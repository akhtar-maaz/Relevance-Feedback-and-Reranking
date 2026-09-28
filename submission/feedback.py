"""
submission/feedback.py -- THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5,
"Submission Interface & Conformance Checking") -- do not rename them,
change their signatures, or move them out of this file.

    prepare(corpus_path: str) -> None
        Called once, before anything else. Load the corpus and build
        whatever collection-wide statistics your unigram LM and
        relevance model need (see submission/lm_utils.py::CollectionStats
        for why this is a cheap, one-time pass and does not require
        building an index). No build/load process split this time --
        grading is in-process, so it is fine to keep everything in
        module-level state here and read it in the two functions below.

    score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        Rerank the PROVIDED candidate pool -- typically the top-100 from
        an undisclosed reference retriever (assignment Section 6) -- using
        your own unigram query-likelihood LM (Section 3.1). You are never
        asked to rank the whole corpus; candidate_doc_ids is the entire
        universe of documents you need to consider for this call. Return
        up to k (doc_id, score) pairs, sorted by score descending, drawn
        ONLY from candidate_doc_ids. This is graded directly as Track A.

    relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        RM1/RM2/RM3-based reranking (Section 3.2). Two DIFFERENT lists
        come in, doing two different jobs -- do not confuse them:
          - pseudo_relevant_doc_ids: the (possibly noise-perturbed) seed
            set used to ESTIMATE the relevance model. Supplied BY THE
            HARNESS -- sometimes exactly your own score_candidates()
            top-k', sometimes a version of that with a fraction swapped
            for off-topic documents (query drift stress-testing;
            assignment Section 7, Track C). You have no legitimate way to
            tell which, and must not try to detect it -- see
            docs/SUBMISSION_INTERFACE.md, "call-order independence".
          - candidate_doc_ids: the pool to RERANK using your estimated
            model -- typically the SAME candidate pool passed to
            score_candidates() for this query, unperturbed. Your
            returned doc_ids must come from this list, not from
            pseudo_relevant_doc_ids and not from outside either list.
        Return up to k (doc_id, score) pairs, sorted by score descending.

The base reranker uses Dirichlet-smoothed query likelihood and a modest
prior from the provided first-pass candidate order. Feedback estimates RM1
or RM2 from the supplied seed, then interpolates a truncated relevance
model with the original query (RM3). The final ranking fuses feedback with
the base ranking to limit drift from contaminated seeds. No ranking or seed
is cached by query; every call uses exactly the supplied documents.

Only Python's standard library and the starter's token/count helpers are
used. The earlier RM3 defaults were selected on 35 TREC-COVID development
topics and checked on 15 reserved topics. Rank fusion was studied afterward
on all 50 public topics; see docs/DEV_RESULTS.md for the tradeoffs.
"""
import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission.lm_utils import CollectionStats, dirichlet_smoothed_log_prob, tokenize

DIRICHLET_MU = 500.0
FEEDBACK_MU = 300.0
QUERY_WEIGHT = 0.5  # RM3 lambda: 1 keeps the query; 0 uses only feedback.
EXPANSION_TERMS = 20
RM_ESTIMATOR = "rm1"  # Either "rm1" or "rm2"; both feed RM3 interpolation.
FIRST_PASS_RANK_WEIGHT = 0.1
FEEDBACK_RANK_WEIGHT = 0.05

_STATS: Optional[CollectionStats] = None


def prepare(corpus_path: str) -> None:
    """Stream corpus statistics in one pass, retaining texts for lazy counts.

    Building a fresh object also resets document caches between corpora.
    Duplicate IDs are rejected because they would corrupt collection counts.
    """
    global _STATS
    stats = CollectionStats()
    with open(corpus_path, "r", encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, 1):
            if not line.strip():
                continue
            document = json.loads(line)
            doc_id, text = document["doc_id"], document["text"]
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise ValueError(f"Corpus line {line_number}: doc_id and text must be strings")
            if doc_id in stats.doc_texts:
                raise ValueError(f"Corpus line {line_number}: duplicate doc_id {doc_id!r}")
            stats.add_document(doc_id, text)
    _STATS = stats


def _require_stats() -> CollectionStats:
    if _STATS is None:
        raise RuntimeError("Call prepare(corpus_path) before scoring or feedback.")
    return _STATS


def _validate_parameters(feedback: bool = False) -> None:
    if not math.isfinite(DIRICHLET_MU) or DIRICHLET_MU <= 0:
        raise ValueError("DIRICHLET_MU must be finite and positive")
    if not feedback:
        return
    if not math.isfinite(FEEDBACK_MU) or FEEDBACK_MU <= 0:
        raise ValueError("FEEDBACK_MU must be finite and positive")
    if not math.isfinite(QUERY_WEIGHT) or not 0 <= QUERY_WEIGHT <= 1:
        raise ValueError("QUERY_WEIGHT must be between 0 and 1")
    if not isinstance(EXPANSION_TERMS, int) or EXPANSION_TERMS < 1:
        raise ValueError("EXPANSION_TERMS must be a positive integer")
    if RM_ESTIMATOR not in ("rm1", "rm2"):
        raise ValueError("RM_ESTIMATOR must be 'rm1' or 'rm2'")


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Rerank the supplied pool with query likelihood and its initial order.

    The input order is a useful, score-free first-pass prior. Query
    likelihood can move documents within it, while the prior prevents
    excessive changes on queries whose words are common in the corpus.
    """
    stats = _require_stats()
    _validate_parameters()
    if k <= 0 or not candidate_doc_ids or not tokenize(query):
        return []
    query_likelihood = _ql_rerank(query, candidate_doc_ids, len(candidate_doc_ids), stats)
    first_pass = list(dict.fromkeys(candidate_doc_ids))
    return _fuse_ranks(first_pass, query_likelihood, FIRST_PASS_RANK_WEIGHT, k)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Estimate feedback from the supplied seed and rerank the candidate pool.

    Seed documents may be outside the candidate pool. Empty seeds fall back
    to query likelihood. Out-of-vocabulary query words cannot supply topical
    evidence, so they do not dilute the query anchor or influence estimation.
    """
    stats = _require_stats()
    _validate_parameters(feedback=True)
    if k <= 0 or not candidate_doc_ids:
        return []
    query_terms = [term for term in tokenize(query) if stats.collection_prob(term) > 0]
    if not query_terms or not pseudo_relevant_doc_ids or QUERY_WEIGHT == 1:
        return score_candidates(query, candidate_doc_ids, k)

    # Canonical ordering makes the supplied set independent of input order.
    # Unknown IDs raise the same clear error as candidate lookups; empty
    # documents contribute no observed vocabulary and supply no feedback.
    seed_ids = [
        doc_id for doc_id in sorted(set(pseudo_relevant_doc_ids))
        if stats.doc_term_counts(doc_id)
    ]
    if not seed_ids:
        return score_candidates(query, candidate_doc_ids, k)
    estimator = _estimate_rm1 if RM_ESTIMATOR == "rm1" else _estimate_rm2
    relevance_model = estimator(query_terms, seed_ids, stats)
    expansion = _select_expansion(relevance_model, stats)
    query_model = _normalize(Counter(query_terms))
    model = _interpolate_rm3(query_model, expansion, QUERY_WEIGHT)
    feedback_ranking = _rank_model(model, candidate_doc_ids, len(candidate_doc_ids), stats)
    base_ranking = score_candidates(query, candidate_doc_ids, len(candidate_doc_ids))
    # A fixed rank coefficient cannot move a document in a tiny pool;
    # strengthen feedback there while keeping the 100-document default.
    feedback_weight = max(FEEDBACK_RANK_WEIGHT, min(2.0, 10.0 / len(base_ranking)))
    return _fuse_ranks([doc_id for doc_id, _ in base_ranking], feedback_ranking,
                       feedback_weight, k)


def _fuse_ranks(
    prior_ids: List[str], evidence_ranking: List[Tuple[str, float]],
    evidence_weight: float, k: int,
) -> List[Tuple[str, float]]:
    """Combine two ranked views of the same pool without score calibration.

    The first-pass scores are not supplied to the submission, and LM log
    likelihood scales with query length. Rank fusion avoids comparing those
    incompatible score scales. Lower ranks are better; exact ties retain
    the prior order.
    """
    if k <= 0:
        return []
    if len(prior_ids) == 1:
        return evidence_ranking[:1]
    evidence_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(evidence_ranking, 1)}
    scored = [
        (doc_id, -float(rank + evidence_weight * evidence_rank[doc_id]))
        for rank, doc_id in enumerate(dict.fromkeys(prior_ids), 1)
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = math.fsum(weights.values())
    if total <= 0:
        return {}
    return {term: weights[term] / total for term in sorted(weights) if weights[term] > 0}


def _softmax(log_weights: Dict[str, float]) -> Dict[str, float]:
    if not log_weights:
        return {}
    highest = max(log_weights.values())
    return _normalize({key: math.exp(value - highest) for key, value in log_weights.items()})


def _seed_posteriors(query_terms: List[str], seed_ids: List[str], stats: CollectionStats) -> Dict[str, float]:
    """P(D|Q) under a uniform seed-document prior, calculated in log space."""
    query_counts = Counter(query_terms)
    log_scores = {}
    for doc_id in sorted(set(seed_ids)):
        counts = stats.doc_term_counts(doc_id)
        length = stats.doc_lengths[doc_id]
        log_scores[doc_id] = math.fsum(
            count * dirichlet_smoothed_log_prob(
                counts.get(term, 0), length, stats.collection_prob(term), FEEDBACK_MU
            )
            for term, count in sorted(query_counts.items())
        )
    return _softmax(log_scores)


def _seed_vocabulary(seed_ids: List[str], stats: CollectionStats) -> List[str]:
    vocabulary = set()
    for doc_id in seed_ids:
        vocabulary.update(stats.doc_term_counts(doc_id))
    return sorted(vocabulary)


def _estimate_rm1(query_terms: List[str], seed_ids: List[str], stats: CollectionStats) -> Dict[str, float]:
    """RM1: sum_D P(w|D) P(D|Q), normalized over the seed vocabulary.

    Accumulate observed counts sparsely and add the shared collection
    smoothing contribution once per word, avoiding a vocabulary x seed loop.
    """
    seed_ids = sorted(set(seed_ids))
    posteriors = _seed_posteriors(query_terms, seed_ids, stats)
    observed = {}
    background_weights = []
    for doc_id in seed_ids:
        # Even if a posterior underflows to zero, its observed words remain
        # in the seed vocabulary and can receive background smoothing.
        posterior = posteriors.get(doc_id, 0.0)
        scale = posterior / (stats.doc_lengths[doc_id] + FEEDBACK_MU)
        background_weights.append(FEEDBACK_MU * scale)
        for term, count in stats.doc_term_counts(doc_id).items():
            observed.setdefault(term, []).append(scale * count)
    background_weight = math.fsum(background_weights)
    return _normalize({
        term: math.fsum(contributions) + background_weight * stats.collection_prob(term)
        for term, contributions in observed.items()
    })


def _estimate_rm2(query_terms: List[str], seed_ids: List[str], stats: CollectionStats) -> Dict[str, float]:
    """RM2: P(w) product_i sum_D P(q_i|D) P(D|w), with one seed prior.

    For n seed documents with uniform prior:
        P(w) = sum_D P(w|D) / n
        P(D|w) = P(w|D) / sum_D P(w|D).
    Using collection P(w|C) for the first factor would mix two different
    priors and break the one-word-query equivalence between RM1 and RM2.
    Collection probabilities enter only through Dirichlet smoothing here.
    """
    seed_ids = sorted(set(seed_ids))
    if not seed_ids:
        return {}
    counts = [stats.doc_term_counts(doc_id) for doc_id in seed_ids]
    inverse_lengths = [1.0 / (stats.doc_lengths[doc_id] + FEEDBACK_MU) for doc_id in seed_ids]
    query_probs = []
    for term, frequency in sorted(Counter(query_terms).items()):
        background = stats.collection_prob(term)
        if background <= 0:
            continue  # An OOV word has no evidence under any seed model.
        probabilities = [
            (doc_counts.get(term, 0) + FEEDBACK_MU * background) * inverse_length
            for doc_counts, inverse_length in zip(counts, inverse_lengths)
        ]
        query_probs.append((frequency, probabilities))

    log_model = {}
    for term in _seed_vocabulary(seed_ids, stats):
        background = stats.collection_prob(term)
        word_probs = [
            (doc_counts.get(term, 0) + FEEDBACK_MU * background) * inverse_length
            for doc_counts, inverse_length in zip(counts, inverse_lengths)
        ]
        word_sum = math.fsum(word_probs)
        log_conditionals = [
            frequency * math.log(math.fsum(
                p_word * p_query for p_word, p_query in zip(word_probs, probabilities)
            ) / word_sum)
            for frequency, probabilities in query_probs
        ]
        log_model[term] = math.log(word_sum / len(seed_ids)) + math.fsum(log_conditionals)
    return _softmax(log_model)


def _select_expansion(model: Dict[str, float], stats: CollectionStats) -> Dict[str, float]:
    """Truncate by model probability times collection self-information.

    This selection favors informative seed terms. The selected terms keep
    their model probabilities, renormalized before RM3 interpolation.
    """
    terms = sorted(model, key=lambda term: (
        -model[term] * math.log1p(1.0 / max(stats.collection_prob(term), 1e-300)),
        term,
    ))[:EXPANSION_TERMS]
    return _normalize({term: model[term] for term in terms})


def _interpolate_rm3(
    query_model: Dict[str, float], relevance_model: Dict[str, float], query_weight: float,
) -> Dict[str, float]:
    """Mix normalized query and feedback models, preserving both endpoints."""
    if not relevance_model:
        return _normalize(query_model)
    if not query_model:
        return _normalize(relevance_model)
    return _normalize({
        term: query_weight * query_model.get(term, 0.0)
        + (1.0 - query_weight) * relevance_model.get(term, 0.0)
        for term in sorted(set(query_model) | set(relevance_model))
    })


def _rank_model(model: Dict[str, float], doc_ids: List[str], k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    """Weighted log likelihood; RM3 ranking is equivalent to negative KL."""
    if k <= 0 or not model:
        return []
    terms = [(term, weight, stats.collection_prob(term)) for term, weight in sorted(model.items())]
    scores = []
    for doc_id in dict.fromkeys(doc_ids):
        counts = stats.doc_term_counts(doc_id)
        length = stats.doc_lengths[doc_id]
        score = math.fsum(
            weight * dirichlet_smoothed_log_prob(counts.get(term, 0), length, background, DIRICHLET_MU)
            for term, weight, background in terms
        )
        scores.append((doc_id, score))
    # Stable sort retains candidate input order for true ties.
    return sorted(scores, key=lambda pair: pair[1], reverse=True)[:k]


def _ql_rerank(query: str, doc_ids: List[str], k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    return _rank_model(Counter(tokenize(query)), doc_ids, k, stats)
