"""
submission/feedback.py -- THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment -- do not rename
them, change their signatures, or move them out of this file.

    prepare(corpus_path: str) -> None
    score_candidates(query, candidate_doc_ids, k=10) -> List[(doc_id, score)]
    relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k=10)
        -> List[(doc_id, score)]

Overview of what is implemented here (everything from scratch, standard
library only):

1. Text analysis (submission/text_utils.py): lowercase alphanumeric
   tokens + Porter stemming for documents and queries; stopwords are
   removed from queries and from relevance-model expansion terms only.

2. score_candidates(): unigram query likelihood
       score(D) = sum_{q in Q} log P(q | D)
   with Dirichlet-prior smoothing (default) or Jelinek-Mercer smoothing
   (SMOOTHING / DIRICHLET_MU / JM_LAMBDA), computed over the provided
   candidate pool only. By default mu = MU_AVGDL_FACTOR x the average
   document length of the corpus given to prepare().

3. relevance_model_feedback(): relevance models (Lavrenko & Croft 2001),
   estimated ONLY from `pseudo_relevant_doc_ids`, used to rerank ONLY
   `candidate_doc_ids`:
     RM1:  P(w|R) ∝ sum_{D in F} P(w|D) P(D|Q),   P(D|Q) ∝ P(Q|D) (uniform P(D))
     RM2:  P(w|R) ∝ P(w) prod_{q in Q} sum_{D in F} P(q|D) P(D|w),
           P(D|w) = P(w|D) P(D) / P(w),   P(w) = sum_D P(w|D) P(D)
     RM3:  P'(w) = λ P(w|Q) + (1-λ) P(w|R)     (P(w|R) from RM_BASE = rm1|rm2)
   Candidates are ranked by the cross entropy between the (truncated,
   renormalised) feedback model θ and each candidate's Dirichlet-smoothed
   document model:  score(D) = sum_w θ(w) log P(w|D)  (rank-equivalent to
   -KL(θ || θ_D)). RM_VARIANT picks which of the three is used.

   Drift-robustness choices (all generic -- nothing here tries to detect
   which seed documents are noise, and nothing reads state left behind by
   score_candidates()):
     * P(D|Q) is recomputed from THIS call's seed with our own smoothed
       query likelihood, so a seed document that matches the query poorly
       (typical of contamination) gets exponentially less say in P(w|R).
       SEED_WEIGHT_TEMP flattens/sharpens that posterior.
     * Optional (off by default): expansion terms must occur in at least
       MIN_SEED_DF seed documents, and seed weights can be multiplied by
       each document's coherence with the rest of the seed
       (COHERENCE_POWER). Both were measured on the dev sweep; neither
       improved retention there, so both are disabled.
     * Only the top FB_TERMS non-stopword terms are kept.
     * The original query keeps weight λ (RM3), anchoring the model.
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from submission.corpus_utils import load_corpus
from submission.lm_utils import dirichlet_smoothed_log_prob, jelinek_mercer_smoothed_log_prob
from submission.text_utils import Analyzer

# ---------------------------------------------------------------------------
# Tunable parameters (chosen on a local dev sweep -- see scripts/sweep.py).
# All are read at call time, so experiments can override them on the module.
# ---------------------------------------------------------------------------
USE_STEMMING = True        # applied in prepare()

# Query likelihood (Track A)
SMOOTHING = "dirichlet"    # "dirichlet" | "jm"
DIRICHLET_MU = None        # None -> MU_AVGDL_FACTOR * average document length
MU_AVGDL_FACTOR = 3.0
JM_LAMBDA = 0.7            # weight of the collection model under JM

# Relevance-model feedback (Tracks B / C)
RM_VARIANT = "rm3"         # "rm1" | "rm2" | "rm3"
RM_BASE = "rm1"            # which estimator RM3 interpolates: "rm1" | "rm2"
FB_LAMBDA = 0.5            # RM3 weight on the ORIGINAL query model P(w|Q)
FB_TERMS = 45              # expansion terms kept from P(w|R)
FB_MAX_DOCS = 50           # safety cap on how many seed documents are used
SEED_WEIGHT_TEMP = 1.0     # P(D|Q) ∝ exp(log P(Q|D) / T)
MIN_SEED_DF = 1            # expansion term must appear in >= this many seed docs
COHERENCE_POWER = 0.0      # P(D|Q) multiplied by (mean cosine to other seed docs)^power
RANK_MU = None             # Dirichlet mu for ranking with θ (None -> same mu as QL)

_LOG_FLOOR = -50.0


class _Collection:
    """Collection statistics for the background model P(w|C), plus lazily
    computed (and cached) per-document term counts -- same design as
    submission/lm_utils.CollectionStats, but in the stemmed term space."""

    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer = analyzer
        self.doc_texts: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.cf: Counter = Counter()
        self.total_len = 0
        self.n_docs = 0
        self._tf_cache: Dict[str, Counter] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = self.analyzer.analyze(text)
        counts = Counter(tokens)
        self.doc_texts[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.cf.update(counts)
        self.total_len += len(tokens)
        self.n_docs += 1

    def has(self, doc_id: str) -> bool:
        return doc_id in self.doc_texts

    def tf(self, doc_id: str) -> Counter:
        counts = self._tf_cache.get(doc_id)
        if counts is None:
            counts = Counter(self.analyzer.analyze(self.doc_texts[doc_id]))
            self._tf_cache[doc_id] = counts
        return counts

    def avg_doc_len(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 0.0

    def p_coll(self, term: str) -> float:
        if self.total_len == 0:
            return 0.0
        return self.cf.get(term, 0) / self.total_len


_COLL: Optional[_Collection] = None


# ---------------------------------------------------------------------------
# Required interface
# ---------------------------------------------------------------------------
def prepare(corpus_path: str) -> None:
    """Load the corpus and build collection-wide statistics. Called once."""
    global _COLL
    coll = _Collection(Analyzer(stem=USE_STEMMING))
    for doc_id, text in load_corpus(corpus_path):
        coll.add(doc_id, text)
    _COLL = coll


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs from candidate_doc_ids, best first,
    under smoothed unigram query likelihood."""
    coll = _require_prepared("score_candidates")
    q_terms = coll.analyzer.analyze_query(query)
    pool = _known_unique(coll, candidate_doc_ids)
    if not q_terms:
        return _top_k({d: 0.0 for d in pool}, k)
    return _top_k({d: query_log_likelihood(coll, q_terms, d) for d in pool}, k)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs from candidate_doc_ids, best first,
    reranked with a relevance model estimated from pseudo_relevant_doc_ids."""
    coll = _require_prepared("relevance_model_feedback")
    q_terms = coll.analyzer.analyze_query(query)
    pool = _known_unique(coll, candidate_doc_ids)
    seed = _known_unique(coll, pseudo_relevant_doc_ids)[:FB_MAX_DOCS]

    if not q_terms:
        return _top_k({d: 0.0 for d in pool}, k)
    if not seed:
        # Nothing to estimate a relevance model from: plain query likelihood.
        return _top_k({d: query_log_likelihood(coll, q_terms, d) for d in pool}, k)

    theta = build_feedback_model(coll, q_terms, seed)
    mu = RANK_MU if RANK_MU is not None else dirichlet_mu(coll)
    return _top_k({d: cross_entropy_score(coll, theta, d, mu) for d in pool}, k)


# ---------------------------------------------------------------------------
# Language-model building blocks (exposed for tests / experiments)
# ---------------------------------------------------------------------------
def dirichlet_mu(coll: _Collection) -> float:
    """The Dirichlet prior mu: DIRICHLET_MU if set, else scaled to the
    collection's average document length (mu acts as a pseudo-document
    length, so its useful range moves with |D|)."""
    if DIRICHLET_MU is not None:
        return float(DIRICHLET_MU)
    return max(1.0, MU_AVGDL_FACTOR * coll.avg_doc_len())


def smoothed_log_prob(coll: _Collection, term: str, doc_id: str, mu: Optional[float] = None) -> float:
    """log P(term | D) under the configured smoothing method."""
    tf = coll.tf(doc_id).get(term, 0)
    dl = coll.doc_lengths[doc_id]
    pc = coll.p_coll(term)
    if SMOOTHING == "jm" and mu is None:
        return jelinek_mercer_smoothed_log_prob(tf, dl, pc, JM_LAMBDA)
    return dirichlet_smoothed_log_prob(tf, dl, pc, dirichlet_mu(coll) if mu is None else mu)


def query_log_likelihood(coll: _Collection, q_terms: Sequence[str], doc_id: str) -> float:
    """log P(Q | D) = sum over query tokens (with repetition) of log P(q|D)."""
    return sum(smoothed_log_prob(coll, t, doc_id) for t in q_terms)


def seed_posteriors(coll: _Collection, q_terms: Sequence[str], seed: Sequence[str]) -> Dict[str, float]:
    """P(D|Q) over the seed set: softmax of log P(Q|D) / SEED_WEIGHT_TEMP
    (uniform document prior)."""
    temp = SEED_WEIGHT_TEMP if SEED_WEIGHT_TEMP > 0 else 1.0
    logs = {d: query_log_likelihood(coll, q_terms, d) / temp for d in seed}
    if COHERENCE_POWER and len(seed) >= 3:
        for d, c in seed_coherence(coll, seed).items():
            logs[d] += COHERENCE_POWER * math.log(max(c, 1e-6))
    m = max(logs.values())
    exp = {d: math.exp(v - m) for d, v in logs.items()}
    z = sum(exp.values())
    return {d: v / z for d, v in exp.items()}


def seed_coherence(coll: _Collection, seed: Sequence[str]) -> Dict[str, float]:
    """Mean cosine similarity (non-stopword tf vectors) of each seed doc to
    the other seed docs. Documents on the seed's dominant topic score high;
    an isolated document scores low."""
    is_stop = coll.analyzer.is_stopword_stem
    vecs = {}
    for d in seed:
        v = {w: c for w, c in coll.tf(d).items() if not is_stop(w)}
        norm = math.sqrt(sum(c * c for c in v.values())) or 1.0
        vecs[d] = {w: c / norm for w, c in v.items()}
    out = {}
    for d in seed:
        vd = vecs[d]
        sims = [sum(p * vecs[o].get(w, 0.0) for w, p in vd.items()) for o in seed if o != d]
        out[d] = sum(sims) / len(sims) if sims else 1.0
    return out


def _doc_ml(coll: _Collection, doc_id: str) -> Dict[str, float]:
    """Maximum-likelihood document model c(w,D)/|D|."""
    dl = coll.doc_lengths[doc_id]
    if dl == 0:
        return {}
    return {w: c / dl for w, c in coll.tf(doc_id).items()}


def estimate_rm1(coll: _Collection, q_terms: Sequence[str], seed: Sequence[str]) -> Dict[str, float]:
    """RM1: P(w|R) ∝ sum_D P(w|D) P(D|Q). Returns a normalised distribution
    over every term that occurs in the seed documents."""
    post = seed_posteriors(coll, q_terms, seed)
    rm: Dict[str, float] = {}
    for d in seed:
        wd = post[d]
        for w, p in _doc_ml(coll, d).items():
            rm[w] = rm.get(w, 0.0) + wd * p
    return _normalize(rm)


def estimate_rm2(coll: _Collection, q_terms: Sequence[str], seed: Sequence[str]) -> Dict[str, float]:
    """RM2 (conditional / iid sampling): the query terms are drawn
    independently, each from its own document chosen given w:
        P(w, q_1..q_k) = P(w) prod_i sum_D P(q_i|D) P(D|w)
    with P(D|w) = P(w|D) P(D) / P(w), uniform P(D) over the seed, P(w|D)
    the ML document model (restricted to seed vocabulary) and P(q|D) the
    smoothed document model. Computed in log space."""
    ml = {d: _doc_ml(coll, d) for d in seed}
    n = len(seed)
    p_w: Dict[str, float] = {}
    for d in seed:
        for w, p in ml[d].items():
            p_w[w] = p_w.get(w, 0.0) + p / n
    q_probs = {d: [math.exp(smoothed_log_prob(coll, q, d)) for q in q_terms] for d in seed}

    log_scores: Dict[str, float] = {}
    for w, pw in p_w.items():
        # P(D|w) for each seed doc
        pdw = [(ml[d].get(w, 0.0) / n) / pw for d in seed]
        s = math.log(pw)
        for i in range(len(q_terms)):
            inner = sum(pdw[j] * q_probs[d][i] for j, d in enumerate(seed))
            s += math.log(inner) if inner > 0 else _LOG_FLOOR
        log_scores[w] = s
    if not log_scores:
        return {}
    m = max(log_scores.values())
    return _normalize({w: math.exp(v - m) for w, v in log_scores.items()})


def select_expansion_terms(
    coll: _Collection, rm: Dict[str, float], seed: Sequence[str], n_terms: int
) -> Dict[str, float]:
    """Keep the n_terms highest-probability non-stopword terms of P(w|R)
    that occur in >= MIN_SEED_DF seed docs (when |seed| >= 3); renormalise."""
    min_df = MIN_SEED_DF if len(seed) >= 3 else 1
    seed_df: Counter = Counter()
    if min_df > 1:
        for d in seed:
            seed_df.update(coll.tf(d).keys())
    is_stop = coll.analyzer.is_stopword_stem
    kept = [
        (w, p) for w, p in rm.items()
        if not is_stop(w) and len(w) > 1 and (min_df <= 1 or seed_df[w] >= min_df)
    ]
    kept.sort(key=lambda wp: (-wp[1], wp[0]))
    return _normalize(dict(kept[:n_terms]))


def query_model(q_terms: Sequence[str]) -> Dict[str, float]:
    """Maximum-likelihood query model P(w|Q) = c(w,Q)/|Q|."""
    return _normalize(dict(Counter(q_terms)))


def interpolate_rm3(p_q: Dict[str, float], p_r: Dict[str, float], lam: float) -> Dict[str, float]:
    """RM3: λ P(w|Q) + (1-λ) P(w|R)."""
    if not p_r:
        return dict(p_q)
    out = {w: lam * p for w, p in p_q.items()}
    for w, p in p_r.items():
        out[w] = out.get(w, 0.0) + (1.0 - lam) * p
    return _normalize(out)


def build_feedback_model(coll: _Collection, q_terms: Sequence[str], seed: Sequence[str]) -> Dict[str, float]:
    """The θ used to rank candidates, for the configured RM_VARIANT."""
    variant = RM_VARIANT.lower()
    base = RM_BASE.lower() if variant == "rm3" else variant
    rm = estimate_rm2(coll, q_terms, seed) if base == "rm2" else estimate_rm1(coll, q_terms, seed)
    p_r = select_expansion_terms(coll, rm, seed, FB_TERMS)
    if variant == "rm3":
        return interpolate_rm3(query_model(q_terms), p_r, FB_LAMBDA)
    return p_r if p_r else query_model(q_terms)


def cross_entropy_score(coll: _Collection, theta: Dict[str, float], doc_id: str, mu: float) -> float:
    """sum_w θ(w) log P(w|D) with Dirichlet-smoothed P(w|D); rank-equivalent
    to -KL(θ || θ_D)."""
    tf = coll.tf(doc_id)
    dl = coll.doc_lengths[doc_id]
    return sum(
        p * dirichlet_smoothed_log_prob(tf.get(w, 0), dl, coll.p_coll(w), mu)
        for w, p in theta.items()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_prepared(name: str) -> _Collection:
    if _COLL is None:
        raise RuntimeError(f"{name}() called before prepare(corpus_path).")
    return _COLL


def _known_unique(coll: _Collection, doc_ids: Sequence[str]) -> List[str]:
    """Order-preserving de-duplication, dropping doc_ids not in the corpus."""
    seen = set()
    out = []
    for d in doc_ids:
        if d not in seen and coll.has(d):
            seen.add(d)
            out.append(d)
    return out


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    z = sum(dist.values())
    if z <= 0:
        return {}
    return {w: v / z for w, v in dist.items()}


def _top_k(scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    """Sort by score descending, ties broken by doc_id for determinism."""
    ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
    return [(d, float(s)) for d, s in ranked[: max(k, 0)]]
