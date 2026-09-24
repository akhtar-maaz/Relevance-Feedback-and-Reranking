"""
Hand-checkable correctness tests for submission/feedback.py (assignment
"Correctness of required components": the unigram QL reranker and the
RM1/RM2/RM3 feedback layer must be verifiable against small examples).

Tiny corpus used throughout (3-letter words are not stemmed):
    d1: "cat cat dog"      |d1| = 3
    d2: "cat fox"          |d2| = 2
    d3: "dog dog fox fox"  |d3| = 4
Collection: cat=3, dog=3, fox=3, |C|=9  ->  P(w|C) = 1/3 for every term.
With Dirichlet mu = 3:  P(w|D) = (c(w,D) + 1) / (|D| + 3).
"""
import json
import math
from fractions import Fraction as F

import pytest

import submission.feedback as fb
from submission.text_utils import porter_stem

DOCS = {"d1": "cat cat dog", "d2": "cat fox", "d3": "dog dog fox fox"}
POOL = ["d1", "d2", "d3"]


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"doc_id": d, "text": t}) + "\n" for d, t in DOCS.items()))
    for name, value in {
        "_COLL": None,
        "USE_STEMMING": True,
        "SMOOTHING": "dirichlet",
        "DIRICHLET_MU": 3.0,
        "RANK_MU": None,
        "SEED_WEIGHT_TEMP": 1.0,
        "FB_TERMS": 100,
        "FB_MAX_DOCS": 50,
        "MIN_SEED_DF": 1,
        "RM_BASE": "rm1",
        "RM_VARIANT": "rm3",
        "FB_LAMBDA": 0.5,
    }.items():
        monkeypatch.setattr(fb, name, value)
    fb.prepare(str(path))
    return fb._COLL


def _approx(dist, expected):
    assert set(dist) == set(expected)
    for w, p in expected.items():
        assert dist[w] == pytest.approx(float(p), abs=1e-12), w


# ---------------------------------------------------------------- QL ----
def test_dirichlet_query_likelihood_matches_hand_computation(tiny):
    # P(cat|d1) = (2+1)/(3+3) = 1/2 ; P(cat|d2) = (1+1)/(2+3) = 2/5 ; P(cat|d3) = 1/7
    res = fb.score_candidates("cat", POOL, 10)
    assert [d for d, _ in res] == ["d1", "d2", "d3"]
    got = dict(res)
    assert got["d1"] == pytest.approx(math.log(1 / 2))
    assert got["d2"] == pytest.approx(math.log(2 / 5))
    assert got["d3"] == pytest.approx(math.log(1 / 7))


def test_multi_term_query_sums_log_probs(tiny):
    # "cat fox": d2 = log(2/5)+log(2/5), d1 = log(1/2)+log(1/6), d3 = log(1/7)+log(3/7)
    got = dict(fb.score_candidates("the cat and the fox", POOL, 10))  # stopwords dropped
    assert got["d2"] == pytest.approx(2 * math.log(2 / 5))
    assert got["d1"] == pytest.approx(math.log(1 / 2) + math.log(1 / 6))
    assert got["d3"] == pytest.approx(math.log(1 / 7) + math.log(3 / 7))


def test_jelinek_mercer_matches_hand_computation(tiny, monkeypatch):
    monkeypatch.setattr(fb, "SMOOTHING", "jm")
    monkeypatch.setattr(fb, "JM_LAMBDA", 0.5)
    # P(cat|d1) = 0.5*2/3 + 0.5*1/3 = 1/2 ; d2: 0.5*1/2+1/6 = 5/12 ; d3: 1/6
    got = dict(fb.score_candidates("cat", POOL, 10))
    assert got["d1"] == pytest.approx(math.log(1 / 2))
    assert got["d2"] == pytest.approx(math.log(5 / 12))
    assert got["d3"] == pytest.approx(math.log(1 / 6))


def test_score_candidates_only_ranks_the_given_pool_and_respects_k(tiny):
    res = fb.score_candidates("cat", ["d3", "d2", "d2", "not_a_doc"], 10)
    assert [d for d, _ in res] == ["d2", "d3"]
    assert len(fb.score_candidates("cat", POOL, 1)) == 1
    assert fb.score_candidates("cat", POOL, 0) == []


# ---------------------------------------------------------- RM1/2/3 -----
def test_seed_posteriors_are_normalised_query_likelihoods(tiny):
    post = fb.seed_posteriors(tiny, ["cat"], ["d1", "d2"])
    # P(D|Q) ∝ P(Q|D): 1/2 : 2/5  ->  5/9 : 4/9
    _approx(post, {"d1": F(5, 9), "d2": F(4, 9)})


def test_rm1_matches_hand_computation(tiny):
    # P(w|R) = 5/9 * ML(d1) + 4/9 * ML(d2)
    #   cat = 5/9*2/3 + 4/9*1/2 = 16/27 ; dog = 5/9*1/3 = 5/27 ; fox = 4/9*1/2 = 6/27
    rm = fb.estimate_rm1(tiny, ["cat"], ["d1", "d2"])
    _approx(rm, {"cat": F(16, 27), "dog": F(5, 27), "fox": F(6, 27)})
    assert sum(rm.values()) == pytest.approx(1.0)


def test_rm2_equals_rm1_for_a_single_term_query(tiny):
    # With |Q| = 1 the RM2 formula P(w) sum_D P(q|D)P(D|w) reduces to
    # sum_D P(w|D)P(D)P(q|D), i.e. exactly RM1.
    rm2 = fb.estimate_rm2(tiny, ["cat"], ["d1", "d2"])
    _approx(rm2, {"cat": F(16, 27), "dog": F(5, 27), "fox": F(6, 27)})


def test_rm2_matches_hand_computation_for_two_terms(tiny):
    # Seed {d1, d2}, uniform P(D) = 1/2, ML doc models:
    #   d1: cat 2/3, dog 1/3      d2: cat 1/2, fox 1/2
    # P(w): cat 7/12, dog 1/6, fox 1/4
    # P(D|w): cat -> (4/7, 3/7), dog -> (1, 0), fox -> (0, 1)
    # P(cat|D) = (1/2, 2/5), P(fox|D) = (1/6, 2/5)
    pq = {"cat": (F(1, 2), F(2, 5)), "fox": (F(1, 6), F(2, 5))}
    pdw = {"cat": (F(4, 7), F(3, 7)), "dog": (F(1), F(0)), "fox": (F(0), F(1))}
    pw = {"cat": F(7, 12), "dog": F(1, 6), "fox": F(1, 4)}
    raw = {}
    for w in pw:
        s = pw[w]
        for q in ("cat", "fox"):
            s *= pdw[w][0] * pq[q][0] + pdw[w][1] * pq[q][1]
        raw[w] = s
    z = sum(raw.values())
    expected = {w: v / z for w, v in raw.items()}   # ≈ cat .569, dog .111, fox .320
    rm2 = fb.estimate_rm2(tiny, ["cat", "fox"], ["d1", "d2"])
    _approx(rm2, expected)
    assert rm2 != pytest.approx(fb.estimate_rm1(tiny, ["cat", "fox"], ["d1", "d2"]))


def test_rm3_interpolation_matches_hand_computation(tiny):
    # λ = 1/2: cat = 1/2*1 + 1/2*16/27 = 43/54 ; dog = 5/54 ; fox = 6/54
    theta = fb.build_feedback_model(tiny, ["cat"], ["d1", "d2"])
    _approx(theta, {"cat": F(43, 54), "dog": F(5, 54), "fox": F(6, 54)})


def test_feedback_ranking_is_cross_entropy_with_the_feedback_model(tiny):
    theta = {"cat": 43 / 54, "dog": 5 / 54, "fox": 6 / 54}
    res = dict(fb.relevance_model_feedback("cat", ["d1", "d2"], POOL, 10))
    for d, (tf, dl) in {"d1": ({"cat": 2, "dog": 1}, 3), "d2": ({"cat": 1, "fox": 1}, 2),
                        "d3": ({"dog": 2, "fox": 2}, 4)}.items():
        expected = sum(p * math.log((tf.get(w, 0) + 1) / (dl + 3)) for w, p in theta.items())
        assert res[d] == pytest.approx(expected)


def test_rm3_with_lambda_one_reproduces_query_likelihood_ranking(tiny, monkeypatch):
    monkeypatch.setattr(fb, "FB_LAMBDA", 1.0)
    for q in ("cat", "fox", "dog cat"):
        ql = [d for d, _ in fb.score_candidates(q, POOL, 10)]
        rm = [d for d, _ in fb.relevance_model_feedback(q, ["d3"], POOL, 10)]
        assert rm == ql


def test_rm1_and_rm2_variants_ignore_the_query_anchor(tiny, monkeypatch):
    for variant in ("rm1", "rm2"):
        monkeypatch.setattr(fb, "RM_VARIANT", variant)
        theta = fb.build_feedback_model(tiny, ["cat"], ["d1", "d2"])
        _approx(theta, {"cat": F(16, 27), "dog": F(5, 27), "fox": F(6, 27)})


def test_min_seed_df_filters_single_document_terms(tiny, monkeypatch):
    monkeypatch.setattr(fb, "MIN_SEED_DF", 2)
    seed = ["d1", "d2", "d3"]  # cat in {d1,d2}, dog in {d1,d3}, fox in {d2,d3}: all kept
    assert set(fb.select_expansion_terms(tiny, fb.estimate_rm1(tiny, ["cat"], seed), seed, 100)) == {"cat", "dog", "fox"}
    seed = ["d1", "d2", "d2x"]  # d2x unknown -> only 2 real docs, filter disabled
    assert fb.relevance_model_feedback("cat", seed, POOL, 10)


# ------------------------------------------------------ interface edge --
def test_feedback_only_returns_candidates_even_if_seed_is_outside_pool(tiny):
    res = fb.relevance_model_feedback("cat", ["d1", "d2"], ["d3"], 10)
    assert [d for d, _ in res] == ["d3"]


def test_feedback_falls_back_to_ql_for_empty_or_unknown_seed(tiny):
    ql = fb.score_candidates("cat", POOL, 10)
    assert fb.relevance_model_feedback("cat", [], POOL, 10) == ql
    assert fb.relevance_model_feedback("cat", ["nope"], POOL, 10) == ql


def test_feedback_depends_on_the_seed_it_is_given(tiny):
    a = fb.relevance_model_feedback("cat", ["d1"], POOL, 10)
    b = fb.relevance_model_feedback("cat", ["d2"], POOL, 10)
    assert a != b


def test_deterministic_and_no_duplicates(tiny):
    a = fb.relevance_model_feedback("cat dog", ["d1", "d1", "d3"], POOL + POOL, 10)
    b = fb.relevance_model_feedback("cat dog", ["d1", "d1", "d3"], POOL + POOL, 10)
    assert a == b
    assert len({d for d, _ in a}) == len(a) == 3


def test_stopword_only_query_does_not_crash(tiny):
    assert len(fb.score_candidates("the of and", POOL, 10)) == 3
    assert len(fb.relevance_model_feedback("", ["d1"], POOL, 10)) == 3


def test_porter_stemmer_reference_examples():
    for word, stem in {"caresses": "caress", "ponies": "poni", "relational": "relat",
                       "hopping": "hop", "generalizations": "gener", "retrieval": "retriev"}.items():
        assert porter_stem(word) == stem
