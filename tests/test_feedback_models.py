"""Numerical and behavioral regressions for the submitted relevance models.

Run with pytest, or without any third-party packages:
    python3 tests/test_feedback_models.py
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from submission import feedback
from submission.lm_utils import CollectionStats


class FeedbackModelTests(unittest.TestCase):
    def setUp(self):
        self.stats = CollectionStats.from_corpus([
            ("d1", "apple apple banana"),
            ("d2", "apple carrot"),
            ("background", "carrot carrot carrot"),
        ])
        self.settings = patch.multiple(
            feedback,
            _STATS=self.stats,
            FEEDBACK_MU=2.0,
            DIRICHLET_MU=2.0,
            QUERY_WEIGHT=0.5,
            EXPANSION_TERMS=20,
            RM_ESTIMATOR="rm2",
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def assertModelAlmostEqual(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        self.assertAlmostEqual(math.fsum(actual.values()), 1.0, places=12)
        for term, probability in expected.items():
            with self.subTest(term=term):
                self.assertTrue(math.isfinite(actual[term]))
                self.assertGreaterEqual(actual[term], 0.0)
                self.assertAlmostEqual(actual[term], probability, places=12)

    def test_rm1_matches_hand_calculated_posterior_mixture(self):
        # P_C(apple, banana, carrot) = (3/8, 1/8, 1/2).
        # With mu=2, P(w|d1)=(11/20, 1/4, 1/5),
        # P(w|d2)=(7/16, 1/16, 1/2). For Q=apple the document
        # posterior is (44/79, 35/79); take that weighted mixture.
        actual = feedback._estimate_rm1(["apple"], ["d1", "d2"], self.stats)
        self.assertModelAlmostEqual(actual, {
            "apple": 3161 / 6320,
            "banana": 211 / 1264,
            "carrot": 263 / 790,
        })

    def test_rm2_single_query_term_matches_rm1(self):
        # Under a uniform seed prior, P(w)=mean_D P(w|D). For a
        # one-token query the RM2 factorization reduces exactly to RM1.
        # Using collection P(w) here instead would violate this identity.
        actual = feedback._estimate_rm2(["apple"], ["d1", "d2"], self.stats)
        self.assertModelAlmostEqual(actual, {
            "apple": 3161 / 6320,
            "banana": 211 / 1264,
            "carrot": 263 / 790,
        })

    def test_rm2_matches_hand_calculated_two_term_query(self):
        # For each w, multiply its seed marginal by
        # P(apple|w) P(banana|w), then normalize. These fractions come
        # from the two document distributions written in the RM1 test.
        unnormalized = {
            "apple": 666971 / 16179200,
            "banana": 3587 / 204800,
            "carrot": 3419 / 179200,
        }
        total = math.fsum(unnormalized.values())
        actual = feedback._estimate_rm2(
            ["apple", "banana"], ["d1", "d2"], self.stats
        )
        self.assertModelAlmostEqual(
            actual, {term: value / total for term, value in unnormalized.items()}
        )

    def test_rm1_repeated_query_tokens_change_document_posteriors(self):
        # P(d1|apple apple) = 44^2 / (44^2 + 35^2).
        p1 = 1936 / 3161
        p2 = 1225 / 3161
        actual = feedback._estimate_rm1(
            ["apple", "apple"], ["d1", "d2"], self.stats
        )
        self.assertModelAlmostEqual(actual, {
            "apple": p1 * 11 / 20 + p2 * 7 / 16,
            "banana": p1 / 4 + p2 / 16,
            "carrot": p1 / 5 + p2 / 2,
        })

    def test_rm_estimators_remain_finite_for_long_queries(self):
        # Direct likelihood products would underflow to zero here.
        for estimator in (feedback._estimate_rm1, feedback._estimate_rm2):
            with self.subTest(estimator=estimator.__name__):
                actual = estimator(["apple"] * 2000, ["d1", "d2"], self.stats)
                self.assertEqual(set(actual), {"apple", "banana", "carrot"})
                self.assertAlmostEqual(math.fsum(actual.values()), 1.0, places=12)
                self.assertTrue(all(math.isfinite(p) and p >= 0 for p in actual.values()))

    def test_rm1_keeps_smoothed_terms_when_a_seed_posterior_underflows(self):
        # A sufficiently long query makes P(d2|Q) round to exactly zero.
        # Carrot occurs only in d2 within the seed, but its probability in
        # d1 remains 1/5 from background smoothing, so it must be retained.
        actual = feedback._estimate_rm1(
            ["apple"] * 20000, ["d1", "d2"], self.stats
        )
        self.assertModelAlmostEqual(actual, {
            "apple": 11 / 20, "banana": 1 / 4, "carrot": 1 / 5,
        })

    def test_rm3_interpolation_normalizes_and_preserves_query_terms(self):
        actual = feedback._interpolate_rm3(
            {"apple": 2 / 3, "banana": 1 / 3},
            {"banana": 1 / 4, "carrot": 3 / 4},
            0.4,
        )
        self.assertModelAlmostEqual(actual, {
            "apple": 4 / 15,
            "banana": 17 / 60,
            "carrot": 9 / 20,
        })

    def test_rm3_interpolation_endpoints_and_empty_feedback(self):
        query = {"apple": 0.75, "banana": 0.25}
        expansion = {"banana": 0.2, "carrot": 0.8}
        self.assertModelAlmostEqual(
            feedback._interpolate_rm3(query, expansion, 1.0), query
        )
        self.assertModelAlmostEqual(
            feedback._interpolate_rm3(query, expansion, 0.0), expansion
        )
        self.assertModelAlmostEqual(
            feedback._interpolate_rm3(query, {}, 0.4), query
        )

    def test_lambda_one_is_exactly_the_base_reranker(self):
        candidates = ["d2", "d1", "background"]
        with patch.object(feedback, "QUERY_WEIGHT", 1.0):
            self.assertEqual(
                feedback.relevance_model_feedback(
                    "apple apple banana", ["background"], candidates, 3
                ),
                feedback.score_candidates("apple apple banana", candidates, 3),
            )

    def test_lambda_zero_scores_using_the_relevance_distribution(self):
        expected_model = {"apple": 3161 / 6320, "banana": 211 / 1264, "carrot": 263 / 790}
        document_probabilities = {
            "d1": {"apple": 11 / 20, "banana": 1 / 4, "carrot": 1 / 5},
            "d2": {"apple": 7 / 16, "banana": 1 / 16, "carrot": 1 / 2},
        }
        with patch.object(feedback, "QUERY_WEIGHT", 0.0):
            actual = dict(feedback.relevance_model_feedback(
                "apple", ["d1", "d2"], ["d1", "d2"], 2
            ))
        self.assertEqual(set(actual), {"d1", "d2"})
        for doc_id, probabilities in document_probabilities.items():
            expected = math.fsum(
                expected_model[term] * math.log(probability)
                for term, probability in probabilities.items()
            )
            self.assertAlmostEqual(actual[doc_id], expected, places=12)

    def test_supplied_seed_affects_ranking_even_outside_candidate_pool(self):
        candidates = ["d1", "d2"]
        with patch.object(feedback, "QUERY_WEIGHT", 0.0):
            apple_seed = feedback.relevance_model_feedback("apple", ["d1"], candidates, 2)
            carrot_seed = feedback.relevance_model_feedback(
                "apple", ["background"], candidates, 2
            )
        self.assertEqual(apple_seed[0][0], "d1")
        self.assertEqual(carrot_seed[0][0], "d2")
        self.assertEqual({doc_id for doc_id, _ in carrot_seed}, set(candidates))

    def test_feedback_is_independent_of_previous_scoring_calls(self):
        candidates = ["d1", "d2"]
        expected = feedback.relevance_model_feedback(
            "apple", ["background"], candidates, 2
        )
        feedback.score_candidates("apple", candidates, 2)
        feedback.relevance_model_feedback("apple", ["d1"], candidates, 2)
        feedback.score_candidates("banana carrot", ["background", "d1"], 1)
        self.assertEqual(
            feedback.relevance_model_feedback("apple", ["background"], candidates, 2),
            expected,
        )

    def test_seed_and_query_permutations_preserve_exact_results(self):
        candidates = ["background", "d2", "d1"]
        self.assertEqual(
            feedback.score_candidates("apple apple banana", candidates, 3),
            feedback.score_candidates("banana apple apple", candidates, 3),
        )
        for estimator in ("rm1", "rm2"):
            with self.subTest(estimator=estimator), patch.object(feedback, "RM_ESTIMATOR", estimator):
                self.assertEqual(
                    feedback.relevance_model_feedback(
                        "apple apple banana", ["d1", "d2"], candidates, 3
                    ),
                    feedback.relevance_model_feedback(
                        "banana apple apple", ["d2", "d1"], candidates, 3
                    ),
                )

    def test_duplicate_seeds_and_candidates_do_not_change_results(self):
        for estimator in ("rm1", "rm2"):
            with self.subTest(estimator=estimator), patch.object(feedback, "RM_ESTIMATOR", estimator):
                expected = feedback.relevance_model_feedback(
                    "apple banana", ["d1", "d2"], ["d1", "d2", "background"], 10
                )
                actual = feedback.relevance_model_feedback(
                    "apple banana", ["d1", "d1", "d2", "d1"],
                    ["d1", "d1", "d2", "background", "d2"], 10,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), len({doc_id for doc_id, _ in actual}))

    def test_empty_seed_falls_back_to_base(self):
        candidates = ["d2", "d1", "background"]
        self.assertEqual(
            feedback.relevance_model_feedback("apple banana", [], candidates, 3),
            feedback.score_candidates("apple banana", candidates, 3),
        )

    def test_empty_queries_and_candidate_pools_return_no_results(self):
        for query in ("", "  !!!  "):
            with self.subTest(query=query):
                self.assertEqual(feedback.score_candidates(query, ["d1"], 10), [])
                self.assertEqual(
                    feedback.relevance_model_feedback(query, ["d2"], ["d1"], 10), []
                )
        self.assertEqual(feedback.score_candidates("apple", [], 10), [])
        self.assertEqual(feedback.relevance_model_feedback("apple", ["d1"], [], 10), [])

    def test_all_oov_query_falls_back_to_finite_base_scores(self):
        candidates = ["d2", "d1", "background"]
        actual = feedback.relevance_model_feedback(
            "unseenword anotherunseenword", ["background"], candidates, 3
        )
        self.assertEqual(actual, feedback.score_candidates(
            "unseenword anotherunseenword", candidates, 3
        ))
        self.assertTrue(all(math.isfinite(score) for _, score in actual))

    def test_oov_tokens_do_not_change_feedback_for_known_query(self):
        candidates = ["d2", "d1", "background"]
        for estimator in ("rm1", "rm2"):
            with self.subTest(estimator=estimator), patch.object(feedback, "RM_ESTIMATOR", estimator):
                expected = feedback.relevance_model_feedback(
                    "apple banana", ["d1", "d2"], candidates, 3
                )
                actual = feedback.relevance_model_feedback(
                    "apple unseenword banana", ["d1", "d2"], candidates, 3
                )
                self.assertEqual(actual, expected)

    def test_nonpositive_k_returns_no_results(self):
        for k in (0, -1, -10):
            with self.subTest(k=k):
                self.assertEqual(feedback.score_candidates("apple", ["d1", "d2"], k), [])
                self.assertEqual(feedback.relevance_model_feedback(
                    "apple", ["d1"], ["d1", "d2"], k
                ), [])

    def test_empty_documents_can_be_scored_and_do_not_supply_feedback(self):
        stats = CollectionStats.from_corpus([
            ("empty", ""), ("d1", "apple apple banana"), ("d2", "apple carrot"),
        ])
        with patch.object(feedback, "_STATS", stats):
            candidates = ["empty", "d1", "d2"]
            actual = feedback.relevance_model_feedback("apple", ["empty"], candidates, 3)
            self.assertEqual(actual, feedback.score_candidates("apple", candidates, 3))
            self.assertTrue(all(math.isfinite(score) for _, score in actual))
            self.assertEqual(
                feedback.relevance_model_feedback("apple", ["d1", "empty"], candidates, 3),
                feedback.relevance_model_feedback("apple", ["d1"], candidates, 3),
            )

    def test_prepare_replaces_previous_collection_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(json.dumps({"doc_id": "fresh", "text": "kiwi kiwi"}) + "\n")
            feedback.prepare(str(path))
        self.assertEqual(feedback.score_candidates("kiwi", ["fresh"], 1), [("fresh", 0.0)])
        self.assertEqual(feedback._STATS.collection_prob("apple"), 0.0)

    def test_prepare_rejects_duplicate_corpus_ids_without_replacing_valid_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text("\n".join(json.dumps(document) for document in [
                {"doc_id": "duplicate", "text": "apple"},
                {"doc_id": "duplicate", "text": "banana"},
            ]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
                feedback.prepare(str(path))
        self.assertIs(feedback._STATS, self.stats)

    def test_results_are_exactly_reproducible_across_process_hash_seeds(self):
        # Set iteration order must not change expansion selection or the
        # floating-point scores. Check both estimators in fresh processes.
        script = """
import json
from submission import feedback
from submission.lm_utils import CollectionStats
words = ['term' + str(i) for i in range(45)]
feedback._STATS = CollectionStats.from_corpus([
    ('a', 'apple apple ' + ' '.join(words)),
    ('b', 'apple banana ' + ' '.join(reversed(words[::2]))),
    ('c', 'carrot carrot ' + ' '.join(words[1::2])),
])
feedback.FEEDBACK_MU = 2.0
feedback.DIRICHLET_MU = 2.0
outputs = {}
for estimator in ['rm1', 'rm2']:
    feedback.RM_ESTIMATOR = estimator
    results = feedback.relevance_model_feedback(
        'apple banana apple', ['a', 'b'], ['c', 'b', 'a'], 3
    )
    outputs[estimator] = [(doc_id, score.hex()) for doc_id, score in results]
print(json.dumps(outputs, sort_keys=True))
"""
        outputs = []
        for hash_seed in ("1", "13", "107"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                text=True, capture_output=True, check=True, timeout=15,
            )
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
