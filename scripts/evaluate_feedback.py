#!/usr/bin/env python3
"""Compare RM1/RM2 and RM3 weights under repeated public practice noise.

Uses only the standard library and repository modules. These are local
development measurements, not leaderboard scores. Do not tune on the toy set.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

# Support both `python scripts/evaluate_feedback.py` and `python -m scripts.evaluate_feedback`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import noise_injection
from harness.metrics import evaluate_run
from harness.run_harness import PRF_DEPTH, _validate_and_sort_results, check_conformance
from harness.trec_io import read_qrels, read_queries
import submission.feedback as feedback


def bounded_fraction(value):
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "queries", "qrels", "candidates"):
        result.add_argument("--" + name, required=True, type=Path)
    result.add_argument("--output", type=Path, help="write full results and per-query nDCG to JSON")
    result.add_argument("--estimators", nargs="+", choices=("rm1", "rm2"), default=["rm1", "rm2"])
    weights = result.add_mutually_exclusive_group()
    weights.add_argument("--lambdas", nargs="+", type=bounded_fraction,
                         help="RM3 original-query weights; default: submission's QUERY_WEIGHT")
    weights.add_argument("--lambda-sweep", action="store_true", help="use 0, 0.25, 0.5, 0.75, 1")
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="distinct base random seeds (default: 0 1 2)")
    result.add_argument("--noise-levels", nargs="+", type=bounded_fraction,
                        default=noise_injection.PRACTICE_NOISE_LEVELS,
                        help="practice replacement fractions (default: 0 0.25 0.5); clean is always included")
    result.add_argument("--prf-depth", type=positive_int, default=PRF_DEPTH,
                        help="number of baseline results used as feedback seeds (default: %(default)s)")
    return result


def checked_candidates(path):
    """Reject malformed/duplicate pools instead of silently overwriting them."""
    pools = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["qid"]
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{path}:{line_number}: qid must be a nonempty string")
            if qid in pools:
                raise ValueError(f"{path}:{line_number}: duplicate candidate qid {qid!r}")
            if not isinstance(item["candidates"], list):
                raise ValueError(f"{path}:{line_number}: candidates must be a list")
            ids = []
            for entry in item["candidates"]:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError(f"{path}:{line_number}: expected [doc_id, score] entries")
                doc_id, score = entry
                if not isinstance(doc_id, str) or not doc_id or not math.isfinite(float(score)):
                    raise ValueError(f"{path}:{line_number}: invalid candidate ID or score")
                ids.append(doc_id)
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}:{line_number}: duplicate document in pool for {qid!r}")
            pools[qid] = ids
    return pools


def checked_ranking(results, qid, k, candidates):
    ranked = _validate_and_sort_results(results, qid, k, set(candidates))
    if any(not math.isfinite(score) for _, score in ranked):
        raise ValueError(f"qid={qid}: submission returned a non-finite score")
    return ranked


def short_ids(ids):
    ordered = sorted(ids)
    return ", ".join(ordered[:10]) + (f" (+{len(ordered) - 10} more)" if len(ordered) > 10 else "")


def measurement(run, qrels, base_seed):
    metrics = evaluate_run(run, qrels)
    return {
        "base_seed": base_seed,
        "ndcg@10": metrics["aggregate"]["ndcg@10"],
        "per_query_ndcg@10": {qid: values["ndcg@10"] for qid, values in metrics["per_query"].items()},
    }


def evaluate(args):
    for name in ("corpus", "queries", "qrels", "candidates"):
        path = getattr(args, name)
        if not path.is_file():
            raise ValueError(f"--{name}: file does not exist: {path}")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be distinct so repeats are not counted twice")
    problems = check_conformance(feedback)
    if problems:
        raise ValueError("Submission interface: " + "; ".join(problems))

    queries = read_queries(str(args.queries))
    query_ids = {qid for qid, _ in queries}
    if not queries or any(not qid or not text.strip() for qid, text in queries):
        raise ValueError("queries must contain nonempty query IDs and query text")
    if len(query_ids) != len(queries):
        raise ValueError("queries contain duplicate query IDs")
    pools = checked_candidates(args.candidates)
    all_qrels = read_qrels(str(args.qrels))
    missing_pools = query_ids - pools.keys()
    if missing_pools:
        raise ValueError("missing candidates for query IDs: " + short_ids(missing_pools))
    empty_pools = {qid for qid in query_ids if not pools[qid]}
    if empty_pools:
        raise ValueError("empty candidate pools for query IDs: " + short_ids(empty_pools))
    missing_qrels = query_ids - all_qrels.keys()
    if missing_qrels:
        raise ValueError("missing qrels for query IDs: " + short_ids(missing_qrels))
    # Evaluate exactly the supplied topics. Report extra qrels topics explicitly.
    qrels = {qid: all_qrels[qid] for qid, _ in queries}
    weights = [0.0, 0.25, 0.5, 0.75, 1.0] if args.lambda_sweep else args.lambdas
    weights = list(dict.fromkeys(weights if weights is not None else [feedback.QUERY_WEIGHT]))
    estimators = list(dict.fromkeys(args.estimators))
    levels = sorted(set(args.noise_levels) | {0.0})

    started = time.perf_counter()
    feedback.prepare(str(args.corpus))
    prepare_seconds = time.perf_counter() - started
    baseline, clean_seeds = {}, {}
    depth = max(10, args.prf_depth)
    for qid, text in queries:
        ranked = checked_ranking(feedback.score_candidates(text, pools[qid], depth), qid, depth, pools[qid])
        baseline[qid] = ranked[:10]
        clean_seeds[qid] = [doc_id for doc_id, _ in ranked[:args.prf_depth]]

    # Draw once per (query, repeat, level); every configuration gets identical
    # lists. Qrels are never passed to a ranking or noise-injection function.
    noisy_seeds = {seed: {level: {} for level in levels if level > 0} for seed in args.seeds}
    for seed in args.seeds:
        for qid, _ in queries:
            samples = noise_injection.sweep(
                clean_seeds[qid], pools[qid],
                seed=noise_injection.stable_seed(qid, base_seed=seed), levels=levels,
            )
            for level, doc_ids in samples:
                if level > 0:
                    noisy_seeds[seed][level][qid] = doc_ids

    def feedback_run(seeds):
        return {
            qid: checked_ranking(
                feedback.relevance_model_feedback(text, seeds[qid], pools[qid], 10), qid, 10, pools[qid]
            )
            for qid, text in queries
        }

    rows = []
    original_settings = feedback.RM_ESTIMATOR, feedback.QUERY_WEIGHT
    try:
        for estimator in estimators:
            feedback.RM_ESTIMATOR = estimator
            for query_weight in weights:
                feedback.QUERY_WEIGHT = query_weight
                clean = measurement(feedback_run(clean_seeds), qrels, None)
                for level in levels:
                    repeats = [clean] if level == 0 else [
                        measurement(feedback_run(noisy_seeds[seed][level]), qrels, seed)
                        for seed in args.seeds
                    ]
                    values = [repeat["ndcg@10"] for repeat in repeats]
                    actual_fractions = [0.0] if level == 0 else [
                        len(set(noisy_seeds[seed][level][qid]) - set(clean_seeds[qid])) / len(clean_seeds[qid])
                        if clean_seeds[qid] else 0.0
                        for seed in args.seeds for qid, _ in queries
                    ]
                    rows.append({
                        "estimator": estimator, "query_weight": query_weight, "noise_fraction": level,
                        "mean_actual_replacement_fraction": statistics.mean(actual_fractions),
                        "ndcg@10_mean": statistics.mean(values),
                        "ndcg@10_sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "repeat_count": len(repeats), "repeats": repeats,
                    })
    finally:
        feedback.RM_ESTIMATOR, feedback.QUERY_WEIGHT = original_settings

    return {
        "note": "Local development measurements, not leaderboard scores. Toy data is for debugging, not tuning.",
        "inputs": {name: str(getattr(args, name).resolve()) for name in ("corpus", "queries", "qrels", "candidates")},
        "feedback_source_sha256": hashlib.sha256(Path(feedback.__file__).read_bytes()).hexdigest(),
        "settings": {
            "estimators": estimators, "query_weights": weights, "base_seeds": args.seeds,
            "noise_levels": levels, "prf_depth": args.prf_depth, "ranking_cutoff": 10,
            "dirichlet_mu": feedback.DIRICHLET_MU, "feedback_mu": feedback.FEEDBACK_MU,
            "expansion_terms": feedback.EXPANSION_TERMS,
        },
        "coverage": {
            "input_queries": len(queries), "queries_with_candidates": len(query_ids & pools.keys()),
            "queries_with_qrels": len(query_ids & all_qrels.keys()), "evaluated_queries": len(qrels),
            "qrels_query_ids_not_requested": sorted(all_qrels.keys() - query_ids),
            "candidate_query_ids_not_requested": sorted(pools.keys() - query_ids),
        },
        "baseline": evaluate_run(baseline, qrels), "feedback": rows,
        "timing": {"prepare_seconds": prepare_seconds, "total_seconds": time.perf_counter() - started},
    }


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        report = evaluate(args)
        if args.output:
            if args.output.resolve() in {getattr(args, name).resolve() for name in ("corpus", "queries", "qrels", "candidates")}:
                raise ValueError("--output must not overwrite an input file")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        cli.error(str(error))
    print(report["note"])
    coverage = report["coverage"]
    print(f"Query coverage: {coverage['evaluated_queries']}/{coverage['input_queries']} supplied queries evaluated")
    print(f"Extra topics excluded: {len(coverage['qrels_query_ids_not_requested'])} qrels, "
          f"{len(coverage['candidate_query_ids_not_requested'])} candidate pools")
    print(f"Baseline nDCG@10: {report['baseline']['aggregate']['ndcg@10']:.6f}")
    print("estimator lambda noise actual_noise repeats  mean_nDCG@10 sample_std")
    for row in report["feedback"]:
        print(f"{row['estimator']:9s} {row['query_weight']:6.2f} {row['noise_fraction']:5.2f} "
              f"{row['mean_actual_replacement_fraction']:12.3f} {row['repeat_count']:7d} "
              f"{row['ndcg@10_mean']:13.6f} {row['ndcg@10_sample_std']:10.6f}")
    if args.output:
        print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
