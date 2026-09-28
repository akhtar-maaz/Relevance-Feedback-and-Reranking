#!/usr/bin/env python3
"""Run a fixed, reproducible development study with a held-out topic split.

Select QL smoothing on training topics, then select RM3 settings by the
mean nDCG@10 across clean, 25%, and 50% public practice noise. Evaluate
only the selected settings and an immutable original reference on the
validation topics. This script never edits the submission defaults.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import noise_injection
from harness.metrics import evaluate_run
from harness.run_harness import check_conformance
from harness.trec_io import read_qrels, read_queries
from scripts.evaluate_feedback import checked_candidates, checked_ranking, short_ids
import submission.feedback as feedback


SPLIT_SEED = 7364
TRAIN_FRACTION = 0.7
TRAIN_NOISE_SEEDS = (0, 1, 2)
VALIDATION_NOISE_SEEDS = (101, 102, 103, 104, 105)
NOISE_LEVELS = (0.0, 0.25, 0.5)
QL_MUS = (500.0, 1500.0, 3000.0)
ESTIMATORS = ("rm1", "rm2")
FEEDBACK_MUS = (10.0, 300.0)
QUERY_WEIGHTS = (0.25, 0.5, 0.75)
DEPTH = 10
EXPANSION_TERMS = 20
REFERENCE = MappingProxyType({
    "dirichlet_mu": 1500.0, "feedback_mu": 10.0, "query_weight": 0.5,
    "expansion_terms": 20, "rm_estimator": "rm2",
})
PARAMETER_ATTRIBUTES = {
    "dirichlet_mu": "DIRICHLET_MU", "feedback_mu": "FEEDBACK_MU",
    "query_weight": "QUERY_WEIGHT", "expansion_terms": "EXPANSION_TERMS",
    "rm_estimator": "RM_ESTIMATOR",
}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "queries", "qrels", "candidates", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    return cli


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_parameters(parameters):
    for key, attribute in PARAMETER_ATTRIBUTES.items():
        setattr(feedback, attribute, parameters[key])


def read_inputs(args):
    input_names = ("corpus", "queries", "qrels", "candidates")
    for name in input_names:
        if not getattr(args, name).is_file():
            raise ValueError(f"--{name}: file does not exist: {getattr(args, name)}")
    if args.output.resolve() in {getattr(args, name).resolve() for name in input_names}:
        raise ValueError("--output must not overwrite an input file")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file")
    problems = check_conformance(feedback)
    if problems:
        raise ValueError("Submission interface: " + "; ".join(problems))
    query_rows = read_queries(str(args.queries))
    queries = dict(query_rows)
    if any(not qid or not text.strip() for qid, text in query_rows):
        raise ValueError("queries must contain nonempty query IDs and query text")
    if len(queries) != len(query_rows):
        raise ValueError("queries contain duplicate query IDs")
    if len(queries) < 10:
        raise ValueError("at least 10 queries are required for the training/validation split")
    pools = checked_candidates(args.candidates)
    qrels = read_qrels(str(args.qrels))
    for label, missing in (
        ("missing candidates", queries.keys() - pools.keys()),
        ("missing qrels", queries.keys() - qrels.keys()),
        ("empty candidate pools", {qid for qid in queries if qid in pools and not pools[qid]}),
    ):
        if missing:
            raise ValueError(label + " for query IDs: " + short_ids(missing))
    return queries, pools, qrels


def ql_run(qids, queries, pools, mu):
    feedback.DIRICHLET_MU = mu
    run = {
        qid: checked_ranking(feedback.score_candidates(queries[qid], pools[qid], DEPTH),
                             qid, DEPTH, pools[qid])
        for qid in qids
    }
    return run, {qid: [doc_id for doc_id, _ in ranked] for qid, ranked in run.items()}


def seed_conditions(qids, clean_seeds, pools, base_seeds):
    """Use the same random draws for every configuration sharing a QL seed.

    With different QL settings, the clean seed and eligible replacement
    pool may differ. Base seeds and the per-topic RNG recipe stay paired.
    """
    conditions = {level: [] for level in NOISE_LEVELS}
    conditions[0.0].append((None, clean_seeds))
    for base_seed in base_seeds:
        perturbed = {level: {} for level in NOISE_LEVELS if level > 0}
        for qid in qids:
            for level, seed in noise_injection.sweep(
                clean_seeds[qid], pools[qid],
                seed=noise_injection.stable_seed(qid, base_seed=base_seed),
                levels=list(NOISE_LEVELS),
            ):
                if level > 0:
                    perturbed[level][qid] = seed
        for level, seeds in perturbed.items():
            conditions[level].append((base_seed, seeds))
    return conditions


def feedback_measurement(qids, queries, pools, qrels, parameters, conditions, clean_seeds):
    started = time.perf_counter()
    apply_parameters(parameters)
    rows = []
    for level in NOISE_LEVELS:
        repeats = []
        actual_fractions = []
        for base_seed, seeds in conditions[level]:
            run = {
                qid: checked_ranking(
                    feedback.relevance_model_feedback(queries[qid], seeds[qid], pools[qid], DEPTH),
                    qid, DEPTH, pools[qid],
                )
                for qid in qids
            }
            repeats.append({"base_seed": base_seed, "metrics": evaluate_run(run, qrels)})
            actual_fractions.extend(
                len(set(seeds[qid]) - set(clean_seeds[qid])) / len(clean_seeds[qid])
                if clean_seeds[qid] else 0.0 for qid in qids
            )
        aggregates = [repeat["metrics"]["aggregate"] for repeat in repeats]
        rows.append({
            "noise_fraction": level, "repeat_count": len(repeats),
            "mean_actual_replacement_fraction": statistics.mean(actual_fractions),
            "aggregate_mean": {
                metric: statistics.mean(values[metric] for values in aggregates)
                for metric in aggregates[0]
            },
            "ndcg@10_sample_std": statistics.stdev(values["ndcg@10"] for values in aggregates)
            if len(aggregates) > 1 else 0.0,
            "per_query_mean": {
                qid: {
                    metric: statistics.mean(repeat["metrics"]["per_query"][qid][metric] for repeat in repeats)
                    for metric in ("ndcg@10", "map@10")
                } for qid in qids
            },
            "repeats": repeats,
        })
    return {
        "parameters": dict(parameters),
        "objective": statistics.mean(row["aggregate_mean"]["ndcg@10"] for row in rows),
        "conditions": rows, "seconds": time.perf_counter() - started,
    }


def paired_delta(selected, reference):
    """Compare topic means; a tie means absolute difference <= 1e-12."""
    result = {}
    for metric in ("ndcg@10", "map@10"):
        deltas = {qid: selected[qid][metric] - reference[qid][metric] for qid in selected}
        wins = sum(delta > 1e-12 for delta in deltas.values())
        losses = sum(delta < -1e-12 for delta in deltas.values())
        result[metric] = {
            "mean_delta": statistics.mean(deltas.values()), "per_query_delta": deltas,
            "wins": wins, "ties": len(deltas) - wins - losses, "losses": losses,
        }
    return result


def compare_validation(selected, reference, qids):
    conditions = []
    for selected_row, reference_row in zip(selected["feedback"]["conditions"], reference["feedback"]["conditions"]):
        conditions.append({
            "noise_fraction": selected_row["noise_fraction"],
            "topic_mean_comparison": paired_delta(selected_row["per_query_mean"], reference_row["per_query_mean"]),
            "paired_repeats": [
                {"base_seed": selected_repeat["base_seed"],
                 "comparison": paired_delta(selected_repeat["metrics"]["per_query"], reference_repeat["metrics"]["per_query"])}
                for selected_repeat, reference_repeat in zip(selected_row["repeats"], reference_row["repeats"])
            ],
        })
    topic_objectives = []
    for result in (selected, reference):
        topic_objectives.append({
            qid: {
                metric: statistics.mean(row["per_query_mean"][qid][metric] for row in result["feedback"]["conditions"])
                for metric in ("ndcg@10", "map@10")
            } for qid in qids
        })
    return {
        "direction": "selected minus original reference",
        "tie_absolute_tolerance": 1e-12,
        "baseline": paired_delta(selected["baseline"]["per_query"], reference["baseline"]["per_query"]),
        "feedback_conditions": conditions,
        "feedback_equal_condition_mean": paired_delta(*topic_objectives),
    }


def evaluate(args):
    started = time.perf_counter()
    queries, pools, all_qrels = read_inputs(args)
    shuffled_qids = sorted(queries)
    random.Random(SPLIT_SEED).shuffle(shuffled_qids)
    train_size = int(len(shuffled_qids) * TRAIN_FRACTION)
    train_qids, validation_qids = shuffled_qids[:train_size], shuffled_qids[train_size:]
    # Keep ALL judgments per topic, including documents outside candidates.
    train_qrels = {qid: all_qrels[qid] for qid in train_qids}
    validation_qrels = {qid: all_qrels[qid] for qid in validation_qids}
    print(f"Frozen split: {len(train_qids)} training, {len(validation_qids)} validation topics.", flush=True)
    print("Preparing the full corpus once; the study may take several minutes.", flush=True)
    prepare_started = time.perf_counter()
    feedback.prepare(str(args.corpus))
    prepare_seconds = time.perf_counter() - prepare_started
    # Inspect only coverage in prepared storage. All rankings and feedback
    # below go through the assignment's public functions.
    corpus_ids = feedback._STATS.doc_texts
    candidate_ids = {doc_id for pool in pools.values() for doc_id in pool}
    missing = candidate_ids - corpus_ids.keys()
    if missing:
        raise ValueError("candidate IDs absent from corpus: " + short_ids(missing))
    original_parameters = {key: getattr(feedback, attribute) for key, attribute in PARAMETER_ATTRIBUTES.items()}
    print(f"Prepared {len(corpus_ids)} documents in {prepare_seconds:.1f}s; all {len(candidate_ids)} candidate IDs found.", flush=True)
    ql_rows, feedback_rows = [], []
    train_started = time.perf_counter()
    try:
        training_seeds = {}
        for mu in QL_MUS:
            row_started = time.perf_counter()
            run, seeds = ql_run(train_qids, queries, pools, mu)
            training_seeds[mu] = seeds
            row = {"dirichlet_mu": mu, "metrics": evaluate_run(run, train_qrels), "seconds": time.perf_counter() - row_started}
            ql_rows.append(row)
            print(f"Training QL mu={mu:g}: nDCG@10={row['metrics']['aggregate']['ndcg@10']:.6f}, MAP@10={row['metrics']['aggregate']['map@10']:.6f} ({row['seconds']:.1f}s)", flush=True)
        best_ql = max(ql_rows, key=lambda row: (
            row["metrics"]["aggregate"]["ndcg@10"], row["metrics"]["aggregate"]["map@10"],
            -abs(row["dirichlet_mu"] - REFERENCE["dirichlet_mu"]),
        ))
        selected_mu = best_ql["dirichlet_mu"]
        clean_seeds = training_seeds[selected_mu]
        conditions = seed_conditions(train_qids, clean_seeds, pools, TRAIN_NOISE_SEEDS)
        for estimator in ESTIMATORS:
            for feedback_mu in FEEDBACK_MUS:
                for query_weight in QUERY_WEIGHTS:
                    parameters = {
                        "dirichlet_mu": selected_mu, "feedback_mu": feedback_mu,
                        "query_weight": query_weight, "expansion_terms": EXPANSION_TERMS,
                        "rm_estimator": estimator,
                    }
                    row = feedback_measurement(train_qids, queries, pools, train_qrels, parameters, conditions, clean_seeds)
                    feedback_rows.append(row)
                    scores = "/".join(f"{condition['aggregate_mean']['ndcg@10']:.6f}" for condition in row["conditions"])
                    print(f"Training {estimator}, feedback_mu={feedback_mu:g}, query_weight={query_weight:g}: objective={row['objective']:.6f}; clean/25%/50%={scores} ({row['seconds']:.1f}s)", flush=True)
        best_feedback = max(feedback_rows, key=lambda row: (row["objective"], row["parameters"]["query_weight"]))
        selected_parameters = dict(best_feedback["parameters"])
        training_seconds = time.perf_counter() - train_started
        print("Training selection frozen: " + json.dumps(selected_parameters, sort_keys=True), flush=True)
        validation_started = time.perf_counter()
        validation = {}
        validation_baselines = {}
        validation_conditions = {}
        for label, parameters in (("original_reference", REFERENCE), ("selected", selected_parameters)):
            mu = parameters["dirichlet_mu"]
            if mu not in validation_baselines:
                run, seeds = ql_run(validation_qids, queries, pools, mu)
                validation_baselines[mu] = (evaluate_run(run, validation_qrels), seeds)
                validation_conditions[mu] = seed_conditions(validation_qids, seeds, pools, VALIDATION_NOISE_SEEDS)
            baseline, seeds = validation_baselines[mu]
            result = feedback_measurement(validation_qids, queries, pools, validation_qrels, parameters, validation_conditions[mu], seeds)
            validation[label] = {"parameters": dict(parameters), "baseline": baseline, "feedback": result}
            print(f"Validation {label}: QL nDCG@10={baseline['aggregate']['ndcg@10']:.6f}, feedback objective={result['objective']:.6f} ({result['seconds']:.1f}s)", flush=True)
        validation_seconds = time.perf_counter() - validation_started
        validation["paired_comparison"] = compare_validation(validation["selected"], validation["original_reference"], validation_qids)
    finally:
        apply_parameters(original_parameters)

    root = Path(__file__).resolve().parents[1]
    source_paths = (Path(__file__).resolve(), Path(feedback.__file__).resolve(), root / "submission/lm_utils.py",
                    root / "scripts/evaluate_feedback.py", root / "harness/metrics.py", root / "harness/noise_injection.py",
                    root / "harness/run_harness.py", root / "harness/trec_io.py")
    inputs = {
        name: {"path": str(getattr(args, name).resolve()), "bytes": getattr(args, name).stat().st_size,
               "sha256": file_hash(getattr(args, name))}
        for name in ("corpus", "queries", "qrels", "candidates")
    }
    return {
        "schema_version": 1,
        "note": "Local development study; validation topics were not used to select parameters. Practice noise is not the hidden grading recipe. No submission defaults were changed.",
        "inputs": inputs,
        "source_sha256": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "loaded_submission_parameters": original_parameters,
        "protocol": {
            "split_seed": SPLIT_SEED, "train_fraction": TRAIN_FRACTION,
            "split_recipe": "sort query IDs; shuffle with random.Random(7364); first floor(0.7*n) train, remaining validation",
            "ql_mus": QL_MUS, "estimators": ESTIMATORS, "feedback_mus": FEEDBACK_MUS,
            "query_weights": QUERY_WEIGHTS, "expansion_terms": EXPANSION_TERMS,
            "prf_depth": DEPTH, "ranking_cutoff": DEPTH, "noise_levels": NOISE_LEVELS,
            "training_noise_base_seeds": TRAIN_NOISE_SEEDS, "validation_noise_base_seeds": VALIDATION_NOISE_SEEDS,
            "noise_recipe": "harness.noise_injection.sweep with stable_seed(qid, base_seed); identical draws across configurations; QL seed is recomputed per mu",
            "ql_selection": "max training nDCG@10, then MAP@10, then closest to original reference mu",
            "feedback_selection": "max arithmetic mean of clean nDCG@10 and repeat-mean nDCG@10 at 25% and 50% noise; exact ties prefer higher query weight, then first grid row",
            "reference_parameters": dict(REFERENCE), "validation_policy": "only selected and original reference, after selection is frozen",
            "qrels_policy": "all judgments per evaluated topic, including documents outside the candidate pool",
        },
        "split": {"training_query_ids": train_qids, "validation_query_ids": validation_qids},
        "coverage": {
            "input_queries": len(queries), "corpus_documents": len(corpus_ids),
            "unique_candidate_documents": len(candidate_ids), "missing_candidate_documents": 0,
            "qrels_query_ids_not_requested": sorted(all_qrels.keys() - queries.keys()),
            "candidate_query_ids_not_requested": sorted(pools.keys() - queries.keys()),
        },
        "training": {"ql": ql_rows, "feedback": feedback_rows,
                     "selected_dirichlet_mu": selected_mu, "selected_parameters": selected_parameters},
        "validation": validation,
        "timing": {"prepare_seconds": prepare_seconds, "training_seconds": training_seconds,
                   "validation_seconds": validation_seconds, "total_seconds": time.perf_counter() - started},
    }


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        report = evaluate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        cli.error(str(error))
    print(f"Full reproducible report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
