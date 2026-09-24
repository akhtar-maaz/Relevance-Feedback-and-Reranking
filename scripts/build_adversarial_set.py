#!/usr/bin/env python
"""
Build submission/adversarial_set.json (optional bonus track, docs/GRADING.md).

Goal: for a handful of dev queries, a pseudo-relevant doc-id list that
makes an UNKNOWN opponent's relevance_model_feedback() drift as much as
possible (largest nDCG@10 drop vs. that opponent's clean-seed run).

Strategy ("plausible but wrong", topically coherent contamination):
  * Only judged NON-relevant documents from the query's candidate pool are
    eligible (every one of them is something the first-pass retriever
    found plausible). Unjudged documents are used only if too few judged
    non-relevant ones exist.
  * Eligible docs are ranked by query likelihood: typical defenders weight
    seed docs by P(D|Q), so high-QL non-relevant docs survive that
    weighting instead of being discounted away.
  * Several candidate seeds are generated -- the top-m by QL, and clusters
    of mutually similar docs grown around each high-QL non-relevant doc
    (coherent vocabulary survives "term must occur in >=2 seed docs"
    filters and concentrates P(w|R) on one wrong sub-topic).
  * Each candidate seed is scored against a panel of surrogate defenders
    (RM1, RM2, and RM3 at several λ / term counts, all from
    submission.feedback), and the one with the largest mean nDCG@10 drop
    wins. Queries are then ranked by that drop and the top --n-queries kept.

Usage:
    python scripts/build_adversarial_set.py --data data/full --n-queries 8
    (defaults to data/cranfield; qids must be the REAL dev qids at
    submission time -- rerun on data/full once candidates_dev.jsonl is
    released.)
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import submission.feedback as fb  # noqa: E402
from harness.candidates_io import read_candidates  # noqa: E402
from harness.metrics import evaluate_run  # noqa: E402
from harness.trec_io import read_qrels, read_queries  # noqa: E402

PRF_DEPTH = 10

SURROGATES = [
    {"RM_VARIANT": "rm1"},
    {"RM_VARIANT": "rm2"},
    {"RM_VARIANT": "rm3", "FB_LAMBDA": 0.2, "FB_TERMS": 50, "MIN_SEED_DF": 1},
    {"RM_VARIANT": "rm3", "FB_LAMBDA": 0.5, "FB_TERMS": 30, "MIN_SEED_DF": 1},
    {"RM_VARIANT": "rm3", "FB_LAMBDA": 0.5, "FB_TERMS": 30, "MIN_SEED_DF": 2},
    {"RM_VARIANT": "rm3", "FB_LAMBDA": 0.7, "FB_TERMS": 20, "MIN_SEED_DF": 2},
]


def _ndcg(run_list, qid, qrels):
    return evaluate_run({qid: run_list}, {qid: qrels[qid]})["aggregate"]["ndcg@10"]


def _with(cfg):
    saved = {k: getattr(fb, k) for k in cfg}
    for k, v in cfg.items():
        setattr(fb, k, v)
    return saved


def _cos(a: Counter, b: Counter) -> float:
    num = sum(v * b.get(t, 0) for t, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cranfield")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "submission", "adversarial_set.json"))
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--seed-size", type=int, default=PRF_DEPTH)
    ap.add_argument("--min-clean-ndcg", type=float, default=0.3,
                    help="only attack queries a defender does well on (room to drop)")
    args = ap.parse_args()

    queries = read_queries(os.path.join(args.data, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(args.data, "qrels_dev.txt"))
    cands = read_candidates(os.path.join(args.data, "candidates_dev.jsonl"))
    fb.prepare(os.path.join(args.data, "corpus.jsonl"))
    coll = fb._COLL

    results = []
    for qid, text in queries:
        if qid not in cands or qid not in qrels:
            continue
        pool = [d for d, _ in cands[qid]]
        ql_rank = [d for d, _ in fb.score_candidates(text, pool, len(pool))]
        clean_seed = ql_rank[:PRF_DEPTH]

        clean = []
        for cfg in SURROGATES:
            saved = _with(cfg)
            clean.append(_ndcg(fb.relevance_model_feedback(text, clean_seed, pool, 10), qid, qrels))
            _with(saved)
        if sum(clean) / len(clean) < args.min_clean_ndcg:
            continue

        judged_nonrel = [d for d in ql_rank if qrels[qid].get(d, None) == 0]
        unjudged = [d for d in ql_rank if d not in qrels[qid]]
        eligible = (judged_nonrel + unjudged)[: max(3 * args.seed_size, 30)]
        if len(eligible) < 2:
            continue
        m = min(args.seed_size, len(eligible))

        options = [eligible[:m]]
        tf = {d: coll.tf(d) for d in eligible}
        for anchor in eligible[:5]:
            sims = sorted(eligible, key=lambda d: (-_cos(tf[anchor], tf[d]), d))
            options.append(sims[:m])
            options.append(sims[: max(2, m // 2)])

        best = None
        for seed in options:
            drops = []
            for cfg, c in zip(SURROGATES, clean):
                saved = _with(cfg)
                drops.append(c - _ndcg(fb.relevance_model_feedback(text, seed, pool, 10), qid, qrels))
                _with(saved)
            mean_drop = sum(drops) / len(drops)
            if best is None or mean_drop > best[0]:
                best = (mean_drop, seed)
        results.append((best[0], qid, best[1]))
        print(f"{qid}: mean surrogate drop {best[0]:.3f} with {len(best[1])} docs")

    results.sort(key=lambda r: (-r[0], r[1]))
    chosen = {qid: seed for _drop, qid, seed in results[: args.n_queries]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
        f.write("\n")
    mean = sum(r[0] for r in results[: args.n_queries]) / max(1, min(args.n_queries, len(results)))
    print(f"wrote {len(chosen)} queries to {args.out} (mean surrogate nDCG@10 drop {mean:.3f})")


if __name__ == "__main__":
    main()
