#!/usr/bin/env python
"""
Parameter sweeps for submission/feedback.py (the data behind the report's
nDCG@10-vs-λ plots and the final parameter choice).

For every configuration it runs the same protocol as harness/run_harness.py
-- clean seed = our own score_candidates() top-PRF_DEPTH -- and then also
feeds relevance_model_feedback() several kinds of contaminated seeds:

  uniform@f   harness.noise_injection (the public practice recipe) at
              fraction f, averaged over N_SEEDS independent noise draws
  adjacent@f  "plausible but wrong" noise: the replaced docs are drawn from
              the next ADJ_WINDOW documents of OUR OWN QL ranking below the
              seed (topically adjacent material) -- a harsher, different
              recipe, so we are not tuning to the practice recipe only
  outside@f   noise drawn from anywhere in the corpus outside the pool

Usage (examples):
    python scripts/sweep.py ql --data data/cranfield --mus 250,500,1000,1500,2000,3000
    python scripts/sweep.py ql --data data/cranfield --smoothing jm --jm 0.1,0.3,0.5,0.7,0.9
    python scripts/sweep.py fb --data data/cranfield \
        --variants rm1,rm2,rm3 --lambdas 0,0.2,0.4,0.5,0.6,0.8,1.0 --terms 30 \
        --out runs/fb_sweep.csv
Extra grid axes over any other constant: --axis COHERENCE_POWER=0,1,2.
Any module-level constant of submission.feedback can also be pinned for a
whole sweep with --set NAME=VALUE (repeatable), e.g. --set MIN_SEED_DF=1.
"""
import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import submission.feedback as fb  # noqa: E402
from harness import noise_injection  # noqa: E402
from harness.candidates_io import read_candidates  # noqa: E402
from harness.metrics import evaluate_run  # noqa: E402
from harness.trec_io import read_qrels, read_queries  # noqa: E402

PRF_DEPTH = 10
ADJ_WINDOW = 20

_G = {}  # worker globals (populated before fork)


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v in ("None", "none"):
        return None
    if v in ("True", "False"):
        return v == "True"
    return v


def load(data_dir: str, sets):
    for name, value in sets.items():
        setattr(fb, name, value)
    queries = read_queries(os.path.join(data_dir, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(data_dir, "qrels_dev.txt"))
    cands = read_candidates(os.path.join(data_dir, "candidates_dev.jsonl"))
    t0 = time.perf_counter()
    fb.prepare(os.path.join(data_dir, "corpus.jsonl"))
    prep = time.perf_counter() - t0
    queries = [(q, t) for q, t in queries if q in cands and q in qrels]
    pools = {q: [d for d, _ in cands[q]] for q, _ in queries}
    return queries, qrels, pools, prep


def noisy_seeds(qid, clean, pool, ql_ranking, all_doc_ids, levels, n_seeds):
    """{condition_name: [perturbed seed lists]}"""
    out = {}
    for s in range(n_seeds):
        seed = noise_injection.stable_seed(qid, base_seed=1000 * s)
        for level, ids in noise_injection.sweep(clean, pool, seed=seed, levels=levels):
            out.setdefault(f"uniform@{level:g}", []).append(ids)
    clean_set = set(clean)
    adjacent = [d for d in ql_ranking if d not in clean_set][:ADJ_WINDOW]
    pool_set = set(pool)
    for s in range(n_seeds):
        rng = random.Random(noise_injection.stable_seed(qid, base_seed=7919 + s))
        for level in levels:
            n_rep = round(len(clean) * level)
            for name, source in (("adjacent", adjacent), ("outside", None)):
                ids = list(clean)
                if n_rep:
                    pos = rng.sample(range(len(clean)), n_rep)
                    if source is None:
                        repl = []
                        while len(repl) < n_rep:
                            d = rng.choice(all_doc_ids)
                            if d not in pool_set and d not in repl:
                                repl.append(d)
                    else:
                        repl = rng.sample(source, min(n_rep, len(source)))
                    for p, d in zip(pos, repl):
                        ids[p] = d
                out.setdefault(f"{name}@{level:g}", []).append(ids)
    return out


def _eval_config(cfg):
    for name, value in cfg.items():
        setattr(fb, name, value)
    queries, qrels, pools = _G["queries"], _G["qrels"], _G["pools"]
    seeds = _G["seeds"]
    runs = {}
    worst = 0.0
    for qid, text in queries:
        pool = pools[qid]
        t0 = time.perf_counter()
        runs.setdefault("clean", {})[qid] = fb.relevance_model_feedback(text, seeds[qid]["clean"], pool, 10)
        worst = max(worst, time.perf_counter() - t0)
        for cond, lists in seeds[qid]["noisy"].items():
            for i, ids in enumerate(lists):
                runs.setdefault((cond, i), {})[qid] = fb.relevance_model_feedback(text, ids, pool, 10)
    res = {"clean": evaluate_run(runs["clean"], qrels)["aggregate"]["ndcg@10"]}
    by_cond = {}
    for key, run in runs.items():
        if key == "clean":
            continue
        by_cond.setdefault(key[0], []).append(evaluate_run(run, qrels)["aggregate"]["ndcg@10"])
    for cond, vals in by_cond.items():
        res[cond] = statistics.mean(vals)
    noisy = [v for c, v in res.items() if c != "clean"]
    res["noisy_mean"] = statistics.mean(noisy) if noisy else res["clean"]
    res["retention"] = res["noisy_mean"] / res["clean"] if res["clean"] else 0.0
    res["max_call_s"] = worst
    return cfg, res


def cmd_ql(args, sets):
    queries, qrels, pools, prep = load(args.data, sets)
    print(f"prepare(): {prep:.1f}s, {len(queries)} queries")
    rows = []
    if args.smoothing == "jm":
        grid = [{"SMOOTHING": "jm", "JM_LAMBDA": v} for v in map(float, args.jm.split(","))]
    else:
        grid = [{"SMOOTHING": "dirichlet", "DIRICHLET_MU": v} for v in map(float, args.mus.split(","))]
    for cfg in grid:
        for n, v in cfg.items():
            setattr(fb, n, v)
        run = {q: fb.score_candidates(t, pools[q], 10) for q, t in queries}
        agg = evaluate_run(run, qrels)["aggregate"]
        row = dict(cfg, **{m: round(agg[m], 4) for m in ("ndcg@10", "map@10", "mrr", "p@10")})
        rows.append(row)
        print(row)
    _write(args.out, rows)


def cmd_fb(args, sets):
    queries, qrels, pools, prep = load(args.data, sets)
    print(f"prepare(): {prep:.1f}s, {len(queries)} queries")
    levels = [float(x) for x in args.levels.split(",")]
    all_ids = list(fb._COLL.doc_texts.keys())
    seeds = {}
    for qid, text in queries:
        ql = [d for d, _ in fb.score_candidates(text, pools[qid], len(pools[qid]))]
        clean = ql[:PRF_DEPTH]
        seeds[qid] = {
            "clean": clean,
            "noisy": noisy_seeds(qid, clean, pools[qid], ql, all_ids, levels, args.n_seeds),
        }
    ql_ndcg = evaluate_run({q: fb.score_candidates(t, pools[q], 10) for q, t in queries}, qrels)["aggregate"]["ndcg@10"]
    print(f"QL (no feedback) nDCG@10 = {ql_ndcg:.4f}")
    _G.update(queries=queries, qrels=qrels, pools=pools, seeds=seeds)

    axes = {
        "RM_VARIANT": args.variants.split(","),
        "FB_LAMBDA": [float(x) for x in args.lambdas.split(",")],
        "FB_TERMS": [int(x) for x in args.terms.split(",")],
        "SEED_WEIGHT_TEMP": [float(x) for x in args.temps.split(",")],
        "MIN_SEED_DF": [int(x) for x in args.min_df.split(",")],
        "RM_BASE": args.rm_base.split(","),
    }
    for spec in args.axis:
        name, values = spec.split("=", 1)
        axes[name] = [_parse_value(v) for v in values.split(",")]
    grid = []
    for combo in itertools.product(*axes.values()):
        cfg = dict(zip(axes.keys(), combo))
        if cfg["RM_VARIANT"] != "rm3":
            # λ and RM_BASE are irrelevant for plain RM1/RM2: evaluate once.
            if cfg["FB_LAMBDA"] != axes["FB_LAMBDA"][0] or cfg["RM_BASE"] != axes["RM_BASE"][0]:
                continue
        grid.append(cfg)
    print(f"{len(grid)} configurations")
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for cfg, res in pool.imap(_eval_config, grid):
            row = dict(cfg, ql_ndcg=round(ql_ndcg, 4), **{k: round(v, 4) for k, v in res.items()})
            rows.append(row)
            print(json.dumps(row))
    _write(args.out, rows)


def _write(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ql", "fb"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data/cranfield")
        p.add_argument("--out", default=None)
        p.add_argument("--set", action="append", default=[], help="NAME=VALUE override on submission.feedback")
    q = sub.choices["ql"]
    q.add_argument("--smoothing", default="dirichlet", choices=["dirichlet", "jm"])
    q.add_argument("--mus", default="100,250,500,750,1000,1500,2000,3000")
    q.add_argument("--jm", default="0.1,0.2,0.3,0.5,0.7,0.9")
    f = sub.choices["fb"]
    f.add_argument("--variants", default="rm3")
    f.add_argument("--rm-base", default="rm1")
    f.add_argument("--lambdas", default="0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    f.add_argument("--terms", default="30")
    f.add_argument("--temps", default="1.0")
    f.add_argument("--min-df", default="2")
    f.add_argument("--levels", default="0.25,0.5,0.75")
    f.add_argument("--n-seeds", type=int, default=3)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--axis", action="append", default=[], help="extra grid axis NAME=v1,v2,...")
    args = ap.parse_args()
    sets = {}
    for s in args.set:
        n, v = s.split("=", 1)
        sets[n] = _parse_value(v)
    {"ql": cmd_ql, "fb": cmd_fb}[args.cmd](args, sets)


if __name__ == "__main__":
    main()
