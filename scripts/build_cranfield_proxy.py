#!/usr/bin/env python
"""
Build a local, realistic stand-in dev set for tuning while the real
assignment dev candidates are unavailable.

Downloads the Cranfield collection (1400 docs, 225 queries, qrels) from a
public GitHub mirror and writes, in exactly the toy/full formats:

    <out>/corpus.jsonl
    <out>/queries_dev.tsv
    <out>/qrels_dev.txt
    <out>/candidates_dev.jsonl   <- top-K per query from a from-scratch BM25
                                    first pass (our own stand-in for the
                                    undisclosed reference retriever)

BM25 deliberately uses NO stemming and NO stopword removal, so that the
first-pass ranker is "unremarkable" and differs from our own reranker
(which does stem) -- the candidate pools therefore have realistic, not
self-serving, recall.

Usage:
    python scripts/build_cranfield_proxy.py                 # -> data/cranfield
    python scripts/build_cranfield_proxy.py --top-k 100 --out data/cranfield
    python scripts/build_cranfield_proxy.py --src-dir /path/with/cran/files   # offline

Also usable to generate candidates for ANY corpus/queries pair in the
standard format (e.g. data/full, if the staff candidates file is late):
    python scripts/build_cranfield_proxy.py --candidates-only \
        --out data/full --top-k 100
"""
import argparse
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness.candidates_io import write_candidates  # noqa: E402
from harness.trec_io import read_corpus, read_queries  # noqa: E402

MIRROR = "https://raw.githubusercontent.com/oussbenk/cranfield-trec-dataset/main/"
FILES = ["cran.all.1400.xml", "cran.qry.xml", "cranqrel.trec.txt"]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fetch(name: str, src_dir: str) -> str:
    if src_dir:
        with open(os.path.join(src_dir, name), "r", encoding="utf-8") as f:
            return f.read()
    with urllib.request.urlopen(MIRROR + name, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _clean(s: str) -> str:
    return " ".join(s.split())


def write_cranfield(out: str, src_dir: str) -> None:
    docs_xml = _fetch(FILES[0], src_dir)
    qry_xml = _fetch(FILES[1], src_dir)
    qrels_txt = _fetch(FILES[2], src_dir)

    n = 0
    with open(os.path.join(out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for block in re.findall(r"<doc>(.*?)</doc>", docs_xml, flags=re.S):
            docno = re.search(r"<docno>\s*(\d+)\s*</docno>", block).group(1)
            text = re.search(r"<text>(.*?)</text>", block, flags=re.S)
            title = re.search(r"<title>(.*?)</title>", block, flags=re.S)
            body = _clean(text.group(1)) if text else ""
            if not body and title:
                body = _clean(title.group(1))
            f.write(json.dumps({"doc_id": docno, "text": body}) + "\n")
            n += 1
    print(f"wrote {n} docs")

    # cranqrel numbers queries 1..225 by POSITION in cran.qry, not by <num>.
    tops = re.findall(r"<top>(.*?)</top>", qry_xml, flags=re.S)
    with open(os.path.join(out, "queries_dev.tsv"), "w", encoding="utf-8") as f:
        for i, block in enumerate(tops, start=1):
            title = re.search(r"<title>(.*?)</title>", block, flags=re.S).group(1)
            f.write(f"{i}\t{_clean(title)}\n")
    print(f"wrote {len(tops)} queries")

    kept = 0
    with open(os.path.join(out, "qrels_dev.txt"), "w", encoding="utf-8") as f:
        for line in qrels_txt.splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _it, doc_id, rel = parts
            rel = int(rel)
            if rel <= 0:
                continue
            f.write(f"{qid} 0 {doc_id} {min(rel, 2)}\n")
            kept += 1
    print(f"wrote {kept} positive qrels")


def bm25_candidates(corpus_path: str, queries_path: str, top_k: int, k1: float = 0.9, b: float = 0.4):
    docs = read_corpus(corpus_path)
    postings = defaultdict(list)  # term -> [(doc_idx, tf)]
    lengths = []
    for idx, (_doc_id, text) in enumerate(docs):
        counts = Counter(_TOKEN_RE.findall(text.lower()))
        lengths.append(sum(counts.values()))
        for term, tf in counts.items():
            postings[term].append((idx, tf))
    n_docs = len(docs)
    avgdl = (sum(lengths) / n_docs) if n_docs else 1.0

    out = {}
    for qid, text in read_queries(queries_path):
        scores = defaultdict(float)
        for term in set(_TOKEN_RE.findall(text.lower())):
            plist = postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, tf in plist:
                norm = tf + k1 * (1 - b + b * lengths[idx] / avgdl)
                scores[idx] += idf * tf * (k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda p: (-p[1], docs[p[0]][0]))[:top_k]
        out[qid] = [(docs[i][0], s) for i, s in ranked]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "cranfield"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--src-dir", default=None, help="read the three cran files from here instead of downloading")
    ap.add_argument("--candidates-only", action="store_true", help="only (re)build candidates_dev.jsonl in --out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.candidates_only:
        write_cranfield(args.out, args.src_dir)
    cands = bm25_candidates(
        os.path.join(args.out, "corpus.jsonl"), os.path.join(args.out, "queries_dev.tsv"), args.top_k
    )
    write_candidates(os.path.join(args.out, "candidates_dev.jsonl"), cands)
    print(f"wrote BM25 top-{args.top_k} candidates for {len(cands)} queries")


if __name__ == "__main__":
    main()
