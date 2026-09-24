#!/usr/bin/env python
"""
Rebuild the BEIR TREC-COVID collection (the assignment's full corpus, i.e.
what `scripts/download_full_corpus.py` fetches via ir_datasets) WITHOUT
ir_datasets, from sources reachable when the BEIR / HuggingFace hosts are not:

  corpus   CORD-19 2020-07-16 metadata.csv (AI2's public S3 bucket): the
           snapshot BEIR's trec-covid corpus was built from. One document
           per unique cord_uid (first row wins), text = title + " " + abstract
           (ir_datasets' default_text() for beir docs is also title + text).
  queries  topics.beir-v1.0.0-trec-covid.test.tsv.gz  (castorini/anserini-tools)
  qrels    qrels.beir-v1.0.0-trec-covid.test.txt       (castorini/anserini-tools)

Writes <out>/corpus.jsonl, queries_dev.tsv, qrels_dev.txt. The staff
candidates_dev.jsonl is NOT reproducible (undisclosed retriever); build a
stand-in BM25 top-100 with:
    python scripts/build_cranfield_proxy.py --candidates-only --out data/full

Usage:
    python scripts/build_trec_covid.py [--out data/full] [--src-dir DIR]
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import urllib.request

CORD19 = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv"
TOOLS = "https://raw.githubusercontent.com/castorini/anserini-tools/master/"
TOPICS = "topics/topics.beir-v1.0.0-trec-covid.test.tsv.gz"
QRELS = "qrels/qrels.beir-v1.0.0-trec-covid.test.txt"


def _get(url: str, src_dir: str, name: str) -> bytes:
    if src_dir:
        with open(os.path.join(src_dir, name), "rb") as f:
            return f.read()
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "full"))
    ap.add_argument("--src-dir", default=None,
                    help="read metadata.csv, topics .tsv.gz and qrels .txt (basenames) from here")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csv.field_size_limit(sys.maxsize)
    meta = _get(CORD19, args.src_dir, "metadata.csv").decode("utf-8")
    seen = set()
    with open(os.path.join(args.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for row in csv.DictReader(io.StringIO(meta)):
            uid = row["cord_uid"]
            if uid in seen:
                continue
            seen.add(uid)
            text = " ".join(f"{row['title'] or ''} {row['abstract'] or ''}".split())
            f.write(json.dumps({"doc_id": uid, "text": text}) + "\n")
    print(f"wrote {len(seen)} docs")

    topics = gzip.decompress(_get(TOOLS + TOPICS, args.src_dir, os.path.basename(TOPICS))).decode("utf-8")
    n = 0
    with open(os.path.join(args.out, "queries_dev.tsv"), "w", encoding="utf-8") as f:
        for line in topics.splitlines():
            if line.strip():
                qid, text = line.split("\t", 1)
                f.write(f"{qid}\t{' '.join(text.split())}\n")
                n += 1
    print(f"wrote {n} queries")

    qrels = _get(TOOLS + QRELS, args.src_dir, os.path.basename(QRELS)).decode("utf-8")
    kept = 0
    with open(os.path.join(args.out, "qrels_dev.txt"), "w", encoding="utf-8") as f:
        for line in qrels.splitlines():
            parts = line.split()
            if len(parts) == 4:
                qid, _it, doc_id, rel = parts
                f.write(f"{qid} 0 {doc_id} {max(int(rel), 0)}\n")
                kept += 1
    print(f"wrote {kept} qrels")


if __name__ == "__main__":
    main()
