#!/usr/bin/env bash
# Run the same checks CI runs, locally, before you push.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Full test suite =="
python3 -m pytest tests/ -v

echo
echo "== Full harness run on the toy set =="
python3 -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --candidates data/toy/candidates_dev.jsonl \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json

echo
echo "All smoke checks passed."
