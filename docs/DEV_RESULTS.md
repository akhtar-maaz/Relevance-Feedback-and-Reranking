# Full development evaluation — 28 September 2026

The supplied `COL/candidates_dev.jsonl` contains 50 query pools of 100
candidates each (4,408 unique documents). Every ID was verified against
the 171,332-document `beir/trec-covid` collection obtained using the
assignment's downloader with `ir_datasets==0.6.3`. All 50 topics match the
66,336 relevance-judgment records. The original candidate file was
preserved and copied to `data/full/candidates_dev.jsonl`.

## Selected settings

| Setting | Previous | Selected |
|---|---:|---:|
| Base Dirichlet mu | 1500 | 500 |
| Feedback Dirichlet mu | 10 | 300 |
| Estimator feeding RM3 | RM2 | RM1 |
| RM3 original-query weight | 0.5 | 0.25 |
| Expansion terms | 20 | 20 |
| Feedback seed depth | 10 | 10 |

The chosen configuration improved aggregate feedback accuracy. Stronger
feedback smoothing gives the collection model more influence on the seed
language models, while the smaller query weight gives expansion more
influence in RM3. These changes were evaluated together; this experiment
does not isolate a causal contribution for every individual setting.

## Reserved-topic validation

The fixed protocol in `scripts/tune_feedback.py` selected parameters on
35 topics and checked only the selected configuration and the previous
reference on the other 15. The split uses seed 7364 and was recorded
before scores were examined. Training noise seeds were 0–2; validation
used fresh seeds 101–105. Clean feedback was evaluated once; noisy numbers
below average five draws.

| Condition | Previous nDCG@10 | Selected nDCG@10 | Change |
|---|---:|---:|---:|
| Base query likelihood | 0.462925 | 0.440159 | -0.022766 |
| Clean feedback | 0.432127 | 0.502741 | +0.070614 |
| 25% requested noise (20% actual) | 0.451799 | 0.516126 | +0.064327 |
| 50% noise | 0.470263 | 0.491629 | +0.021366 |

For the equal-weight mean across clean and both noisy conditions, the
selected model gained 0.052102 nDCG@10 on the reserved
set. Topic outcomes were 5 wins, 2 ties, and
8 losses: the positive aggregate gain is concentrated in
some topics, not a uniform improvement. Base query-likelihood accuracy
also fell on this partition. These 15 public development topics are a
limited check, not the course's private test set or a guarantee of
statistical significance.

## All 50 development topics

The unchanged public harness was then run once per configuration using
its default noise seed. These numbers include the 35 tuning topics, so
use the reserved-topic comparison above when assessing generalization.

| Condition | Previous nDCG@10 | Selected nDCG@10 | Change |
|---|---:|---:|---:|
| Base query likelihood | 0.532472 | 0.531282 | -0.001191 |
| Clean feedback | 0.539225 | 0.590632 | +0.051407 |
| 25% requested noise (20% actual) | 0.557699 | 0.587183 | +0.029484 |
| 50% noise | 0.542847 | 0.583669 | +0.040822 |

Practice retention changed from 1.000000
to 0.991185, because clean accuracy increased
more than noisy accuracy. Retention alone was not the tuning objective.
The hidden leaderboard uses class percentiles and an undisclosed noise
recipe; these are raw development measurements only. Actual replacement
at the requested 25% level is 2 of 10 seed documents under the public
rounding rule.

## Reproduce and inspect

Run the bounded comparison from the repository root:

```bash
python3 scripts/tune_feedback.py \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --candidates data/full/candidates_dev.jsonl \
  --output runs/full-dev/tuning.json
```

The original reference is fixed in that script and is independent of the
current submission defaults. The [experiment guide](EXPERIMENTS.md)
explains the predeclared search grid, objective, complete qrels, and paired
noise sampling. No qrels or topic-specific rules are used by the submission
when ranking documents.

Local results are in `runs/full-dev/`: `data_manifest.json` records input
checksums, `tuning.json` records every training row, topic partition,
validation repeat, and paired per-topic change; `before_report.json` and
`after_report.json` hold the full-harness results. Data and local run
artifacts are ignored by Git. A compact aggregate record is kept in
[dev_results.json](dev_results.json).

All 53 tests pass after applying these settings. The full corpus prepared
in about 7.6 seconds in the final
harness run on this machine. Dependencies were installed in a temporary
local environment; the submission itself still uses only the standard
library and provided helpers.
