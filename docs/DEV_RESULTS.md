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

The original 0.25 configuration passed all 53 tests. The follow-up changed
only the RM3 query weight to 0.5; the expanded, five-seed public-data
evaluation was rerun with this weight. The full corpus prepared in about
7.6 seconds in the original harness run on this machine. Dependencies
were installed in a temporary local environment; the submission itself
still uses only the standard library and provided helpers.


## Follow-up after the leaderboard screenshot

The screenshot after the previous commit showed rank 15, Track A nDCG@10
0.4960, Track B nDCG@10 0.4896, and Track C retention 0.8565. Because
those topics and the grading noise recipe are held out, the public scores
cannot pinpoint the cause. The low retention motivated testing a stronger
query anchor on the available development topics.

With QL mu 500, feedback mu 300, RM1, and 20 expansion terms held fixed,
the evaluator compared query weights 0.25 and 0.5 under five paired noise
draws at requested replacement levels 0%, 25%, 50%, 75%, and 100%. It
uses the same 50 public topics as the earlier evaluation, so these results
are a follow-up sensitivity check, not a fresh holdout. At seed depth 10,
requested 25% again replaces 2 documents (20% actual).

| Requested noise | Lambda 0.25 nDCG@10 | Lambda 0.5 nDCG@10 | Change |
|---|---:|---:|---:|
| Clean | 0.590632 | 0.585438 | -0.005194 |
| 25% (20% actual) | 0.589965 | 0.589392 | -0.000573 |
| 50% | 0.576737 | 0.583118 | +0.006381 |
| 75% | 0.534893 | 0.562460 | +0.027567 |
| 100% | 0.514539 | 0.548298 | +0.033760 |

Across the five conditions, mean nDCG@10 rose from 0.561353
to 0.573741. Mean noisy nDCG divided by clean nDCG rose
from 0.9380 to 0.9750. The 0.5 query weight loses
-0.005194 clean nDCG@10 and gains more as
contamination grows. This is why the current default uses 0.5: it retains
most of the clean gain while reducing drift in this extended practice
stress test. At lambda 1, feedback is exactly the base query-likelihood
ranking.

The follow-up report records all seven tested query weights, repeat
variation, per-query scores, settings, and source/input hashes in the
ignored local file `runs/full-dev/anchor_sensitivity_verified.json`. The
reproduction command is in [EXPERIMENTS.md](EXPERIMENTS.md). This provides
evidence about public development topics only; it does not establish that
the next hidden leaderboard result will improve.
