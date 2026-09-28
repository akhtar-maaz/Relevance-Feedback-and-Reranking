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

## Rank fusion after the second leaderboard result

The second screenshot showed rank 14, Track A 0.4960, Track B 0.4990,
Track C retention 0.9300, and final score 0.1257. This was a small gain
from rank 15, Track B 0.4896, retention 0.8565, and final score 0.0829.
Track A did not move because the previous change affected only feedback.

The supplied candidate list is already ranked by a first-pass retriever,
but its scores are not passed to the submission. The new base ranks every
candidate by Dirichlet query likelihood, then combines its rank with the
candidate's original position: `first_pass_rank + 0.5 * QL_rank` (smaller
is better). Feedback still estimates RM1 and interpolates RM3. Its final
ranking combines the new base rank with the RM3 rank as
`base_rank + 0.25 * RM3_rank` for the 100-document public pools. For very
small pools the feedback weight rises, so the supplied seed can still
change the result. This is a real, seed-dependent feedback ranking, with
the query and the first pass anchoring noisy estimates.

On all 50 public development topics, the unchanged local harness reported:

| Measure | Previous submission | Rank fusion |
|---|---:|---:|
| Track A nDCG@10 | 0.531282 | 0.573883 |
| Track B clean nDCG@10 | 0.585438 | 0.583922 |
| Track C public practice retention | not comparable to the five-seed result below | 0.9994 |

The previous Track B figure is from the five-seed sensitivity study,
whose clean condition was evaluated once. The public practice retention
above uses the harness's two default noisy conditions and one draw each.
In an additional five-draw sweep, the new model scored 0.583922 clean,
0.584011 at requested 25% noise, 0.584820 at 50%, 0.583940 at 75%, and
0.583144 at 100%. The noisy-to-clean ratio across those four noisy
conditions is approximately 1.0001. Results, per-query measurements, and
repeat variation are in the ignored local file
`runs/full-dev/rank_fusion_sensitivity.json`.

Rank weights were explored after looking at both groups of the earlier
35/15 split. Those groups are no longer independent validation evidence
for this change. The class leaderboard uses held-out topics, a separate
noise recipe, and percentile scores, so this result does not guarantee
a similar leaderboard gain. The candidate order may also be less useful
on another pool. The unigram likelihood and RM1/RM3 computations remain
independently inspectable in `submission/feedback.py`.

## Opening-text query likelihood after the third leaderboard result

The next leaderboard submission improved to Track A 0.5021, MAP 0.4267,
Track B 0.5027, retention 0.9968, and final score 0.1929. Retention was
already strong, while the raw accuracy remained below the leading teams.
The next development change therefore targeted ranking accuracy.

The tested base used the supplied candidate order, full-text query
likelihood, and a separate Dirichlet query likelihood over the first 15
tokens of each document. These opening tokens often contain the paper
title in this corpus. The fusion is `candidate_rank + 0.1 * full_QL_rank`,
then `base_rank + 0.15 * opening_QL_rank`. The opening model uses mu 100.
Feedback still uses RM1/RM3 from the supplied seed; its rank weight is
0.1 for the 100-document public pools to reduce drift.

| Public development measure | Previous commit | Opening-text model |
|---|---:|---:|
| Track A nDCG@10 | 0.573883 | 0.623566 |
| Track B clean nDCG@10 | 0.583922 | 0.625311 |
| Public practice retention, default draws | 0.9994 | 1.0000 |

Across five separate public noise draws, the new model scored 0.625311
clean, 0.625711 at requested 25% noise, 0.626344 at 50%, 0.623728 at
75%, and 0.616901 at 100%. The average noisy-to-clean ratio across those
four noisy conditions is about 0.9966. The complete local measurements
are in ignored `runs/full-dev/lead_fusion_final_report.json` and
`runs/full-dev/lead_fusion_final_sensitivity.json`.

Several opening-window sizes and rank weights were explored on the same
50 public topics, including the earlier 35/15 split. The split was not
an untouched holdout for this change, and the public recipe differs from
the grading recipe.

The hidden leaderboard result for this version was substantially worse:
Track A 0.3915, MAP 0.2843, Track B 0.3974, retention 0.9991, and final
score 0.0750. The large public gains failed to transfer to hidden topics.
The opening-text model and its changed rank weights were removed; the
submission code was restored to commit `491d357`, whose leaderboard score
was 0.1929. The later dependency-only conformance fix remains in place.

## Conservative rank-fusion trial

The restored leaderboard result was Track A nDCG@10 0.5021, MAP@10
0.4267, Track B nDCG@10 0.5027, retention 0.9968, and final score
0.1929. Several leading leaderboard rows had equal Track A and Track B
scores to four decimal places and retention 1.0000. This is consistent
with very conservative feedback, although their implementations are not
visible and this pattern does not establish what they did.

This trial changes only two rank weights: the full-text query-likelihood
rank receives weight 0.1 instead of 0.5 against the supplied first-pass
candidate order, and the RM3 feedback rank receives weight 0.1 instead
of 0.25 against the base rank. RM1/RM3 estimation and all corpus/query
processing remain the same. Query likelihood still changed the top-10
ordering on 24 of 50 public topics and brought 8 documents into the
top 10 that were not in the first-pass top 10.

| Public development measure | Restored version | Conservative trial |
|---|---:|---:|
| Track A nDCG@10 | 0.573883 | 0.584466 |
| Track B clean nDCG@10 | 0.583922 | 0.585689 |
| Public practice retention, default draws | 0.9994 | 1.0000 |

Across five public noise draws, clean feedback scored 0.585689;
requested 25%, 50%, 75%, and 100% noise scored 0.587434, 0.588407,
0.586811, and 0.585743 respectively. The mean noisy-to-clean ratio
exceeded 1 and is capped at 1.0 by the leaderboard's retention rule.
The complete local measurements are in ignored
`runs/full-dev/conservative_fusion_report.json` and
`runs/full-dev/conservative_fusion_sensitivity.json`.

The same public topics have been inspected repeatedly, so these results
are exploratory. On the hidden leaderboard this trial reached Track A
nDCG@10 0.5028, MAP@10 0.4273, Track B nDCG@10 0.5018, retention
0.9990, and final score 0.2440. This was a higher final score than the
restored 0.1929 version, although the class-relative score can also
change when other teams submit.

## Three-round model study

The configured feedback weight was briefly changed from 0.1 to 0.05.
The code enforces `max(configured_weight, 10 / candidate_count)` for
feedback rank fusion, so both settings have the same effective 0.1
weight on the public 100-document pools. That trial did not test a
different public model. A separate scratch comparison lowered the floor
to make 0.05 effective; it had mixed results across three held-out topic
folds and was not promoted. The configured weight was restored to 0.1,
matching the best known hidden result of 0.2440.

One agent compared a fixed compact grid covering RM1/RM2, Dirichlet
smoothing, original-query anchoring, candidate-order weight, and feedback
rank weight. The 50 public topics were split into three folds of 17, 17,
and 16 topics. Each model was measured on two training folds and the
remaining test fold in each of three rounds, with independent train/test
noise draws at 50% and 100% replacement. All feedback calls used the
provided seeds and candidate pools.

| Model | Mean test Track A nDCG@10 | Mean test Track B nDCG@10 | Mean capped retention |
|---|---:|---:|---:|
| Current effective feedback weight 0.1 | 0.5841 | 0.5854 | 0.9947 |
| Feedback weight 0.15 | 0.5841 | 0.5915 | 0.9907 |
| Feedback weight 0.2 | 0.5841 | 0.5945 | 0.9905 |
| Dirichlet mu 1000 | 0.5852 | 0.5898 | 0.9919 |

Stronger feedback improved clean accuracy in each test fold but reduced
retention in some folds. None of the tested settings improved a proxy
weighted like the public leaderboard components in all three test folds.
The proxy used raw metrics; the real leaderboard percentile-ranks class
results, so this study cannot predict a hidden score. Local detail is in
ignored `runs/full-dev/three_round_study.json` and
`runs/full-dev/half_feedback_floor.json`.
