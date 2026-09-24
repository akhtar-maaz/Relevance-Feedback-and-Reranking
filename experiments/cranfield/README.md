Sweep outputs from `scripts/sweep.py` on the Cranfield proxy dev set, which `scripts/build_cranfield_proxy.py` builds (BM25 top-100 pools, 225 queries). These are the raw data for the report.

- `ql_dir.csv` / `ql_jm.csv`: Track A. The Dirichlet μ sweep and the JM λ sweep, with stemming and query stopwords.
- `fb_first.csv`: RM1 vs RM2 vs RM3 over λ, with μ = 500.
- `fb_grid.csv`: RM3 over λ × terms × seed-weight temperature × min seed-df × RM1/RM2 base, with μ = 500.
- `fb_coh.csv`: coherence-weighting ablation, with auto μ = 3 × avgdl.
- `fb_refine.csv`: final neighbourhood, including temp = 1000 (≈ uniform seed weights). Three noise draws per condition.
- `adversarial_cranfield.json`: the adversarial builder's output on Cranfield, for validation only.

Noise columns:
- `uniform@f`: the public practice recipe.
- `adjacent@f`: replacements drawn from QL ranks 11–30.
- `outside@f`: replacements drawn from outside the pool.

`retention` = the mean over all noisy columns ÷ `clean`.
