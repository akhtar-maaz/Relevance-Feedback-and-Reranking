`scripts/sweep.py` outputs on TREC-COVID. `scripts/build_trec_covid.py` rebuilds the collection from CORD-19 2020-07-16 plus the BEIR topics and qrels: 191,175 docs, which is the official round-5 collection, and 50 queries. The candidate pools are our own BM25 top-100 (`scripts/build_cranfield_proxy.py --candidates-only --out data/full`), a stand-in for the staff `candidates_dev.jsonl`. The columns mean the same as in `../cranfield/README.md`.

- `covid_ql_dir.csv` / `covid_ql_jm.csv`: Track A, Dirichlet μ vs JM λ.
- `covid_fb.csv`: RM1/RM2/RM3 × λ × {20, 45, 80} terms, with MIN_SEED_DF = 2 and 3 noise draws.
- `covid_ablation.csv`: λ = 0.5 and 45 terms, crossing seed-weight temperature (1 vs 1000 ≈ uniform) × min seed-df × coherence.
