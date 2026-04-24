# Port Checklist

Tracking the port of [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension)
→ `torchid`. Legend: `[x]` done, `[~]` in progress, `[ ]` todo.

## Phase 0 — Repo hygiene

- [x] Update `pyproject.toml` metadata (author, description, deps trimmed to `numpy`/`torch`)
- [x] Remove template scripts
- [x] Add `validation` dep group (`scikit-dimension`, `scikit-learn`, `pandas`, `matplotlib`)
- [x] Rewrite `README.md`

## Phase 1 — Core primitives (`src/torchid/_primitives.py`)

- [x] `pairwise_sqdist(X, Y=None, chunk=...)` — chunked, bounded memory
- [x] `knn(X, k, chunk=...)` — returns `(dists, idx)` (verified vs sklearn)
- [x] `gather_neighbors(X, idx)` → `(N, k, d)`
- [x] `batched_local_pca(X_nbrs)` — batched SVD → eigenvalues
- [x] `log_knn_ratios(dists)` — shared by MLE/TwoNN/MiND_ML/MADA
- [x] `sample_combinations(k, p, m)` — for ESS

## Phase 2 — Base + infra

- [x] `estimators/base.py` — `GlobalEstimator`, `LocalEstimator`, `.fit(X)`, `.dimension_`, device/dtype
- [x] `datasets.py` — focused subset (hyperball, hypersphere, affine subspace w/ noise, swiss roll). Full 23-manifold port deferred (skdim.datasets is available via the validation group)
- [x] `parity.py` — sklearn-dim parity harness + per-estimator tolerance table

## Phase 3 — Estimator ports (12 total)

Order: easy GPU wins first, then per-point-loop-heavy, then exotic.

- [x] lPCA (FO, Fan, maxgap, ratio, Kaiser, broken_stick, participation_ratio) — parity green on all 7 heuristics
- [x] TwoNN — parity green
- [x] MLE (Levina–Bickel closed form) — parity green across `comb` × `unbiased`; skdim's py313 bug worked around via a patched subclass in the test
- [x] CorrInt — parity green
- [x] MiND_ML (MLi + MLk via dense-grid refinement) — parity green
- [x] MOM — parity green (note: default `_N_NEIGHBORS=100` to match skdim)
- [x] MADA — implementation done, parity pending final suite run
- [x] TLE — implementation done (fully batched (N, k, k) tensor ops), parity pending
- [x] KNN — implementation done; integer output so parity is ±1 on the small-sample cases
- [x] DANCo — implementation done; calibration uses torch hyperballs (different RNG than skdim numpy → loose tolerance expected)
- [x] ESS — implementation done for ver='a'/'b' at d=1 (closed form); d>1 ver='a' via Monte-Carlo p-subsets
- [x] FisherS — implementation done; Lambert-W inversion still numpy (~20 scalars, not worth porting)
- [ ] TwoNN
- [ ] MLE (Levina–Bickel closed form; skip `scipy.integrate.quad`)
- [ ] CorrInt
- [ ] MiND_ML
- [ ] MOM
- [ ] MADA
- [ ] KNN (graph-length regression, batched bootstrap)
- [ ] TLE
- [ ] DANCo (angle concentration + KL calibration, batched)
- [ ] ESS (Monte-Carlo p-subset sampling)
- [ ] FisherS

For each estimator:

- [ ] implementation in `estimators/<name>.py`
- [ ] parity test `tests/test_parity_<name>.py` (vs scikit-dimension)
- [ ] manifold-recovery test
- [ ] benchmark script `benchmarks/bench_<name>.py`

## Phase 4 — Benchmarks

- [x] Unified runner at `benchmarks/bench.py` covering all 12 estimators
- [x] `BENCHMARKS.md` seeded with `--small` sweep (n ∈ {500, 2000}, D=20) — CPU + RTX 3090
- [x] CUDA-enabled numbers (torch.cuda.Event timing + peak memory) — headline **6393× ESS**, **248× MADA**, **98× TLE**, **69× FisherS**
- [ ] Larger sweep (n ∈ {10k, 100k}, D ∈ {100, 1000})
- [ ] Log-log plots

## Phase 5 — Polish

- [ ] Update `README.md` with real examples + benchmark headline numbers
- [ ] Ensure `ruff check .`, `ty check`, `pytest` all green in CI
- [ ] Tag `v0.1.0` on PyPI once all estimators + parity tests pass

## Phase 6 — Docs site

- [x] Zensical config (`mkdocs.yml`) mirrored from `contourrs`
- [x] `docs/{index,getting-started,estimators,api,architecture,performance,parity}.md`
- [x] `.github/workflows/docs.yml` — auto-deploy to GitHub Pages on push-to-main
- [x] Local build verified (`uv run --with zensical zensical build --clean`)

## Phase 7 — faiss-cpu CPU knn backend

- [x] `faiss-cpu` added as runtime dep; `_primitives.knn` dispatches on device type
- [x] GPU path unchanged (pure torch)
- [x] All 25 parity tests still green after the swap
- [x] Re-ran large sweep; headline CPU speedups: MADA 77×, TwoNN 10×, MLE/CorrInt 4×, TLE 14×

## Known loose ends

- All 25 parity tests green. ESS and DANCo use tighter (n=500) test cases because skdim's per-point Python loops make n=2000 impractical in the test suite.
- KNN uses numpy RNG (`np.random.randint`) to share a stream with skdim's bootstrap; this is the one place we can't use torch's generator without breaking parity.
- `scipy.special.lambertw` is still imported inside FisherS (called on ~20 scalar values — not a perf bottleneck).
- `torch.special.digamma/i0/i1` required for DANCo; available in torch ≥ 2.x.
- skdim's `MLE.__init__` mutates `frame.f_locals` which is read-only on Python 3.13; the parity test and benchmark both construct skdim MLE via `__new__` + manual attribute assignment as a workaround.
