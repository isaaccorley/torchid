# Estimators

Twelve intrinsic-dimension estimators, ported one-for-one from scikit-dimension. All live under `torchid.estimators`. Split by output type:

## Global estimators — single scalar per dataset

### `lPCA`
PCA-eigenvalue-based. Fits a covariance, then applies one of 7 thresholding heuristics: `FO` (Fukunaga-Olsen), `Fan`, `maxgap`, `ratio`, `participation_ratio`, `Kaiser`, `broken_stick`. The workhorse of the library and the fastest estimator we have.

Reference: Cangelosi & Goriely 2007; Fan et al. 2010; Johnsson 2015.

```python
lPCA(ver="FO", alphaFO=0.05).fit(X).dimension_
```

### `TwoNN`
Two-nearest-neighbor ratio method. Under a locally uniform density, `-log(1 - Femp(mu))` is linear in `log(mu)` with slope = intrinsic dimension.

Reference: Facco et al. 2017.

### `MLE`
Levina–Bickel maximum-likelihood estimator. Closed form:
`d_i = (k - 1) / Σ log(R_k(i) / R_j(i))`.
Aggregated by harmonic mean (default), mean, or median.

Only the zero-noise branch is implemented — the Haro integral (skdim's `integral_approximation="Haro"` with a non-zero `sigma`) used `scipy.integrate.quad` per neighbor, which is what made skdim's MLE slow in the first place.

Reference: Levina & Bickel 2005; Haro et al. 2008.

### `CorrInt`
Correlation-integral (Grassberger–Procaccia). Estimates
`d = [log C(r2) − log C(r1)] / log(r2 / r1)`
where `C(r)` is the fraction of pairwise distances below `r`.

### `MiND_ML`
Maximum-likelihood integer / continuous dimension from `ρ_i = d_1(i) / d_k(i)`. `ver="MLi"` returns the best integer; `ver="MLk"` refines via a 1-D dense grid (in place of `scipy.optimize.L-BFGS-B`).

Reference: Rozza et al. 2012.

### `KNN`
k-nearest-neighbor graph-length regression (Carter et al. 2010). Bootstrapped least-squares over `m = 1..D`. Uses `np.random.randint` internally to share the RNG stream with skdim's parity tests.

### `DANCo`
Dimensionality from Angle and Norm Concentration. Compares observed `(d_hat, μ_nu, μ_tau)` from the data's pairwise angles against a calibration set generated from `D` synthetic hyperballs. The candidate dimension that minimizes KL wins; fractal mode interpolates.

Reference: Ceruti et al. 2012.

### `FisherS`
Fisher-separability-based. Projects onto a sphere, counts inseparable-point fractions across a grid of α thresholds, inverts the theoretical probability via Lambert-W.

Reference: Albergante et al. 2019.

## Local estimators — per-point tensor

Local estimators expose both a scalar `dimension_` (aggregated) and a `dimension_pw_` tensor of shape `(N,)`.

### `MOM`
Method of moments: `d_i = -m1 / (m1 - w)` where `w = d_k`, `m1 = mean(d_j)`. Defaults to `n_neighbors=100` to match skdim's LocalEstimator default.

Reference: Amsaleg et al. 2018.

### `MADA`
Manifold-adaptive dimension — ratio of k-NN to k/2-NN distances:
`d_i = log 2 / log(R_k / R_k/2)`.

Reference: Farahmand & Szepesvári 2007.

### `TLE`
Tight Local Estimator — builds a `(k, k)` matrix of pairwise neighbor distances per point and applies a closed-form formula with several boundary-case adjustments. torchid does this in a single batched `(N, k, k)` tensor operation.

Reference: Amsaleg et al. 2019.

### `ESS`
Expected Simplex Skewness. For `d=1, ver='a'` (the default and dominant case), the per-neighborhood ESS collapses to a mean |sin(angle)| over all pairs of centered vectors — a single batched matmul + triu selection. For `d>1` we fall back to Monte-Carlo p-subset sampling.

Reference: Johnsson et al. 2015.

## Common parameters

Most estimators share:

- `n_neighbors` (default 20 for `MLE`, `MADA`, `TLE`; 100 for `MOM`, `ESS`).
- Device/dtype are inherited from the input tensor; no `.to()` needed.
- All estimators return `self` from `fit()` so chaining works.

## Not yet ported

- Multi-dataset batched fitting (`estimate_many(X_list)`) — scope parked for a future release.
- skdim's `asPointwise(X, global_est)` wrapper — trivial to reimplement once needed.
- The `fit_explained_variance=True` shortcut on `lPCA` — supported, but parity with the skdim semantic is not tested.
