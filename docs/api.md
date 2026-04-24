# API Reference

## `torchid.estimators`

All twelve estimators follow a uniform API:

```python
est = Estimator(**params).fit(X)   # X: (N, D) torch.Tensor, numpy, or array-like
est.dimension_                     # float
est.dimension_pw_                  # torch.Tensor of shape (N,), for local estimators
```

### Global estimators

```python
class lPCA(ver: str = "FO",
           alphaRatio: float = 0.05,
           alphaFO: float = 0.05,
           alphaFan: float = 10.0,
           betaFan: float = 0.8,
           PFan: float = 0.95,
           fit_explained_variance: bool = False)
```

`ver` ∈ {`FO`, `Fan`, `maxgap`, `ratio`, `participation_ratio`, `Kaiser`, `broken_stick`}.
Attributes after `fit`: `dimension_`, `explained_var_` (torch.Tensor), `gap_` (np.ndarray).

```python
class TwoNN(discard_fraction: float = 0.1, dist: bool = False)
```

Attributes: `dimension_`, `x_` (log(mu)), `y_` (-log(1 - Femp)).

```python
class MLE(unbiased: bool = False, ...)
```

`fit(X, *, n_neighbors: int | None = None, comb: str = "mle")`; `comb` ∈ {`mle`, `mean`, `median`}. Only the zero-noise branch of skdim's MLE is supported.

```python
class CorrInt(k1: int = 10, k2: int = 20, DM: bool = False)
```

```python
class MiND_ML(k: int = 20, D: int = 10, ver: str = "MLk")
```

`ver` ∈ {`MLi`, `MLk`}.

```python
class KNN(k: int | None = None, ps: np.ndarray | None = None, M: int = 1, gamma: int = 2)
```

Uses `np.random.randint` internally; seed with `np.random.seed` for reproducibility.

```python
class DANCo(k: int = 10,
            D: int | None = None,
            calibration_data: dict | None = None,
            ver: str = "DANCo",
            fractal: bool = True,
            random_state: int | None = None)
```

`ver` ∈ {`DANCo`, `MIND_MLi`, `MIND_MLk`}. Calibration data is generated on-device from torch hyperballs; reused across calls when `calibration_data` is supplied.

```python
class FisherS(conditional_number: float = 10.0,
              project_on_sphere: bool = True,
              alphas: np.ndarray | None = None,
              limit_maxdim: bool = False)
```

Lambert-W inversion is delegated to `scipy.special.lambertw` on a ~20-element alpha grid.

### Local estimators

```python
class MOM(n_neighbors: int | None = None)            # default _N_NEIGHBORS = 100
class MADA(DM: bool = False, n_neighbors: int | None = None)   # default 20
class TLE(epsilon: float = 1e-4, n_neighbors: int | None = None)  # default 20
class ESS(ver: str = "a", d: int = 1, random_state: int | None = None,
          n_neighbors: int | None = None)            # default 100
```

All local estimators share: `.fit(X)` → `self`, `.dimension_pw_` (Tensor), `.dimension_` (float).

## `torchid.datasets`

A focused subset of the scikit-dimension `BenchmarkManifolds`, torch-native.

```python
torchid.datasets.hyperball(n, d, *, radius=1.0, generator=None, device="cpu", dtype=torch.float32)
torchid.datasets.hypersphere(n, d, *, generator=None, device="cpu", dtype=torch.float32)
torchid.datasets.affine_subspace(n, d, ambient, *, noise_std=0.0,
                                 generator=None, device="cpu", dtype=torch.float32)
torchid.datasets.swiss_roll(n, *, generator=None, device="cpu", dtype=torch.float32)
```

`noise_std > 0` on `affine_subspace` adds isotropic Gaussian noise in the full ambient space — recommended for parity tests, since the degenerate zero-eigenvalue tail otherwise makes some ratio-based estimators (lPCA/maxgap) numerically unstable.

The full 23-manifold benchmark is available via `skdim.datasets.BenchmarkManifolds` when the `validation` dep group is installed.

## `torchid._primitives`

Shared batched building blocks used by every estimator. Private, but stable enough to call from user code:

```python
as_tensor(X, *, dtype=None, device=None) -> Tensor
pairwise_sqdist(X, Y=None, *, chunk=4096, clamp_min=0.0) -> Tensor
knn(X, k, *, chunk=4096, include_self=False, Y=None) -> (dists, idx)
gather_neighbors(X, idx) -> Tensor              # (N, k, D)
batched_local_pca(X_nbrs, *, center=True) -> (eigvals, eigvecs)
log_knn_ratios(dists, *, eps=1e-12) -> Tensor   # log(d_k / d_j), j=0..k-2
sample_combinations(k, p, m, *, device=None, generator=None) -> Tensor  # (m, p)
```

`knn` dispatches to `faiss.IndexFlatL2` when `X.device.type == "cpu"` and stays pure-torch on CUDA — see [Architecture](architecture.md).

## `torchid.parity`

Helpers for cross-checking torchid against scikit-dimension. Requires the `validation` dep group.

```python
from torchid.parity import Case, DEFAULT_CASES, compare_global, assert_parity

rows = compare_global(
    torch_cls=TwoNN,
    skdim_cls=skid.TwoNN,
    cases=DEFAULT_CASES,
    atol=1e-4, rtol=5e-3,
)
assert_parity(rows)
```
