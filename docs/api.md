# API Reference

## At a glance

| import                                          | what it gives you                                             |
| ----------------------------------------------- | ------------------------------------------------------------- |
| `from torchid.estimators import lPCA, TwoNN, …` | the 12 estimator classes                                      |
| `from torchid import estimate_many`             | fit one estimator across many datasets                        |
| `from torchid import asPointwise`               | turn any global estimator into a per-point local              |
| `from torchid import IntrinsicDimension`        | torchmetrics-compatible streaming ID                          |
| `from torchid import IntrinsicDimensionLoss`    | differentiable ID as a minimizable loss                       |
| `from torchid import functional`                | differentiable functional estimates (`mle_id`, `twonn_id`, …) |
| `from torchid import datasets`                  | `hyperball`, `hypersphere`, `affine_subspace`, `swiss_roll`   |
| `from torchid import primitives`                | shared batched primitives (`knn`, `pairwise_sqdist`, …)       |

Every estimator follows the same pattern:

```python
est = Estimator(**params).fit(X)   # X: (N, D) torch.Tensor / numpy / list
est.dimension_                     # float
est.dimension_pw_                  # (N,) tensor — local estimators only
```

The output lives on the same device as the input.

## Global estimators

One scalar dimension per dataset. Constructor signatures below; defaults match
scikit-dimension's where possible.

### `lPCA`

PCA-eigenvalue thresholding via one of seven heuristics.

```python
lPCA(
    ver: str = "FO",
    alphaRatio: float = 0.05,
    alphaFO: float = 0.05,
    alphaFan: float = 10.0,
    betaFan: float = 0.8,
    PFan: float = 0.95,
    fit_explained_variance: bool = False,
)
```

- `ver` ∈ `{"FO", "Fan", "maxgap", "ratio", "participation_ratio", "Kaiser", "broken_stick"}`.
- After `fit`: `dimension_` (float), `explained_var_` (`Tensor`), `gap_` (`np.ndarray`).

### `TwoNN`

Facco et al. 2017. Linear regression on log-ratios of the two nearest neighbors.

```python
TwoNN(discard_fraction: float = 0.1, dist: bool = False)
```

- After `fit`: `dimension_`, `x_` = `log(mu)`, `y_` = `-log(1 - F_emp)`.

### `MLE`

Levina–Bickel maximum likelihood (zero-noise closed form).

```python
MLE(unbiased: bool = False)
# fit signature has extra kwargs:
MLE().fit(X, *, n_neighbors: int | None = None, comb: str = "mle")
```

- `comb` ∈ `{"mle", "mean", "median"}` — aggregation across pointwise estimates.
- The Haro-integral noise branch from skdim is **not** ported (parity tests
    cover the zero-noise branch only).

### `CorrInt`

Grassberger–Procaccia correlation integral.

```python
CorrInt(k1: int = 10, k2: int = 20, DM: bool = False)
```

`DM=True` (precomputed distance matrix) is not supported.

### `MiND_ML`

Maximum-likelihood integer / continuous dim from $\rho_i = d_1(i) / d_k(i)$.

```python
MiND_ML(k: int = 20, D: int = 10, ver: str = "MLk")
```

- `ver` ∈ `{"MLi", "MLk"}`. `MLk` refines `MLi` via a 1-D dense grid (in place
    of `scipy.optimize.L-BFGS-B`).

### `KNN`

Carter et al. 2010 — graph-length regression over bootstrap resamples.

```python
KNN(k: int | None = None, ps: np.ndarray | None = None, M: int = 1, gamma: int = 2)
```

Uses `np.random.randint`; seed via `np.random.seed(...)` for reproducibility.

### `DANCo`

Ceruti et al. 2012 — dimensionality from angle and norm concentration.

```python
DANCo(
    k: int = 10,
    D: int | None = None,
    calibration_data: dict | None = None,
    ver: str = "DANCo",
    fractal: bool = True,
    random_state: int | None = None,
)
```

- `ver` ∈ `{"DANCo", "MIND_MLi", "MIND_MLk"}`.
- Calibration data is generated on-device from torch hyperballs; pass back via
    `calibration_data` to reuse across calls.

### `FisherS`

Albergante et al. 2019 — Fisher separability.

```python
FisherS(
    conditional_number: float = 10.0,
    project_on_sphere: bool = True,
    alphas: np.ndarray | None = None,
    limit_maxdim: bool = False,
)
```

Lambert-W inversion is delegated to `scipy.special.lambertw` on a ~20-element
alpha grid (not a perf bottleneck).

## Local estimators

One dimension per point. Output has both `dimension_pw_` (`Tensor` of shape
`(N,)`) and `dimension_` (the aggregated scalar).

### `MOM`

Method of moments (Amsaleg et al. 2018).

```python
MOM(n_neighbors: int | None = None)   # default 100, matches skdim
```

### `MADA`

Manifold-adaptive (Farahmand & Szepesvári 2007). Ratio of `k`-NN to `k/2`-NN
distances.

```python
MADA(DM: bool = False, n_neighbors: int | None = None)   # default 20
```

### `TLE`

Tight Local Estimator (Amsaleg et al. 2019). Closed-form formula on a
per-point `(k, k)` neighbor-distance matrix.

```python
TLE(epsilon: float = 1e-4, n_neighbors: int | None = None)   # default 20
```

### `ESS`

Expected Simplex Skewness (Johnsson et al. 2015).

```python
ESS(
    ver: str = "a",
    d: int = 1,
    random_state: int | None = None,
    n_neighbors: int | None = None,    # default 100
)
```

For `d=1` (the dominant case) the per-neighborhood ESS has a closed form. For
`d > 1, ver="a"` we fall back to Monte-Carlo p-subset sampling. `ver="b"`
requires `d=1`.

## High-level wrappers

### `estimate_many(datasets, estimator, **kwargs) -> list[float]`

Fit one estimator independently across a list of datasets. Per-entry shapes
may differ. `**kwargs` are forwarded to every fit.

```python
from torchid import estimate_many
from torchid.estimators import TwoNN

dims = estimate_many([X_resnet, X_vit, X_dino], TwoNN)
# → [d_resnet, d_vit, d_dino]
```

### `asPointwise(X, estimator, n_neighbors=100, **kwargs) -> Tensor`

Mirrors `skdim.asPointwise`. For each point, fit `estimator` on its `k`-NN
patch; return an `(N,)` tensor of per-point dimensions. Complements the native
local estimators by working with *any* global estimator (including `FisherS`,
`CorrInt`, `DANCo`) at the cost of `N` fits — use a fast estimator for large
`N`.

```python
from torchid import asPointwise
from torchid.estimators import lPCA

ids = asPointwise(X, lPCA, n_neighbors=50)   # (N,) tensor
# threshold ids to find off-manifold / high-ID points
```

## `torchid.IntrinsicDimension`

torchmetrics-compatible streaming metric. Buffers feature batches across
`update()` and runs the chosen estimator on `compute()`. DDP-aware — state
reduces via `cat`.

```python
class IntrinsicDimension(
    method: str = "lpca",
    max_samples: int | None = 10_000,
    **estimator_kwargs,
)
```

- `method` is the lowercase estimator name: `"lpca"`, `"twonn"`, `"mle"`,
    `"corrint"`, `"mind_ml"`, `"mom"`, `"mada"`, `"knn"`, `"tle"`, `"danco"`,
    `"ess"`, `"fishers"`.
- `max_samples` caps retained state with random-priority reservoir sampling.
    `None` keeps everything.

```python
from torchid import IntrinsicDimension

metric = IntrinsicDimension(method="lpca").to(device)
for batch in val_loader:
    feats = model.encode(batch)
    metric.update(feats)
print(metric.compute())             # 0-D tensor
```

## `torchid.functional`

Differentiable functional forms of the closed-form estimators. They return
0-D tensors (the classes return floats) and keep the autograd graph alive, so
an ID estimate can be a training objective.

```python
mle_id(X, *, n_neighbors=20, unbiased=False, comb="mle")
twonn_id(X, *, discard_fraction=0.1)
mom_id(X, *, n_neighbors=100)
mada_id(X, *, n_neighbors=20)
pr_id(X)                                   # participation ratio (Σλ)²/Σλ²
intrinsic_dimension(X, method="twonn", **kwargs)   # name-based dispatch
```

```python
import torch
from torchid.functional import twonn_id

X = encoder(batch)          # (B, D), requires_grad
d = twonn_id(X)             # 0-D tensor, differentiable w.r.t. X
(-d).backward()             # push representations toward higher ID
```

How gradients work: neighbor *selection* is discrete, so indices come from a
no-grad kNN (FAISS on non-macOS CPU, chunked top-k on macOS and CUDA) and are
treated as constants; distances are then recomputed differentiably from the
gathered coordinates. Away from neighbor-order ties the estimate is piecewise
smooth in `X` and the gradient is exact.

Only estimators whose formula is smooth in the distances are exposed. The
counting/calibration estimators (`lPCA` thresholds, `CorrInt`, `DANCo`,
`FisherS`, `KNN`, `MiND_ML`) are integer- or argmin-valued with zero gradient
almost everywhere; `pr_id` — the participation ratio, identical to
`lPCA(ver="participation_ratio")` — is the smooth spectral alternative and
the best-behaved objective (no kNN graph at all).

Without gradients (`X.requires_grad == False`) the functionals take the same
kNN code path as the estimator classes and reproduce them exactly.

## `torchid.losses`

`IntrinsicDimensionLoss` wraps the functional estimates into a
`torch.nn.Module` that is always something to *minimize*. Raw ID is
normalized to the ratio `id / D` (`D` = embedding width):

```python
class IntrinsicDimensionLoss(
    method: str = "twonn",     # 'mada' | 'mle' | 'mom' | 'pr' | 'twonn'
    mode: str = "maximize",    # 'maximize' | 'minimize' | 'target'
    target: float | None = None,
    normalize: bool = True,
    **method_kwargs,           # forwarded to the estimate, e.g. n_neighbors=10
)
```

- `mode="maximize"` → loss = `1 - id/D` (drive ID toward the ambient width)
- `mode="minimize"` → loss = `id/D`
- `mode="target"` → loss = `((id - target)/D)²`
- `normalize=False` drops the `1/D` scaling (`-id` / `id` / `(id - target)²`).

```python
from torchid.losses import IntrinsicDimensionLoss

id_loss = IntrinsicDimensionLoss(method="mle", mode="maximize", n_neighbors=10)

feats = encoder(batch)                    # (B, D)
loss = task_loss + 0.1 * id_loss(feats)  # regularize toward higher ID
loss.backward()
id_loss.dimension_                        # detached raw estimate, for logging
```

The estimate is computed over the batch, so the batch is the sample: use a
reasonably large `B` (hundreds) for the kNN-based methods; `pr` is smooth and
usable at smaller `B`. Estimates can slightly exceed `D` (MLE in particular),
so the `maximize` loss can go a little negative — harmless for optimization.

## `torchid.datasets`

A focused subset of skdim's `BenchmarkManifolds`, torch-native. The full 23
manifolds are reachable via `skdim.datasets.BenchmarkManifolds` from the
`validation` dep group.

```python
hyperball(n, d, *, radius=1.0, generator=None, device="cpu", dtype=torch.float32)
hypersphere(n, d, *, generator=None, device="cpu", dtype=torch.float32)
affine_subspace(n, d, ambient, *, noise_std=0.0,
                generator=None, device="cpu", dtype=torch.float32)
swiss_roll(n, *, generator=None, device="cpu", dtype=torch.float32)
```

`affine_subspace(..., noise_std=0)` produces exact zero tail-eigenvalues; pass
a small positive `noise_std` (e.g. `0.01`) for parity testing — the
ratio-based heuristics in `lPCA` go numerically unstable otherwise.

## `torchid.primitives`

Shared batched building blocks — public API. Use these directly when you
want a fast batched kNN, neighborhood gather, or per-neighborhood SVD
without going through an estimator:

| function                                               | shape signature                  |
| ------------------------------------------------------ | -------------------------------- |
| `as_tensor(X, *, dtype=None, device=None)`             | array-like → `(N, D)` Tensor     |
| `pairwise_sqdist(X, Y=None, *, chunk=4096)`            | `(N, M)` Tensor                  |
| `knn(X, k, *, chunk=4096, include_self=False, Y=None)` | `(dists, idx)` of shape `(N, k)` |
| `gather_neighbors(X, idx)`                             | `(N, k, D)` Tensor               |
| `batched_local_pca(X_nbrs, *, center=True)`            | `(eigvals, eigvecs)`             |
| `log_knn_ratios(dists, *, eps=1e-12)`                  | `dists.shape[:-1] + (k-1,)`      |
| `sample_combinations(k, p, m, *, generator=None)`      | `(m, p)` index tensor            |

`knn` dispatches to `faiss.IndexFlatL2` on non-macOS CPU and to the chunked
torch top-k path on macOS or CUDA. See [Architecture](architecture.md).

## Parity harness (dev-only)

The parity harness is intentionally **not** part of the public `torchid`
namespace — it lives at `tests/_parity.py` because it imports `skdim` and
exists only to assert numerical parity in the test suite. To use it from
your own tests, install the `validation` dep group and import via the test
directory:

```python
from ._parity import Case, DEFAULT_CASES, compare_global, assert_parity
import skdim.id as skid
from torchid.estimators import TwoNN

rows = compare_global(
    torch_cls=TwoNN,
    skdim_cls=skid.TwoNN,
    cases=DEFAULT_CASES,
    atol=1e-4,
    rtol=5e-3,
)
assert_parity(rows)
```

`compare_global` returns one row per case with `torch_dim`, `skdim_dim`,
`abs_err`, `rel_err`, `pass`. `assert_parity` raises a readable diff when
fewer than `min_fraction` of rows pass. Default tolerances per estimator are
documented in [Parity](parity.md).
