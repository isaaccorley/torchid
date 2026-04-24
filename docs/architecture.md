# Architecture

The library is deliberately thin — shared primitives plus one file per estimator, each <500 LOC.

```
src/torchid/
├── _primitives.py       # pdist, knn (faiss on CPU, torch on CUDA), local-PCA, log-ratios
├── datasets.py          # hyperball, hypersphere, affine_subspace, swiss_roll
├── parity.py            # sklearn-dim cross-check harness (validation group only)
└── estimators/
    ├── base.py          # GlobalEstimator, LocalEstimator
    ├── lpca.py          # + 11 more estimator files
    └── ...
```

## Design principles

### 1. One tensor op beats one Python loop per point

scikit-dimension estimators are mostly `for i in range(N): per_point_fn(X[neighbors[i]])`. Each per-point call is a small numpy computation that would finish in microseconds, but the Python-loop overhead dominates when N is large.

torchid replaces every such loop with a batched tensor operation over `(N, k, …)`. Concretely:

- **DANCo** builds one `(N, k, k)` angle tensor instead of looping `_loc_angles` per point.
- **TLE** builds one `(N, k, k)` pairwise-distance tensor instead of `squareform(pdist(X[nbrs]))` per point.
- **ESS (d=1)** computes the per-neighborhood skewness as a single matmul + `triu_indices` slice.
- **lPCA** calls `torch.linalg.svdvals` on a `(N, k, D)` batched neighborhood tensor for the local variant.

### 2. Device-in, device-out

Every estimator is written against `torch.Tensor`, so the same code runs on CPU or CUDA. Intermediate values live on the device; only the final scalar is forced to the host via `float(...)`.

### 3. CPU path uses faiss, GPU path uses torch

The one primitive that differs between devices is `knn`. Pure-torch is O(n²) — fine when the dataset fits in GPU memory as one blob, but loses to scikit-dimension's sklearn `NearestNeighbors` (BallTree) at n ≥ 10k on CPU.

So `torchid._primitives.knn` dispatches on `X.device.type`:

- **CPU** → `faiss.IndexFlatL2` (SIMD + OpenMP, O(n log n) in practice).
- **CUDA** → torch chunked top-k over a streamed distance matrix.

This is the only dispatch in the codebase. No faiss-gpu, no conditional imports elsewhere.

### 4. Parity over novelty

Every numerical choice matches the reference. Where we diverge (fine-grid instead of `scipy.optimize.L-BFGS-B` inside `MiND_ML`; closed-form instead of `scipy.integrate.quad` inside `MLE`; Monte-Carlo p-subsets instead of full enumeration for `ESS` at d>1) the test suite has a tolerance band and a comment explaining why.

## Base classes

`torchid.estimators.base` provides two thin superclasses:

```python
class GlobalEstimator:
    def fit(self, X, y=None) -> Self: ...
    dimension_: float

class LocalEstimator:
    def fit(self, X, y=None) -> Self: ...
    dimension_pw_: Tensor            # (N,)
    dimension_: float                # .mean() of dimension_pw_
```

Neither is a `sklearn.base.BaseEstimator` — torchid avoids importing sklearn at runtime. For sklearn-style pipelines, wrap with a `FunctionTransformer` or write a one-liner adapter.

## Known sharp edges

- **skdim's `MLE.__init__`** mutates `inspect.getargvalues().f_locals`, which is read-only on Python 3.13. Parity tests and benchmarks patch this by instantiating skdim via `__new__` + manual attribute assignment. torchid's own `MLE` has no such issue.
- **KNN's bootstrap** uses `np.random.randint` to share its RNG stream with skdim's parity test — the one place we can't use torch's generator without breaking reproducibility against the reference.
- **`affine_subspace(..., noise_std=0)`** produces exact zero tail-eigenvalues, which makes ratio-based PCA heuristics (like `lPCA(ver="maxgap")`) numerically unstable. The parity harness defaults to `noise_std=0.01`.

## Future work

- Multi-dataset batched fitting (`estimate_many(X_list)`) — the primitives already support broadcasting over a leading dimension; only the estimator-level wrappers need updating.
- A `TorchMetric`-style streaming interface for evaluating ID across training epochs without keeping all activations resident.
- `torch.compile` integration for the closed-form estimators (MLE, TwoNN, MOM, MADA) — the control flow is already static.
