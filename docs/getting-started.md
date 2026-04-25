# Getting Started

## Install

```bash
pip install torchid
```

Requires Python 3.13+ and PyTorch 2.x. To run on a CUDA host match your driver's CUDA version when installing torch (for a 12.x driver: `pip install torch --index-url https://download.pytorch.org/whl/cu128`). `faiss-cpu` and `numpy` come in automatically.

For running parity tests or benchmarks against scikit-dimension:

```bash
pip install "torchid[validation]"
# or with uv:
uv sync --group validation
```

## The estimator API

Every estimator follows the scikit-learn / scikit-dimension `fit` pattern. Global estimators return a single scalar; local estimators return both the per-point array and its mean.

```python
import torch
from torchid.estimators import lPCA, MLE

X = torch.randn(5000, 30, device="cuda")

# Global — single scalar
est = lPCA(ver="FO").fit(X)
print(est.dimension_)          # float
print(est.explained_var_)      # torch.Tensor

# Local — per-point tensor + aggregated mean
est = MLE().fit(X, comb="mle")
print(est.dimension_)          # float (harmonic mean over points)
print(est.dimension_pw_)       # torch.Tensor of shape (N,)
```

Inputs can be any 2-D tensor, NumPy array, or Python list. The output lives on the same device as the input — pass a CUDA tensor in, get GPU-resident intermediates, only the final scalar is moved to CPU.

## Streaming during training

Use `torchid.IntrinsicDimension` to log ID per epoch / per validation step
without writing a buffer manually. It's a `torchmetrics.Metric`:

```python
from torchid import IntrinsicDimension

metric = IntrinsicDimension(method="lpca", max_samples=10_000).to(device)
for batch in val_loader:
    feats = model.encode(batch)
    metric.update(feats)
print(metric.compute())             # 0-D tensor
```

`max_samples` reservoir-caps memory. The metric is DDP-aware (state reduces
via `cat`).

## Multi-dataset and per-point helpers

Two thin wrappers compose with any of the 12 estimators:

```python
from torchid import asPointwise, estimate_many
from torchid.estimators import TwoNN, lPCA

# one estimate per dataset (e.g. across model variants or layers)
dims = estimate_many([X_resnet, X_vit, X_dino], TwoNN)

# per-point local ID via any global estimator (anomaly maps, etc.)
ids = asPointwise(X, lPCA, n_neighbors=50)   # (N,) tensor
```

## Choosing an estimator

A rough decision tree:

| You want                                                | Use                         |
| ------------------------------------------------------- | --------------------------- |
| A global dimension via PCA eigenvalue thresholding      | `lPCA`                      |
| Fast, model-free global estimate from nearest neighbors | `TwoNN`, `MLE`, `CorrInt`   |
| A per-point local dimension map                         | `MOM`, `MADA`, `TLE`, `ESS` |
| Reference-calibrated global with anisotropy correction  | `DANCo`                     |
| Separability-based global robust to noise               | `FisherS`                   |
| Integer likelihood-based dimension                      | `MiND_ML`, `KNN`            |

All 12 are described in [Estimators](estimators.md) with references and parameter notes.

## Device placement

`torchid` is designed so you never need to think about device transfers:

```python
X = torch.randn(10_000, 100, device="cuda")
d = lPCA().fit(X).dimension_      # runs entirely on GPU

X_cpu = X.cpu()
d = lPCA().fit(X_cpu).dimension_  # runs on CPU (knn dispatches to faiss-cpu)
```

The one primitive that branches internally is `torchid.primitives.knn`: on CPU tensors it calls `faiss.IndexFlatL2` (O(n log n) in practice thanks to SIMD + OpenMP); on CUDA it stays pure-torch with a chunked top-k kernel.

## Reproducing the benchmarks

```bash
uv run --group validation python -m benchmarks.bench --small     # ~5 min, n ∈ {500, 2000}
uv run --group validation python -m benchmarks.bench             # ~25 min, n ∈ {2k, 10k, 20k}
```

Results write to `BENCHMARKS.md`.

## Running parity tests

```bash
uv run --group validation pytest tests/ -q
```

The full suite includes parity, primitives, wrappers, and metrics tests. CUDA
device-parity tests are auto-skipped on hosts without a GPU. See
[Parity](parity.md) for per-estimator tolerances and the caveats around
skdim's Python-3.13-incompatible `MLE.__init__`.
