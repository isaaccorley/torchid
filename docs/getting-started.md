# Getting Started

## Install

```bash
pip install torchid
```

Supports Python 3.11 through 3.15, including free-threaded CPython 3.14t. For CUDA, choose the wheel for your platform and driver using the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

For running parity tests or benchmarks against scikit-dimension:

```bash
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

Inputs can be 2-D tensors, NumPy arrays, or Python lists. Per-point estimates stay on the input device. The aggregate `dimension_` is a Python float.

## Streaming during training

Install `torchid[metrics]` to use `torchid.IntrinsicDimension` to log ID per epoch / per validation step
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

## Using ID as a training objective

The estimator classes are fit-only. When you want gradients — e.g. to
regularize representations toward higher (or lower, or a target) intrinsic
dimension — use the differentiable functional forms in `torchid.functional`
or the ready-made loss:

```python
from torchid import IntrinsicDimensionLoss

id_loss = IntrinsicDimensionLoss(method="twonn", mode="maximize")

feats = encoder(batch)                    # (B, D), requires_grad
loss = task_loss + 0.1 * id_loss(feats)  # minimizing 1 - id/D maximizes ID
loss.backward()
```

`mode` is one of `maximize` (`1 - id/D`), `minimize` (`id/D`), or `target`
(`((id - target)/D)²`). Five methods are differentiable: `mle`, `twonn`,
`mom`, `mada`, and `pr` (participation ratio — smooth, no kNN graph). See the
[API reference](api.md#torchidlosses) for the gradient semantics and caveats.

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

On Apple silicon, use a float32 MPS tensor. TorchID tests MPS with PyTorch 2.15, whose Metal backend requires macOS 15 or later. The same estimator accepts CPU and CUDA tensors.

```python
if torch.backends.mps.is_available():
    X = torch.randn(10_000, 100, dtype=torch.float32, device="mps")
    est = MLE().fit(X)
    print(est.dimension_pw_.device)  # mps:0
```

The `KNN` graph-length estimator computes distances and fits its regression in float64 on every backend. For MPS input, it emits a warning and copies the coordinates to CPU for the fit. To retain float64 source coordinates, pass them directly on CPU or CUDA.

`torchid.primitives.knn` is exact brute force in plain torch: a chunked distance matrix followed by `torch.topk`, on the device of the input.

## Reproducing the benchmarks

```bash
uv run --group validation python -m benchmarks.bench --small     # ~5 min, n ∈ {500, 2000}
uv run --group validation python -m benchmarks.bench             # ~25 min, n ∈ {2k, 10k, 20k}
```

Results write to `BENCHMARKS.md`.

## Running parity tests

```bash
uv run --extra metrics --group validation pytest tests/ -q
```

The full suite includes parity, primitives, wrappers, and metrics tests. CUDA and MPS tests run when the corresponding device is available. On an Apple silicon Mac, run `uv run pytest -m mps` for the device checks. See
[Parity](parity.md) for per-estimator tolerances and the caveats around
skdim's Python-3.13-incompatible `MLE.__init__`.
