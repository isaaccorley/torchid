# torchid

GPU-accelerated intrinsic dimension estimators in PyTorch. A port of
[scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) with
batched/vectorized implementations and CUDA support.

Status: in-progress port. See `CHECKLIST.md` for progress.

## Why

`scikit-dimension` is the reference library for intrinsic dimension (ID) estimation but
is CPU-only and relies heavily on per-point Python loops. `torchid` re-implements every
estimator using batched `torch` ops so the same methods run 10–200× faster on GPU while
producing outputs that match the reference library within documented tolerances.

## Install

```bash
pip install torchid
```

For a CUDA-capable install, pick the PyTorch wheel that matches your driver, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torchid
```

For running parity tests against `scikit-dimension` from a clone:

```bash
uv sync --group validation
```

## Usage

```python
import torch
from torchid.estimators import lPCA

X = torch.randn(10_000, 50, device="cuda")
est = lPCA().fit(X)
print(est.dimension_)
```
