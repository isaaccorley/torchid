"""Batched torch primitives shared by every estimator.

Public API. Use these directly when you want a batched kNN, a streamed
pairwise distance, neighborhood gather, or per-neighborhood SVD without
going through an estimator:

::

    import torch
    from torchid.primitives import knn, batched_local_pca, gather_neighbors

    X = torch.randn(10_000, 50, device="cuda")
    dists, idx = knn(X, k=20)             # (10_000, 20) each
    nbrs = gather_neighbors(X, idx)        # (10_000, 20, 50)
    eigvals, eigvecs = batched_local_pca(nbrs)

Distances, neighbors, and local PCA are computed against ``(N, ...)`` tensors
without per-point Python loops.
``knn`` uses ``faiss.IndexFlatL2`` on non-macOS CPU and chunked torch top-k on
macOS or CUDA; see :doc:`/architecture` for why.
"""

import math
import sys
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "as_tensor",
    "batched_local_pca",
    "gather_neighbors",
    "knn",
    "log_knn_ratios",
    "pairwise_sqdist",
    "sample_combinations",
]


def as_tensor(
    X: object,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Coerce array-like input to a 2-D float tensor on the requested device."""
    t = X if isinstance(X, Tensor) else torch.as_tensor(X)
    if dtype is not None:
        t = t.to(dtype)
    elif t.dtype not in (torch.float32, torch.float64):
        t = t.to(torch.float32)
    if device is not None:
        t = t.to(device)
    if t.ndim != 2:
        raise ValueError(f"expected 2-D input, got shape {tuple(t.shape)}")
    return t


def pairwise_sqdist(
    X: Tensor, Y: Tensor | None = None, *, chunk: int = 4096, clamp_min: float = 0.0
) -> Tensor:
    """Squared Euclidean distances between rows of ``X`` and ``Y``.

    Streams in row chunks so peak memory is bounded at ``chunk * Y.shape[0]``
    regardless of ``X.shape[0]``. When ``Y`` is ``None`` the matrix is
    symmetric and the diagonal is forced to zero.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if X.ndim != 2:
        raise ValueError(f"expected 2-D X, got shape {tuple(X.shape)}")
    if Y is not None and Y.ndim != 2:
        raise ValueError(f"expected 2-D Y, got shape {tuple(Y.shape)}")
    self_pair = Y is None
    if self_pair:
        Y = X
    assert Y is not None
    _validate_reference(X, Y)
    n, m = X.shape[0], Y.shape[0]
    y_sq = (Y * Y).sum(dim=1)
    out = torch.empty((n, m), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        out[start:end] = x_sq + y_sq.unsqueeze(0) - 2.0 * (xb @ Y.T)
    out.clamp_(min=clamp_min)
    if self_pair:
        diag = torch.arange(min(n, m), device=X.device)
        out[diag, diag] = 0.0
    return out


def knn(
    X: Tensor,
    k: int,
    *,
    chunk: int = 4096,
    include_self: bool = False,
    Y: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(distances, indices)`` of the ``k`` nearest neighbors per row of ``X``.

    Distances are Euclidean (not squared). When ``Y`` is ``None`` and
    ``include_self`` is False, self-matches are excluded.

    Dispatches on device: CPU tensors use ``faiss.IndexFlatL2`` except on
    macOS, where PyTorch and FAISS load conflicting OpenMP runtimes. CUDA and
    macOS tensors use the pure-torch chunked path.
    """
    if X.ndim != 2:
        raise ValueError(f"expected 2-D X, got shape {tuple(X.shape)}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if Y is not None:
        if Y.ndim != 2:
            raise ValueError(f"expected 2-D Y, got shape {tuple(Y.shape)}")
        _validate_reference(X, Y)
    _k_eff(
        k,
        X.shape[0] if Y is None else Y.shape[0],
        self_pair=Y is None,
        include_self=include_self,
    )
    if X.device.type == "cpu" and sys.platform != "darwin":
        return _knn_faiss(X, k, Y=Y, include_self=include_self)
    return _knn_torch(X, k, Y=Y, include_self=include_self, chunk=chunk)


def _knn_faiss(X: Tensor, k: int, *, Y: Tensor | None, include_self: bool) -> tuple[Tensor, Tensor]:
    try:
        import faiss  # optional runtime dep, only needed on the non-macOS CPU path
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for non-macOS CPU-path kNN. Install it with:\n"
            "  pip install torchid[cpu]   # faiss-cpu\n"
            "  pip install torchid[cuda]  # faiss-cuda-cu128 (GPU-enabled, also works on CPU)"
        ) from exc

    self_pair = Y is None
    ref = X if self_pair else Y
    assert ref is not None
    k_eff, drop_self = _k_eff(k, ref.shape[0], self_pair=self_pair, include_self=include_self)
    ref_np = ref.detach().contiguous().to(torch.float32).numpy()
    q_np = ref_np if self_pair else X.detach().contiguous().to(torch.float32).numpy()
    index = cast("Any", faiss.IndexFlatL2(ref_np.shape[1]))
    index.add(ref_np)
    d2, idx = index.search(q_np, k_eff)
    d = np.sqrt(np.maximum(d2, 0.0))
    if drop_self:
        d, idx = _drop_self_numpy(d, idx)
    return (
        torch.from_numpy(d[:, :k]).to(X.dtype),
        torch.from_numpy(idx[:, :k]).to(torch.long),
    )


def _knn_torch(
    X: Tensor, k: int, *, Y: Tensor | None, include_self: bool, chunk: int
) -> tuple[Tensor, Tensor]:
    self_pair = Y is None
    ref = X if self_pair else Y
    assert ref is not None
    n = X.shape[0]
    k_eff, drop_self = _k_eff(k, ref.shape[0], self_pair=self_pair, include_self=include_self)

    d_out = torch.empty((n, k), dtype=X.dtype, device=X.device)
    i_out = torch.empty((n, k), dtype=torch.long, device=X.device)
    y_sq = (ref * ref).sum(dim=1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        d = (x_sq + y_sq.unsqueeze(0) - 2.0 * (xb @ ref.T)).clamp_(min=0.0)
        vals, idx = torch.topk(d, k_eff, dim=1, largest=False, sorted=True)
        if drop_self:
            row_ids = torch.arange(start, end, device=X.device).unsqueeze(1)
            is_self = idx == row_ids
            drop = torch.where(
                is_self.any(dim=1, keepdim=True),
                is_self.int().argmax(dim=1, keepdim=True),
                torch.full_like(row_ids, k_eff - 1),
            )
            keep = torch.ones_like(idx, dtype=torch.bool).scatter_(1, drop, False)
            idx = idx[keep].view(end - start, k_eff - 1)
            vals = vals[keep].view(end - start, k_eff - 1)
        d_out[start:end] = vals[:, :k].sqrt()
        i_out[start:end] = idx[:, :k]
    return d_out, i_out


def _k_eff(k: int, m: int, *, self_pair: bool, include_self: bool) -> tuple[int, bool]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    drop_self = self_pair and not include_self
    k_eff = k + (1 if drop_self else 0)
    if k_eff > m:
        raise ValueError(f"requested k={k} but only {m} reference points available")
    return k_eff, drop_self


def _effective_n_neighbors(n_neighbors: int, n_samples: int, *, minimum: int = 1) -> int:
    """Validate and cap a requested neighbor count against the sample count."""
    if n_neighbors < minimum:
        raise ValueError(f"n_neighbors must be >= {minimum}, got {n_neighbors}")
    available = n_samples - 1
    if available < minimum:
        raise ValueError(
            f"at least {minimum + 1} samples are required for {minimum} neighbors, got {n_samples}"
        )
    return min(n_neighbors, available)


def _validate_reference(X: Tensor, Y: Tensor) -> None:
    """Require compatible tensors for cross-reference distance calculations."""
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension, got {X.shape[1]} and {Y.shape[1]}"
        )
    if X.device != Y.device:
        raise ValueError(f"X and Y must be on the same device, got {X.device} and {Y.device}")
    if X.dtype != Y.dtype:
        raise ValueError(f"X and Y must have the same dtype, got {X.dtype} and {Y.dtype}")


def _drop_self_numpy(d: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """faiss usually returns self at column 0, but not under ties — detect and drop."""
    n, k_eff = idx.shape
    row_ids = np.arange(n)[:, None]
    is_self = idx == row_ids
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k_eff - 1)
    mask = np.ones_like(idx, dtype=bool)
    mask[np.arange(n), drop] = False
    return d[mask].reshape(n, k_eff - 1), idx[mask].reshape(n, k_eff - 1)


def gather_neighbors(X: Tensor, idx: Tensor) -> Tensor:
    """Gather neighbor coordinates given an ``(N, k)`` index tensor → ``(N, k, D)``."""
    n, k = idx.shape
    return X[idx.reshape(-1)].reshape(n, k, X.shape[1])


def batched_local_pca(X_nbrs: Tensor, *, center: bool = True) -> tuple[Tensor, Tensor]:
    """Batched local PCA over neighborhoods.

    Args:
        X_nbrs: ``(N, k, D)`` neighborhoods.
        center: subtract the per-neighborhood mean before SVD.

    Returns:
        ``(eigvals, eigvecs)`` where ``eigvals`` is ``(N, min(k, D))`` of
        covariance eigenvalues (descending) and ``eigvecs`` is
        ``(N, D, min(k, D))`` (columns = principal axes).
    """
    _, k, _ = X_nbrs.shape
    Xc = X_nbrs - X_nbrs.mean(dim=1, keepdim=True) if center else X_nbrs
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    eigvals = (S * S) / max(k - 1, 1)
    return eigvals, Vh.transpose(-1, -2)


def log_knn_ratios(dists: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Row-wise ``log(d_k / d_j)`` for ``j < k``, stable under zero ties.

    ``dists`` is assumed sorted ascending on the last axis (as returned by
    :func:`knn`). Output shape is ``dists.shape`` minus one on the last axis.
    """
    if dists.ndim < 2:
        raise ValueError("expected at least 2-D input")
    return torch.log(dists[..., -1:].clamp_min(eps)) - torch.log(dists[..., :-1].clamp_min(eps))


def sample_combinations(
    k: int,
    p: int,
    m: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``m`` uniform p-subsets of ``{0, …, k-1}`` without replacement.

    Falls back to exact enumeration via :func:`torch.combinations` when the
    requested ``m`` exceeds the total count ``C(k, p)``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if p < 1 or p > k:
        raise ValueError(f"p must satisfy 1 <= p <= k, got p={p}, k={k}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m >= math.comb(k, p):
        return torch.combinations(torch.arange(k, device=device), r=p)
    keys = torch.rand((m, k), device=device, generator=generator)
    return keys.argsort(dim=1)[:, :p]
