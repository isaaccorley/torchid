"""Batched torch primitives shared by all estimators.

Every estimator in :mod:`torchid.estimators` is built on top of these helpers.
They are written to avoid per-point Python loops: distances, neighbors, and
local-PCA are computed in batched form against `(N, ...)` tensors.
"""

import math

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
    t = torch.as_tensor(X, dtype=dtype) if not isinstance(X, Tensor) else X
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if t.dtype not in (torch.float32, torch.float64):
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

    Computed in row chunks so peak memory stays bounded at ``chunk * M * 4B``
    regardless of ``X``'s row count. When ``Y`` is ``None``, the matrix is
    symmetric and the diagonal is clamped to 0.
    """
    if Y is None:
        Y = X
    n, m = X.shape[0], Y.shape[0]
    y_sq = (Y * Y).sum(dim=1)
    out = torch.empty((n, m), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        d = x_sq + y_sq.unsqueeze(0) - 2.0 * (xb @ Y.T)
        out[start:end] = d
    out.clamp_(min=clamp_min)
    if Y is X:
        # numerical slop on the diagonal
        idx = torch.arange(min(n, m), device=X.device)
        out[idx, idx] = 0.0
    return out


def knn(
    X: Tensor,
    k: int,
    *,
    chunk: int = 4096,
    include_self: bool = False,
    Y: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(distances, indices)`` of the ``k`` nearest neighbors for each row of ``X``.

    Distances are Euclidean (not squared). When ``Y`` is ``None`` and
    ``include_self`` is False, self-matches are excluded.

    On CPU this dispatches to ``faiss.IndexFlatL2`` (O(n log n)-ish thanks to SIMD
    batching + parallelism) which is much faster than the torch O(n²) path at
    n ≥ ~5k. On CUDA we stay on the torch path — faiss-gpu is not used.
    """
    self_search = Y is None
    ref = X if self_search else Y
    if X.device.type == "cpu":
        return _knn_faiss(X, k, Y=Y if not self_search else None, include_self=include_self)
    return _knn_torch(
        X, k, chunk=chunk, include_self=include_self, Y=Y if not self_search else None,
    )


def _knn_faiss(
    X: Tensor, k: int, *, Y: Tensor | None, include_self: bool,
) -> tuple[Tensor, Tensor]:
    import faiss  # local import; optional CPU-only dep

    self_search = Y is None
    ref = X if self_search else Y
    m = ref.shape[0]
    k_eff = k + (1 if (self_search and not include_self) else 0)
    if k_eff > m:
        raise ValueError(f"requested k={k} but only {m} reference points available")

    ref_np = ref.detach().contiguous().to(torch.float32).numpy()
    q_np = X.detach().contiguous().to(torch.float32).numpy() if not self_search else ref_np
    index = faiss.IndexFlatL2(ref_np.shape[1])
    index.add(ref_np)
    d2, idx = index.search(q_np, k_eff)  # d2: squared L2
    d = np.sqrt(np.maximum(d2, 0.0))

    if self_search and not include_self:
        # faiss returns the query itself at column 0 in almost all cases; under
        # ties it may appear elsewhere, so we detect and drop explicitly.
        row_ids = np.arange(X.shape[0])[:, None]
        is_self = idx == row_ids
        has_self = is_self.any(axis=1)
        # for rows where self is present, drop that column; otherwise drop the last
        drop = np.where(has_self, is_self.argmax(axis=1), k_eff - 1)
        mask = np.ones_like(idx, dtype=bool)
        mask[np.arange(idx.shape[0]), drop] = False
        idx = idx[mask].reshape(X.shape[0], k_eff - 1)
        d = d[mask].reshape(X.shape[0], k_eff - 1)

    d = d[:, :k]
    idx = idx[:, :k]
    return (
        torch.from_numpy(d).to(X.dtype),
        torch.from_numpy(idx).to(torch.long),
    )


def _knn_torch(
    X: Tensor,
    k: int,
    *,
    chunk: int,
    include_self: bool,
    Y: Tensor | None,
) -> tuple[Tensor, Tensor]:
    self_search = Y is None
    ref = X if self_search else Y
    n = X.shape[0]
    m = ref.shape[0]
    k_eff = k + (1 if (self_search and not include_self) else 0)
    if k_eff > m:
        raise ValueError(f"requested k={k} but only {m} reference points available")

    d_out = torch.empty((n, k), dtype=X.dtype, device=X.device)
    i_out = torch.empty((n, k), dtype=torch.long, device=X.device)
    y_sq = (ref * ref).sum(dim=1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        d = x_sq + y_sq.unsqueeze(0) - 2.0 * (xb @ ref.T)
        d.clamp_(min=0.0)
        vals, idx = torch.topk(d, k_eff, dim=1, largest=False, sorted=True)
        if self_search and not include_self:
            row_ids = torch.arange(start, end, device=X.device).unsqueeze(1)
            is_self = idx == row_ids
            any_self = is_self.any(dim=1, keepdim=True)
            drop_col = torch.where(any_self, is_self.float().argmax(dim=1, keepdim=True), k_eff - 1)
            keep = torch.ones_like(idx, dtype=torch.bool)
            keep.scatter_(1, drop_col, False)
            idx = idx[keep].view(end - start, k_eff - 1)
            vals = vals[keep].view(end - start, k_eff - 1)
        d_out[start:end] = vals[:, :k].sqrt()
        i_out[start:end] = idx[:, :k]
    return d_out, i_out


def gather_neighbors(X: Tensor, idx: Tensor) -> Tensor:
    """Gather neighbor coordinates given a ``(N, k)`` index tensor → ``(N, k, D)``."""
    n, k = idx.shape
    d = X.shape[1]
    return X[idx.reshape(-1)].reshape(n, k, d)


def batched_local_pca(
    X_nbrs: Tensor, *, center: bool = True
) -> tuple[Tensor, Tensor]:
    """Batched local PCA over neighborhoods.

    Args:
        X_nbrs: ``(N, k, D)`` neighborhoods.
        center: subtract the per-neighborhood mean before SVD.

    Returns:
        ``(eigvals, eigvecs)`` where ``eigvals`` is ``(N, min(k, D))`` of
        covariance eigenvalues (largest first) and ``eigvecs`` is
        ``(N, D, min(k, D))`` (columns are principal axes).
    """
    n, k, d = X_nbrs.shape
    Xc = X_nbrs - X_nbrs.mean(dim=1, keepdim=True) if center else X_nbrs
    # SVD: Xc = U S V^T; covariance eigvals = S^2 / (k-1)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    denom = max(k - 1, 1)
    eigvals = (S * S) / denom
    eigvecs = Vh.transpose(-1, -2)
    return eigvals, eigvecs


def log_knn_ratios(dists: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Row-wise log(d_k / d_j) for j < k, stable under zero-distance ties.

    ``dists`` is assumed sorted ascending on the last axis (as returned by :func:`knn`).
    Output shape matches ``dists`` minus one along the last axis (the reference column).
    """
    if dists.ndim < 2:
        raise ValueError("expected at least 2-D input")
    d_ref = dists[..., -1:].clamp_min(eps)
    d_j = dists[..., :-1].clamp_min(eps)
    return torch.log(d_ref) - torch.log(d_j)


def sample_combinations(
    k: int, p: int, m: int, *, device: torch.device | str | None = None, generator: torch.Generator | None = None
) -> Tensor:
    """Sample ``m`` uniform p-subsets of ``{0, ..., k-1}`` without replacement.

    Used by ESS in place of enumerating all C(k, p) combinations when that is
    prohibitive. For small ``k``/``p`` the caller may prefer an exact enumeration.
    """
    if p > k:
        raise ValueError(f"p={p} > k={k}")
    total = math.comb(k, p)
    if m >= total:
        # fall back to exact enumeration via a greedy index trick
        idx = torch.combinations(torch.arange(k, device=device), r=p)
        return idx
    # rejection-free: argsort random keys, take first p columns
    keys = torch.rand((m, k), device=device, generator=generator)
    return keys.argsort(dim=1)[:, :p]
