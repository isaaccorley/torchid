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
``knn`` is exact brute force: a chunked distance matrix followed by ``torch.topk`` on
whichever device the tensors live on; see :doc:`/architecture`.
"""

import math

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


def _euclidean_distances(x: Tensor, y: Tensor) -> Tensor:
    """Subtract coordinates before squaring, including for large or batched inputs."""
    if x.dtype in (torch.float16, torch.bfloat16):
        x, y = x.float(), y.float()
    elif not x.is_floating_point():
        dtype = torch.float32 if x.device.type == "mps" else torch.float64
        x, y = x.to(dtype), y.to(dtype)
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")


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
    out = torch.empty((n, m), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = _euclidean_distances(X[start:end], Y).square()
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

    The search is exact and runs on the device of ``X``, in chunks of ``chunk`` query rows.
    """
    if X.ndim != 2:
        raise ValueError(f"expected 2-D X, got shape {tuple(X.shape)}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if Y is not None:
        if Y.ndim != 2:
            raise ValueError(f"expected 2-D Y, got shape {tuple(Y.shape)}")
        _validate_reference(X, Y)
    self_pair = Y is None
    ref = X if Y is None else Y
    n = X.shape[0]
    k_eff, drop_self = _k_eff(k, ref.shape[0], self_pair=self_pair, include_self=include_self)

    d_out = torch.empty((n, k), dtype=X.dtype, device=X.device)
    i_out = torch.empty((n, k), dtype=torch.long, device=X.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = _euclidean_distances(X[start:end], ref)
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
        d_out[start:end] = vals[:, :k]
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
