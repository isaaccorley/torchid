"""Differentiable functional forms of the closed-form estimators.

Each returns a 0-D tensor that carries gradients w.r.t. ``X``: kNN indices are
found under ``no_grad`` and treated as constants, distances are recomputed from
the gathered coordinates. The counting/argmin estimators (``lPCA`` thresholds,
``CorrInt``, ``DANCo``, ``FisherS``, ``KNN``, ``MiND_ML``) have zero gradient
almost everywhere and are not exposed. See :mod:`torchid.losses`.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

from torchid.primitives import as_tensor, gather_neighbors, knn, log_knn_ratios

__all__ = [
    "intrinsic_dimension",
    "mada_id",
    "mle_id",
    "mom_id",
    "pr_id",
    "twonn_id",
]


def _knn_dists(X: Tensor, k: int) -> Tensor:
    """kNN distances that carry gradients when ``X`` does; grad-free inputs use plain knn."""
    if not X.requires_grad:
        return knn(X, k=k)[0]
    with torch.no_grad():
        _, idx = knn(X, k=k)
    nbrs = gather_neighbors(X, idx)  # (N, k, D)
    d2 = (X.unsqueeze(1) - nbrs).pow(2).sum(dim=2)
    # clamp before sqrt: duplicate points give d2=0 whose sqrt-grad is inf
    return d2.clamp_min(torch.finfo(X.dtype).eps ** 2).sqrt()


def mle_id(
    X: object,
    *,
    n_neighbors: int = 20,
    unbiased: bool = False,
    comb: str = "mle",
) -> Tensor:
    """Differentiable Levina–Bickel MLE (:class:`torchid.estimators.MLE`)."""
    Xt = as_tensor(X)
    k = min(n_neighbors, Xt.shape[0] - 1)
    dists = _knn_dists(Xt, k)
    logs = log_knn_ratios(dists)
    kfac = k - 2 if unbiased else k - 1
    d_pw = kfac / logs.sum(dim=1).clamp_min(torch.finfo(Xt.dtype).tiny)
    if comb == "mle":
        return 1.0 / (1.0 / d_pw).mean()
    if comb == "mean":
        return d_pw.mean()
    if comb == "median":
        return d_pw.median()
    raise ValueError(f"comb must be one of 'mle','mean','median', got {comb!r}")


def twonn_id(X: object, *, discard_fraction: float = 0.1) -> Tensor:
    """Differentiable TwoNN (:class:`torchid.estimators.TwoNN`)."""
    Xt = as_tensor(X)
    d = _knn_dists(Xt, 2)
    mu = d[:, 1] / d[:, 0].clamp_min(torch.finfo(Xt.dtype).tiny)
    N = mu.shape[0]
    keep = int(N * (1 - discard_fraction))
    mu_sorted, _ = torch.sort(mu)
    femp = torch.arange(keep, device=Xt.device, dtype=Xt.dtype) / N
    x = torch.log(mu_sorted[:keep])
    y = -torch.log1p(-femp)
    return (x * y).sum() / (x * x).sum().clamp_min(torch.finfo(Xt.dtype).tiny)


def mom_id(X: object, *, n_neighbors: int = 100) -> Tensor:
    """Differentiable method of moments (:class:`torchid.estimators.MOM`); the
    ``m1 - w`` denominator is clamped away from zero."""
    Xt = as_tensor(X)
    k = min(n_neighbors, Xt.shape[0] - 1)
    dists = _knn_dists(Xt, k)
    w = dists[:, -1]
    m1 = dists.mean(dim=1)
    denom = (m1 - w).clamp_max(-torch.finfo(Xt.dtype).tiny)
    return (-m1 / denom).mean()


def mada_id(X: object, *, n_neighbors: int = 20) -> Tensor:
    """Differentiable MADA (:class:`torchid.estimators.MADA`); the log-ratio
    denominator is clamped away from zero."""
    Xt = as_tensor(X)
    k = min(n_neighbors, Xt.shape[0] - 1)
    if k < 2:
        raise ValueError(f"MADA needs at least 2 neighbors, got k={k}")
    dists = _knn_dists(Xt, k)
    RK = dists[:, k - 1]
    RK2 = dists[:, k // 2 - 1]
    log_ratio = torch.log(RK / RK2.clamp_min(torch.finfo(Xt.dtype).tiny))
    return (math.log(2.0) / log_ratio.clamp_min(torch.finfo(Xt.dtype).tiny)).mean()


def pr_id(X: object) -> Tensor:
    """Participation ratio ``(Σλ)² / Σλ²`` of the covariance spectrum — same as
    ``lPCA(ver='participation_ratio')``; fully smooth, no kNN graph."""
    Xt = as_tensor(X)
    Xc = Xt - Xt.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    ev = s * s  # the 1/(n-1) covariance normalizer cancels in the ratio
    return ev.sum() ** 2 / (ev * ev).sum().clamp_min(torch.finfo(Xt.dtype).tiny)


_REGISTRY: dict[str, Callable[..., Tensor]] = {
    "mada": mada_id,
    "mle": mle_id,
    "mom": mom_id,
    "pr": pr_id,
    "twonn": twonn_id,
}


def intrinsic_dimension(X: object, method: str = "twonn", **kwargs: object) -> Tensor:
    """Dispatch by name — ``'mada'``, ``'mle'``, ``'mom'``, ``'pr'``, ``'twonn'``."""
    key = method.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown method {method!r}. choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key](X, **kwargs)
