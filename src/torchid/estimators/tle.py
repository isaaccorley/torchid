"""TLE (Amsaleg et al. 2019) — Tight Local intrinsic dimensionality Estimator.

Fully batched per-point computation. For every point, skdim builds a ``(k, k)``
matrix of pairwise neighbor distances and applies a closed-form ID formula with
several boundary-case adjustments. Here we broadcast over the leading point
axis to get an ``(N, k, k)`` tensor and apply every step vectorized.
"""

import torch
from torch import Tensor

from torchid.estimators.base import LocalEstimator
from torchid.primitives import gather_neighbors, knn


class TLE(LocalEstimator):
    _N_NEIGHBORS = 20

    def __init__(self, epsilon: float = 1e-4, n_neighbors: int | None = None) -> None:
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors

    def get_params(self) -> dict[str, object]:
        return {"epsilon": self.epsilon, "n_neighbors": self.n_neighbors}

    def fit(self, X: object, y: object = None) -> "TLE":
        Xt = self._prepare(X)
        k = self.n_neighbors or self._N_NEIGHBORS
        k = min(k, Xt.shape[0] - 1)
        dists, idx = knn(Xt, k=k)
        nbrs = gather_neighbors(Xt, idx)  # (N, k, D)
        self.dimension_pw_ = _tle_batch(nbrs, dists, epsilon=self.epsilon)
        self.dimension_ = float(self.dimension_pw_.mean())
        return self


def _tle_batch(nbrs: Tensor, dists: Tensor, *, epsilon: float) -> Tensor:
    # nbrs: (N, k, D); dists: (N, k) sorted ascending
    N, k, D = nbrs.shape
    eps = torch.tensor(epsilon, dtype=nbrs.dtype, device=nbrs.device)
    r = dists[:, -1]  # (N,)
    if (r == 0).any():
        raise ValueError("All k-NN distances are zero for some point")
    # V[i,j] = ||nbrs[i] - nbrs[j]|| within each point's neighborhood
    V2 = torch.cdist(nbrs, nbrs, p=2).pow(2)  # (N, k, k); squared distances suffice below
    V = V2.sqrt()
    Di = dists.unsqueeze(2).expand(N, k, k)  # (N, k, k), Di[n,i,j] = dists[n,i]
    Dj = dists.unsqueeze(1).expand(N, k, k)
    r_b = r.view(N, 1, 1)
    Di2 = Di * Di
    Dj2 = Dj * Dj
    r2 = r_b * r_b

    Z2 = 2 * Di2 + 2 * Dj2 - V2

    denom_main = 2.0 * (r2 - Di2)
    # add epsilon to denominators to guard; we overwrite Dr rows below anyway
    safe = denom_main + (denom_main.abs() < torch.finfo(nbrs.dtype).tiny).to(nbrs.dtype) * 1e-30

    a_num = Di2 + V2 - Dj2
    rad_S = (a_num**2 + 4 * V2 * (r2 - Di2)).clamp_min(0.0).sqrt()
    S = r_b * (rad_S - a_num) / safe

    a_num_t = Di2 + Z2 - Dj2
    rad_T = (a_num_t**2 + 4 * Z2 * (r2 - Di2)).clamp_min(0.0).sqrt()
    T = r_b * (rad_T - a_num_t) / safe

    # Boundary case 1: Di == r (row i touches the sphere)
    Dr = Di == r_b  # (N, k, k)
    S_Dr = r_b * V2 / (r2 + V2 - Dj2).clamp_min(torch.finfo(nbrs.dtype).tiny)
    T_Dr = r_b * Z2 / (r2 + Z2 - Dj2).clamp_min(torch.finfo(nbrs.dtype).tiny)
    S = torch.where(Dr, S_Dr, S)
    T = torch.where(Dr, T_Dr, T)
    # Boundary 2: Di == 0
    Di0 = Di == 0
    S = torch.where(Di0, Dj, S)
    T = torch.where(Di0, Dj, T)
    # Boundary 3: Dj == 0
    Dj0 = Dj == 0
    rv_rpv = r_b * V / (r_b + V).clamp_min(torch.finfo(nbrs.dtype).tiny)
    S = torch.where(Dj0, rv_rpv, S)
    T = torch.where(Dj0, rv_rpv, T)
    # Boundary 4: V == 0 (off-diagonal). skdim zeroes the diagonal of V0.
    V0 = V == 0
    diag = torch.eye(k, dtype=torch.bool, device=nbrs.device).unsqueeze(0).expand(N, k, k)
    V0 = V0 & ~diag
    S = torch.where(V0, r_b.expand_as(S), S)
    T = torch.where(V0, r_b.expand_as(T), T)
    nV0 = V0.to(torch.int64).sum(dim=(1, 2))  # (N,)

    # Drop T/S below epsilon (but not on the diagonal)
    TSeps = (eps > T) | (eps > S)
    TSeps = TSeps & ~diag
    nTSeps = TSeps.to(torch.int64).sum(dim=(1, 2))
    T = torch.where(TSeps, r_b.expand_as(T), T)
    S = torch.where(TSeps, r_b.expand_as(S), S)

    logT = torch.log(T / r_b.clamp_min(torch.finfo(nbrs.dtype).tiny))
    logS = torch.log(S / r_b.clamp_min(torch.finfo(nbrs.dtype).tiny))
    # Zero diagonals
    logT = logT.masked_fill(diag, 0.0)
    logS = logS.masked_fill(diag, 0.0)

    s1t = logT.sum(dim=(1, 2))
    s1s = logS.sum(dim=(1, 2))

    # s2: sum log(dists/r) over dists >= epsilon; skdim drops the first few and keeps the tail
    Deps = dists < eps
    nDeps = Deps.to(torch.int64).sum(dim=1)
    safe_dists = dists.clamp_min(torch.finfo(nbrs.dtype).tiny)
    log_d = torch.log(safe_dists / r.unsqueeze(1).clamp_min(torch.finfo(nbrs.dtype).tiny))
    log_d = torch.where(Deps, torch.zeros_like(log_d), log_d)
    s2 = log_d.sum(dim=1)

    num = -2.0 * (k * k - nTSeps - nDeps - nV0).to(nbrs.dtype)
    denom = (s1t + s1s + 2 * s2).clamp_max(-torch.finfo(nbrs.dtype).tiny)
    return num / denom
