"""MiND_ML (Rozza et al. 2012) intrinsic dimension estimator.

``rho_i = d_1(i) / d_k(i)`` — ratio of nearest to k-th NN distance. For an
integer d the log-likelihood is::

    ℓ(d) = N log(k d) + (d - 1) Σ log ρ_i + (k - 1) Σ log(1 - ρ_i^d)

MLi picks the maximizing integer in [1, D]; MLk refines via a dense 1-D grid
(in place of ``scipy.optimize.L-BFGS-B``) — the objective is smooth and bounded
so a grid at 0.001 spacing matches scipy's precision within tolerance.
"""

import torch
from torch import Tensor

from torchid._primitives import knn
from torchid.estimators.base import GlobalEstimator


class MiND_ML(GlobalEstimator):
    """MiND_ML{i,k} intrinsic dimension estimator."""

    def __init__(self, k: int = 20, D: int = 10, ver: str = "MLk") -> None:
        if ver not in ("MLi", "MLk"):
            raise ValueError(f"ver must be 'MLi' or 'MLk', got {ver!r}")
        self.k = k
        self.D = D
        self.ver = ver

    def get_params(self) -> dict[str, object]:
        return {"k": self.k, "D": self.D, "ver": self.ver}

    def _fit(self, X: Tensor) -> Tensor:
        n = X.shape[0]
        k = min(self.k + 1, n - 1)
        dists, _ = knn(X, k=k)
        rhos = dists[:, 0] / dists[:, -1].clamp_min(torch.finfo(X.dtype).tiny)
        rhos = rhos.clamp(min=torch.finfo(X.dtype).tiny, max=1 - torch.finfo(X.dtype).eps)
        sum_log_rho = torch.log(rhos).sum()
        N = rhos.shape[0]
        # integer MLE over d = 1..D
        d_int = torch.arange(1, self.D + 1, device=X.device, dtype=X.dtype)
        ll_int = _lld(d_int, rhos, N, k, sum_log_rho)
        mli = torch.argmax(ll_int) + 1
        if self.ver == "MLi":
            return mli.to(X.dtype)
        # MLk: dense grid refinement over [max(mli-1, 0), min(mli+1, D)]
        lo = float(max(int(mli.item()) - 1, 0))
        hi = float(min(int(mli.item()) + 1, self.D))
        grid = torch.linspace(
            max(lo, 1e-6), hi, steps=int((hi - lo) / 0.001) + 1, device=X.device, dtype=X.dtype
        )
        ll_c = _lld(grid, rhos, N, k, sum_log_rho)
        return grid[torch.argmax(ll_c)]


def _lld(d: Tensor, rhos: Tensor, N: int, k: int, sum_log_rho: Tensor) -> Tensor:
    # d: (M,) or scalar. rhos: (N,). Returns (M,) log-likelihood.
    d = d.unsqueeze(-1) if d.ndim == 1 else d  # (M, 1)
    rho_d = rhos.unsqueeze(0) ** d  # (M, N)
    log_k_d = torch.log((k * d).clamp_min(torch.finfo(rhos.dtype).tiny)).squeeze(-1)
    term1 = N * log_k_d
    term2 = (d.squeeze(-1) - 1) * sum_log_rho
    term3 = (k - 1) * torch.log1p(-rho_d).sum(dim=-1)
    return term1 + term2 + term3
