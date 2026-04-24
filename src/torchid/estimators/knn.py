"""KNN graph-length regression intrinsic dimension estimator (Carter et al. 2010)."""

import numpy as np
import torch
from torch import Tensor

from torchid._primitives import pairwise_sqdist
from torchid.estimators.base import GlobalEstimator


class KNN(GlobalEstimator):
    """Carter-et-al.-style kNN graph-length regression."""

    def __init__(
        self,
        k: int | None = None,
        ps: np.ndarray | None = None,
        M: int = 1,
        gamma: int = 2,
    ) -> None:
        self.k = k
        self.ps = ps
        self.M = M
        self.gamma = gamma

    def get_params(self) -> dict[str, object]:
        return {"k": self.k, "ps": self.ps, "M": self.M, "gamma": self.gamma}

    def _fit(self, X: Tensor) -> Tensor:
        n = X.shape[0]
        k_eff = 2 if self.k is None else self.k
        ps = np.arange(k_eff + 1, k_eff + 5) if self.ps is None else np.asarray(self.ps)
        Q = len(ps)
        if min(ps) <= k_eff or max(ps) > n:
            raise ValueError("ps must satisfy k < ps <= n")

        # Full pairwise distance matrix — O(n^2), but that's what the estimator requires.
        # Use float64 to match skdim's scipy.spatial.distance.pdist precision (the
        # argmin over integer d is sensitive to small cumulative errors).
        X64 = X.to(torch.float64)
        D2 = pairwise_sqdist(X64)
        D = D2.sqrt()

        L = torch.zeros((Q, self.M), dtype=torch.float64, device=X.device)
        for i in range(Q):
            p = int(ps[i])
            for j in range(self.M):
                samp_ind_np = np.random.randint(0, n, p)
                samp_ind = torch.as_tensor(samp_ind_np, dtype=torch.long, device=X.device)
                sub = D[samp_ind][:, samp_ind]  # (p, p)
                # per row: sort, take the k closest non-self entries (indices 1..k)
                vals, _ = torch.sort(sub, dim=1)
                L[i, j] = (vals[:, 1 : k_eff + 1] ** self.gamma).sum()

        d = X.shape[1]
        ps_t = torch.as_tensor(ps, dtype=torch.float64, device=X.device)
        L_sum_M = L.sum(dim=1)
        epsilons = torch.empty(d, dtype=torch.float64, device=X.device)
        for m0 in range(d):
            m = m0 + 1
            alpha = (m - self.gamma) / m
            ps_alpha = ps_t**alpha
            hat_c = (ps_alpha * L_sum_M).sum() / ((ps_alpha**2).sum() * self.M)
            epsilons[m0] = ((L - (hat_c * ps_alpha).unsqueeze(1)) ** 2).sum()

        de = int(torch.argmin(epsilons).item()) + 1
        self.residual_ = float(epsilons[de - 1])
        return torch.tensor(float(de), dtype=X.dtype, device=X.device)
