"""KNN graph-length regression intrinsic dimension estimator (Carter et al. 2010)."""

import warnings

import numpy as np
import torch
from torch import Tensor

from torchid.estimators.base import GlobalEstimator
from torchid.primitives import knn


class KNN(GlobalEstimator):
    """Carter-et-al.-style kNN graph-length regression."""

    def __init__(
        self,
        k: int | None = None,
        ps: np.ndarray | None = None,
        M: int = 1,
        gamma: int = 2,
    ) -> None:
        """Configure the bootstrap graphs.

        Each of the ``M`` samples per size in ``ps`` uses ``k`` neighbors per point
        and sums edge lengths raised to ``gamma``.
        """
        self.k = k
        self.ps = ps
        self.M = M
        self.gamma = gamma

    def get_params(self) -> dict[str, object]:
        """Return the graph and bootstrap parameters used for the fit."""
        return {"k": self.k, "ps": self.ps, "M": self.M, "gamma": self.gamma}

    def _fit(self, X: Tensor) -> Tensor:
        """Estimate dimension from how float64 kNN graph lengths scale with bootstrap size."""
        n = X.shape[0]
        k_eff = 2 if self.k is None else self.k
        ps = np.arange(k_eff + 1, k_eff + 5) if self.ps is None else np.asarray(self.ps)
        Q = len(ps)
        if min(ps) <= k_eff or max(ps) > n:
            raise ValueError("ps must satisfy k < ps <= n")

        coords = X
        if X.device.type == "mps":
            warnings.warn(
                "KNN computes in float64. Moving the fit to CPU because MPS lacks float64 support.",
                stacklevel=3,
            )
            coords = X.cpu()
        coords = coords.to(torch.float64)

        L = coords.new_zeros((Q, self.M))
        for i in range(Q):
            p = int(ps[i])
            for j in range(self.M):
                samp_ind_np = np.random.randint(0, n, p)
                samp_ind = torch.as_tensor(samp_ind_np, dtype=torch.long, device=coords.device)
                distances, _ = knn(coords[samp_ind], k_eff)
                L[i, j] = (distances**self.gamma).sum()

        d = X.shape[1]
        ps_t = torch.as_tensor(ps, dtype=coords.dtype, device=coords.device)
        L_sum_M = L.sum(dim=1)
        epsilons = coords.new_empty(d)
        for m0 in range(d):
            m = m0 + 1
            alpha = (m - self.gamma) / m
            ps_alpha = ps_t**alpha
            hat_c = (ps_alpha * L_sum_M).sum() / ((ps_alpha**2).sum() * self.M)
            epsilons[m0] = ((L - (hat_c * ps_alpha).unsqueeze(1)) ** 2).sum()

        de = int(torch.argmin(epsilons).item()) + 1
        self.residual_ = float(epsilons[de - 1])
        return torch.tensor(float(de), dtype=X.dtype, device=X.device)
