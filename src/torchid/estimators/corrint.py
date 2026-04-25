"""Correlation-integral intrinsic dimension (Grassberger–Procaccia).

d = [log C(r2) − log C(r1)] / log(r2 / r1), where C(r) is the fraction of
pairwise distances below r. Both thresholds come from median kNN distances at
``k1`` and ``k2``.
"""

import torch
from torch import Tensor

from torchid.estimators.base import GlobalEstimator
from torchid.primitives import knn


class CorrInt(GlobalEstimator):
    """Correlation-integral estimator (Grassberger 1983)."""

    def __init__(self, k1: int = 10, k2: int = 20, DM: bool = False) -> None:
        self.k1 = k1
        self.k2 = k2
        self.DM = DM

    def get_params(self) -> dict[str, object]:
        return {"k1": self.k1, "k2": self.k2, "DM": self.DM}

    def _fit(self, X: Tensor) -> Tensor:
        if self.DM:
            raise NotImplementedError("precomputed DM mode not supported")
        n = X.shape[0]
        k2 = min(self.k2, n - 1)
        k1 = min(self.k1, k2 - 1) if self.k1 > k2 else self.k1
        dists, _ = knn(X, k=k2)
        r1 = dists[:, k1 - 1].median()
        r2 = dists[:, k2 - 1].median()
        s1, s2 = _count_pairs(X, r1, r2)
        total_pairs = n * n
        cr1 = s1 / total_pairs
        cr2 = s2 / total_pairs
        return (torch.log(cr2) - torch.log(cr1)) / torch.log(r2 / r1)


def _count_pairs(X: Tensor, r1: Tensor, r2: Tensor, *, chunk: int = 2048) -> tuple[Tensor, Tensor]:
    """Count pairs (i != j) with Euclidean distance below each threshold."""
    n = X.shape[0]
    y_sq = (X * X).sum(dim=1)
    r1_sq = r1 * r1
    r2_sq = r2 * r2
    s1 = torch.zeros((), dtype=X.dtype, device=X.device)
    s2 = torch.zeros((), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        d = x_sq + y_sq.unsqueeze(0) - 2.0 * (xb @ X.T)
        d.clamp_(min=0.0)
        s1 += (d < r1_sq).sum()
        s2 += (d < r2_sq).sum()
    # subtract diagonal (self-matches) that pass the threshold trivially
    s1 = s1 - n
    s2 = s2 - n
    return s1.to(X.dtype), s2.to(X.dtype)
