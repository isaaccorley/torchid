"""MADA (Farahmand 2007) — manifold-adaptive dimension via two NN radii."""

import math

import torch

from torchid._primitives import knn
from torchid.estimators.base import LocalEstimator


class MADA(LocalEstimator):
    """Local information dimension from the ratio of ``k``-NN to ``k/2``-NN distances."""

    _N_NEIGHBORS = 20

    def __init__(self, DM: bool = False, n_neighbors: int | None = None) -> None:
        if DM:
            raise NotImplementedError("precomputed DM mode not supported")
        self.DM = DM
        self.n_neighbors = n_neighbors

    def get_params(self) -> dict[str, object]:
        return {"DM": self.DM, "n_neighbors": self.n_neighbors}

    def fit(self, X: object, y: object = None) -> "MADA":
        Xt = self._prepare(X)
        k = self.n_neighbors or self._N_NEIGHBORS
        k = min(k, Xt.shape[0] - 1)
        dists, _ = knn(Xt, k=k)
        RK = dists[:, k - 1]
        RK2 = dists[:, int(math.floor(k / 2) - 1)]
        self.dimension_pw_ = math.log(2.0) / torch.log(RK / RK2)
        self.dimension_ = float(self.dimension_pw_.mean())
        return self
