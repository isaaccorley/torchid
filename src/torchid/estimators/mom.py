"""Method of Moments (Amsaleg et al. 2018) intrinsic dimension estimator.

Per-point: ``d_i = -m1 / (m1 - w)`` where ``w = d_k`` (last NN distance) and
``m1 = mean(d_j, j=1..k)``. Fully vectorized.
"""

from torch import Tensor

from torchid.estimators.base import LocalEstimator
from torchid.primitives import _effective_n_neighbors, knn


class MOM(LocalEstimator):
    """Method-of-moments local intrinsic dimension."""

    _N_NEIGHBORS = 100  # matches skdim LocalEstimator default

    def __init__(self, n_neighbors: int | None = None) -> None:
        self.n_neighbors = n_neighbors

    def get_params(self) -> dict[str, object]:
        return {"n_neighbors": self.n_neighbors}

    def fit(self, X: object, y: object = None) -> "MOM":
        Xt = self._prepare(X)
        requested = self.n_neighbors if self.n_neighbors is not None else self._N_NEIGHBORS
        k = _effective_n_neighbors(requested, Xt.shape[0], minimum=2)
        dists, _ = knn(Xt, k=k)
        self.dimension_pw_ = self._mom(dists)
        self.dimension_ = float(self.dimension_pw_.mean())
        return self

    def _mom(self, dists: Tensor) -> Tensor:
        w = dists[:, -1]
        m1 = dists.mean(dim=1)
        return -m1 / (m1 - w)
