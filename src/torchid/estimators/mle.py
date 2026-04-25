"""MLE (Levina–Bickel) intrinsic dimension estimator.

Closed-form pointwise estimator::

    d_i = (k - 1) / Σ_{j=1..k-1} log(R_k(i) / R_j(i))

aggregated by harmonic mean (comb='mle'), mean, or median. Fully batched —
no per-point Python loops and no ``scipy.integrate.quad``. Only the zero-noise
branch of skdim's MLE is implemented; noisy integral approximations are rarely
used in practice and would reintroduce per-point numerical integration.
"""

import torch
from torch import Tensor

from torchid.estimators.base import LocalEstimator
from torchid.primitives import knn, log_knn_ratios


class MLE(LocalEstimator):
    """Levina–Bickel / Haro-at-σ=0 MLE.

    Parameters
    ----------
    K: unused (present for skdim parity); the neighborhood size is controlled
        by ``n_neighbors``.
    unbiased: use (k-2) rather than (k-1) in the numerator.
    comb: aggregation across pointwise estimates (``'mle'`` → harmonic mean,
        ``'mean'``, ``'median'``).
    n_neighbors: k used for kNN distances. Defaults to 20 (matches skdim).
    """

    _N_NEIGHBORS = 20

    def __init__(
        self,
        dnoise: object = None,
        sigma: float = 0.0,
        n: int | None = None,
        integral_approximation: str = "Haro",
        unbiased: bool = False,
        neighborhood_based: bool = True,
        K: int = 5,
    ) -> None:
        if dnoise is not None or sigma != 0.0:
            raise NotImplementedError(
                "torchid.MLE only supports the zero-noise closed-form "
                "(dnoise=None, sigma=0); use skdim.id.MLE for the Haro integral branch."
            )
        self.dnoise = dnoise
        self.sigma = sigma
        self.n = n
        self.integral_approximation = integral_approximation
        self.unbiased = unbiased
        self.neighborhood_based = neighborhood_based
        self.K = K

    def get_params(self) -> dict[str, object]:
        return {"unbiased": self.unbiased, "neighborhood_based": self.neighborhood_based}

    def fit(  # type: ignore[override]
        self,
        X: object,
        y: object = None,
        *,
        n_neighbors: int | None = None,
        comb: str = "mle",
    ) -> "MLE":
        Xt = self._prepare(X)
        k = n_neighbors if n_neighbors is not None else self._N_NEIGHBORS
        k = min(k, Xt.shape[0] - 1)
        self.n_neighbors = k
        self.comb = comb

        dists, _ = knn(Xt, k=k)
        self.dimension_pw_ = self._pointwise(dists)

        if comb == "mle":
            self.dimension_ = float(1.0 / (1.0 / self.dimension_pw_).mean())
        elif comb == "mean":
            self.dimension_ = float(self.dimension_pw_.mean())
        elif comb == "median":
            self.dimension_ = float(self.dimension_pw_.median())
        else:
            raise ValueError(f"comb must be one of 'mle','mean','median', got {comb!r}")
        return self

    def _pointwise(self, dists: Tensor) -> Tensor:
        # log_knn_ratios returns (N, k-1) of log(R_k / R_j), j=0..k-2
        logs = log_knn_ratios(dists)
        k = dists.shape[1]
        kfac = k - 2 if self.unbiased else k - 1
        denom = logs.sum(dim=1).clamp_min(torch.finfo(dists.dtype).tiny)
        return kfac / denom

    def _fit(self, X: Tensor) -> Tensor:  # pragma: no cover - fit is overridden
        raise NotImplementedError
