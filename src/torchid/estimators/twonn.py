"""TwoNN estimator (Facco et al. 2017).

Closed-form: for every point compute ``mu = d2 / d1`` (ratio of 2nd-nearest to
1st-nearest distance). Under a locally-uniform density, ``-log(1 - Femp(mu))``
is linear in ``log(mu)`` with slope = intrinsic dimension.

Fully vectorized on GPU; no per-point loops.
"""

import torch
from torch import Tensor

from torchid.estimators.base import GlobalEstimator
from torchid.primitives import knn


class TwoNN(GlobalEstimator):
    """Intrinsic dimension via the TwoNN algorithm.

    Parameters
    ----------
    discard_fraction: fraction (0..1) of largest ``mu`` values to discard (the
        paper's heuristic for tail sensitivity).
    dist: when True, ``X`` is already a ``(N, 2)`` matrix of (d1, d2) distances.
    """

    def __init__(self, discard_fraction: float = 0.1, dist: bool = False) -> None:
        _validate_discard_fraction(discard_fraction)
        self.discard_fraction = discard_fraction
        self.dist = dist

    def get_params(self) -> dict[str, object]:
        return {"discard_fraction": self.discard_fraction, "dist": self.dist}

    def _fit(self, X: Tensor) -> Tensor:
        if self.dist:
            if X.shape[1] != 2:
                raise ValueError(
                    f"dist=True expects an (N, 2) distance matrix, got {tuple(X.shape)}"
                )
            mu = X[:, 1] / X[:, 0].clamp_min(torch.finfo(X.dtype).tiny)
        else:
            d, _ = knn(X, k=2)
            mu = d[:, 1] / d[:, 0].clamp_min(torch.finfo(X.dtype).tiny)

        N = mu.shape[0]
        keep = _retained_count(N, self.discard_fraction)
        mu_sorted, _ = torch.sort(mu)
        mu_kept = mu_sorted[:keep]
        # use skdim's F_emp = arange(keep) / N  (not / keep) — intentionally divides by full N
        femp = torch.arange(keep, device=X.device, dtype=X.dtype) / N
        x = torch.log(mu_kept)
        y = -torch.log1p(-femp)
        # zero-intercept OLS: slope = sum(xy) / sum(x^2)
        slope = (x * y).sum() / (x * x).sum().clamp_min(torch.finfo(X.dtype).tiny)
        self.x_ = x
        self.y_ = y
        return slope


def _validate_discard_fraction(discard_fraction: float) -> None:
    if not 0 <= discard_fraction < 1:
        raise ValueError(f"discard_fraction must be in [0, 1), got {discard_fraction}")


def _retained_count(n_samples: int, discard_fraction: float) -> int:
    _validate_discard_fraction(discard_fraction)
    keep = int(n_samples * (1 - discard_fraction))
    if keep < 2:
        raise ValueError(
            "TwoNN needs at least 2 retained samples; "
            f"got {keep} from n_samples={n_samples} and discard_fraction={discard_fraction}"
        )
    return keep
