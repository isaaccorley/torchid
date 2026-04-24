"""Base classes for intrinsic dimension estimators.

Mirrors scikit-dimension's `GlobalEstimator` / `LocalEstimator` fit/attribute
pattern but lets inputs stay on a GPU tensor end-to-end.
"""

import torch
from torch import Tensor

from torchid._primitives import as_tensor


class _BaseEstimator:
    """Shared plumbing: input coercion, device/dtype handling, sklearn-style repr."""

    def __init__(self) -> None:
        self._fitted: bool = False

    def _prepare(self, X: object) -> Tensor:
        Xt = as_tensor(X)
        self._device = Xt.device
        self._dtype = Xt.dtype
        self._n_, self._d_ = Xt.shape
        return Xt

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be .fit(X) before accessing results")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def get_params(self) -> dict[str, object]:
        """Mimics sklearn's :meth:`get_params` for introspection in tests."""
        # subclasses override when they have params
        return {}


class GlobalEstimator(_BaseEstimator):
    """Estimators that emit one intrinsic dimension for the whole dataset."""

    dimension_: float

    def fit(self, X: object, y: object = None) -> "GlobalEstimator":  # noqa: ARG002
        Xt = self._prepare(X)
        self.dimension_ = float(self._fit(Xt))
        self._fitted = True
        return self

    def _fit(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def fit_transform(self, X: object, y: object = None) -> float:  # noqa: ARG002
        return self.fit(X).dimension_


class LocalEstimator(_BaseEstimator):
    """Estimators that emit one intrinsic dimension per point."""

    dimension_pw_: Tensor
    dimension_: float  # mean of pointwise dimensions

    def fit(self, X: object, y: object = None) -> "LocalEstimator":  # noqa: ARG002
        Xt = self._prepare(X)
        pw = self._fit(Xt)
        self.dimension_pw_ = pw
        self.dimension_ = float(pw.mean())
        self._fitted = True
        return self

    def _fit(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def dimension_pw_numpy(self) -> "object":
        """Return pointwise dimensions as a numpy array (convenience for tests/plots)."""
        self._check_fitted()
        return self.dimension_pw_.detach().cpu().numpy()
