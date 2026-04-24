"""Base classes for intrinsic dimension estimators.

Two thin superclasses that do the minimum needed to mirror scikit-dimension's
`fit(X) -> self` / `.dimension_` pattern while letting inputs stay on a GPU
tensor end-to-end.
"""

from torch import Tensor

from torchid._primitives import as_tensor


class _BaseEstimator:
    """Input coercion + sklearn-style ``repr``."""

    def _prepare(self, X: object) -> Tensor:
        return as_tensor(X)

    def get_params(self) -> dict[str, object]:
        """Subclasses override to expose constructor args for introspection."""
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


class GlobalEstimator(_BaseEstimator):
    """Estimators that emit one intrinsic dimension for the whole dataset.

    Subclasses implement :meth:`_fit` to return a 0-D tensor; the base class
    wraps it in ``float(...)`` and stores the result on ``self.dimension_``.
    """

    dimension_: float

    def fit(self, X: object, y: object = None) -> "GlobalEstimator":  # noqa: ARG002
        self.dimension_ = float(self._fit(self._prepare(X)))
        return self

    def _fit(self, X: Tensor) -> Tensor:
        raise NotImplementedError


class LocalEstimator(_BaseEstimator):
    """Estimators that emit one intrinsic dimension per point.

    All local estimators override :meth:`fit` directly because they each have
    a slightly different ``k``-neighbor / aggregation signature. The base
    class exists only so ``isinstance`` checks are meaningful.
    """

    dimension_pw_: Tensor
    dimension_: float
