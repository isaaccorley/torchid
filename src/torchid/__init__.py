"""torchid — GPU-accelerated intrinsic dimension estimators."""

from importlib.util import find_spec
from typing import TYPE_CHECKING

from torchid import datasets, estimators, functional, primitives
from torchid.losses import IntrinsicDimensionLoss
from torchid.wrappers import asPointwise, estimate_many

if TYPE_CHECKING:
    from torchid.metrics import IntrinsicDimension

__all__ = [
    "IntrinsicDimensionLoss",
    "asPointwise",
    "datasets",
    "estimate_many",
    "estimators",
    "functional",
    "primitives",
]

if find_spec("torchmetrics") is not None:
    __all__ += ["IntrinsicDimension"]


def __getattr__(name: str) -> "type[IntrinsicDimension]":
    if name == "IntrinsicDimension":
        from torchid.metrics import IntrinsicDimension

        return IntrinsicDimension
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
