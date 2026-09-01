"""Intrinsic dimension as a minimizable training objective (see :mod:`torchid.functional`)."""

from typing import Any

from torch import Tensor, nn

from torchid.functional import _REGISTRY, intrinsic_dimension

__all__ = ["IntrinsicDimensionLoss"]

_MODES = ("maximize", "minimize", "target")


class IntrinsicDimensionLoss(nn.Module):
    """Differentiable ID as a loss: ``1 - id/D`` ('maximize'), ``id/D``
    ('minimize'), or ``((id - target)/D)²`` ('target'); ``normalize=False``
    drops the ``1/D``. ``method_kwargs`` forward to the estimate; the detached
    raw estimate lands on ``self.dimension_`` after each forward."""

    dimension_: Tensor | None

    def __init__(
        self,
        method: str = "twonn",
        mode: str = "maximize",
        target: float | None = None,
        normalize: bool = True,
        **method_kwargs: Any,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if mode == "target" and target is None:
            raise ValueError("mode='target' requires a target dimension")
        if mode != "target" and target is not None:
            raise ValueError(f"target is only valid with mode='target', got mode={mode!r}")
        if method.lower() not in _REGISTRY:
            raise ValueError(f"unknown method {method!r}. choose from {sorted(_REGISTRY)}")
        self.method = method.lower()
        self.mode = mode
        self.target = target
        self.normalize = normalize
        self.method_kwargs = method_kwargs
        self.dimension_ = None

    def forward(self, X: Tensor) -> Tensor:
        """Return the 0-D loss for a ``(B, D)`` batch."""
        d = intrinsic_dimension(X, method=self.method, **self.method_kwargs)
        self.dimension_ = d.detach()
        scale = float(X.shape[1]) if self.normalize else 1.0
        if self.mode == "maximize":
            return 1.0 - d / scale if self.normalize else -d
        if self.mode == "minimize":
            return d / scale
        assert self.target is not None
        return ((d - self.target) / scale) ** 2

    def extra_repr(self) -> str:
        parts = [f"method={self.method!r}", f"mode={self.mode!r}"]
        if self.target is not None:
            parts.append(f"target={self.target}")
        if not self.normalize:
            parts.append("normalize=False")
        parts.extend(f"{k}={v!r}" for k, v in self.method_kwargs.items())
        return ", ".join(parts)
