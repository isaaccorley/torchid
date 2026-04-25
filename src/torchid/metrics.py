"""torchmetrics-compatible streaming intrinsic dimension estimators.

Designed for the deep-learning use case: log ID per epoch / per validation step
without writing a one-shot fit harness.

::

    from torchid.metrics import IntrinsicDimension

    metric = IntrinsicDimension(method="lpca").to(device)
    for batch in val_loader:
        feats = model.encode(batch)            # (B, D)
        metric.update(feats)
    print(metric.compute())                    # 0-D tensor

The metric buffers features across ``update`` calls and runs the chosen
estimator on the concatenation in ``compute``. ``max_samples`` caps memory
via reservoir-style subsampling — useful when a full epoch's features would
not fit on the device.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from torchid.estimators import (
    ESS,
    KNN,
    MADA,
    MLE,
    MOM,
    TLE,
    CorrInt,
    DANCo,
    FisherS,
    MiND_ML,
    TwoNN,
    lPCA,
)

_REGISTRY: dict[str, type] = {
    "lpca": lPCA,
    "twonn": TwoNN,
    "mle": MLE,
    "corrint": CorrInt,
    "mind_ml": MiND_ML,
    "mom": MOM,
    "mada": MADA,
    "knn": KNN,
    "tle": TLE,
    "danco": DANCo,
    "ess": ESS,
    "fishers": FisherS,
}


class IntrinsicDimension(Metric):
    """Streaming intrinsic dimension as a :class:`torchmetrics.Metric`.

    Parameters
    ----------
    method
        Name of the estimator (case-insensitive). One of ``lpca``, ``twonn``,
        ``mle``, ``corrint``, ``mind_ml``, ``mom``, ``mada``, ``knn``, ``tle``,
        ``danco``, ``ess``, ``fishers``.
    max_samples
        Hard cap on the number of features kept across ``update`` calls. If
        more arrive, a uniform random subset is retained (reservoir-style,
        seeded by the global torch RNG). ``None`` keeps everything.
    estimator_kwargs
        Forwarded to the estimator constructor.

    Notes
    -----
    State is accumulated as a list of feature tensors with
    ``dist_reduce_fx="cat"``, so the metric is DDP-compatible — calling
    ``compute`` from rank 0 sees the concatenation across all ranks.
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    features: list[Tensor]

    def __init__(
        self,
        method: str = "lpca",
        max_samples: int | None = 10_000,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__()
        key = method.lower()
        if key not in _REGISTRY:
            raise ValueError(f"unknown method {method!r}. choose from {sorted(_REGISTRY)}")
        self.method = key
        self.max_samples = max_samples
        self.estimator_kwargs = estimator_kwargs
        self.add_state("features", default=[], dist_reduce_fx="cat")

    def update(self, features: Tensor) -> None:  # type: ignore[override]
        """Append a ``(B, D)`` batch of features to the running buffer."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError(f"expected (B, D) features, got shape {tuple(features.shape)}")
        self.features.append(features.detach())

    def compute(self) -> Tensor:
        """Concatenate buffered features, cap to ``max_samples``, fit, return scalar."""
        if not self.features:
            raise RuntimeError("compute() called before any update()")
        X = torch.cat(self.features, dim=0)
        if self.max_samples is not None and X.shape[0] > self.max_samples:
            idx = torch.randperm(X.shape[0], device=X.device)[: self.max_samples]
            X = X[idx]
        cls = _REGISTRY[self.method]
        est = cls(**self.estimator_kwargs).fit(X)
        return torch.tensor(est.dimension_, device=X.device, dtype=X.dtype)
