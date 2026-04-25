"""High-level wrappers around the estimator API.

- :func:`estimate_many` fits the same estimator across a list of datasets.
- :func:`asPointwise` turns any global estimator into a per-point local one.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from torchid._primitives import as_tensor, gather_neighbors, knn

__all__ = ["asPointwise", "estimate_many"]


def estimate_many(
    datasets: Iterable[object],
    estimator: type,
    **kwargs: Any,
) -> list[float]:
    """Fit ``estimator`` independently on each entry in ``datasets``.

    Useful for hyperparameter / representation sweeps where you want one
    intrinsic-dimension estimate per input without writing the loop yourself.

    Args:
        datasets: iterable of array-likes, each ``(N_i, D_i)``. Per-entry
            ``N_i`` and ``D_i`` may differ.
        estimator: an estimator class from :mod:`torchid.estimators` (e.g.
            ``lPCA``, ``TwoNN``).
        **kwargs: forwarded to ``estimator(**kwargs)`` for every fit.

    Returns:
        List of scalar dimensions, one per input dataset.

    Example:
        >>> import torch
        >>> from torchid import estimate_many
        >>> from torchid.estimators import TwoNN
        >>>
        >>> rng = torch.Generator().manual_seed(0)
        >>> low_d = torch.randn(2000, 5, generator=rng) @ torch.randn(5, 30, generator=rng)
        >>> high_d = torch.randn(2000, 30, generator=rng)
        >>> dims = estimate_many([low_d, high_d], TwoNN)
        >>> # `dims[0]` is approximately 5; `dims[1]` is much larger.
    """
    return [estimator(**kwargs).fit(X).dimension_ for X in datasets]


def asPointwise(
    X: object,
    estimator: type,
    n_neighbors: int = 100,
    **kwargs: Any,
) -> Tensor:
    """Per-point local ID by running ``estimator`` on each point's ``k``-NN patch.

    Mirrors :func:`skdim.asPointwise`. For each point ``i`` in ``X`` we take
    its ``n_neighbors`` nearest neighbors as a local patch and fit
    ``estimator`` on that patch — the resulting dimension becomes the local
    ID at point ``i``. Output is a ``(N,)`` tensor.

    This complements the native local estimators (``MOM``, ``MADA``, ``TLE``,
    ``ESS``): those have closed-form per-point formulas, while this wrapper
    works with *any* global estimator at the cost of ``N`` fits.

    Use it when you want:

    - a per-point ID map for visualization or anomaly detection (high local
      ID = noisy / off-manifold points),
    - to localize a method that has no native local form (``FisherS``,
      ``CorrInt``, ``DANCo``),
    - to study how ID varies across regions of a manifold.

    Args:
        X: ``(N, D)`` array-like.
        estimator: a global estimator class from :mod:`torchid.estimators`.
        n_neighbors: neighborhood size for each per-point fit.
        **kwargs: forwarded to ``estimator(**kwargs)``.

    Returns:
        ``(N,)`` tensor of per-point dimensions on the same device as ``X``.

    Example:
        >>> import torch
        >>> from torchid import asPointwise
        >>> from torchid.estimators import lPCA
        >>>
        >>> X = torch.randn(500, 10, generator=torch.Generator().manual_seed(0))
        >>> ids = asPointwise(X, lPCA, n_neighbors=50)
        >>> ids.shape
        torch.Size([500])

    Performance:
        ``N`` estimator fits — pick a fast estimator (``lPCA``, ``TwoNN``,
        ``MLE``) for large ``N``. ``DANCo`` / ``ESS`` will be slow.
    """
    Xt = as_tensor(X)
    n = Xt.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    _, idx = knn(Xt, k=n_neighbors)
    nbrs = gather_neighbors(Xt, idx)  # (N, k, D)
    out = torch.empty(n, device=Xt.device, dtype=Xt.dtype)
    for i in range(n):
        out[i] = estimator(**kwargs).fit(nbrs[i]).dimension_
    return out
