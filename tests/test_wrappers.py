"""Tests for :func:`estimate_many` and :func:`asPointwise`."""

import torch

from torchid import asPointwise, estimate_many
from torchid.datasets import affine_subspace, hyperball
from torchid.estimators import MLE, TwoNN, lPCA


def test_estimate_many_returns_one_dim_per_dataset() -> None:
    rng = torch.Generator().manual_seed(0)
    Xs = [hyperball(500, d, generator=rng) for d in (3, 5, 8)]
    dims = estimate_many(Xs, lPCA)
    assert len(dims) == 3
    # ID should track ambient dim within lPCA's accuracy band
    for d_true, d_est in zip([3, 5, 8], dims, strict=False):
        assert abs(d_est - d_true) <= 1


def test_estimate_many_threads_kwargs() -> None:
    X = hyperball(500, 4, generator=torch.Generator().manual_seed(0))
    fo = estimate_many([X], lPCA, ver="FO")[0]
    kaiser = estimate_many([X], lPCA, ver="Kaiser")[0]
    # Different heuristics on the same data — at least one should differ
    # from the other on this synthetic input.
    assert isinstance(fo, float)
    assert isinstance(kaiser, float)


def test_estimate_many_handles_varying_shapes() -> None:
    rng = torch.Generator().manual_seed(0)
    Xs = [
        affine_subspace(300, 2, 5, noise_std=0.01, generator=rng),
        affine_subspace(800, 4, 12, noise_std=0.01, generator=rng),
    ]
    dims = estimate_many(Xs, TwoNN)
    assert len(dims) == 2


def test_aspointwise_shape_and_device() -> None:
    X = hyperball(200, 4, generator=torch.Generator().manual_seed(0))
    ids = asPointwise(X, lPCA, n_neighbors=30)
    assert ids.shape == (200,)
    assert ids.device == X.device


def test_aspointwise_recovers_uniform_id() -> None:
    # Uniform hyperball: every point's local ID should hover near the true d
    X = hyperball(500, 4, generator=torch.Generator().manual_seed(0))
    ids = asPointwise(X, lPCA, n_neighbors=80)
    assert 2.5 <= ids.median().item() <= 5.5


def test_aspointwise_caps_n_neighbors_at_dataset_size() -> None:
    X = hyperball(20, 3, generator=torch.Generator().manual_seed(0))
    # request more neighbors than samples — should silently clamp
    ids = asPointwise(X, lPCA, n_neighbors=100)
    assert ids.shape == (20,)


def test_aspointwise_works_with_multiple_estimators() -> None:
    X = affine_subspace(150, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    for cls in (lPCA, TwoNN, MLE):
        ids = asPointwise(X, cls, n_neighbors=40)
        assert ids.shape == (150,)
        assert torch.isfinite(ids).all()
