"""Tests for the streaming :class:`IntrinsicDimension` torchmetrics adapter."""

import pytest
import torch

import torchid.metrics as metrics_module
from torchid.datasets import affine_subspace, hyperball
from torchid.estimators import lPCA
from torchid.metrics import IntrinsicDimension


def test_metric_matches_one_shot_fit() -> None:
    X = affine_subspace(1000, 4, 12, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    expected = lPCA().fit(X).dimension_

    metric = IntrinsicDimension(method="lpca", max_samples=None)
    for chunk in X.split(100):
        metric.update(chunk)
    assert abs(float(metric.compute()) - expected) < 1e-5


def test_metric_max_samples_subsamples() -> None:
    X = hyperball(5000, 5, generator=torch.Generator().manual_seed(0))
    metric = IntrinsicDimension(method="twonn", max_samples=500)
    metric.update(X)
    out = float(metric.compute())
    # With reservoir down to 500 points the estimate should still bracket the
    # true ID = 5 within the looser TwoNN finite-sample band.
    assert 3 < out < 8


def test_metric_max_samples_caps_stored_features(monkeypatch) -> None:
    fitted_sizes = []

    class SpyEstimator:
        def fit(self, X):
            fitted_sizes.append(X.shape[0])
            self.dimension_ = 1.0
            return self

    monkeypatch.setitem(metrics_module._REGISTRY, "spy", SpyEstimator)
    metric = IntrinsicDimension(method="spy", max_samples=25)
    for _ in range(10):
        metric.update(torch.randn(20, 4))
        assert sum(batch.shape[0] for batch in metric.features) <= 25
    metric.compute()
    assert fitted_sizes == [25]


@pytest.mark.parametrize("max_samples", [0, -1])
def test_metric_rejects_invalid_max_samples(max_samples) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        IntrinsicDimension(max_samples=max_samples)


def test_metric_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        IntrinsicDimension(method="bogus")


def test_metric_compute_before_update_raises() -> None:
    with pytest.raises(RuntimeError, match="before any update"):
        IntrinsicDimension().compute()


def test_metric_invalid_input_shape_raises() -> None:
    metric = IntrinsicDimension()
    with pytest.raises(ValueError, match=r"expected \(B, D\)"):
        metric.update(torch.zeros(2, 3, 4))


def test_metric_rejects_empty_or_inconsistent_updates() -> None:
    metric = IntrinsicDimension()
    with pytest.raises(ValueError, match="at least one"):
        metric.update(torch.empty(0, 3))
    metric.update(torch.randn(2, 3))
    with pytest.raises(ValueError, match="same feature dimension"):
        metric.update(torch.randn(2, 4))


def test_metric_1d_input_promoted_to_batch() -> None:
    metric = IntrinsicDimension(method="lpca", max_samples=None)
    metric.update(torch.randn(8))  # treated as (1, 8)
    metric.update(torch.randn(20, 8))
    out = metric.compute()
    assert out.ndim == 0


def test_metric_reset_clears_state() -> None:
    metric = IntrinsicDimension()
    metric.update(torch.randn(50, 6))
    metric.reset()
    with pytest.raises(RuntimeError, match="before any update"):
        metric.compute()


def test_metric_estimator_kwargs_threaded_through(monkeypatch) -> None:
    class SpyEstimator:
        def __init__(self, result):
            self.result = result

        def fit(self, X):
            self.dimension_ = self.result
            return self

    monkeypatch.setitem(metrics_module._REGISTRY, "spy", SpyEstimator)
    metric = IntrinsicDimension(method="spy", result=3.5, max_samples=None)
    metric.update(torch.randn(10, 2))
    assert float(metric.compute()) == 3.5
