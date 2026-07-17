"""Tests for the differentiable functional API (:mod:`torchid.functional`)."""

import pytest
import torch

from torchid.datasets import affine_subspace
from torchid.estimators import MADA, MLE, MOM, TwoNN, lPCA
from torchid.functional import intrinsic_dimension, mada_id, mle_id, mom_id, pr_id, twonn_id

FUNCTIONALS = {
    "mada": mada_id,
    "mle": mle_id,
    "mom": mom_id,
    "pr": pr_id,
    "twonn": twonn_id,
}


@pytest.fixture
def X():
    return affine_subspace(600, 4, 12, noise_std=0.05, generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# parity: without gradients the functionals must reproduce the estimator classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comb", ["mle", "mean", "median"])
def test_mle_id_matches_estimator(X, comb):
    expected = MLE().fit(X, comb=comb).dimension_
    assert float(mle_id(X, comb=comb)) == pytest.approx(expected, rel=1e-6)


def test_twonn_id_matches_estimator(X):
    expected = TwoNN().fit(X).dimension_
    assert float(twonn_id(X)) == pytest.approx(expected, rel=1e-6)


def test_mom_id_matches_estimator(X):
    expected = MOM().fit(X).dimension_
    assert float(mom_id(X)) == pytest.approx(expected, rel=1e-6)


def test_mada_id_matches_estimator(X):
    expected = MADA().fit(X).dimension_
    assert float(mada_id(X)) == pytest.approx(expected, rel=1e-6)


def test_pr_id_matches_lpca_participation_ratio(X):
    expected = lPCA(ver="participation_ratio").fit(X).dimension_
    assert float(pr_id(X)) == pytest.approx(expected, rel=1e-6)


def test_dispatcher_matches_direct_call(X):
    for name, fn in FUNCTIONALS.items():
        assert float(intrinsic_dimension(X, method=name)) == pytest.approx(float(fn(X)))


def test_dispatcher_unknown_method_raises(X):
    with pytest.raises(ValueError, match="unknown method"):
        intrinsic_dimension(X, method="bogus")


def test_mle_id_unknown_comb_raises(X):
    with pytest.raises(ValueError, match="comb must be"):
        mle_id(X, comb="bogus")


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONALS))
def test_grad_path_value_matches_no_grad(X, name):
    # recomputed (differentiable) distances must agree with the kNN distances
    Xg = X.clone().requires_grad_(True)
    with_grad = float(FUNCTIONALS[name](Xg).detach())
    without = float(FUNCTIONALS[name](X))
    assert with_grad == pytest.approx(without, rel=1e-4)


@pytest.mark.parametrize("name", sorted(FUNCTIONALS))
def test_backward_produces_finite_nonzero_grads(X, name):
    Xg = X.clone().requires_grad_(True)
    d = FUNCTIONALS[name](Xg)
    assert d.ndim == 0
    assert d.requires_grad
    d.backward()
    assert Xg.grad is not None
    assert torch.isfinite(Xg.grad).all()
    assert Xg.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("mada", {"n_neighbors": 6}),
        ("mle", {"n_neighbors": 6}),
        ("mom", {"n_neighbors": 6}),
        ("pr", {}),
        ("twonn", {}),
    ],
)
def test_gradcheck(name, kwargs):
    # float64 + well-separated random points so the numeric jacobian never
    # crosses a neighbor-order tie
    X = torch.randn(40, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    X = X.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda inp: FUNCTIONALS[name](inp, **kwargs), (X,), eps=1e-6, atol=1e-4
    )


def test_no_grad_input_returns_no_grad_output(X):
    d = twonn_id(X)
    assert not d.requires_grad


@pytest.mark.cuda
def test_functional_grads_on_cuda():
    X = torch.randn(512, 16, generator=torch.Generator().manual_seed(0)).cuda()
    Xg = X.clone().requires_grad_(True)
    d = mle_id(Xg, n_neighbors=10)
    d.backward()
    assert Xg.grad is not None
    assert Xg.grad.device.type == "cuda"
    assert torch.isfinite(Xg.grad).all()
