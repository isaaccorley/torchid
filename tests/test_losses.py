"""Tests for :class:`torchid.losses.IntrinsicDimensionLoss`."""

import pytest
import torch
from torch import nn

from torchid.functional import mle_id, pr_id
from torchid.losses import IntrinsicDimensionLoss


def _features(n=400, d=3, D=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen) @ torch.randn(d, D, generator=gen)


# ---------------------------------------------------------------------------
# construction / semantics
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        IntrinsicDimensionLoss(mode="bogus")


def test_target_mode_requires_target():
    with pytest.raises(ValueError, match="requires a target"):
        IntrinsicDimensionLoss(mode="target")


def test_target_without_target_mode_raises():
    with pytest.raises(ValueError, match="only valid with mode='target'"):
        IntrinsicDimensionLoss(mode="maximize", target=3.0)


def test_unknown_method_raises_at_construction():
    with pytest.raises(ValueError, match="unknown method"):
        IntrinsicDimensionLoss(method="bogus")


def test_normalized_modes_are_complementary():
    X = _features()
    lo = float(IntrinsicDimensionLoss(method="pr", mode="maximize")(X))
    hi = float(IntrinsicDimensionLoss(method="pr", mode="minimize")(X))
    assert lo + hi == pytest.approx(1.0)


def test_unnormalized_maximize_is_negative_id():
    X = _features()
    loss = IntrinsicDimensionLoss(method="pr", mode="maximize", normalize=False)(X)
    assert float(loss) == pytest.approx(-float(pr_id(X)))


def test_target_mode_value():
    X = _features()
    d = float(pr_id(X))
    loss = IntrinsicDimensionLoss(method="pr", mode="target", target=d)(X)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_dimension_attribute_stores_raw_estimate():
    X = _features()
    loss_fn = IntrinsicDimensionLoss(method="twonn", mode="minimize")
    loss_fn(X)
    assert loss_fn.dimension_ is not None
    assert not loss_fn.dimension_.requires_grad
    assert float(loss_fn.dimension_) == pytest.approx(float(loss_fn(X) * X.shape[1]))


def test_method_kwargs_forwarded():
    X = _features()
    l1 = float(IntrinsicDimensionLoss(method="mle", mode="minimize", n_neighbors=5)(X))
    l2 = float(IntrinsicDimensionLoss(method="mle", mode="minimize", n_neighbors=50)(X))
    assert l1 != l2
    assert float(IntrinsicDimensionLoss(method="mle", mode="minimize", n_neighbors=5)(X)) == (
        pytest.approx(float(mle_id(X, n_neighbors=5)) / X.shape[1])
    )


def test_repr_mentions_config():
    r = repr(IntrinsicDimensionLoss(method="mle", mode="target", target=4.0, n_neighbors=10))
    assert "method='mle'" in r
    assert "target=4.0" in r
    assert "n_neighbors=10" in r


# ---------------------------------------------------------------------------
# optimization: the losses actually move ID in the requested direction
# ---------------------------------------------------------------------------


def test_maximize_pr_recovers_rank():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(512, 8, generator=gen)
    # near-rank-1 map: maximizing ID should spread the spectrum back out
    W0 = torch.outer(torch.randn(8, generator=gen), torch.randn(8, generator=gen))
    W0 = W0 + 0.01 * torch.randn(8, 8, generator=gen)
    W = nn.Parameter(W0.clone())
    loss_fn = IntrinsicDimensionLoss(method="pr", mode="maximize")
    opt = torch.optim.Adam([W], lr=0.05)
    before = float(pr_id(Z @ W0))
    for _ in range(100):
        opt.zero_grad()
        loss_fn(Z @ W).backward()
        opt.step()
    after = float(pr_id((Z @ W).detach()))
    assert before < 2.0
    assert after > before + 2.0


def test_minimize_mle_collapses_id():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(256, 6, generator=gen)
    W = nn.Parameter(torch.eye(6) + 0.01 * torch.randn(6, 6, generator=gen))
    loss_fn = IntrinsicDimensionLoss(method="mle", mode="minimize", n_neighbors=10)
    opt = torch.optim.Adam([W], lr=0.05)
    before = float(mle_id((Z @ W).detach(), n_neighbors=10))
    for _ in range(60):
        opt.zero_grad()
        loss_fn(Z @ W).backward()
        opt.step()
    after = float(mle_id((Z @ W).detach(), n_neighbors=10))
    assert after < before - 1.0


def test_target_mode_converges_to_target():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(512, 8, generator=gen)
    W = nn.Parameter(torch.eye(8) + 0.01 * torch.randn(8, 8, generator=gen))
    loss_fn = IntrinsicDimensionLoss(method="pr", mode="target", target=3.0)
    opt = torch.optim.Adam([W], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss_fn(Z @ W).backward()
        opt.step()
    after = float(pr_id((Z @ W).detach()))
    assert after == pytest.approx(3.0, abs=0.3)


def test_gradients_reach_module_parameters():
    gen = torch.Generator().manual_seed(0)
    encoder = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 16))
    X = torch.randn(300, 12, generator=gen)
    loss = IntrinsicDimensionLoss(method="twonn", mode="maximize")(encoder(X))
    loss.backward()
    grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.cuda
def test_loss_on_cuda():
    X = torch.randn(512, 16).cuda().requires_grad_(True)
    loss = IntrinsicDimensionLoss(method="mle", mode="maximize", n_neighbors=10)(X)
    loss.backward()
    assert X.grad is not None
    assert X.grad.device.type == "cuda"
    assert torch.isfinite(X.grad).all()
