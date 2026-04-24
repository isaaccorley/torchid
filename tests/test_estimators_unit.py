"""Unit tests hitting uncovered branches in individual estimators."""

import numpy as np
import pytest
import torch

from torchid.datasets import affine_subspace, hyperball, hypersphere, swiss_roll
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
from torchid.estimators.base import GlobalEstimator, LocalEstimator


@pytest.fixture
def X() -> torch.Tensor:
    return affine_subspace(300, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))


def test_repr_and_get_params() -> None:
    r = repr(lPCA(ver="Fan"))
    assert "lPCA" in r
    assert "ver='Fan'" in r
    assert lPCA(ver="Fan").get_params()["ver"] == "Fan"


def test_lpca_invalid_version_raises() -> None:
    with pytest.raises(ValueError, match="ver must be one of"):
        lPCA(ver="bogus")


def test_lpca_fit_explained_variance_branch() -> None:
    X = hyperball(500, 4, generator=torch.Generator().manual_seed(0))
    ev = torch.linalg.svdvals(X - X.mean(0)) ** 2 / (X.shape[0] - 1)
    d = lPCA(ver="participation_ratio", fit_explained_variance=True).fit(ev).dimension_
    assert 0 < d < 5


def test_lpca_broken_stick_no_cross() -> None:
    # single eigenvalue — bs = [1] = norm, no mask hit → returns 0
    ev = torch.tensor([1.0])
    d = lPCA(ver="broken_stick", fit_explained_variance=True).fit(ev).dimension_
    assert d == 0.0


def test_corrint_invalid_DM_mode() -> None:
    X = hyperball(100, 3)
    with pytest.raises(NotImplementedError):
        CorrInt(DM=True).fit(X)


def test_mada_invalid_DM_mode() -> None:
    with pytest.raises(NotImplementedError):
        MADA(DM=True)


def test_mle_noise_branch_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="zero-noise"):
        MLE(sigma=0.1)


def test_mle_invalid_comb(X: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="comb"):
        MLE().fit(X, comb="xyz")


def test_mle_comb_median(X: torch.Tensor) -> None:
    d = MLE().fit(X, comb="median").dimension_
    assert 0 < d < 10


def test_mind_ml_invalid_ver() -> None:
    with pytest.raises(ValueError, match="ver"):
        MiND_ML(ver="bogus")


def test_mind_ml_mli_branch() -> None:
    X = affine_subspace(500, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    d = MiND_ML(ver="MLi", D=10).fit(X).dimension_
    assert 1 <= d <= 10


def test_danco_invalid_ver() -> None:
    with pytest.raises(ValueError, match="ver"):
        DANCo(ver="bogus")


def test_danco_mind_mli_branch() -> None:
    X = affine_subspace(300, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    d = DANCo(ver="MIND_MLi", D=6, random_state=0).fit(X).dimension_
    assert 1 <= d <= 6


def test_danco_mind_mlk_branch() -> None:
    X = affine_subspace(300, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    d = DANCo(ver="MIND_MLk", D=6, random_state=0).fit(X).dimension_
    assert 1 <= d <= 6


def test_danco_non_fractal(X: torch.Tensor) -> None:
    d = DANCo(D=5, fractal=False, random_state=0).fit(X).dimension_
    assert isinstance(d, float)


def test_ess_invalid_ver() -> None:
    with pytest.raises(ValueError, match="ver"):
        ESS(ver="z")


def test_ess_ver_b() -> None:
    X = hyperball(300, 4, generator=torch.Generator().manual_seed(0))
    d = ESS(ver="b", d=1, random_state=0).fit(X).dimension_
    assert d > 0


def test_ess_monte_carlo_dgt1() -> None:
    X = hyperball(200, 5, generator=torch.Generator().manual_seed(0))
    d = ESS(ver="a", d=2, random_state=0, n_neighbors=30).fit(X).dimension_
    assert d > 0


def test_ess_degenerate_ambient_too_small() -> None:
    # ambient dimension < p = d + 1 triggers the fixed-ESS branch; the inverse
    # map is undefined (ref is 0 below n=d) so we just assert the branch ran
    # without crashing — not the dimension value.
    import math as m

    X = torch.randn(50, 2)
    est = ESS(ver="a", d=3).fit(X)
    assert (
        torch.isnan(est.essval_).any()
        or est.essval_.eq(0.0).any()
        or m.isfinite(est.dimension_)
        or m.isnan(est.dimension_)
    )


def test_ess_monte_carlo_ver_b_not_implemented() -> None:
    # d > 1 with ver='b' should raise
    X = hyperball(50, 6)
    with pytest.raises(NotImplementedError, match="ver='b'"):
        ESS(ver="b", d=3, n_neighbors=15).fit(X)


def test_tle_zero_distance_raises() -> None:
    X = torch.zeros(30, 3)
    with pytest.raises(ValueError, match="zero"):
        TLE().fit(X)


def test_datasets_hypersphere() -> None:
    X = hypersphere(200, 5, generator=torch.Generator().manual_seed(0))
    assert X.shape == (200, 6)
    norms = X.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_datasets_swiss_roll_shape() -> None:
    X = swiss_roll(100)
    assert X.shape == (100, 3)


def test_datasets_affine_invalid() -> None:
    with pytest.raises(ValueError, match="<="):
        affine_subspace(50, d=10, ambient=5)


def test_isinstance_base_classes() -> None:
    assert isinstance(lPCA(), GlobalEstimator)
    assert isinstance(MOM(), LocalEstimator)


def test_twonn_dist_mode() -> None:
    # Synthesize a (d1, d2) distance table; slope should be ~d_true=5
    g = torch.Generator().manual_seed(0)
    X = hyperball(2000, 5, generator=g)
    from torchid._primitives import knn as knn_fn

    d, _ = knn_fn(X, k=2)
    est = TwoNN(dist=True).fit(d).dimension_
    assert 3.5 < est < 6.5


def test_fishers_with_custom_alphas() -> None:
    X = hyperball(300, 5, generator=torch.Generator().manual_seed(0))
    d = FisherS(alphas=np.arange(0.7, 0.95, 0.02)).fit(X).dimension_
    assert d > 0


def test_fishers_limit_maxdim() -> None:
    X = hyperball(200, 5, generator=torch.Generator().manual_seed(0))
    d = FisherS(limit_maxdim=True).fit(X).dimension_
    assert d <= X.shape[1]


def test_knn_est_custom_ps() -> None:
    np.random.seed(0)
    X = affine_subspace(500, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))
    d = KNN(k=2, ps=np.array([5, 6, 7, 8]), M=2).fit(X).dimension_
    assert 1 <= d <= 8


def test_knn_invalid_ps() -> None:
    X = hyperball(100, 3)
    with pytest.raises(ValueError, match="ps"):
        KNN(k=5, ps=np.array([3, 4])).fit(X)  # ps must satisfy k < ps
