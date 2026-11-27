"""CPU agreement and device placement on CUDA and MPS."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from torchid.datasets import affine_subspace
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


@pytest.fixture(scope="module")
def X_cpu() -> torch.Tensor:
    return affine_subspace(500, 3, 8, noise_std=0.01, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (lPCA, {}),
        (TwoNN, {}),
        (MLE, {}),
        (CorrInt, {}),
        (MiND_ML, {}),
        (MOM, {}),
        (MADA, {}),
        (TLE, {}),
        (ESS, {"random_state": 0}),
        (DANCo, {"D": 5, "random_state": 0, "fractal": False}),
        (FisherS, {}),
    ],
)
def test_cpu_accelerator_agreement(
    cls: type, kwargs: dict, X_cpu: torch.Tensor, accelerator: torch.device
) -> None:
    X_gpu = X_cpu.to(accelerator)
    d_cpu = cls(**kwargs).fit(X_cpu).dimension_
    fitted = cls(**kwargs).fit(X_gpu)
    if hasattr(fitted, "dimension_pw_"):
        assert fitted.dimension_pw_.device == X_gpu.device
    d_gpu = fitted.dimension_
    # DANCo draws calibration samples from a device-specific RNG.
    atol = 1.0 if cls is DANCo else 1e-3
    rtol = 0.1 if cls is DANCo else 1e-2
    assert d_gpu == pytest.approx(d_cpu, abs=atol, rel=rtol)


def test_knn_cpu_accelerator_agreement(accelerator: torch.device) -> None:
    # Float32 distance arithmetic gives the wrong dimension for these two clusters.
    x = torch.randn(128, 4, generator=torch.Generator().manual_seed(0))
    x[:64, 0] += 10_000
    x[64:, 0] -= 10_000
    sizes = np.array([12, 20, 32, 44])
    np.random.seed(0)
    cpu = KNN(ps=sizes, M=3).fit(x)
    np.random.seed(0)
    notice = (
        pytest.warns(UserWarning, match="float64.*CPU")
        if accelerator.type == "mps"
        else nullcontext()
    )
    with notice:
        result = KNN(ps=sizes, M=3).fit(x.to(accelerator))
    assert result.dimension_ == cpu.dimension_
    assert result.residual_ == pytest.approx(cpu.residual_, rel=1e-7)
