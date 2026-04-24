"""CUDA device-parity tests.

Every estimator should produce the same dimension (within fp32 tolerance)
whether you feed it a CPU tensor or a CUDA tensor. Skipped automatically on
hosts without CUDA — see ``tests/conftest.py``.
"""

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


@pytest.mark.cuda
def test_cuda_available_selfcheck() -> None:
    assert torch.cuda.is_available()


@pytest.mark.cuda
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
def test_cpu_cuda_agreement(cls: type, kwargs: dict, X_cpu: torch.Tensor) -> None:
    X_gpu = X_cpu.cuda()
    d_cpu = cls(**kwargs).fit(X_cpu).dimension_
    d_gpu = cls(**kwargs).fit(X_gpu).dimension_
    # fp32 accumulation slop; DANCo also differs via separate RNG streams on
    # different devices so it gets a looser band.
    atol = 1.0 if cls is DANCo else 1e-3
    rtol = 0.1 if cls is DANCo else 1e-2
    assert abs(d_cpu - d_gpu) <= atol or abs(d_cpu - d_gpu) / max(abs(d_cpu), 1e-9) <= rtol, (
        f"{cls.__name__}: cpu={d_cpu} gpu={d_gpu}"
    )


@pytest.mark.cuda
def test_knn_cpu_cuda_agreement(X_cpu: torch.Tensor) -> None:
    np.random.seed(0)
    d_cpu = KNN(k=2, M=3).fit(X_cpu).dimension_
    np.random.seed(0)
    d_gpu = KNN(k=2, M=3).fit(X_cpu.cuda()).dimension_
    assert abs(d_cpu - d_gpu) <= 1
