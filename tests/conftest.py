import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(
    params=[
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ]
)
def accelerator(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    return torch.device(request.param)
