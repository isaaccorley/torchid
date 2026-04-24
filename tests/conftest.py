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


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.cuda`` tests when no GPU is visible.

    GitHub Actions runners don't have CUDA, so CUDA-tagged tests need to be
    quietly skipped there. Developers with a local GPU still run them.
    """
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
