"""Parity: torchid.MiND_ML vs skdim.id.MiND_ML."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import MiND_ML
from torchid.parity import DEFAULT_CASES, assert_parity, compare_global


@pytest.mark.parametrize("ver", ["MLi", "MLk"])
def test_mind_ml_matches_skdim(ver: str) -> None:
    rows = compare_global(
        MiND_ML,
        skid.MiND_ML,
        cases=DEFAULT_CASES,
        torch_kwargs={"ver": ver, "D": 15},
        skdim_kwargs={"ver": ver, "D": 15},
        atol=0.05,
        rtol=2e-2,
    )
    assert_parity(rows)
