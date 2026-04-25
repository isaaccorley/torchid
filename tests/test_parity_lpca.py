"""Parity: torchid.estimators.lPCA vs skdim.id.lPCA across all heuristics."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import lPCA

from ._parity import DEFAULT_CASES, assert_parity, compare_global


@pytest.mark.parametrize(
    "ver",
    ["FO", "Fan", "maxgap", "ratio", "participation_ratio", "Kaiser", "broken_stick"],
)
def test_lpca_matches_skdim(ver: str) -> None:
    rows = compare_global(
        lPCA,
        skid.lPCA,
        cases=DEFAULT_CASES,
        torch_kwargs={"ver": ver},
        skdim_kwargs={"ver": ver, "verbose": False},
        atol=1e-4,
        rtol=1e-3,
    )
    assert_parity(rows)
