"""Parity: torchid.CorrInt vs skdim.id.CorrInt."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import CorrInt
from torchid.parity import DEFAULT_CASES, assert_parity, compare_global


def test_corrint_matches_skdim() -> None:
    rows = compare_global(
        CorrInt,
        skid.CorrInt,
        cases=DEFAULT_CASES,
        atol=1e-4,
        rtol=5e-3,
    )
    assert_parity(rows)
