"""Parity: torchid.MOM vs skdim.id.MOM."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import MOM
from torchid.parity import DEFAULT_CASES, assert_parity, compare_global


def test_mom_matches_skdim() -> None:
    rows = compare_global(
        MOM, skid.MOM, cases=DEFAULT_CASES, atol=1e-4, rtol=5e-3,
    )
    assert_parity(rows)
