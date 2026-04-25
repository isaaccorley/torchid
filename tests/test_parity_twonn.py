"""Parity: torchid.TwoNN vs skdim.id.TwoNN."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import TwoNN

from ._parity import DEFAULT_CASES, assert_parity, compare_global


def test_twonn_matches_skdim() -> None:
    rows = compare_global(
        TwoNN,
        skid.TwoNN,
        cases=DEFAULT_CASES,
        atol=1e-4,
        rtol=5e-3,
    )
    assert_parity(rows)
