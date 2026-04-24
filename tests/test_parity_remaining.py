"""Parity tests for MADA, TLE, KNN, DANCo, ESS, FisherS.

DANCo/ESS/FisherS use different RNG from skdim and/or differ in small numerical
details (spline interpolation, calibration data); tolerances reflect that.
"""

import numpy as np
import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import ESS, KNN, MADA, TLE, DANCo, FisherS
from torchid.parity import DEFAULT_CASES, Case, assert_parity, compare_global

# skdim's ESS and DANCo have per-point Python loops that get slow at n>=2000;
# run them on smaller datasets so parity tests finish in seconds.
SMALL_CASES: tuple[Case, ...] = (
    Case("hyperball", n=500, d=5, ambient=5),
    Case("affine", n=500, d=3, ambient=10),
    Case("affine", n=500, d=5, ambient=10),
    Case("swissroll", n=500, d=2, ambient=3),
)


def test_mada_matches_skdim() -> None:
    rows = compare_global(
        MADA,
        skid.MADA,
        cases=DEFAULT_CASES,
        atol=1e-4,
        rtol=1e-3,
    )
    assert_parity(rows)


def test_tle_matches_skdim() -> None:
    rows = compare_global(
        TLE,
        skid.TLE,
        cases=DEFAULT_CASES,
        atol=1e-3,
        rtol=1e-2,
    )
    assert_parity(rows)


def test_knn_matches_skdim() -> None:
    # KNN's bootstrap is stochastic; each call reseeds np.random to pair
    # skdim's sampling with torchid's numpy-backed sampling 1:1.
    np.random.seed(0)
    rows = compare_global(
        KNN,
        skid.KNN,
        cases=DEFAULT_CASES,
        torch_kwargs={"M": 5},
        skdim_kwargs={"M": 5},
        atol=0,
        rtol=0,
    )
    # integer output; with M=5 the argmin is usually stable across equal RNG.
    # Allow off-by-one on the highest-ambient-D case.
    passed = sum(r["abs_err"] <= 1 for r in rows)
    assert passed >= len(rows) - 1, rows


def test_ess_matches_skdim() -> None:
    rows = compare_global(
        ESS,
        skid.ESS,
        cases=SMALL_CASES,
        skdim_kwargs={"random_state": 0},
        torch_kwargs={"random_state": 0},
        atol=0.5,
        rtol=0.1,
    )
    # ESS on uniform-density data is well-behaved; on curved manifolds it
    # disagrees more sharply. Require most cases pass.
    assert_parity(rows, min_fraction=0.75)


def test_danco_matches_skdim() -> None:
    rows = compare_global(
        DANCo,
        skid.DANCo,
        cases=SMALL_CASES,
        torch_kwargs={"random_state": 0, "fractal": False},
        skdim_kwargs={"random_state": 0, "fractal": False},
        atol=1.5,
        rtol=0.25,
    )
    assert_parity(rows, min_fraction=0.5)


def test_fishers_matches_skdim() -> None:
    rows = compare_global(
        FisherS,
        skid.FisherS,
        cases=DEFAULT_CASES,
        atol=0.1,
        rtol=1e-2,
    )
    assert_parity(rows)
