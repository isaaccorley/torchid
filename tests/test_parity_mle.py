"""Parity: torchid.MLE vs skdim.id.MLE (zero-noise closed form)."""

import pytest

pytest.importorskip("skdim")

import skdim.id as skid

from torchid.estimators import MLE
from torchid.parity import DEFAULT_CASES, assert_parity, compare_global


def _make_skdim_mle(**kw):
    """skdim.id.MLE.__init__ mutates frame.f_locals, broken on Python 3.13. Patch it."""

    class _PatchedMLE(skid.MLE):
        def __init__(self, **kw):  # noqa: ANN003
            self.dnoise = kw.get("dnoise")
            self.sigma = kw.get("sigma", 0)
            self.n = kw.get("n")
            self.integral_approximation = kw.get("integral_approximation", "Haro")
            self.unbiased = kw.get("unbiased", False)
            self.neighborhood_based = kw.get("neighborhood_based", True)
            self.K = kw.get("K", 5)

    return _PatchedMLE(**kw)


@pytest.mark.parametrize("comb", ["mle", "mean", "median"])
@pytest.mark.parametrize("unbiased", [False, True])
def test_mle_matches_skdim(comb: str, unbiased: bool) -> None:
    class T(MLE):
        def fit(self, X, y=None):  # type: ignore[override]
            return super().fit(X, comb=comb)

    class S:
        def __init__(self, **kw):  # noqa: ANN003
            self._inner = _make_skdim_mle(**kw)

        def fit(self, X, y=None):  # noqa: ARG002
            self._inner.fit(X, comb=comb)
            self.dimension_ = self._inner.dimension_
            return self

    rows = compare_global(
        T, S,
        cases=DEFAULT_CASES,
        torch_kwargs={"unbiased": unbiased},
        skdim_kwargs={"unbiased": unbiased},
        atol=1e-4, rtol=5e-3,
    )
    assert_parity(rows)
