"""Harness for checking torchid parity against scikit-dimension.

Importing this module requires the ``validation`` dep group (it imports
``skdim``). Parity tests in ``tests/test_parity_*.py`` consume these helpers.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

try:
    import skdim  # type: ignore
    import skdim.id as skid  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchid.parity requires the 'validation' dep group: uv sync --group validation"
    ) from e

from torchid import datasets


@dataclass
class Case:
    """A single parity test configuration."""

    name: str
    n: int = 2000
    d: int = 5
    ambient: int = 20
    noise_std: float = 0.01
    seed: int = 0

    def build(self) -> np.ndarray:
        gen = torch.Generator().manual_seed(self.seed)
        match self.name:
            case "hyperball":
                X = datasets.hyperball(self.n, self.d, generator=gen)
            case "hypersphere":
                X = datasets.hypersphere(self.n, self.d, generator=gen)
            case "affine":
                X = datasets.affine_subspace(
                    self.n, self.d, self.ambient, noise_std=self.noise_std, generator=gen
                )
            case "swissroll":
                X = datasets.swiss_roll(self.n, generator=gen)
            case _:
                raise ValueError(f"unknown case {self.name}")
        return X.numpy().astype(np.float64)


DEFAULT_CASES: tuple[Case, ...] = (
    Case("hyperball", d=5, ambient=5),
    Case("hyperball", d=10, ambient=10),
    Case("affine", d=3, ambient=15),
    Case("affine", d=8, ambient=30),
    Case("swissroll", d=2, ambient=3),
)


def compare_global(
    torch_cls,
    skdim_cls,
    *,
    cases: tuple[Case, ...] = DEFAULT_CASES,
    torch_kwargs: dict | None = None,
    skdim_kwargs: dict | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> list[dict]:
    """Run ``torch_cls`` and ``skdim_cls`` across cases. Returns a list of rows.

    Each row: ``{case, torch_dim, skdim_dim, abs_err, rel_err, pass}``. Caller
    decides how strictly to assert — some estimators (ESS, DANCo) tolerate
    looser bounds than those defaults.
    """
    torch_kwargs = torch_kwargs or {}
    skdim_kwargs = skdim_kwargs or {}
    rows: list[dict] = []
    for case in cases:
        X = case.build()
        td = float(torch_cls(**torch_kwargs).fit(torch.from_numpy(X).float()).dimension_)
        sd = float(skdim_cls(**skdim_kwargs).fit(X).dimension_)
        ae = abs(td - sd)
        re = ae / max(abs(sd), 1e-12)
        rows.append(
            {
                "case": f"{case.name}(d={case.d},n={case.n},D={case.ambient})",
                "torch_dim": td,
                "skdim_dim": sd,
                "abs_err": ae,
                "rel_err": re,
                "pass": ae <= atol or re <= rtol,
            }
        )
    return rows


def assert_parity(rows: list[dict], *, min_fraction: float = 1.0) -> None:
    """Assert that at least ``min_fraction`` of rows pass. Surfaces a readable diff."""
    passed = sum(r["pass"] for r in rows)
    frac = passed / len(rows)
    if frac < min_fraction:
        lines = [f"  {r['case']}: torch={r['torch_dim']:.6g} skdim={r['skdim_dim']:.6g} "
                 f"abs={r['abs_err']:.3g} rel={r['rel_err']:.3g} pass={r['pass']}"
                 for r in rows]
        raise AssertionError(
            f"parity failed: {passed}/{len(rows)} cases passed "
            f"(needed {min_fraction:.0%}):\n" + "\n".join(lines)
        )


__all__ = ["Case", "DEFAULT_CASES", "assert_parity", "compare_global"]
