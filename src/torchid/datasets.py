"""Synthetic manifolds for testing / validation.

A focused subset of the scikit-dimension ``BenchmarkManifolds`` that covers the
cases needed for parity testing: linear subspaces, hyperballs, swiss roll,
hyperspheres. The full 23-manifold benchmark is still available via
``skdim.datasets.BenchmarkManifolds`` when the ``validation`` dep group is
installed; this module exists so users can reproduce tests without skdim.
"""

import math

import torch
from torch import Tensor

__all__ = [
    "affine_subspace",
    "hyperball",
    "hypersphere",
    "swiss_roll",
]


def _gen(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def hyperball(
    n: int,
    d: int,
    *,
    radius: float = 1.0,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``n`` uniform samples from the ``d``-dimensional unit ball (ambient = d).

    True intrinsic dimension is ``d``.
    """
    u = _gen((n, d), generator=generator, device=device, dtype=dtype)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = torch.rand(n, 1, generator=generator, device=device, dtype=dtype) ** (1.0 / d)
    return radius * r * u


def hypersphere(
    n: int,
    d: int,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``n`` uniform samples from the ``d``-sphere embedded in R^{d+1}.

    True intrinsic dimension is ``d``.
    """
    u = _gen((n, d + 1), generator=generator, device=device, dtype=dtype)
    return u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)


def affine_subspace(
    n: int,
    d: int,
    ambient: int,
    *,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``n`` points on a random ``d``-dim affine subspace of R^ambient.

    True ID is ``d``. With ``noise_std > 0`` the output is perturbed with
    isotropic Gaussian noise in the full ambient space — that breaks the
    degenerate zero-eigenvalue floor which otherwise makes some ratio-based
    estimators (lPCA/maxgap) numerically unstable.
    """
    if d > ambient:
        raise ValueError("d must be <= ambient")
    coords = _gen((n, d), generator=generator, device=device, dtype=dtype)
    basis = _gen((d, ambient), generator=generator, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(basis.T)
    offset = _gen((1, ambient), generator=generator, device=device, dtype=dtype)
    X = coords @ Q.T + offset
    if noise_std > 0:
        X = X + noise_std * _gen((n, ambient), generator=generator, device=device, dtype=dtype)
    return X


def swiss_roll(
    n: int,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Classic 2-D swiss roll embedded in R^3 (true ID = 2)."""
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=generator, device=device, dtype=dtype))
    h = 21.0 * torch.rand(n, generator=generator, device=device, dtype=dtype)
    x = t * torch.cos(t)
    y = h
    z = t * torch.sin(t)
    return torch.stack([x, y, z], dim=1)
