"""ESS (Johnsson et al. 2015) — Expected Simplex Skewness.

Batched implementation of the default (``ver='a'``, ``d=1``) case, which covers
the vast majority of ESS usage. For d=1 ver='a' the per-neighborhood ESS has a
closed form in terms of pairwise vector angles — no actual combinatorial
enumeration needed, no sampling required.

For d > 1 or ver='b' the estimator falls back to a Monte-Carlo approximation
over p-subsets; see :func:`_ess_monte_carlo`.
"""

import math

import torch
from torch import Tensor

from torchid._primitives import gather_neighbors, knn, sample_combinations
from torchid.estimators.base import LocalEstimator


class ESS(LocalEstimator):
    _N_NEIGHBORS = 100

    def __init__(
        self,
        ver: str = "a",
        d: int = 1,
        random_state: int | None = None,
        n_neighbors: int | None = None,
    ) -> None:
        if ver not in ("a", "b"):
            raise ValueError(f"ver={ver!r}")
        self.ver = ver
        self.d = d
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def get_params(self) -> dict[str, object]:
        return {"ver": self.ver, "d": self.d}

    def fit(self, X: object, y: object = None) -> "ESS":
        Xt = self._prepare(X)
        k = self.n_neighbors or self._N_NEIGHBORS
        k = min(k, Xt.shape[0] - 1)
        _, idx = knn(Xt, k=k)
        nbrs = gather_neighbors(Xt, idx)  # (N, k, D)
        gen = torch.Generator(device=Xt.device).manual_seed(
            0 if self.random_state is None else int(self.random_state)
        )
        essvals = _batched_ess(nbrs, ver=self.ver, d=self.d, gen=gen)
        self.essval_ = essvals
        self.dimension_pw_ = _ess_to_dim(essvals, ver=self.ver, d=self.d)
        self.dimension_ = float(self.dimension_pw_.mean())
        return self


def _batched_ess(nbrs: Tensor, *, ver: str, d: int, gen: torch.Generator) -> Tensor:
    # center each neighborhood
    vec = nbrs - nbrs.mean(dim=1, keepdim=True)  # (N, k, D)
    N, k, D = vec.shape
    p = d + 1
    if p > D:
        # degenerate: fixed ESS per skdim's convention
        return torch.full((N,), 0.0 if ver == "a" else 1.0, dtype=nbrs.dtype, device=nbrs.device)

    if ver == "a" and d == 1:
        lens = vec.norm(dim=2)  # (N, k)
        dots = torch.einsum("nid,njd->nij", vec, vec)  # (N, k, k)
        i_idx, j_idx = torch.triu_indices(k, k, offset=1, device=nbrs.device)
        v_i = lens[:, i_idx]  # (N, P) P = k*(k-1)/2
        v_j = lens[:, j_idx]
        d_ij = dots[:, i_idx, j_idx]
        weight = v_i * v_j
        # vol = sqrt(|v_i|^2 |v_j|^2 - (v_i . v_j)^2)
        vol = (weight**2 - d_ij**2).clamp_min(0.0).sqrt()
        return vol.sum(dim=1) / weight.sum(dim=1).clamp_min(torch.finfo(nbrs.dtype).tiny)

    if ver == "b" and d == 1:
        dots = torch.einsum("nid,njd->nij", vec, vec)  # (N, k, k)
        lens = vec.norm(dim=2)
        i_idx, j_idx = torch.triu_indices(k, k, offset=1, device=nbrs.device)
        proj = dots[:, i_idx, j_idx].abs()
        weight = lens[:, i_idx] * lens[:, j_idx]
        return proj.sum(dim=1) / weight.sum(dim=1).clamp_min(torch.finfo(nbrs.dtype).tiny)

    return _ess_monte_carlo(vec, ver=ver, d=d, gen=gen)


def _ess_monte_carlo(
    vec: Tensor, *, ver: str, d: int, gen: torch.Generator, m: int = 5000
) -> Tensor:
    """Monte-Carlo p-subset ESS estimate for ver='a' d>1."""
    if ver != "a":
        raise NotImplementedError("ver='b' is only supported for d=1")
    N, k, D = vec.shape
    p = d + 1
    combs = sample_combinations(k, p, m, device=vec.device, generator=gen)  # (m, p)
    sel = vec[:, combs]  # (N, m, p, D)
    lens = sel.norm(dim=3)  # (N, m, p)
    weight = lens.prod(dim=2)  # (N, m)
    gram = torch.einsum("nmpd,nmqd->nmpq", sel, sel)  # (N, m, p, p)
    sign, logdet = torch.linalg.slogdet(gram)
    vol = torch.exp(0.5 * logdet) * (sign > 0).to(vec.dtype)
    return vol.sum(dim=1) / weight.sum(dim=1).clamp_min(torch.finfo(vec.dtype).tiny)


def _ess_to_dim(essvals: Tensor, *, ver: str, d: int, maxdim: int = 160) -> Tensor:
    """Invert the ESS reference curve to a dimension.

    ``ver='a'`` ref is increasing with n; ``ver='b'`` is decreasing. We bracket
    each essval, then linearly interpolate between the integer endpoints.
    """
    ref = _ess_reference(maxdim, ver=ver, d=d, device=essvals.device, dtype=essvals.dtype)
    if ver == "a":
        de_int = torch.bucketize(essvals, ref).clamp(min=1, max=maxdim - 1)
    else:
        # decreasing ref: count how many ref values exceed each essval
        de_int = (ref.unsqueeze(0) > essvals.unsqueeze(1)).sum(dim=1).clamp(min=1, max=maxdim - 1)
    lo = ref[de_int - 1]
    hi = ref[de_int]
    # ref is strictly monotone so hi - lo is non-zero; its sign flips with ver
    frac = (essvals - lo) / (hi - lo)
    return de_int.to(essvals.dtype) + frac


def _ess_reference(
    maxdim: int,
    *,
    ver: str,
    d: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    if ver == "a":
        dims = torch.arange(1, maxdim + 1, device=device, dtype=dtype)
        # factor1(n) = gamma(n/2)/gamma((n+1)/2) — via lgamma
        lg_n = torch.lgamma(dims / 2)
        lg_np1 = torch.lgamma((dims + 1) / 2)
        factor1 = torch.exp(lg_n - lg_np1)
        lg_nmd = torch.lgamma((dims - d) / 2)
        factor2 = torch.exp(lg_n - lg_nmd)
        # factor2 is 0/undefined for n <= d
        factor2 = torch.where(dims > d, factor2, torch.zeros_like(factor2))
        return factor1**d * factor2
    if ver == "b" and d == 1:
        dims = torch.arange(1, maxdim + 1, device=device, dtype=dtype)
        lg_np2 = torch.lgamma((dims + 2) / 2)
        lg_np1 = torch.lgamma((dims + 1) / 2)
        return torch.exp(lg_np1 - lg_np2) * 2 / math.sqrt(math.pi) / dims
    raise NotImplementedError("ver='b' requires d=1")
