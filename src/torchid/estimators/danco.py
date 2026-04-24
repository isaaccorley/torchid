"""DANCo (Ceruti et al. 2012) — Dimensionality from Angle and Norm Concentration.

Implementation notes
--------------------
- The per-point angle computation is fully batched: a single ``(N, k, k)``
  angle tensor is constructed in one pass rather than the per-point Python
  loop in skdim's ``_angles``.
- Von-Mises MLE (``nu``, ``tau``) is computed in closed form per point,
  then averaged.
- MIND_MLk is reused from :mod:`torchid.estimators.mind_ml`.
- Calibration data is generated on-device from torch hyperballs. Because
  torch's RNG differs from numpy's, exact parity with skdim is not expected;
  tolerances in the parity tests reflect this.
- The ``fractal`` mode minimizes a 1-D cubic spline of the KL curve on a
  dense grid rather than via scipy's ``interp1d + minimize``. Same end result
  up to grid spacing.
"""

import math

import torch
from torch import Tensor

from torchid._primitives import gather_neighbors, knn
from torchid.datasets import hyperball
from torchid.estimators.base import GlobalEstimator
from torchid.estimators.mind_ml import _lld


class DANCo(GlobalEstimator):
    """DANCo intrinsic dimension estimator."""

    def __init__(
        self,
        k: int = 10,
        D: int | None = None,
        calibration_data: dict | None = None,
        ver: str = "DANCo",
        fractal: bool = True,
        random_state: int | None = None,
    ) -> None:
        if ver not in ("DANCo", "MIND_MLi", "MIND_MLk"):
            raise ValueError(f"ver={ver!r}")
        self.k = k
        self.D = D
        self.calibration_data = calibration_data
        self.ver = ver
        self.fractal = fractal
        self.random_state = random_state

    def get_params(self) -> dict[str, object]:
        return {"k": self.k, "D": self.D, "ver": self.ver, "fractal": self.fractal}

    def _fit(self, X: Tensor) -> Tensor:
        n = X.shape[0]
        k = min(self.k, n - 2)
        D = self.D if self.D is not None else X.shape[1]
        gen = torch.Generator(device=X.device).manual_seed(
            0 if self.random_state is None else int(self.random_state)
        )
        obs = _danco_stats(X, k=k, D=D, gen=gen)

        if self.ver == "MIND_MLi":
            return torch.tensor(float(_mind_mli(X, k, D)), dtype=X.dtype, device=X.device)
        if self.ver == "MIND_MLk":
            return obs["dhat"]

        # full DANCo: build calibration across dims 1..D and pick argmin KL
        cal = self.calibration_data
        if cal is None:
            cal = {"k": k, "N": n, "maxdim": 0, "calibration_data": []}
        while cal["maxdim"] < D:
            nd = cal["maxdim"] + 1
            ball = hyperball(n, nd, generator=gen, device=X.device, dtype=X.dtype)
            if ball.shape[1] < 2:
                # pad to 2D for knn to work
                ball = torch.cat([ball, torch.zeros_like(ball)], dim=1)
            cal["calibration_data"].append(_danco_stats(ball, k=k, D=max(nd * 2 + 5, D), gen=gen))
            cal["maxdim"] = nd
        self.calibration_data_ = cal

        kl = torch.stack([_kl(obs, c, k) for c in cal["calibration_data"]])
        de_int = int(torch.argmin(kl).item()) + 1

        if not self.fractal:
            self.kl_divergence_ = float(kl[de_int - 1])
            return torch.tensor(float(de_int), dtype=X.dtype, device=X.device)

        # fractal: dense 1-D interpolation over kl
        grid = torch.linspace(1.0, float(D), steps=1000, device=X.device, dtype=X.dtype)
        dom = torch.arange(1, D + 1, dtype=X.dtype, device=X.device)
        lo_idx = torch.clamp(torch.bucketize(grid, dom) - 1, 0, D - 2)
        w = (grid - dom[lo_idx]) / (dom[lo_idx + 1] - dom[lo_idx])
        kl_interp = kl[lo_idx] * (1 - w) + kl[lo_idx + 1] * w
        best = torch.argmin(kl_interp)
        self.kl_divergence_ = float(kl[de_int - 1])
        return grid[best]


def _danco_stats(X: Tensor, *, k: int, D: int, gen: torch.Generator) -> dict:
    dists, idx = knn(X, k=k + 1)
    rhos = (dists[:, 0] / dists[:, -1].clamp_min(torch.finfo(X.dtype).tiny)).clamp(
        min=torch.finfo(X.dtype).tiny, max=1 - torch.finfo(X.dtype).eps
    )
    # MIND_MLi integer
    N = rhos.shape[0]
    sum_log_rho = torch.log(rhos).sum()
    d_grid = torch.arange(1, D + 1, device=X.device, dtype=X.dtype)
    ll = _lld(d_grid, rhos, N, k + 1, sum_log_rho)
    mli = int(torch.argmax(ll).item()) + 1
    # MIND_MLk continuous refinement
    lo, hi = max(mli - 1, 1e-6), min(mli + 1, D)
    grid = torch.linspace(lo, hi, steps=int((hi - lo) / 0.001) + 1, device=X.device, dtype=X.dtype)
    ll_c = _lld(grid, rhos, N, k + 1, sum_log_rho)
    dhat = grid[torch.argmax(ll_c)]
    # angles across the k neighbors (drop the first idx column per skdim convention)
    nbrs = gather_neighbors(X, idx[:, :k])
    nu, tau = _von_mises_mle(X, nbrs)
    return {"dhat": dhat, "mu_nu": nu.mean(), "mu_tau": tau.mean()}


def _von_mises_mle(X: Tensor, nbrs: Tensor) -> tuple[Tensor, Tensor]:
    # for each point: vectors = nbrs - X[i]; angles between all pairs of vectors.
    vec = nbrs - X.unsqueeze(1)  # (N, k, D)
    vec_n = vec / vec.norm(dim=2, keepdim=True).clamp_min(torch.finfo(X.dtype).tiny)
    # cos between all pairs (k, k)
    cos_mat = torch.einsum("nkd,njd->nkj", vec_n, vec_n).clamp(min=-1.0, max=1.0)
    # take upper-triangular (i < j) pairs
    N, k, _ = cos_mat.shape
    i_idx, j_idx = torch.triu_indices(k, k, offset=1, device=X.device)
    cos = cos_mat[:, i_idx, j_idx]
    th = torch.arccos(cos)
    sinth = torch.sin(th)
    costh = torch.cos(th)
    nu = torch.atan2(sinth.sum(dim=1), costh.sum(dim=1))
    eta = torch.sqrt(costh.mean(dim=1) ** 2 + sinth.mean(dim=1) ** 2).clamp(
        min=torch.finfo(X.dtype).tiny, max=1 - torch.finfo(X.dtype).eps
    )
    tau = _Ainv(eta)
    return nu, tau


def _Ainv(eta: Tensor) -> Tensor:
    e2 = eta * eta
    e3 = e2 * eta
    e5 = e3 * e2
    branch_a = 2 * eta + e3 + 5 * e5 / 6
    branch_b = -0.4 + 1.39 * eta + 0.43 / (1 - eta).clamp_min(torch.finfo(eta.dtype).tiny)
    branch_c = 1 / (e3 - 4 * e2 + 3 * eta).clamp_min(torch.finfo(eta.dtype).tiny)
    return torch.where(eta < 0.53, branch_a, torch.where(eta < 0.85, branch_b, branch_c))


def _mind_mli(X: Tensor, k: int, D: int) -> int:
    dists, _ = knn(X, k=k + 1)
    rhos = (dists[:, 0] / dists[:, -1].clamp_min(torch.finfo(X.dtype).tiny)).clamp(
        min=torch.finfo(X.dtype).tiny, max=1 - torch.finfo(X.dtype).eps
    )
    N = rhos.shape[0]
    sum_log_rho = torch.log(rhos).sum()
    d_grid = torch.arange(1, D + 1, device=X.device, dtype=X.dtype)
    ll = _lld(d_grid, rhos, N, k + 1, sum_log_rho)
    return int(torch.argmax(ll).item()) + 1


def _kl(obs: dict, cal: dict, k: int) -> Tensor:
    return _kl_d(obs["dhat"], cal["dhat"], k) + _kl_nutau(
        obs["mu_nu"], cal["mu_nu"], obs["mu_tau"], cal["mu_tau"]
    )


def _kl_d(dhat: Tensor, dcal: Tensor, k: int) -> Tensor:
    dhat = dhat.clamp_min(torch.finfo(dhat.dtype).tiny)
    quo = dcal / dhat
    H_k = sum(1.0 / j for j in range(1, k + 1))
    # sum_{i=0..k} (-1)^i * C(k,i) * digamma(1 + i/quo)
    i = torch.arange(k + 1, device=dhat.device, dtype=dhat.dtype)
    log_binom = (
        torch.lgamma(torch.tensor(float(k + 1), device=dhat.device, dtype=dhat.dtype))
        - torch.lgamma(i + 1)
        - torch.lgamma(torch.tensor(float(k), device=dhat.device, dtype=dhat.dtype) - i + 1)
    )
    binom = torch.exp(log_binom)
    sign = torch.pow(torch.tensor(-1.0, device=dhat.device, dtype=dhat.dtype), i)
    a = sign * binom * torch.special.digamma(1 + i / quo)
    return H_k * quo - torch.log(quo) - (k - 1) * a.sum()


def _kl_nutau(nu1: Tensor, nu2: Tensor, tau1: Tensor, tau2: Tensor) -> Tensor:
    i0_1 = torch.special.i0(tau1).clamp_min(torch.finfo(tau1.dtype).tiny)
    i0_2 = torch.special.i0(tau2).clamp_min(torch.finfo(tau2.dtype).tiny)
    i1_1 = torch.special.i1(tau1)
    return torch.log(i0_2 / i0_1) + i1_1 / i0_1 * (tau1 - tau2 * torch.cos(nu1 - nu2))
