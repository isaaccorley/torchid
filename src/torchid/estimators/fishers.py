"""FisherS (Albergante et al. 2019) — Fisher separability intrinsic dimension.

The heavy inner loop of skdim's implementation is a pairwise-dot-product scan
across alpha thresholds. We replace that with a single batched matmul plus
threshold-counting in torch. Lambert-W inversion uses numpy (~20 scalar alpha
values — not worth porting).
"""

import math

import numpy as np
import torch
from torch import Tensor

from torchid.estimators.base import GlobalEstimator


class FisherS(GlobalEstimator):
    """Fisher-separability intrinsic dimension estimator."""

    def __init__(
        self,
        conditional_number: float = 10.0,
        project_on_sphere: bool = True,
        alphas: np.ndarray | None = None,
        limit_maxdim: bool = False,
    ) -> None:
        self.conditional_number = conditional_number
        self.project_on_sphere = project_on_sphere
        self.alphas = alphas
        self.limit_maxdim = limit_maxdim

    def get_params(self) -> dict[str, object]:
        return {
            "conditional_number": self.conditional_number,
            "project_on_sphere": self.project_on_sphere,
            "limit_maxdim": self.limit_maxdim,
        }

    def _fit(self, X: Tensor) -> Tensor:
        alphas_np = (
            np.arange(0.6, 1.0, 0.02) if self.alphas is None else np.asarray(self.alphas).flatten()
        )
        self._alphas = alphas_np[None]  # keep skdim's 2-D convention
        Xp = _preprocess(
            X,
            conditional_number=self.conditional_number,
            project_on_sphere=self.project_on_sphere,
        )
        self.Xp_ = Xp
        p_alpha = _check_separability(Xp, alphas_np)  # (A, N)
        self.p_alpha_ = p_alpha
        py_mean = p_alpha.mean(dim=1).cpu().numpy()
        n_alpha, n_single = _invert_dim(py_mean, alphas_np)
        self.n_alpha_ = torch.as_tensor(n_alpha, dtype=X.dtype, device=X.device)
        self.alphas_ = torch.as_tensor(alphas_np, dtype=X.dtype, device=X.device)
        self.separable_fraction_ = (p_alpha == 0).to(X.dtype).mean(dim=1)
        if self.limit_maxdim:
            n_single = min(n_single, X.shape[1])
        return torch.tensor(float(n_single), dtype=X.dtype, device=X.device)


def _preprocess(X: Tensor, *, conditional_number: float, project_on_sphere: bool) -> Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    # PCA via SVD; keep components with explained_variance > max / conditional_number
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    s2 = (S * S) / max(Xc.shape[0] - 1, 1)
    ratio = s2 / s2[0].clamp_min(torch.finfo(X.dtype).tiny)
    keep = ratio > (1.0 / conditional_number)
    V_keep = Vh[keep].T
    Xr = Xc @ V_keep
    # Whiten: divide by std (ddof=1) of each retained PC
    std = Xr.std(dim=0, unbiased=True).clamp_min(torch.finfo(X.dtype).tiny)
    Xr = Xr / std
    if project_on_sphere:
        norms = Xr.norm(dim=1, keepdim=True).clamp_min(torch.finfo(X.dtype).tiny)
        Xr = Xr / norms
    return Xr


def _check_separability(data: Tensor, alphas: np.ndarray) -> Tensor:
    """Returns ``(A, N)`` tensor where entry (a, i) is the fraction of other
    points ``j`` with ``<x_i, x_j> / <x_i, x_i>  >  alphas[a]``.

    Matches skdim's reverse-cumulative histogram semantics over alpha.
    """
    n = data.shape[0]
    # xy[i, j] = <x_i, x_j>; leng[i] = <x_i, x_i>. Keeping a full (N, N)
    # matrix is fine up to ~30k but a (N, N, A) broadcast is not, so we loop
    # over alphas and reduce in place.
    xy = data @ data.T
    leng = xy.diag().unsqueeze(1)  # (N, 1)
    xy = xy - torch.diag(torch.diag(xy))  # zero diagonal
    ratios = xy / leng.clamp_min(torch.finfo(data.dtype).tiny)  # (N, N)
    A = len(alphas)
    py = torch.empty((A, n), dtype=data.dtype, device=data.device)
    for a_idx, a in enumerate(alphas):
        py[a_idx] = (ratios > float(a)).sum(dim=1).to(data.dtype) / n
    return py


def _invert_dim(py_mean: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy.special import lambertw  # only used on ~20 scalars

    n = np.full(alphas.shape, np.nan)
    for i, a in enumerate(alphas):
        p = py_mean[i]
        if p == 0:
            continue
        a2 = a * a
        w = math.log(1 - a2)
        n[i] = np.real(lambertw(-(w / (2 * math.pi * p * p * a2 * (1 - a2))))) / (-w)
    n[np.isinf(n)] = np.nan
    inds = np.where(~np.isnan(n))[0]
    if len(inds) == 0:
        return n, float("nan")
    alpha_max = alphas[inds].max()
    alpha_ref = alpha_max * 0.9
    k = np.argmin(np.abs(alphas[inds] - alpha_ref))
    n_single = float(n[inds[k]])
    return n, n_single
