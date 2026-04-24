"""Local PCA-based intrinsic dimension estimator.

Ports :class:`skdim.id.lPCA`. Given a neighborhood (or a full dataset treated as
one), eigen-decompose its sample covariance and read off the intrinsic dimension
via one of several classical heuristics (FO, Fan, maxgap, ratio, Kaiser,
broken-stick, participation ratio).
"""

import torch
from torch import Tensor

from torchid.estimators.base import GlobalEstimator

_VERSIONS = ("FO", "Fan", "maxgap", "ratio", "participation_ratio", "Kaiser", "broken_stick")


class lPCA(GlobalEstimator):
    """PCA-eigenvalue-based intrinsic dimension estimator.

    Matches the API of :class:`skdim.id.lPCA`. Supply a single *local* patch (or
    the whole dataset if you want a global estimate) as ``X``. The chosen
    ``ver`` heuristic determines how eigenvalues are thresholded into a
    dimension.
    """

    def __init__(
        self,
        ver: str = "FO",
        alphaRatio: float = 0.05,
        alphaFO: float = 0.05,
        alphaFan: float = 10.0,
        betaFan: float = 0.8,
        PFan: float = 0.95,
        fit_explained_variance: bool = False,
    ) -> None:
        if ver not in _VERSIONS:
            raise ValueError(f"ver must be one of {_VERSIONS}, got {ver!r}")
        self.ver = ver
        self.alphaRatio = alphaRatio
        self.alphaFO = alphaFO
        self.alphaFan = alphaFan
        self.betaFan = betaFan
        self.PFan = PFan
        self.fit_explained_variance = fit_explained_variance

    def get_params(self) -> dict[str, object]:
        return {
            "ver": self.ver,
            "alphaRatio": self.alphaRatio,
            "alphaFO": self.alphaFO,
            "alphaFan": self.alphaFan,
            "betaFan": self.betaFan,
            "PFan": self.PFan,
            "fit_explained_variance": self.fit_explained_variance,
        }

    def fit(self, X: object, y: object = None) -> "lPCA":  # noqa: ARG002
        if self.fit_explained_variance:
            ev = torch.as_tensor(X).to(torch.float32).flatten()
        else:
            ev = _explained_variance(self._prepare(X))
        self.explained_var_ = ev
        self.dimension_ = float(self._pick(ev))
        self.gap_ = _gaps(ev).detach().cpu().numpy()
        return self

    def _pick(self, ev: Tensor) -> Tensor:
        match self.ver:
            case "FO":
                return (ev > self.alphaFO * ev[0]).sum().to(ev.dtype)
            case "maxgap":
                g = _gaps(ev)
                return (torch.argmax(torch.nan_to_num(g, nan=-float("inf"))) + 1).to(ev.dtype)
            case "ratio":
                cum = torch.cumsum(ev, dim=0)
                norm = cum / cum.max()
                return ((norm < self.alphaRatio).sum() + 1).to(ev.dtype)
            case "participation_ratio":
                return ev.sum() ** 2 / (ev * ev).sum()
            case "Kaiser":
                return (ev > ev.mean()).sum().to(ev.dtype)
            case "broken_stick":
                return _broken_stick(ev).to(ev.dtype)
            case "Fan":
                return _fan(ev, alphaFan=self.alphaFan, betaFan=self.betaFan, PFan=self.PFan)
        raise AssertionError  # unreachable


def _explained_variance(X: Tensor) -> Tensor:
    """sklearn-PCA's ``explained_variance_``: singular-values squared over (n-1)."""
    n = X.shape[0]
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    return (s * s) / max(n - 1, 1)


def _gaps(ev: Tensor) -> Tensor:
    # ev assumed sorted descending (svdvals is descending)
    return ev[:-1] / ev[1:].clamp_min(torch.finfo(ev.dtype).tiny)


def _broken_stick(ev: Tensor) -> Tensor:
    dim = ev.shape[0]
    # bs[i] = (1/dim) * sum_{j=i..dim-1} 1/(j+1)
    inv = 1.0 / torch.arange(1, dim + 1, device=ev.device, dtype=ev.dtype)
    bs = torch.flip(torch.cumsum(torch.flip(inv, dims=(0,)), dim=0), dims=(0,)) / dim
    norm = ev / ev.sum().clamp_min(torch.finfo(ev.dtype).tiny)
    mask = bs > norm
    if not mask.any():
        return torch.tensor(0.0, device=ev.device, dtype=ev.dtype)
    return (torch.argmax(mask.to(torch.int8)) + 1).to(ev.dtype)


def _fan(ev: Tensor, *, alphaFan: float, betaFan: float, PFan: float) -> Tensor:
    cum = torch.cumsum(ev, dim=0)
    total = cum[-1]
    r = int(torch.argmax((cum / total > PFan).to(torch.int8)).item())
    sigma = ev[r:].mean()
    evd = ev - sigma
    g = evd[:-1] / evd[1:].clamp_min(torch.finfo(ev.dtype).tiny)
    gap_hits = torch.nonzero(g > alphaFan, as_tuple=False).flatten()
    cum2 = torch.cumsum(evd, dim=0) / evd.sum().clamp_min(torch.finfo(ev.dtype).tiny)
    beta_hits = torch.nonzero(cum2 > betaFan, as_tuple=False).flatten()
    hits = torch.cat([gap_hits, beta_hits])
    if hits.numel() == 0:
        return torch.tensor(float(ev.shape[0]), device=ev.device, dtype=ev.dtype)
    return (hits.min() + 1).to(ev.dtype)
