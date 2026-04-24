"""Unified benchmark runner: time every torchid estimator vs skdim.

Usage::

    uv run --group validation python -m benchmarks.bench --out BENCHMARKS.md

Runs each estimator across a grid of (n, D_ambient), timing sklearn,
torch-cpu, and (if available) torch-cuda. Peak memory is measured via
``torch.cuda.max_memory_allocated`` when CUDA is available. Output is a
Markdown table suitable for pasting into ``BENCHMARKS.md``.
"""

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
import skdim.id as skid
import torch

from torchid import datasets, estimators


@dataclass
class Row:
    estimator: str
    n: int
    d_ambient: int
    skdim_ms: float
    torch_cpu_ms: float
    torch_cuda_ms: float | None
    cuda_peak_mb: float | None


ESTIMATORS: dict[str, tuple[type, type, dict, int]] = {
    # name → (torch_cls, skdim_cls, kwargs, max_n_for_skdim)
    "lPCA": (estimators.lPCA, skid.lPCA, {}, 100_000),
    "TwoNN": (estimators.TwoNN, skid.TwoNN, {}, 100_000),
    "MLE": (estimators.MLE, skid.MLE, {}, 50_000),
    "CorrInt": (estimators.CorrInt, skid.CorrInt, {}, 20_000),
    "MiND_ML": (estimators.MiND_ML, skid.MiND_ML, {}, 50_000),
    "MOM": (estimators.MOM, skid.MOM, {}, 50_000),
    "MADA": (estimators.MADA, skid.MADA, {}, 20_000),
    "KNN": (estimators.KNN, skid.KNN, {}, 10_000),
    "TLE": (estimators.TLE, skid.TLE, {}, 5_000),
    "DANCo": (estimators.DANCo, skid.DANCo, {"fractal": False, "random_state": 0}, 1_000),
    "ESS": (estimators.ESS, skid.ESS, {"random_state": 0}, 1_000),
    "FisherS": (estimators.FisherS, skid.FisherS, {}, 10_000),
}


def _time_once(fn, repeat: int = 3) -> float:
    ts = []
    fn()  # warmup
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)


def _time_cuda(fn, repeat: int = 3) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return statistics.median(ts), peak_mb


def _build(n: int, d_ambient: int, seed: int = 0) -> np.ndarray:
    gen = torch.Generator().manual_seed(seed)
    X = datasets.affine_subspace(n, min(5, d_ambient - 1), d_ambient, noise_std=0.01, generator=gen)
    return X.numpy().astype(np.float64)


def run(sizes: list[tuple[int, int]], repeat: int = 3) -> list[Row]:
    rows: list[Row] = []
    cuda_ok = torch.cuda.is_available()
    for name, (Tc, Sc, kw, nmax_sk) in ESTIMATORS.items():
        for n, d in sizes:
            X = _build(n, d)
            Xt_cpu = torch.from_numpy(X).float()

            if n <= nmax_sk:
                sk_ms = _time_once(lambda: _skdim_ctor(name, Sc, kw).fit(X), repeat)
            else:
                sk_ms = float("nan")  # skip — too slow
            tc_ms = _time_once(lambda: Tc(**kw).fit(Xt_cpu), repeat)
            if cuda_ok:
                Xt_gpu = Xt_cpu.cuda()
                tg_ms, peak = _time_cuda(lambda: Tc(**kw).fit(Xt_gpu), repeat)
            else:
                tg_ms, peak = None, None
            rows.append(Row(name, n, d, sk_ms, tc_ms, tg_ms, peak))
            sk_str = f"{sk_ms:8.1f}ms" if sk_ms == sk_ms else "   skip "
            print(
                f"{name:10s} n={n:6d} D={d:4d}  sk={sk_str}  "
                f"tc={tc_ms:8.1f}ms  tg={('%.1f' % tg_ms + 'ms') if tg_ms else '--':>12s}"
            )
    return rows


def _skdim_kwargs(name: str, kw: dict) -> dict:
    return kw


def _skdim_ctor(name: str, Sc: type, kw: dict):
    """Instantiate a skdim estimator, working around its py3.13 MLE bug."""
    if name == "MLE":
        inst = Sc.__new__(Sc)
        inst.dnoise = kw.get("dnoise")
        inst.sigma = kw.get("sigma", 0)
        inst.n = kw.get("n")
        inst.integral_approximation = kw.get("integral_approximation", "Haro")
        inst.unbiased = kw.get("unbiased", False)
        inst.neighborhood_based = kw.get("neighborhood_based", True)
        inst.K = kw.get("K", 5)
        return inst
    return Sc(**kw)


def to_markdown(rows: list[Row]) -> str:
    lines = [
        "| estimator | n | D | skdim (ms) | torch-cpu (ms) | torch-cuda (ms) | "
        "cpu speedup | cuda speedup | cuda peak (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:

        def _f(x):
            return "—" if (x is None or (isinstance(x, float) and x != x)) else f"{x:.1f}"

        cpu_sp = (
            (r.skdim_ms / r.torch_cpu_ms)
            if (r.torch_cpu_ms and r.skdim_ms == r.skdim_ms)
            else float("nan")
        )
        cuda_sp = (
            (r.skdim_ms / r.torch_cuda_ms)
            if (r.torch_cuda_ms and r.skdim_ms == r.skdim_ms)
            else float("nan")
        )
        lines.append(
            f"| {r.estimator} | {r.n} | {r.d_ambient} | {_f(r.skdim_ms)} | "
            f"{_f(r.torch_cpu_ms)} | {_f(r.torch_cuda_ms)} | "
            f"{_f(cpu_sp)}× | {_f(cuda_sp)}× | {_f(r.cuda_peak_mb)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="BENCHMARKS.md")
    parser.add_argument("--small", action="store_true", help="quick smoke run")
    args = parser.parse_args()

    if args.small:
        sizes = [(500, 20), (2000, 20)]
    else:
        sizes = [(2000, 20), (10000, 20), (20000, 50)]

    rows = run(sizes)
    md = to_markdown(rows)
    header = (
        "# Benchmarks\n\n"
        "Synthetic affine-subspace data (true ID = 5) in R^D. Times are median of 3 "
        "runs after warmup. CUDA timings use `torch.cuda.Event`; `peak` is "
        "`torch.cuda.max_memory_allocated`.\n\n"
    )
    with open(args.out, "w") as f:
        f.write(header + md + "\n")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
