# Performance

All numbers below are synthetic affine-subspace data (true ID = 5) in `R^D`, median of 3 runs after warmup, measured with `torch.cuda.Event`. Speedups are `skdim / torch-*`. CUDA columns measured on three GPUs: NVIDIA RTX 3090, A100 80 GB PCIe, and H100 80 GB HBM3. Raw tables live in `BENCHMARKS.md`.

## Headline (H100)

| estimator | size        |                              CUDA speedup | CPU speedup |
| --------- | ----------- | ----------------------------------------: | ----------: |
| MADA      | n=20k, D=50 |                                 **2725×** |         26× |
| MADA      | n=10k, D=20 |                                 **1774×** |     **32×** |
| TwoNN     | n=20k, D=50 |                                      361× |        3.4× |
| CorrInt   | n=2k, D=20  |                                      234× |     **10×** |
| MiND_ML   | n=2k, D=20  |                                      174× |        9.8× |
| FisherS   | n=10k, D=20 |                                      168× |        2.6× |
| CorrInt   | n=20k, D=50 |                                      123× |        1.5× |
| KNN       | n=10k, D=20 |                                      126× |        2.9× |
| TLE       | n=2k, D=20  |                                      113× |        7.5× |
| MLE       | n=2k, D=20  |                                       36× |        2.0× |
| ESS       | n=500, D=20 | **9000×+** (skdim only runs at this size) |        924× |

## GPU scaling

H100 lands ≈ 2-3× the 3090 across the board on n=20k workloads. KNN is the standout — **9.1×** faster on H100, because faiss-style chunked dist scans saturate HBM3 bandwidth.

| estimator | n=20k, D=50 | 3090 (ms) | A100 (ms) | H100 (ms) | H100 vs 3090 |
| --------- | ----------- | --------: | --------: | --------: | -----------: |
| MADA      |             |      31.3 |      22.1 |      12.8 |         2.4× |
| TwoNN     |             |      31.4 |      22.1 |      12.9 |         2.4× |
| CorrInt   |             |      69.5 |      45.9 |      26.8 |         2.6× |
| KNN       |             |     166.5 |      33.7 |      18.3 |     **9.1×** |
| FisherS   |             |     215.7 |     124.9 |      68.6 |         3.1× |
| DANCo     |             |    1667.9 |    1199.6 |     713.3 |         2.3× |

## What's fast, what's slow, and why

**On CUDA, everything wins.** The closed-form estimators (MLE, TwoNN, MOM, MADA) finish in the millisecond range because each one reduces to a single knn + elementwise fused kernel. The heavier per-point estimators (TLE, DANCo, ESS) fit a `(N, k, k)` tensor in memory and vectorize over it — each amounts to one batched matmul + a handful of elementwise kernels.

**On CPU, `MADA` / `TLE` / `CorrInt` still win** because skdim materializes a full `squareform(pdist)` plus Python-level boundary-case logic per point. torchid's CPU path uses faiss-cpu for the neighbor search (SIMD + OpenMP) and a single vectorized numpy equivalent of the estimator math.

**On CPU, `MLE` / `MOM` / `MiND_ML` trail skdim at n ≥ 10k.** Their torch-side overhead (many small tensor allocations, `dists.mean(dim=1)`, etc.) adds up to more than skdim's numpy-native equivalent. It's not the knn — faiss has already won that fight. `torch.compile` over these short closed-form kernels is the obvious future win; they ship unfused today.

**`ESS` and `DANCo` skdim numbers are empty at n ≥ 2000** because the reference implementation's per-point Python loop is too slow to benchmark. The smaller `--small` sweep (n ∈ {500, 2000}) exposes this: skdim's ESS takes 19 seconds at n=500 while torchid finishes in 16 ms on CPU, 3 ms on CUDA.

## Memory

Peak CUDA memory maxes out at ~6.5 GB for FisherS at n=20k, D=50. Everything else stays under 4 GB at those sizes. There's plenty of room to push to n=100k on the 3090 for lPCA / TwoNN / MLE; on the 80 GB A100/H100 even the O(n²) estimators (CorrInt, KNN) fit at n≈100k. On a 24 GB 3090 you'd need to chunk harder or subsample for those sizes.

## Reproducing

```bash
# Small sweep (n ∈ {500, 2000}, ~5 min) — includes ESS/DANCo vs skdim
uv run --group validation python -m benchmarks.bench --small --out BENCHMARKS-small.md

# Default sweep (n ∈ {2k, 10k, 20k}, ~25 min)
uv run --group validation python -m benchmarks.bench --out BENCHMARKS.md
```

The bench harness lives at `benchmarks/bench.py`. Each estimator has an `n_max_sk` bound — above that, skdim is skipped to keep wall-time in check. torch-cpu and torch-cuda are measured at every requested size.

## Caveats

- Bench times `fit(X)` only; it doesn't cross-check numerical output. Correctness is covered separately by the parity tests in `tests/test_parity_*.py`.
- All runs use `torch.float32`. For problems where fp32 accumulation matters (very large n or very small distances), cast `X` to `float64` before `fit`.
- KNN's bootstrap uses `np.random.randint`; the bench results are not perfectly reproducible across runs unless you set `np.random.seed` before each call.
