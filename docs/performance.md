# Performance

All numbers below are synthetic affine-subspace data (true ID = 5) in `R^D`, median of 3 runs after warmup, measured with `torch.cuda.Event` on an RTX 3090. Speedups are `skdim / torch-*`. Raw tables live in `BENCHMARKS.md`.

## Headline

| estimator | size        |                             CUDA speedup | CPU speedup |
| --------- | ----------- | ---------------------------------------: | ----------: |
| MADA      | n=20k, D=50 |                                **1000×** |         63× |
| MADA      | n=10k, D=20 |                                     809× |     **77×** |
| TwoNN     | n=20k, D=50 |                                     169× |         10× |
| TLE       | n=2k, D=20  |                                     103× |         14× |
| FisherS   | n=10k, D=20 |                                      99× |        2.1× |
| CorrInt   | n=10k, D=20 |                                      60× |        1.9× |
| KNN       | n=10k, D=20 |                                      39× |        1.6× |
| MLE       | n=2k, D=20  |                                      21× |        4.2× |
| ESS       | n=500, D=20 | **6393×** (skdim only runs at this size) |        924× |

## What's fast, what's slow, and why

**On CUDA, everything wins.** The closed-form estimators (MLE, TwoNN, MOM, MADA) finish in the millisecond range because each one reduces to a single knn + elementwise fused kernel. The heavier per-point estimators (TLE, DANCo, ESS) fit a `(N, k, k)` tensor in memory and vectorize over it — each amounts to one batched matmul + a handful of elementwise kernels.

**On CPU, `MADA` / `TLE` / `CorrInt` still win** because skdim materializes a full `squareform(pdist)` plus Python-level boundary-case logic per point. torchid's CPU path uses faiss-cpu for the neighbor search (SIMD + OpenMP) and a single vectorized numpy equivalent of the estimator math.

**On CPU, `MLE` / `MOM` / `MiND_ML` trail skdim at n ≥ 10k.** Their torch-side overhead (many small tensor allocations, `dists.mean(dim=1)`, etc.) adds up to more than skdim's numpy-native equivalent. It's not the knn — faiss has already won that fight. `torch.compile` over these short closed-form kernels is the obvious future win; they ship unfused today.

**`ESS` and `DANCo` skdim numbers are empty at n ≥ 2000** because the reference implementation's per-point Python loop is too slow to benchmark. The smaller `--small` sweep (n ∈ {500, 2000}) exposes this: skdim's ESS takes 19 seconds at n=500 while torchid finishes in 16 ms on CPU, 3 ms on CUDA.

## Memory

Peak CUDA memory maxes out at ~6.5 GB for FisherS at n=20k, D=50. Everything else stays under 4 GB at those sizes. There's plenty of room to push to n=100k on the 3090 for lPCA / TwoNN / MLE. At n=100k the O(n²) estimators (CorrInt, KNN) would blow through 24 GB even with chunking; you'd either chunk harder or live with subsampling.

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
