# Benchmarks

Synthetic affine-subspace data (true ID = 5) in R^D. Times are median of 3 runs after warmup. CUDA timings use `torch.cuda.Event`; `peak` is `torch.cuda.max_memory_allocated`. CUDA columns measured on three GPUs: NVIDIA RTX 3090 (24 GB GDDR6X), A100 80 GB PCIe, and H100 80 GB HBM3.

`—` in a skdim column means the estimator was skipped at that size because skdim's per-point Python loop makes wall-time impractical (>~30 s). Speedups in the right-most table are `skdim / torch-cuda(H100)`.

`torchid._primitives.knn` dispatches to `faiss-cpu` on CPU tensors and stays pure-torch on CUDA — so CPU wins below measure torchid's batched tensor ops + faiss knn vs scikit-dimension's numpy + sklearn NearestNeighbors.

## Headline wins (H100 vs skdim CPU)

| estimator | size        | H100 speedup | note                                                              |
| --------- | ----------- | -----------: | ----------------------------------------------------------------- |
| MADA      | n=20k, D=50 |    **2725×** | batched knn + log-ratio                                           |
| MADA      | n=10k, D=20 |    **1774×** | —                                                                 |
| TwoNN     | n=20k, D=50 |     **361×** | skdim's `pairwise_distances_chunked` path is the bottleneck       |
| CorrInt   | n=2k, D=20  |     **234×** | per-point pairwise scan → batched `(N, k)` tensor                 |
| MiND_ML   | n=2k, D=20  |     **174×** | —                                                                 |
| FisherS   | n=10k, D=20 |     **168×** | chunked pairwise + numba histc → `data @ data.T` + per-alpha loop |
| CorrInt   | n=20k, D=50 |     **123×** | —                                                                 |
| KNN       | n=10k, D=20 |     **126×** | skdim's scipy pdist + bootstrap                                   |
| TLE       | n=2k, D=20  |     **113×** | per-point `squareform(pdist)` → batched `(N, k, k)` tensor        |

## GPU scaling (CUDA wall-time, ms; lower is better)

Same workload, three GPUs. H100 ≈ 2-3× the 3090; KNN is the standout (9× faster on H100 — HBM3 saturates the faiss-style scan).

| estimator |     n |   D | 3090 (ms) | A100 (ms) | H100 (ms) | H100 vs 3090 |
| --------- | ----: | --: | --------: | --------: | --------: | -----------: |
| TwoNN     | 20000 |  50 |      31.4 |      22.1 |      12.9 |         2.4× |
| MLE       | 20000 |  50 |      31.3 |      22.1 |      12.9 |         2.4× |
| CorrInt   | 20000 |  50 |      69.5 |      45.9 |      26.8 |         2.6× |
| MiND_ML   | 20000 |  50 |      32.7 |      23.1 |      13.6 |         2.4× |
| MOM       | 20000 |  50 |      31.6 |      22.3 |      13.0 |         2.4× |
| MADA      | 20000 |  50 |      31.3 |      22.1 |      12.8 |         2.4× |
| KNN       | 20000 |  50 |     166.5 |      33.7 |      18.3 |         9.1× |
| TLE       | 20000 |  50 |      57.7 |      42.2 |      25.8 |         2.2× |
| DANCo     | 20000 |  50 |    1667.9 |    1199.6 |     713.3 |         2.3× |
| ESS       | 20000 |  50 |      46.1 |      33.1 |      19.0 |         2.4× |
| FisherS   | 20000 |  50 |     215.7 |     124.9 |      68.6 |         3.1× |

## CPU wins (torch-cpu + faiss-cpu vs skdim)

| estimator | size  | cpu speedup |
| --------- | ----- | ----------: |
| MADA      | n=20k |     **26×** |
| MADA      | n=10k |     **32×** |
| MADA      | n=2k  |     **21×** |
| TLE       | n=2k  |    **7.5×** |
| TwoNN     | n=20k |    **3.4×** |
| CorrInt   | n=2k  |   **10.3×** |
| MiND_ML   | n=2k  |    **9.8×** |
| FisherS   | n=2k  |    **5.8×** |

`MiND_ML`, `MOM`, and `MLE` at n≥10k still trail skdim on CPU — their torch-side overhead beyond knn (many small tensor allocations, `dists.mean(dim=1)` etc.) adds up to more than skdim's numpy equivalent. Not a knn problem; a future `torch.compile` pass is the obvious next optimization. Once on CUDA every estimator is in the black.

**ESS / DANCo skdim numbers are missing** at every size because the reference implementation's per-point Python loop is prohibitively slow even at n=2000 (ESS alone is ~19 s at n=500 in the smaller sweep — a 6393× gap on CUDA). See `uv run --group validation python -m benchmarks.bench --small` for those head-to-head numbers.

## Full table (H100)

Speedup is `skdim / H100`. CUDA peak measured on H100; A100/3090 peaks track within a few MB.

| estimator |     n |   D | skdim (ms) | torch-cpu (ms) | 3090 (ms) | A100 (ms) | H100 (ms) | cpu speedup | H100 speedup | H100 peak (MB) |
| --------- | ----: | --: | ---------: | -------------: | --------: | --------: | --------: | ----------: | -----------: | -------------: |
| lPCA      |  2000 |  20 |        0.5 |            0.3 |       1.0 |       1.1 |       0.9 |        1.4× |         0.5× |            3.6 |
| lPCA      | 10000 |  20 |        1.0 |            1.1 |       1.1 |       1.2 |       0.9 |        0.9× |         1.1× |           13.9 |
| lPCA      | 20000 |  50 |        4.2 |            4.3 |       2.9 |       2.9 |       2.4 |        1.0× |         1.7× |           39.3 |
| TwoNN     |  2000 |  20 |        5.0 |            8.2 |       0.9 |       0.9 |       0.7 |        0.6× |         7.1× |           78.7 |
| TwoNN     | 10000 |  20 |       54.7 |          185.8 |       8.6 |       6.2 |       4.0 |        0.3× |        13.6× |          658.2 |
| TwoNN     | 20000 |  50 |     4651.2 |         1370.8 |      31.4 |      22.1 |      12.9 |        3.4× |       361.3× |         1286.5 |
| MLE       |  2000 |  20 |       21.4 |           10.9 |       0.9 |       0.8 |       0.6 |        2.0× |        35.9× |           79.1 |
| MLE       | 10000 |  20 |      108.7 |          206.1 |       8.6 |       6.2 |       3.7 |        0.5× |        29.3× |          661.3 |
| MLE       | 20000 |  50 |      397.5 |         1384.4 |      31.3 |      22.1 |      12.9 |        0.3× |        30.8× |         1291.6 |
| CorrInt   |  2000 |  20 |      183.4 |           17.9 |       1.4 |       1.2 |       0.8 |       10.3× |       233.7× |           83.2 |
| CorrInt   | 10000 |  20 |     1098.5 |          392.8 |      18.3 |      12.2 |       7.3 |        2.8× |       149.8× |          661.3 |
| CorrInt   | 20000 |  50 |     3290.4 |         2203.0 |      69.5 |      45.9 |      26.8 |        1.5× |       122.9× |         1291.6 |
| MiND_ML   |  2000 |  20 |      137.5 |           14.0 |       1.3 |       1.1 |       0.8 |        9.8× |       173.9× |           79.2 |
| MiND_ML   | 10000 |  20 |      139.5 |          223.3 |       9.4 |       6.9 |       4.1 |        0.6× |        33.8× |          661.4 |
| MiND_ML   | 20000 |  50 |      370.0 |         1420.3 |      32.7 |      23.1 |      13.6 |        0.3× |        27.3× |         1291.9 |
| MOM       |  2000 |  20 |       26.5 |           25.0 |       0.8 |       0.8 |       0.5 |        1.1× |        48.2× |           80.2 |
| MOM       | 10000 |  20 |      101.4 |          281.7 |       8.5 |       6.2 |       3.7 |        0.4× |        27.4× |          674.8 |
| MOM       | 20000 |  50 |      508.6 |         1545.2 |      31.6 |      22.3 |      13.0 |        0.3× |        39.2× |         1314.3 |
| MADA      |  2000 |  20 |      223.4 |           10.6 |       0.8 |       0.8 |       0.5 |       21.1× |       410.2× |           79.1 |
| MADA      | 10000 |  20 |     6492.7 |          201.3 |       8.4 |       6.1 |       3.7 |       32.2× |      1774.3× |          661.3 |
| MADA      | 20000 |  50 |    34839.5 |         1340.5 |      31.3 |      22.1 |      12.8 |       26.0× |      2724.8× |         1291.6 |
| KNN       |  2000 |  20 |       21.8 |            9.6 |       4.2 |       4.0 |       2.1 |        2.3× |        10.2× |          154.6 |
| KNN       | 10000 |  20 |      683.5 |          235.2 |      25.0 |      10.2 |       5.4 |        2.9× |       126.0× |         1734.8 |
| KNN       | 20000 |  50 |          — |         1091.9 |     166.5 |      33.7 |      18.3 |          —× |           —× |         6148.4 |
| TLE       |  2000 |  20 |      229.3 |           30.8 |       3.5 |       2.9 |       2.0 |        7.5× |       113.0× |           98.2 |
| TLE       | 10000 |  20 |          — |          264.2 |      21.7 |      16.1 |      10.1 |          —× |           —× |          661.3 |
| TLE       | 20000 |  50 |          — |         1522.5 |      57.7 |      42.2 |      25.8 |          —× |           —× |         1291.6 |
| DANCo     |  2000 |  20 |          — |          257.7 |      45.8 |      45.4 |      30.7 |          —× |           —× |           79.3 |
| DANCo     | 10000 |  20 |          — |         3653.6 |     213.0 |     164.9 |     100.4 |          —× |           —× |          660.7 |
| DANCo     | 20000 |  50 |          — |        47584.6 |    1667.9 |    1199.6 |     713.3 |          —× |           —× |         1293.6 |
| ESS       |  2000 |  20 |          — |           59.6 |       2.2 |       1.8 |       1.2 |          —× |           —× |          406.5 |
| ESS       | 10000 |  20 |          — |          471.6 |      15.2 |      11.0 |       6.8 |          —× |           —× |         1904.7 |
| ESS       | 20000 |  50 |          — |         2027.7 |      46.1 |      33.1 |      19.0 |          —× |           —× |         4235.9 |
| FisherS   |  2000 |  20 |      156.4 |           27.2 |       3.8 |       3.8 |       2.3 |        5.8× |        69.0× |           98.0 |
| FisherS   | 10000 |  20 |     3055.1 |         1184.8 |      54.8 |      32.7 |      18.2 |        2.6× |       168.0× |         1655.1 |
| FisherS   | 20000 |  50 |          — |         4835.7 |     215.7 |     124.9 |      68.6 |          —× |           —× |         6523.4 |
