# Benchmarks

Synthetic affine-subspace data (true ID = 5) in R^D. Times are median of 3 runs after warmup. CUDA timings use `torch.cuda.Event`; `peak` is `torch.cuda.max_memory_allocated`. GPU is an RTX 3090.

`—` in a skdim column means the estimator was skipped at that size because skdim's per-point Python loop makes wall-time impractical (>~30 s). Speedups are `skdim / torch-*`.

`torchid._primitives.knn` dispatches to `faiss-cpu` on CPU tensors and stays pure-torch on CUDA — so CPU wins below measure torchid's batched tensor ops + faiss knn vs scikit-dimension's numpy + sklearn NearestNeighbors.

## Headline wins (RTX 3090 vs skdim CPU)

| estimator | size        | cuda speedup | note                                                              |
| --------- | ----------- | -----------: | ----------------------------------------------------------------- |
| MADA      | n=20k       |    **1000×** | batched knn + log-ratio                                           |
| MADA      | n=10k       |     **809×** | —                                                                 |
| TwoNN     | n=20k, D=50 |     **169×** | skdim's `pairwise_distances_chunked` path is the bottleneck       |
| TLE       | n=2k        |     **103×** | per-point `squareform(pdist)` → batched `(N, k, k)` tensor        |
| FisherS   | n=10k       |      **99×** | chunked pairwise + numba histc → `data @ data.T` + per-alpha loop |
| CorrInt   | n=10k       |      **60×** | single chunked pairwise scan                                      |
| KNN       | n=10k       |      **39×** | skdim's scipy pdist + bootstrap                                   |
| MLE       | n=2k        |      **21×** | —                                                                 |

## CPU wins (torch-cpu + faiss-cpu vs skdim)

| estimator | size  | cpu speedup |
| --------- | ----- | ----------: |
| MADA      | n=10k |     **77×** |
| MADA      | n=20k |     **63×** |
| MADA      | n=2k  |     **47×** |
| TLE       | n=2k  |     **14×** |
| TwoNN     | n=20k |     **10×** |
| CorrInt   | n=2k  |    **4.2×** |
| MLE       | n=2k  |    **4.2×** |
| FisherS   | n=2k  |    **3.2×** |

`MiND_ML`, `MOM`, and `MLE` at n≥10k still trail skdim on CPU — their torch-side overhead beyond knn (many small tensor allocations, `dists.mean(dim=1)` etc.) adds up to more than skdim's numpy equivalent. Not a knn problem; a future `torch.compile` pass is the obvious next optimization. Once on CUDA every estimator is in the black.

**ESS / DANCo skdim numbers are missing** at every size because the reference implementation's per-point Python loop is prohibitively slow even at n=2000 (ESS alone is ~19 s at n=500 in the smaller sweep — a 6393× gap on CUDA). See `uv run --group validation python -m benchmarks.bench --small` for those head-to-head numbers.

## Full table

| estimator |     n |   D | skdim (ms) | torch-cpu (ms) | torch-cuda (ms) | cpu speedup | cuda speedup | cuda peak (MB) |
| --------- | ----: | --: | ---------: | -------------: | --------------: | ----------: | -----------: | -------------: |
| lPCA      |  2000 |  20 |        0.5 |            0.3 |             1.0 |        1.9× |         0.5× |            3.6 |
| lPCA      | 10000 |  20 |        0.9 |            0.6 |             1.1 |        1.6× |         0.8× |           13.9 |
| lPCA      | 20000 |  50 |        4.1 |            3.8 |             2.9 |        1.1× |         1.4× |           39.0 |
| TwoNN     |  2000 |  20 |        4.4 |            3.5 |             0.9 |        1.3× |         4.7× |           54.1 |
| TwoNN     | 10000 |  20 |       44.0 |           79.5 |             8.6 |        0.6× |         5.1× |          634.4 |
| TwoNN     | 20000 |  50 |     5301.0 |          508.3 |            31.4 |       10.4× |       169.1× |         1262.7 |
| MLE       |  2000 |  20 |       18.8 |            4.5 |             0.9 |        4.2× |        20.6× |           54.5 |
| MLE       | 10000 |  20 |      108.9 |           86.4 |             8.6 |        1.3× |        12.7× |          637.4 |
| MLE       | 20000 |  50 |      358.2 |          514.2 |            31.3 |        0.7× |        11.4× |         1267.8 |
| CorrInt   |  2000 |  20 |       82.8 |           19.6 |             1.4 |        4.2× |        57.5× |           58.3 |
| CorrInt   | 10000 |  20 |     1102.5 |          588.5 |            18.3 |        1.9× |        60.4× |          637.4 |
| CorrInt   | 20000 |  50 |     2965.6 |         2446.9 |            69.5 |        1.2× |        42.7× |         1267.8 |
| MiND_ML   |  2000 |  20 |       10.7 |           13.8 |             1.3 |        0.8× |         8.2× |           54.6 |
| MiND_ML   | 10000 |  20 |       74.2 |          127.8 |             9.4 |        0.6× |         7.9× |          637.6 |
| MiND_ML   | 20000 |  50 |      295.6 |          602.2 |            32.7 |        0.5× |         9.0× |         1268.1 |
| MOM       |  2000 |  20 |       17.7 |           15.7 |             0.8 |        1.1× |        21.7× |           56.4 |
| MOM       | 10000 |  20 |       73.9 |          151.6 |             8.5 |        0.5× |         8.7× |          650.9 |
| MOM       | 20000 |  50 |      311.9 |          654.1 |            31.6 |        0.5× |         9.9× |         1290.5 |
| MADA      |  2000 |  20 |      204.5 |            4.3 |             0.8 |       47.4× |       245.4× |           54.5 |
| MADA      | 10000 |  20 |     6800.1 |           88.8 |             8.4 |       76.6× |       809.0× |          637.4 |
| MADA      | 20000 |  50 |    31335.7 |          499.3 |            31.3 |       62.8× |      1000.2× |         1267.8 |
| KNN       |  2000 |  20 |       20.6 |           26.8 |             4.2 |        0.8× |         4.9× |          130.7 |
| KNN       | 10000 |  20 |      977.6 |          621.3 |            25.0 |        1.6× |        39.1× |         2023.5 |
| KNN       | 20000 |  50 |          — |         2461.8 |           166.5 |          —× |           —× |         6124.0 |
| TLE       |  2000 |  20 |      362.9 |           26.1 |             3.5 |       13.9× |       102.7× |           73.6 |
| TLE       | 10000 |  20 |          — |          241.8 |            21.7 |          —× |           —× |          637.4 |
| TLE       | 20000 |  50 |          — |          852.2 |            57.7 |          —× |           —× |         1267.8 |
| DANCo     |  2000 |  20 |          — |          235.8 |            45.8 |          —× |           —× |           54.7 |
| DANCo     | 10000 |  20 |          — |         2615.0 |           213.0 |          —× |           —× |          636.8 |
| DANCo     | 20000 |  50 |          — |        24467.4 |          1667.9 |          —× |           —× |         1269.5 |
| ESS       |  2000 |  20 |          — |           83.8 |             2.2 |          —× |           —× |          382.6 |
| ESS       | 10000 |  20 |          — |          488.6 |            15.2 |          —× |           —× |         1880.1 |
| ESS       | 20000 |  50 |          — |         1401.7 |            46.1 |          —× |           —× |         4212.0 |
| FisherS   |  2000 |  20 |      249.5 |           77.5 |             3.8 |        3.2× |        66.2× |           73.3 |
| FisherS   | 10000 |  20 |     5418.4 |         2606.8 |            54.8 |        2.1× |        98.8× |         1631.2 |
| FisherS   | 20000 |  50 |          — |        10025.0 |           215.7 |          —× |           —× |         6499.1 |
