# Parity with scikit-dimension

Every torchid estimator is cross-checked against its scikit-dimension counterpart on the same synthetic datasets — all twelve estimators land within the per-estimator tolerance band documented below.

## Test matrix

The default parity cases live in `tests/_parity.DEFAULT_CASES`:

| case                      |    n | true ID | ambient |
| ------------------------- | ---: | ------: | ------: |
| `hyperball`               | 2000 |       5 |       5 |
| `hyperball`               | 2000 |      10 |      10 |
| `affine` (noise_std=0.01) | 2000 |       3 |      15 |
| `affine` (noise_std=0.01) | 2000 |       8 |      30 |
| `swissroll`               | 2000 |       2 |       3 |

ESS and DANCo parity runs on a smaller grid (`SMALL_CASES`, n=500) because skdim's per-point Python loops are prohibitively slow at n=2000 — see [Performance](performance.md).

## Per-estimator tolerances

| estimator               | atol | rtol | notes                                           |
| ----------------------- | ---: | ---: | ----------------------------------------------- |
| lPCA (all 7 heuristics) | 1e-4 | 1e-3 | —                                               |
| TwoNN                   | 1e-4 | 5e-3 | —                                               |
| MLE                     | 1e-4 | 5e-3 | skdim patched via `__new__` for py3.13          |
| CorrInt                 | 1e-4 | 5e-3 | —                                               |
| MiND_ML (MLi, MLk)      | 0.05 | 2e-2 | grid refinement replaces scipy L-BFGS-B         |
| MOM                     | 1e-4 | 5e-3 | —                                               |
| MADA                    | 1e-4 | 1e-3 | —                                               |
| TLE                     | 1e-3 | 1e-2 | —                                               |
| KNN                     |    — |    — | integer output; allow off-by-one on noisy cases |
| DANCo                   |  1.5 | 0.25 | different RNG for hyperball calibration         |
| ESS                     |  0.5 |  0.1 | Monte-Carlo for d > 1                           |
| FisherS                 |  0.1 | 1e-2 | —                                               |

`DANCo` has the loosest bounds because the calibration step generates synthetic hyperballs using **torch's RNG** rather than numpy's, so the reference set differs bin-for-bin from skdim's. The underlying algorithm converges to the same answer, just not the same numerical trajectory.

## Running parity tests

```bash
uv sync --group validation
uv run pytest tests/ -q
# 25 passed in ~80s
```

The harness lives in `tests/_parity`:

```python
from ._parity import Case, DEFAULT_CASES, compare_global, assert_parity
from torchid.estimators import TwoNN
import skdim.id as skid

rows = compare_global(
    torch_cls=TwoNN,
    skdim_cls=skid.TwoNN,
    cases=DEFAULT_CASES,
    atol=1e-4,
    rtol=5e-3,
)
assert_parity(rows)
```

`compare_global` returns one row per case with `torch_dim`, `skdim_dim`, `abs_err`, `rel_err`, `pass`; `assert_parity` raises a readable diff when the fraction of passing rows falls below the threshold.

## Known caveats

### skdim's MLE.__init__ on Python 3.13

skdim's `MLE` binds its constructor args via `inspect.getargvalues()` and mutates `frame.f_locals`. That's read-only starting in 3.13 and the call raises `ValueError: cannot remove local variables from FrameLocalsProxy`. Our parity test and bench both work around it by constructing skdim MLE via `__new__` + manual attribute assignment.

### Degenerate-eigenvalue cases

An exact-rank affine subspace (`noise_std=0`) has `ambient - d` zero eigenvalues. sklearn's PCA clips them to 0, which makes `lPCA(ver="maxgap")` pick a different index than ours (torch keeps them as tiny-but-positive). The parity harness defaults to `noise_std=0.01` so tails stay finite.

### KNN's argmin sensitivity

`KNN` returns the integer `m = argmin` over a least-squares residual. At ambient D=30, n=2000, M=1 the residuals for competing `m` are close enough that RNG or fp32-vs-fp64 slop swaps the winner. torchid's KNN runs the inner math in float64 (matching skdim's scipy pdist) and uses `np.random.randint` (matching skdim's bootstrap RNG). The test allows off-by-one.

### ESS d > 1

We enumerate all `C(k, p)` simple elements when it fits, otherwise fall back to Monte-Carlo sampling `m=5000` p-subsets. For the default `d=1` this doesn't matter — the per-neighborhood ESS has a closed form in terms of pairwise angles.
