"""Unit tests for torchid._primitives — covers both the CPU faiss path and the
torch chunked path (via the explicit internal entry points)."""

import numpy as np
import pytest
import torch

from torchid import _primitives as P


def test_as_tensor_float64_preserved() -> None:
    X = torch.randn(10, 3, dtype=torch.float64)
    out = P.as_tensor(X)
    assert out.dtype == torch.float64


def test_as_tensor_list_input() -> None:
    out = P.as_tensor([[1, 2, 3], [4, 5, 6]])
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32


def test_as_tensor_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        P.as_tensor(torch.randn(10))


def test_as_tensor_dtype_override() -> None:
    out = P.as_tensor(torch.randn(3, 2, dtype=torch.float64), dtype=torch.float32)
    assert out.dtype == torch.float32


def test_pairwise_sqdist_chunked_matches_unchunked() -> None:
    X = torch.randn(200, 5, generator=torch.Generator().manual_seed(0))
    a = P.pairwise_sqdist(X, chunk=16)
    b = P.pairwise_sqdist(X, chunk=1000)
    assert torch.allclose(a, b, atol=1e-5)
    assert a.diag().abs().max() < 1e-6


def test_pairwise_sqdist_cross_YX() -> None:
    g = torch.Generator().manual_seed(1)
    X = torch.randn(30, 4, generator=g)
    Y = torch.randn(50, 4, generator=g)
    d = P.pairwise_sqdist(X, Y)
    assert d.shape == (30, 50)


def test_knn_torch_path_matches_faiss_path() -> None:
    # We exercise the CUDA chunked path directly via the private helper so this
    # test runs on any host.
    X = torch.randn(100, 6, generator=torch.Generator().manual_seed(0))
    d_faiss, i_faiss = P._knn_faiss(X, k=5, Y=None, include_self=False)
    d_torch, i_torch = P._knn_torch(X, k=5, Y=None, include_self=False, chunk=32)
    # Indices can differ under ties; distances must match
    assert torch.allclose(
        torch.sort(d_faiss, dim=1).values,
        torch.sort(d_torch, dim=1).values,
        atol=1e-5,
    )


def test_knn_too_large_k_raises() -> None:
    X = torch.randn(10, 2)
    with pytest.raises(ValueError, match="requested k"):
        P.knn(X, k=20)


def test_knn_cross_reference() -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.randn(20, 4, generator=g)
    Y = torch.randn(50, 4, generator=g)
    d, idx = P.knn(X, k=5, Y=Y)
    assert d.shape == (20, 5)
    assert idx.max() < 50


def test_gather_neighbors() -> None:
    X = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    idx = torch.tensor([[1, 4], [0, 9]])
    out = P.gather_neighbors(X, idx)
    assert out.shape == (2, 2, 3)
    assert torch.equal(out[0, 0], X[1])
    assert torch.equal(out[1, 1], X[9])


def test_batched_local_pca() -> None:
    g = torch.Generator().manual_seed(0)
    nbrs = torch.randn(4, 15, 3, generator=g)
    evals, evecs = P.batched_local_pca(nbrs)
    assert evals.shape == (4, 3)
    assert evecs.shape == (4, 3, 3)
    assert (evals >= -1e-6).all()
    # Descending
    assert torch.all(evals[:, :-1] >= evals[:, 1:] - 1e-6)


def test_batched_local_pca_no_center() -> None:
    nbrs = torch.randn(3, 10, 4, generator=torch.Generator().manual_seed(1))
    e1, _ = P.batched_local_pca(nbrs, center=True)
    e2, _ = P.batched_local_pca(nbrs, center=False)
    # With a non-zero mean the uncentered version gains one large eigenvalue
    assert (e2[:, 0] >= e1[:, 0]).all()


def test_log_knn_ratios_shape() -> None:
    dists = torch.tensor([[1.0, 2.0, 4.0, 8.0], [1.0, 1.5, 3.0, 5.0]])
    out = P.log_knn_ratios(dists)
    assert out.shape == (2, 3)
    # log(8/1) = log 8
    assert torch.allclose(out[0, 0], torch.log(torch.tensor(8.0)))


def test_log_knn_ratios_rejects_1d() -> None:
    with pytest.raises(ValueError, match="at least 2-D"):
        P.log_knn_ratios(torch.tensor([1.0, 2.0, 3.0]))


def test_sample_combinations_random_path() -> None:
    idx = P.sample_combinations(k=20, p=3, m=50)
    assert idx.shape == (50, 3)
    assert idx.min() >= 0
    assert idx.max() < 20
    # each row is a p-subset (no duplicates within a row)
    assert all(torch.unique(idx[i]).numel() == 3 for i in range(50))


def test_sample_combinations_exact_enum() -> None:
    # requesting more than C(k, p) falls back to full enumeration
    idx = P.sample_combinations(k=5, p=2, m=999)
    assert idx.shape == (10, 2)  # C(5, 2) = 10


def test_sample_combinations_invalid_p() -> None:
    with pytest.raises(ValueError, match="p="):
        P.sample_combinations(k=3, p=5, m=1)
