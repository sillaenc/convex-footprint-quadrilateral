import itertools

import numpy as np
import pytest

from cfq import cross_products, decode_cfq, invalid_rate, is_strictly_convex, is_valid_quad, reorder_by_angle
from cfq.encode import encode_cfq

EXTREMES = np.array([-1e30, -1e4, -20.0, 0.0, 20.0, 1e4, 1e30])


def _check_positive(quads):
    assert np.isfinite(quads).all()
    cross = cross_products(quads)
    assert (cross > 0).all(), cross.min()
    assert is_strictly_convex(quads).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cross_products_positive_random(rng, dtype):
    n = 50000
    params = rng.normal(0.0, 1.0, (n, 7)) * rng.choice([0.1, 1.0, 5.0, 50.0], (n, 1))
    centers = rng.uniform((0.0, 0.0), (320.0, 180.0), (n, 2))
    _check_positive(decode_cfq(centers.astype(dtype), params.astype(dtype)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cross_products_positive_extreme(rng, dtype):
    n = 50000
    params = EXTREMES[rng.integers(0, len(EXTREMES), (n, 7))]
    centers = rng.uniform((0.0, 0.0), (320.0, 180.0), (n, 2))
    quads = decode_cfq(centers.astype(dtype), params.astype(dtype))
    assert ((params[:, 0] == 0) & (params[:, 1] == 0)).any()
    _check_positive(quads)


def test_cross_products_positive_exhaustive_grid():
    # Every combination of extreme values for the five shape outputs.
    grid = np.asarray(list(itertools.product(EXTREMES, repeat=5)))
    for direction in ([0.0, 0.0], [1.0, 0.0], [-1e30, 1e30], [3.0, -4.0]):
        params = np.column_stack((np.tile(direction, (len(grid), 1)), grid)).astype(np.float32)
        centers = np.full((len(grid), 2), 100.0, np.float32)
        _check_positive(decode_cfq(centers, params))


def test_cross_products_positive_torch(rng):
    torch = pytest.importorskip("torch")
    from cfq.decode import decode_cfq_torch

    n = 50000
    params = np.concatenate((rng.normal(0.0, 3.0, (n, 7)), EXTREMES[rng.integers(0, len(EXTREMES), (n, 7))]))
    centers = rng.uniform((0.0, 0.0), (320.0, 180.0), (2 * n, 2))
    quads = decode_cfq_torch(torch.as_tensor(centers, dtype=torch.float32), torch.as_tensor(params, dtype=torch.float32))
    _check_positive(quads.numpy())


def test_cross_products_formula(valid_quads):
    # Eq. (7): e1 x e2 = 2(L b_f + K), ..., with K = C tanh(s).
    for quad in valid_quads[:200]:
        _, z = encode_cfq(quad)
        length = np.exp(z[2])
        b_f, b_r = length * np.exp(z[3]), length * np.exp(z[4])
        k = ((1 - 1e-3) * min(length * b_f, length * b_r) - 1e-2) * np.tanh(z[6])
        expected = 2 * np.array([length * b_f + k, length * b_r + k, length * b_r - k, length * b_f - k])
        assert np.allclose(cross_products(quad), expected, rtol=1e-9, atol=1e-9)


def test_validity_cases():
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float64)
    assert is_valid_quad(square)
    assert is_valid_quad(square[::-1])  # either winding
    assert not is_valid_quad(square[[0, 2, 1, 3]])  # bow tie
    assert not is_valid_quad(np.array([[0, 0], [1, 0], [2, 0], [1, 1]]))  # collinear triple
    assert not is_valid_quad(np.array([[0, 0], [2, 0], [0.5, 0.5], [0, 2]]))  # concave
    assert not is_valid_quad(np.array([[0, 0], [0, 0], [1, 1], [1, 0]]))  # repeated vertex
    assert not is_valid_quad(np.array([[0, 0], [0, 0.01], [0.01, 0.01], [0.01, 0]]))  # tiny area
    assert not is_valid_quad(np.array([[0, 0], [0, np.inf], [1, 1], [1, 0]]))
    assert invalid_rate(np.stack([square, square[[0, 2, 1, 3]]])) == 0.5
    assert invalid_rate(np.empty((0, 4, 2))) == 0.0


def test_reorder_by_angle():
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32)
    fixed, ok = reorder_by_angle(square[[0, 2, 1, 3]])
    assert ok and is_valid_quad(fixed)
    assert sorted(map(tuple, fixed)) == sorted(map(tuple, square))
    unchanged, ok = reorder_by_angle(square)
    assert ok and np.array_equal(unchanged, square)
    _, ok = reorder_by_angle(np.array([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32))
    assert not ok
