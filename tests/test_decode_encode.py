import numpy as np
import pytest

from cfq import CFQEncodingError, canonicalize_orientation, decode_cfq, encode_cfq, is_encodable
from cfq.constants import MAX_EXTENT, MIN_HALF_WIDTH


def test_round_trip_random(valid_quads):
    worst = 0.0
    encoded = 0
    for quad in valid_quads:
        try:
            center, params = encode_cfq(quad)
        except CFQEncodingError:
            continue
        encoded += 1
        worst = max(worst, np.abs(decode_cfq(center, params) - quad).max())
    assert encoded > 0.99 * len(valid_quads)
    assert worst < 1e-9


@pytest.mark.parametrize("stride", [1, 4, 8])
def test_round_trip_with_stride(valid_quads, stride):
    for quad in [q * stride for q in valid_quads[:200] if is_encodable(q)]:
        center, params = encode_cfq(quad, stride=stride)
        assert np.allclose(decode_cfq(center, params) * stride, quad, rtol=0, atol=1e-9 * stride * 512)


def test_round_trip_float32_params(valid_quads):
    for quad in valid_quads[:500]:
        if not is_encodable(quad):
            continue
        center, params = encode_cfq(quad, dtype=np.float32)
        assert np.abs(decode_cfq(center, params) - quad).max() < 2e-3


EDGE_CASES = {
    "axis_aligned_rectangle": [[0, 0], [0, 4], [10, 4], [10, 0]],
    "rotated_rectangle": [[1.0, 0.0], [-1.0, 2.0], [7.0, 10.0], [9.0, 8.0]],
    "trapezoid": [[0, -1], [0, 1], [10, 3], [10, -3]],
    "parallelogram": [[0, 0], [2, 4], [12, 4], [10, 0]],
    "narrow_but_encodable": [[0, -0.11], [0, 0.11], [20, 0.11], [20, -0.11]],
    "large": [[0, 0], [0, 300], [400, 300], [400, 0]],
    "near_triangle": [[0, -2], [0, 2], [10, 0.4], [10, -0.4]],
}


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_round_trip_edge_cases(name):
    quad = canonicalize_orientation(np.asarray([EDGE_CASES[name]], np.float64))[0].astype(np.float64)
    center, params = encode_cfq(quad)
    assert np.allclose(decode_cfq(center, params), quad, rtol=0, atol=1e-9)


def test_center_is_vertex_mean(valid_quads):
    for quad in valid_quads[:100]:
        center, _ = encode_cfq(quad)
        assert np.allclose(center, quad.mean(axis=0))


REJECTED = {
    "reversed_order": [[0, 0], [0, 4], [10, 4], [10, 0]],
    "bow_tie": [[0, 0], [0, 4], [10, 0], [10, 4]],
    "too_narrow": [[0, 0.05], [0, -0.05], [10, -0.05], [10, 0.05]],
    "concave": [[0, -2], [0, 2], [10, 2], [1, 0]],
    "collinear_front_and_rear": [[0, 0], [0, 0], [10, 0], [10, 0]],
    "too_long": [[0, 0], [0, 10], [MAX_EXTENT + 10, 10], [MAX_EXTENT + 10, 0]],
    "nonfinite": [[0, 0], [0, np.nan], [10, 4], [10, 0]],
}


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_unencodable(name):
    quad = np.asarray(REJECTED[name], np.float64)
    if name not in ("reversed_order", "nonfinite"):
        quad = canonicalize_orientation(quad[None])[0].astype(np.float64)
    with pytest.raises(CFQEncodingError):
        encode_cfq(quad)
    assert not is_encodable(quad)


def test_minimum_half_width_rule():
    half = MIN_HALF_WIDTH
    inside = [[0, -1.01 * half], [0, 1.01 * half], [10, 1.01 * half], [10, -1.01 * half]]
    outside = [[0, -0.99 * half], [0, 0.99 * half], [10, 0.99 * half], [10, -0.99 * half]]
    assert is_encodable(canonicalize_orientation(np.asarray([inside]))[0])
    assert not is_encodable(canonicalize_orientation(np.asarray([outside]))[0])


def test_canonicalize_orientation():
    quad = np.asarray([[[0, 0], [0, 4], [10, 4], [10, 0]]], np.float32)  # negative y-down area
    assert not is_encodable(quad[0])
    fixed = canonicalize_orientation(quad)
    assert np.array_equal(fixed[0], quad[0, [1, 0, 3, 2]])
    assert is_encodable(fixed[0])
    assert np.array_equal(canonicalize_orientation(fixed), fixed)


def test_torch_decoder_matches_numpy(rng):
    torch = pytest.importorskip("torch")
    from cfq.decode import decode_cfq_torch

    params = rng.normal(0.0, 3.0, (5000, 7))
    centers = rng.uniform(0.0, 300.0, (5000, 2))
    expected = decode_cfq(centers, params)
    got = decode_cfq_torch(torch.as_tensor(centers), torch.as_tensor(params)).numpy()
    assert np.abs(got - expected).max() < 1e-9
    got32 = decode_cfq_torch(torch.as_tensor(centers, dtype=torch.float32), torch.as_tensor(params, dtype=torch.float32))
    assert got32.dtype == torch.float32
    assert np.abs(got32.numpy() - decode_cfq(centers.astype(np.float32), params.astype(np.float32))).max() < 1e-3
