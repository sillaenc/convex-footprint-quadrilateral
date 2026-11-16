"""CFQ encoder: ground-truth quadrilateral -> CFQ parameters (inverse decoder).

The encoder inverts Eqs. (1)-(4). A quadrilateral that lies outside the range
the decoder can represent raises :class:`CFQEncodingError`; in training such a
vehicle is treated as an ignored ("excluded") object rather than background
(Section 3.4). For every accepted quadrilateral, decoding the returned
parameters reproduces its vertices up to floating-point rounding.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    MARGIN,
    MAX_ASPECT,
    MAX_EXTENT,
    MIN_HALF_WIDTH,
    MIN_LENGTH,
    Q_ASPECT,
    RAW_LIMIT,
    TURN_FLOOR,
)

# Rounding overshoot at the open interval (-1, 1) is clipped before atanh.
_UNIT_CLIP = 1.0 - 1e-7


class CFQEncodingError(ValueError):
    """The quadrilateral cannot be represented by the CFQ decoder."""


def canonicalize_orientation(quads) -> np.ndarray:
    """Return copies of ``(N, 4, 2)`` quads with positive y-down signed area.

    A quad with negative signed area is mapped by the lateral swap
    ``[FL, FR, RR, RL] -> [FR, FL, RL, RR]``, which keeps the front and rear
    edges. Degenerate and self-intersecting quads are left unchanged (the
    encoder rejects them).
    """

    values = np.asarray(quads, dtype=np.float32).copy()
    if values.size == 0:
        return values.reshape(-1, 4, 2)
    if values.ndim != 3 or values.shape[1:] != (4, 2):
        raise ValueError("quads must have shape (N, 4, 2)")
    area2 = (values[..., 0] * np.roll(values[..., 1], -1, axis=1)).sum(axis=1) - (
        values[..., 1] * np.roll(values[..., 0], -1, axis=1)
    ).sum(axis=1)
    reverse = area2 < 0.0
    values[reverse] = values[reverse][:, (1, 0, 3, 2)]
    return values


def encode_cfq(quad, stride: float = 1.0, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Encode one quadrilateral ``[FL, FR, RR, RL]`` into CFQ parameters.

    Args:
        quad: ``(4, 2)`` vertices in image pixels, ordered so that the signed
            area is positive in y-down image coordinates (see
            :func:`canonicalize_orientation`).
        stride: output stride; pixels are divided by it to obtain the
            feature-map units in which the decoder works.
        dtype: dtype of the returned arrays. All arithmetic is float64.

    Returns:
        ``(center, params)``: the center ``c`` (mean of the vertices, feature
        units) and ``z = (v_x, v_y, l, rho_f, rho_r, q0, s)`` with ``v`` the
        unit axis direction, so that ``decode_cfq(center, params)`` returns
        ``quad / stride``.

    Raises:
        CFQEncodingError: if the quadrilateral is outside the representable
            range (wrong orientation, too narrow, too elongated, outside the
            slope bounds, or larger than the decoder's feature extent).
    """

    if not stride > 0:
        raise ValueError("stride must be positive")
    points = np.asarray(quad, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise CFQEncodingError("quad must be a finite (4, 2) array")
    signed_area2 = float(
        np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if not signed_area2 > 0.0:
        raise CFQEncodingError("quad must have positive y-down signed area in [FL, FR, RR, RL] order")

    # Axis, length and half-vectors (Eqs. (1)-(2) inverted).
    m_f = 0.5 * (points[0] + points[1])
    m_r = 0.5 * (points[2] + points[3])
    axis = (m_r - m_f) / float(stride)
    length = float(np.linalg.norm(axis))
    if not length > MIN_HALF_WIDTH:
        raise CFQEncodingError("front and rear midpoints are too close")
    t = axis / length
    n = np.array((-t[1], t[0]))
    s_f = (points[0] - m_f) / float(stride)
    s_r = (points[3] - m_r) / float(stride)
    a_f, b_f = float(np.dot(s_f, t)), float(np.dot(s_f, n))
    a_r, b_r = float(np.dot(s_r, t)), float(np.dot(s_r, n))
    if not (b_f > MIN_HALF_WIDTH and b_r > MIN_HALF_WIDTH):
        raise CFQEncodingError("a half-width is not above the minimum half-width")

    # Numerical limits of Section 3.1.
    min_b = max(length / MAX_ASPECT, 2.0 * TURN_FLOOR / ((1.0 - MARGIN) * length))
    max_b = min(MAX_EXTENT, MAX_ASPECT * length)
    if not (MIN_LENGTH <= length <= MAX_EXTENT and min_b <= b_f <= max_b and min_b <= b_r <= max_b):
        raise CFQEncodingError("length or half-width is outside the representable range")

    # Eq. (4) inverted: common slope and slope difference.
    q = 0.5 * (a_f / b_f + a_r / b_r)
    q_max = min(RAW_LIMIT, Q_ASPECT * length / max(b_f, b_r))
    capacity = (1.0 - MARGIN) * min(length * b_f, length * b_r) - TURN_FLOOR
    if not capacity > 0.0:
        raise CFQEncodingError("quad is below the turn floor")
    q_unit = q / q_max
    shear_unit = b_f * b_r * (a_f / b_f - a_r / b_r) / capacity
    if not (abs(q_unit) < 1.0 and abs(shear_unit) < 1.0):
        raise CFQEncodingError("common slope or slope difference is outside its open bound")
    q_unit = float(np.clip(q_unit, -_UNIT_CLIP, _UNIT_CLIP))
    shear_unit = float(np.clip(shear_unit, -_UNIT_CLIP, _UNIT_CLIP))

    params = np.asarray(
        (
            t[0],
            t[1],
            math.log(length),
            math.log(b_f / length),
            math.log(b_r / length),
            np.arctanh(q_unit),
            np.arctanh(shear_unit),
        ),
        dtype=dtype,
    )
    center = (points.mean(axis=0) / float(stride)).astype(dtype)
    return center, params


def is_encodable(quad, stride: float = 1.0) -> bool:
    """Return whether :func:`encode_cfq` accepts ``quad``."""

    try:
        encode_cfq(quad, stride)
    except CFQEncodingError:
        return False
    return True
