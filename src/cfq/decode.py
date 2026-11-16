"""CFQ decoder: center + seven raw outputs -> four vertices [FL, FR, RR, RL].

The decoder implements Eqs. (1)-(4) of the paper. For every finite input it
returns a strictly convex quadrilateral whose four consecutive-edge cross
products are positive (Eq. (7)); see :mod:`cfq.convexity`.

Two implementations with identical arithmetic are provided:

* :func:`decode_cfq` -- NumPy, computes in the floating dtype of its inputs
  (float32 inputs are decoded in float32, everything else in float64).
* :func:`decode_cfq_torch` -- PyTorch, differentiable. Geometry runs in
  float32 (half-precision inputs are upcast, as under mixed precision
  training); float64 inputs are decoded in float64.

Coordinates are feature-map units. Multiply the vertices by the output stride
to obtain input-image pixels.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    DIRECTION_EPS,
    MARGIN,
    MAX_ASPECT,
    MAX_EXTENT,
    MIN_HALF_WIDTH,
    MIN_LENGTH,
    NUM_PARAMS,
    Q_ASPECT,
    RAW_LIMIT,
    TURN_FLOOR,
)

_LOG_MIN_LENGTH = math.log(MIN_LENGTH)
_LOG_MAX_EXTENT = math.log(MAX_EXTENT)
_LOG_MAX_ASPECT = math.log(MAX_ASPECT)


def _check_shapes(center_shape, params_shape) -> None:
    if center_shape[-1] < 2:
        raise ValueError("center must have at least 2 values in its last axis")
    if params_shape[-1] != NUM_PARAMS:
        raise ValueError(f"params must have {NUM_PARAMS} values in its last axis")


def decode_cfq(center, params) -> np.ndarray:
    """Decode CFQ parameters into vertices with NumPy.

    Args:
        center: ``(..., 2)`` center ``c`` (mean of the four vertices).
        params: ``(..., 7)`` raw outputs ``(v_x, v_y, l, rho_f, rho_r, q0, s)``.

    Returns:
        ``(..., 4, 2)`` vertices in the order FL, FR, RR, RL.
    """

    center = np.asarray(center)
    params = np.asarray(params)
    _check_shapes(center.shape, params.shape)
    dtype = np.float32 if center.dtype == np.float32 and params.dtype == np.float32 else np.float64
    center = center[..., :2].astype(dtype, copy=False)
    params = params.astype(dtype, copy=False)

    # Eq. (1): direction, with an explicit unit fallback for a zero vector.
    v = np.clip(params[..., 0:2], -RAW_LIMIT, RAW_LIMIT)
    norm = np.sqrt(v[..., 0:1] * v[..., 0:1] + v[..., 1:2] * v[..., 1:2])
    unit = v / np.maximum(norm, dtype(DIRECTION_EPS))
    fallback = np.zeros_like(unit)
    fallback[..., 0] = 1.0
    t = np.where(norm >= DIRECTION_EPS, unit, fallback)
    n = np.stack((-t[..., 1], t[..., 0]), axis=-1)

    # Eq. (1): length and half-widths, within the numerical limits of Sec. 3.1.
    length = np.exp(np.clip(params[..., 2:3], _LOG_MIN_LENGTH, _LOG_MAX_EXTENT))
    min_b = np.maximum(
        np.maximum(np.full_like(length, MIN_HALF_WIDTH), length / MAX_ASPECT),
        2.0 * TURN_FLOOR / ((1.0 - MARGIN) * length),
    )
    max_b = np.minimum(np.full_like(length, MAX_EXTENT), length * MAX_ASPECT)
    rho_f = np.clip(params[..., 3:4], -_LOG_MAX_ASPECT, _LOG_MAX_ASPECT)
    rho_r = np.clip(params[..., 4:5], -_LOG_MAX_ASPECT, _LOG_MAX_ASPECT)
    b_f = np.minimum(np.maximum(length * np.exp(rho_f), min_b), max_b)
    b_r = np.minimum(np.maximum(length * np.exp(rho_r), min_b), max_b)

    # Eq. (4): bounded common slope and slope difference.
    q_max = np.minimum(np.full_like(length, RAW_LIMIT), Q_ASPECT * length / np.maximum(b_f, b_r))
    q = q_max * np.tanh(np.clip(params[..., 5:6], -RAW_LIMIT, RAW_LIMIT))
    capacity = (1.0 - MARGIN) * np.minimum(length * b_f, length * b_r) - TURN_FLOOR
    delta_q = capacity / (b_f * b_r) * np.tanh(np.clip(params[..., 6:7], -RAW_LIMIT, RAW_LIMIT))

    # Eqs. (2)-(3): midpoints, half-vectors and vertices.
    a_f = b_f * (q + 0.5 * delta_q)
    a_r = b_r * (q - 0.5 * delta_q)
    m_f = center - 0.5 * length * t
    m_r = center + 0.5 * length * t
    s_f = a_f * t + b_f * n
    s_r = a_r * t + b_r * n
    return np.stack((m_f + s_f, m_f - s_f, m_r - s_r, m_r + s_r), axis=-2)


def decode_cfq_torch(center, params):
    """Differentiable PyTorch version of :func:`decode_cfq`.

    Args:
        center: ``(..., 2)`` tensor, center ``c``.
        params: ``(..., 7)`` tensor, raw outputs ``(v_x, v_y, l, rho_f, rho_r, q0, s)``.

    Returns:
        ``(..., 4, 2)`` tensor of vertices FL, FR, RR, RL.
    """

    import torch

    _check_shapes(center.shape, params.shape)
    dtype = torch.float64 if center.dtype == torch.float64 and params.dtype == torch.float64 else torch.float32
    center = center[..., :2].to(dtype)
    params = params.to(dtype)

    v = params[..., 0:2].clamp(-RAW_LIMIT, RAW_LIMIT)
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    unit = v / norm.clamp_min(DIRECTION_EPS)
    fallback = torch.zeros_like(unit)
    fallback[..., 0] = 1.0
    t = torch.where(norm >= DIRECTION_EPS, unit, fallback)
    n = torch.stack((-t[..., 1], t[..., 0]), dim=-1)

    length = params[..., 2:3].clamp(_LOG_MIN_LENGTH, _LOG_MAX_EXTENT).exp()
    min_b = torch.maximum(
        torch.maximum(torch.full_like(length, MIN_HALF_WIDTH), length / MAX_ASPECT),
        2.0 * TURN_FLOOR / ((1.0 - MARGIN) * length),
    )
    max_b = torch.minimum(torch.full_like(length, MAX_EXTENT), length * MAX_ASPECT)
    rho_f = params[..., 3:4].clamp(-_LOG_MAX_ASPECT, _LOG_MAX_ASPECT)
    rho_r = params[..., 4:5].clamp(-_LOG_MAX_ASPECT, _LOG_MAX_ASPECT)
    b_f = (length * rho_f.exp()).clamp(min=min_b, max=max_b)
    b_r = (length * rho_r.exp()).clamp(min=min_b, max=max_b)

    q_max = torch.minimum(torch.full_like(length, RAW_LIMIT), Q_ASPECT * length / torch.maximum(b_f, b_r))
    q = q_max * torch.tanh(params[..., 5:6].clamp(-RAW_LIMIT, RAW_LIMIT))
    capacity = (1.0 - MARGIN) * torch.minimum(length * b_f, length * b_r) - TURN_FLOOR
    delta_q = capacity / (b_f * b_r) * torch.tanh(params[..., 6:7].clamp(-RAW_LIMIT, RAW_LIMIT))

    a_f = b_f * (q + 0.5 * delta_q)
    a_r = b_r * (q - 0.5 * delta_q)
    m_f = center - 0.5 * length * t
    m_r = center + 0.5 * length * t
    s_f = a_f * t + b_f * n
    s_r = a_r * t + b_r * n
    return torch.stack((m_f + s_f, m_f - s_f, m_r - s_r, m_r + s_r), dim=-2)
