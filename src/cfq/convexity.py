"""Convexity check, validity of predicted quadrilaterals and the invalid rate.

* :func:`cross_products` returns the four consecutive-edge cross products of
  Eq. (7). The CFQ decoder makes all four strictly positive.
* :func:`is_valid_quad` is the validity rule of the evaluation (Section 3.6):
  a prediction is valid if it is a strictly convex quadrilateral (either
  winding) with no repeated vertex and a non-negligible area. Collinear
  triples, concave vertices and self-intersections are invalid.
* :func:`invalid_rate` is the fraction of invalid quadrilaterals.
* :func:`reorder_by_angle` is the optional repair applied to free 4-point
  outputs: an invalid quad is reconnected in the order of the vertex angles
  around its mean.
"""

from __future__ import annotations

import numpy as np

#: Minimum edge length (pixels) below which two vertices count as repeated.
MIN_EDGE = 1e-4
#: Minimum absolute polygon area (squared pixels).
MIN_AREA = 1e-3


def cross_products(quads) -> np.ndarray:
    """Cross products ``e_k x e_{k+1}`` of consecutive edges, Eq. (7).

    Args:
        quads: ``(..., 4, 2)`` vertices in the order ``p_1 .. p_4``.

    Returns:
        ``(..., 4)`` values ``[e1 x e2, e2 x e3, e3 x e4, e4 x e1]`` with
        ``e_k = p_{k+1} - p_k``. For a CFQ output ``[FL, FR, RR, RL]`` these are
        ``2(L b_f + K), 2(L b_r + K), 2(L b_r - K), 2(L b_f - K)``.
    """

    points = np.asarray(quads)
    edges = np.roll(points, -1, axis=-2) - points
    following = np.roll(edges, -1, axis=-2)
    return edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]


def is_strictly_convex(quads) -> np.ndarray:
    """Whether all four cross products are finite and strictly positive.

    This is the convexity guarantee of Section 3.2: the vertices turn in the
    same (positive, y-down) direction at every corner.
    """

    cross = cross_products(quads)
    return np.isfinite(cross).all(axis=-1) & (cross > 0.0).all(axis=-1)


def _turns_consistently(points: np.ndarray) -> np.ndarray:
    # Strict turn-sign test over consecutive edges, evaluated in float32 like
    # the contour convexity test of common image libraries. A zero turn
    # (collinear triple) is not convex.
    edges = np.roll(points, -1, axis=-2) - points
    previous = np.roll(edges, 1, axis=-2)
    lhs = edges[..., 1] * previous[..., 0]
    rhs = edges[..., 0] * previous[..., 1]
    positive = (lhs > rhs).all(axis=-1)
    negative = (lhs < rhs).all(axis=-1)
    return positive | negative


def polygon_area(quads) -> np.ndarray:
    """Signed shoelace area of ``(..., N, 2)`` polygons (float64)."""

    points = np.asarray(quads, dtype=np.float64)
    nxt = np.roll(points, -1, axis=-2)
    return 0.5 * (points[..., 0] * nxt[..., 1] - points[..., 1] * nxt[..., 0]).sum(axis=-1)


def is_valid_quad(quads) -> np.ndarray:
    """Validity of ``(..., 4, 2)`` quadrilaterals used by the evaluation.

    A quad is valid if its vertices are finite, consecutive vertices are more
    than :data:`MIN_EDGE` apart, its absolute area exceeds :data:`MIN_AREA`,
    and it is strictly convex in either winding. Vertices are rounded to
    float32 first, as in the evaluation of the paper.
    """

    points = np.asarray(quads, dtype=np.float32)
    if points.shape[-2:] != (4, 2):
        raise ValueError("quads must have shape (..., 4, 2)")
    finite = np.isfinite(points).all(axis=(-1, -2))
    safe = np.where(finite[..., None, None], points, np.float32(0.0))
    edge = np.linalg.norm(safe - np.roll(safe, -1, axis=-2), axis=-1)
    separated = (edge > MIN_EDGE).all(axis=-1)
    area = np.abs(polygon_area(safe)) > MIN_AREA
    return finite & separated & area & _turns_consistently(safe)


def invalid_rate(quads) -> float:
    """Fraction of ``(N, 4, 2)`` quads that are not valid (0 for N = 0)."""

    valid = is_valid_quad(np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2))
    return float(1.0 - valid.mean()) if valid.size else 0.0


def reorder_by_angle(quad) -> tuple[np.ndarray, bool]:
    """Reconnect an invalid quad in the order of its angles around the mean.

    Valid quads are returned unchanged. For an invalid one, the eight
    adjacency-preserving orderings of the angle-sorted vertices (four starting
    points, two windings) are tried, and the valid candidate closest to the
    input (sum of squared vertex displacements, ties broken by candidate
    index) is returned.

    Returns:
        ``(quad, ok)`` where ``ok`` is False if no valid ordering exists.
    """

    original = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if is_valid_quad(original):
        return original.copy(), True
    if not np.isfinite(original).all():
        return original.copy(), False
    centre = original.mean(axis=0)
    angles = np.arctan2(original[:, 1] - centre[1], original[:, 0] - centre[0])
    base = np.argsort(angles, kind="stable")
    best, best_cost = None, None
    for winding in (base, base[::-1]):
        for offset in range(4):
            candidate = original[np.roll(winding, -offset)]
            if not is_valid_quad(candidate):
                continue
            cost = float(np.square(candidate - original).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = candidate, cost
    if best is None:
        return original.copy(), False
    return best.copy(), True
