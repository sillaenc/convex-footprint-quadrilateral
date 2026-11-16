"""Numerical constants of the CFQ decoder and encoder.

All lengths are in feature-map units (input pixels divided by the output
stride). The values are the ones used for every experiment in the paper.
"""

import math

#: Relative margin ``mu`` in Eq. (4); keeps ``|K|`` away from ``min(L*b_f, L*b_r)``.
MARGIN = 1e-3
#: Absolute margin ``tau`` in Eq. (4); guarantees ``C >= tau > 0``.
TURN_FLOOR = 1e-2
#: Smallest half-width ``b_f``, ``b_r`` (feature units).
MIN_HALF_WIDTH = 0.1
#: Largest length and half-width (feature units).
MAX_EXTENT = 512.0
#: Largest ratio between a half-width and the length (both ways).
MAX_ASPECT = 256.0
#: Factor of the common-slope bound ``q_max = min(4 L / max(b_f, b_r), 1e4)``.
Q_ASPECT = 4.0
#: Magnitude to which every raw output is clamped before use.
RAW_LIMIT = 1.0e4
#: Direction vectors shorter than this fall back to ``t = (1, 0)``.
DIRECTION_EPS = 1e-6

#: Smallest length ``L``. With it the half-width interval is never empty.
MIN_LENGTH = 2.0 * math.sqrt(TURN_FLOOR / (1.0 - MARGIN))

#: Number of raw geometric outputs ``z = (v_x, v_y, l, rho_f, rho_r, q0, s)``.
NUM_PARAMS = 7
