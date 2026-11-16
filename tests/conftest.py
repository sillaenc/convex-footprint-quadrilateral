import numpy as np
import pytest

from cfq import decode_cfq


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_params(rng, n):
    """Raw CFQ outputs that decode to typical vehicle-like footprints."""
    return np.column_stack(
        (
            rng.normal(0.0, 1.0, (n, 2)),
            rng.normal(2.5, 1.0, n),
            rng.normal(-0.7, 0.6, (n, 2)),
            rng.normal(0.0, 0.8, (n, 2)),
        )
    )


@pytest.fixture
def valid_quads(rng):
    """Decoded CFQ quads in feature units (all strictly convex)."""
    n = 2000
    centers = rng.uniform((0.0, 0.0), (320.0, 180.0), size=(n, 2))
    return decode_cfq(centers, random_params(rng, n))
