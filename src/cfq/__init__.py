"""CFQ: Convex Footprint Quadrilateral.

NumPy functions are exported here. The PyTorch parts are imported explicitly:
``from cfq.decode import decode_cfq_torch`` and ``from cfq.loss import corner_loss``.
"""

from .convexity import cross_products, invalid_rate, is_strictly_convex, is_valid_quad, reorder_by_angle
from .decode import decode_cfq
from .encode import CFQEncodingError, canonicalize_orientation, encode_cfq, is_encodable
from .metrics import FootprintEvaluator, greedy_match, iou_matrix, nce, polygon_iou

__version__ = "1.0.0"

__all__ = [
    "CFQEncodingError",
    "FootprintEvaluator",
    "canonicalize_orientation",
    "cross_products",
    "decode_cfq",
    "encode_cfq",
    "greedy_match",
    "invalid_rate",
    "iou_matrix",
    "is_encodable",
    "is_strictly_convex",
    "is_valid_quad",
    "nce",
    "polygon_iou",
    "reorder_by_angle",
]
