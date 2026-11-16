"""Minimal CFQ usage: decode, encode, check convexity, and evaluate."""

import numpy as np

from cfq import (
    FootprintEvaluator,
    canonicalize_orientation,
    cross_products,
    decode_cfq,
    encode_cfq,
    invalid_rate,
    nce,
    polygon_iou,
)

STRIDE = 4

# 1. Decode raw network outputs at one location. The center is in feature
#    units (grid cell + predicted offset); z = (v_x, v_y, l, rho_f, rho_r, q0, s).
center = np.array([120.3, 60.7])
z = np.array([0.8, 0.6, np.log(12.0), np.log(0.25), np.log(0.20), 0.1, -0.2])
quad = decode_cfq(center, z) * STRIDE  # [FL, FR, RR, RL] in input pixels
print("decoded vertices:\n", quad)
print("cross products (all > 0):", cross_products(quad))

# 2. Encode a ground-truth footprint given in pixels. The encoder expects a
#    positive y-down signed area; canonicalize_orientation swaps left/right
#    vertices of a footprint labelled with the other winding.
gt = np.array([[400.0, 260.0], [430.0, 300.0], [520.0, 250.0], [490.0, 210.0]])
gt = canonicalize_orientation(gt[None])[0].astype(np.float64)
gt_center, gt_z = encode_cfq(gt, stride=STRIDE)
print("encoded parameters:", gt_z)
print("round-trip error (px):", np.abs(decode_cfq(gt_center, gt_z) * STRIDE - gt).max())

# 3. Metrics between one prediction and one GT.
prediction = gt + np.array([[2.0, -1.0], [1.0, 1.5], [-1.0, 0.5], [0.0, -2.0]])
print("IoU:", polygon_iou(prediction, gt))
print("NCE:", nce(prediction[[2, 3, 0, 1]], gt))  # vertex order does not matter
print("invalid rate:", invalid_rate(np.stack([prediction, prediction[[0, 2, 1, 3]]])))

# 4. Dataset-level evaluation (one image shown).
evaluator = FootprintEvaluator(max_detections=100)
evaluator.update(pred_quads=np.stack([prediction]), scores=np.array([0.9]), gt_quads=gt[None])
print(evaluator.average_precision()["AP50"], evaluator.point_metrics(score_floor=0.05))
