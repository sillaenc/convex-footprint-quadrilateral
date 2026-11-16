# CFQ: Convex Footprint Quadrilateral

This repository contains the reference implementation of the **Convex Footprint
Quadrilateral (CFQ)** representation and the evaluation metrics from the paper
*Convex Footprint Quadrilateral: Axis-Based Amodal Vehicle Footprint Detection
in Parking Surveillance*.

CFQ describes the ground footprint of a vehicle in the image as a quadrilateral
`[FL, FR, RR, RL]`. A detector predicts the center `c` (mean of the four
vertices) and seven raw outputs `z = (v_x, v_y, l, rho_f, rho_r, q0, s)`: a
direction vector for the vehicle axis, the log length between the front and
rear edge midpoints, the log ratios of the front and rear half-widths to the
length, a common edge slope and a slope difference. The decoder builds the four
vertices from these values with bounded, smooth transformations.

The bound on the slope difference makes every decoded quadrilateral strictly
convex, for any finite network output. The four cross products of consecutive
edges are `2(L b_f ± K)` and `2(L b_r ± K)` with `|K| < min(L b_f, L b_r)`, so
all of them are positive. No vertex reordering, convex hull or post-hoc repair
is needed. The decoder is piecewise differentiable, so a corner loss on the
decoded vertices trains the center and the shape parameters end to end.

The package contains:

- `cfq.decode`: the decoder, in NumPy and differentiable PyTorch
- `cfq.encode`: the inverse encoder for ground-truth footprints, including the
  rule for footprints that cannot be encoded
- `cfq.convexity`: the cross-product convexity check, the validity rule of
  the evaluation, the invalid rate, and the optional angle reordering used for
  free 4-point outputs
- `cfq.loss`: the order-invariant corner loss (PyTorch)
- `cfq.metrics`: polygon IoU, the order-invariant normalized corner error
  (NCE), greedy matching, and an evaluator for AP50, AP50:95, NCE, IoU and the
  invalid rate

It does not contain training code, the detector, the synthetic-data generator,
datasets, model weights, or deployment code.

## Installation

```bash
pip install .            # NumPy only: decoder, encoder, convexity, metrics
pip install ".[torch]"   # adds the PyTorch decoder and the corner loss
```

Python 3.9 or later and NumPy are required. PyTorch is optional. Polygon IoU
uses a self-contained convex-polygon clipping routine and has no further
dependencies.

## Usage

All CFQ geometry is in feature-map units: input-image pixels divided by the
output stride (4 in the paper).

```python
import numpy as np
from cfq import decode_cfq, encode_cfq, canonicalize_orientation, cross_products

# Decode one location: center (grid cell + offset) and seven raw outputs.
center = np.array([120.3, 60.7])
z = np.array([0.8, 0.6, np.log(12.0), np.log(0.25), np.log(0.20), 0.1, -0.2])
quad = decode_cfq(center, z) * 4              # (4, 2) FL, FR, RR, RL in pixels
assert (cross_products(quad) > 0).all()       # strictly convex, Eq. (7)

# Encode a ground-truth footprint given in pixels.
gt = canonicalize_orientation(np.array([[[400., 260.], [430., 300.], [520., 250.], [490., 210.]]]))[0]
gt_center, gt_z = encode_cfq(gt, stride=4)    # raises CFQEncodingError if not representable
assert np.allclose(decode_cfq(gt_center, gt_z) * 4, gt)
```

Both `decode_cfq` and `decode_cfq_torch` take arrays with any leading shape:
`(..., 2)` centers and `(..., 7)` parameters give `(..., 4, 2)` vertices.

Training loss (PyTorch):

```python
from cfq.decode import decode_cfq_torch
from cfq.loss import corner_loss

pred = decode_cfq_torch(center_t, params_t)       # (N, 4, 2), feature units
loss = corner_loss(pred, gt_vertices_t, mask_t)   # Eq. (9); the GT vertex order does not matter
```

Metrics:

```python
from cfq import polygon_iou, nce, invalid_rate, FootprintEvaluator

polygon_iou(pred_quad, gt_quad)    # 0 if either quad is not a valid convex quad
nce(pred_quad, gt_quad)            # Eq. (10), minimum over the 24 vertex correspondences
invalid_rate(pred_quads)           # fraction of non-convex / degenerate quads

ev = FootprintEvaluator(max_detections=100)
for image in dataset:              # coordinates in original-image pixels
    ev.update(image.pred_quads, image.scores, image.gt_quads,
              gt_in_frame=image.gt_mask, ignore_quads=image.ignore_quads)
ev.average_precision()             # {'AP50', 'AP50_95', 'per_threshold'}
ev.point_metrics(score_floor=0.05) # precision, recall, NCE, IoU, invalid rate
```

`examples/basic_usage.py` runs all of the above.

## Correspondence with the paper

| Paper | Code |
|---|---|
| Eq. (1): `t = v/‖v‖₂` (fallback `(1, 0)` if `‖v‖₂ < 1e-6`), `n = (−t_y, t_x)`, `L = exp(l)`, `b = L exp(ρ)` | `decode_cfq`, `decode_cfq_torch` |
| Eq. (2): midpoints `m_f, m_r`, half-vectors `s_f, s_r`, `a_f = b_f(q + Δq/2)`, `a_r = b_r(q − Δq/2)` | `decode_cfq`, `decode_cfq_torch` |
| Eq. (3): vertex order `FL, FR, RR, RL` | `decode_cfq`, `decode_cfq_torch` |
| Eq. (4): `q = q_max tanh(q0)`, `C = (1−μ) min(L b_f, L b_r) − τ`, `Δq = C/(b_f b_r) tanh(s)` | `decode_cfq`, `decode_cfq_torch` |
| Constants `μ = 1e−3`, `τ = 1e−2`, `q_max = min(4L/max(b_f, b_r), 1e4)` | `cfq.constants` (`MARGIN`, `TURN_FLOOR`, `Q_ASPECT`, `RAW_LIMIT`) |
| Sec. 3.1 limits: `L ∈ [2√(τ/(1−μ)), 512]`, `b_f, b_r ∈ [max(0.1, L/256, 2τ/((1−μ)L)), min(512, 256L)]` | `cfq.constants` (`MIN_LENGTH`, `MAX_EXTENT`, `MIN_HALF_WIDTH`, `MAX_ASPECT`) |
| Eqs. (5)–(7): cross products of consecutive edges | `cross_products`, `is_strictly_convex` |
| Sec. 3.4: GT → parameters by inverting the decoder; GTs that cannot be encoded are ignored in training | `encode_cfq`, `is_encodable`, `CFQEncodingError` |
| Eq. (9): diagonal-normalized Smooth L1 (`β = 0.05`), minimum over the eight orderings of D₄ | `cfq.loss.corner_cost`, `corner_loss`, `D4_ORDERS` |
| Eq. (10): NCE, minimum over 24 correspondences, normalized by the mean GT diagonal | `cfq.metrics.nce` |
| Sec. 3.6: polygon IoU (0 for invalid predictions), one-to-one greedy matching, AP50, AP50:95 | `polygon_iou`, `iou_matrix`, `greedy_match`, `FootprintEvaluator` |
| Sec. 3.6: invalid rate (three collinear vertices or not a convex quad) | `is_valid_quad`, `invalid_rate` |
| Sec. 3.6: reconnection of free 4-point outputs by vertex angle | `reorder_by_angle` |

Notes on the implementation:

- Raw outputs are clamped to `±1e4`, and `l`, `ρ_f` and `ρ_r` are clamped to
  the ranges that the limits above imply. As a result, every finite input
  decodes to finite vertices. The PyTorch decoder computes in float32 (half
  precision inputs are upcast) unless both inputs are float64.
- The encoder accepts a quad only if it has positive signed area in y-down
  image coordinates, `L` and both half-widths lie within the decoder limits
  (half-widths strictly above 0.1), and `|q| < q_max` and `|K| < C` hold
  strictly. For every accepted quad, the decode–encode round trip is exact up
  to floating-point rounding. `canonicalize_orientation` maps a footprint
  labelled with the opposite winding to the expected one by swapping its left
  and right vertices.
- `FootprintEvaluator` follows the protocol of the paper. Per image, at most
  100 predictions are ranked by score. An unmatched prediction whose center
  lies in an ignore polygon is neither a true nor a false positive. NCE and
  IoU are averaged over the true positives at IoU 0.5. The invalid rate is
  computed over the counted predictions (true positives and non-ignored false
  positives) with score ≥ 0.05.

## Tests

```bash
pip install ".[test]"
pytest -q
```

The tests check:

- the exact encode–decode round trip, on random footprints and edge cases
  (rectangles, trapezoids, parallelograms, very narrow, large and
  near-triangular footprints)
- the rejection rule for footprints that cannot be encoded
- positivity of all four cross products for 100,000 random and extreme raw
  outputs (values up to `±1e30`, zero direction vectors) in float32 and
  float64, for an exhaustive grid of extreme values, and for the PyTorch
  decoder
- D₄ permutation invariance of the corner loss and its agreement with Eq. (9)
- invariance of NCE to all 24 vertex permutations
- polygon IoU on cases with known values, the validity rule, and the evaluator

## Citation

```bibtex
@article{kim2026cfq,
  title   = {Convex Footprint Quadrilateral: Axis-Based Amodal Vehicle Footprint Detection in Parking Surveillance},
  author  = {Kim, NamJu and Kang, Jae-Mo},
  journal = {Machine Learning and Knowledge Extraction},
  publisher = {MDPI},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License. Copyright (c) 2026 SILLA ENC Co., Ltd. See [LICENSE](LICENSE).
