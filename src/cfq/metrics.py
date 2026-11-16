"""Evaluation metrics of Section 3.6.

* :func:`polygon_iou` -- IoU of two footprint polygons; 0 if either is invalid.
* :func:`nce` -- order-invariant normalized corner error, Eq. (10): the mean
  vertex distance under the best of the 24 one-to-one correspondences, divided
  by the mean length of the two GT diagonals.
* :func:`greedy_match` -- score-descending greedy one-to-one matching on IoU.
* :class:`FootprintEvaluator` -- COCO-style 101-point AP50 and AP50:95 with
  ignore regions, and NCE, IoU and invalid rate over the true positives and
  predictions above a score floor.

Everything is self-contained NumPy; the convex-polygon intersection uses
Sutherland-Hodgman clipping.
"""

from __future__ import annotations

import itertools

import numpy as np

from .convexity import is_valid_quad, polygon_area

#: All 4! = 24 one-to-one assignments of four predicted vertices to GT vertices.
PERMUTATIONS_24 = np.asarray(list(itertools.permutations(range(4))))


def _diagonal(gt: np.ndarray) -> float:
    return max(float((np.linalg.norm(gt[0] - gt[2]) + np.linalg.norm(gt[1] - gt[3])) / 2.0), 1e-6)


def nce(prediction, gt) -> float:
    """Order-invariant normalized corner error, Eq. (10).

    Args:
        prediction: ``(4, 2)`` predicted vertices in any order.
        gt: ``(4, 2)`` ground-truth vertices; ``gt[0], gt[2]`` and
            ``gt[1], gt[3]`` are its diagonals.
    """

    p = np.asarray(prediction, dtype=np.float64).reshape(4, 2)
    g = np.asarray(gt, dtype=np.float64).reshape(4, 2)
    distances = np.linalg.norm(p[None] - g[PERMUTATIONS_24], axis=-1).mean(axis=-1)
    return float(distances.min() / _diagonal(g))


def _positive_winding(points: np.ndarray) -> np.ndarray:
    return points if polygon_area(points) >= 0.0 else points[::-1]


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Intersection of two convex polygons with positive winding."""

    output = subject
    for i in range(len(clipper)):
        if not len(output):
            break
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        side = edge[0] * (output[:, 1] - a[1]) - edge[1] * (output[:, 0] - a[0])
        points = []
        for j in range(len(output)):
            cur, nxt = output[j], output[(j + 1) % len(output)]
            s_cur, s_nxt = side[j], side[(j + 1) % len(output)]
            if s_cur >= 0.0:
                points.append(cur)
            if (s_cur >= 0.0) != (s_nxt >= 0.0):
                points.append(cur + (s_cur / (s_cur - s_nxt)) * (nxt - cur))
        output = np.asarray(points).reshape(-1, 2)
    return output


def polygon_iou(prediction, gt) -> float:
    """IoU of two quadrilaterals; 0 if either is not a valid convex quad."""

    p = np.asarray(prediction, dtype=np.float32).reshape(4, 2)
    g = np.asarray(gt, dtype=np.float32).reshape(4, 2)
    if not (is_valid_quad(p) and is_valid_quad(g)):
        return 0.0
    p = _positive_winding(p.astype(np.float64))
    g = _positive_winding(g.astype(np.float64))
    inter = _clip(p, g)
    intersection = abs(float(polygon_area(inter))) if len(inter) >= 3 else 0.0
    union = abs(float(polygon_area(p))) + abs(float(polygon_area(g))) - intersection
    return intersection / union if union > 1e-6 else 0.0


def iou_matrix(predictions, gts) -> np.ndarray:
    """``(P, G)`` polygon IoU between ``(P, 4, 2)`` predictions and ``(G, 4, 2)`` GTs."""

    p = np.asarray(predictions, dtype=np.float64).reshape(-1, 4, 2)
    g = np.asarray(gts, dtype=np.float64).reshape(-1, 4, 2)
    ious = np.zeros((len(p), len(g)))
    if not len(p) or not len(g):
        return ious
    # Disjoint bounding boxes cannot intersect; this prefilter is exact.
    p_lo, p_hi = p.min(axis=1), p.max(axis=1)
    g_lo, g_hi = g.min(axis=1), g.max(axis=1)
    overlap = (
        (p_lo[:, None, 0] <= g_hi[None, :, 0])
        & (g_lo[None, :, 0] <= p_hi[:, None, 0])
        & (p_lo[:, None, 1] <= g_hi[None, :, 1])
        & (g_lo[None, :, 1] <= p_hi[:, None, 1])
    )
    for i, j in zip(*np.nonzero(overlap)):
        ious[i, j] = polygon_iou(p[i], g[j])
    return ious


def greedy_match(ious, threshold: float = 0.5) -> np.ndarray:
    """Greedy one-to-one matching of score-sorted predictions to GTs.

    Args:
        ious: ``(P, G)`` IoU matrix whose rows are sorted by descending score.
        threshold: minimum IoU of a match.

    Returns:
        ``(P,)`` matched GT index per prediction, -1 if unmatched. Each
        prediction takes the unmatched GT with the highest IoU (lowest index
        on ties).
    """

    ious = np.asarray(ious)
    matched = np.full(len(ious), -1, dtype=np.int64)
    used = np.zeros(ious.shape[1] if ious.ndim == 2 else 0, dtype=bool)
    for i in range(len(ious)):
        candidates = np.flatnonzero((ious[i] >= threshold) & ~used)
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            used[j] = True
            matched[i] = j
    return matched


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Even-odd test; points on the boundary count as inside."""

    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross == 0.0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_ignore_regions(points, ignore_quads) -> np.ndarray:
    """Whether each ``(N, 2)`` point lies inside (or on) any ignore polygon."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hit = np.zeros(len(points), dtype=bool)
    if ignore_quads is None:
        return hit
    for polygon in np.asarray(ignore_quads, dtype=np.float32).astype(np.float64).reshape(-1, 4, 2):
        for k in np.flatnonzero(~hit):
            hit[k] = _point_in_polygon(points[k], polygon)
    return hit


def _ap101(tp: np.ndarray, num_gt: int) -> float:
    true = np.cumsum(tp, dtype=np.float64)
    false = np.cumsum(~tp, dtype=np.float64)
    recall = true / num_gt
    precision = true / np.maximum(true + false, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    sampled = np.zeros(101)
    valid = indices < len(envelope)
    sampled[valid] = envelope[indices[valid]]
    return float(sampled.mean())


class FootprintEvaluator:
    """Accumulates per-image results and reports AP, NCE, IoU and invalid rate.

    Protocol (Section 3.6): per image, at most ``max_detections`` predictions
    are ranked by score and greedily matched one to one to GTs by polygon IoU.
    An unmatched prediction whose center (mean of its vertices) falls in an
    ignore polygon is neither a true nor a false positive. Invalid predictions
    have IoU 0 and cannot match. AP is the 101-point interpolated AP per IoU
    threshold. NCE and IoU are averaged over the true positives at IoU 0.5
    with score at least ``score_floor``; the invalid rate is the fraction of
    invalid quads among the counted (non-ignored) predictions above the floor.
    """

    def __init__(self, max_detections: int = 100, iou_thresholds=None):
        self.max_detections = int(max_detections)
        self.iou_thresholds = (
            np.linspace(0.5, 0.95, 10) if iou_thresholds is None else np.asarray(iou_thresholds, dtype=np.float64)
        )
        if not np.any(np.isclose(self.iou_thresholds, 0.5)):
            raise ValueError("iou_thresholds must contain 0.5")
        self.num_gt = 0
        self._scores, self._tp, self._ignored = [], [], []
        # Per counted prediction at IoU 0.5: score, tp, nce, iou, invalid.
        self._records = []

    def update(self, pred_quads, scores, gt_quads, gt_in_frame=None, ignore_quads=None):
        """Add one image.

        Args:
            pred_quads: ``(P, 4, 2)`` predicted vertices.
            scores: ``(P,)`` confidence scores.
            gt_quads: ``(G, 4, 2)`` ground-truth vertices.
            gt_in_frame: optional ``(G,)`` mask of the GTs that are evaluated;
                the others are dropped.
            ignore_quads: optional ``(R, 4, 2)`` ignore polygons.
        """

        quads = np.asarray(pred_quads, dtype=np.float32).reshape(-1, 4, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(quads):
            raise ValueError("scores and pred_quads differ in length")
        order = np.argsort(-scores, kind="stable")[: self.max_detections]
        quads, scores = quads[order], scores[order]
        gts = np.asarray(gt_quads, dtype=np.float32).reshape(-1, 4, 2)
        if gt_in_frame is not None:
            gts = gts[np.asarray(gt_in_frame, dtype=bool)]
        self.num_gt += len(gts)

        ious = iou_matrix(quads, gts)
        in_ignore = in_ignore_regions(quads.mean(axis=1), ignore_quads)
        tp = np.zeros((len(self.iou_thresholds), len(quads)), dtype=bool)
        ignored = np.zeros_like(tp)
        for k, threshold in enumerate(self.iou_thresholds):
            matched = greedy_match(ious, threshold)
            tp[k] = matched >= 0
            ignored[k] = (matched < 0) & in_ignore
            if np.isclose(threshold, 0.5):
                for i, j in enumerate(matched):
                    if j >= 0:
                        self._records.append((scores[i], 1.0, nce(quads[i], gts[j]), ious[i, j], 0.0))
                    elif not in_ignore[i]:
                        self._records.append((scores[i], 0.0, 0.0, 0.0, float(not is_valid_quad(quads[i]))))
        self._scores.append(scores)
        self._tp.append(tp)
        self._ignored.append(ignored)

    def average_precision(self) -> dict:
        """AP per IoU threshold, AP50 and AP50:95."""

        if not self.num_gt:
            return {"AP50": None, "AP50_95": None, "per_threshold": {}}
        scores = np.concatenate(self._scores) if self._scores else np.empty(0)
        tp = np.concatenate(self._tp, axis=1) if self._tp else np.zeros((len(self.iou_thresholds), 0), bool)
        ignored = np.concatenate(self._ignored, axis=1) if self._ignored else np.zeros_like(tp)
        order = np.argsort(-scores, kind="stable")
        per_threshold = {}
        for k, threshold in enumerate(self.iou_thresholds):
            keep = order[~ignored[k, order]]
            per_threshold[round(float(threshold), 2)] = _ap101(tp[k, keep], self.num_gt)
        ap50 = per_threshold[0.5]
        return {"AP50": ap50, "AP50_95": float(np.mean(list(per_threshold.values()))), "per_threshold": per_threshold}

    def point_metrics(self, score_floor: float = 0.05) -> dict:
        """Precision, recall, mean NCE, mean IoU and invalid rate above a score floor."""

        records = np.asarray(self._records, dtype=np.float64).reshape(-1, 5)
        records = records[records[:, 0] >= score_floor]
        tp = records[:, 1].sum()
        count = len(records)
        return {
            "predictions": count,
            "tp": int(tp),
            "precision": float(tp / count) if count else 0.0,
            "recall": float(tp / self.num_gt) if self.num_gt else 0.0,
            "nce": float(records[:, 2].sum() / tp) if tp else None,
            "iou": float(records[:, 3].sum() / tp) if tp else None,
            "invalid_rate": float(records[:, 4].sum() / count) if count else 0.0,
        }
