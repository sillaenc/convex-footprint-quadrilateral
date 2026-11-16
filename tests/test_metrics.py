import itertools

import numpy as np
import pytest

from cfq import FootprintEvaluator, greedy_match, iou_matrix, nce, polygon_iou
from cfq.metrics import in_ignore_regions

SQUARE = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], np.float64)


def test_iou_sanity():
    assert polygon_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    assert polygon_iou(SQUARE, SQUARE[::-1]) == pytest.approx(1.0)
    assert polygon_iou(SQUARE, SQUARE + [1, 0]) == pytest.approx(2 / 6)
    assert polygon_iou(SQUARE, SQUARE + [1, 1]) == pytest.approx(1 / 7)
    assert polygon_iou(SQUARE, SQUARE + [5, 0]) == 0.0
    assert polygon_iou(SQUARE, SQUARE + [2, 0]) == pytest.approx(0.0, abs=1e-12)
    inner = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])
    assert polygon_iou(SQUARE, inner) == pytest.approx(0.25)
    diamond = np.array([[1, 0], [0, 1], [1, 2], [2, 1]], np.float64)
    assert polygon_iou(SQUARE, diamond) == pytest.approx(0.5)
    bow_tie = SQUARE[[0, 2, 1, 3]]
    assert polygon_iou(bow_tie, SQUARE) == 0.0
    assert polygon_iou(SQUARE, bow_tie) == 0.0


def test_iou_symmetric_and_order_invariant(valid_quads, rng):
    for i in range(200):
        a, b = valid_quads[i], valid_quads[i] + rng.normal(0, 1.0, (4, 2)) * 0.2
        value = polygon_iou(a, b)
        assert 0.0 <= value <= 1.0
        assert polygon_iou(b, a) == pytest.approx(value, abs=1e-9)
        assert polygon_iou(np.roll(a, 1, axis=0), b) == pytest.approx(value, abs=1e-9)


def test_nce_permutation_invariance(valid_quads, rng):
    for quad in valid_quads[:100]:
        prediction = quad + rng.normal(0, 0.5, (4, 2))
        reference = nce(prediction, quad)
        for perm in itertools.permutations(range(4)):
            assert nce(prediction[list(perm)], quad) == pytest.approx(reference, rel=1e-12, abs=1e-15)


def test_nce_values():
    assert nce(SQUARE, SQUARE) == 0.0
    assert nce(SQUARE[[2, 0, 3, 1]], SQUARE) == 0.0
    diagonal = 2 * np.sqrt(2)
    assert nce(SQUARE + [0.1, 0], SQUARE) == pytest.approx(0.1 / diagonal)
    assert nce(SQUARE * 10 + [1, 0], SQUARE * 10) == pytest.approx(nce(SQUARE + [0.1, 0], SQUARE))


def test_greedy_match():
    ious = np.array([[0.9, 0.6], [0.95, 0.0], [0.4, 0.55]])
    assert greedy_match(ious).tolist() == [0, -1, 1]
    assert greedy_match(ious, 0.7).tolist() == [0, -1, -1]
    assert greedy_match(np.zeros((2, 0))).tolist() == [-1, -1]


def test_iou_matrix():
    m = iou_matrix(np.stack([SQUARE, SQUARE + [1, 0]]), np.stack([SQUARE, SQUARE + 10]))
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(1) and m[1, 0] == pytest.approx(1 / 3) and m[:, 1].max() == 0


def test_ignore_regions():
    points = np.array([[1, 1], [2, 1], [3, 3]])
    assert in_ignore_regions(points, SQUARE[None]).tolist() == [True, True, False]
    assert not in_ignore_regions(points, None).any()


def test_evaluator_perfect_and_ignore():
    gts = np.stack([SQUARE, SQUARE + 10, SQUARE + 20])
    ev = FootprintEvaluator()
    ev.update(gts, np.array([0.9, 0.8, 0.7]), gts)
    ap = ev.average_precision()
    assert ap["AP50"] == pytest.approx(1.0) and ap["AP50_95"] == pytest.approx(1.0)
    pm = ev.point_metrics()
    assert pm["precision"] == 1.0 and pm["recall"] == 1.0 and pm["nce"] == 0.0 and pm["iou"] == pytest.approx(1.0)

    ev = FootprintEvaluator()
    bow_tie = (SQUARE + 40)[[0, 2, 1, 3]]
    ignored_fp = SQUARE + 60
    preds = np.stack([gts[0], bow_tie, ignored_fp])
    ev.update(preds, np.array([0.9, 0.8, 0.7]), gts[:1], ignore_quads=(SQUARE * 2 + 59)[None])
    pm = ev.point_metrics()
    assert pm["predictions"] == 2 and pm["tp"] == 1 and pm["invalid_rate"] == 0.5
    assert ev.average_precision()["AP50"] == pytest.approx(1.0)


def test_evaluator_ranking():
    gts = np.stack([SQUARE, SQUARE + 10])
    ev = FootprintEvaluator()
    ev.update(np.stack([SQUARE + 30, gts[0], gts[1]]), np.array([0.9, 0.8, 0.7]), gts)
    # Precision envelope 2/3 from recall 0 to 1.
    assert ev.average_precision()["AP50"] == pytest.approx(2 / 3)
    ev = FootprintEvaluator()
    ev.update(np.stack([gts[0]]), np.array([0.9]), gts, gt_in_frame=[True, False])
    assert ev.num_gt == 1 and ev.average_precision()["AP50"] == pytest.approx(1.0)
