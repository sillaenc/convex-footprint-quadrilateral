import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cfq.decode import decode_cfq_torch  # noqa: E402
from cfq.loss import D4_ORDERS, corner_cost, corner_loss  # noqa: E402


def _batch(seed=0, n=512):
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(n, 2, generator=g, dtype=torch.float64) * 100
    params = torch.randn(n, 7, generator=g, dtype=torch.float64)
    params[:, 2] += 2.5
    prediction = decode_cfq_torch(centers, params)
    target = prediction.detach() + torch.randn(n, 4, 2, generator=g, dtype=torch.float64)
    return prediction, target


def test_d4_orders():
    assert len(set(D4_ORDERS)) == 8
    for order in D4_ORDERS:
        # Adjacent vertices stay adjacent.
        positions = {v: i for i, v in enumerate(order)}
        for k in range(4):
            assert (positions[(k + 1) % 4] - positions[k]) % 4 in (1, 3)


def test_permutation_invariance():
    prediction, target = _batch()
    reference = corner_cost(prediction, target)
    for order in D4_ORDERS:
        assert torch.allclose(corner_cost(prediction, target[:, order]), reference, rtol=0, atol=1e-15)
        assert torch.allclose(corner_cost(prediction[:, order], target), reference, rtol=0, atol=1e-15)


def test_not_invariant_to_non_adjacent_permutation():
    prediction, target = _batch()
    swapped = corner_cost(prediction, target[:, [0, 2, 1, 3]])
    assert (swapped > corner_cost(prediction, target) + 1e-6).any()


def test_zero_at_target_and_scale_invariance():
    prediction, target = _batch()
    assert corner_cost(target, target[:, [2, 3, 0, 1]]).abs().max() == 0
    small = corner_cost(prediction, target)
    large = corner_cost(prediction * 10, target * 10)
    assert torch.allclose(small, large, rtol=1e-12, atol=1e-15)


def test_matches_explicit_formula():
    prediction, target = _batch(n=64)
    diag = 0.5 * ((target[:, 2] - target[:, 0]).norm(dim=-1) + (target[:, 3] - target[:, 1]).norm(dim=-1))

    def huber(x, beta=0.05):
        return torch.where(x.abs() < beta, 0.5 * x * x / beta, x.abs() - 0.5 * beta)

    expected = torch.stack(
        [huber((prediction - target[:, list(o)]) / diag[:, None, None]).sum(dim=(-1, -2)) / 8 for o in D4_ORDERS]
    ).min(dim=0).values
    assert torch.allclose(corner_cost(prediction, target), expected, rtol=1e-12, atol=1e-15)


def test_mask_and_gradient():
    prediction, target = _batch(n=32)
    mask = torch.zeros(32, dtype=torch.float64)
    mask[:5] = 1
    loss = corner_loss(prediction, target, mask)
    assert torch.allclose(loss, corner_cost(prediction, target)[:5].mean())
    assert corner_loss(prediction, target, torch.zeros(32)).item() == 0.0
    centers = torch.zeros(3, 2, requires_grad=True)
    params = torch.zeros(3, 7, requires_grad=True)
    corner_loss(decode_cfq_torch(centers, params), torch.ones(3, 4, 2) * torch.tensor([[0.0, 0], [0, 1], [2, 1], [2, 0]])).backward()
    assert torch.isfinite(params.grad).all() and torch.isfinite(centers.grad).all()
    assert params.grad.abs().sum() > 0
