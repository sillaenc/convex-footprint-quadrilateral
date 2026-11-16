"""Order-invariant corner loss, Eq. (9) (requires PyTorch).

For each regression location the decoded vertices are compared with the GT
vertices under each of the eight adjacency-preserving orderings of the GT
(the dihedral group D4: four cyclic shifts and their reversals), and the
smallest cost is used. The cost of one ordering is the Smooth L1 loss with
``beta = 0.05`` of the coordinate differences divided by the mean length of
the two GT diagonals, averaged over the eight coordinates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: The eight orderings of D4: cyclic shifts, then their reversals.
D4_ORDERS: tuple[tuple[int, int, int, int], ...] = tuple(
    tuple((k + i) % 4 for i in range(4)) for k in range(4)
) + tuple(tuple((k - i) % 4 for i in range(4)) for k in range(4))

#: Smooth L1 transition point used in the paper.
BETA = 0.05


def corner_cost(prediction: torch.Tensor, target: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    """Per-location order-invariant corner cost.

    Args:
        prediction: ``(..., 4, 2)`` decoded vertices.
        target: ``(..., 4, 2)`` GT vertices; ``target[..., 0, :], target[..., 2, :]``
            and ``target[..., 1, :], target[..., 3, :]`` are its diagonals.
        beta: Smooth L1 transition point.

    Returns:
        ``(...)`` cost: ``min over D4 of (1/8) sum_k sum_a h_beta(dp / d)``.
    """

    target = target.to(prediction.dtype)
    diagonal = (
        0.5
        * (
            torch.linalg.vector_norm(target[..., 2, :] - target[..., 0, :], dim=-1)
            + torch.linalg.vector_norm(target[..., 3, :] - target[..., 1, :], dim=-1)
        )
    ).detach().clamp_min(1e-4)
    best = None
    for order in D4_ORDERS:
        difference = (prediction - target[..., order, :]) / diagonal[..., None, None]
        cost = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none", beta=beta).mean(
            dim=(-1, -2)
        )
        best = cost if best is None else torch.minimum(best, cost)
    return best


def corner_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = BETA,
) -> torch.Tensor:
    """Corner loss of Eq. (9): mean of :func:`corner_cost` over valid locations.

    Args:
        prediction: ``(..., 4, 2)`` decoded vertices.
        target: ``(..., 4, 2)`` GT vertices.
        mask: optional ``(...)`` weights (1 for a valid regression location,
            0 otherwise). Without a mask every location is used.
        beta: Smooth L1 transition point.

    Returns:
        Scalar ``sum(mask * cost) / max(sum(mask), 1)``.
    """

    cost = corner_cost(prediction, target, beta)
    if mask is None:
        mask = torch.ones_like(cost)
    mask = mask.to(cost.dtype)
    return (cost * mask).sum() / mask.sum().clamp_min(1.0)
