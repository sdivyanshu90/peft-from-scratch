"""Learning-rate schedulers: linear-decay-with-warmup and cosine-with-warmup.

Both follow the standard "warm up linearly, then decay" recipe used for
fine-tuning transformers. They are thin :class:`~torch.optim.lr_scheduler.LambdaLR`
factories so they compose with any optimizer and with gradient accumulation
(``scheduler.step()`` is called once per optimizer step).
"""

from __future__ import annotations

import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from peft_lib.core.exceptions import ConfigError

__all__ = [
    "SchedulerName",
    "build_scheduler",
    "get_linear_schedule_with_warmup",
    "get_warmup_cosine_schedule",
]

SchedulerName = Literal["cosine", "linear", "constant", "none"]


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup for ``num_warmup_steps`` then linear decay to 0.

    Args:
        optimizer: The optimizer whose base ``lr`` is scaled.
        num_warmup_steps: Steps to ramp the multiplier 0 -> 1.
        num_training_steps: Total steps; the multiplier hits 0 at the end.
        last_epoch: Index of the last step (``-1`` to start fresh).

    Returns:
        A :class:`LambdaLR` scheduler.

    Example:
        >>> import torch
        >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        >>> sched = get_linear_schedule_with_warmup(opt, 2, 10)
        >>> round(opt.param_groups[0]["lr"], 3)  # step 0: 0/2 of base lr
        0.0
    """

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        return max(0.0, remaining / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the base lr.

    Args:
        optimizer: The optimizer whose base ``lr`` is scaled.
        num_warmup_steps: Steps to ramp the multiplier 0 -> 1.
        num_training_steps: Total steps.
        num_cycles: Number of cosine half-cycles (0.5 = a single decay to the
            minimum, the usual choice).
        min_lr_ratio: Floor multiplier (e.g. 0.1 keeps 10% of the base lr).
        last_epoch: Index of the last step.

    Returns:
        A :class:`LambdaLR` scheduler.

    Example:
        >>> import torch
        >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        >>> sched = get_warmup_cosine_schedule(opt, 0, 10)
        >>> round(opt.param_groups[0]["lr"], 4)  # step 0, no warmup -> full lr
        1.0
    """
    if not 0.0 <= min_lr_ratio < 1.0:
        raise ConfigError(f"min_lr_ratio must be in [0, 1), got {min_lr_ratio}.")

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, cosine)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_scheduler(
    name: SchedulerName,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR | None:
    """Construct a scheduler by name (used by :class:`PEFTTrainer`).

    Args:
        name: ``"cosine"``, ``"linear"``, ``"constant"``, or ``"none"``.
        optimizer: The optimizer to schedule.
        num_warmup_steps: Warmup steps.
        num_training_steps: Total steps.

    Returns:
        A :class:`LambdaLR`, or ``None`` when ``name`` is ``"none"``.

    Raises:
        ConfigError: If ``name`` is unknown.

    Example:
        >>> import torch
        >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-3)
        >>> build_scheduler("none", opt, 0, 10) is None
        True
    """
    if name == "none":
        return None
    if name == "cosine":
        return get_warmup_cosine_schedule(optimizer, num_warmup_steps, num_training_steps)
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if name == "constant":
        return LambdaLR(
            optimizer,
            lambda step: 1.0 if step >= num_warmup_steps else step / max(1, num_warmup_steps),
        )
    raise ConfigError(f"Unknown scheduler {name!r}; choose cosine|linear|constant|none.")
