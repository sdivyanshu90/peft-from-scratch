r"""Model Soup: average the weights of several fine-tuned models (Wortsman et al., 2022).

================================================================================
INTUITION
================================================================================
Fine-tuning the same pretrained model with different hyperparameters/seeds yields
models in the same loss basin. Averaging their weights ("souping") often beats
any single member and costs nothing at inference (one model, no ensembling).

This module averages **state dicts** (or adapter state dicts), supporting a
uniform mean and an arbitrary weighted mean:

        theta_soup = sum_i w_i * theta_i ,   sum_i w_i = 1 (uniform: w_i = 1/n).

Only tensors present in *all* inputs with matching shape are averaged; others are
reported as an error so silent mismatches never slip through.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from peft_lib.core.exceptions import MergeError

__all__ = ["uniform_soup", "weighted_soup"]


def _validate(state_dicts: Sequence[dict[str, torch.Tensor]]) -> list[str]:
    """Validate that all state dicts share the same keys and shapes.

    Returns:
        The sorted list of shared keys.

    Raises:
        MergeError: If empty, keys differ, or shapes mismatch.
    """
    if not state_dicts:
        raise MergeError("Cannot make a soup from zero models.")
    keys = set(state_dicts[0])
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd) != keys:
            missing = keys.symmetric_difference(sd)
            raise MergeError(f"State dict {i} has mismatched keys: {sorted(missing)[:5]}...")
    for key in keys:
        shape = state_dicts[0][key].shape
        for i, sd in enumerate(state_dicts):
            if sd[key].shape != shape:
                raise MergeError(
                    f"Shape mismatch for {key!r} in model {i}: {sd[key].shape} != {shape}."
                )
    return sorted(keys)


def weighted_soup(
    state_dicts: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Return the weighted average of several state dicts.

    Args:
        state_dicts: The models' (or adapters') state dicts; identical key sets.
        weights: One coefficient per state dict. Normalised to sum to 1.

    Returns:
        A new averaged state dict.

    Raises:
        MergeError: On count mismatch, empty input, or key/shape mismatch.

    Example:
        >>> import torch
        >>> a = {"w": torch.zeros(3)}
        >>> b = {"w": torch.ones(3)}
        >>> weighted_soup([a, b], [1.0, 3.0])["w"].tolist()
        [0.75, 0.75, 0.75]
    """
    if len(state_dicts) != len(weights):
        raise MergeError(f"{len(state_dicts)} models but {len(weights)} weights.")
    total = float(sum(weights))
    if total == 0.0:
        raise MergeError("Soup weights sum to zero.")
    keys = _validate(state_dicts)
    norm = [w / total for w in weights]
    soup: dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, coeff in zip(state_dicts, norm, strict=True):
            acc = acc + coeff * sd[key].to(torch.float32)
        soup[key] = acc.to(state_dicts[0][key].dtype)
    return soup


def uniform_soup(state_dicts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Return the uniform (equal-weight) average of several state dicts.

    Args:
        state_dicts: The models' state dicts; identical key sets.

    Returns:
        A new averaged state dict.

    Raises:
        MergeError: On empty input or key/shape mismatch.

    Example:
        >>> import torch
        >>> a = {"w": torch.tensor([0.0, 2.0])}
        >>> b = {"w": torch.tensor([4.0, 0.0])}
        >>> uniform_soup([a, b])["w"].tolist()
        [2.0, 1.0]
    """
    return weighted_soup(state_dicts, [1.0] * len(state_dicts))
