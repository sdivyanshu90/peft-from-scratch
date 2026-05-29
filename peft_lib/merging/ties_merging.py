r"""TIES-Merging: Trim, Elect Sign & Disjoint Merge (Yadav et al., 2023).

================================================================================
INTUITION & ALGORITHM
================================================================================
Naively averaging several task-specific fine-tunes cancels out useful updates,
because different tasks push the *same* parameter in *opposite* directions. TIES
resolves these conflicts in three steps, operating on **task vectors**
``tau_i = theta_i - theta_base`` (each fine-tune's delta from the shared base):

1. **Trim.** Per task vector, keep only the top-``density`` fraction of entries by
   magnitude; zero the rest. Small, noisy updates are discarded.
2. **Elect Sign.** For each parameter, sum the (trimmed) values across tasks and
   take the sign of that sum — the direction the majority "votes" for.
3. **Disjoint Merge.** For each parameter, average only the task values whose sign
   agrees with the elected sign (ignoring zeros and dissenters).

The merged model is ``theta_base + scaling * tau_merged``.

This removes the destructive interference that plagues plain averaging while
keeping the agreeing signal at full strength.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from peft_lib.core.exceptions import ConfigError, MergeError

__all__ = ["make_task_vector", "ties_merge", "ties_merge_into"]


def make_task_vector(
    finetuned: Mapping[str, torch.Tensor],
    base: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the task vector ``finetuned - base`` over shared keys.

    Args:
        finetuned: A fine-tuned model's state dict.
        base: The shared base model's state dict.

    Returns:
        The per-parameter delta for every key present in both.

    Example:
        >>> import torch
        >>> ft = {"w": torch.tensor([2.0, 3.0])}
        >>> base = {"w": torch.tensor([1.0, 1.0])}
        >>> make_task_vector(ft, base)["w"].tolist()
        [1.0, 2.0]
    """
    return {k: finetuned[k] - base[k] for k in finetuned if k in base}


def _trim_tensor(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top-``density`` fraction of ``tensor`` by magnitude; zero the rest."""
    flat = tensor.flatten()
    n = flat.numel()
    k = max(1, round(density * n))
    if k >= n:
        return tensor.clone()
    _, idx = flat.abs().topk(k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).view_as(tensor)


def ties_merge(
    task_vectors: Sequence[Mapping[str, torch.Tensor]],
    *,
    density: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Merge task vectors via Trim -> Elect Sign -> Disjoint Merge.

    Args:
        task_vectors: One delta state dict per task (see :func:`make_task_vector`).
            All must share the same keys/shapes.
        density: Fraction of entries to keep per task vector during Trim
            (``0 < density <= 1``). Typical: 0.1-0.3.

    Returns:
        The merged task vector (same keys as the inputs).

    Raises:
        ConfigError: If ``density`` is out of range.
        MergeError: If inputs are empty or have mismatched keys.

    Example:
        >>> import torch
        >>> # Two tasks; on dim 0 they agree (+), on dim 1 they conflict.
        >>> t1 = {"w": torch.tensor([1.0, 2.0])}
        >>> t2 = {"w": torch.tensor([3.0, -2.0])}
        >>> merged = ties_merge([t1, t2], density=1.0)["w"]
        >>> merged[0].item()  # agree -> mean(1, 3) = 2.0
        2.0
    """
    if not 0.0 < density <= 1.0:
        raise ConfigError(f"density must be in (0, 1], got {density}.")
    if not task_vectors:
        raise MergeError("ties_merge needs at least one task vector.")
    keys = set(task_vectors[0])
    for i, tv in enumerate(task_vectors[1:], start=1):
        if set(tv) != keys:
            raise MergeError(f"Task vector {i} has mismatched keys.")

    merged: dict[str, torch.Tensor] = {}
    for key in task_vectors[0]:
        orig_dtype = task_vectors[0][key].dtype
        # (T, *shape) after per-task trimming.
        trimmed = torch.stack(
            [_trim_tensor(tv[key].to(torch.float32), density) for tv in task_vectors]
        )
        elected_sign = torch.sign(trimmed.sum(dim=0))  # (*shape), in {-1, 0, 1}
        agree = (torch.sign(trimmed) == elected_sign) & (elected_sign != 0)  # (T, *shape)
        kept = torch.where(agree, trimmed, torch.zeros_like(trimmed))
        count = agree.sum(dim=0).clamp(min=1)  # avoid div-by-zero
        merged[key] = (kept.sum(dim=0) / count).to(orig_dtype)
    return merged


def ties_merge_into(
    base: Mapping[str, torch.Tensor],
    task_vectors: Sequence[Mapping[str, torch.Tensor]],
    *,
    density: float = 0.2,
    scaling: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply a TIES merge of ``task_vectors`` on top of ``base``.

    Computes ``theta = base + scaling * ties_merge(task_vectors)``.

    Args:
        base: The shared base model's state dict.
        task_vectors: One delta per task.
        density: Trim density (see :func:`ties_merge`).
        scaling: Multiplier ``lambda`` on the merged task vector.

    Returns:
        The merged model state dict (keys from ``base``).

    Example:
        >>> import torch
        >>> base = {"w": torch.zeros(2)}
        >>> t1 = {"w": torch.tensor([1.0, 2.0])}
        >>> t2 = {"w": torch.tensor([3.0, -2.0])}
        >>> ties_merge_into(base, [t1, t2], density=1.0)["w"][0].item()
        2.0
    """
    merged = ties_merge(task_vectors, density=density)
    out: dict[str, torch.Tensor] = {}
    for key, base_tensor in base.items():
        if key in merged:
            out[key] = base_tensor + scaling * merged[key].to(base_tensor.dtype)
        else:
            out[key] = base_tensor.clone()
    return out
