r"""Merge LoRA-family adapters back into a base model for zero-overhead inference.

This is a thin, well-typed wrapper over each method's own ``merge`` logic plus a
helper for **multi-adapter weighted merging**: combining several trained adapters
into one set of base weights, e.g. ``W0 + sum_i w_i * ΔW_i``.

Single-adapter merging is just ``peft_model.merge_and_unload()`` (defined on the
model). The functions here add (a) a convenience wrapper that returns the bare
backbone and (b) a linear combination of multiple adapter checkpoints — for the
*additive*-delta methods (LoRA, DoRA, VeRA). IA³ is multiplicative and is not
supported by the weighted sum.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTModel
from peft_lib.core.exceptions import MergeError
from peft_lib.core.utils import set_submodule

__all__ = ["merge_and_unload", "weighted_merge_adapters"]


@runtime_checkable
class _AdditiveAdapter(Protocol):
    """An injection adapter with an explicit additive weight delta (LoRA/DoRA/VeRA)."""

    base_layer: nn.Module
    merged: bool
    fan_in_fan_out: bool

    def get_delta_weight(self) -> torch.Tensor: ...


def merge_and_unload(model: PEFTModel) -> nn.Module:
    """Fold a model's adapters into the backbone and return the plain module.

    Equivalent to ``model.merge_and_unload()``; provided here so merging code can
    be imported from one place.

    Args:
        model: A foldable PEFT model (LoRA/DoRA/VeRA/IA³).

    Returns:
        The unwrapped backbone with adapters folded in.

    Raises:
        MergeError: If the method is not foldable.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> from peft_lib.merging import merge_and_unload
        >>> _ = torch.manual_seed(0)
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> bare = merge_and_unload(peft)
        >>> isinstance(bare, nn.Linear)
        True
    """
    return model.merge_and_unload()


def weighted_merge_adapters(
    model: InjectionPEFTModel,
    adapter_state_dicts: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> nn.Module:
    r"""Merge several adapter checkpoints into the base weights as a weighted sum.

    Computes ``W0 + sum_i w_i * ΔW_i`` per adapted layer, where each ``ΔW_i`` is
    the delta implied by the i-th adapter checkpoint loaded into the model's
    layers. This is the building block for combining task-specific adapters into a
    single merged model (cf. LoRA-hub / adapter averaging).

    Args:
        model: A LoRA/DoRA/VeRA model (additive deltas; IA³ is unsupported).
        adapter_state_dicts: One adapter ``state_dict`` per adapter to merge (keys
            must match :meth:`PEFTModel.adapter_state_dict`).
        weights: Mixing coefficient per adapter (same length as the state dicts).

    Returns:
        The unwrapped backbone with the weighted delta folded in.

    Raises:
        MergeError: If lengths mismatch, an adapter is not additive, or a key set
            does not match the model.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> from peft_lib.merging import weighted_merge_adapters
        >>> _ = torch.manual_seed(0)
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> sd = {k: torch.randn_like(v) for k, v in peft.adapter_state_dict().items()}
        >>> bare = weighted_merge_adapters(peft, [sd], [0.5])
        >>> isinstance(bare, nn.Linear)
        True
    """
    if len(adapter_state_dicts) != len(weights):
        raise MergeError(f"Got {len(adapter_state_dicts)} adapters but {len(weights)} weights.")
    if not adapter_state_dicts:
        raise MergeError("No adapters to merge.")

    expected_keys = set(model.adapter_state_dict())
    for i, sd in enumerate(adapter_state_dicts):
        if set(sd) != expected_keys:
            raise MergeError(f"Adapter {i} keys do not match the model's adapter keys.")
    for name, raw in model.adapter_layers.items():
        if not isinstance(raw, _AdditiveAdapter):
            raise MergeError(
                f"Adapter at {name!r} ({type(raw).__name__}) has no additive delta; "
                "weighted_merge_adapters supports LoRA/DoRA/VeRA only."
            )

    # Accumulate the weighted, correctly-oriented delta for each layer by loading
    # one adapter checkpoint at a time and reading off its per-layer delta.
    adapters = {name: cast(_AdditiveAdapter, m) for name, m in model.adapter_layers.items()}
    totals: dict[str, torch.Tensor] = {
        name: torch.zeros_like(ad.base_layer.weight) for name, ad in adapters.items()
    }
    for sd, coeff in zip(adapter_state_dicts, weights, strict=True):
        model.load_state_dict(sd, strict=False)
        for name, ad in adapters.items():
            weight = ad.base_layer.weight
            delta = ad.get_delta_weight().to(weight.dtype)
            oriented = delta.transpose(0, 1) if ad.fan_in_fan_out else delta
            totals[name] = totals[name] + coeff * oriented

    for name, ad in adapters.items():
        with torch.no_grad():
            ad.base_layer.weight.add_(totals[name])
        ad.merged = True
        if name == "":
            model.base_model = ad.base_layer
        else:
            set_submodule(model.base_model, name, ad.base_layer)
    model.adapter_layers.clear()
    return model.base_model
