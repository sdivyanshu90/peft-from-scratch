"""Shared type aliases and structural protocols for :mod:`peft_lib`.

Centralising these keeps signatures across the package consistent and gives
``mypy --strict`` a single source of truth. We prefer *structural* typing
(:class:`typing.Protocol`) over nominal base classes wherever an interface is
all we need, so that third-party modules (e.g. HuggingFace layers) satisfy our
contracts without subclassing anything we own.

Notation (used throughout the codebase and docstrings):
    ``B`` batch, ``S`` sequence length, ``D`` model dim, ``H`` heads,
    ``r`` rank, ``alpha`` LoRA scale numerator, ``s = alpha / r`` effective scale.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "DType",
    "Device",
    "MergeableLayer",
    "Number",
    "StateDict",
    "TargetSpec",
    "TensorDict",
]

# ---------------------------------------------------------------------------
# Plain aliases
# ---------------------------------------------------------------------------
StateDict = dict[str, torch.Tensor]
"""A PyTorch ``state_dict``: parameter/buffer name -> tensor."""

TensorDict = dict[str, torch.Tensor]
"""A generic name -> tensor mapping (forward kwargs, cached activations, ...)."""

Number = int | float
"""A real scalar accepted by config fields (e.g. ``alpha``, ``dropout``)."""

Device = str | torch.device
"""Anything accepted by ``tensor.to(...)`` as a device."""

DType = torch.dtype
"""Alias for readability in signatures that thread a compute dtype."""

TargetSpec = str | list[str]
"""How a config names the submodules to adapt.

Either a list of *suffixes* matched against ``module.named_modules()`` keys
(e.g. ``["q_proj", "v_proj"]``), or the sentinel ``"all-linear"`` meaning
"every :class:`torch.nn.Linear` except the LM head".
"""


# ---------------------------------------------------------------------------
# Structural protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class MergeableLayer(Protocol):
    """An adapter layer whose delta can be folded into (and out of) its base weight.

    Any PEFT layer that supports zero-overhead inference implements this. The
    contract is intentionally minimal so :func:`isinstance` checks at runtime
    (``@runtime_checkable``) stay cheap and unambiguous.

    Attributes:
        merged: ``True`` iff the adapter delta currently lives inside the base
            weight (so ``forward`` must *not* re-apply it).
        base_layer: The wrapped, frozen module whose weight the delta folds into.

    Note:
        ``merge`` must be the exact inverse of ``unmerge`` up to floating-point
        error; the test-suite asserts ``atol=1e-5`` round-trips.
    """

    merged: bool
    base_layer: nn.Module

    def merge(self) -> None:
        """Fold the adapter delta into the base weight in-place."""
        ...

    def unmerge(self) -> None:
        """Subtract the adapter delta back out of the base weight in-place."""
        ...


@runtime_checkable
class SupportsForward(Protocol):
    """Minimal callable-module protocol: ``forward(x) -> Tensor``.

    Used to type the base layer that an adapter wraps without committing to
    :class:`torch.nn.Linear` specifically (HF ``Conv1D`` also qualifies).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an input tensor to an output tensor."""
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Invoke :meth:`forward` (provided by :class:`torch.nn.Module`)."""
        ...


def is_linear_like(module: nn.Module) -> bool:
    """Return whether ``module`` is a weight-carrying linear projection we can adapt.

    Recognises :class:`torch.nn.Linear` and HuggingFace's ``transformers``
    ``Conv1D`` (used by GPT-2), which is a linear layer storing its weight
    transposed as ``(in_features, out_features)``.

    Args:
        module: Any :class:`torch.nn.Module`.

    Returns:
        ``True`` if ``module`` exposes a 2-D ``weight`` we know how to wrap.

    Example:
        >>> import torch.nn as nn
        >>> is_linear_like(nn.Linear(4, 8))
        True
        >>> is_linear_like(nn.LayerNorm(8))
        False
    """
    if isinstance(module, nn.Linear):
        return True
    # Duck-type HF Conv1D without importing transformers at module load time.
    return type(module).__name__ == "Conv1D" and hasattr(module, "weight")
