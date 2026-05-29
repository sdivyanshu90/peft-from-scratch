"""bitsandbytes 4-bit / 8-bit quantization helpers for QLoRA.

``bitsandbytes`` is an optional dependency that requires a CUDA GPU. Every entry
point here degrades gracefully: import is lazy, availability is checked up front,
and a clear :class:`~peft_lib.core.exceptions.DeviceError` is raised when the
backend is missing rather than an opaque ``ImportError`` deep in a forward pass.

Quantizing a model replaces its :class:`torch.nn.Linear` layers with
``bitsandbytes`` ``Linear4bit`` / ``Linear8bitLt`` layers. The actual
quantization happens when the model is moved to CUDA (``model.cuda()``).
"""

from __future__ import annotations

import importlib.util
from typing import Literal

import torch
from torch import nn

from peft_lib.core.exceptions import ConfigError, DeviceError

__all__ = ["dtype_from_str", "is_bnb_available", "replace_with_bnb_linear", "require_bnb"]

QuantType = Literal["nf4", "fp4"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def dtype_from_str(name: str) -> torch.dtype:
    """Resolve a JSON-friendly dtype name (e.g. ``"bfloat16"``) to a ``torch.dtype``.

    Args:
        name: One of ``"float32"``, ``"float16"``, ``"bfloat16"``.

    Returns:
        The corresponding :class:`torch.dtype`.

    Raises:
        ConfigError: If ``name`` is not a supported dtype string.

    Example:
        >>> dtype_from_str("bfloat16")
        torch.bfloat16
    """
    try:
        return _DTYPES[name]
    except KeyError:
        raise ConfigError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPES)}.") from None


def is_bnb_available() -> bool:
    """Return whether ``bitsandbytes`` is importable in this environment.

    Returns:
        ``True`` if the package is installed (does not check for a usable GPU).

    Example:
        >>> isinstance(is_bnb_available(), bool)
        True
    """
    return importlib.util.find_spec("bitsandbytes") is not None


def require_bnb() -> None:
    """Raise a clear error if ``bitsandbytes`` is unavailable.

    Raises:
        DeviceError: If the package is not installed.
    """
    if not is_bnb_available():
        raise DeviceError(
            "bitsandbytes is required for QLoRA / 4-bit / 8-bit quantization but is "
            "not installed. Install with `pip install peft-lib[quant]` (needs a CUDA GPU)."
        )


def replace_with_bnb_linear(
    model: nn.Module,
    *,
    bits: int = 4,
    quant_type: QuantType = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
    skip_modules: frozenset[str] = frozenset({"lm_head", "score", "classifier"}),
) -> nn.Module:
    """Replace ``nn.Linear`` layers in ``model`` with bitsandbytes quantized linears.

    The original FP weights are copied into the quantized layer; quantization
    materialises when the model is moved to CUDA.

    Args:
        model: The model to convert (mutated in place).
        bits: ``4`` (``Linear4bit``) or ``8`` (``Linear8bitLt``).
        quant_type: 4-bit quantization scheme (``"nf4"`` or ``"fp4"``).
        compute_dtype: Dtype string for the de-quantized compute path.
        double_quant: Whether to use 4-bit double quantization.
        skip_modules: Name leaves to leave in full precision (typically heads).

    Returns:
        The same ``model``, with linears replaced.

    Raises:
        DeviceError: If ``bitsandbytes`` is not installed.
        ConfigError: If ``bits`` is not 4 or 8.

    Example:
        >>> from peft_lib.quantization import is_bnb_available
        >>> # Quantization is gated on a usable backend; without it, callers get a
        >>> # clear DeviceError (see require_bnb) rather than an opaque ImportError.
        >>> isinstance(is_bnb_available(), bool)
        True
    """
    require_bnb()
    if bits not in (4, 8):
        raise ConfigError(f"bits must be 4 or 8, got {bits}.")
    # The actual bnb replacement requires a CUDA build of bitsandbytes; it is
    # exercised by the GPU-only `quant`-marked test, not the CPU coverage run.
    import bitsandbytes as bnb  # pragma: no cover

    compute = dtype_from_str(compute_dtype)  # pragma: no cover
    for parent in model.modules():  # pragma: no cover
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or child_name in skip_modules:
                continue
            has_bias = child.bias is not None
            if bits == 4:
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    compute_dtype=compute,
                    quant_type=quant_type,
                    compress_statistics=double_quant,
                )
            else:
                new = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                )
            new.weight = bnb.nn.Params4bit(child.weight.data) if bits == 4 else new.weight
            with torch.no_grad():
                if bits != 4:
                    new.weight.data.copy_(child.weight.data)
                if has_bias:
                    new.bias.data.copy_(child.bias.data)
            setattr(parent, child_name, new)
    return model
