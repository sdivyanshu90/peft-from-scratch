"""Quantization helpers (bitsandbytes 4-bit / 8-bit). Optional dependency."""

from __future__ import annotations

from peft_lib.quantization.bnb_utils import (
    is_bnb_available,
    replace_with_bnb_linear,
    require_bnb,
)

__all__ = ["is_bnb_available", "replace_with_bnb_linear", "require_bnb"]
