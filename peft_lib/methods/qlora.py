r"""QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023).

================================================================================
MATHEMATICAL DERIVATION  (notation as for LoRA; W0 stored in 4-bit)
================================================================================

**(1) Problem setup.** LoRA already shrinks the *trainable* footprint, but the
frozen base weights still sit in fp16 (14 GB for a 7B model). QLoRA shrinks the
*frozen* footprint too by storing ``W0`` in 4-bit, while keeping LoRA adapters in
higher precision.

**(2) Quantization.** ``W0`` is quantized with the **4-bit NormalFloat (NF4)**
scheme — a quantile-based code optimal for the (roughly Gaussian) distribution of
neural-network weights — optionally with **double quantization** of the scales.
Let ``Q(W0)`` denote the 4-bit code and ``dequant(Q(W0))`` its de-quantized form.

**(3) Forward pass.** The frozen quantized weight is de-quantized on the fly and
the LoRA term (Eq. 4 of :mod:`peft_lib.methods.lora`) is added in compute dtype:

        y = dequant(Q(W0)) x + b0 + s (dropout(x) A^T) B^T.            (Eq. 1)

Only ``A, B`` (fp16/bf16) are trained; ``Q(W0)`` is fixed.

**(4) Initialization.** Identical to LoRA: ``A ~ kaiming``, ``B = 0`` (zero delta
on top of the quantized base).

**(5) Backward.** Gradients flow only to ``A, B``; the 4-bit weights receive no
gradient. Optimizer state (paged optimizers in the paper) covers only the
adapters.

**(6) Parameter count.** *Trainable* params are exactly LoRA's
``r*(in+out)`` per layer. The win is *memory*: ``W0`` drops from 16-bit to ~4-bit
(plus small per-block scales), roughly a 4x reduction in frozen-weight VRAM.

**(7) Merge note.** ``dequant(Q(W0)) + sBA`` cannot be folded back into a 4-bit
weight without precision loss, so :meth:`QLoRAModel.merge_and_unload` raises;
de-quantize the base to fp16 first, then merge as ordinary LoRA.

**(8) Requirements.** Needs ``bitsandbytes`` + a CUDA GPU. Constructing a
``QLoRAModel`` without them raises a clear
:class:`~peft_lib.core.exceptions.DeviceError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

from peft_lib.core.base import PEFTConfig
from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.core.registry import register_peft
from peft_lib.methods.lora import LoRAConfig, LoRAModel
from peft_lib.quantization.bnb_utils import replace_with_bnb_linear

__all__ = ["QLoRAConfig", "QLoRAModel"]

QuantType = Literal["nf4", "fp4"]


@dataclass(kw_only=True)
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (LoRA over a 4-bit / 8-bit quantized base).

    Inherits every LoRA field (``r``, ``alpha``, ``dropout``, ``target_modules``,
    ...) and adds the quantization knobs.

    Attributes:
        bits: ``4`` (NF4/FP4) or ``8`` (LLM.int8()).
        quant_type: 4-bit scheme: ``"nf4"`` (default) or ``"fp4"``.
        double_quant: Use 4-bit double quantization of the block scales.
        compute_dtype: Compute dtype for the de-quantized path (string, e.g.
            ``"bfloat16"``).
        quantize_base: If ``True`` (default), quantize the backbone at construction
            time; set ``False`` if the model is already quantized.

    Example:
        >>> import peft_lib
        >>> from peft_lib import QLoRAConfig
        >>> cfg = QLoRAConfig(r=16, bits=4, quant_type="nf4", target_modules=["q_proj"])
        >>> cfg.peft_type, cfg.bits
        ('qlora', 4)
    """

    bits: int = 4
    quant_type: QuantType = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    quantize_base: bool = True

    def validate(self) -> None:
        """Validate QLoRA hyperparameters (LoRA's plus quantization).

        Raises:
            ConfigError: If ``bits`` not in {4, 8} or ``quant_type`` invalid.
        """
        super().validate()
        if self.bits not in (4, 8):
            raise ConfigError(f"bits must be 4 or 8, got {self.bits}.")
        if self.quant_type not in ("nf4", "fp4"):
            raise ConfigError(f"quant_type must be 'nf4'|'fp4', got {self.quant_type!r}.")


@register_peft("qlora")
class QLoRAModel(LoRAModel):
    """LoRA over a bitsandbytes-quantized backbone.

    Quantizes the base model (unless ``config.quantize_base`` is ``False`` or it is
    already quantized), then injects standard LoRA adapters. Requires
    ``bitsandbytes`` + CUDA; otherwise construction raises ``DeviceError``.

    Example:
        >>> import torch.nn as nn, peft_lib  # doctest: +SKIP
        >>> from peft_lib import QLoRAConfig, get_peft_model  # doctest: +SKIP
        >>> base = nn.Linear(32, 32)  # doctest: +SKIP
        >>> cfg = QLoRAConfig(r=8, target_modules=[""])  # doctest: +SKIP
        >>> peft = get_peft_model(base, cfg).cuda()  # doctest: +SKIP
    """

    config_class = QLoRAConfig

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        assert isinstance(config, QLoRAConfig)
        if config.quantize_base:
            # Raises DeviceError if bitsandbytes is unavailable (e.g. CPU-only CI).
            replace_with_bnb_linear(
                base_model,
                bits=config.bits,
                quant_type=config.quant_type,
                compute_dtype=config.compute_dtype,
                double_quant=config.double_quant,
            )
        super().__init__(base_model, config)

    def merge_and_unload(self) -> nn.Module:
        """QLoRA cannot fold a delta into a 4-bit weight losslessly.

        Raises:
            MergeError: Always; de-quantize the base to fp16 first, then merge.
        """
        raise MergeError(
            "QLoRA adapters cannot be merged into a 4-bit base losslessly. "
            "De-quantize the base model to fp16/bf16, then merge as ordinary LoRA."
        )
