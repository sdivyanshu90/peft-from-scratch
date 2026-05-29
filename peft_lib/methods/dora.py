r"""DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

================================================================================
MATHEMATICAL DERIVATION  (notation: out, in, r, s = scaling; ||.||_c = column norm)
================================================================================

**(1) Problem setup.** LoRA constrains the *whole* update ``ΔW = s B A`` to be
low rank. Liu et al. observe that full fine-tuning changes a weight's **magnitude**
and **direction** in ways LoRA struggles to reproduce jointly. DoRA fixes this by
decomposing the weight and adapting the two parts separately.

**(2) Decomposition.** Any weight ``W in R^{out x in}`` factorises (per output row)
into a magnitude scalar and a unit-norm direction:

        W = m ⊙ ( V / ||V||_c ),                                       (Eq. 1)

where ``||V||_c`` is the row-wise (per-output) L2 norm, ``m in R^{out}`` is the
magnitude vector, and ``V / ||V||_c`` has unit-norm rows (the direction).

**(3) Parameterization.** DoRA trains:
* ``m in R^{out}`` — the magnitude (a full, cheap vector).
* a LoRA pair ``A in R^{r x in}``, ``B in R^{out x r}`` that updates only the
  **direction**: ``V = W0 + s B A``.

Effective weight:

        W' = m ⊙ ( (W0 + s B A) / ||W0 + s B A||_c ).                  (Eq. 2)

**(4) Initialization (zero-delta).** ``B = 0`` (so ``V = W0``) and
``m = ||W0||_c``. Then ``W' = ||W0||_c ⊙ (W0 / ||W0||_c) = W0`` exactly — the
adapted model is identical to the pretrained one at step 0 (the LoRA zero-delta
invariant, extended to the decomposed form).

**(5) Forward pass (efficient form).** Materialising ``W'`` each step is wasteful.
Using ``x W'^T = (m / ||V||_c) ⊙ (x W0^T + s x (BA)^T)`` we compute:

        base = x W0^T                          (no bias)
        lora = s (dropout(x) A^T) B^T
        scale = m / ||W0 + s B A||_c            (per output row)
        y = scale ⊙ (base + lora) + b0                                 (Eq. 3)

Following the paper, ``||V||_c`` is treated as a **constant w.r.t. backprop**
(``.detach()``), which removes a large activation-memory term while leaving the
forward values exact.

**(6) Parameter count (derived).** Per adapted layer of shape ``(out, in)``:

        params = r*in (A) + out*r (B) + out (m) = r*(in + out) + out.   (Eq. 4)

DoRA costs exactly ``out`` more parameters per layer than LoRA — the magnitude
vector.

**(7) Scaling analysis.** Trainable params ~ LoRA + ``out`` per layer (negligible).
Forward adds the LoRA FLOPs plus one ``(out x in)`` norm; the norm is ``O(out*in)``
but elementwise and cheap relative to the matmuls.

**(8) Connection to related methods.** DoRA == LoRA on the direction + an explicit
magnitude vector. Setting ``m := ||W0 + sBA||_c`` (i.e. dropping the magnitude
parameter) recovers LoRA exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.exceptions import ConfigError, MergeError, ShapeError
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec
from peft_lib.methods.lora import _infer_linear_dims

__all__ = ["DoRAConfig", "DoRALinear", "DoRAModel"]


@dataclass(kw_only=True)
class DoRAConfig(PEFTConfig):
    """Configuration for DoRA (weight-decomposed LoRA).

    Attributes:
        r: Rank of the directional LoRA update. Must be >= 1.
        alpha: Scale numerator; ``s = alpha / r`` (or ``alpha/sqrt(r)`` if rsLoRA).
        dropout: Input dropout on the LoRA path.
        target_modules: Suffix list or ``"all-linear"``; ``None`` infers per arch.
        use_rslora: Use ``alpha/sqrt(r)`` scaling if ``True``.
        init_lora_weights: ``A`` init scheme (``B`` always zero; ``m`` always the
            row norm of ``W0``).

    Example:
        >>> import peft_lib
        >>> from peft_lib import DoRAConfig
        >>> DoRAConfig(r=8, target_modules=["q_proj"]).peft_type
        'dora'
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: TargetSpec | None = None
    use_rslora: bool = False
    init_lora_weights: bool = True

    def validate(self) -> None:
        """Validate DoRA hyperparameters.

        Raises:
            ConfigError: If ``r < 1``, ``alpha <= 0``, or ``dropout`` not in [0, 1).
        """
        super().validate()
        if self.r < 1:
            raise ConfigError(f"DoRA rank r must be >= 1, got {self.r}.")
        if self.alpha <= 0:
            raise ConfigError(f"DoRA alpha must be > 0, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}.")

    @property
    def scaling(self) -> float:
        """Effective scale ``s`` (alpha/r, or alpha/sqrt(r) under rsLoRA)."""
        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r


class DoRALinear(nn.Module):
    r"""A frozen linear layer adapted via magnitude + low-rank direction (Eq. 2-3).

    Args:
        base_layer: The linear-like module to adapt (``nn.Linear`` or HF ``Conv1D``).
        r: Rank of the directional update.
        alpha: Scale numerator.
        dropout: Input dropout on the LoRA path.
        use_rslora: Use ``alpha/sqrt(r)`` scaling.
        init_lora_weights: ``A`` init scheme.

    Attributes:
        merged: Whether ``W'`` currently lives in ``base_layer.weight``.
        magnitude: The trainable magnitude vector ``m`` of shape ``(out,)``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> lin = torch.nn.Linear(16, 32)
        >>> dora = DoRALinear(lin, r=4, alpha=8)
        >>> x = torch.randn(2, 16)
        >>> torch.allclose(dora(x), lin(x), atol=1e-5)  # zero-delta init
        True
    """

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ConfigError(f"DoRA rank r must be >= 1, got {r}.")
        self.base_layer = base_layer
        self.in_features, self.out_features, self.fan_in_fan_out = _infer_linear_dims(base_layer)
        self.r = r
        self.alpha = alpha
        self.scaling: float = alpha / math.sqrt(r) if use_rslora else alpha / r
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
        # Magnitude m initialised to the row-norm of W0 (-> zero-delta with B=0).
        w0 = self._canonical_weight().detach()
        self.magnitude = nn.Parameter(torch.linalg.norm(w0, dim=1))  # (out,)

        if init_lora_weights:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.normal_(self.lora_A, std=1.0 / r)
        nn.init.zeros_(self.lora_B)

        self.merged = False
        self._merge_delta: torch.Tensor | None = None

    def _canonical_weight(self) -> torch.Tensor:
        """Return ``W0`` in canonical ``(out, in)`` orientation (transposing Conv1D)."""
        weight = self.base_layer.weight
        assert isinstance(weight, torch.Tensor)
        return weight.transpose(0, 1) if self.fan_in_fan_out else weight

    def _effective_weight(self) -> torch.Tensor:
        r"""Compute ``W' = m ⊙ (V / ||V||_c)`` with ``V = W0 + s B A`` (Eq. 2)."""
        w0 = self._canonical_weight()
        v = w0 + self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
        norm_v: torch.Tensor = torch.linalg.norm(v, dim=1, keepdim=True)  # (out, 1)
        return self.magnitude.unsqueeze(1) * v / norm_v  # (out, in)

    def get_delta_weight(self) -> torch.Tensor:
        """Return ``W' - W0`` in canonical orientation (valid only when not merged).

        Returns:
            The dense delta of shape ``(out, in)``.
        """
        return self._effective_weight() - self._canonical_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the DoRA output via the efficient form (Eq. 3).

        Args:
            x: Input of shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``.

        Raises:
            ShapeError: If ``x``'s last dim is not ``in_features``.
        """
        if x.shape[-1] != self.in_features:
            raise ShapeError(
                "DoRALinear input has wrong feature dim",
                expected=(self.in_features,),
                actual=(x.shape[-1],),
            )
        if self.merged:
            result: torch.Tensor = self.base_layer(x)
            return result

        dtype = self.magnitude.dtype
        w0 = self._canonical_weight().to(dtype)
        bias = getattr(self.base_layer, "bias", None)

        base: torch.Tensor = F.linear(x.to(dtype), w0)  # x W0^T, no bias
        x_drop: torch.Tensor = self.lora_dropout(x).to(dtype)
        lora = (x_drop @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        lora = self.scaling * lora

        v = w0 + self.scaling * (self.lora_B @ self.lora_A)
        # Detach the norm (paper's memory-efficient gradient); forward value exact.
        norm_v: torch.Tensor = torch.linalg.norm(v, dim=1).detach()  # (out,)
        scale = self.magnitude / norm_v  # (out,)
        out = scale * (base + lora)  # broadcast (out,) over (..., out)
        if bias is not None:
            out = out + bias.to(dtype)
        return out.to(x.dtype)

    def merge(self) -> None:
        """Fold ``W' - W0`` into ``base_layer.weight`` in place (idempotent-guarded).

        Raises:
            MergeError: If already merged.
        """
        if self.merged:
            raise MergeError("DoRALinear is already merged.")
        delta = self.get_delta_weight().detach().to(self.base_layer.weight.dtype)
        self._merge_delta = delta
        with torch.no_grad():
            self.base_layer.weight.add_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self.merged = True

    def unmerge(self) -> None:
        """Subtract the cached ``W' - W0`` back out of ``base_layer.weight``.

        Raises:
            MergeError: If not currently merged.
        """
        if not self.merged or self._merge_delta is None:
            raise MergeError("DoRALinear is not merged; nothing to unmerge.")
        delta = self._merge_delta
        with torch.no_grad():
            self.base_layer.weight.sub_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self._merge_delta = None
        self.merged = False

    def extra_repr(self) -> str:
        """Return a concise representation for ``print(model)``."""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, scaling={self.scaling:.4g}"
        )


@register_peft("dora")
class DoRAModel(InjectionPEFTModel):
    """DoRA-wrapped model: injects :class:`DoRALinear` adapters into the frozen backbone.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import DoRAConfig, get_peft_model
        >>> _ = torch.manual_seed(0)
        >>> peft = get_peft_model(nn.Linear(32, 16), DoRAConfig(r=8, target_modules=[""]))
        >>> trn, _ = peft.get_nb_trainable_parameters()
        >>> trn  # r*(in+out) + out = 8*(32+16) + 16
        400
    """

    config_class = DoRAConfig

    @property
    def dora_config(self) -> DoRAConfig:
        """Return the config narrowed to :class:`DoRAConfig`."""
        assert isinstance(self.config, DoRAConfig)
        return self.config

    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        cfg = self.dora_config
        return DoRALinear(
            base_layer,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            use_rslora=cfg.use_rslora,
            init_lora_weights=cfg.init_lora_weights,
        )
