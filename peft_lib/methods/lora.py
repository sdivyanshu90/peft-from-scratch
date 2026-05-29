r"""LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021).

================================================================================
MATHEMATICAL DERIVATION  (notation: B=batch, S=seq, D=dim, r=rank, alpha, s=alpha/r)
================================================================================

**(1) Problem setup.** Full fine-tuning of a pretrained linear map
``y = W0 x`` (``W0 in R^{out x in}``) learns an update ``W0 -> W0 + ΔW`` with
``out * in`` trainable parameters. For a 7B model that is billions of numbers per
checkpoint. We want a cheaper ``ΔW``.

**(2) Core hypothesis / constraint.** Aghajanyan et al. (2020) showed the update
that adaptation needs lies on a *low intrinsic dimension*. LoRA imposes this as a
hard rank constraint:

        ΔW = s * B A ,   with   rank(B A) <= r ,                         (Eq. 1)

where ``A in R^{r x in}``, ``B in R^{out x r}``, ``r << min(in, out)``, and
``s = alpha / r`` is a fixed scalar (Eq. 3). We learn ``A, B`` and freeze ``W0``.

**(3) Parameterization, shapes, init, update rule.**
        A in R^{r x in}    init: A ~ kaiming_uniform(a=sqrt 5)
        B in R^{out x r}    init: B = 0                                  (Eq. 2)
The ``B = 0`` initialization is *load-bearing*: it makes ``ΔW = 0`` at step 0, so
the adapted model is bit-for-bit the pretrained model before any gradient step
(Hu et al. §4.1). Initialising ``B`` randomly would inject untrained noise into
every layer — a HARD-banned mistake in this library.

**(4) Forward pass.** For an input ``x in R^{... x in}``:

        h_base = W0 x + b0                                              (frozen)
        h_lora = s * ( dropout(x) A^T ) B^T                            (Eq. 4)
        y      = h_base + h_lora

Note dropout is applied to ``x`` *before* ``A`` (never between ``A`` and ``B``):
the regulariser perturbs the input subspace the adapter reads, matching Hu et
al.'s reference implementation. Banned mistake: dropout between A and B.

**(5) Backward / gradient analysis.** With loss ``L``:

        dL/dB = s * (dL/dy)^T (x A^T)        -> shape (out, r)
        dL/dA = s * B^T (dL/dy) x^T          -> shape (r, in)

At step 0, ``B = 0`` so ``dL/dA = 0``: the *A* matrix does not move on the first
step, but ``dL/dB != 0`` (it sees ``x A^T``), so ``B`` lifts off zero immediately
and training proceeds. There is no dead-gradient stall because ``A`` is random,
not zero.

**(6) Parameter count (derived).** Per adapted layer of shape ``(out, in)``:

        params = r*in (matrix A) + out*r (matrix B) = r*(in + out).     (Eq. 5)

Summed over a target set ``T``:  ``Σ_{(in,out) in T} r*(in+out)``. For r=8 on
BERT-base (D=768) adapting query+value across 12 layers:
``12 * 2 * 8 * (768+768) = 294,912``  (verified in tests/regression).

**(7) Scaling analysis.**
* Memory: trainable params drop from ``out*in`` to ``r*(in+out)``; for a 768x768
  layer at r=8 that is 12,288 vs 589,824 — a 48x reduction. Optimizer state
  (Adam: 2x params) scales with the *trainable* count, the real VRAM win.
* FLOPs (forward): base ``2*out*in`` MACs per token; LoRA adds ``2*r*(in+out)``.
  At r=8, D=768 that is ~3% extra — the library's <5% overhead budget.

**(8) Connection to related methods.**
* **rsLoRA** (Kalajdzievski, 2023): replaces ``s = alpha/r`` with
  ``s = alpha/sqrt(r)`` so that learning is stable as ``r`` grows (Eq. 6).
* **LoRA+** (Hayou et al., 2024): keeps the math but sets ``lr_B = ratio * lr_A``
  (ratio ~ 16); see :func:`peft_lib.training.trainer.build_lora_plus_param_groups`.
* **DoRA** (Liu et al., 2024): decomposes ``W`` into magnitude + direction and
  applies LoRA only to the direction; see :mod:`peft_lib.methods.dora`.
* **Tied-LoRA** (Renduchintala et al., 2024): shares ``A, B`` across layers; the
  ``tie_weights`` flag here implements the weight-sharing core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.exceptions import ConfigError, MergeError, ShapeError
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec

__all__ = ["LoRAConfig", "LoRALinear", "LoRAModel"]

BiasMode = Literal["none", "all", "lora_only"]


def _infer_linear_dims(layer: nn.Module) -> tuple[int, int, bool]:
    """Infer ``(in_features, out_features, fan_in_fan_out)`` for a linear-like layer.

    Handles :class:`torch.nn.Linear` (weight ``(out, in)``) and HuggingFace
    ``Conv1D`` (weight ``(in, out)``, used by GPT-2). ``fan_in_fan_out`` is
    ``True`` when the weight is stored transposed (Conv1D), which the merge logic
    needs in order to add the delta in the right orientation.

    Args:
        layer: The module to inspect.

    Returns:
        ``(in_features, out_features, fan_in_fan_out)``.

    Raises:
        ShapeError: If the layer has no 2-D ``weight`` we can interpret.
    """
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features, False
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2:
        raise ShapeError(f"{type(layer).__name__} is not a supported linear-like layer.")
    if type(layer).__name__ == "Conv1D":  # transformers Conv1D: weight is (in, out)
        return weight.shape[0], weight.shape[1], True
    # Generic fallback: assume nn.Linear convention (out, in).
    return weight.shape[1], weight.shape[0], False


@dataclass(kw_only=True)
class LoRAConfig(PEFTConfig):
    """Configuration for LoRA and its weight-sharing / scaling variants.

    Attributes:
        r: Rank of the low-rank update (Eq. 1). Must be >= 1. Typical: 4-64.
        alpha: Scaling numerator; effective scale is ``s = alpha / r`` (or
            ``alpha / sqrt(r)`` when ``use_rslora``). Convention: ``alpha = 2*r``.
        dropout: Dropout probability applied to the input *before* ``A`` (Eq. 4).
        target_modules: Suffix list (e.g. ``["q_proj", "v_proj"]``) or
            ``"all-linear"``; ``None`` triggers per-architecture inference.
        bias: Which biases to train: ``"none"`` (default, cleanest checkpoint),
            ``"all"`` (every base bias), or ``"lora_only"`` (only adapted layers').
        use_rslora: If ``True``, use rank-stabilised scaling ``alpha/sqrt(r)``.
        lora_plus_lr_ratio: If set, the LoRA+ ratio ``lr_B / lr_A``; consumed by
            the optimizer builder, not by the forward pass.
        tie_weights: If ``True``, share one ``(A, B)`` pair across all adapted
            layers of identical shape (Tied-LoRA core).
        init_lora_weights: If ``True``, ``A ~ kaiming_uniform``; if ``False``,
            ``A ~ N(0, 1/r^2)``. ``B`` is always zero either way (zero-delta init).

    Raises:
        ConfigError: On any invalid field (see :meth:`validate`).

    Example:
        >>> import peft_lib
        >>> from peft_lib import LoRAConfig
        >>> cfg = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
        >>> round(8 / 8, 3)  # scaling s = alpha / r when r==alpha/2 -> here alpha=16
        1.0
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: TargetSpec | None = None
    bias: BiasMode = "none"
    use_rslora: bool = False
    lora_plus_lr_ratio: float | None = None
    tie_weights: bool = False
    init_lora_weights: bool = True
    # Kept for forward-compat / documentation; not yet auto-applied.
    modules_to_save: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate LoRA hyperparameters.

        Raises:
            ConfigError: If ``r < 1``, ``alpha <= 0``, ``dropout`` not in [0, 1),
                ``bias`` not one of the allowed modes, or a non-positive LoRA+
                ratio.
        """
        super().validate()
        if self.r < 1:
            raise ConfigError(f"LoRA rank r must be >= 1, got {self.r}.")
        if self.alpha <= 0:
            raise ConfigError(f"LoRA alpha must be > 0, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.bias not in ("none", "all", "lora_only"):
            raise ConfigError(f"bias must be 'none'|'all'|'lora_only', got {self.bias!r}.")
        if self.lora_plus_lr_ratio is not None and self.lora_plus_lr_ratio <= 0:
            raise ConfigError(f"lora_plus_lr_ratio must be > 0, got {self.lora_plus_lr_ratio}.")

    @property
    def scaling(self) -> float:
        """Effective LoRA scale ``s`` (Eq. 3 / Eq. 6).

        Returns:
            ``alpha / sqrt(r)`` if ``use_rslora`` else ``alpha / r``.
        """
        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r


class LoRALinear(nn.Module):
    r"""A frozen linear layer augmented with a trainable low-rank update.

    Wraps any linear-like ``base_layer`` (``nn.Linear`` or HF ``Conv1D``) and
    computes ``y = base_layer(x) + s * (dropout(x) A^T) B^T`` (Eq. 4). The base
    layer's parameters are expected to be frozen by the owning model.

    Args:
        base_layer: The linear-like module to adapt. Its weights stay frozen.
        r: Rank (Eq. 1).
        alpha: Scale numerator.
        dropout: Input dropout probability (applied before ``A``).
        use_rslora: Use ``alpha/sqrt(r)`` scaling if ``True``.
        init_lora_weights: Init scheme for ``A`` (``B`` is always zero).
        shared_A: Optional shared ``A`` parameter (Tied-LoRA); when given,
            ``shared_B`` must also be given and local init is skipped.
        shared_B: Optional shared ``B`` parameter (Tied-LoRA).

    Attributes:
        merged: Whether the delta currently lives in ``base_layer.weight``.
        scaling: The effective scale ``s``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> lin = torch.nn.Linear(16, 32)
        >>> lora = LoRALinear(lin, r=4, alpha=8)
        >>> x = torch.randn(2, 16)
        >>> torch.allclose(lora(x), lin(x))  # zero-delta init -> identical
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
        shared_A: nn.Parameter | None = None,
        shared_B: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ConfigError(f"LoRA rank r must be >= 1, got {r}.")
        self.base_layer = base_layer
        self.in_features, self.out_features, self.fan_in_fan_out = _infer_linear_dims(base_layer)
        self.r = r
        self.alpha = alpha
        self.use_rslora = use_rslora
        self.scaling: float = alpha / math.sqrt(r) if use_rslora else alpha / r
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if shared_A is not None or shared_B is not None:
            if shared_A is None or shared_B is None:
                raise ConfigError("Tied LoRA requires both shared_A and shared_B.")
            if tuple(shared_A.shape) != (r, self.in_features) or tuple(shared_B.shape) != (
                self.out_features,
                r,
            ):
                raise ShapeError(
                    "Shared LoRA parameters do not match this layer's shape",
                    expected=(r, self.in_features),
                    actual=tuple(shared_A.shape),
                )
            self.lora_A = shared_A
            self.lora_B = shared_B
            self.tied = True
        else:
            # A: (r, in)  ;  B: (out, r)  -- Eq. 2 shapes.
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
            self.tied = False
            self.reset_lora_parameters(init_lora_weights)

        self.merged = False

    def reset_lora_parameters(self, init_lora_weights: bool = True) -> None:
        """(Re)initialise ``A`` and ``B`` to the zero-delta state (Eq. 2).

        Args:
            init_lora_weights: ``True`` -> ``A ~ kaiming_uniform(a=sqrt 5)``;
                ``False`` -> ``A ~ N(0, 1/r^2)``. ``B`` is always set to zero so
                the initial delta is exactly zero.

        Example:
            >>> import torch
            >>> lora = LoRALinear(torch.nn.Linear(8, 8), r=2)
            >>> float(lora.lora_B.abs().sum())
            0.0
        """
        if init_lora_weights:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.normal_(self.lora_A, std=1.0 / self.r)
        nn.init.zeros_(self.lora_B)

    def get_delta_weight(self) -> torch.Tensor:
        r"""Return the effective weight delta ``ΔW = s * B A`` (Eq. 1), shape ``(out, in)``.

        Returns:
            The dense delta in ``(out_features, in_features)`` orientation,
            regardless of ``fan_in_fan_out``.

        Example:
            >>> import torch
            >>> lora = LoRALinear(torch.nn.Linear(4, 6), r=2)
            >>> tuple(lora.get_delta_weight().shape)
            (6, 4)
        """
        # (out, r) @ (r, in) -> (out, in)
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the adapted output (Eq. 4).

        Args:
            x: Input of shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``; equals the base output when
            merged or at zero-delta init.

        Raises:
            ShapeError: If ``x``'s last dim is not ``in_features``.
        """
        if x.shape[-1] != self.in_features:
            raise ShapeError(
                "LoRALinear input has wrong feature dim",
                expected=(self.in_features,),
                actual=(x.shape[-1],),
            )
        result: torch.Tensor = self.base_layer(x)  # (..., out)
        if self.merged:
            return result
        # Cast input to the adapter's dtype; compute delta; cast back to result dtype.
        x_drop: torch.Tensor = self.lora_dropout(x).to(self.lora_A.dtype)
        delta = (x_drop @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)  # (..., out)
        return result + (self.scaling * delta).to(result.dtype)

    def merge(self) -> None:
        """Fold ``ΔW`` into ``base_layer.weight`` in place (idempotent-guarded).

        Raises:
            MergeError: If already merged.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> lora = LoRALinear(torch.nn.Linear(8, 8), r=4)
            >>> lora.lora_B.data.normal_()  # make the delta non-trivial  # doctest: +ELLIPSIS
            tensor(...)
            >>> x = torch.randn(3, 8); before = lora(x)
            >>> lora.merge()
            >>> torch.allclose(before, lora(x), atol=1e-5)
            True
        """
        if self.merged:
            raise MergeError("LoRALinear is already merged.")
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            # Conv1D stores weight transposed; add delta in matching orientation.
            self.base_layer.weight.add_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self.merged = True

    def unmerge(self) -> None:
        """Subtract ``ΔW`` back out of ``base_layer.weight`` in place.

        Raises:
            MergeError: If not currently merged.
        """
        if not self.merged:
            raise MergeError("LoRALinear is not merged; nothing to unmerge.")
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            self.base_layer.weight.sub_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self.merged = False

    def extra_repr(self) -> str:
        """Return a concise representation for ``print(model)``."""
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4g}, tied={self.tied}"
        )


@register_peft("lora")
class LoRAModel(InjectionPEFTModel):
    """LoRA-wrapped model: freezes the backbone and injects :class:`LoRALinear` adapters.

    Supports the standard method plus rsLoRA, LoRA+ (via the optimizer builder),
    and tied-weight LoRA. Build it with :func:`peft_lib.get_peft_model`.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> _ = torch.manual_seed(0)
        >>> base = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8))
        >>> cfg = LoRAConfig(r=8, alpha=16, target_modules=["0", "2"])
        >>> peft = get_peft_model(base, cfg)
        >>> trn, tot = peft.get_nb_trainable_parameters()
        >>> trn  # 8*(32+32) + 8*(32+8)
        832
    """

    config_class = LoRAConfig

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        # Storage for tied parameters, keyed by (in, out); used during injection.
        self._tied_params: dict[tuple[int, int], tuple[nn.Parameter, nn.Parameter]] = {}
        super().__init__(base_model, config)
        self._mark_bias_trainable()

    @property
    def lora_config(self) -> LoRAConfig:
        """Return the config, narrowed to :class:`LoRAConfig` for type-checkers."""
        assert isinstance(self.config, LoRAConfig)
        return self.config

    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        """Build a :class:`LoRALinear` for one target, wiring up tying if requested."""
        cfg = self.lora_config
        shared_A: nn.Parameter | None = None
        shared_B: nn.Parameter | None = None
        if cfg.tie_weights:
            in_f, out_f, _ = _infer_linear_dims(base_layer)
            key = (in_f, out_f)
            if key not in self._tied_params:
                a = nn.Parameter(torch.empty(cfg.r, in_f))
                b = nn.Parameter(torch.empty(out_f, cfg.r))
                if cfg.init_lora_weights:
                    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                else:
                    nn.init.normal_(a, std=1.0 / cfg.r)
                nn.init.zeros_(b)
                self._tied_params[key] = (a, b)
            shared_A, shared_B = self._tied_params[key]
        return LoRALinear(
            base_layer,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            use_rslora=cfg.use_rslora,
            init_lora_weights=cfg.init_lora_weights,
            shared_A=shared_A,
            shared_B=shared_B,
        )

    def _mark_bias_trainable(self) -> None:
        """Enable gradients on biases according to ``config.bias`` (after freezing)."""
        mode = self.lora_config.bias
        if mode == "none":
            return
        if mode == "all":
            for name, param in self.base_model.named_parameters():
                if name.endswith("bias"):
                    param.requires_grad_(True)
        elif mode == "lora_only":
            for adapter in self.adapter_layers.values():
                bias = getattr(adapter.base_layer, "bias", None)
                if bias is not None:
                    bias.requires_grad_(True)
