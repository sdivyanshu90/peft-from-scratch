r"""Bottleneck Adapters (Houlsby et al., 2019; Pfeiffer et al., 2020).

================================================================================
MATHEMATICAL DERIVATION  (notation: D=hidden, b=bottleneck dim)
================================================================================

**(1) Problem setup.** Insert a small trainable module *inside* each transformer
sub-block whose output we want to adapt, leaving all original weights frozen. The
module is a residual bottleneck: down-project, non-linearity, up-project.

**(2) Parameterization.** For a hidden vector ``h in R^{D}``:

        Adapter(h) = h + W_up · phi( W_down · h ),                     (Eq. 1)

with ``W_down in R^{b x D}``, ``W_up in R^{D x b}``, ``b << D``, and ``phi`` a
non-linearity (GELU/ReLU). An optional LayerNorm on ``h`` precedes the bottleneck
(common in the Houlsby variant).

**(3) Placement.**
* **Houlsby**: an adapter after *both* the attention and the feed-forward
  sub-layers (two per transformer block).
* **Pfeiffer**: an adapter after the feed-forward sub-layer only (one per block) —
  fewer parameters, similar quality.
This library realises both by *wrapping the output projection* of the chosen
sub-layer(s); the choice is encoded in ``target_modules``.

**(4) Initialization (zero-delta).** ``W_up = 0`` (and its bias) so
``Adapter(h) = h`` exactly at step 0 — the pretrained model is preserved.

**(5) Forward pass.** Exactly Eq. 1, applied to the *output* of the wrapped layer.

**(6) Parameter count (derived).** Per adapter on a ``D``-dim output:

        params = (D·b + b)  [down]  + (b·D + D)  [up]  (+ 2D if LayerNorm)
               = 2 D b + b + D  (+ 2D).                                (Eq. 2)

**(7) Non-mergeability.** Because ``phi`` is non-linear, the adapter cannot be
folded into a linear weight; :meth:`AdapterModel.merge_and_unload` raises. Deploy
the adapter modules in place (they are tiny).

**(8) Connection to related methods.** Adapters add a *non-linear* residual;
LoRA/DoRA add a *linear* (foldable) one. Adapters generally use more parameters
but can model non-linear corrections the linear methods cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.exceptions import ConfigError, ShapeError
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec
from peft_lib.methods.lora import _infer_linear_dims

__all__ = ["AdapterConfig", "AdapterLayer", "AdapterModel"]

Activation = Literal["relu", "gelu", "tanh"]
AdapterType = Literal["houlsby", "pfeiffer"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


@dataclass(kw_only=True)
class AdapterConfig(PEFTConfig):
    """Configuration for bottleneck adapters.

    Attributes:
        target_modules: Output projections to attach adapters after (e.g.
            ``["o_proj", "down_proj"]`` for Houlsby, ``["down_proj"]`` for
            Pfeiffer), or ``"all-linear"``; ``None`` infers per arch.
        bottleneck_dim: The reduced dimension ``b`` (Eq. 1).
        non_linearity: Activation ``phi``: ``"relu"`` | ``"gelu"`` | ``"tanh"``.
        use_layernorm: Prepend a LayerNorm on ``h`` before the bottleneck.
        adapter_type: ``"houlsby"`` or ``"pfeiffer"`` — informational; the actual
            placement is governed by ``target_modules``.

    Example:
        >>> import peft_lib
        >>> from peft_lib import AdapterConfig
        >>> AdapterConfig(target_modules=["down_proj"], bottleneck_dim=16).peft_type
        'adapter'
    """

    target_modules: TargetSpec | None = None
    bottleneck_dim: int = 16
    non_linearity: Activation = "gelu"
    use_layernorm: bool = False
    adapter_type: AdapterType = "pfeiffer"

    def validate(self) -> None:
        """Validate adapter hyperparameters.

        Raises:
            ConfigError: If ``bottleneck_dim < 1``, or invalid ``non_linearity`` /
                ``adapter_type``.
        """
        super().validate()
        if self.bottleneck_dim < 1:
            raise ConfigError(f"bottleneck_dim must be >= 1, got {self.bottleneck_dim}.")
        if self.non_linearity not in _ACTIVATIONS:
            raise ConfigError(f"non_linearity must be one of {sorted(_ACTIVATIONS)}.")
        if self.adapter_type not in ("houlsby", "pfeiffer"):
            raise ConfigError(
                f"adapter_type must be 'houlsby'|'pfeiffer', got {self.adapter_type!r}."
            )


class AdapterLayer(nn.Module):
    r"""A frozen layer wrapped with a residual bottleneck adapter on its output (Eq. 1).

    Args:
        base_layer: The linear-like module whose output is adapted.
        bottleneck_dim: Reduced dimension ``b``.
        non_linearity: Activation key (``"relu"``/``"gelu"``/``"tanh"``).
        use_layernorm: Whether to LayerNorm the output before the bottleneck.

    Attributes:
        base_layer: The wrapped frozen module (exposed for inspection/unload).

    Note:
        This module intentionally does **not** implement ``merge``/``unmerge`` — a
        non-linear adapter is not foldable into a linear weight.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> lin = torch.nn.Linear(16, 32)
        >>> adapter = AdapterLayer(lin, bottleneck_dim=4)
        >>> x = torch.randn(2, 16)
        >>> torch.allclose(adapter(x), lin(x))  # W_up = 0 -> identity at init
        True
    """

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        bottleneck_dim: int = 16,
        non_linearity: Activation = "gelu",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 1:
            raise ConfigError(f"bottleneck_dim must be >= 1, got {bottleneck_dim}.")
        self.base_layer = base_layer
        _, self.out_features, _ = _infer_linear_dims(base_layer)
        dim = self.out_features

        self.adapter_norm: nn.Module = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.adapter_down = nn.Linear(dim, bottleneck_dim)
        self.adapter_act = _ACTIVATIONS[non_linearity]()
        self.adapter_up = nn.Linear(bottleneck_dim, dim)

        # Zero-delta init: up-projection starts at exactly zero (Eq. 4).
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped layer, then add the residual bottleneck (Eq. 1).

        Args:
            x: Input to the wrapped layer, shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``.

        Raises:
            ShapeError: If the wrapped layer's output dim is unexpected.
        """
        h: torch.Tensor = self.base_layer(x)  # (..., out)
        if h.shape[-1] != self.out_features:
            raise ShapeError(
                "AdapterLayer output dim mismatch",
                expected=(self.out_features,),
                actual=(h.shape[-1],),
            )
        dtype = self.adapter_down.weight.dtype
        z = self.adapter_norm(h.to(dtype))
        z = self.adapter_act(self.adapter_down(z))
        delta: torch.Tensor = self.adapter_up(z)
        return h + delta.to(h.dtype)

    def extra_repr(self) -> str:
        """Return a concise representation for ``print(model)``."""
        return f"out={self.out_features}, bottleneck={self.adapter_down.out_features}"


@register_peft("adapter")
class AdapterModel(InjectionPEFTModel):
    """Bottleneck-adapter wrapper: inserts :class:`AdapterLayer` residuals after targets.

    Because the adapters are non-linear, :meth:`merge_and_unload` (inherited) will
    raise :class:`~peft_lib.core.exceptions.MergeError`.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import AdapterConfig, get_peft_model
        >>> _ = torch.manual_seed(0)
        >>> base = nn.Sequential(nn.Linear(32, 32))
        >>> cfg = AdapterConfig(target_modules=["0"], bottleneck_dim=8)
        >>> peft = get_peft_model(base, cfg)
        >>> peft.get_nb_trainable_parameters()[0]  # 2*32*8 + 8 + 32
        552
    """

    config_class = AdapterConfig

    @property
    def adapter_config(self) -> AdapterConfig:
        """Return the config narrowed to :class:`AdapterConfig`."""
        assert isinstance(self.config, AdapterConfig)
        return self.config

    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        cfg = self.adapter_config
        return AdapterLayer(
            base_layer,
            bottleneck_dim=cfg.bottleneck_dim,
            non_linearity=cfg.non_linearity,
            use_layernorm=cfg.use_layernorm,
        )
