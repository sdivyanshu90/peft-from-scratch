r"""IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations (Liu et al., 2022).

================================================================================
MATHEMATICAL DERIVATION  (notation: out, in; ⊙ = elementwise product)
================================================================================

**(1) Problem setup.** The cheapest possible adaptation rescales activations
rather than adding new weight matrices. IA³ learns one vector per targeted
projection that elementwise-multiplies an activation, ``inhibiting`` (<1) or
``amplifying`` (>1) each feature.

**(2) Two placements.** IA³ rescales three activation sites in a transformer:
keys, values, and the feed-forward inner activation. Two algebraic forms cover
all three:

* **Output rescaling** (keys/values, and any "non-feedforward" target). With a
  learned ``l in R^{out}``:

        y = l ⊙ (W0 x + b0).                                          (Eq. 1)

* **Input rescaling** (the feed-forward up-projection). With ``l in R^{in}``:

        y = W0 (l ⊙ x) + b0.                                         (Eq. 2)

**(3) Initialization (zero-delta).** ``l = 1`` (all ones) -> the rescaling is the
identity, so the adapted model equals the pretrained one at step 0.

**(4) Forward pass.** Exactly Eq. 1 / Eq. 2 — a single broadcast multiply plus the
frozen base layer; no extra matmul.

**(5) Backward.** ``dL/dl`` is an elementwise product of the upstream gradient
with the (frozen) base output (Eq. 1) or input (Eq. 2). ``W0`` receives no
gradient (frozen).

**(6) Parameter count (derived).** Per adapted layer:

        params = out   (output rescaling)   or   in   (input rescaling).  (Eq. 3)

This is the smallest of any method here — one scalar per feature, no rank.

**(7) Merge equivalence.** Both forms fold losslessly into the base weight:
        output:  W' = diag(l) W0 ,  b' = l ⊙ b0      (scale rows)
        input:   W' = W0 diag(l)                      (scale columns)
so merged inference has exactly zero overhead.

**(8) Connection to related methods.** IA³ is VeRA with ``r -> `` and only the
output scaling kept, or LoRA with the low-rank term replaced by a diagonal — the
extreme end of the parameter-efficiency / expressivity trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.exceptions import ConfigError, MergeError, ShapeError
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec
from peft_lib.methods.lora import _infer_linear_dims

__all__ = ["IA3Config", "IA3Linear", "IA3Model"]


@dataclass(kw_only=True)
class IA3Config(PEFTConfig):
    """Configuration for IA³.

    Attributes:
        target_modules: Suffix list of projections to rescale (e.g. ``["k", "v",
            "wi"]`` for T5), or ``"all-linear"``; ``None`` infers per arch.
        feedforward_modules: Subset of ``target_modules`` treated as feed-forward
            (input rescaling, Eq. 2). Others use output rescaling (Eq. 1). Must be
            a subset of the targets.
        dropout: Input dropout on the rescaled path.
        init_ia3_weights: If ``True`` (default), initialise ``l = 1`` (zero delta).

    Example:
        >>> import peft_lib
        >>> from peft_lib import IA3Config
        >>> IA3Config(target_modules=["k", "v", "wi"], feedforward_modules=["wi"]).peft_type
        'ia3'
    """

    target_modules: TargetSpec | None = None
    feedforward_modules: list[str] = field(default_factory=list)
    dropout: float = 0.0
    init_ia3_weights: bool = True

    def validate(self) -> None:
        """Validate IA³ hyperparameters.

        Raises:
            ConfigError: If ``dropout`` not in [0, 1), or ``feedforward_modules``
                is not a subset of an explicit ``target_modules`` list.
        """
        super().validate()
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.feedforward_modules and isinstance(self.target_modules, list):
            extra = set(self.feedforward_modules) - set(self.target_modules)
            if extra:
                raise ConfigError(f"feedforward_modules {sorted(extra)} are not in target_modules.")


class IA3Linear(nn.Module):
    r"""A frozen linear layer with a learned IA³ rescaling vector (Eq. 1 / Eq. 2).

    Args:
        base_layer: The linear-like module to adapt.
        is_feedforward: If ``True``, rescale the **input** (Eq. 2, ``l`` length
            ``in``); else rescale the **output** (Eq. 1, ``l`` length ``out``).
        dropout: Input dropout on the rescaled path.
        init_ia3_weights: If ``True``, initialise ``l = 1`` (zero delta).

    Attributes:
        merged: Whether ``l`` currently lives in ``base_layer.weight``.
        ia3_l: The trainable rescaling vector.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> lin = torch.nn.Linear(16, 32)
        >>> ia3 = IA3Linear(lin, is_feedforward=False)
        >>> x = torch.randn(2, 16)
        >>> torch.allclose(ia3(x), lin(x))  # l == 1 -> identity
        True
    """

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        is_feedforward: bool = False,
        dropout: float = 0.0,
        init_ia3_weights: bool = True,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features, self.out_features, self.fan_in_fan_out = _infer_linear_dims(base_layer)
        self.is_feedforward = is_feedforward
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        length = self.in_features if is_feedforward else self.out_features
        init = torch.ones(length) if init_ia3_weights else torch.empty(length).uniform_(0.9, 1.1)
        self.ia3_l = nn.Parameter(init)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the IA³ output (Eq. 1 for output-rescale, Eq. 2 for input-rescale).

        Args:
            x: Input of shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``.

        Raises:
            ShapeError: If ``x``'s last dim is not ``in_features``.
        """
        if x.shape[-1] != self.in_features:
            raise ShapeError(
                "IA3Linear input has wrong feature dim",
                expected=(self.in_features,),
                actual=(x.shape[-1],),
            )
        if self.merged:
            merged_out: torch.Tensor = self.base_layer(x)
            return merged_out
        if self.is_feedforward:
            scaled = self.lora_dropout(x) * self.ia3_l.to(x.dtype)  # (..., in)
            ff_out: torch.Tensor = self.base_layer(scaled)
            return ff_out
        out: torch.Tensor = self.base_layer(x)  # (..., out)
        return out * self.ia3_l.to(out.dtype)

    def merge(self) -> None:
        """Fold ``l`` into ``base_layer.weight`` (and bias for output-rescale).

        Raises:
            MergeError: If already merged.
        """
        if self.merged:
            raise MergeError("IA3Linear is already merged.")
        weight = self.base_layer.weight
        factor = self.ia3_l.to(weight.dtype)
        bias = getattr(self.base_layer, "bias", None)
        with torch.no_grad():
            if self.is_feedforward:
                # Scale columns (input dim). Canonical (out,in)->dim1; Conv1D (in,out)->dim0.
                weight.mul_(factor.view(-1, 1) if self.fan_in_fan_out else factor.view(1, -1))
            else:
                # Scale rows (output dim). Canonical (out,in)->dim0; Conv1D (in,out)->dim1.
                weight.mul_(factor.view(1, -1) if self.fan_in_fan_out else factor.view(-1, 1))
                if bias is not None:
                    bias.mul_(factor)
        self.merged = True

    def unmerge(self) -> None:
        """Divide ``l`` back out of ``base_layer.weight`` (inverse of :meth:`merge`).

        Raises:
            MergeError: If not merged, or if any ``l`` entry is ~0 (non-invertible).
        """
        if not self.merged:
            raise MergeError("IA3Linear is not merged; nothing to unmerge.")
        if torch.any(self.ia3_l.abs() < 1e-8):
            raise MergeError("IA3 rescaling has a ~zero entry; merge is not invertible.")
        weight = self.base_layer.weight
        factor = self.ia3_l.to(weight.dtype)
        bias = getattr(self.base_layer, "bias", None)
        with torch.no_grad():
            if self.is_feedforward:
                weight.div_(factor.view(-1, 1) if self.fan_in_fan_out else factor.view(1, -1))
            else:
                weight.div_(factor.view(1, -1) if self.fan_in_fan_out else factor.view(-1, 1))
                if bias is not None:
                    bias.div_(factor)
        self.merged = False

    def extra_repr(self) -> str:
        """Return a concise representation for ``print(model)``."""
        kind = "feedforward(input)" if self.is_feedforward else "output"
        return f"in={self.in_features}, out={self.out_features}, rescale={kind}"


@register_peft("ia3")
class IA3Model(InjectionPEFTModel):
    """IA³-wrapped model: injects :class:`IA3Linear` rescalers into the frozen backbone.

    A target is treated as feed-forward (input rescaling) iff its name leaf is in
    ``config.feedforward_modules``.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import IA3Config, get_peft_model
        >>> _ = torch.manual_seed(0)
        >>> base = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        >>> cfg = IA3Config(target_modules=["0", "1"], feedforward_modules=["1"])
        >>> peft = get_peft_model(base, cfg)
        >>> peft.get_nb_trainable_parameters()[0]  # out(16) + in(16) = 32
        32
    """

    config_class = IA3Config

    @property
    def ia3_config(self) -> IA3Config:
        """Return the config narrowed to :class:`IA3Config`."""
        assert isinstance(self.config, IA3Config)
        return self.config

    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        cfg = self.ia3_config
        leaf = module_name.rpartition(".")[2]
        is_ff = leaf in cfg.feedforward_modules
        return IA3Linear(
            base_layer,
            is_feedforward=is_ff,
            dropout=cfg.dropout,
            init_ia3_weights=cfg.init_ia3_weights,
        )
