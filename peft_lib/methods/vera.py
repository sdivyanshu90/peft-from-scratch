r"""VeRA: Vector-based Random Matrix Adaptation (Kopiczko et al., 2024).

================================================================================
MATHEMATICAL DERIVATION  (notation: out, in, r; Λ = diag(.))
================================================================================

**(1) Problem setup.** LoRA's trainable count grows with ``r*(in+out)``. VeRA
asks: what if the low-rank *bases* ``A, B`` were **frozen, random, and shared**
across all layers, and we only learned tiny per-layer scaling vectors? Random
projections preserve enough structure (Johnson-Lindenstrauss) that learning how
to *combine* them suffices.

**(2) Parameterization.** Fix two random matrices, shared across every adapted
layer and never trained:

        A in R^{r x in}   (frozen),     B in R^{out x r}   (frozen).

Learn two small per-layer vectors:

        d in R^{r}        (scales the r shared directions),
        b in R^{out}      (scales the out outputs).

The update is:

        ΔW = Λ_b B Λ_d A = diag(b) · B · diag(d) · A.                  (Eq. 1)

**(3) Initialization (zero-delta).** ``A, B`` ~ uniform (deterministically seeded
so they can be regenerated, never stored). ``d = d_init`` (a small constant, e.g.
0.1) and ``b = 0``. With ``b = 0``, ``ΔW = 0`` -> the adapted model equals the
pretrained one at step 0.

**(4) Forward pass.** Never materialise ``ΔW``; apply factor by factor:

        h1 = dropout(x) A^T      -> (..., r)
        h2 = h1 ⊙ d              -> (..., r)
        h3 = h2 B^T              -> (..., out)
        ΔY = h3 ⊙ b              -> (..., out)
        y  = W0 x + b0 + ΔY                                            (Eq. 2)

**(5) Backward.** Gradients flow only to ``d`` and ``b`` (``A, B`` are frozen):
``dL/db = (dL/dy) ⊙ h3`` and ``dL/dd = ((dL/dy ⊙ b) B) ⊙ h1`` — both cheap
elementwise/low-rank products.

**(6) Parameter count (derived).** Per adapted layer:

        params = r (vector d) + out (vector b) = r + out.              (Eq. 3)

The frozen shared ``A, B`` cost **zero** trainable parameters and are *not stored*
in the checkpoint (regenerated from ``projection_seed``). For r=256 on a 768-dim
layer that is ``256 + 768 = 1024`` trainable vs LoRA's ``r*(in+out)`` — a ~10x
reduction at equal expressive rank.

**(7) Scaling analysis.** Trainable params are independent of ``in`` for the ``d``
vector, enabling very high effective rank at near-constant cost. The frozen
matrices add ``r*(in+out)`` *non-trainable* memory per shape (shared in a
production setting; this reference impl regenerates them per layer for clarity).

**(8) Connection to related methods.** VeRA = LoRA with ``A, B`` frozen+shared and
two diagonal scalings made trainable. IA³ is the degenerate ``r -> `` case that
keeps only the output scaling ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.exceptions import ConfigError, MergeError, ShapeError
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec
from peft_lib.methods.lora import _infer_linear_dims

__all__ = ["VeRAConfig", "VeRALinear", "VeRAModel"]


def _seeded_uniform(rows: int, cols: int, fan_in: int, seed: int) -> torch.Tensor:
    """Deterministically generate a uniform-init matrix from ``seed``.

    Uses a private :class:`torch.Generator` so the result is independent of global
    RNG state — essential because these matrices are *regenerated* (not stored) at
    load time and must match bit-for-bit.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        fan_in: Fan-in used for the Kaiming-style uniform bound.
        seed: Generator seed.

    Returns:
        A ``(rows, cols)`` tensor of values in ``[-bound, bound]``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bound = (1.0 / fan_in) ** 0.5
    return torch.empty(rows, cols).uniform_(-bound, bound, generator=generator)


@dataclass(kw_only=True)
class VeRAConfig(PEFTConfig):
    """Configuration for VeRA.

    Attributes:
        r: Shared random-projection rank. VeRA uses *large* r (params don't scale
            with it); defaults to 256.
        target_modules: Suffix list or ``"all-linear"``; ``None`` infers per arch.
        d_init: Initial value of the ``d`` scaling vector.
        projection_seed: Seed for the frozen shared ``A, B`` (regenerated on load).
        dropout: Input dropout on the adapter path.

    Example:
        >>> import peft_lib
        >>> from peft_lib import VeRAConfig
        >>> VeRAConfig(r=64, target_modules=["q_proj"]).peft_type
        'vera'
    """

    r: int = 256
    target_modules: TargetSpec | None = None
    d_init: float = 0.1
    projection_seed: int = 0
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate VeRA hyperparameters.

        Raises:
            ConfigError: If ``r < 1`` or ``dropout`` not in [0, 1).
        """
        super().validate()
        if self.r < 1:
            raise ConfigError(f"VeRA rank r must be >= 1, got {self.r}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}.")


class VeRALinear(nn.Module):
    r"""A frozen linear layer adapted by VeRA's shared-random-matrix scheme (Eq. 1-2).

    Args:
        base_layer: The linear-like module to adapt.
        r: Shared projection rank.
        d_init: Initial value of the ``d`` vector.
        projection_seed: Seed for the frozen ``A, B`` matrices.
        dropout: Input dropout on the adapter path.

    Attributes:
        merged: Whether ``ΔW`` currently lives in ``base_layer.weight``.
        vera_lambda_d: Trainable ``d`` vector of shape ``(r,)``.
        vera_lambda_b: Trainable ``b`` vector of shape ``(out,)``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> lin = torch.nn.Linear(16, 32)
        >>> vera = VeRALinear(lin, r=8)
        >>> x = torch.randn(2, 16)
        >>> torch.allclose(vera(x), lin(x), atol=1e-6)  # zero-delta (b=0)
        True
    """

    vera_A: torch.Tensor
    vera_B: torch.Tensor

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        r: int = 256,
        d_init: float = 0.1,
        projection_seed: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ConfigError(f"VeRA rank r must be >= 1, got {r}.")
        self.base_layer = base_layer
        self.in_features, self.out_features, self.fan_in_fan_out = _infer_linear_dims(base_layer)
        self.r = r
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Frozen, shared (by seed), regenerated-on-load random projections.
        self.register_buffer(
            "vera_A", _seeded_uniform(r, self.in_features, self.in_features, seed=projection_seed)
        )
        self.register_buffer(
            "vera_B", _seeded_uniform(self.out_features, r, r, seed=projection_seed + 1)
        )

        # Trainable per-layer scaling vectors (Eq. 1).
        self.vera_lambda_d = nn.Parameter(torch.full((r,), float(d_init)))
        self.vera_lambda_b = nn.Parameter(torch.zeros(self.out_features))

        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:
        r"""Return ``ΔW = diag(b) B diag(d) A`` (Eq. 1), shape ``(out, in)``."""
        # diag(d) A -> scale rows of A; B @ that; diag(b) -> scale rows.
        scaled_a = self.vera_lambda_d.unsqueeze(1) * self.vera_A  # (r, in)
        bda = self.vera_B @ scaled_a  # (out, in)
        return self.vera_lambda_b.unsqueeze(1) * bda  # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the VeRA output (Eq. 2).

        Args:
            x: Input of shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``.

        Raises:
            ShapeError: If ``x``'s last dim is not ``in_features``.
        """
        if x.shape[-1] != self.in_features:
            raise ShapeError(
                "VeRALinear input has wrong feature dim",
                expected=(self.in_features,),
                actual=(x.shape[-1],),
            )
        result: torch.Tensor = self.base_layer(x)
        if self.merged:
            return result
        dtype = self.vera_lambda_d.dtype
        x_drop: torch.Tensor = self.lora_dropout(x).to(dtype)
        h1 = x_drop @ self.vera_A.to(dtype).transpose(0, 1)  # (..., r)
        h2 = h1 * self.vera_lambda_d  # (..., r)
        h3 = h2 @ self.vera_B.to(dtype).transpose(0, 1)  # (..., out)
        delta = h3 * self.vera_lambda_b  # (..., out)
        return result + delta.to(result.dtype)

    def merge(self) -> None:
        """Fold ``ΔW`` into ``base_layer.weight`` in place (idempotent-guarded).

        Raises:
            MergeError: If already merged.
        """
        if self.merged:
            raise MergeError("VeRALinear is already merged.")
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            self.base_layer.weight.add_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self.merged = True

    def unmerge(self) -> None:
        """Subtract ``ΔW`` back out of ``base_layer.weight`` in place.

        Raises:
            MergeError: If not currently merged.
        """
        if not self.merged:
            raise MergeError("VeRALinear is not merged; nothing to unmerge.")
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            self.base_layer.weight.sub_(delta.transpose(0, 1) if self.fan_in_fan_out else delta)
        self.merged = False

    def extra_repr(self) -> str:
        """Return a concise representation for ``print(model)``."""
        return f"in={self.in_features}, out={self.out_features}, r={self.r}"


@register_peft("vera")
class VeRAModel(InjectionPEFTModel):
    """VeRA-wrapped model: injects :class:`VeRALinear` adapters into the frozen backbone.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import VeRAConfig, get_peft_model
        >>> _ = torch.manual_seed(0)
        >>> peft = get_peft_model(nn.Linear(32, 16), VeRAConfig(r=64, target_modules=[""]))
        >>> peft.get_nb_trainable_parameters()[0]  # r + out = 64 + 16
        80
    """

    config_class = VeRAConfig

    @property
    def vera_config(self) -> VeRAConfig:
        """Return the config narrowed to :class:`VeRAConfig`."""
        assert isinstance(self.config, VeRAConfig)
        return self.config

    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        cfg = self.vera_config
        return VeRALinear(
            base_layer,
            r=cfg.r,
            d_init=cfg.d_init,
            projection_seed=cfg.projection_seed,
            dropout=cfg.dropout,
        )
