r"""Prefix Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021).

================================================================================
MATHEMATICAL DERIVATION  (notation: B=batch, P=num_virtual_tokens, L=layers,
H=heads, d_h=head_dim, D=H*d_h=hidden)
================================================================================

**(1) Problem setup.** Prompt Tuning prepends learnable vectors only at the input
layer. Prefix Tuning instead prepends learnable **key/value pairs at every
attention layer**, giving the frozen model far more steering capacity per virtual
token (at the cost of depth-dependent parameters).

**(2) Parameterization.** For each layer ``l`` and each virtual position ``p`` we
learn a key ``k_{l,p}`` and value ``v_{l,p}`` in ``R^{H x d_h}``. Stacked, that is
a table

        Prefix in R^{P x (2 L D)}.                                     (Eq. 1)

Optionally (``prefix_projection=True``) this table is *reparameterised* through a
small MLP from a ``P x D`` embedding — Li & Liang found this stabilises
optimisation:

        Prefix = MLP( E ),  E in R^{P x D}.                            (Eq. 2)

**(3) Forward pass.** The prefix is reshaped to per-layer past key/values

        pkv[l] = (K_l, V_l),  K_l, V_l in R^{B x H x P x d_h},          (Eq. 3)

injected via the model's KV cache, and the attention mask is left-padded with
``P`` ones. Each layer's attention then attends over ``P + S`` positions; the
backbone weights are frozen.

**(4) Initialization.** The prefix table / embedding is initialised randomly
(N(0, sigma^2)). Unlike LoRA there is **no zero-delta init** — the prefixes
perturb attention from step 0 (this is intrinsic to the method, not a defect).

**(5) Backward.** Gradients flow into the prefix table (or the reparameterisation
MLP) through every attention layer; no backbone weight is updated.

**(6) Parameter count (derived).**
* Without projection: ``params = P * 2 * L * D``.                      (Eq. 4)
* With projection: ``P*D (embedding) + D*D_mlp + D_mlp + D_mlp*(2*L*D) + 2*L*D``.

For GPT-2 (L=12, D=768) with P=10 and no projection:
``10 * 2 * 12 * 768 = 184,320`` trainable parameters (see tests/regression). This
is the standard Prefix-Tuning count; configurations with a reparameterisation MLP
are larger.

**(7) Scaling analysis.** Parameters scale linearly with depth ``L`` — more
expensive than Prompt Tuning but more expressive, especially for small models.

**(8) Connection to related methods.** Prompt Tuning is Prefix Tuning truncated to
the input layer. Both are input-augmentation methods with no foldable weight
delta, so ``merge_and_unload`` raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from peft_lib.core.base import PEFTConfig, PEFTModel
from peft_lib.core.exceptions import ConfigError, DeviceError, MergeError
from peft_lib.core.registry import register_peft
from peft_lib.core.utils import freeze_model

__all__ = ["PrefixConfig", "PrefixEncoder", "PrefixModel"]


def _model_dims(model: nn.Module) -> tuple[int, int, int]:
    """Extract ``(hidden, num_layers, num_heads)`` from a HuggingFace model config.

    Args:
        model: A backbone exposing ``model.config`` with the usual attribute names.

    Returns:
        ``(hidden_size, num_layers, num_heads)``.

    Raises:
        DeviceError: If the dimensions cannot be read from the config.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise DeviceError("Prefix Tuning requires a model with a `.config`.")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if not (hidden and layers and heads):
        raise DeviceError(
            "Could not infer (hidden, layers, heads) from the model config; "
            "Prefix Tuning supports standard decoder transformers (e.g. GPT-2)."
        )
    return int(hidden), int(layers), int(heads)


@dataclass(kw_only=True)
class PrefixConfig(PEFTConfig):
    """Configuration for Prefix Tuning.

    Attributes:
        num_virtual_tokens: Number of prefix positions ``P`` per layer.
        prefix_projection: If ``True``, reparameterise the prefix through an MLP
            (Eq. 2); if ``False`` (default), learn the table directly (Eq. 1).
        encoder_hidden_size: Hidden width of the reparameterisation MLP; defaults
            to the model hidden size.
        init_std: Std-dev for random init.

    Example:
        >>> import peft_lib
        >>> from peft_lib import PrefixConfig
        >>> PrefixConfig(num_virtual_tokens=10).peft_type
        'prefix_tuning'
    """

    num_virtual_tokens: int = 10
    prefix_projection: bool = False
    encoder_hidden_size: int | None = None
    init_std: float = 0.02

    def validate(self) -> None:
        """Validate prefix-tuning hyperparameters.

        Raises:
            ConfigError: If ``num_virtual_tokens < 1``.
        """
        super().validate()
        if self.num_virtual_tokens < 1:
            raise ConfigError(f"num_virtual_tokens must be >= 1, got {self.num_virtual_tokens}.")


class PrefixEncoder(nn.Module):
    r"""Produces per-layer past key/values from a learnable prefix table (Eq. 1-3).

    Args:
        num_virtual_tokens: Prefix length ``P``.
        num_layers: Number of transformer layers ``L``.
        num_heads: Number of attention heads ``H``.
        hidden: Model hidden size ``D = H * d_h``.
        prefix_projection: Whether to reparameterise via an MLP (Eq. 2).
        encoder_hidden_size: MLP width (defaults to ``hidden``).
        init_std: Std-dev for random init.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> enc = PrefixEncoder(num_virtual_tokens=4, num_layers=2, num_heads=2, hidden=16)
        >>> pkv = enc(batch_size=3)
        >>> len(pkv), pkv[0][0].shape  # 2 layers; (B, H, P, d_h)
        (2, torch.Size([3, 2, 4, 8]))
    """

    def __init__(
        self,
        num_virtual_tokens: int,
        num_layers: int,
        num_heads: int,
        hidden: int,
        *,
        prefix_projection: bool = False,
        encoder_hidden_size: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ConfigError(f"hidden ({hidden}) must be divisible by num_heads ({num_heads}).")
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden = hidden
        self.head_dim = hidden // num_heads
        self.prefix_projection = prefix_projection
        out_dim = 2 * num_layers * hidden  # key + value per layer

        # arange(P) used to index the embedding; persistent buffer (not trained).
        self.register_buffer("prefix_tokens", torch.arange(num_virtual_tokens).long())

        if prefix_projection:
            enc_hidden = encoder_hidden_size or hidden
            self.embedding = nn.Embedding(num_virtual_tokens, hidden)
            self.transform: nn.Module = nn.Sequential(
                nn.Linear(hidden, enc_hidden),
                nn.Tanh(),
                nn.Linear(enc_hidden, out_dim),
            )
            nn.init.normal_(self.embedding.weight, std=init_std)
        else:
            self.embedding = nn.Embedding(num_virtual_tokens, out_dim)
            self.transform = nn.Identity()
            nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, batch_size: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Build the past-key-value tuple for a batch.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            A tuple of ``L`` ``(key, value)`` pairs, each tensor shaped
            ``(B, H, P, d_h)``.
        """
        tokens = self.prefix_tokens.to(self.embedding.weight.device)
        prefix = self.transform(self.embedding(tokens))  # (P, 2*L*D)
        # (P, 2L, H, d_h) -> (2L, P, H, d_h) per the cache layout.
        prefix = prefix.view(
            self.num_virtual_tokens, 2 * self.num_layers, self.num_heads, self.head_dim
        )
        prefix = prefix.permute(1, 0, 2, 3)  # (2L, P, H, d_h)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            key = prefix[2 * layer]  # (P, H, d_h)
            value = prefix[2 * layer + 1]
            # -> (B, H, P, d_h)
            key = key.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
            value = value.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
            out.append((key.contiguous(), value.contiguous()))
        return tuple(out)


@register_peft("prefix_tuning")
class PrefixModel(PEFTModel):
    """Prefix-Tuning wrapper: injects per-layer learnable key/values via the KV cache.

    Supports standard decoder transformers (e.g. GPT-2) whose ``forward`` accepts a
    legacy ``past_key_values`` tuple.

    Example:
        >>> from transformers import GPT2LMHeadModel, GPT2Config  # doctest: +SKIP
        >>> from peft_lib import PrefixConfig, get_peft_model  # doctest: +SKIP
        >>> m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32))  # doctest: +SKIP
        >>> peft = get_peft_model(m, PrefixConfig(num_virtual_tokens=5))  # doctest: +SKIP
        >>> peft.get_nb_trainable_parameters()[0]  # P*2*L*D  # doctest: +SKIP
        640
    """

    config_class = PrefixConfig

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        super().__init__(base_model, config)
        freeze_model(self.base_model)
        hidden, layers, heads = _model_dims(self.base_model)
        cfg = self.prefix_config
        self.prefix_encoder = PrefixEncoder(
            cfg.num_virtual_tokens,
            num_layers=layers,
            num_heads=heads,
            hidden=hidden,
            prefix_projection=cfg.prefix_projection,
            encoder_hidden_size=cfg.encoder_hidden_size,
            init_std=cfg.init_std,
        )

    @property
    def prefix_config(self) -> PrefixConfig:
        """Return the config narrowed to :class:`PrefixConfig`."""
        assert isinstance(self.config, PrefixConfig)
        return self.config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Attach the learnable prefix as past key/values and delegate (Eq. 3).

        Args:
            input_ids: Token ids ``(B, S)``.
            attention_mask: ``(B, S)`` mask; left-padded with ``P`` ones for the
                prefix. Built as all-ones if omitted.
            labels: ``(B, S)`` labels — left unchanged (the prefix lives in the KV
                cache, not in the token sequence).
            **kwargs: Forwarded to the backbone.

        Returns:
            The backbone's output.

        Raises:
            ConfigError: If ``input_ids`` is not provided.
        """
        if input_ids is None:
            raise ConfigError("Prefix Tuning requires input_ids.")
        batch, seq = input_ids.shape[0], input_ids.shape[1]
        n = self.prefix_config.num_virtual_tokens

        past_key_values = self.prefix_encoder(batch)
        if attention_mask is None:
            attention_mask = torch.ones(batch, seq, device=input_ids.device)
        prefix_mask = torch.ones(batch, n, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+S)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=False,
            **kwargs,
        )

    def merge_and_unload(self) -> nn.Module:
        """Prefix Tuning has no foldable weight delta.

        Raises:
            MergeError: Always; deploy by supplying the prefix KV cache at inference.
        """
        raise MergeError(
            "Prefix Tuning has no weight delta to merge; serve it by injecting the "
            "learned past_key_values at inference."
        )
