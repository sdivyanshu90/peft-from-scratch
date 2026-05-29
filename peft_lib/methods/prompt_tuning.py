r"""Prompt Tuning: The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021).

================================================================================
MATHEMATICAL DERIVATION  (notation: B=batch, S=seq, D=token_dim, P=num_virtual_tokens)
================================================================================

**(1) Problem setup.** Instead of changing any model weight, prepend a short
sequence of *learnable* embedding vectors ("soft prompt") to the input. The
frozen model conditions on them exactly as it would on real token embeddings.

**(2) Parameterization.** A single trainable matrix:

        P_e in R^{P x D},                                              (Eq. 1)

where ``D`` is the model's embedding dimension. Nothing else is trained.

**(3) Forward pass.** Given token ids with embeddings ``E in R^{B x S x D}``:

        E'   = [ P_e ; E ]            (prepend P virtual tokens)        (Eq. 2)
        mask'= [ 1_{BxP} ; mask ]     (attend to the prompt)
        y    = frozen_model(inputs_embeds=E', attention_mask=mask')

For loss, label positions for the virtual tokens are set to -100 (ignored).

**(4) Initialization.** Either ``P_e ~ N(0, sigma^2)`` (random) or sampled from
the model's own vocabulary embeddings (``init_from_vocab``), which Lester et al.
find trains faster on small models.

**(5) Backward.** Gradients flow only into ``P_e`` through the frozen model's
first attention layer. No weight in the backbone is updated.

**(6) Parameter count (derived).**

        params = P * D.                                                (Eq. 3)

For P=20 virtual tokens on a 768-dim model: ``20 * 768 = 15,360`` — independent of
model depth (contrast Prefix Tuning, which scales with the number of layers).

**(7) Scaling analysis.** The cheapest method by far for deep models; cost is
constant in depth. The trade-off is reduced capacity vs Prefix Tuning, which
Lester et al. show closes as model scale grows.

**(8) Connection to related methods.** Prompt Tuning == Prefix Tuning restricted
to the *input layer only* (no per-layer key/value prefixes). It has no weight
delta, so :meth:`PromptModel.merge_and_unload` raises — deploy by saving the
soft prompt alongside the frozen model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from peft_lib.core.base import PEFTConfig, PEFTModel
from peft_lib.core.exceptions import ConfigError, DeviceError, MergeError
from peft_lib.core.registry import register_peft
from peft_lib.core.utils import freeze_model

__all__ = ["PromptModel", "PromptTuningConfig", "SoftPromptEmbedding"]

PromptInit = Literal["random", "vocab"]


@dataclass(kw_only=True)
class PromptTuningConfig(PEFTConfig):
    """Configuration for soft Prompt Tuning.

    Attributes:
        num_virtual_tokens: Number of learnable prompt vectors ``P`` to prepend.
        token_dim: Embedding dimension ``D``; ``None`` infers it from the model.
        prompt_init: ``"random"`` (Gaussian) or ``"vocab"`` (sample real token
            embeddings).
        init_std: Std-dev for ``"random"`` init.

    Example:
        >>> import peft_lib
        >>> from peft_lib import PromptTuningConfig
        >>> PromptTuningConfig(num_virtual_tokens=20).peft_type
        'prompt_tuning'
    """

    num_virtual_tokens: int = 20
    token_dim: int | None = None
    prompt_init: PromptInit = "random"
    init_std: float = 0.02

    def validate(self) -> None:
        """Validate prompt-tuning hyperparameters.

        Raises:
            ConfigError: If ``num_virtual_tokens < 1`` or ``prompt_init`` invalid.
        """
        super().validate()
        if self.num_virtual_tokens < 1:
            raise ConfigError(f"num_virtual_tokens must be >= 1, got {self.num_virtual_tokens}.")
        if self.prompt_init not in ("random", "vocab"):
            raise ConfigError(f"prompt_init must be 'random'|'vocab', got {self.prompt_init!r}.")


class SoftPromptEmbedding(nn.Module):
    r"""The trainable soft-prompt matrix ``P_e in R^{P x D}`` (Eq. 1).

    Args:
        num_virtual_tokens: Number of prompt vectors ``P``.
        token_dim: Embedding dimension ``D``.
        init_std: Std-dev used for random Gaussian init.
        init_embeddings: Optional ``(P, D)`` tensor (e.g. sampled vocab vectors)
            to initialise from instead of Gaussian noise.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> sp = SoftPromptEmbedding(num_virtual_tokens=4, token_dim=8)
        >>> sp(batch_size=2).shape
        torch.Size([2, 4, 8])
    """

    def __init__(
        self,
        num_virtual_tokens: int,
        token_dim: int,
        *,
        init_std: float = 0.02,
        init_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.token_dim = token_dim
        if init_embeddings is not None:
            if tuple(init_embeddings.shape) != (num_virtual_tokens, token_dim):
                raise ConfigError(
                    f"init_embeddings must be {(num_virtual_tokens, token_dim)}, "
                    f"got {tuple(init_embeddings.shape)}."
                )
            weight = init_embeddings.detach().clone()
        else:
            weight = torch.randn(num_virtual_tokens, token_dim) * init_std
        self.embedding = nn.Parameter(weight)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return the soft prompt broadcast to a batch.

        Args:
            batch_size: The batch size ``B``.

        Returns:
            A tensor of shape ``(B, P, D)`` (a view via ``expand``).
        """
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)


@register_peft("prompt_tuning")
class PromptModel(PEFTModel):
    """Prompt-Tuning wrapper: prepends a learnable soft prompt to the input embeddings.

    Requires a backbone exposing ``get_input_embeddings()`` and accepting an
    ``inputs_embeds`` keyword (all HuggingFace transformer models do).

    Example:
        >>> from transformers import GPT2LMHeadModel, GPT2Config  # doctest: +SKIP
        >>> from peft_lib import PromptTuningConfig, get_peft_model  # doctest: +SKIP
        >>> m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32))  # doctest: +SKIP
        >>> peft = get_peft_model(m, PromptTuningConfig(num_virtual_tokens=5))  # doctest: +SKIP
        >>> peft.get_nb_trainable_parameters()[0]  # P*D  # doctest: +SKIP
        160
    """

    config_class = PromptTuningConfig

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        super().__init__(base_model, config)
        freeze_model(self.base_model)
        cfg = self.prompt_config
        token_dim = cfg.token_dim or self._infer_token_dim()
        init_embeddings = self._sampled_vocab(token_dim) if cfg.prompt_init == "vocab" else None
        self.prompt_encoder = SoftPromptEmbedding(
            cfg.num_virtual_tokens,
            token_dim,
            init_std=cfg.init_std,
            init_embeddings=init_embeddings,
        )

    @property
    def prompt_config(self) -> PromptTuningConfig:
        """Return the config narrowed to :class:`PromptTuningConfig`."""
        assert isinstance(self.config, PromptTuningConfig)
        return self.config

    def _input_embeddings(self) -> nn.Module:
        get = getattr(self.base_model, "get_input_embeddings", None)
        if get is None:
            raise DeviceError(
                f"{type(self.base_model).__name__} has no get_input_embeddings(); "
                "Prompt Tuning needs an embedding-based backbone."
            )
        embed: nn.Module = get()
        return embed

    def _embedding_weight(self) -> torch.Tensor:
        weight = self._input_embeddings().weight
        assert isinstance(weight, torch.Tensor)
        return weight

    def _infer_token_dim(self) -> int:
        return int(self._embedding_weight().shape[1])

    def _sampled_vocab(self, token_dim: int) -> torch.Tensor:
        """Sample ``num_virtual_tokens`` real vocabulary vectors for init (Eq. 4)."""
        weight = self._embedding_weight()
        n = self.prompt_config.num_virtual_tokens
        idx = torch.randint(0, weight.shape[0], (n,))
        return weight[idx].detach().clone()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Prepend the soft prompt and delegate to the frozen backbone (Eq. 2).

        Args:
            input_ids: Token ids ``(B, S)``; embedded via the frozen embedding if
                ``inputs_embeds`` is not given.
            attention_mask: ``(B, S)`` mask; extended with ones for the prompt.
            inputs_embeds: Precomputed ``(B, S, D)`` embeddings (alternative to ids).
            labels: ``(B, S)`` labels; prepended with -100 for the virtual tokens.
            **kwargs: Forwarded to the backbone.

        Returns:
            The backbone's output (e.g. a HuggingFace ``CausalLMOutput``).

        Raises:
            ConfigError: If neither ``input_ids`` nor ``inputs_embeds`` is given.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ConfigError("Provide either input_ids or inputs_embeds.")
            inputs_embeds = self._input_embeddings()(input_ids)
        batch = inputs_embeds.shape[0]
        n = self.prompt_config.num_virtual_tokens

        prompts = self.prompt_encoder(batch).to(inputs_embeds.dtype)  # (B, P, D)
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)  # (B, P+S, D)

        if attention_mask is not None:
            prefix = torch.ones(batch, n, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix, attention_mask], dim=1)
        if labels is not None:
            ignore = torch.full((batch, n), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore, labels], dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def merge_and_unload(self) -> nn.Module:
        """Prompt Tuning has no weight delta to fold.

        Raises:
            MergeError: Always; deploy by saving the soft prompt with the model.
        """
        raise MergeError(
            "Prompt Tuning has no weight delta to merge. Save the soft prompt "
            "(save_pretrained) and prepend it at inference instead."
        )
