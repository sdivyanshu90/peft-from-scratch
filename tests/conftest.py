"""Shared pytest fixtures and helpers for the peft_lib test suite.

Two families of fixtures:

* **Synthetic** (`tiny_model`, `random_ids`, `random_input`): tiny hand-rolled
  modules with predictably named ``nn.Linear`` submodules. Fast, deterministic,
  dependency-free — used by every unit/property test.
* **HuggingFace** (`tiny_gpt2`, `tiny_llama`, `tiny_t5`, `tiny_bert`): real
  architectures built *from config* (no network/pretrained download). Used by
  integration and regression tests; gated behind ``pytest.importorskip``.

A session-wide autouse fixture seeds ``torch.manual_seed(42)`` before every test,
honouring the project's "consistent seed" rule without per-test boilerplate.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

SEED = 42


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed torch (and CUDA, if present) to 42 before each test."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Synthetic transformer-shaped model
# ---------------------------------------------------------------------------
class TinyBlock(nn.Module):
    """A minimal transformer-ish block with conventionally named projections.

    Exposes ``q_proj``/``k_proj``/``v_proj``/``o_proj`` (square, ``d x d``) and an
    MLP ``up_proj`` (``d -> ff``) / ``down_proj`` (``ff -> d``), so tests can target
    modules by the same suffixes used for real LLaMA-style models.
    """

    def __init__(self, d: int = 32, ff: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.up_proj = nn.Linear(d, ff)
        self.down_proj = nn.Linear(ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        x = x + attn
        return x + self.down_proj(torch.relu(self.up_proj(x)))


class TinyModel(nn.Module):
    """A tiny token model: embedding -> N TinyBlocks -> linear head.

    Attributes:
        d: Hidden dim. ff: MLP inner dim. vocab: vocabulary size.
    """

    def __init__(self, d: int = 32, ff: int = 64, n_layers: int = 2, vocab: int = 50) -> None:
        super().__init__()
        self.d = d
        self.ff = ff
        self.vocab = vocab
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyBlock(d, ff) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)  # (B, S, d)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)  # (B, S, vocab)


@pytest.fixture
def tiny_model() -> TinyModel:
    """A freshly initialised :class:`TinyModel` (d=32, ff=64, 2 layers, vocab=50)."""
    torch.manual_seed(SEED)
    return TinyModel()


@pytest.fixture
def random_ids() -> torch.Tensor:
    """Integer token ids of shape ``(B=2, S=8)`` valid for :class:`TinyModel`."""
    torch.manual_seed(SEED)
    return torch.randint(0, 50, (2, 8))


@pytest.fixture
def random_input() -> torch.Tensor:
    """A float activation batch of shape ``(B=2, S=8, D=32)`` for direct layer tests."""
    torch.manual_seed(SEED)
    return torch.randn(2, 8, 32)


# ---------------------------------------------------------------------------
# Real (config-only) HuggingFace models — no pretrained download
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_gpt2() -> nn.Module:
    """A 2-layer GPT-2 LM built from config (uses HF ``Conv1D`` projections)."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(SEED)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=64)
    return GPT2LMHeadModel(cfg)


@pytest.fixture
def tiny_llama() -> nn.Module:
    """A 2-layer LLaMA model built from config (uses ``nn.Linear`` projections)."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


@pytest.fixture
def tiny_bert() -> nn.Module:
    """A 2-layer BERT built from config (projections named ``query``/``value``)."""
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertModel

    torch.manual_seed(SEED)
    cfg = BertConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=64,
        max_position_embeddings=64,
    )
    return BertModel(cfg)


@pytest.fixture
def tiny_t5() -> nn.Module:
    """A tiny T5 (encoder-decoder) built from config; projections named ``q``/``v``/``wi``."""
    pytest.importorskip("transformers")
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(SEED)
    cfg = T5Config(
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_layers=2,
        num_heads=4,
        vocab_size=64,
    )
    return T5ForConditionalGeneration(cfg)


@pytest.fixture
def trainable_grad_names():
    """Return a helper that lists trainable param names containing a marker substring."""

    def _names(model: nn.Module, marker: str) -> list[str]:
        return [n for n, p in model.named_parameters() if p.requires_grad and marker in n]

    return _names
