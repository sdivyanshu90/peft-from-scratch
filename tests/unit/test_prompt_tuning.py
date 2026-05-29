"""Unit tests for Prompt Tuning (input-augmentation; no foldable delta).

Uses a config-built tiny GPT-2 (no network). Marked ``hf`` since it needs the
transformers stack.
"""

from __future__ import annotations

import pytest
import torch

from peft_lib import PromptModel, PromptTuningConfig, get_peft_model
from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.methods.prompt_tuning import SoftPromptEmbedding

pytestmark = pytest.mark.hf

P = 5
DIM = 32


def test_soft_prompt_shape():
    sp = SoftPromptEmbedding(num_virtual_tokens=P, token_dim=DIM)
    assert sp(batch_size=3).shape == (3, P, DIM)
    assert sp.embedding.shape == (P, DIM)


def test_forward_prepends_prompt(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    ids = torch.randint(0, 64, (2, 6))
    out = peft(input_ids=ids, attention_mask=torch.ones(2, 6, dtype=torch.long))
    # Sequence length grows by P virtual tokens.
    assert out.logits.shape == (2, 6 + P, tiny_gpt2.config.vocab_size)


def test_trainable_param_count(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    trn, _ = peft.get_nb_trainable_parameters()
    assert trn == P * DIM  # 5 * 32 = 160


def test_only_prompt_is_trainable(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert trainable == ["prompt_encoder.embedding"]


def test_gradient_flows(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    ids = torch.randint(0, 64, (2, 6))
    out = peft(input_ids=ids, labels=ids)
    out.loss.backward()
    grad = peft.prompt_encoder.embedding.grad
    assert grad is not None and grad.abs().sum().item() > 0


def test_save_load_roundtrip(tiny_gpt2, tmp_path):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    peft.prompt_encoder.embedding.data.normal_()
    peft.save_pretrained(tmp_path)

    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    fresh = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=64)
    )
    reloaded = PromptModel.from_pretrained(fresh, tmp_path)
    assert torch.equal(
        peft.prompt_encoder.embedding.detach(),
        reloaded.prompt_encoder.embedding.detach(),
    )


def test_vocab_init_uses_real_embeddings(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P, prompt_init="vocab"))
    vocab_weight = tiny_gpt2.get_input_embeddings().weight.detach()
    # Each soft-prompt row must equal some real vocabulary row.
    for row in peft.prompt_encoder.embedding.detach():
        assert torch.any(torch.all(torch.isclose(vocab_weight, row, atol=1e-6), dim=1))


def test_merge_and_unload_raises(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PromptTuningConfig(num_virtual_tokens=P))
    with pytest.raises(MergeError, match="no weight delta"):
        peft.merge_and_unload()


def test_config_serialization(tmp_path):
    cfg = PromptTuningConfig(num_virtual_tokens=8, prompt_init="vocab", token_dim=64)
    assert PromptTuningConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert PromptTuningConfig.load(cfg.save(tmp_path)).num_virtual_tokens == 8


@pytest.mark.parametrize("kwargs", [{"num_virtual_tokens": 0}, {"prompt_init": "bad"}])
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        PromptTuningConfig(**kwargs)
