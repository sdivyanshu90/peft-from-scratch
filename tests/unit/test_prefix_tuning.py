"""Unit tests for Prefix Tuning (per-layer KV prefixes; no foldable delta)."""

from __future__ import annotations

import pytest
import torch

from peft_lib import PrefixConfig, PrefixModel, get_peft_model
from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.methods.prefix_tuning import PrefixEncoder

pytestmark = pytest.mark.hf

P = 5
L = 2
H = 2
DIM = 32


def test_encoder_produces_correct_pkv_shapes():
    enc = PrefixEncoder(num_virtual_tokens=P, num_layers=L, num_heads=H, hidden=DIM)
    pkv = enc(batch_size=3)
    assert len(pkv) == L
    for key, value in pkv:
        assert key.shape == (3, H, P, DIM // H)
        assert value.shape == (3, H, P, DIM // H)


def test_forward_runs_and_keeps_seq_len(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    ids = torch.randint(0, 64, (2, 6))
    out = peft(input_ids=ids)
    # Prefix lives in the KV cache, so the output seq length is unchanged.
    assert out.logits.shape == (2, 6, tiny_gpt2.config.vocab_size)


def test_trainable_param_count_no_projection(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    trn, _ = peft.get_nb_trainable_parameters()
    assert trn == P * 2 * L * DIM  # 5 * 2 * 2 * 32 = 640


def test_only_prefix_is_trainable(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
    assert all(n.startswith("prefix_encoder") for n in trainable)
    assert any("base_model" in n for n, p in peft.named_parameters() if not p.requires_grad)


def test_gradient_flows(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    ids = torch.randint(0, 64, (2, 6))
    peft(input_ids=ids, labels=ids).loss.backward()
    grad = peft.prefix_encoder.embedding.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0


def test_projection_variant_has_mlp(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P, prefix_projection=True))
    names = {n for n, p in peft.named_parameters() if p.requires_grad}
    # The reparameterisation MLP adds transform.* parameters.
    assert any("transform" in n for n in names)
    ids = torch.randint(0, 64, (2, 6))
    assert peft(input_ids=ids).logits.shape == (2, 6, 64)


def test_save_load_roundtrip(tiny_gpt2, tmp_path):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    peft.prefix_encoder.embedding.weight.data.normal_()
    peft.save_pretrained(tmp_path)

    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    fresh = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=64)
    )
    reloaded = PrefixModel.from_pretrained(fresh, tmp_path)
    assert torch.equal(
        peft.prefix_encoder.embedding.weight.detach(),
        reloaded.prefix_encoder.embedding.weight.detach(),
    )


def test_merge_and_unload_raises(tiny_gpt2):
    peft = get_peft_model(tiny_gpt2, PrefixConfig(num_virtual_tokens=P))
    with pytest.raises(MergeError, match="no weight delta"):
        peft.merge_and_unload()


def test_config_serialization(tmp_path):
    cfg = PrefixConfig(num_virtual_tokens=10, prefix_projection=True, encoder_hidden_size=128)
    assert PrefixConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert PrefixConfig.load(cfg.save(tmp_path)).prefix_projection is True


def test_invalid_config():
    with pytest.raises(ConfigError):
        PrefixConfig(num_virtual_tokens=0)
