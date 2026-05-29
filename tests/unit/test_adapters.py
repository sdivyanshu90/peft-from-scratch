"""Unit tests for bottleneck Adapters (non-linear, non-mergeable)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import AdapterConfig, get_peft_model
from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.methods.adapters import AdapterLayer

DIM = 32
B = 8


def test_forward_shape(tiny_model, random_ids):
    peft = get_peft_model(tiny_model, AdapterConfig(target_modules=["down_proj"], bottleneck_dim=B))
    assert peft(random_ids).shape == (2, 8, tiny_model.vocab)


def test_zero_delta_init():
    torch.manual_seed(0)
    lin = nn.Linear(DIM, DIM)
    x = torch.randn(2, 4, DIM)
    adapter = AdapterLayer(lin, bottleneck_dim=B)
    # W_up = 0 -> Adapter(h) == h.
    assert torch.allclose(adapter(x), lin(x), atol=1e-6)
    assert torch.count_nonzero(adapter.adapter_up.weight) == 0


def test_trainable_param_count():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(DIM, DIM))
    peft = get_peft_model(base, AdapterConfig(target_modules=["0"], bottleneck_dim=B))
    trn, _ = peft.get_nb_trainable_parameters()
    # down: D*b + b ; up: b*D + D  -> 2*D*b + b + D
    assert trn == 2 * DIM * B + B + DIM  # 552


def test_layernorm_adds_params():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(DIM, DIM))
    peft = get_peft_model(
        base, AdapterConfig(target_modules=["0"], bottleneck_dim=B, use_layernorm=True)
    )
    trn, _ = peft.get_nb_trainable_parameters()
    assert trn == 2 * DIM * B + B + DIM + 2 * DIM  # +LayerNorm weight & bias


def test_only_adapter_params_trainable(tiny_model):
    peft = get_peft_model(tiny_model, AdapterConfig(target_modules=["down_proj"]))
    trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert all("adapter_" in n for n in trainable)


def test_gradient_flows(tiny_model, random_ids):
    peft = get_peft_model(tiny_model, AdapterConfig(target_modules=["down_proj"], bottleneck_dim=B))
    peft(random_ids).pow(2).mean().backward()
    grads = {n: p.grad for n, p in peft.named_parameters() if p.requires_grad}
    assert all(g is not None for g in grads.values())
    # At step 0 the up-projection (initialised to 0) moves first; the down-proj
    # has zero gradient (it is gated by W_up == 0), mirroring LoRA's B-then-A.
    up = [g for n, g in grads.items() if "adapter_up" in n]
    assert any(g.abs().sum().item() > 0 for g in up)


def test_merge_and_unload_raises(tiny_model):
    peft = get_peft_model(tiny_model, AdapterConfig(target_modules=["down_proj"]))
    with pytest.raises(MergeError, match="cannot be merged"):
        peft.merge_and_unload()


def test_houlsby_targets_two_sublayers(tiny_model):
    # Houlsby = adapters after both attention-out and ff-out projections.
    peft = get_peft_model(
        tiny_model,
        AdapterConfig(target_modules=["o_proj", "down_proj"], adapter_type="houlsby"),
    )
    n_adapters = sum(1 for m in peft.modules() if isinstance(m, AdapterLayer))
    assert n_adapters == 2 * tiny_model.n_layers  # 2 sublayers * 2 layers


def test_save_load_roundtrip(tiny_model, tmp_path):
    peft = get_peft_model(tiny_model, AdapterConfig(target_modules=["down_proj"], bottleneck_dim=B))
    for _, p in peft.named_parameters():
        if p.requires_grad:
            p.data.normal_(std=0.1)
    ids = torch.randint(0, tiny_model.vocab, (2, 8))
    before = peft(ids).detach().clone()
    peft.save_pretrained(tmp_path)

    torch.manual_seed(42)
    fresh = type(tiny_model)()
    from peft_lib import AdapterModel

    reloaded = AdapterModel.from_pretrained(fresh, tmp_path)
    for k, v in peft.adapter_state_dict().items():
        assert torch.equal(v, reloaded.adapter_state_dict()[k])
    assert torch.allclose(before, reloaded(ids), atol=1e-6)


def test_config_serialization(tmp_path):
    cfg = AdapterConfig(target_modules=["down_proj"], bottleneck_dim=32, non_linearity="relu")
    assert AdapterConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert AdapterConfig.load(cfg.save(tmp_path)).bottleneck_dim == 32


@pytest.mark.parametrize(
    "kwargs", [{"bottleneck_dim": 0}, {"non_linearity": "bad"}, {"adapter_type": "x"}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        AdapterConfig(target_modules=["q"], **kwargs)
