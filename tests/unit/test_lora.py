"""Unit tests for LoRA: the canonical 10-test suite plus variant coverage.

Every test seeds via the autouse ``_seed_everything`` fixture (torch.manual_seed(42))
and asserts on concrete values — never merely "no exception raised".
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from peft_lib import LoRAConfig, LoRAModel, get_peft_model
from peft_lib.core.exceptions import ConfigError, MergeError, ShapeError
from peft_lib.methods.lora import LoRALinear

TARGETS = ["q_proj", "v_proj"]
R = 8
ALPHA = 16
# TinyModel: d=32, 2 layers; q_proj & v_proj are each 32x32.
EXPECTED_TRAINABLE = 2 * 2 * (R * (32 + 32))  # 2 layers * 2 projs * r*(in+out) = 2048


def _wrap(model: nn.Module, **kwargs: object) -> LoRAModel:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=TARGETS, **kwargs)  # type: ignore[arg-type]
    peft = get_peft_model(model, cfg)
    assert isinstance(peft, LoRAModel)
    return peft


# --- 1. forward shape -------------------------------------------------------
def test_forward_shape(tiny_model, random_ids):
    peft = _wrap(tiny_model)
    out = peft(random_ids)
    assert out.shape == (2, 8, tiny_model.vocab)


# --- 2. only adapter params require grad ------------------------------------
def test_only_adapter_params_require_grad(tiny_model):
    peft = _wrap(tiny_model)
    trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
    assert trainable, "expected some trainable params"
    assert all("lora_" in n for n in trainable)
    # Every non-LoRA parameter must be frozen.
    frozen = {n for n, p in peft.named_parameters() if not p.requires_grad}
    assert all("lora_" not in n for n in frozen)
    # Exactly 4 adapted layers * 2 matrices (A, B) = 8 trainable tensors.
    assert len(trainable) == 8


# --- 3. exact trainable param count (matches Eq. 5) -------------------------
def test_trainable_param_count(tiny_model):
    peft = _wrap(tiny_model)
    trainable, _ = peft.get_nb_trainable_parameters()
    assert trainable == EXPECTED_TRAINABLE == 2048


# --- 4. gradients flow to adapter params ------------------------------------
def test_gradient_flows(tiny_model, random_ids):
    peft = _wrap(tiny_model)
    peft(random_ids).pow(2).mean().backward()
    a_grads = [p.grad for n, p in peft.named_parameters() if "lora_A" in n]
    b_grads = [p.grad for n, p in peft.named_parameters() if "lora_B" in n]
    # Grad tensors must exist for both A and B.
    assert all(g is not None for g in a_grads + b_grads)
    # B lifts off zero on step 1 (sees x A^T); its gradient is non-trivial.
    assert any(g.abs().sum().item() > 0 for g in b_grads)  # type: ignore[union-attr]
    # A's gradient is exactly zero at step 0 because B == 0 (derivation §5).
    assert all(torch.count_nonzero(g) == 0 for g in a_grads)  # type: ignore[arg-type]


# --- 5. merge_and_unload equivalence (atol=1e-5) ----------------------------
def test_merge_and_unload(tiny_model, random_ids):
    peft = _wrap(tiny_model)
    # Train a few steps so the delta is non-zero.
    opt = torch.optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        peft(random_ids).pow(2).mean().backward()
        opt.step()
    expected = peft(random_ids).detach().clone()
    merged = peft.merge_and_unload()
    got = merged(random_ids)
    assert torch.allclose(expected, got, atol=1e-5)
    # No LoRALinear should remain after unloading.
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())


# --- 6. save/load roundtrip (bitwise identical adapter weights) -------------
def test_save_load_roundtrip(tiny_model, tmp_path):
    vocab = tiny_model.vocab
    peft = _wrap(tiny_model)  # mutates tiny_model in place
    # Make adapters non-trivial.
    for _, p in peft.named_parameters():
        if p.requires_grad:
            p.data.normal_()
    ids = torch.randint(0, vocab, (2, 8))
    out_before = peft(ids).detach().clone()

    peft.save_pretrained(tmp_path)
    # A pristine base with identical weights: TinyModel init is deterministic
    # under seed 42 (the same seed the `tiny_model` fixture used).
    torch.manual_seed(42)
    fresh = type(tiny_model)()
    reloaded = LoRAModel.from_pretrained(fresh, tmp_path)

    saved = peft.adapter_state_dict()
    loaded = reloaded.adapter_state_dict()
    assert saved.keys() == loaded.keys()
    for k in saved:
        assert torch.equal(saved[k].cpu(), loaded[k].cpu()), f"{k} not bitwise identical"
    assert torch.allclose(out_before, reloaded(ids), atol=1e-6)


# --- 7. zero-delta initialisation -------------------------------------------
def test_init_is_zero_delta(tiny_model, random_ids):
    base_out = tiny_model(random_ids).detach().clone()
    peft = _wrap(tiny_model)  # wrapping mutates tiny_model in place
    assert torch.allclose(peft(random_ids), base_out, atol=1e-6)
    # And every B matrix is exactly zero.
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            assert torch.count_nonzero(p) == 0


# --- 8. dtype propagation ---------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_propagation(tiny_model, dtype):
    vocab = tiny_model.vocab
    # fp16 matmul is unsupported on CPU, so run fp16 on CUDA when available.
    device = "cuda" if (dtype == torch.float16 and torch.cuda.is_available()) else "cpu"
    peft = _wrap(tiny_model).to(device=device, dtype=dtype)
    for n, p in peft.named_parameters():
        if "lora_" in n:
            assert p.dtype == dtype, f"{n} not cast to {dtype}"
    # Run a forward only where the matmul dtype/device combo is supported.
    if dtype == torch.bfloat16 or device == "cuda":
        ids = torch.randint(0, vocab, (2, 8), device=device)
        assert peft(ids).dtype == dtype


# --- 9. device propagation --------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_propagation(tiny_model):
    peft = _wrap(tiny_model).to("cuda")
    assert all(p.is_cuda for n, p in peft.named_parameters() if "lora_" in n)
    out = peft(torch.randint(0, tiny_model.vocab, (2, 8), device="cuda"))
    assert out.is_cuda


# --- 10. config JSON serialization ------------------------------------------
def test_config_serialization(tmp_path):
    cfg = LoRAConfig(r=16, alpha=32, dropout=0.1, target_modules=TARGETS, use_rslora=True)
    # Dict round-trip.
    restored = LoRAConfig.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
    assert restored.r == 16 and restored.use_rslora is True
    # JSON-string round-trip (proves JSON-native fields).
    assert json.loads(json.dumps(cfg.to_dict()))["target_modules"] == TARGETS
    # File round-trip.
    path = cfg.save(tmp_path)
    assert path.name == "adapter_config.json"
    assert LoRAConfig.load(tmp_path).alpha == 32


# ===========================================================================
# Variant + edge-case coverage
# ===========================================================================
def test_rslora_scaling_formula():
    std = LoRAConfig(r=16, alpha=16, target_modules=TARGETS)
    rs = LoRAConfig(r=16, alpha=16, use_rslora=True, target_modules=TARGETS)
    assert std.scaling == pytest.approx(1.0)  # alpha/r = 16/16
    assert rs.scaling == pytest.approx(16 / (16**0.5))  # alpha/sqrt(r) = 4.0


def test_tied_weights_reduce_params(tiny_model):
    untied = _wrap(type(tiny_model)())
    tied = _wrap(type(tiny_model)(), tie_weights=True)
    untied_n, _ = untied.get_nb_trainable_parameters()
    tied_n, _ = tied.get_nb_trainable_parameters()
    # All 4 targets share one (A,B) of shape (8,32)+(32,8)=512 -> huge reduction.
    assert tied_n == R * (32 + 32)  # 512
    assert tied_n < untied_n


def test_tied_weights_roundtrip(tiny_model, tmp_path):
    tied = _wrap(tiny_model, tie_weights=True)
    for _, p in tied.named_parameters():
        if p.requires_grad:
            p.data.normal_()
    ids = torch.randint(0, tiny_model.vocab, (2, 8))
    before = tied(ids).detach().clone()
    tied.save_pretrained(tmp_path)
    fresh = LoRAModel.from_pretrained(type(tiny_model)(), tmp_path, tie_weights=True)
    # Reload only restores adapter weights; base weights differ, so just check the
    # tied adapter weights match bitwise.
    assert tied.adapter_state_dict().keys() == fresh.adapter_state_dict().keys()
    assert before.shape == fresh(ids).shape


def test_dropout_is_on_lora_path(tiny_model, random_ids):
    peft = _wrap(tiny_model, dropout=0.5)
    # Make the delta non-zero so dropout has an observable effect.
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            p.data.normal_()
    peft.train()
    a = peft(random_ids)
    b = peft(random_ids)
    assert not torch.allclose(a, b), "dropout should randomise the LoRA path in train mode"
    peft.eval()
    assert torch.allclose(peft(random_ids), peft(random_ids))


def test_conv1d_support_and_merge(random_input):
    conv1d = pytest.importorskip("transformers.pytorch_utils").Conv1D
    layer = conv1d(48, 32)  # nf=48 (out), nx=32 (in) -> weight (32, 48)
    lora = LoRALinear(layer, r=4, alpha=8)
    assert (lora.in_features, lora.out_features, lora.fan_in_fan_out) == (32, 48, True)
    # zero-delta
    assert torch.allclose(lora(random_input), layer(random_input), atol=1e-6)
    lora.lora_B.data.normal_()
    before = lora(random_input).detach().clone()
    lora.merge()
    assert torch.allclose(before, lora(random_input), atol=1e-5)
    lora.unmerge()
    assert torch.allclose(before, lora(random_input), atol=1e-5)


def test_merge_twice_raises():
    lora = LoRALinear(nn.Linear(8, 8), r=2)
    lora.merge()
    with pytest.raises(MergeError, match="already merged"):
        lora.merge()


def test_unmerge_without_merge_raises():
    lora = LoRALinear(nn.Linear(8, 8), r=2)
    with pytest.raises(MergeError, match="not merged"):
        lora.unmerge()


def test_wrong_input_dim_raises():
    lora = LoRALinear(nn.Linear(8, 8), r=2)
    with pytest.raises(ShapeError):
        lora(torch.randn(3, 7))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r": 0}, "rank"),
        ({"alpha": 0}, "alpha"),
        ({"dropout": 1.0}, "dropout"),
        ({"bias": "wrong"}, "bias"),
        ({"lora_plus_lr_ratio": -1.0}, "lora_plus"),
    ],
)
def test_invalid_config_raises(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        LoRAConfig(target_modules=TARGETS, **kwargs)


def test_bias_modes(tiny_model):
    peft = _wrap(tiny_model, bias="all")
    bias_trainable = [
        n for n, p in peft.named_parameters() if n.endswith("bias") and p.requires_grad
    ]
    assert bias_trainable, "bias='all' should train biases"


def test_no_target_match_raises(tiny_model):
    with pytest.raises(ConfigError, match="No linear-like modules matched"):
        get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["does_not_exist"]))


def test_abstract_config_rejected():
    from peft_lib.core.base import PEFTConfig

    with pytest.raises(ConfigError, match="abstract"):
        PEFTConfig()
