"""The canonical 10-test contract, parametrised across all injection methods.

Each test runs for LoRA, DoRA, VeRA, and IA³ (ids show as e.g.
``test_merge_and_unload[dora]``), so every method gets the full suite while the
assertions stay in one place. Method-specific behaviour lives in
``test_dora.py`` / ``test_vera.py`` / ``test_ia3.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from peft_lib import (
    DoRAConfig,
    IA3Config,
    LoRAConfig,
    PEFTConfig,
    VeRAConfig,
    get_peft_model,
)

# TinyModel q_proj/v_proj are 32x32; 2 layers * 2 projs = 4 adapted layers.
TARGETS = ["q_proj", "v_proj"]
N_LAYERS = 4
DIM = 32


@dataclass
class Case:
    """One method's parametrisation of the shared contract."""

    name: str
    make_config: Callable[[], PEFTConfig]
    markers: tuple[str, ...]
    expected_trainable: int


CASES = [
    Case(
        "lora",
        lambda: LoRAConfig(r=8, alpha=16, target_modules=TARGETS),
        ("lora_",),
        N_LAYERS * 8 * (DIM + DIM),
    ),
    Case(
        "dora",
        lambda: DoRAConfig(r=8, alpha=16, target_modules=TARGETS),
        ("lora_", "magnitude"),
        N_LAYERS * (8 * (DIM + DIM) + DIM),
    ),
    Case(
        "vera",
        lambda: VeRAConfig(r=16, target_modules=TARGETS),
        ("vera_lambda",),
        N_LAYERS * (16 + DIM),
    ),
    Case("ia3", lambda: IA3Config(target_modules=TARGETS), ("ia3_l",), N_LAYERS * DIM),
]
IDS = [c.name for c in CASES]


@pytest.fixture(params=CASES, ids=IDS)
def case(request) -> Case:
    return request.param


def _wrap(model: nn.Module, case: Case):
    return get_peft_model(model, case.make_config())


# --- 1 --------------------------------------------------------------------
def test_forward_shape(tiny_model, random_ids, case):
    out = _wrap(tiny_model, case)(random_ids)
    assert out.shape == (2, 8, tiny_model.vocab)


# --- 2 --------------------------------------------------------------------
def test_only_adapter_params_require_grad(tiny_model, case):
    peft = _wrap(tiny_model, case)
    trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert trainable
    assert all(any(m in n for m in case.markers) for n in trainable)
    # The backbone must still carry frozen weights.
    assert any(not p.requires_grad for p in peft.parameters())


# --- 3 --------------------------------------------------------------------
def test_trainable_param_count(tiny_model, case):
    trn, _ = _wrap(tiny_model, case).get_nb_trainable_parameters()
    assert trn == case.expected_trainable


# --- 4 --------------------------------------------------------------------
def test_gradient_flows(tiny_model, random_ids, case):
    peft = _wrap(tiny_model, case)
    peft(random_ids).pow(2).mean().backward()
    grads = [p.grad for n, p in peft.named_parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    # At least one adapter parameter receives a non-zero gradient on step 0.
    assert any(g.abs().sum().item() > 0 for g in grads)  # type: ignore[union-attr]


# --- 5 --------------------------------------------------------------------
def test_merge_and_unload(tiny_model, random_ids, case):
    peft = _wrap(tiny_model, case)
    opt = torch.optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.3)
    for _ in range(5):
        opt.zero_grad()
        peft(random_ids).pow(2).mean().backward()
        opt.step()
    expected = peft(random_ids).detach().clone()
    merged = peft.merge_and_unload()
    assert torch.allclose(expected, merged(random_ids), atol=1e-4)


# --- 6 --------------------------------------------------------------------
def test_save_load_roundtrip(tiny_model, case, tmp_path):
    vocab = tiny_model.vocab
    peft = _wrap(tiny_model, case)
    for _, p in peft.named_parameters():
        if p.requires_grad:
            p.data.normal_(std=0.1)
    ids = torch.randint(0, vocab, (2, 8))
    before = peft(ids).detach().clone()
    peft.save_pretrained(tmp_path)

    torch.manual_seed(42)  # reproduce the fixture's deterministic base weights
    fresh = type(tiny_model)()
    reloaded = type(peft).from_pretrained(fresh, tmp_path)

    saved, loaded = peft.adapter_state_dict(), reloaded.adapter_state_dict()
    assert saved.keys() == loaded.keys()
    for k in saved:
        assert torch.equal(saved[k].cpu(), loaded[k].cpu())
    assert torch.allclose(before, reloaded(ids), atol=1e-5)


# --- 7 --------------------------------------------------------------------
def test_init_is_zero_delta(tiny_model, random_ids, case):
    base_out = tiny_model(random_ids).detach().clone()
    peft = _wrap(tiny_model, case)
    assert torch.allclose(peft(random_ids), base_out, atol=1e-5)


# --- 8 --------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_propagation(tiny_model, case, dtype):
    vocab = tiny_model.vocab
    device = "cuda" if (dtype == torch.float16 and torch.cuda.is_available()) else "cpu"
    peft = _wrap(tiny_model, case).to(device=device, dtype=dtype)
    for n, p in peft.named_parameters():
        if p.requires_grad:
            assert p.dtype == dtype, f"{n} not cast to {dtype}"
    if dtype == torch.bfloat16 or device == "cuda":
        ids = torch.randint(0, vocab, (2, 8), device=device)
        assert peft(ids).dtype == dtype


# --- 9 --------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_propagation(tiny_model, case):
    peft = _wrap(tiny_model, case).to("cuda")
    assert all(p.is_cuda for p in peft.parameters() if p.requires_grad)
    assert peft(torch.randint(0, tiny_model.vocab, (2, 8), device="cuda")).is_cuda


# --- 10 -------------------------------------------------------------------
def test_config_serialization(case, tmp_path):
    cfg = case.make_config()
    assert PEFTConfig.from_dict.__func__  # sanity: classmethod exists
    restored = type(cfg).from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
    assert restored.peft_type == cfg.peft_type
    path = cfg.save(tmp_path)
    assert type(cfg).load(path).to_dict() == cfg.to_dict()
