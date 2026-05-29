"""Integration: every injection method trains end-to-end through PEFTTrainer."""

from __future__ import annotations

import math

import pytest
import torch

from peft_lib import (
    AdapterConfig,
    DoRAConfig,
    IA3Config,
    LoRAConfig,
    VeRAConfig,
    get_peft_model,
)
from peft_lib.training import PEFTTrainer, TrainerConfig

CONFIGS = {
    "lora": lambda: LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
    "dora": lambda: DoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
    "vera": lambda: VeRAConfig(r=16, target_modules=["q_proj", "v_proj"]),
    "ia3": lambda: IA3Config(target_modules=["q_proj", "v_proj"]),
    "adapter": lambda: AdapterConfig(target_modules=["down_proj"], bottleneck_dim=8),
}


def _dataset(vocab: int):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (4, 8))
    return [{"input_ids": ids, "labels": ids}]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_method_trains_through_trainer(tiny_model, name):
    peft = get_peft_model(tiny_model, CONFIGS[name]())
    before = {n: p.detach().clone() for n, p in peft.named_parameters() if p.requires_grad}
    cfg = TrainerConfig(learning_rate=5e-3, max_steps=15, device="cpu", scheduler="none")
    trainer = PEFTTrainer(peft, cfg, _dataset(tiny_model.vocab) * 15)
    history = trainer.train()

    assert trainer.state.global_step == 15
    assert all(math.isfinite(h["loss"]) for h in history if "loss" in h)
    # At least one trainable parameter moved during training.
    moved = any(
        not torch.equal(before[n], p)
        for n, p in peft.named_parameters()
        if p.requires_grad and n in before
    )
    assert moved, f"{name}: no trainable parameter changed"


@pytest.mark.parametrize("name", ["lora", "dora", "vera", "ia3"])
def test_train_then_merge_is_equivalent(tiny_model, name):
    peft = get_peft_model(tiny_model, CONFIGS[name]())
    data = _dataset(tiny_model.vocab)
    PEFTTrainer(
        peft,
        TrainerConfig(learning_rate=5e-3, max_steps=10, device="cpu", scheduler="none"),
        data * 10,
    ).train()
    eval_ids = torch.randint(0, tiny_model.vocab, (2, 8))
    peft.eval()
    expected = peft(eval_ids).detach().clone()
    merged = peft.merge_and_unload()
    assert torch.allclose(expected, merged(eval_ids), atol=1e-4)
