"""Integration tests for PEFTTrainer, schedulers, and callbacks."""

from __future__ import annotations

import pytest
import torch

from peft_lib import LoRAConfig, get_peft_model
from peft_lib.training import (
    CheckpointSaver,
    EarlyStopping,
    PEFTTrainer,
    TrainableParamLogger,
    TrainerConfig,
    build_lora_plus_param_groups,
)

ALL_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def _dataset(vocab: int, n: int = 4):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (n, 8))
    return [{"input_ids": ids, "labels": ids}]


def test_training_reduces_loss(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=8, alpha=16, target_modules=ALL_PROJ))
    data = _dataset(tiny_model.vocab)
    cfg = TrainerConfig(
        learning_rate=1e-2, max_steps=40, scheduler="cosine", device="cpu", logging_steps=1
    )
    trainer = PEFTTrainer(peft, cfg, data * 40)
    history = trainer.train()
    losses = [h["loss"] for h in history if "loss" in h]
    assert losses[-1] < losses[0], f"loss did not improve: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert trainer.state.global_step == 40


def test_only_adapter_weights_change(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=8, target_modules=["q_proj"]))
    # Snapshot a frozen base weight and the embedding.
    base_w = peft.adapter_layers["layers.0.q_proj"].base_layer.weight.detach().clone()
    embed_w = tiny_model.embed.weight.detach().clone()
    data = _dataset(tiny_model.vocab)
    trainer = PEFTTrainer(
        peft, TrainerConfig(learning_rate=1e-2, max_steps=10, device="cpu"), data * 10
    )
    trainer.train()
    assert torch.equal(base_w, peft.adapter_layers["layers.0.q_proj"].base_layer.weight)
    assert torch.equal(embed_w, tiny_model.embed.weight)
    # And the adapter actually moved.
    assert peft.adapter_layers["layers.0.q_proj"].lora_B.abs().sum().item() > 0


def test_gradient_accumulation_step_count(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["q_proj"]))
    data = _dataset(tiny_model.vocab) * 8
    cfg = TrainerConfig(max_steps=2, gradient_accumulation_steps=4, device="cpu", scheduler="none")
    trainer = PEFTTrainer(peft, cfg, data)
    trainer.train()
    # 8 micro-batches / 4 accumulation = 2 optimizer steps.
    assert trainer.state.global_step == 2


def test_scheduler_warmup_then_decay(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["q_proj"]))
    data = _dataset(tiny_model.vocab) * 20
    cfg = TrainerConfig(
        learning_rate=1.0,
        max_steps=20,
        warmup_steps=5,
        scheduler="cosine",
        device="cpu",
        logging_steps=1,
    )
    trainer = PEFTTrainer(peft, cfg, data)
    trainer.train()
    lrs = [h["lr"] for h in trainer.log_history if "lr" in h]
    assert lrs[0] < lrs[4]  # warming up
    assert lrs[-1] < max(lrs)  # then decaying


def test_lora_plus_param_groups_have_scaled_lr(tiny_model):
    peft = get_peft_model(
        tiny_model, LoRAConfig(r=8, target_modules=["q_proj"], lora_plus_lr_ratio=16.0)
    )
    trainer = PEFTTrainer(
        peft,
        TrainerConfig(learning_rate=1e-3, max_steps=1, device="cpu"),
        _dataset(tiny_model.vocab),
    )
    lrs = sorted(g["lr"] for g in trainer.optimizer.param_groups)
    assert lrs == pytest.approx([1e-3, 1.6e-2])


def test_early_stopping_triggers(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["q_proj"]))
    data = _dataset(tiny_model.vocab)
    # Eval loss won't improve on a constant eval set with a tiny lr -> patience hit.
    es = EarlyStopping(monitor="eval_loss", patience=1)
    cfg = TrainerConfig(
        learning_rate=0.0, max_steps=50, eval_steps=1, device="cpu", scheduler="none"
    )
    trainer = PEFTTrainer(peft, cfg, data * 50, eval_dataloader=data, callbacks=[es])
    trainer.train()
    assert trainer.state.should_stop
    assert trainer.state.global_step < 50  # stopped early


def test_checkpoint_saver_writes(tiny_model, tmp_path):
    peft = get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["q_proj"]))
    saver = CheckpointSaver(tmp_path, every_steps=2)
    cfg = TrainerConfig(max_steps=4, device="cpu", scheduler="none")
    PEFTTrainer(peft, cfg, _dataset(tiny_model.vocab) * 4, callbacks=[saver]).train()
    assert (tmp_path / "step-2" / "adapter_config.json").exists()
    assert (tmp_path / "step-4" / "adapter_model.safetensors").exists()


def test_trainable_param_logger_runs(tiny_model, capsys):
    peft = get_peft_model(tiny_model, LoRAConfig(r=4, target_modules=["q_proj"]))
    cfg = TrainerConfig(max_steps=1, device="cpu", scheduler="none")
    PEFTTrainer(peft, cfg, _dataset(tiny_model.vocab), callbacks=[TrainableParamLogger()]).train()
    out = capsys.readouterr().out
    assert "trainable:" in out


def test_invalid_trainer_config():
    from peft_lib.core.exceptions import ConfigError

    with pytest.raises(ConfigError):
        TrainerConfig(gradient_accumulation_steps=0)


def test_build_lora_plus_groups_partition(tiny_model):
    peft = get_peft_model(tiny_model, LoRAConfig(r=8, target_modules=["q_proj", "v_proj"]))
    groups = build_lora_plus_param_groups(peft, 1e-3, 16.0)
    n_b = sum(p.numel() for p in groups[1]["params"])
    n_other = sum(p.numel() for p in groups[0]["params"])
    # 4 layers * (B: out*r=32*8) ; A: (r*in=8*32) -> equal here, both 1024 per layer.
    assert n_b == 4 * 32 * 8
    assert n_other == 4 * 8 * 32
