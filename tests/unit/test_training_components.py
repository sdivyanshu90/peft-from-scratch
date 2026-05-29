"""Unit tests for schedulers, callbacks, and trainer helper branches."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from peft_lib import LoRAConfig, get_peft_model
from peft_lib.core.exceptions import ConfigError
from peft_lib.training import (
    CheckpointSaver,
    EarlyStopping,
    PEFTTrainer,
    TrainerConfig,
    TrainerState,
    build_scheduler,
    causal_lm_loss,
    default_compute_loss,
    get_linear_schedule_with_warmup,
    get_warmup_cosine_schedule,
)


def _opt(lr: float = 1.0) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=lr)


# --- schedulers -------------------------------------------------------------
def test_linear_schedule_warmup_and_decay():
    opt = _opt()
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=6)
    lrs = []
    for _ in range(6):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.0)  # step 0
    assert lrs[2] == pytest.approx(1.0)  # peak after warmup
    assert lrs[5] < lrs[2]  # decaying


def test_cosine_schedule_floor_and_shape():
    opt = _opt()
    sched = get_warmup_cosine_schedule(opt, 0, 4, min_lr_ratio=0.1)
    lrs = []
    for _ in range(4):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1.0)
    assert all(v >= 0.1 - 1e-9 for v in lrs)  # never below the floor
    assert lrs[-1] < lrs[0]


def test_cosine_invalid_min_lr_ratio():
    with pytest.raises(ConfigError):
        get_warmup_cosine_schedule(_opt(), 0, 4, min_lr_ratio=1.5)


@pytest.mark.parametrize("name", ["cosine", "linear", "constant"])
def test_build_scheduler_known(name):
    assert build_scheduler(name, _opt(), 1, 4) is not None


def test_build_scheduler_none_and_unknown():
    assert build_scheduler("none", _opt(), 0, 4) is None
    with pytest.raises(ConfigError):
        build_scheduler("bogus", _opt(), 0, 4)  # type: ignore[arg-type]


def test_constant_scheduler_warms_up_then_holds():
    opt = _opt()
    sched = build_scheduler("constant", opt, 2, 6)
    assert sched is not None
    vals = []
    for _ in range(5):
        vals.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert vals[0] == pytest.approx(0.0)
    assert vals[2] == pytest.approx(1.0) and vals[4] == pytest.approx(1.0)


# --- callbacks --------------------------------------------------------------
def _model() -> nn.Module:
    return get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))


def test_checkpoint_saver_saves_best_on_improvement(tmp_path):
    saver = CheckpointSaver(tmp_path, save_best=True, monitor="eval_loss")
    model = _model()
    state = TrainerState()
    saver.on_evaluate(state, {"eval_loss": 1.0}, model)
    assert (tmp_path / "best" / "adapter_config.json").exists()
    assert saver.best_value == 1.0
    # Worse metric -> no overwrite (best_value unchanged).
    saver.on_evaluate(state, {"eval_loss": 2.0}, model)
    assert saver.best_value == 1.0


def test_checkpoint_saver_ignores_missing_metric(tmp_path):
    saver = CheckpointSaver(tmp_path, save_best=True, monitor="eval_loss")
    saver.on_evaluate(TrainerState(), {"other": 1.0}, _model())
    assert saver.best_value is None


def test_early_stopping_patience_and_improvement():
    es = EarlyStopping(monitor="eval_loss", patience=2)
    state = TrainerState()
    es.on_evaluate(state, {"eval_loss": 1.0}, _model())  # first -> baseline
    es.on_evaluate(state, {"eval_loss": 0.5}, _model())  # improves
    assert es.num_bad_evals == 0 and not state.should_stop
    es.on_evaluate(state, {"eval_loss": 0.6}, _model())  # worse (1)
    es.on_evaluate(state, {"eval_loss": 0.6}, _model())  # worse (2) -> stop
    assert state.should_stop


def test_early_stopping_greater_is_better():
    es = EarlyStopping(monitor="acc", patience=1, greater_is_better=True)
    state = TrainerState()
    es.on_evaluate(state, {"acc": 0.8}, _model())
    es.on_evaluate(state, {"acc": 0.9}, _model())  # improves
    assert not state.should_stop
    es.on_evaluate(state, {"acc": 0.85}, _model())  # worse -> stop
    assert state.should_stop


def test_early_stopping_ignores_missing_metric():
    es = EarlyStopping(monitor="eval_loss")
    state = TrainerState()
    es.on_evaluate(state, {"nope": 1.0}, _model())
    assert not state.should_stop and es.best_value is None


# --- trainer helpers --------------------------------------------------------
def test_default_compute_loss_requires_labels():
    with pytest.raises(ConfigError, match="requires a 'labels'"):
        default_compute_loss(nn.Linear(4, 4), {"x": torch.randn(2, 4)})


def test_causal_lm_loss_shifts():
    torch.manual_seed(0)

    class LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(4, 6)

        def forward(self, input_ids):
            return self.head(torch.randn(input_ids.shape[0], input_ids.shape[1], 4))

    batch = {"input_ids": torch.randint(0, 6, (2, 5)), "labels": torch.randint(0, 6, (2, 5))}
    loss = causal_lm_loss(LM(), batch)
    assert loss.ndim == 0 and math.isfinite(float(loss))


def test_trainer_requires_max_steps_for_lenless_loader():
    peft = _model()

    def gen():
        yield {"input_ids": torch.randn(2, 8), "labels": torch.randint(0, 8, (2,))}

    with pytest.raises(ConfigError, match="no length"):
        PEFTTrainer(peft, TrainerConfig(num_epochs=1, device="cpu"), gen())


def test_trainer_evaluate_returns_metrics():
    torch.manual_seed(0)
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    data = [{"x": torch.randn(4, 8), "labels": torch.randint(0, 8, (4,))}]

    def loss_fn(m, b):
        return nn.functional.cross_entropy(m(b["x"]), b["labels"])

    cfg = TrainerConfig(max_steps=1, device="cpu", scheduler="none")
    trainer = PEFTTrainer(peft, cfg, data, eval_dataloader=data, compute_loss=loss_fn)
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics and math.isfinite(metrics["eval_loss"])


def test_trainer_epoch_based_runs_full_pass():
    torch.manual_seed(0)
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    data = [{"x": torch.randn(4, 8), "labels": torch.randint(0, 8, (4,))} for _ in range(3)]

    def loss_fn(m, b):
        return nn.functional.cross_entropy(m(b["x"]), b["labels"])

    cfg = TrainerConfig(num_epochs=2, device="cpu", scheduler="linear", logging_steps=1)
    trainer = PEFTTrainer(peft, cfg, data, compute_loss=loss_fn)
    trainer.train()
    assert trainer.state.global_step == 6  # 3 batches * 2 epochs
