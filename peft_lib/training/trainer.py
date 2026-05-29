"""PEFTTrainer: a compact, dependency-light training loop for PEFT models.

It deliberately stays small and readable rather than reimplementing the full
HuggingFace ``Trainer``: gradient accumulation, gradient clipping, warmup/decay
scheduling, periodic evaluation, callbacks, and LoRA+ parameter grouping — the
pieces that matter for parameter-efficient fine-tuning — and nothing else.

The loss is computed by a pluggable ``compute_loss(model, batch)``; the default
pops ``labels`` from the batch, runs the model, and applies position-wise cross
entropy to the logits. For causal-LM next-token shifting, pass a custom
``compute_loss`` (see :func:`causal_lm_loss`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from peft_lib.core.base import PEFTModel
from peft_lib.core.exceptions import ConfigError
from peft_lib.methods.lora import LoRAConfig
from peft_lib.training.callbacks import TrainerCallback, TrainerState
from peft_lib.training.schedulers import SchedulerName, build_scheduler

__all__ = [
    "LossFn",
    "PEFTTrainer",
    "TrainerConfig",
    "build_lora_plus_param_groups",
    "causal_lm_loss",
    "default_compute_loss",
]

Batch = Mapping[str, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


def default_compute_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Position-wise cross-entropy on the model's logits.

    Pops ``labels`` from ``batch``, runs ``model(**rest)``, and computes
    ``cross_entropy(logits, labels, ignore_index=-100)``. Works for both plain
    ``nn.Module`` backbones (logits tensor) and HuggingFace outputs (``.logits``).

    Args:
        model: The model (or PEFT wrapper).
        batch: A mapping that must contain ``"labels"`` plus the model's inputs.

    Returns:
        A scalar loss tensor.

    Raises:
        ConfigError: If ``batch`` has no ``"labels"`` key.
    """
    if "labels" not in batch:
        raise ConfigError("default_compute_loss requires a 'labels' entry in the batch.")
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    out = model(**inputs)
    logits = out.logits if hasattr(out, "logits") else out
    labels = batch["labels"]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


def causal_lm_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Next-token cross-entropy with the standard causal shift.

    Args:
        model: A backbone producing ``(B, S, V)`` logits.
        batch: Must contain ``"labels"`` and the model inputs.

    Returns:
        A scalar next-token loss tensor.

    Raises:
        ConfigError: If ``batch`` has no ``"labels"`` key.
    """
    if "labels" not in batch:
        raise ConfigError("causal_lm_loss requires a 'labels' entry in the batch.")
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    out = model(**inputs)
    logits = out.logits if hasattr(out, "logits") else out
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = batch["labels"][..., 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def build_lora_plus_param_groups(
    model: nn.Module,
    learning_rate: float,
    lr_ratio: float,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Split trainable params into LoRA+ groups: ``B`` gets ``lr_ratio x`` the lr.

    LoRA+ (Hayou et al., 2024) shows that giving the ``B`` matrix a higher learning
    rate than ``A`` (ratio ~ 16) speeds convergence at no extra cost.

    Args:
        model: A LoRA-wrapped model.
        learning_rate: Base learning rate (applied to ``A`` and everything else).
        lr_ratio: Multiplier for ``B``-matrix parameters.
        weight_decay: Weight decay applied to both groups.

    Returns:
        Two optimizer param groups: ``{A & others @ lr}`` and ``{B @ lr*ratio}``.

    Example:
        >>> import torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> groups = build_lora_plus_param_groups(peft, 1e-3, 16.0)
        >>> [round(g["lr"], 4) for g in groups]
        [0.001, 0.016]
    """
    b_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (b_params if "lora_B" in name else other_params).append(param)
    return [
        {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": b_params, "lr": learning_rate * lr_ratio, "weight_decay": weight_decay},
    ]


@dataclass
class TrainerConfig:
    """Hyperparameters for :class:`PEFTTrainer`.

    Attributes:
        learning_rate: Base AdamW learning rate.
        weight_decay: AdamW weight decay (adapters only).
        num_epochs: Passes over the training dataloader (ignored if ``max_steps``).
        max_steps: Hard cap on optimizer steps; overrides ``num_epochs`` if set.
        gradient_accumulation_steps: Micro-batches per optimizer step.
        max_grad_norm: Gradient-norm clip threshold (``None`` to disable).
        warmup_ratio: Fraction of total steps used for LR warmup.
        warmup_steps: Absolute warmup steps; overrides ``warmup_ratio`` if set.
        scheduler: ``"cosine"`` | ``"linear"`` | ``"constant"`` | ``"none"``.
        logging_steps: Emit logs every N optimizer steps.
        eval_steps: Evaluate every N steps (``None`` to evaluate only per-epoch).
        seed: Seed set at the start of training.
        device: Target device (``None`` -> CUDA if available else CPU).
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    warmup_ratio: float = 0.0
    warmup_steps: int | None = None
    scheduler: SchedulerName = "cosine"
    logging_steps: int = 10
    eval_steps: int | None = None
    seed: int = 42
    device: str | None = None

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("gradient_accumulation_steps must be >= 1.")
        if self.num_epochs < 1 and self.max_steps is None:
            raise ConfigError("Set num_epochs >= 1 or an explicit max_steps.")

    def resolved_device(self) -> str:
        """Return the configured device, defaulting to CUDA when available."""
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class PEFTTrainer:
    """A minimal trainer specialised for PEFT models.

    Args:
        model: The :class:`PEFTModel` to train (only its adapters update).
        config: A :class:`TrainerConfig`.
        train_dataloader: Iterable of ``dict[str, Tensor]`` batches.
        eval_dataloader: Optional evaluation dataloader.
        optimizer: Optional optimizer; built (AdamW, LoRA+-aware) if ``None``.
        compute_loss: ``(model, batch) -> scalar``; defaults to
            :func:`default_compute_loss`.
        callbacks: Optional list of :class:`TrainerCallback`.

    Attributes:
        state: The live :class:`TrainerState`.
        log_history: List of scalar log dicts emitted during training.

    Example:
        >>> import torch, torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> from peft_lib.training import PEFTTrainer, TrainerConfig
        >>> _ = torch.manual_seed(0)
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> batch = {"x": torch.randn(4, 8), "labels": torch.randint(0, 8, (4,))}
        >>> # custom loss for a bare Linear classifier
        >>> def loss_fn(m, b):
        ...     return torch.nn.functional.cross_entropy(m(b["x"]), b["labels"])
        >>> cfg = TrainerConfig(max_steps=3, scheduler="none", device="cpu", logging_steps=100)
        >>> tr = PEFTTrainer(peft, cfg, [batch] * 3, compute_loss=loss_fn)
        >>> _ = tr.train()
        >>> tr.state.global_step
        3
    """

    def __init__(
        self,
        model: PEFTModel,
        config: TrainerConfig,
        train_dataloader: Iterable[Batch],
        *,
        eval_dataloader: Iterable[Batch] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        compute_loss: LossFn | None = None,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_loss: LossFn = compute_loss or default_compute_loss
        self.callbacks: list[TrainerCallback] = list(callbacks or [])
        self.device = config.resolved_device()
        self.state = TrainerState()
        self.log_history: list[dict[str, float]] = []

        self.optimizer = optimizer or self._build_optimizer()
        self._max_steps = self._compute_max_steps()
        self.state.max_steps = self._max_steps
        warmup = self._compute_warmup_steps(self._max_steps)
        self.scheduler = build_scheduler(config.scheduler, self.optimizer, warmup, self._max_steps)

    # -- setup helpers -------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        model_cfg = self.model.config
        if isinstance(model_cfg, LoRAConfig) and model_cfg.lora_plus_lr_ratio:
            groups = build_lora_plus_param_groups(
                self.model, cfg.learning_rate, model_cfg.lora_plus_lr_ratio, cfg.weight_decay
            )
        else:
            groups = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "weight_decay": cfg.weight_decay,
                }
            ]
        return torch.optim.AdamW(groups, lr=cfg.learning_rate)

    def _compute_max_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        try:
            num_batches = len(self.train_dataloader)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigError(
                "train_dataloader has no length; set TrainerConfig.max_steps explicitly."
            ) from exc
        steps_per_epoch = math.ceil(num_batches / self.config.gradient_accumulation_steps)
        return max(1, steps_per_epoch * self.config.num_epochs)

    def _compute_warmup_steps(self, max_steps: int) -> int:
        if self.config.warmup_steps is not None:
            return self.config.warmup_steps
        return int(max_steps * self.config.warmup_ratio)

    # -- callback dispatch ---------------------------------------------------
    def _emit(self, hook: str, *args: Any) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self.state, *args)

    def _log(self, logs: dict[str, float]) -> None:
        self.log_history.append({"step": float(self.state.global_step), **logs})
        self._emit("on_log", logs)

    def _move(self, batch: Batch) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    # -- training loop -------------------------------------------------------
    def train(self) -> list[dict[str, float]]:
        """Run the training loop and return the scalar log history.

        Returns:
            A list of log dicts, each containing at least ``"step"`` and ``"loss"``.
        """
        torch.manual_seed(self.config.seed)
        self.model.to(self.device)
        self.model.train()
        self._emit("on_train_begin", self.model)

        accum = self.config.gradient_accumulation_steps
        micro = 0
        self.optimizer.zero_grad()
        done = False

        epoch = 0
        while not done:
            for batch in self.train_dataloader:
                loss = self.compute_loss(self.model, self._move(batch)) / accum
                loss.backward()  # type: ignore[no-untyped-call]
                micro += 1
                self.state.latest_loss = float(loss.item() * accum)
                if micro % accum != 0:
                    continue

                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad),
                        self.config.max_grad_norm,
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
                self._emit("on_step_end", self.model)

                if self.state.global_step % self.config.logging_steps == 0:
                    self._log({"loss": self.state.latest_loss, "lr": self._current_lr()})
                if self.config.eval_steps and self.state.global_step % self.config.eval_steps == 0:
                    self.evaluate()

                if self.state.global_step >= self._max_steps or self.state.should_stop:
                    done = True
                    break
            epoch += 1
            self.state.epoch = epoch
            if self.eval_dataloader is not None and not done:
                self.evaluate()
            if self.config.max_steps is None and epoch >= self.config.num_epochs:
                done = True

        self._log({"loss": self.state.latest_loss, "lr": self._current_lr()})
        self._emit("on_train_end", self.model)
        return self.log_history

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run one pass over ``eval_dataloader`` and return ``{"eval_loss": ...}``.

        Returns:
            A metrics dict (empty if no eval dataloader is configured).
        """
        if self.eval_dataloader is None:
            return {}
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.eval_dataloader:
            loss = self.compute_loss(self.model, self._move(batch))
            total += float(loss.item())
            count += 1
        self.model.train()
        metrics = {"eval_loss": total / max(1, count)}
        self.state.metrics = metrics
        self._emit("on_evaluate", metrics, self.model)
        self._log(metrics)
        return metrics

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
