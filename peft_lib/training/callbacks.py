"""Training callbacks: logging, checkpointing, early stopping.

Callbacks observe (and can steer) a run via a small set of hooks invoked by
:class:`~peft_lib.training.trainer.PEFTTrainer`. A callback may request an early
stop by setting ``state.should_stop = True``. State is passed by reference so
callbacks share a single mutable :class:`TrainerState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from peft_lib.core.base import PEFTModel
from peft_lib.core.utils import human_readable

__all__ = [
    "CheckpointSaver",
    "EarlyStopping",
    "TrainableParamLogger",
    "TrainerCallback",
    "TrainerState",
]


@dataclass
class TrainerState:
    """Mutable training state shared with callbacks.

    Attributes:
        global_step: Optimizer steps taken so far.
        epoch: Current (0-based) epoch.
        max_steps: Planned total optimizer steps.
        latest_loss: Most recent training loss value.
        metrics: Latest evaluation metrics (name -> value).
        should_stop: Set ``True`` by a callback to request an early stop.
    """

    global_step: int = 0
    epoch: int = 0
    max_steps: int = 0
    latest_loss: float = float("nan")
    metrics: dict[str, float] = field(default_factory=dict)
    should_stop: bool = False


class TrainerCallback:
    """Base class for callbacks. Override only the hooks you need.

    All hooks are no-ops by default, so subclasses stay minimal.
    """

    def on_train_begin(self, state: TrainerState, model: PEFTModel) -> None:
        """Called once before the first optimizer step."""

    def on_step_end(self, state: TrainerState, model: PEFTModel) -> None:
        """Called after every optimizer step."""

    def on_log(self, state: TrainerState, logs: dict[str, float]) -> None:
        """Called whenever the trainer emits scalar logs."""

    def on_evaluate(self, state: TrainerState, metrics: dict[str, float], model: PEFTModel) -> None:
        """Called after an evaluation pass produces ``metrics``."""

    def on_train_end(self, state: TrainerState, model: PEFTModel) -> None:
        """Called once after the final step."""


class TrainableParamLogger(TrainerCallback):
    """Logs the trainable / total parameter breakdown at the start of training.

    Example:
        >>> import torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> from peft_lib.training import TrainableParamLogger, TrainerState
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> TrainableParamLogger().on_train_begin(TrainerState(), peft)  # doctest: +ELLIPSIS
        [peft_lib] trainable: 64 / ... (...%) ...
    """

    def on_train_begin(self, state: TrainerState, model: PEFTModel) -> None:
        """Print a one-line trainable-parameter summary."""
        trainable, total = model.get_nb_trainable_parameters()
        pct = 100.0 * trainable / total if total else 0.0
        print(
            f"[peft_lib] trainable: {trainable} / {total} "
            f"({pct:.4f}%)  [{human_readable(trainable)} / {human_readable(total)}]"
        )


class CheckpointSaver(TrainerCallback):
    """Saves adapter checkpoints periodically and/or on best metric.

    Args:
        output_dir: Directory under which ``step-<n>`` / ``best`` subdirs are written.
        every_steps: If set, save every ``every_steps`` optimizer steps.
        save_best: If ``True``, save to ``best/`` whenever ``monitor`` improves.
        monitor: Metric name to watch for ``save_best``.
        greater_is_better: Whether a higher ``monitor`` is better.

    Attributes:
        best_value: The best observed ``monitor`` value (``None`` until first eval).

    Example:
        >>> CheckpointSaver("out", every_steps=100).every_steps
        100
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        every_steps: int | None = None,
        save_best: bool = False,
        monitor: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.every_steps = every_steps
        self.save_best = save_best
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.best_value: float | None = None

    def on_step_end(self, state: TrainerState, model: PEFTModel) -> None:
        """Save a periodic checkpoint if ``every_steps`` is configured."""
        if self.every_steps and state.global_step > 0 and state.global_step % self.every_steps == 0:
            model.save_pretrained(self.output_dir / f"step-{state.global_step}")

    def on_evaluate(self, state: TrainerState, metrics: dict[str, float], model: PEFTModel) -> None:
        """Save to ``best/`` when the monitored metric improves."""
        if not self.save_best or self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        improved = self.best_value is None or (
            value > self.best_value if self.greater_is_better else value < self.best_value
        )
        if improved:
            self.best_value = value
            model.save_pretrained(self.output_dir / "best")


class EarlyStopping(TrainerCallback):
    """Requests an early stop when a monitored metric stops improving.

    Args:
        monitor: Metric name to watch.
        patience: Number of evaluations without improvement to tolerate.
        min_delta: Minimum change counted as an improvement.
        greater_is_better: Whether a higher metric is better.

    Attributes:
        best_value: Best observed value so far.
        num_bad_evals: Consecutive evaluations without improvement.

    Example:
        >>> es = EarlyStopping(monitor="eval_loss", patience=2)
        >>> es.patience
        2
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        *,
        patience: int = 3,
        min_delta: float = 0.0,
        greater_is_better: bool = False,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self.best_value: float | None = None
        self.num_bad_evals = 0

    def on_evaluate(self, state: TrainerState, metrics: dict[str, float], model: PEFTModel) -> None:
        """Update the patience counter and flip ``should_stop`` when exhausted."""
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if self.best_value is None:
            self.best_value = value
            return
        delta = value - self.best_value
        improved = (delta > self.min_delta) if self.greater_is_better else (delta < -self.min_delta)
        if improved:
            self.best_value = value
            self.num_bad_evals = 0
        else:
            self.num_bad_evals += 1
            if self.num_bad_evals >= self.patience:
                state.should_stop = True
