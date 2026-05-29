"""A small GLUE evaluation harness for PEFT-adapted classifiers.

GLUE tasks use different metrics (accuracy, Matthews correlation for CoLA, F1 +
accuracy for MRPC/QQP, Pearson/Spearman for STS-B). The metric functions here are
implemented with NumPy (no scikit-learn dependency) and are unit-tested. Dataset
loading/tokenisation lazily import ``datasets`` / ``transformers`` so the rest of
the library has no hard dependency on them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from peft_lib.core.exceptions import ConfigError

FloatArray = npt.NDArray[Any]

__all__ = [
    "GLUE_TASKS",
    "GlueTask",
    "evaluate_glue",
    "glue_metrics",
]


@dataclass(frozen=True)
class GlueTask:
    """Static metadata for a GLUE task.

    Attributes:
        name: Task key (e.g. ``"sst2"``).
        num_labels: Number of classes (1 for the STS-B regression task).
        text_fields: The dataset column(s) holding the input text.
        is_regression: Whether the task is regression (STS-B).
    """

    name: str
    num_labels: int
    text_fields: tuple[str, ...]
    is_regression: bool = False


GLUE_TASKS: dict[str, GlueTask] = {
    "cola": GlueTask("cola", 2, ("sentence",)),
    "sst2": GlueTask("sst2", 2, ("sentence",)),
    "mrpc": GlueTask("mrpc", 2, ("sentence1", "sentence2")),
    "qqp": GlueTask("qqp", 2, ("question1", "question2")),
    "stsb": GlueTask("stsb", 1, ("sentence1", "sentence2"), is_regression=True),
    "mnli": GlueTask("mnli", 3, ("premise", "hypothesis")),
    "qnli": GlueTask("qnli", 2, ("question", "sentence")),
    "rte": GlueTask("rte", 2, ("sentence1", "sentence2")),
    "wnli": GlueTask("wnli", 2, ("sentence1", "sentence2")),
}


def _accuracy(preds: FloatArray, labels: FloatArray) -> float:
    return float((preds == labels).mean())


def _f1_binary(preds: FloatArray, labels: FloatArray) -> float:
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _matthews_corrcoef(preds: FloatArray, labels: FloatArray) -> float:
    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom else 0.0


def _pearson(preds: FloatArray, labels: FloatArray) -> float:
    if preds.std() == 0 or labels.std() == 0:
        return 0.0
    return float(np.corrcoef(preds, labels)[0, 1])


def _spearman(preds: FloatArray, labels: FloatArray) -> float:
    pr = preds.argsort().argsort().astype(np.float64)
    lr = labels.argsort().argsort().astype(np.float64)
    return _pearson(pr, lr)


def glue_metrics(
    task: str,
    predictions: FloatArray | list[float],
    labels: FloatArray | list[float],
) -> dict[str, float]:
    """Compute the canonical GLUE metric(s) for a task.

    Args:
        task: A GLUE task key (see :data:`GLUE_TASKS`).
        predictions: Predicted class indices (or scores for STS-B).
        labels: Ground-truth labels.

    Returns:
        A metric-name -> value dict (e.g. ``{"accuracy": ..., "f1": ...}``).

    Raises:
        ConfigError: If ``task`` is unknown.

    Example:
        >>> glue_metrics("sst2", [1, 0, 1, 1], [1, 0, 0, 1])["accuracy"]
        0.75
        >>> round(glue_metrics("stsb", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["pearson"], 3)
        1.0
    """
    if task not in GLUE_TASKS:
        raise ConfigError(f"Unknown GLUE task {task!r}; choose from {sorted(GLUE_TASKS)}.")
    preds = np.asarray(predictions)
    targets = np.asarray(labels)
    if task == "cola":
        return {"matthews_correlation": _matthews_corrcoef(preds, targets)}
    if task == "stsb":
        return {"pearson": _pearson(preds, targets), "spearmanr": _spearman(preds, targets)}
    if task in ("mrpc", "qqp"):
        return {"accuracy": _accuracy(preds, targets), "f1": _f1_binary(preds, targets)}
    return {"accuracy": _accuracy(preds, targets)}


@torch.no_grad()
def evaluate_glue(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    task: str,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Run a classifier over ``dataloader`` and compute GLUE metrics.

    Each batch must contain ``"labels"`` and the model's inputs. The model is
    expected to return logits (``out.logits`` or a tensor). For regression
    (STS-B) the single output is used directly; otherwise ``argmax`` is taken.

    Args:
        model: A classification/regression model (optionally PEFT-wrapped).
        dataloader: Iterable of input batches.
        task: GLUE task key.
        device: Device to run on.

    Returns:
        The task's metric dict (see :func:`glue_metrics`).

    Raises:
        ConfigError: If ``task`` is unknown.
    """
    if task not in GLUE_TASKS:
        raise ConfigError(f"Unknown GLUE task {task!r}.")
    spec = GLUE_TASKS[task]
    model.to(device).eval()
    all_preds: list[float] = []
    all_labels: list[float] = []
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out
        preds = logits.squeeze(-1) if spec.is_regression else logits.argmax(dim=-1)
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(batch["labels"].flatten().tolist())
    return glue_metrics(task, all_preds, all_labels)
