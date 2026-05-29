"""Memory and latency profiling for PEFT vs full fine-tuning.

Two complementary tools:

* **Analytical** (``lora_flop_overhead_ratio``, ``optimizer_state_bytes``):
  deterministic, environment-independent estimates derived from the math — ideal
  for regression assertions (e.g. "LoRA adds < 5% forward FLOPs").
* **Empirical** (``profile_call``, ``measure_forward_overhead``): wall-clock and
  (on CUDA) peak-memory measurements of real calls, for reporting.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = [
    "MemoryReport",
    "lora_flop_overhead_ratio",
    "measure_forward_overhead",
    "optimizer_state_bytes",
    "profile_call",
]


@dataclass
class MemoryReport:
    """Result of :func:`profile_call`.

    Attributes:
        wall_time_s: Median wall-clock seconds per call (after warmup).
        peak_bytes: Peak CUDA memory during the timed region (0 on CPU).
        device: The device the call ran on.
    """

    wall_time_s: float
    peak_bytes: int
    device: str


def lora_flop_overhead_ratio(in_features: int, out_features: int, rank: int) -> float:
    """Analytical forward-FLOP overhead of a LoRA layer vs the base linear.

    A linear forward costs ~``2 * in * out`` MACs/token; LoRA adds
    ``2 * rank * (in + out)``. The ratio is therefore::

        overhead = (rank * (in + out)) / (in * out).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.

    Returns:
        The fractional overhead (e.g. ``0.0208`` = 2.08%).

    Example:
        >>> round(lora_flop_overhead_ratio(768, 768, 8), 4)
        0.0208
    """
    return rank * (in_features + out_features) / (in_features * out_features)


def optimizer_state_bytes(model: nn.Module, *, bytes_per_state: int = 8) -> int:
    """Estimate optimizer-state bytes for the *trainable* parameters.

    Adam keeps two fp32 moments per parameter (~8 bytes). Because PEFT trains only
    the adapters, this is where the headline VRAM saving comes from.

    Args:
        model: The (PEFT-wrapped) model.
        bytes_per_state: Bytes of optimizer state per trainable element (Adam: 8).

    Returns:
        Estimated optimizer-state size in bytes.

    Example:
        >>> import torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> optimizer_state_bytes(peft)  # 64 trainable * 8
        512
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable * bytes_per_state


def profile_call(
    fn: Callable[[], Any],
    *,
    device: str = "cpu",
    warmup: int = 2,
    iters: int = 5,
) -> MemoryReport:
    """Time a callable and (on CUDA) record peak memory.

    Args:
        fn: A zero-arg callable performing the work to measure.
        device: ``"cpu"`` or ``"cuda"``.
        warmup: Untimed warmup calls (to amortise allocation/JIT).
        iters: Timed iterations; the median wall time is reported.

    Returns:
        A :class:`MemoryReport`.

    Example:
        >>> import torch
        >>> r = profile_call(lambda: torch.randn(4, 4) @ torch.randn(4, 4), iters=2, warmup=0)
        >>> r.wall_time_s >= 0.0
        True
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    for _ in range(max(1, iters)):
        start = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    peak = int(torch.cuda.max_memory_allocated()) if is_cuda else 0
    times.sort()
    return MemoryReport(wall_time_s=times[len(times) // 2], peak_bytes=peak, device=device)


def measure_forward_overhead(
    base_layer: nn.Module,
    peft_layer: nn.Module,
    example_input: torch.Tensor,
    *,
    device: str = "cpu",
    iters: int = 20,
) -> float:
    """Empirical forward-time overhead of ``peft_layer`` vs ``base_layer``.

    Args:
        base_layer: The original module.
        peft_layer: The adapted module (same I/O signature).
        example_input: A representative input tensor.
        device: Device to run on.
        iters: Timed iterations.

    Returns:
        ``(peft_time / base_time) - 1`` (e.g. ``0.03`` for 3% slower). Note that
        wall-clock ratios are noisy; prefer :func:`lora_flop_overhead_ratio` for
        assertions.

    Example:
        >>> import torch
        >>> from peft_lib.methods.lora import LoRALinear
        >>> lin = torch.nn.Linear(64, 64)
        >>> ov = measure_forward_overhead(lin, LoRALinear(lin, r=8),
        ...                               torch.randn(4, 64), iters=2)
        >>> isinstance(ov, float)
        True
    """
    base_layer = base_layer.to(device).eval()
    peft_layer = peft_layer.to(device).eval()
    example_input = example_input.to(device)
    gc.collect()
    with torch.no_grad():
        base = profile_call(lambda: base_layer(example_input), device=device, iters=iters)
        peft = profile_call(lambda: peft_layer(example_input), device=device, iters=iters)
    if base.wall_time_s <= 0:
        return 0.0
    return peft.wall_time_s / base.wall_time_s - 1.0
