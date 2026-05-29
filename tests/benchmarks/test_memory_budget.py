"""Benchmark-budget tests: LoRA forward overhead and optimizer-memory savings.

Analytical assertions are deterministic and run everywhere. The wall-clock test
is marked ``slow`` and uses a lenient bound (timing is noisy, especially on CPU).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import LoRAConfig, get_peft_model
from peft_lib.benchmarks import (
    lora_flop_overhead_ratio,
    measure_forward_overhead,
    optimizer_state_bytes,
)
from peft_lib.methods.lora import LoRALinear


def test_lora_forward_flop_overhead_under_5_percent():
    # The spec's budget: < 5% on (B=8, S=512, D=768) at r=8.
    overhead = lora_flop_overhead_ratio(768, 768, 8)
    assert overhead < 0.05
    assert overhead == pytest.approx(0.0208, abs=1e-3)


@pytest.mark.parametrize("rank", [1, 4, 8, 16, 32])
def test_overhead_grows_linearly_with_rank(rank):
    base = lora_flop_overhead_ratio(768, 768, 1)
    assert lora_flop_overhead_ratio(768, 768, rank) == pytest.approx(base * rank)


def test_optimizer_state_savings_vs_full_finetune():
    torch.manual_seed(0)
    linear = nn.Linear(768, 768)
    peft = get_peft_model(linear, LoRAConfig(r=8, target_modules=[""]))
    full = sum(p.numel() for p in nn.Linear(768, 768).parameters()) * 8
    lora = optimizer_state_bytes(peft)
    # LoRA optimizer state should be at least 40x smaller than full fine-tuning.
    assert lora * 40 < full
    assert lora == 8 * (768 + 768) * 8  # 8 bytes per trainable element


@pytest.mark.slow
def test_measured_forward_overhead_is_bounded():
    torch.manual_seed(0)
    linear = nn.Linear(768, 768)
    lora = LoRALinear(linear, r=8)
    x = torch.randn(8, 512, 768)
    overhead = measure_forward_overhead(linear, lora, x, iters=30)
    # Wall-clock ratios are noisy; assert a generous sanity bound only.
    assert overhead < 1.0, f"LoRA forward overhead unexpectedly high: {overhead:.2%}"
