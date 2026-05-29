"""Property-based tests (hypothesis): invariants that must hold for *all* shapes.

We assert the universally-true properties of each method — output shape, exact
parameter count, and zero-delta initialisation — across randomly sampled
dimensions, ranks, and batch/seq sizes.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from peft_lib.methods.dora import DoRALinear
from peft_lib.methods.ia3 import IA3Linear
from peft_lib.methods.lora import LoRALinear
from peft_lib.methods.vera import VeRALinear

# Disable hypothesis's per-example deadline: tensor ops vary in wall-time.
SETTINGS = settings(deadline=None, max_examples=40)

dims = st.integers(min_value=1, max_value=48)
ranks = st.integers(min_value=1, max_value=64)
batches = st.integers(min_value=1, max_value=8)
seqs = st.integers(min_value=1, max_value=32)


@SETTINGS
@given(batch=batches, seq=seqs, rank=ranks, in_f=dims, out_f=dims)
def test_lora_output_shape_is_always_correct(batch, seq, rank, in_f, out_f):
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(in_f, out_f), r=rank)
    out = lora(torch.randn(batch, seq, in_f))
    assert out.shape == (batch, seq, out_f)


@SETTINGS
@given(rank=ranks, in_f=dims, out_f=dims)
def test_lora_param_count_matches_formula(rank, in_f, out_f):
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(in_f, out_f), r=rank)
    n = sum(p.numel() for name, p in lora.named_parameters() if not name.startswith("base_layer"))
    assert n == rank * (in_f + out_f)


@SETTINGS
@given(rank=ranks, in_f=dims, out_f=dims)
def test_dora_param_count_matches_formula(rank, in_f, out_f):
    torch.manual_seed(0)
    dora = DoRALinear(nn.Linear(in_f, out_f), r=rank)
    n = sum(p.numel() for name, p in dora.named_parameters() if not name.startswith("base_layer"))
    assert n == rank * (in_f + out_f) + out_f  # + magnitude


@SETTINGS
@given(rank=ranks, in_f=dims, out_f=dims)
def test_vera_param_count_matches_formula(rank, in_f, out_f):
    torch.manual_seed(0)
    vera = VeRALinear(nn.Linear(in_f, out_f), r=rank)
    n = sum(p.numel() for name, p in vera.named_parameters() if not name.startswith("base_layer"))
    assert n == rank + out_f  # d (r) + b (out)


@SETTINGS
@given(batch=batches, in_f=dims, out_f=dims, rank=ranks)
def test_lora_zero_delta_init_for_any_shape(batch, in_f, out_f, rank):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f)
    lora = LoRALinear(lin, r=rank)
    x = torch.randn(batch, in_f)
    assert torch.allclose(lora(x), lin(x), atol=1e-5)


@SETTINGS
@given(batch=batches, in_f=dims, out_f=dims)
def test_ia3_zero_delta_init_for_any_shape(batch, in_f, out_f):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f)
    for ff in (True, False):
        ia3 = IA3Linear(lin, is_feedforward=ff)
        x = torch.randn(batch, in_f)
        assert torch.allclose(ia3(x), lin(x), atol=1e-5)


@SETTINGS
@given(batch=batches, in_f=dims, out_f=dims, rank=st.integers(1, 16))
def test_lora_merge_equivalence_for_any_shape(batch, in_f, out_f, rank):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f)
    lora = LoRALinear(lin, r=rank)
    lora.lora_B.data.normal_()  # make the delta non-trivial
    x = torch.randn(batch, in_f)
    before = lora(x).detach().clone()
    lora.merge()
    assert torch.allclose(before, lora(x), atol=1e-4)
