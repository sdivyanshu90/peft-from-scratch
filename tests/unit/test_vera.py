"""VeRA-specific unit tests (frozen shared projections); the 10-test contract
lives in ``test_injection_contract.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib.core.exceptions import ConfigError
from peft_lib.methods.vera import VeRAConfig, VeRALinear


def test_projection_matrices_are_frozen():
    vera = VeRALinear(nn.Linear(16, 8), r=32)
    # vera_A / vera_B are buffers (not parameters) -> never trained, never saved.
    param_names = {n for n, _ in vera.named_parameters()}
    assert "vera_A" not in param_names and "vera_B" not in param_names
    buffer_names = {n for n, _ in vera.named_buffers()}
    assert {"vera_A", "vera_B"} <= buffer_names


def test_only_d_and_b_are_trainable():
    vera = VeRALinear(nn.Linear(16, 8), r=32)
    # Adapter-only trainable params (a standalone layer does not freeze its base).
    trainable = {
        n for n, p in vera.named_parameters() if p.requires_grad and not n.startswith("base_layer")
    }
    assert trainable == {"vera_lambda_d", "vera_lambda_b"}
    assert vera.vera_lambda_d.shape == (32,)
    assert vera.vera_lambda_b.shape == (8,)


def test_same_seed_reproduces_projections():
    a = VeRALinear(nn.Linear(16, 8), r=32, projection_seed=7)
    b = VeRALinear(nn.Linear(16, 8), r=32, projection_seed=7)
    c = VeRALinear(nn.Linear(16, 8), r=32, projection_seed=8)
    assert torch.equal(a.vera_A, b.vera_A) and torch.equal(a.vera_B, b.vera_B)
    assert not torch.equal(a.vera_A, c.vera_A)


def test_b_vector_zero_init_gives_zero_delta():
    vera = VeRALinear(nn.Linear(16, 8), r=32)
    assert torch.count_nonzero(vera.vera_lambda_b) == 0
    assert torch.count_nonzero(vera.get_delta_weight()) == 0


def test_d_init_value():
    vera = VeRALinear(nn.Linear(16, 8), r=4, d_init=0.25)
    assert torch.allclose(vera.vera_lambda_d, torch.full((4,), 0.25))


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"dropout": 1.0}])
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        VeRAConfig(target_modules=["q"], **kwargs)
