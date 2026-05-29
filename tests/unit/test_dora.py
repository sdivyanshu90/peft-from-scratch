"""DoRA-specific unit tests (magnitude decomposition); the 10-test contract lives
in ``test_injection_contract.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib.core.exceptions import ConfigError
from peft_lib.methods.dora import DoRAConfig, DoRALinear


def test_magnitude_initialised_to_row_norm():
    torch.manual_seed(42)
    lin = nn.Linear(16, 8, bias=False)
    dora = DoRALinear(lin, r=4)
    expected = torch.linalg.norm(lin.weight, dim=1)  # (out=8,)
    assert dora.magnitude.shape == (8,)
    assert torch.allclose(dora.magnitude, expected, atol=1e-6)


def test_effective_weight_equals_w0_at_init():
    torch.manual_seed(42)
    lin = nn.Linear(16, 8)
    dora = DoRALinear(lin, r=4)
    # W' == W0 because B=0 and m=||W0||.
    assert torch.allclose(dora._effective_weight(), lin.weight, atol=1e-5)
    assert torch.count_nonzero(dora.get_delta_weight().abs() > 1e-5) == 0


def test_magnitude_changes_weight_norm_after_training():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    dora = DoRALinear(lin, r=4)
    x = torch.randn(4, 16)
    opt = torch.optim.SGD(dora.parameters(), lr=0.5)
    for _ in range(10):
        opt.zero_grad()
        dora(x).pow(2).mean().backward()
        opt.step()
    # After training the magnitude vector has moved away from the initial norm.
    init_norm = torch.linalg.norm(lin.weight, dim=1)
    assert not torch.allclose(dora.magnitude, init_norm, atol=1e-3)


def test_dora_costs_exactly_out_more_than_lora():
    torch.manual_seed(0)
    dora = DoRALinear(nn.Linear(16, 8), r=4)
    # Count adapter-only params (a standalone layer does not freeze its base_layer;
    # that is the owning model's responsibility).
    n = sum(p.numel() for name, p in dora.named_parameters() if not name.startswith("base_layer"))
    assert n == 4 * (16 + 8) + 8  # r*(in+out) + out (magnitude) = 104


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"alpha": -1}, {"dropout": 1.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        DoRAConfig(target_modules=["q"], **kwargs)
