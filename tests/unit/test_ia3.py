"""IA³-specific unit tests (output vs input rescaling, merge algebra); the
10-test contract lives in ``test_injection_contract.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.methods.ia3 import IA3Config, IA3Linear


def test_output_rescale_vector_length_is_out():
    ia3 = IA3Linear(nn.Linear(16, 8), is_feedforward=False)
    assert ia3.ia3_l.shape == (8,)


def test_input_rescale_vector_length_is_in():
    ia3 = IA3Linear(nn.Linear(16, 8), is_feedforward=True)
    assert ia3.ia3_l.shape == (16,)


def test_ones_init_is_identity():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    assert torch.allclose(IA3Linear(lin, is_feedforward=False)(x), lin(x))
    assert torch.allclose(IA3Linear(lin, is_feedforward=True)(x), lin(x))


def test_output_merge_scales_rows_and_bias():
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    w0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
    ia3 = IA3Linear(lin, is_feedforward=False)
    ia3.ia3_l.data = torch.tensor([2.0, 3.0, 4.0])
    ia3.merge()
    assert torch.allclose(lin.weight, w0 * torch.tensor([2.0, 3.0, 4.0]).view(-1, 1))
    assert torch.allclose(lin.bias, b0 * torch.tensor([2.0, 3.0, 4.0]))


def test_input_merge_scales_columns_only():
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    w0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
    ia3 = IA3Linear(lin, is_feedforward=True)
    ia3.ia3_l.data = torch.tensor([2.0, 3.0, 4.0, 5.0])
    ia3.merge()
    assert torch.allclose(lin.weight, w0 * torch.tensor([2.0, 3.0, 4.0, 5.0]).view(1, -1))
    assert torch.allclose(lin.bias, b0)  # bias untouched for input rescaling


def test_merge_unmerge_roundtrip():
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    ia3 = IA3Linear(lin, is_feedforward=False)
    ia3.ia3_l.data.uniform_(0.5, 1.5)
    x = torch.randn(3, 8)
    before = ia3(x).detach().clone()
    ia3.merge()
    assert torch.allclose(before, ia3(x), atol=1e-5)
    ia3.unmerge()
    assert torch.allclose(before, ia3(x), atol=1e-5)


def test_unmerge_with_zero_entry_raises():
    ia3 = IA3Linear(nn.Linear(8, 8), is_feedforward=False)
    ia3.ia3_l.data[0] = 0.0
    ia3.merge()
    with pytest.raises(MergeError, match="not invertible"):
        ia3.unmerge()


def test_feedforward_subset_validation():
    with pytest.raises(ConfigError, match="not in target_modules"):
        IA3Config(target_modules=["k", "v"], feedforward_modules=["wi"])


def test_model_assigns_feedforward_by_name():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    from peft_lib import get_peft_model

    cfg = IA3Config(target_modules=["0", "1"], feedforward_modules=["1"])
    peft = get_peft_model(base, cfg)
    assert peft.adapter_layers["0"].is_feedforward is False  # type: ignore[attr-defined]
    assert peft.adapter_layers["1"].is_feedforward is True  # type: ignore[attr-defined]
    # length: layer 0 output-rescale -> 16 ; layer 1 input-rescale -> 16
    assert peft.get_nb_trainable_parameters()[0] == 16 + 16
