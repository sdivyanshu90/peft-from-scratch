"""Integration tests for the merging module: LoRA folding, soups, TIES."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import IA3Config, LoRAConfig, get_peft_model
from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.merging import (
    make_task_vector,
    ties_merge,
    ties_merge_into,
    uniform_soup,
    weighted_merge_adapters,
    weighted_soup,
)


# --- weighted_merge_adapters ------------------------------------------------
def test_weighted_merge_matches_scaled_delta():
    torch.manual_seed(0)
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    peft.adapter_layers[""].lora_B.data.normal_()
    w0 = peft.adapter_layers[""].base_layer.weight.detach().clone()
    delta = peft.adapter_layers[""].get_delta_weight().detach().clone()
    sd = {k: v.clone() for k, v in peft.adapter_state_dict().items()}

    bare = weighted_merge_adapters(peft, [sd, sd], [0.3, 0.2])  # combined weight 0.5
    assert isinstance(bare, nn.Linear)
    assert torch.allclose(bare.weight, w0 + 0.5 * delta, atol=1e-5)


def test_weighted_merge_rejects_ia3():
    peft = get_peft_model(nn.Linear(8, 8), IA3Config(target_modules=[""]))
    sd = peft.adapter_state_dict()
    with pytest.raises(MergeError, match="additive"):
        weighted_merge_adapters(peft, [sd], [1.0])


def test_weighted_merge_validates_lengths_and_keys():
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    sd = peft.adapter_state_dict()
    with pytest.raises(MergeError, match="weights"):
        weighted_merge_adapters(peft, [sd], [1.0, 2.0])
    with pytest.raises(MergeError, match="keys"):
        weighted_merge_adapters(peft, [{"bad.key": torch.zeros(1)}], [1.0])


# --- model soup -------------------------------------------------------------
def test_uniform_soup_averages():
    a = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([1.0])}
    b = {"w": torch.tensor([4.0, 0.0]), "b": torch.tensor([3.0])}
    soup = uniform_soup([a, b])
    assert soup["w"].tolist() == [2.0, 1.0]
    assert soup["b"].tolist() == [2.0]


def test_weighted_soup_normalises():
    a = {"w": torch.zeros(2)}
    b = {"w": torch.ones(2) * 4}
    soup = weighted_soup([a, b], [1.0, 3.0])  # -> 0.25*0 + 0.75*4 = 3
    assert torch.allclose(soup["w"], torch.full((2,), 3.0))


def test_soup_rejects_mismatched_keys():
    with pytest.raises(MergeError):
        uniform_soup([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}])


def test_soup_rejects_empty():
    with pytest.raises(MergeError):
        uniform_soup([])


# --- TIES -------------------------------------------------------------------
def test_make_task_vector():
    ft = {"w": torch.tensor([2.0, 3.0])}
    base = {"w": torch.tensor([1.0, 1.0])}
    assert make_task_vector(ft, base)["w"].tolist() == [1.0, 2.0]


def test_ties_resolves_sign_conflict():
    # dim 0: both positive (agree) -> mean(1,3)=2 ; dim 1: +2 vs -2 conflict.
    t1 = {"w": torch.tensor([1.0, 2.0])}
    t2 = {"w": torch.tensor([3.0, -2.0])}
    merged = ties_merge([t1, t2], density=1.0)["w"]
    assert merged[0].item() == pytest.approx(2.0)
    # Conflicting dim: elected sign decided by the sum (2 + -2 = 0 -> sign 0 -> 0).
    assert merged[1].item() == pytest.approx(0.0)


def test_ties_elects_majority_sign():
    t1 = {"w": torch.tensor([5.0])}
    t2 = {"w": torch.tensor([1.0])}
    t3 = {"w": torch.tensor([-1.0])}
    # sum = 5 -> elected +; average of agreeing (5, 1) = 3.
    merged = ties_merge([t1, t2, t3], density=1.0)["w"]
    assert merged[0].item() == pytest.approx(3.0)


def test_ties_trim_keeps_top_magnitude():
    # density 0.5 keeps the single largest-magnitude entry per task.
    t1 = {"w": torch.tensor([0.1, 5.0])}
    merged = ties_merge([t1], density=0.5)["w"]
    assert merged[0].item() == pytest.approx(0.0)  # trimmed
    assert merged[1].item() == pytest.approx(5.0)  # kept


def test_ties_merge_into_adds_to_base():
    base = {"w": torch.zeros(2)}
    t1 = {"w": torch.tensor([1.0, 2.0])}
    t2 = {"w": torch.tensor([3.0, -2.0])}
    out = ties_merge_into(base, [t1, t2], density=1.0, scaling=2.0)
    assert out["w"][0].item() == pytest.approx(4.0)  # 0 + 2 * mean(1,3)


def test_ties_invalid_density():
    with pytest.raises(ConfigError):
        ties_merge([{"w": torch.zeros(2)}], density=0.0)


def test_ties_mismatched_keys():
    with pytest.raises(MergeError):
        ties_merge([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}])
