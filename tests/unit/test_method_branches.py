"""Targeted tests for method/merge edge branches (errors, reprs, init flags).

These cover the deterministic, environment-independent branches that the headline
contract/behaviour tests do not exercise, keeping coverage high and the error
paths honest.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import (
    AdapterConfig,
    LoRAConfig,
    PrefixConfig,
    PromptTuningConfig,
    QLoRAConfig,
    get_peft_model,
)
from peft_lib.core.exceptions import ConfigError, DeviceError, MergeError, ShapeError
from peft_lib.methods.dora import DoRALinear
from peft_lib.methods.ia3 import IA3Linear
from peft_lib.methods.lora import LoRALinear, _infer_linear_dims
from peft_lib.methods.prompt_tuning import SoftPromptEmbedding
from peft_lib.methods.vera import VeRALinear


# --- _infer_linear_dims edge cases ------------------------------------------
def test_infer_dims_rejects_non_2d_weight():
    with pytest.raises(ShapeError):
        _infer_linear_dims(nn.LayerNorm(8))  # 1-D weight


def test_infer_dims_generic_2d_fallback():
    class Custom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 3))  # (out, in) convention

    in_f, out_f, fifo = _infer_linear_dims(Custom())
    assert (in_f, out_f, fifo) == (3, 5, False)


# --- extra_repr (cheap, but exercises the formatting branch) ----------------
def test_extra_reprs():
    assert "r=4" in repr(LoRALinear(nn.Linear(8, 8), r=4))
    assert "r=4" in repr(DoRALinear(nn.Linear(8, 8), r=4))
    assert "r=4" in repr(VeRALinear(nn.Linear(8, 8), r=4))
    assert "rescale" in repr(IA3Linear(nn.Linear(8, 8)))


# --- init_lora_weights=False (the Gaussian-A branch) ------------------------
def test_lora_init_false_keeps_zero_delta():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(8, 8), r=4, init_lora_weights=False)
    assert torch.count_nonzero(lora.lora_A) > 0  # A is non-zero (Gaussian)
    assert torch.count_nonzero(lora.lora_B) == 0  # B is still zero -> zero delta


def test_dora_init_false_branch():
    torch.manual_seed(0)
    dora = DoRALinear(nn.Linear(8, 8), r=4, init_lora_weights=False)
    assert torch.count_nonzero(dora.lora_A) > 0


# --- shape errors on wrong input dim ----------------------------------------
@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: DoRALinear(nn.Linear(8, 8), r=4),
        lambda: VeRALinear(nn.Linear(8, 8), r=4),
        lambda: IA3Linear(nn.Linear(8, 8)),
    ],
)
def test_wrong_input_dim_raises(layer_factory):
    with pytest.raises(ShapeError):
        layer_factory()(torch.randn(2, 7))


# --- merge/unmerge guards for DoRA & VeRA -----------------------------------
def test_dora_merge_unmerge_guards():
    torch.manual_seed(0)
    dora = DoRALinear(nn.Linear(8, 8), r=4)
    dora.lora_B.data.normal_()
    with pytest.raises(MergeError, match="not merged"):
        dora.unmerge()
    dora.merge()
    with pytest.raises(MergeError, match="already merged"):
        dora.merge()
    dora.unmerge()  # roundtrip back


def test_vera_merge_unmerge_guards():
    torch.manual_seed(0)
    vera = VeRALinear(nn.Linear(8, 8), r=4)
    vera.vera_lambda_b.data.normal_()
    with pytest.raises(MergeError, match="not merged"):
        vera.unmerge()
    x = torch.randn(2, 8)
    before = vera(x).detach().clone()
    vera.merge()
    with pytest.raises(MergeError, match="already merged"):
        vera.merge()
    assert torch.allclose(before, vera(x), atol=1e-5)  # merged forward path
    vera.unmerge()
    assert torch.allclose(before, vera(x), atol=1e-5)


def test_ia3_merge_twice_guard():
    ia3 = IA3Linear(nn.Linear(8, 8))
    ia3.merge()
    with pytest.raises(MergeError, match="already merged"):
        ia3.merge()


def test_dora_invalid_rank():
    with pytest.raises(ConfigError):
        DoRALinear(nn.Linear(8, 8), r=0)


def test_vera_invalid_rank():
    with pytest.raises(ConfigError):
        VeRALinear(nn.Linear(8, 8), r=0)


def test_lora_invalid_rank():
    with pytest.raises(ConfigError):
        LoRALinear(nn.Linear(8, 8), r=0)


def test_vera_shared_param_shape_mismatch():
    # Tied LoRA path: shared A/B with wrong shape raises ShapeError.
    bad_a = nn.Parameter(torch.randn(3, 99))
    bad_b = nn.Parameter(torch.randn(8, 3))
    with pytest.raises(ShapeError):
        LoRALinear(nn.Linear(8, 8), r=3, shared_A=bad_a, shared_B=bad_b)


def test_lora_tied_requires_both_shared():
    with pytest.raises(ConfigError, match="both shared_A and shared_B"):
        LoRALinear(nn.Linear(8, 8), r=3, shared_A=nn.Parameter(torch.randn(3, 8)))


# --- LoRA bias modes --------------------------------------------------------
def test_lora_bias_lora_only_trains_adapted_bias():
    base = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    peft = get_peft_model(base, LoRAConfig(r=4, target_modules=["0"], bias="lora_only"))
    # The adapted layer's bias should be trainable; the other layer's should not.
    assert peft.adapter_layers["0"].base_layer.bias.requires_grad
    assert not base[1].bias.requires_grad


# --- Adapter activations & layernorm ----------------------------------------
@pytest.mark.parametrize("act", ["relu", "gelu", "tanh"])
def test_adapter_activations(act):
    base = nn.Sequential(nn.Linear(8, 8))
    peft = get_peft_model(base, AdapterConfig(target_modules=["0"], non_linearity=act))
    assert peft(torch.randn(2, 8)).shape == (2, 8)


# --- Prefix / Prompt error paths --------------------------------------------
def test_prefix_requires_model_config():
    class NoConfig(nn.Module):
        pass

    with pytest.raises(DeviceError, match="`.config`"):
        get_peft_model(NoConfig(), PrefixConfig(num_virtual_tokens=2))


def test_prefix_requires_full_dims():
    class PartialCfg:
        hidden_size = 8  # missing layers/heads

    class M(nn.Module):
        config = PartialCfg()

    with pytest.raises(DeviceError, match="infer"):
        get_peft_model(M(), PrefixConfig(num_virtual_tokens=2))


def test_prefix_forward_requires_input_ids():
    class Cfg:
        hidden_size = 8
        num_hidden_layers = 1
        num_attention_heads = 2

    class M(nn.Module):
        config = Cfg()

    peft = get_peft_model(M(), PrefixConfig(num_virtual_tokens=2))
    with pytest.raises(ConfigError, match="requires input_ids"):
        peft(input_ids=None)


class _MiniEmbModel(nn.Module):
    """A minimal embedding-based model for Prompt Tuning tests (no transformers)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.out = nn.Linear(8, 10)

    def get_input_embeddings(self) -> nn.Module:
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None):
        return self.out(inputs_embeds)


def test_prompt_forward_requires_some_input():
    peft = get_peft_model(_MiniEmbModel(), PromptTuningConfig(num_virtual_tokens=2))
    with pytest.raises(ConfigError, match="input_ids or inputs_embeds"):
        peft()


def test_prompt_needs_embeddings():
    class NoEmb(nn.Module):
        pass

    with pytest.raises(DeviceError, match="get_input_embeddings"):
        get_peft_model(NoEmb(), PromptTuningConfig(num_virtual_tokens=2))


def test_soft_prompt_init_embeddings_shape_check():
    with pytest.raises(ConfigError, match="init_embeddings must be"):
        SoftPromptEmbedding(4, 8, init_embeddings=torch.randn(3, 8))


# --- QLoRA branches without bitsandbytes ------------------------------------
def test_qlora_without_quantize_base_builds_and_merge_raises():
    # quantize_base=False skips the bnb path -> plain LoRA over the base.
    torch.manual_seed(0)
    peft = get_peft_model(
        nn.Linear(8, 8), QLoRAConfig(r=4, target_modules=[""], quantize_base=False)
    )
    assert peft.get_nb_trainable_parameters()[0] == 4 * (8 + 8)
    with pytest.raises(MergeError, match="4-bit"):
        peft.merge_and_unload()


def test_replace_with_bnb_linear_rejects_bad_bits():
    from peft_lib.quantization import bnb_utils

    if not bnb_utils.is_bnb_available():
        pytest.skip("bitsandbytes not importable")
    with pytest.raises(ConfigError, match="bits must be 4 or 8"):
        bnb_utils.replace_with_bnb_linear(nn.Linear(8, 8), bits=3)
