"""Regression tests: exact trainable-parameter counts that must never drift.

Two tiers:

* **Synthetic** (fast, always run): exact closed-form counts for each method on
  hand-built linear stacks — the core guarantee that the parameterisation never
  silently changes.
* **Real architectures** (``hf`` + ``slow``): the canonical literature anchors,
  verified against config-built BERT/GPT-2/T5.

Spec-figure reconciliation
--------------------------
The project spec quotes three anchors. One matches exactly; two do not, because
they assume non-standard configurations. We assert the *derived, verified* counts
and document the difference rather than reverse-engineering a magic constant:

* BERT-base, LoRA r=8 on query+value  -> **294,912**  (matches spec exactly:
  12 layers x 2 projections x 8 x (768+768)).
* GPT-2, Prefix l=10                  -> **184,320**  = 2 x L x l x H
  (= 2x12x10x768, the standard no-reparam count). The spec's 7,864,320 does not
  correspond to any standard prefix parameterisation.
* T5-small, IA3 on k,v,wi             -> **24,576**   (48 rescale vectors x 512).
  The spec's 28,672 implies a different vector-length convention for the FF term.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import (
    AdapterConfig,
    DoRAConfig,
    IA3Config,
    LoRAConfig,
    PrefixConfig,
    PromptTuningConfig,
    VeRAConfig,
    get_peft_model,
)


# ---------------------------------------------------------------------------
# Synthetic, fast, always-on exact counts
# ---------------------------------------------------------------------------
def _stack(dims: list[tuple[int, int]]) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(*[nn.Linear(i, o) for i, o in dims])


def test_lora_exact_count():
    model = _stack([(64, 64), (64, 128)])
    peft = get_peft_model(model, LoRAConfig(r=8, target_modules=["0", "1"]))
    # 8*(64+64) + 8*(64+128) = 1024 + 1536
    assert peft.get_nb_trainable_parameters()[0] == 8 * (64 + 64) + 8 * (64 + 128) == 2560


def test_dora_exact_count():
    model = _stack([(64, 64)])
    peft = get_peft_model(model, DoRAConfig(r=8, target_modules=["0"]))
    assert peft.get_nb_trainable_parameters()[0] == 8 * (64 + 64) + 64  # + magnitude


def test_vera_exact_count():
    model = _stack([(64, 64), (64, 64)])
    peft = get_peft_model(model, VeRAConfig(r=32, target_modules=["0", "1"]))
    assert peft.get_nb_trainable_parameters()[0] == 2 * (32 + 64)  # (d=r) + (b=out)


def test_ia3_exact_count():
    model = _stack([(64, 128), (128, 64)])
    # layer 0 output-rescale (out=128), layer 1 input-rescale (in=128)
    peft = get_peft_model(model, IA3Config(target_modules=["0", "1"], feedforward_modules=["1"]))
    assert peft.get_nb_trainable_parameters()[0] == 128 + 128


def test_adapter_exact_count():
    model = _stack([(64, 64)])
    peft = get_peft_model(model, AdapterConfig(target_modules=["0"], bottleneck_dim=16))
    # 2*D*b + b + D = 2*64*16 + 16 + 64
    assert peft.get_nb_trainable_parameters()[0] == 2 * 64 * 16 + 16 + 64


# ---------------------------------------------------------------------------
# Real-architecture anchors (heavier; gated behind hf + slow)
# ---------------------------------------------------------------------------
@pytest.mark.hf
@pytest.mark.slow
def test_bert_base_lora_is_294912():
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    model = BertModel(BertConfig())  # bert-base: hidden 768, 12 layers
    peft = get_peft_model(model, LoRAConfig(r=8, target_modules=["query", "value"]))
    assert peft.get_nb_trainable_parameters()[0] == 294_912


@pytest.mark.hf
@pytest.mark.slow
def test_gpt2_prefix_l10_is_184320():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config())  # 12 layers, hidden 768
    peft = get_peft_model(model, PrefixConfig(num_virtual_tokens=10))
    # 2 * L(12) * l(10) * H(768) = 184320  (standard no-reparam prefix count).
    assert peft.get_nb_trainable_parameters()[0] == 2 * 12 * 10 * 768 == 184_320


@pytest.mark.hf
@pytest.mark.slow
def test_gpt2_lora_c_attn_is_294912():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config())
    peft = get_peft_model(model, LoRAConfig(r=8))  # infers c_attn (Conv1D 768->2304)
    # 12 layers * 8 * (768 + 2304) = 294912.
    assert peft.get_nb_trainable_parameters()[0] == 294_912


@pytest.mark.hf
@pytest.mark.slow
def test_t5_small_ia3_is_24576():
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    cfg = T5Config(d_model=512, d_ff=2048, d_kv=64, num_layers=6, num_heads=8, vocab_size=32128)
    model = T5ForConditionalGeneration(cfg)
    peft = get_peft_model(
        model, IA3Config(target_modules=["k", "v", "wi"], feedforward_modules=["wi"])
    )
    # 48 rescale vectors of length 512 (enc: 6*(k,v,wi); dec: 6*(k,v,k,v,wi)).
    assert peft.get_nb_trainable_parameters()[0] == 24_576


@pytest.mark.hf
@pytest.mark.slow
def test_gpt2_prompt_tuning_count():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config())  # hidden 768
    peft = get_peft_model(model, PromptTuningConfig(num_virtual_tokens=20))
    assert peft.get_nb_trainable_parameters()[0] == 20 * 768  # 15360
