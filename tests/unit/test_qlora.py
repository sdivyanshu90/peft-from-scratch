"""Unit tests for QLoRA + bitsandbytes helpers.

bitsandbytes needs a CUDA GPU; this environment may have only a broken CPU stub.
So config/validation/guard tests run everywhere, while the construction +
merge-raises tests are gated behind a genuinely usable bnb (``_bnb_usable``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft_lib import QLoRAConfig, get_peft_model
from peft_lib.core.exceptions import ConfigError, DeviceError, MergeError
from peft_lib.quantization import bnb_utils


def _bnb_usable() -> bool:
    """Return True iff bitsandbytes can actually build a 4-bit layer here."""
    if not bnb_utils.is_bnb_available():
        return False
    try:
        import bitsandbytes as bnb

        bnb.nn.Linear4bit(8, 8, bias=False)
        return torch.cuda.is_available()
    except Exception:
        return False


# --- config / validation (no bnb needed) -----------------------------------
def test_peft_type_and_inherited_lora_fields():
    cfg = QLoRAConfig(r=16, alpha=32, bits=4, target_modules=["q_proj"])
    assert cfg.peft_type == "qlora"
    assert cfg.r == 16 and cfg.alpha == 32 and cfg.scaling == 2.0


@pytest.mark.parametrize("kwargs", [{"bits": 3}, {"quant_type": "int4"}, {"r": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        QLoRAConfig(target_modules=["q_proj"], **kwargs)


def test_config_serialization(tmp_path):
    cfg = QLoRAConfig(r=8, bits=4, quant_type="nf4", double_quant=True, target_modules=["q"])
    assert QLoRAConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert QLoRAConfig.load(cfg.save(tmp_path)).bits == 4


# --- graceful guard when bnb is unavailable ---------------------------------
def test_require_bnb_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(bnb_utils, "is_bnb_available", lambda: False)
    with pytest.raises(DeviceError, match="bitsandbytes is required"):
        bnb_utils.require_bnb()


def test_replace_raises_without_bnb(monkeypatch):
    monkeypatch.setattr(bnb_utils, "is_bnb_available", lambda: False)
    with pytest.raises(DeviceError):
        bnb_utils.replace_with_bnb_linear(nn.Linear(8, 8))


def test_qlora_construction_raises_without_bnb(monkeypatch):
    monkeypatch.setattr(bnb_utils, "is_bnb_available", lambda: False)
    with pytest.raises(DeviceError):
        get_peft_model(nn.Linear(32, 32), QLoRAConfig(r=8, target_modules=[""]))


def test_dtype_helper():
    assert bnb_utils.dtype_from_str("bfloat16") == torch.bfloat16
    with pytest.raises(ConfigError):
        bnb_utils.dtype_from_str("float8")


# --- real construction (needs working bnb + CUDA) ---------------------------
@pytest.mark.quant
@pytest.mark.skipif(not _bnb_usable(), reason="needs a working bitsandbytes + CUDA")
def test_qlora_builds_and_merge_raises():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(32, 32))
    peft = get_peft_model(base, QLoRAConfig(r=8, target_modules=["0"])).cuda()
    trn, _ = peft.get_nb_trainable_parameters()
    assert trn == 8 * (32 + 32)  # LoRA-equivalent trainable count
    with pytest.raises(MergeError, match="4-bit"):
        peft.merge_and_unload()
