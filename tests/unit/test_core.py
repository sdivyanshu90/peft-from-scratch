"""Unit tests for the core layer: config (de)serialisation, registry, utils, and
the PEFTModel base behaviours (generate passthrough, from_pretrained guards).
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from peft_lib import (
    DoRAConfig,
    DoRAModel,
    LoRAConfig,
    LoRAModel,
    available_methods,
    get_peft_model,
)
from peft_lib.core.base import PEFTConfig, infer_target_modules
from peft_lib.core.exceptions import ConfigError, MergeError, PEFTError, ShapeError
from peft_lib.core.registry import get_entry, register_peft
from peft_lib.core.utils import (
    get_submodule,
    human_readable,
    iter_target_modules,
    match_target,
    set_submodule,
)


# --- exceptions -------------------------------------------------------------
def test_shape_error_formats_shapes():
    err = ShapeError("bad", expected=(2, 3), actual=(2, 4))
    assert "expected (2, 3)" in str(err) and "got (2, 4)" in str(err)
    assert isinstance(err, PEFTError)


# --- config serialisation ---------------------------------------------------
def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ConfigError, match="Unknown config keys"):
        LoRAConfig.from_dict({"r": 8, "bogus_field": 1})


def test_config_save_to_explicit_json_path(tmp_path):
    path = tmp_path / "nested" / "my_cfg.json"
    written = LoRAConfig(r=8, target_modules=["q"]).save(path)
    assert written == path and path.exists()
    assert json.loads(path.read_text())["r"] == 8


def test_abstract_peft_config_cannot_be_used():
    with pytest.raises(ConfigError, match="abstract"):
        PEFTConfig()


# --- registry ---------------------------------------------------------------
def test_available_methods_contains_all_builtins():
    assert {
        "lora",
        "dora",
        "vera",
        "ia3",
        "prefix_tuning",
        "prompt_tuning",
        "adapter",
        "qlora",
    } <= set(available_methods())


def test_get_entry_unknown_raises():
    with pytest.raises(ConfigError, match="Unknown PEFT method"):
        get_entry("does_not_exist")


def test_register_duplicate_raises():
    with pytest.raises(ConfigError, match="already registered"):
        register_peft("lora")(LoRAModel)


def test_register_without_config_class_raises():
    class Bare(nn.Module):
        pass

    with pytest.raises(ConfigError, match="config_class"):
        register_peft("brand-new-method")(Bare)


# --- utils ------------------------------------------------------------------
def test_get_submodule_resolves_and_errors():
    m = nn.Sequential(nn.Linear(2, 2))
    assert get_submodule(m, "0").out_features == 2
    assert get_submodule(m, "") is m
    with pytest.raises(AttributeError):
        get_submodule(m, "nope")


def test_set_submodule_rejects_root():
    with pytest.raises(ConfigError, match="root module"):
        set_submodule(nn.Linear(2, 2), "", nn.Linear(2, 2))


def test_match_target_single_string():
    assert match_target("a.b.q_proj", "q_proj")
    assert not match_target("a.b.q_proj", "v_proj")


def test_iter_target_modules_all_linear_excludes_heads():
    m = nn.ModuleDict({"q": nn.Linear(4, 4), "lm_head": nn.Linear(4, 8), "ln": nn.LayerNorm(4)})
    names = [n for n, _ in iter_target_modules(m, "all-linear")]
    assert names == ["q"]  # lm_head excluded, LayerNorm is not linear


def test_iter_target_modules_no_match_raises():
    with pytest.raises(ConfigError, match="No linear-like modules matched"):
        iter_target_modules(nn.Linear(4, 4), ["nonexistent"])


@pytest.mark.parametrize(
    ("count", "expected"),
    [(500, "500"), (294_912, "294.9K"), (7_864_320, "7.9M"), (2_000_000_000, "2.0B")],
)
def test_human_readable(count, expected):
    assert human_readable(count) == expected


# --- target inference -------------------------------------------------------
def test_infer_target_modules_known_and_unknown():
    class Cfg:
        model_type = "llama"

    class Known(nn.Module):
        config = Cfg()

    assert infer_target_modules(Known()) == ["q_proj", "v_proj"]

    class UnknownCfg:
        model_type = "totally-unknown-arch"

    class Unknown(nn.Module):
        config = UnknownCfg()

    assert infer_target_modules(Unknown()) is None
    assert infer_target_modules(nn.Linear(4, 4)) is None  # no .config


def test_get_peft_model_requires_targets_when_uninferable():
    with pytest.raises(ConfigError, match="target_modules` is required"):
        get_peft_model(nn.Linear(4, 4), LoRAConfig(r=4))  # None targets, no config


# --- PEFTModel base behaviours ----------------------------------------------
class _GenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + 1.0


def test_generate_passthrough_and_missing():
    torch.manual_seed(0)
    peft = get_peft_model(_GenModel(), LoRAConfig(r=2, target_modules=["proj"]))
    x = torch.randn(2, 4)
    assert torch.allclose(peft.generate(x), peft(x) + 1.0)

    no_gen = get_peft_model(nn.Linear(4, 4), LoRAConfig(r=2, target_modules=[""]))
    with pytest.raises(AttributeError, match="has no `generate`"):
        no_gen.generate(x)


def test_print_trainable_parameters_format(capsys):
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    peft.print_trainable_parameters()
    out = capsys.readouterr().out
    assert "trainable params: 64" in out and "trainable%:" in out


def test_construct_with_wrong_config_type_raises():
    with pytest.raises(ConfigError, match="expects a LoRAConfig"):
        LoRAModel(nn.Linear(4, 4), DoRAConfig(r=4, target_modules=[""]))


def test_from_pretrained_peft_type_mismatch(tmp_path):
    torch.manual_seed(0)
    peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
    peft.save_pretrained(tmp_path)
    with pytest.raises(ConfigError, match="called on DoRAModel"):
        DoRAModel.from_pretrained(nn.Linear(8, 8), tmp_path)


def test_from_pretrained_key_mismatch(tmp_path):
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    peft = get_peft_model(base, LoRAConfig(r=4, target_modules=["0", "1"]))
    peft.save_pretrained(tmp_path)
    # Reconstruct with fewer targets -> the checkpoint has extra keys.
    torch.manual_seed(0)
    fresh = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    with pytest.raises(MergeError, match="does not match"):
        LoRAModel.from_pretrained(fresh, tmp_path, target_modules=["0"])
