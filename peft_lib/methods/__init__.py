"""Concrete PEFT methods. Importing a method module registers it.

The top-level :mod:`peft_lib` package imports each of these for their
registration side effects, so end users rarely import from here directly.
"""

from __future__ import annotations

from peft_lib.methods.adapters import AdapterConfig, AdapterLayer, AdapterModel
from peft_lib.methods.dora import DoRAConfig, DoRALinear, DoRAModel
from peft_lib.methods.ia3 import IA3Config, IA3Linear, IA3Model
from peft_lib.methods.lora import LoRAConfig, LoRALinear, LoRAModel
from peft_lib.methods.prefix_tuning import PrefixConfig, PrefixEncoder, PrefixModel
from peft_lib.methods.prompt_tuning import PromptModel, PromptTuningConfig, SoftPromptEmbedding
from peft_lib.methods.qlora import QLoRAConfig, QLoRAModel
from peft_lib.methods.vera import VeRAConfig, VeRALinear, VeRAModel

__all__ = [
    "AdapterConfig",
    "AdapterLayer",
    "AdapterModel",
    "DoRAConfig",
    "DoRALinear",
    "DoRAModel",
    "IA3Config",
    "IA3Linear",
    "IA3Model",
    "LoRAConfig",
    "LoRALinear",
    "LoRAModel",
    "PrefixConfig",
    "PrefixEncoder",
    "PrefixModel",
    "PromptModel",
    "PromptTuningConfig",
    "QLoRAConfig",
    "QLoRAModel",
    "SoftPromptEmbedding",
    "VeRAConfig",
    "VeRALinear",
    "VeRAModel",
]
