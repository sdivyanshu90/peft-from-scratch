"""peft_lib: a from-scratch, type-safe Parameter-Efficient Fine-Tuning library.

This module is the public surface. Importing it registers every built-in method,
so :func:`get_peft_model` and :meth:`PEFTModel.from_pretrained` can resolve any
method by name.

Quickstart:
    >>> import torch.nn as nn
    >>> from peft_lib import LoRAConfig, get_peft_model
    >>> base = nn.Sequential(nn.Linear(16, 16))
    >>> peft = get_peft_model(base, LoRAConfig(r=8, alpha=16, target_modules=["0"]))
    >>> peft.print_trainable_parameters()  # doctest: +ELLIPSIS
    trainable params: 256 || all params: ... || trainable%: ...
"""

from __future__ import annotations

from peft_lib.core import (
    ConfigError,
    DeviceError,
    InjectionPEFTModel,
    MergeError,
    PEFTConfig,
    PEFTError,
    PEFTModel,
    ShapeError,
    available_methods,
    freeze_model,
    get_nb_trainable_parameters,
    get_peft_model,
    register_peft,
)

# --- Method registration (import for side effects) --------------------------
# Each import runs an @register_peft decorator, populating the registry.
from peft_lib.methods.adapters import AdapterConfig, AdapterLayer, AdapterModel
from peft_lib.methods.dora import DoRAConfig, DoRALinear, DoRAModel
from peft_lib.methods.ia3 import IA3Config, IA3Linear, IA3Model
from peft_lib.methods.lora import LoRAConfig, LoRALinear, LoRAModel
from peft_lib.methods.prefix_tuning import PrefixConfig, PrefixEncoder, PrefixModel
from peft_lib.methods.prompt_tuning import PromptModel, PromptTuningConfig, SoftPromptEmbedding
from peft_lib.methods.qlora import QLoRAConfig, QLoRAModel
from peft_lib.methods.vera import VeRAConfig, VeRALinear, VeRAModel

__version__ = "0.1.0"

__all__ = [
    "AdapterConfig",
    "AdapterLayer",
    "AdapterModel",
    "ConfigError",
    "DeviceError",
    "DoRAConfig",
    "DoRALinear",
    "DoRAModel",
    "IA3Config",
    "IA3Linear",
    "IA3Model",
    "InjectionPEFTModel",
    "LoRAConfig",
    "LoRALinear",
    "LoRAModel",
    "MergeError",
    "PEFTConfig",
    "PEFTError",
    "PEFTModel",
    "PrefixConfig",
    "PrefixEncoder",
    "PrefixModel",
    "PromptModel",
    "PromptTuningConfig",
    "QLoRAConfig",
    "QLoRAModel",
    "ShapeError",
    "SoftPromptEmbedding",
    "VeRAConfig",
    "VeRALinear",
    "VeRAModel",
    "__version__",
    "available_methods",
    "freeze_model",
    "get_nb_trainable_parameters",
    "get_peft_model",
    "register_peft",
]
