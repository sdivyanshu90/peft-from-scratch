"""Core layer: config/model base classes, registry, exceptions, and helpers.

Most users import from the top-level :mod:`peft_lib` package; this subpackage is
the stable, method-agnostic foundation that every method builds on.
"""

from __future__ import annotations

from peft_lib.core.base import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    InjectionPEFTModel,
    PEFTConfig,
    PEFTModel,
    infer_target_modules,
)
from peft_lib.core.exceptions import (
    ConfigError,
    DeviceError,
    MergeError,
    PEFTError,
    ShapeError,
)
from peft_lib.core.registry import (
    RegistryEntry,
    available_methods,
    get_entry,
    get_peft_model,
    register_peft,
)
from peft_lib.core.utils import (
    freeze_model,
    get_nb_trainable_parameters,
    get_submodule,
    human_readable,
    iter_target_modules,
    match_target,
    set_submodule,
)

__all__ = [
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_WEIGHTS_NAME",
    "ConfigError",
    "DeviceError",
    "InjectionPEFTModel",
    "MergeError",
    "PEFTConfig",
    "PEFTError",
    "PEFTModel",
    "RegistryEntry",
    "ShapeError",
    "available_methods",
    "freeze_model",
    "get_entry",
    "get_nb_trainable_parameters",
    "get_peft_model",
    "get_submodule",
    "human_readable",
    "infer_target_modules",
    "iter_target_modules",
    "match_target",
    "register_peft",
    "set_submodule",
]
