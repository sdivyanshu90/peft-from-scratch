"""Method registry: the single mapping from a method name to its implementation.

Every PEFT method registers exactly one *model* class via the
:func:`register_peft` decorator. The decorator is the single source of truth for
a method's canonical name: it stamps that name onto the associated config class
(``config_cls.peft_type``), so config serialization and model lookup can never
drift apart.

Design (see ARCHITECTURE.md, Decision 3):
    * No global mutable state beyond this append-only registry, which is
      populated at import time and never mutated thereafter.
    * Lookup is by string name, enabling polymorphic ``from_pretrained`` (read
      ``peft_type`` from JSON -> resolve both config and model classes).

Example:
    >>> from peft_lib.core.registry import available_methods
    >>> "lora" in available_methods()
    True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from torch import nn

from peft_lib.core.exceptions import ConfigError

if TYPE_CHECKING:
    from peft_lib.core.base import PEFTConfig, PEFTModel

__all__ = [
    "RegistryEntry",
    "available_methods",
    "get_entry",
    "get_peft_model",
    "register_peft",
]

_M = TypeVar("_M", bound="type[PEFTModel]")


@dataclass(frozen=True)
class RegistryEntry:
    """An immutable record describing one registered PEFT method.

    Attributes:
        name: Canonical method name (e.g. ``"lora"``). Stable across versions;
            it is what gets written into ``adapter_config.json``.
        model_cls: The :class:`~peft_lib.core.base.PEFTModel` subclass.
        config_cls: The :class:`~peft_lib.core.base.PEFTConfig` subclass that
            ``model_cls`` consumes.
    """

    name: str
    model_cls: type[PEFTModel]
    config_cls: type[PEFTConfig]


_REGISTRY: dict[str, RegistryEntry] = {}


def register_peft(name: str) -> Callable[[_M], _M]:
    """Class decorator registering a :class:`PEFTModel` subclass under ``name``.

    The decorated class must declare a ``config_class`` attribute pointing at its
    :class:`PEFTConfig` subclass. The decorator stamps ``name`` onto that config
    class's ``peft_type`` so that the name lives in exactly one place.

    Args:
        name: Canonical, lowercase, hyphen/underscore-free method name.

    Returns:
        The decorator, which returns the class unchanged (so it composes with
        other decorators and preserves the type for ``mypy``).

    Raises:
        ConfigError: If ``name`` is already registered, or the decorated class
            lacks a valid ``config_class``.

    Example:
        >>> from peft_lib.core.base import PEFTConfig, PEFTModel
        >>> # Real registrations happen in peft_lib/methods/*.py at import time.
        >>> callable(register_peft("demo-method"))
        True
    """

    def decorator(model_cls: _M) -> _M:
        if name in _REGISTRY:
            raise ConfigError(
                f"PEFT method {name!r} is already registered to "
                f"{_REGISTRY[name].model_cls.__name__}."
            )
        config_cls = getattr(model_cls, "config_class", None)
        if config_cls is None:
            raise ConfigError(
                f"{model_cls.__name__} must set `config_class` before @register_peft."
            )
        # Stamp the canonical name onto the config class (single source of truth).
        config_cls.peft_type = name
        _REGISTRY[name] = RegistryEntry(name=name, model_cls=model_cls, config_cls=config_cls)
        return model_cls

    return decorator


def get_entry(name: str) -> RegistryEntry:
    """Return the :class:`RegistryEntry` for ``name``.

    Args:
        name: A registered method name.

    Returns:
        The matching entry.

    Raises:
        ConfigError: If ``name`` is unknown, listing the available methods.

    Example:
        >>> import peft_lib  # triggers registration of all built-in methods
        >>> get_entry("lora").config_cls.__name__
        'LoRAConfig'
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"Unknown PEFT method {name!r}. Available: {sorted(_REGISTRY)}."
        ) from None


def available_methods() -> list[str]:
    """Return the sorted list of registered method names.

    Returns:
        Method names, sorted for deterministic output.

    Example:
        >>> import peft_lib
        >>> methods = available_methods()
        >>> {"lora", "dora", "ia3"} <= set(methods)
        True
    """
    return sorted(_REGISTRY)


def get_peft_model(base_model: nn.Module, config: PEFTConfig) -> PEFTModel:
    """Wrap ``base_model`` with the PEFT method named by ``config.peft_type``.

    This is the library's primary entry point. It looks the method up by name and
    constructs the corresponding :class:`PEFTModel`, which freezes the backbone
    and injects (zero-initialised) adapters.

    Args:
        base_model: The backbone :class:`torch.nn.Module` to fine-tune.
        config: A concrete :class:`PEFTConfig` describing the method.

    Returns:
        A ready-to-train :class:`PEFTModel`.

    Raises:
        ConfigError: If ``config.peft_type`` is not registered.

    Example:
        >>> import torch.nn as nn
        >>> import peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> base = nn.Sequential(nn.Linear(8, 8))
        >>> peft = get_peft_model(base, LoRAConfig(r=4, target_modules=["0"]))
        >>> peft.get_nb_trainable_parameters()[0]  # 4*(8+8) = 64
        64
    """
    entry = get_entry(config.peft_type)
    return entry.model_cls(base_model, config)
