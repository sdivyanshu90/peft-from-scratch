"""Core abstractions: :class:`PEFTConfig`, :class:`PEFTModel`, :class:`InjectionPEFTModel`.

These three classes define the contract every method implements. Read this file
top to bottom for the full lifecycle of a PEFT run:

1.  The user builds a typed, self-validating :class:`PEFTConfig`.
2.  :func:`peft_lib.get_peft_model` constructs a :class:`PEFTModel`, which freezes
    the backbone and injects zero-initialised adapters (so the wrapped model is
    *numerically identical* to the base model until training begins).
3.  Training updates only the adapter parameters.
4.  ``save_pretrained`` writes *only* the adapter weights + the config JSON.
5.  ``from_pretrained`` reconstructs the wrapper and loads those weights.
6.  ``merge_and_unload`` folds foldable adapters back into the base weights for
    zero-overhead inference.

The module-injection family (LoRA, DoRA, VeRA, IA³, bottleneck adapters) shares
:class:`InjectionPEFTModel`, which implements find-and-replace and a generic
merge. Input-augmentation methods (Prefix, Prompt) subclass :class:`PEFTModel`
directly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from safetensors.torch import load_file, save_file
from torch import nn

from peft_lib.core.exceptions import ConfigError, MergeError
from peft_lib.core.typing import MergeableLayer, StateDict, TargetSpec
from peft_lib.core.utils import (
    freeze_model,
    get_nb_trainable_parameters,
    iter_target_modules,
    set_submodule,
)

__all__ = [
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_WEIGHTS_NAME",
    "InjectionPEFTModel",
    "PEFTConfig",
    "PEFTModel",
    "infer_target_modules",
]

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

_C = TypeVar("_C", bound="PEFTConfig")
_P = TypeVar("_P", bound="PEFTModel")

# Per-architecture defaults for `target_modules`, used when the user does not
# pass one explicitly. Keyed by HuggingFace `config.model_type`. Conservative
# choices that match the literature (query/value for BERT-family, etc.).
_DEFAULT_TARGETS: dict[str, list[str]] = {
    "gpt2": ["c_attn"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "distilbert": ["q_lin", "v_lin"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "t5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
}


def infer_target_modules(base_model: nn.Module) -> TargetSpec | None:
    """Guess sensible ``target_modules`` from a HuggingFace model's ``config.model_type``.

    Args:
        base_model: A backbone, ideally exposing ``base_model.config.model_type``.

    Returns:
        A list of target suffixes for known architectures, otherwise ``None``
        (the caller must then require an explicit ``target_modules``).

    Example:
        >>> class Cfg: model_type = "llama"
        >>> class M(nn.Module):
        ...     config = Cfg()
        >>> infer_target_modules(M())
        ['q_proj', 'v_proj']
    """
    model_cfg = getattr(base_model, "config", None)
    model_type = getattr(model_cfg, "model_type", None)
    if isinstance(model_type, str):
        return _DEFAULT_TARGETS.get(model_type)
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class PEFTConfig:
    """Abstract base for every method's typed configuration.

    Concrete configs are :func:`dataclasses.dataclass` subclasses that add their
    own hyperparameters and override :meth:`validate`. Two invariants make the
    config layer trustworthy:

    * **JSON-native fields only.** Every field must be a ``str``/``int``/``float``/
      ``bool``/``list``/``None`` so configs round-trip losslessly through JSON
      (a ``torch.dtype`` is stored as its string name, e.g. ``"bfloat16"``).
    * **Eager validation.** Constraints are checked in ``__post_init__`` so a bad
      config fails at construction, not mid-forward-pass.

    Attributes:
        peft_type: Canonical method name, stamped by :func:`~peft_lib.core.registry.register_peft`.
            Empty on this abstract base, which makes the base un-instantiable.
        base_model_name: Optional provenance string recorded in the checkpoint.

    Example:
        >>> import peft_lib
        >>> from peft_lib import LoRAConfig
        >>> cfg = LoRAConfig(r=8, alpha=16, target_modules=["q", "v"])
        >>> cfg.peft_type
        'lora'
        >>> LoRAConfig.from_dict(cfg.to_dict()).r
        8
    """

    peft_type: ClassVar[str] = ""

    base_model_name: str | None = None

    def __post_init__(self) -> None:
        """Run :meth:`validate` immediately after dataclass initialisation."""
        self.validate()

    def validate(self) -> None:
        """Validate field values; subclasses override and call ``super().validate()``.

        Raises:
            ConfigError: If invoked on the abstract base (no registered
                ``peft_type``), or — in subclasses — on any invalid field.
        """
        if not self.peft_type:
            raise ConfigError(
                "PEFTConfig is abstract. Instantiate a concrete, registered "
                "subclass (e.g. LoRAConfig)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-native dict, including ``peft_type``.

        Returns:
            A dict with every dataclass field plus ``"peft_type"``. Tuples are
            converted to lists for JSON fidelity.

        Example:
            >>> import peft_lib
            >>> from peft_lib import LoRAConfig
            >>> d = LoRAConfig(r=4, target_modules=["q"]).to_dict()
            >>> d["peft_type"], d["r"]
            ('lora', 4)
        """
        out: dict[str, Any] = {"peft_type": self.peft_type}
        for field in fields(self):
            value = getattr(self, field.name)
            out[field.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls: type[_C], data: Mapping[str, Any]) -> _C:
        """Reconstruct a config from a dict produced by :meth:`to_dict`.

        Args:
            data: A mapping of field names to values. A ``"peft_type"`` key is
                tolerated and ignored (the class already fixes the type).

        Returns:
            A validated config instance of ``cls``.

        Raises:
            ConfigError: If ``data`` contains keys that are not fields of ``cls``.

        Example:
            >>> import peft_lib
            >>> from peft_lib import LoRAConfig
            >>> LoRAConfig.from_dict({"r": 16, "peft_type": "lora"}).r
            16
        """
        payload = dict(data)
        payload.pop("peft_type", None)
        valid = {f.name for f in fields(cls)}
        unknown = set(payload) - valid
        if unknown:
            raise ConfigError(
                f"Unknown config keys for {cls.__name__}: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}."
            )
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        """Write the config to ``adapter_config.json``.

        Args:
            path: A directory (the file is created inside it) or an explicit
                ``.json`` path.

        Returns:
            The path actually written.

        Raises:
            ConfigError: If a field is not JSON-serialisable.

        Example:
            >>> import tempfile, peft_lib
            >>> from peft_lib import LoRAConfig
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = LoRAConfig(target_modules=["q"]).save(d)
            ...     p.name
            'adapter_config.json'
        """
        path = Path(path)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            path = path / ADAPTER_CONFIG_NAME
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        except TypeError as exc:
            raise ConfigError(f"Config is not JSON-serialisable: {exc}") from exc
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def load(cls: type[_C], path: str | Path) -> _C:
        """Load a config from ``adapter_config.json``.

        Args:
            path: A directory containing the file, or the file path itself.

        Returns:
            A validated config instance of ``cls``.

        Example:
            >>> import tempfile, peft_lib
            >>> from peft_lib import LoRAConfig
            >>> with tempfile.TemporaryDirectory() as d:
            ...     _ = LoRAConfig(r=32, target_modules=["q"]).save(d)
            ...     LoRAConfig.load(d).r
            32
        """
        path = Path(path)
        if path.is_dir():
            path = path / ADAPTER_CONFIG_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PEFTModel(nn.Module, ABC):
    """Abstract wrapper around a frozen backbone plus trainable adapters.

    Subclasses must set the ``config_class`` class attribute and implement
    :meth:`merge_and_unload`. The wrapper is itself an :class:`torch.nn.Module`,
    so it composes with ``.to()``, ``.train()``, autograd, DDP/FSDP, and
    ``torch.compile`` transparently.

    Attributes:
        config_class: The :class:`PEFTConfig` subclass this model consumes.
        base_model: The wrapped backbone (its non-adapter params are frozen).
        config: The validated config driving the adaptation.

    Example:
        >>> import torch.nn as nn, peft_lib
        >>> from peft_lib import LoRAConfig, get_peft_model
        >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
        >>> isinstance(peft, peft_lib.PEFTModel)
        True
    """

    config_class: ClassVar[type[PEFTConfig]]

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        super().__init__()
        if not isinstance(config, self.config_class):
            raise ConfigError(
                f"{type(self).__name__} expects a {self.config_class.__name__}, "
                f"got {type(config).__name__}."
            )
        self.base_model = base_model
        self.config = config

    # -- parameter accounting ------------------------------------------------
    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts (see core.utils).

        Returns:
            A ``(trainable, total)`` pair.

        Example:
            >>> import torch.nn as nn, peft_lib
            >>> from peft_lib import LoRAConfig, get_peft_model
            >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
            >>> peft.get_nb_trainable_parameters()[0]
            64
        """
        return get_nb_trainable_parameters(self)

    def print_trainable_parameters(self) -> None:
        """Print a one-line trainable/total/percentage summary.

        Example:
            >>> import torch.nn as nn, peft_lib
            >>> from peft_lib import LoRAConfig, get_peft_model
            >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
            >>> peft.print_trainable_parameters()  # doctest: +ELLIPSIS
            trainable params: 64 || all params: 136 || trainable%: 47.05...
        """
        trainable, total = self.get_nb_trainable_parameters()
        pct = 100.0 * trainable / total if total else 0.0
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    # -- forward / generate passthrough -------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped backbone's ``forward`` (adapters are inside it)."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to ``base_model.generate`` if the backbone supports generation.

        Raises:
            AttributeError: If the backbone has no ``generate`` method.
        """
        generate = getattr(self.base_model, "generate", None)
        if generate is None:
            raise AttributeError(f"{type(self.base_model).__name__} has no `generate`.")
        return generate(*args, **kwargs)

    # -- (de)serialisation ---------------------------------------------------
    def adapter_state_dict(self) -> StateDict:
        """Return *only* the trainable adapter tensors, detached and on CPU-ready form.

        Selection is by ``requires_grad``: after construction the backbone is
        frozen and every adapter parameter is trainable, so this captures exactly
        the adapter weights and nothing of the (large, redundant) base model.
        Methods that depend on a non-trainable but non-reconstructible buffer
        (none of the built-ins do) should override this.

        Returns:
            A name -> tensor mapping suitable for ``safetensors`` serialisation.

        Example:
            >>> import torch.nn as nn, peft_lib
            >>> from peft_lib import LoRAConfig, get_peft_model
            >>> base = nn.Sequential(nn.Linear(8, 8))
            >>> peft = get_peft_model(base, LoRAConfig(r=4, target_modules=["0"]))
            >>> sorted(k.split(".")[-1] for k in peft.adapter_state_dict())
            ['lora_A', 'lora_B']
        """
        return {
            name: param.detach() for name, param in self.named_parameters() if param.requires_grad
        }

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Save the config + adapter weights (never the base model) to ``save_dir``.

        Two files are written: :data:`ADAPTER_CONFIG_NAME` (JSON) and
        :data:`ADAPTER_WEIGHTS_NAME` (safetensors). The base weights are
        deliberately excluded — that is the whole point of PEFT checkpoints.

        Args:
            save_dir: Destination directory (created if absent).

        Example:
            >>> import tempfile, torch.nn as nn, peft_lib
            >>> from peft_lib import LoRAConfig, get_peft_model
            >>> peft = get_peft_model(nn.Linear(8, 8), LoRAConfig(r=4, target_modules=[""]))
            >>> with tempfile.TemporaryDirectory() as d:
            ...     peft.save_pretrained(d)
            ...     import os; sorted(os.listdir(d))
            ['adapter_config.json', 'adapter_model.safetensors']
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(save_dir)
        state = {name: tensor.contiguous() for name, tensor in self.adapter_state_dict().items()}
        save_file(
            state,
            str(save_dir / ADAPTER_WEIGHTS_NAME),
            metadata={"format": "pt", "peft_type": self.config.peft_type},
        )

    @classmethod
    def from_pretrained(
        cls: type[_P],
        base_model: nn.Module,
        load_dir: str | Path,
        **config_overrides: Any,
    ) -> PEFTModel:
        """Rebuild a PEFT model from a saved adapter directory.

        The method name (``peft_type``) is read from the saved config, so this
        works both polymorphically (``PEFTModel.from_pretrained(...)``) and from a
        concrete subclass (``LoRAModel.from_pretrained(...)``); in the latter case
        a mismatch raises.

        Args:
            base_model: A freshly loaded backbone matching the one used at save
                time (its weights are reused, never read from the checkpoint).
            load_dir: Directory containing the adapter config + weights.
            **config_overrides: Fields to override on the loaded config (e.g.
                changing ``dropout`` for further training).

        Returns:
            A :class:`PEFTModel` with adapter weights loaded.

        Raises:
            ConfigError: If the saved ``peft_type`` is unknown or mismatches a
                concrete ``cls``.
            MergeError: If the checkpoint's adapter keys do not match the
                reconstructed model exactly.

        Example:
            >>> import tempfile, torch, torch.nn as nn, peft_lib
            >>> from peft_lib import LoRAConfig, LoRAModel, get_peft_model
            >>> _ = torch.manual_seed(0)
            >>> base = nn.Linear(8, 8)
            >>> peft = get_peft_model(base, LoRAConfig(r=4, target_modules=[""]))
            >>> with tempfile.TemporaryDirectory() as d:
            ...     peft.save_pretrained(d)
            ...     reloaded = LoRAModel.from_pretrained(nn.Linear(8, 8), d)
            >>> isinstance(reloaded, LoRAModel)
            True
        """
        # Local import avoids a registry<->base import cycle at module load.
        from peft_lib.core.registry import get_entry

        load_dir = Path(load_dir)
        config_path = load_dir / ADAPTER_CONFIG_NAME
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        peft_type = raw.get("peft_type", "")
        entry = get_entry(peft_type)
        if cls is not PEFTModel and entry.model_cls is not cls:
            raise ConfigError(
                f"Checkpoint is for {peft_type!r} ({entry.model_cls.__name__}), "
                f"but from_pretrained was called on {cls.__name__}."
            )
        config = entry.config_cls.from_dict({**raw, **config_overrides})
        model = entry.model_cls(base_model, config)

        weights = load_file(str(load_dir / ADAPTER_WEIGHTS_NAME))
        expected = set(model.adapter_state_dict())
        got = set(weights)
        if expected != got:
            raise MergeError(
                "Adapter checkpoint does not match the reconstructed model. "
                f"Missing: {sorted(expected - got)}; unexpected: {sorted(got - expected)}."
            )
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise MergeError(f"Unexpected keys while loading adapter: {unexpected}.")
        return model

    # -- merging -------------------------------------------------------------
    @abstractmethod
    def merge_and_unload(self) -> nn.Module:
        """Fold adapters into the backbone and return the plain, unwrapped model.

        Foldable methods (LoRA, DoRA, VeRA, IA³) return a backbone whose forward
        is numerically equivalent (``atol=1e-5``) to the adapted model, with zero
        adapter overhead. Non-foldable methods (Prefix, Prompt, bottleneck
        adapters) raise :class:`~peft_lib.core.exceptions.MergeError`.

        Returns:
            The unwrapped backbone module.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Injection family
# ---------------------------------------------------------------------------
class InjectionPEFTModel(PEFTModel, ABC):
    """Base for methods that *replace* target submodules with adapter wrappers.

    Implements the find-and-replace injection and a generic, protocol-driven
    :meth:`merge_and_unload`. Subclasses implement a single hook,
    :meth:`_create_adapter`, deciding what wrapper to build for each matched
    submodule.

    Attributes:
        adapter_layers: Mapping of dotted module name -> the injected adapter.
    """

    def __init__(self, base_model: nn.Module, config: PEFTConfig) -> None:
        super().__init__(base_model, config)
        freeze_model(self.base_model)
        self.adapter_layers: dict[str, nn.Module] = {}
        self._inject()

    @abstractmethod
    def _create_adapter(self, module_name: str, base_layer: nn.Module) -> nn.Module:
        """Build the adapter wrapper for one matched submodule.

        Args:
            module_name: Dotted name of the matched submodule.
            base_layer: The original (frozen) submodule to wrap.

        Returns:
            A new :class:`torch.nn.Module` that will replace ``base_layer``. It
            must store the wrapped layer as ``self.base_layer`` so the generic
            merge logic can recover it.
        """
        raise NotImplementedError

    def _resolve_targets(self) -> list[tuple[str, nn.Module]]:
        """Resolve the concrete ``(name, module)`` targets for this run.

        Returns:
            The list of submodules to adapt.

        Raises:
            ConfigError: If no ``target_modules`` were given and none could be
                inferred from the model architecture.
        """
        targets: TargetSpec | None = getattr(self.config, "target_modules", None)
        if targets is None:
            targets = infer_target_modules(self.base_model)
        if targets is None:
            raise ConfigError(
                "`target_modules` is required: it could not be inferred from the "
                "model. Pass an explicit list, e.g. target_modules=['q_proj', 'v_proj']."
            )
        return iter_target_modules(self.base_model, targets)

    def _inject(self) -> None:
        """Replace every resolved target with its adapter wrapper, in place.

        The special name ``""`` means the backbone *is itself* the target layer
        (e.g. ``get_peft_model(nn.Linear(...), cfg)``); in that case the wrapper
        replaces :attr:`base_model` directly rather than a submodule of it.
        """
        for name, layer in self._resolve_targets():
            adapter = self._create_adapter(name, layer)
            if name == "":
                self.base_model = adapter
            else:
                set_submodule(self.base_model, name, adapter)
            self.adapter_layers[name] = adapter

    def merge_adapter(self) -> None:
        """Fold every adapter delta into its base weight, keeping the wrappers.

        Useful for a quick equivalence check without unloading. Idempotent guards
        live in each layer's ``merge``.

        Raises:
            MergeError: If any injected adapter is not foldable.
        """
        for name, adapter in self.adapter_layers.items():
            if not isinstance(adapter, MergeableLayer):
                raise MergeError(
                    f"Adapter at {name!r} ({type(adapter).__name__}) cannot be merged."
                )
            adapter.merge()

    def unmerge_adapter(self) -> None:
        """Undo :meth:`merge_adapter`, restoring pristine base weights + live adapters.

        Raises:
            MergeError: If any injected adapter is not foldable.
        """
        for name, adapter in self.adapter_layers.items():
            if not isinstance(adapter, MergeableLayer):
                raise MergeError(
                    f"Adapter at {name!r} ({type(adapter).__name__}) cannot be unmerged."
                )
            adapter.unmerge()

    def merge_and_unload(self) -> nn.Module:
        """Fold all adapters in and replace each wrapper with its updated base layer.

        Returns:
            The unwrapped backbone, ready for zero-overhead inference.

        Raises:
            MergeError: If any injected adapter is not foldable (e.g. a non-linear
                bottleneck adapter).
        """
        for name, adapter in self.adapter_layers.items():
            if not isinstance(adapter, MergeableLayer):
                raise MergeError(
                    f"Adapter at {name!r} ({type(adapter).__name__}) cannot be merged "
                    "into the base weights; deploy it as a wrapped module instead."
                )
            adapter.merge()
            base_layer = adapter.base_layer
            if name == "":
                self.base_model = base_layer
            else:
                set_submodule(self.base_model, name, base_layer)
        self.adapter_layers.clear()
        return self.base_model
