"""Stateless helpers shared by every PEFT method.

These functions never hold state and never mutate global configuration; they
operate purely on the :class:`torch.nn.Module` / config objects passed in. This
keeps the library's behaviour deterministic and ``torch.compile``-friendly.
"""

from __future__ import annotations

from torch import nn

from peft_lib.core.exceptions import ConfigError
from peft_lib.core.typing import TargetSpec, is_linear_like

__all__ = [
    "freeze_model",
    "get_nb_trainable_parameters",
    "get_submodule",
    "human_readable",
    "iter_target_modules",
    "match_target",
    "set_submodule",
]


def get_nb_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs. total parameters of a model.

    A parameter contributes to the *trainable* count iff ``requires_grad`` is
    ``True``. Quantized weights (e.g. ``bitsandbytes`` ``Params4bit``) report
    their *packed* element count via ``numel()``; to keep the "total" figure
    comparable to full fine-tuning we multiply 4-bit params back up by 2, since
    two 4-bit values are packed into each ``uint8`` element.

    Args:
        model: The (possibly PEFT-wrapped) module to inspect.

    Returns:
        A ``(trainable, total)`` pair of parameter counts.

    Example:
        >>> import torch.nn as nn
        >>> m = nn.Linear(8, 4)
        >>> m.bias.requires_grad_(False)  # doctest: +ELLIPSIS
        Parameter containing:...
        >>> get_nb_trainable_parameters(m)
        (32, 36)
    """
    trainable = 0
    total = 0
    for param in model.parameters():
        num = param.numel()
        # bitsandbytes packs two 4-bit values per byte; un-pack for a fair total.
        if num > 0 and param.__class__.__name__ == "Params4bit":
            num *= 2
        total += num
        if param.requires_grad:
            trainable += num
    return trainable, total


def freeze_model(model: nn.Module) -> None:
    """Set ``requires_grad = False`` on every parameter of ``model`` in-place.

    Used by every PEFT method as step one: freeze the entire backbone, then
    selectively re-enable only the newly injected adapter parameters.

    Args:
        model: The module to freeze. Mutated in place.

    Example:
        >>> import torch.nn as nn
        >>> m = nn.Linear(4, 4)
        >>> freeze_model(m)
        >>> any(p.requires_grad for p in m.parameters())
        False
    """
    for param in model.parameters():
        param.requires_grad_(False)


def get_submodule(root: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a dotted submodule path, e.g. ``"transformer.h.0.attn.c_attn"``.

    Args:
        root: The module to start from.
        dotted_name: Dot-separated attribute path. Empty string returns ``root``.

    Returns:
        The resolved submodule.

    Raises:
        AttributeError: If any path component does not exist.

    Example:
        >>> import torch.nn as nn
        >>> m = nn.Sequential(nn.Linear(2, 2))
        >>> get_submodule(m, "0").__class__.__name__
        'Linear'
    """
    if dotted_name == "":
        return root
    module: nn.Module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at ``dotted_name`` with ``new_module`` in-place.

    The standard way PEFT injects adapters: locate a target ``nn.Linear`` and
    swap it for an adapter that wraps it.

    Args:
        root: The module tree to mutate.
        dotted_name: Dot-separated path to the submodule to replace.
        new_module: The replacement module.

    Raises:
        ConfigError: If ``dotted_name`` is empty (cannot replace the root).
        AttributeError: If the parent path does not exist.

    Example:
        >>> import torch.nn as nn
        >>> m = nn.Sequential(nn.Linear(2, 2))
        >>> set_submodule(m, "0", nn.Linear(2, 3))
        >>> m[0].out_features
        3
    """
    if dotted_name == "":
        raise ConfigError("Cannot replace the root module (empty submodule name).")
    parent_name, _, child_name = dotted_name.rpartition(".")
    parent = get_submodule(root, parent_name)
    setattr(parent, child_name, new_module)


def match_target(module_name: str, target: TargetSpec) -> bool:
    """Decide whether a module (by its dotted name) should be adapted.

    Matching is by *suffix on the final dotted component*: ``"v_proj"`` matches
    ``"model.layers.3.self_attn.v_proj"`` but not ``"...v_proj_extra"``. The
    sentinel ``"all-linear"`` is handled by :func:`iter_target_modules`, not
    here, so this function focuses purely on explicit name lists.

    Args:
        module_name: Dotted name from ``model.named_modules()``.
        target: A list of suffixes, or a single suffix string.

    Returns:
        ``True`` if ``module_name``'s last component equals one of the targets.

    Example:
        >>> match_target("transformer.h.0.attn.c_attn", ["c_attn", "c_proj"])
        True
        >>> match_target("transformer.h.0.attn.c_proj_x", ["c_proj"])
        False
    """
    targets = [target] if isinstance(target, str) else target
    leaf = module_name.rpartition(".")[2]
    return leaf in targets


def iter_target_modules(
    model: nn.Module,
    target: TargetSpec,
    *,
    exclude: frozenset[str] = frozenset({"lm_head", "score", "classifier"}),
) -> list[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` pairs that a method should adapt.

    Two modes:

    * ``target == "all-linear"``: every linear-like leaf (``nn.Linear`` or HF
      ``Conv1D``) whose name leaf is not in ``exclude``. This mirrors the common
      "adapt everything but the output head" recipe.
    * Otherwise: every module whose name matches ``target`` per
      :func:`match_target` *and* is linear-like.

    Args:
        model: The backbone to scan.
        target: Target specification (see :data:`~peft_lib.core.typing.TargetSpec`).
        exclude: Name leaves to skip in ``"all-linear"`` mode (typically heads,
            which should keep full-rank task capacity).

    Returns:
        A list of ``(dotted_name, module)`` pairs, in ``named_modules`` order.

    Raises:
        ConfigError: If no module matches (almost always a typo in
            ``target_modules``), surfaced early rather than silently training
            zero adapters.

    Example:
        >>> import torch.nn as nn
        >>> m = nn.ModuleDict({"q": nn.Linear(4, 4), "ln": nn.LayerNorm(4)})
        >>> [n for n, _ in iter_target_modules(m, ["q"])]
        ['q']
    """
    matches: list[tuple[str, nn.Module]] = []
    all_linear = target == "all-linear"
    for name, module in model.named_modules():
        if not is_linear_like(module):
            continue
        leaf = name.rpartition(".")[2]
        if all_linear:
            if leaf not in exclude:
                matches.append((name, module))
        elif match_target(name, target):
            matches.append((name, module))
    if not matches:
        raise ConfigError(
            f"No linear-like modules matched target spec {target!r}. "
            "Check `target_modules` against the model's named_modules()."
        )
    return matches


def human_readable(count: int) -> str:
    """Format a parameter count compactly, e.g. ``294912 -> '294.9K'``.

    Args:
        count: A non-negative integer.

    Returns:
        A short human-readable string with a K/M/B suffix (or the raw integer
        below 1,000).

    Example:
        >>> human_readable(294_912)
        '294.9K'
        >>> human_readable(7_864_320)
        '7.9M'
    """
    if count < 1_000:
        return str(count)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if count >= divisor:
            return f"{count / divisor:.1f}{suffix}"
    return str(count)  # pragma: no cover - unreachable given the guard above
