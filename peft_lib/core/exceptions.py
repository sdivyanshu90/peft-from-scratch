"""Custom exception hierarchy for :mod:`peft_lib`.

All errors raised intentionally by the library derive from :class:`PEFTError`,
so user code can catch the whole family with a single ``except PEFTError``.
The four concrete subclasses partition the failure space into orthogonal,
actionable categories:

``ConfigError``
    The user-supplied configuration is internally inconsistent or violates a
    documented constraint (raised eagerly in ``__post_init__`` validators).
``ShapeError``
    A tensor entering an adapter has a rank/size incompatible with the layer it
    wraps. Carries the expected and actual shapes for fast debugging.
``DeviceError``
    A device/dtype precondition failed (e.g. a CUDA-only kernel requested on
    CPU, or a quantization backend that is not installed).
``MergeError``
    A merge/unmerge operation cannot be performed losslessly (e.g. merging an
    already-merged adapter, or merging into a quantized base weight).

Example:
    >>> from peft_lib.core.exceptions import ConfigError, PEFTError
    >>> try:
    ...     raise ConfigError("rank must be positive")
    ... except PEFTError as exc:
    ...     print(type(exc).__name__, exc)
    ConfigError rank must be positive
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DeviceError",
    "MergeError",
    "PEFTError",
    "ShapeError",
]


class PEFTError(Exception):
    """Base class for every error raised by :mod:`peft_lib`.

    Catch this to handle any library-originated failure uniformly. Never raised
    directly; always one of the concrete subclasses.
    """


class ConfigError(PEFTError):
    """Raised when a :class:`~peft_lib.core.base.PEFTConfig` is invalid.

    Raised eagerly from dataclass ``__post_init__`` validators so that
    misconfiguration fails at construction time rather than deep inside a
    forward pass.
    """


class ShapeError(PEFTError):
    """Raised when tensor shapes are incompatible with an adapter layer.

    Args:
        message: Human-readable description of the mismatch.
        expected: The shape (or partial shape) that was required, if known.
        actual: The shape that was actually observed, if known.

    Attributes:
        expected: As passed in; ``None`` when not supplied.
        actual: As passed in; ``None`` when not supplied.

    Example:
        >>> raise ShapeError("bad input", expected=(8, 512, 768), actual=(8, 512, 64))
        Traceback (most recent call last):
        ...
        peft_lib.core.exceptions.ShapeError: bad input (expected (8, 512, 768), got (8, 512, 64))
    """

    def __init__(
        self,
        message: str,
        *,
        expected: tuple[int, ...] | None = None,
        actual: tuple[int, ...] | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        if expected is not None or actual is not None:
            message = f"{message} (expected {expected}, got {actual})"
        super().__init__(message)


class DeviceError(PEFTError):
    """Raised when a device or dtype precondition is not met.

    Typical causes: requesting a CUDA-only quantization kernel on CPU, or using
    a backend (e.g. ``bitsandbytes``) that is not installed in the environment.
    """


class MergeError(PEFTError):
    """Raised when an adapter cannot be merged into / unmerged from a base weight.

    Typical causes: merging an adapter that is already merged, unmerging one that
    was never merged, or attempting a lossless merge into a quantized weight.
    """
