# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-29

Initial release: a complete, from-scratch PEFT library.

### Added

- **Core** (`peft_lib.core`): `PEFTConfig` / `PEFTModel` / `InjectionPEFTModel`
  base classes, a name → implementation registry (`@register_peft`,
  `get_peft_model`), parameter-counting / freezing / module-surgery utilities, a
  `PEFTError` exception hierarchy (`ConfigError`, `ShapeError`, `DeviceError`,
  `MergeError`), and structural typing (`MergeableLayer` protocol).
- **Methods**:
  - LoRA with variants — standard, **rsLoRA** scaling, **LoRA+** optimizer
    grouping, and **tied-weight** LoRA. Supports `nn.Linear` and HF `Conv1D`.
  - **DoRA** (weight-decomposed: magnitude + low-rank direction).
  - **VeRA** (shared frozen random projections + per-layer scaling vectors).
  - **IA³** (output / input activation rescaling).
  - **Bottleneck adapters** (Houlsby + Pfeiffer placements).
  - **Prefix Tuning** (per-layer learnable KV prefixes, optional reparameterisation).
  - **Prompt Tuning** (soft prompt, random or vocab init).
  - **QLoRA** (LoRA over a `bitsandbytes` 4-bit/8-bit base, graceful fallback).
- **Training** (`peft_lib.training`): `PEFTTrainer` (gradient accumulation,
  clipping, eval, callbacks), warmup-cosine / linear-decay / constant schedulers,
  and `TrainableParamLogger` / `CheckpointSaver` / `EarlyStopping` callbacks.
- **Merging** (`peft_lib.merging`): `merge_and_unload`, multi-adapter
  `weighted_merge_adapters`, uniform/weighted **model soups**, and **TIES**
  (trim → elect-sign → disjoint-merge).
- **Quantization** (`peft_lib.quantization`): `bitsandbytes` 4-bit/8-bit helpers
  with availability guards.
- **Benchmarks** (`peft_lib.benchmarks`): analytical + empirical memory/latency
  profiling and a NumPy-based GLUE metric/eval harness.
- **Every public class/function**: type-annotated, Google-style docstrings with
  runnable examples; the package ships `py.typed`.
- **Docs**: `ARCHITECTURE.md` (system diagram, data-flow shapes, 13 design
  decisions, extension guide) and a per-method conceptual guide.
- **Tests**: unit (the 10-test contract per method), integration (trainer +
  merging + multi-method), property (Hypothesis), regression (exact parameter
  counts incl. BERT-base LoRA = 294,912), and benchmark-budget tests. ~96% line
  coverage; `mypy --strict` and `ruff` clean.
- **CI**: GitHub Actions matrix over Python 3.11/3.12 × Ubuntu/macOS.

### Notes

- Parameter-count anchors are asserted against *derived, verified* values. Two
  figures from the original spec assume non-standard configurations and are
  documented in `tests/regression/test_parameter_counts.py`: Prefix l=10 on GPT-2
  is the standard `2·L·l·H = 184,320` (not 7,864,320), and IA³ on T5-small is
  `24,576` (not 28,672). BERT-base LoRA matches the spec exactly at 294,912.
