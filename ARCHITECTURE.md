# Architecture

`peft_lib` is a from-scratch Parameter-Efficient Fine-Tuning library. This
document describes how the modules compose, how tensors flow through an adapted
model, the non-obvious design decisions (with rejected alternatives), and how to
add a new method in under 50 lines.

> Notation used throughout: **B** = batch, **S** = sequence length,
> **D** = model dim, **H** = heads, **r** = rank, **α** = LoRA alpha,
> **s = α/r** = effective scale.

---

## 1. System diagram

```
                        ┌───────────────────────────────────────────┐
                        │                peft_lib                    │
                        └───────────────────────────────────────────┘

  user code
     │  cfg = LoRAConfig(r=8, target_modules=[...])
     │  peft = get_peft_model(base_model, cfg)
     ▼
┌──────────────────┐   peft_type    ┌──────────────────┐   builds   ┌────────────────────┐
│  core.registry   │ ─────────────▶ │  RegistryEntry   │ ─────────▶ │  core.base         │
│  @register_peft  │   "lora" ->    │ (model_cls,      │            │  PEFTModel (ABC)    │
│  get_peft_model  │   LoRAModel    │  config_cls)     │            │  PEFTConfig (ABC)   │
└──────────────────┘                └──────────────────┘            │  InjectionPEFTModel │
                                                                     └─────────┬──────────┘
                                                                               │ subclassed by
                ┌──────────────────────────────────────────────────────────────┤
                ▼                                ▼                               ▼
   ┌────────────────────────┐     ┌────────────────────────┐      ┌──────────────────────────┐
   │  methods (injection)   │     │  methods (injection)   │      │  methods (augmentation)  │
   │  lora / dora / vera /  │     │  adapters (bottleneck) │      │  prefix_tuning           │
   │  ia3  (replace Linear) │     │  (non-foldable)        │      │  prompt_tuning           │
   └───────────┬────────────┘     └───────────┬────────────┘      └─────────────┬────────────┘
               │                              │                                 │
               ▼                              ▼                                 ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  core.utils: freeze_model · iter_target_modules · set_submodule · get_nb_trainable...  │
   │  core.typing: MergeableLayer (Protocol) · StateDict · TargetSpec                       │
   │  core.exceptions: PEFTError → ConfigError · ShapeError · DeviceError · MergeError       │
   └──────────────────────────────────────────────────────────────────────────────────────┘

  training/                 merging/                    quantization/         benchmarks/
  ├─ PEFTTrainer            ├─ merge_and_unload          └─ bnb_utils          ├─ memory_profiler
  ├─ schedulers             ├─ weighted_merge_adapters      (4-bit / 8-bit)    └─ glue_eval
  └─ callbacks              ├─ model_soup (uniform/wtd)
                            └─ ties_merging
```

The dependency arrows point **inward**: methods depend on `core`; `training`,
`merging`, `quantization`, `benchmarks` depend on `core` (+ methods). `core`
depends on nothing in the package. There is no global mutable state except the
append-only method registry, populated once at import.

---

## 2. Two families of method

| Family | Mechanism | Foldable? | Members | Base class |
|---|---|---|---|---|
| **Module injection** | Replace target `nn.Linear`/`Conv1D` with an adapter wrapper | LoRA/DoRA/VeRA/IA³: ✅; Adapters: ❌ (non-linear) | LoRA, DoRA, VeRA, IA³, Adapters, QLoRA | `InjectionPEFTModel` |
| **Input augmentation** | Prepend learnable embeddings / KV prefixes; backbone untouched | ❌ (no weight delta) | Prefix Tuning, Prompt Tuning | `PEFTModel` |

`InjectionPEFTModel` implements the entire find-and-replace + generic merge once;
each injection method supplies only a `_create_adapter(name, base_layer)` hook.

---

## 3. Data flow (concrete shapes: B=2, S=512, D=768, r=8)

### 3a. LoRA forward (one adapted projection, D→D)

```
x  : (2, 512, 768)                              input activations
                       base path (frozen)        ┌── h_base = W0·x + b0  : (2, 512, 768)
x ──┬───────────────────────────────────────────┤
    │                  adapter path              │   x·Aᵀ : (2, 512, 8)     A : (8, 768)
    └── dropout(x) ──▶ (·)·Aᵀ ──▶ (·)·Bᵀ ──▶ ×s ─┤   (·)·Bᵀ: (2, 512, 768)  B : (768, 8)
                                                  └── h_lora = s·(x·Aᵀ)·Bᵀ : (2, 512, 768)
y = h_base + h_lora : (2, 512, 768)
```

Adapter parameters: `A (8×768) + B (768×8) = 12,288` vs the frozen `W0 (768×768)
= 589,824` — a **48× reduction** in trainable params for this layer.

### 3b. Full model lifecycle

```
get_peft_model(model, cfg)
   │  1. PEFTModel.__init__  → store base_model + validated config
   │  2. freeze_model(base)  → every backbone param.requires_grad = False
   │  3. _resolve_targets()  → iter_target_modules(base, cfg.target_modules)
   │  4. for each (name, layer): set_submodule(base, name, _create_adapter(...))
   ▼
training:  PEFTTrainer.train()  → only adapter params receive gradients
   ▼
save_pretrained(dir)
   │  adapter_config.json     (JSON-native config)
   │  adapter_model.safetensors  (ONLY requires_grad params; never base weights)
   ▼
from_pretrained(fresh_base, dir)
   │  read peft_type → registry → rebuild wrapper (zero-delta) → load adapter weights
   ▼
merge_and_unload()  → W0 += s·B·A per layer; return the bare backbone (zero overhead)
```

### 3c. Prefix Tuning (KV injection)

```
input_ids (2, 512) ──▶ frozen backbone
                              ▲
prefix_encoder(B=2) ──────────┘  past_key_values: L × (K, V),  each (2, H, P, D/H)
attention_mask (2, 512) ──▶ left-pad P ones ──▶ (2, P+512)
```

---

## 4. Design decisions (with rejected alternatives)

**D1 — Registry decorates the *model*, not the config.** `@register_peft("lora")`
sits on `LoRAModel` (which declares `config_class = LoRAConfig`) and stamps the
canonical name onto the config. *Rejected:* decorating the config — then the
config would need a forward reference to its (later-defined) model class, creating
an import-order hazard. Single source of truth for the name = the decorator arg.

**D2 — `peft_type` is a `ClassVar` set by the decorator.** *Rejected:* a hand-set
string on each config (drifts from the registered name) and an abstract property
(can't be a dataclass field). The empty default makes the abstract `PEFTConfig`
un-instantiable via its own validator.

**D3 — Configs hold only JSON-native fields.** A compute dtype is stored as
`"bfloat16"`, not `torch.bfloat16`. *Rejected:* storing live objects — breaks
`json.dumps`, makes checkpoints non-portable, and couples config to torch
internals. Serialization is therefore trivial and lossless.

**D4 — Checkpoints store only `requires_grad` params.** `adapter_state_dict()`
selects by `requires_grad`, which after construction is exactly the adapter set.
*Rejected:* name-prefix matching (`"lora_"`) — method-specific and fragile;
the gradient flag is method-agnostic and always correct. VeRA's frozen random
matrices are *regenerated from a seed*, never stored (the spec's "never store base
weights" taken to its logical end).

**D5 — A single `InjectionPEFTModel` base does find-and-replace + merge.** Methods
implement only `_create_adapter`. *Rejected:* per-method injection code —
duplicated, drift-prone. The generic `merge_and_unload` uses the
`MergeableLayer` protocol so non-foldable adapters (bottleneck) fail loudly with
`MergeError` instead of silently corrupting weights.

**D6 — Structural typing (`Protocol`) for "mergeable" and "additive".** A layer
is mergeable iff it *has* `merge`/`unmerge`/`merged`/`base_layer` — no nominal
base class required. *Rejected:* an ABC mixin — would force multiple inheritance
onto every adapter and couple unrelated layers. `@runtime_checkable` gives cheap
`isinstance` gating at merge time.

**D7 — `fan_in_fan_out` auto-detection for Conv1D.** GPT-2 stores its weight
transposed `(in, out)`. The forward is layout-agnostic (it only needs `A`/`B`
shapes), but `merge` transposes the delta when `fan_in_fan_out`. *Rejected:*
supporting only `nn.Linear` — would exclude the entire GPT-2 family.

**D8 — Zero-delta initialization is an invariant, enforced and tested.** `B = 0`
(LoRA/DoRA), `b = 0` (VeRA), `l = 1` (IA³), `W_up = 0` (adapters), `m = ‖W₀‖`
(DoRA). The adapted model is *bit-identical* to the base at step 0. *Rejected:*
random `B` init — injects untrained noise into every layer (an explicitly banned
mistake). The first-projection-starved-at-step-0 gradient behaviour is a tested
consequence, not a bug.

**D9 — DoRA detaches the column norm in backprop.** `‖V‖_c` is treated as a
constant during the backward pass (per the paper), removing a large
activation-memory term while leaving forward values exact. *Rejected:* full
differentiation — correct but materially heavier; the merge-equivalence test still
holds to `atol=1e-4`.

**D10 — Input vs output rescaling unifies IA³.** Keys/values rescale the *output*
(`l` length = out); the FF up-projection rescales the *input* (`l` length = in).
Both fold losslessly (`diag(l)·W₀` rows / `W₀·diag(l)` cols). *Rejected:* a single
placement — would not match the method's three documented rescale sites.

**D11 — VeRA regenerates frozen matrices from a seed per shape.** Trainable params
are just `d (r) + b (out)` regardless of where the frozen `A,B` live. *Rejected:*
physically sharing one buffer across layers — better memory but fragile under
`.to()`/DDP tensor replacement; this reference impl trades frozen-memory for
correctness and clarity, and documents the production optimization.

**D12 — The trainer pops `labels` and computes loss externally by default.** This
works for both bare `nn.Module` backbones and HuggingFace models without passing
`labels` into incompatible `forward` signatures. *Rejected:* always using the
model's internal `.loss` — bare modules don't have one; `causal_lm_loss` is
provided for next-token shifting when needed.

**D13 — QLoRA construction fails fast without a usable backend.** Building a
`QLoRAModel` quantizes the base via `bitsandbytes`; if it is absent, a clear
`DeviceError` is raised at construction, not deep in a forward pass. Merging into a
4-bit weight raises `MergeError` (dequantize first).

---

## 5. Extension guide — add a method in < 50 lines

A foldable injection method needs three pieces: a config, an adapter layer, and a
model. Example — a hypothetical "ScaleLoRA" that adds a learnable per-output gate:

```python
from dataclasses import dataclass
import torch
from torch import nn
from peft_lib.core.base import InjectionPEFTModel, PEFTConfig
from peft_lib.core.registry import register_peft
from peft_lib.core.typing import TargetSpec
from peft_lib.methods.lora import _infer_linear_dims

@dataclass(kw_only=True)
class ScaleConfig(PEFTConfig):
    target_modules: TargetSpec | None = None
    def validate(self) -> None:
        super().validate()  # add field checks here

class ScaleLinear(nn.Module):                       # must expose base_layer/merged
    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer
        _, out, self.fan_in_fan_out = _infer_linear_dims(base_layer)
        self.gate = nn.Parameter(torch.ones(out))   # zero-delta: ones
        self.merged = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) * self.gate
    def merge(self) -> None: ...                     # fold gate into weight rows
    def unmerge(self) -> None: ...

@register_peft("scale_lora")
class ScaleModel(InjectionPEFTModel):
    config_class = ScaleConfig
    def _create_adapter(self, name: str, base_layer: nn.Module) -> nn.Module:
        return ScaleLinear(base_layer)
```

Import the module once (e.g. from `peft_lib/methods/__init__.py`) so the decorator
runs. `get_peft_model`, `save_pretrained`/`from_pretrained`, parameter counting,
and (if `merge`/`unmerge` are implemented) `merge_and_unload` all work for free.
Then add the 10-test contract entry in `tests/unit/test_injection_contract.py`.

---

## 6. Quality gates

| Gate | Tool | Contract |
|---|---|---|
| Format | `ruff format` (line-length 100) | clean |
| Lint | `ruff check` (E/F/I/N/UP/B/SIM/RUF/D/ANN/PT) | zero warnings |
| Types | `mypy --strict` | zero errors, ships `py.typed` |
| Tests | `pytest` (unit/integration/property/regression/benchmark) | green |
| Docs | Google-style docstrings + runnable doctests | `pytest --doctest-modules` green |

CUDA-only (`gpu`) and `bitsandbytes`-only (`quant`) tests skip cleanly when the
hardware/backend is absent.
