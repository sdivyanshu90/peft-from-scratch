# peft_lib

A standalone, **from-scratch** Parameter-Efficient Fine-Tuning (PEFT) library for
PyTorch — built for clarity, type-safety, and production trust. Every method ships
with a first-principles math derivation, Google-style docstrings with runnable
examples, and a value-asserting test suite (`mypy --strict`, `ruff`, ~96% line
coverage).

> Not a wrapper around 🤗 `peft` — a clean, independent reimplementation you can
> read end-to-end. It interoperates with `transformers` models but depends only on
> `torch` + `safetensors`.

## Methods

| Method | Idea | Mergeable | Module |
|---|---|---|---|
| **LoRA** (+ rsLoRA, LoRA+, tied) | low-rank weight delta `s·BA` | ✅ | `methods/lora.py` |
| **DoRA** | magnitude + low-rank direction | ✅ | `methods/dora.py` |
| **VeRA** | shared frozen random bases + tiny scaling vectors | ✅ | `methods/vera.py` |
| **IA³** | learned activation rescaling | ✅ | `methods/ia3.py` |
| **Adapters** (Houlsby/Pfeiffer) | bottleneck residual MLP | ❌ | `methods/adapters.py` |
| **Prefix Tuning** | per-layer learnable KV prefixes | ❌ | `methods/prefix_tuning.py` |
| **Prompt Tuning** | learnable soft prompt | ❌ | `methods/prompt_tuning.py` |
| **QLoRA** | LoRA over a 4-bit base | ❌ | `methods/qlora.py` |

See [`peft_lib/methods/README.md`](peft_lib/methods/README.md) for the conceptual
guide (intuition, comparisons, when-to-use, failure modes, hyperparameters) and
[`ARCHITECTURE.md`](ARCHITECTURE.md) for system design and the extension guide.

## Install

```bash
pip install -e ".[dev]"          # core + dev tooling
pip install -e ".[hf]"           # + transformers/datasets (tutorials, GLUE)
pip install -e ".[quant]"        # + bitsandbytes (QLoRA; needs CUDA)
pip install -e ".[all]"          # everything
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

## Quickstart

```python
import torch.nn as nn
from peft_lib import LoRAConfig, get_peft_model

base = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))

cfg  = LoRAConfig(r=8, alpha=16, target_modules=["0", "2"])
peft = get_peft_model(base, cfg)
peft.print_trainable_parameters()
# trainable params: 24,576 || all params: 1,205,760 || trainable%: 2.0382

# ... train only the adapters ...

peft.save_pretrained("my-adapter")           # adapter weights + config only
bare = peft.merge_and_unload()                # fold in for zero-overhead inference
```

With a HuggingFace model (targets inferred from `config.model_type`):

```python
from transformers import AutoModelForCausalLM
from peft_lib import LoRAConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
peft  = get_peft_model(model, LoRAConfig(r=8, alpha=16))   # -> adapts c_attn
```

### Training

```python
from peft_lib.training import PEFTTrainer, TrainerConfig, TrainableParamLogger

trainer = PEFTTrainer(
    peft,
    TrainerConfig(learning_rate=2e-4, num_epochs=3, scheduler="cosine", warmup_ratio=0.1),
    train_dataloader,
    eval_dataloader=eval_dataloader,
    callbacks=[TrainableParamLogger()],
)
trainer.train()
```

### Merging adapters

```python
from peft_lib.merging import weighted_merge_adapters, uniform_soup, ties_merge_into
```

## Development

```bash
ruff format peft_lib tests && ruff check peft_lib tests   # format + lint (0 warnings)
mypy peft_lib                                             # strict types
pytest -m "not gpu and not quant"                         # full suite (CPU)
pytest --doctest-modules peft_lib                         # runnable docstrings
```

Markers: `gpu` (CUDA), `quant` (bitsandbytes), `hf` (transformers), `slow`
(real-model regression). All gate cleanly when their dependency is absent.

## License

Apache-2.0.
