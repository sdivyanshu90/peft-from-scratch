# PEFT methods — conceptual guide

This is the **Layer-1 explainer**: intuition, comparisons, and practical
heuristics. The full mathematical derivation for each method (problem setup →
objective → parameterization → forward → backward → parameter count → scaling →
connections) lives in the corresponding module's top docstring
(`lora.py`, `dora.py`, …). The exact, runnable API reference is in the docstrings
of every public class.

---

## The one-sentence intuition for each method

| Method | Analogy |
|---|---|
| **LoRA** | Instead of rewriting a 768×768 page of notes, scribble a thin two-step correction (`down` to a tiny summary, `up` back out) in the margin. |
| **rsLoRA** | The same margin note, but with a font size that stays readable as you add more lines (`α/√r` instead of `α/r`). |
| **LoRA+** | Give the "expand back out" pen (B) a faster ink than the "summarize" pen (A). |
| **DoRA** | Separate *how loud* a weight is (magnitude) from *which way it points* (direction); only re-aim the direction with LoRA, tune loudness directly. |
| **VeRA** | Everyone shares the same fixed pair of random dictionaries; each layer only learns two small volume knobs. |
| **IA³** | Don't add anything — just turn each feature's volume knob up or down. |
| **Adapters** | Insert a tiny "translate → think → translate back" booth after a sub-layer, wired so it does nothing until trained. |
| **Prefix Tuning** | Hand every attention layer a few learned "cheat-sheet" key/value cards to attend to. |
| **Prompt Tuning** | Whisper a few learned imaginary words at the start of the input; the frozen model takes it from there. |
| **QLoRA** | Compress the frozen book to 4-bit so it fits in your bag, then LoRA on top. |

---

## Comparison table

`D` = hidden dim, `L` = layers, `r` = rank, `b` = bottleneck dim, `P` = virtual tokens.

| Method | Trainable params / adapted unit | Mergeable (zero-overhead infer) | Adds inference latency unmerged | Extra frozen memory | Touches |
|---|---|---|---|---|---|
| LoRA | `r·(in+out)` | ✅ | ~`r·(in+out)/(in·out)` FLOPs | none | linear weights |
| rsLoRA | `r·(in+out)` | ✅ | same as LoRA | none | linear weights |
| DoRA | `r·(in+out) + out` | ✅ | LoRA + 1 norm | none | linear weights |
| VeRA | `r + out` (per layer) | ✅ | two small matmuls | frozen `A,B` per shape | linear weights |
| IA³ | `out` (or `in` for FF) | ✅ | one broadcast multiply | none | activations |
| Adapters | `2·D·b + b + D` | ❌ (non-linear) | a bottleneck MLP | none | sub-layer outputs |
| Prefix | `2·L·P·D` | ❌ | extends attention by `P` | none | KV cache |
| Prompt | `P·D` (depth-independent) | ❌ | extends sequence by `P` | none | input embeddings |
| QLoRA | `r·(in+out)` (trainable) | ❌ (4-bit base) | LoRA + dequant | **4× smaller** base | linear weights (4-bit) |

---

## When to use — and when not to

**LoRA** — the default. Reach for it first for almost any decoder/encoder
fine-tune. *Not* when you need to change a weight's *magnitude* a lot (DoRA) or
when even `r·(in+out)` is too many params at your scale (VeRA/IA³).

**rsLoRA** — when you want **large `r`** (≥ 64). Standard `α/r` over-shrinks the
update as `r` grows; `α/√r` keeps step sizes stable. No reason not to use it at
high rank.

**LoRA+** — free convergence speedup; set `lora_plus_lr_ratio≈16`. Harmless to
leave on. Skip only if you've hand-tuned a single global LR you trust.

**DoRA** — when LoRA underfits and you suspect magnitude shifts matter (often
helps at *low* rank). *Not* when memory/latency is razor-tight — it adds an
`out`-vector and a norm per layer.

**VeRA** — extreme parameter budgets (serving thousands of adapters, on-device).
Use **large `r`** (256–1024) since params don't grow with it. *Not* when you need
maximum quality from a single adapter — the frozen random bases cap expressivity.

**IA³** — the cheapest option; great for very large models where even LoRA's
params add up, and for tasks that are "reweight what you already know". *Not* for
tasks needing genuinely new directions in weight space.

**Adapters (Houlsby/Pfeiffer)** — when a *non-linear* correction helps and you
don't need merged (zero-overhead) inference. Pfeiffer (FF-only) is the efficient
default; Houlsby (attention + FF) for maximum capacity. *Not* for latency-critical
serving (cannot be folded into the base weights).

**Prefix Tuning** — generation tasks on smaller models where per-layer steering
helps. *Not* for very deep models on a tight budget (params scale with `L`).

**Prompt Tuning** — the leanest input-side method; shines as model scale grows
(Lester et al.). *Not* for small models or tasks needing strong per-layer control
(use Prefix).

**QLoRA** — when the frozen model won't fit in VRAM in fp16. Needs a CUDA GPU +
`bitsandbytes`. *Not* on CPU, and not when you need to merge losslessly
(dequantize first).

---

## Known failure modes & mitigations

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss doesn't move at all | `target_modules` matched nothing / wrong names | We raise `ConfigError`; print `model.named_modules()` and pick real suffixes. |
| Training diverges at high `r` | `α/r` scaling too large or too small | Enable `use_rslora`; or set `α = 2r`. |
| LoRA underfits a hard task | Rank too low / too few targets | Raise `r`, add `o_proj`/MLP projections, or switch to DoRA. |
| Merged model ≠ adapted model | Merging a non-foldable method | Prefix/Prompt/Adapters/QLoRA aren't foldable — we raise `MergeError`. |
| `merge_and_unload` then bad outputs | Merged twice, or unmerge without merge | Guarded with `MergeError`; merge once. |
| IA³ `unmerge` errors | A learned scale hit ~0 (non-invertible) | We raise `MergeError`; keep the wrapped module instead of unmerging. |
| Prefix forward errors on a new arch | Non-standard KV cache format | Prefix supports standard decoders (GPT-2 family); check `model.config` dims. |
| Soft prompt trains slowly on a small model | Random init | Use `prompt_init="vocab"`. |

---

## Key hyperparameters & sensitivity

| Hyperparameter | Typical | Sensitivity | Notes |
|---|---|---|---|
| LoRA `r` | 8–64 | medium | Diminishing returns past task's intrinsic rank; cost is linear in `r`. |
| LoRA `α` | `2r` | low–medium | Only the ratio `α/r` matters; conventionally `α = 2r`. |
| `dropout` | 0.0–0.1 | low | Applied to the input *before* A; small values regularise. |
| `target_modules` | attn `q,v` (min) → all-linear (max) | **high** | The single biggest quality lever. q,v is the classic minimal set. |
| `use_rslora` | off | low | Turn on for `r ≥ 64`. |
| `lora_plus_lr_ratio` | 16 | low | Speeds convergence; rarely hurts. |
| DoRA `r` | 4–16 | medium | Often matches higher-`r` LoRA at lower rank. |
| VeRA `r` | 256–1024 | low | Large is fine — params don't scale with it. |
| VeRA `d_init` | 0.1 | medium | Too large can destabilise early training. |
| IA³ targets | `k,v` + FF | medium | The FF term (`feedforward_modules`) often matters most. |
| Adapter `bottleneck_dim` | 16–64 | medium | The capacity/parameter knob. |
| Prefix/Prompt `P` | 10–30 | medium | More tokens = more capacity but longer effective sequence. |
