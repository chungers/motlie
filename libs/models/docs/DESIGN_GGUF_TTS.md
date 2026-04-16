# Qwen3-TTS Backend Evaluation Design

## Status: Proposed

Tracking issue: `#188` "Qwen3-TTS backend evaluation — qwen3_tts_rs, F5-TTS ONNX, and GGUF paths"

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-15 | @codex-asr | Broadened the document from a pure GGUF question into a backend-evaluation note after validating `qwen3_tts_rs` end to end on real weights. Added the current backend priority ranking and recorded the Rust-native feasibility result, including the local source-build caveat. |
| 2026-04-15 | @codex-asr | Initial brownfield design for evaluating GGUF-based Qwen3-TTS without adding a fourth runtime engine. Focuses on whether `llama.cpp` can own the large causal LM stage, how ORT can own the smaller post-LM stages, and why the current public `qwen3-tts.cpp` ecosystem is not yet the right engine boundary for Motlie. |

This document is brownfield product work. It assumes Motlie keeps the existing `libs/model` / `libs/models` layering, the existing checkpoint/runtime separation from [DESIGN_CHECKPOINT_SEPARATION.md](../../model/docs/DESIGN_CHECKPOINT_SEPARATION.md), and the current engine set:

- `llama.cpp` for GGUF causal LM inference
- `whisper.cpp` for ASR
- ONNX Runtime for graph-defined speech models and small auxiliary networks

The specific question here is narrower than "can Motlie do TTS with Qwen3-TTS?":

Can Motlie reuse `llama.cpp` for the large text-to-speech-token stage of Qwen3-TTS so the system stays at three engines total, with ONNX Runtime handling only the smaller non-LLM speech stages?

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Research Summary](#research-summary)
- [Key Finding: Can `llama.cpp` Run Qwen3-TTS GGUF Today?](#key-finding-can-llamacpp-run-qwen3-tts-gguf-today)
- [Qwen3-TTS Pipeline Decomposition](#qwen3-tts-pipeline-decomposition)
- [Checkpoint and Engine Decoupling](#checkpoint-and-engine-decoupling)
- [Recommended Architecture for Motlie](#recommended-architecture-for-motlie)
- [Shared GGUF / GGML Substrate Design](#shared-gguf--ggml-substrate-design)
- [Curated Bundle Design Impact](#curated-bundle-design-impact)
- [Backend Candidate Comparison](#backend-candidate-comparison)
- [Recommendation](#recommendation)
- [Risks and Open Questions](#risks-and-open-questions)
- [References](#references)

---

## Overview

The earlier Qwen3-TTS ONNX path proved that Motlie can integrate a multi-stage TTS family, but it also exposed a hard limitation: the public ONNX export story is adapter-heavy and brittle. At the same time, multiple community GGUF conversions of Qwen3-TTS now exist on Hugging Face, and Motlie already has working `llama.cpp` bindings plus existing GGUF artifact resolution patterns.

That makes a `GGUF + llama.cpp + ORT` architecture attractive for two reasons:

1. It reuses engines Motlie already carries.
2. It matches the checkpoint/runtime separation Motlie is already moving toward.

The design question is not whether GGUF exists. GGUF clearly exists for Qwen3-TTS. The question is whether the public GGUFs are compatible with stock `llama.cpp`, or whether they still require a specialized engine.

The answer from current upstream evidence is:

- not directly, not today, for the public Qwen3-TTS GGUF artifacts
- but a `llama.cpp`-first Motlie architecture is still viable if Motlie deliberately exports or isolates the causal talker stage rather than treating the community full-pipeline GGUFs as drop-in `llama.cpp` checkpoints

Since the initial draft, one additional path moved from research to feasibility evidence: `qwen3_tts_rs` now has a verified end-to-end local run on this host with real Qwen weights. That changes the short-term evaluation order even though it does not change the long-term checkpoint/runtime decoupling goal.

## Goals and Non-Goals

### Goals

- Keep Motlie at three local engines:
  - `llama.cpp`
  - `whisper.cpp`
  - ONNX Runtime
- Reuse the existing checkpoint/runtime separation already implemented in `libs/model`
- Treat checkpoint format and runtime engine as orthogonal concerns
- Determine whether Qwen3-TTS can be decomposed into:
  - a large GGUF causal stage for `llama.cpp`
  - smaller ONNX stages for ORT
- Design a shared GGUF / GGML substrate that can serve:
  - `llama.cpp`
  - `whisper.cpp`
  - any future GGUF-based speech runtime Motlie may temporarily need to evaluate

### Non-Goals

- Implementing Qwen3-TTS in this PR
- Adding `qwen3-tts.cpp` as a new supported Motlie engine
- Designing browser or mobile TTS deployment
- Locking Motlie into the current public community GGUF packaging if a cleaner export path is needed
- Defining a generic user-supplied arbitrary TTS checkpoint loader

## Research Summary

### Current Priority Ranking

As of 2026-04-15, the backend evaluation order is:

1. `qwen3_tts_rs`
   Top priority because it is real, Rust-native, Apache-2.0 on crates.io, and already demonstrated speech generation from real Qwen3-TTS weights on this host without any C++ FFI.
2. `F5-TTS via ONNX`
   Next-best option because it reuses Motlie's existing ORT substrate and appears to have a healthier export story than the failed Qwen3-TTS ONNX path.
3. `qwen3-tts.cpp`
   Technically viable and already proven to produce speech locally, but still a C++ FFI path with a weaker license and maintenance posture than the Rust-native alternative.
4. `fish-speech.rs`
   Still technically credible, but the default model-family and checkpoint-license story remain mismatched with Motlie's preferred direction.
5. `index-tts-rust`
   Kept as a low-priority placeholder only; no convincing mature Rust implementation surfaced in this research pass.

### `qwen3_tts_rs`

`second-state/qwen3_tts_rs` is now a confirmed candidate rather than a hypothetical one.

Signals checked on 2026-04-15:

- crate: `qwen3-tts-rs = 0.2.2`
- repository: `second-state/qwen3_tts_rs`
- crates.io license: Apache-2.0
- GitHub stars: `202`
- GitHub forks: `30`
- last push timestamp: `2026-03-29T08:13:51Z`
- default backend: `tch` / libtorch
- alternate backend: MLX on Apple Silicon

What the project claims upstream:

- CLI for named-voice TTS
- voice cloning
- OpenAI-compatible API server
- streaming PCM output mode

Local feasibility result on this host:

- cloned repository revision: `4f98eb7`
- real model used: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- generated `output.wav` successfully from the prompt `Hello, this is a test of Qwen3 text to speech.`
- output properties:
  - `24 kHz`
  - mono
  - 16-bit PCM
  - `4.24 s`
- runtime from the official ARM64 release binary on this host:
  - wall time: `40.69 s`
  - peak RSS: `6.85 GB`

Important caveat:

- the local source build did not complete on this host because `audiopus_sys` attempted to autogen bundled `libopus` and required missing system `libtool` tooling
- the runtime feasibility result is therefore based on the project's official ARM64 release binary plus real downloaded weights, not on a locally compiled binary

Interpretation:

- `qwen3_tts_rs` is already strong enough to justify becoming the top near-term evaluation target
- it does not satisfy the original "reuse only `llama.cpp` + ORT" constraint, because it is a separate `tch` / libtorch runtime family
- it is still a better immediate prototype target than `qwen3-tts.cpp` because it avoids C++ FFI and has a clearer packaging and license story

### Upstream Qwen3-TTS

The official `QwenLM/Qwen3-TTS` repository is a Python-first runtime and model family. The repository currently advertises:

- streaming speech generation
- voice design
- voice cloning from short references
- multilingual support
- official examples centered on Transformers / PyTorch and CUDA-first execution

The official model internals are important for Motlie because they show the real stage boundaries:

- the top-level model type is `qwen3_tts`
- the large text-conditioned speech model is a custom `qwen3_tts_talker`
- that talker is paired with a smaller `qwen3_tts_talker_code_predictor`
- final waveform generation is performed by a speech-tokenizer decoder, not by a generic standalone HiFi-GAN-only path

The public code shows the concrete token structure for the 12 Hz family:

- text vocabulary: `151936`
- talker codec vocabulary: `3072`
- code predictor vocabulary: `2048`
- number of code groups / codebooks: `16`
- special codec IDs such as BOS / EOS / pad / language IDs are explicit in config

Inference from the official code:

- Qwen3-TTS is not just a standard text LM followed by a generic vocoder
- the causal stage emits multi-codebook acoustic tokens, with codebook 0 produced by the talker and codebooks 1-15 produced by a smaller autoregressive code predictor
- the post-token stage is more accurately "speech-tokenizer decode to PCM" than "mel + generic vocoder"

### Community GGUF Options

As of 2026-04-15, the public GGUF ecosystem includes at least these Qwen3-TTS repositories:

| Repo | Current public packaging signal | Notes |
|------|---------------------------------|-------|
| `Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF` | includes `qwen3-tts-0.6b-*.gguf` and `qwen3-tts-tokenizer-*.gguf`; HF API reported `585` downloads | explicitly branded for `qwen3tts.cpp`, not for stock `llama.cpp` |
| `mradermacher/Qwen3-1.7B-Multilingual-TTS-GGUF` | multiple quantizations from `Q2_K` through `Q8_0`; HF API reported `3980` downloads | the strongest public popularity signal among the checked GGUF repos |
| `offbeatengineer/qwen3-tts-gguf` | F16 files for `0.6B`, `1.7B`, and specialized variants plus tokenizer assets; HF API reported `288` downloads | useful evidence that the artifact family is real, but not evidence of stock `llama.cpp` compatibility |
| `OpenVoiceOS/qwen3-tts-0.6b-f16` | paired TTS and tokenizer GGUFs; HF API reported `115` downloads | used by the Python wrapper around `qwen3-tts.cpp` |

Packaging conclusion:

- the community has converged on a two-artifact GGUF layout:
  - main TTS model GGUF
  - tokenizer / decoder GGUF
- that is already enough to justify Motlie treating Qwen3-TTS as a multi-checkpoint runtime family rather than a single-file bundle

### `qwen3-tts.cpp`

The most relevant specialized runtime is `predict-woo/qwen3-tts.cpp`.

Current public GitHub signals checked on 2026-04-15:

- stars: `152`
- forks: `46`
- created: `2026-02-06`
- last GitHub push timestamp: `2026-03-09T05:56:19Z`
- commit count on default branch: `29`
- contributor count from GitHub API: `5`

What it is:

- a specialized GGML runtime for Qwen3-TTS
- not a wrapper around stock `llama.cpp`
- not an ONNX engine
- not the same project as `mmwillet/TTS.cpp`

What the code shows:

- it defines a custom GGUF architecture named `qwen3-tts`
- it reimplements:
  - text tokenizer
  - speaker encoder
  - talker transformer
  - code predictor
  - audio tokenizer decoder
- it uses GGML directly and vendors GGML as a submodule
- it exposes a real `extern "C"` C API in `src/qwen3tts_c_api.h`

What that means for Motlie:

- this is evidence that GGUF-based Qwen3-TTS is feasible
- this is not evidence that stock `llama.cpp` can load the same checkpoints today
- this is a separate runtime family with its own architecture and metadata expectations

Licensing concern:

- unlike the official Qwen code repo and `fish-speech.rs`, `predict-woo/qwen3-tts.cpp` did not present a detected repository license via GitHub API or a clear local `LICENSE` file in the inspected snapshot
- until that changes, it is not a good Motlie dependency even if the technical path is interesting

### `fish-speech.rs`

`EndlessReform/fish-speech.rs` remains relevant as a comparison point because it is a Rust-native speech runtime.

Current public GitHub signals checked on 2026-04-15:

- stars: `110`
- commit count on default branch: `108`
- contributor count: `1`
- last GitHub push timestamp: `2025-06-05T02:28:42Z`
- repository license: Apache-2.0

However, the runtime-license story and the weight-license story diverge:

- code: Apache-2.0
- common Fish Speech model cards checked on Hugging Face:
  - `fishaudio/fish-speech-1.5`
  - `fishaudio/fish-speech-1.4`
  - reported `license = cc-by-nc-sa-4.0`

That makes Fish Speech an awkward Motlie default if commercial-friendly curated bundles are a requirement.

## Key Finding: Can `llama.cpp` Run Qwen3-TTS GGUF Today?

### Short Answer

Not directly for the currently available public Qwen3-TTS GGUF artifacts.

### Why

The strongest direct evidence comes from the working community runtime itself:

- the public conversion tooling writes GGUF with `arch = "qwen3-tts"`
- the runtime expects custom metadata such as:
  - `qwen3-tts.text_vocab_size`
  - `qwen3-tts.num_code_groups`
  - `qwen3-tts.code_predictor.*`
  - `qwen3-tts.codec.*`
- the runtime contains custom execution logic for:
  - talker prefill layout
  - codebook-0 generation
  - delayed autoregressive generation for codebooks 1-15
  - tokenizer decoder / vocoder steps

This is not the shape of a stock `llama.cpp` text-generation load path.

### Precise Conclusion

The current public GGUF checkpoints are better described as:

- "Qwen3-TTS GGUF for a specialized GGML runtime"

not:

- "standard Qwen3 GGUF that `llama.cpp` can already run unchanged"

### What Still Makes a `llama.cpp` Path Viable

A `llama.cpp` path is still viable if Motlie does one of these on purpose:

1. Export only the large causal talker stage into a `llama.cpp`-compatible GGUF layout and run the smaller code-predictor / decoder stages elsewhere.
2. Extend upstream `llama.cpp` with support for the `qwen3_tts_talker` architecture and keep the rest of the pipeline in ORT.

The first option fits Motlie best because it minimizes engine count without forcing Motlie to own an upstream `llama.cpp` model-architecture fork immediately.

## Qwen3-TTS Pipeline Decomposition

### Actual Stage Boundaries

The official 12 Hz family maps more accurately to this pipeline:

1. Text tokens + role / TTS special tokens
   Input uses Qwen-family text tokenization plus TTS-specific special IDs.
2. Talker LM
   Large causal model conditioned on text and optional speaker / prompt signals.
3. Code predictor
   Small autoregressive model that fills the remaining acoustic codebooks per frame.
4. Speech-tokenizer decoder
   Converts the multi-codebook acoustic representation into PCM.
5. Optional speaker encoder / prompt encoder
   Used for cloning / conditioning.

### Tokens and Sampling

Relevant config and generation details observed in upstream code and model inspection:

- text vocabulary: `151936`
- talker hidden size: `1024`
- talker layers: `28`
- talker attention heads: `16`
- talker KV heads: `8`
- talker codec vocab: `3072`
- code predictor layers: `5`
- code predictor vocab: `2048`
- code groups: `16`
- default generation knobs at wrapper level:
  - `top_k = 50`
  - `top_p = 1.0`
  - `temperature = 0.9`
  - separate "subtalker" sampling knobs for the smaller code predictor path

Special IDs from inspected config include:

- `im_start_token_id = 151644`
- `im_end_token_id = 151645`
- `tts_pad_token_id = 151671`
- `tts_bos_token_id = 151672`
- `tts_eos_token_id = 151673`
- `codec_pad_id = 2148`
- `codec_bos_id = 2149`
- `codec_eos_token_id = 2150`

The language IDs are also embedded into the acoustic-token generation prompt. That matters because a Motlie `SpeechRequest` has to preserve language selection all the way into the talker prefill builder.

### Implication for ORT

If Motlie keeps the three-engine target, ONNX Runtime should not be framed as only "generic vocoder support."

For Qwen3-TTS specifically, ORT is the likely home for:

- code predictor, if Motlie does not teach `llama.cpp` that stage
- speech-tokenizer decoder
- speaker encoder / prompt encoder, when cloning is enabled

That is still consistent with the engine-minimization goal because ORT is already a Motlie runtime.

## Checkpoint and Engine Decoupling

Motlie already has the right core abstraction in `libs/model`:

- `ModelIdentity`
- `ModelCheckpoint`
- `BackendAdapter`
- `CheckpointFormat::{Gguf,Ggml,Onnx,Safetensors}`

The new requirement from Qwen3-TTS is not a new philosophical split. It is a more demanding case of the same split:

- one logical model family may need multiple checkpoint artifacts
- one runtime adapter may orchestrate more than one engine internally

### Proposed Refinement

For hybrid speech bundles, keep the existing public contract but allow a bundle spec to declare an engine-specific asset set:

```rust
pub struct RuntimeArtifactSet {
    pub gguf: Vec<ModelCheckpoint>,
    pub onnx: Vec<ModelCheckpoint>,
    pub auxiliary: Vec<ModelCheckpoint>,
}
```

This does not need to become a new `libs/model` public type immediately. It can remain a backend-local or bundle-local specification pattern first.

What matters is the design rule:

- checkpoint format stays orthogonal to engine
- the curated bundle chooses which checkpoints are loaded by which engine

### Example Mapping for a Future Qwen3-TTS Hybrid Bundle

| Logical stage | Checkpoint format | Engine |
|---------------|-------------------|--------|
| talker LM | GGUF | `llama.cpp` |
| code predictor | ONNX | ORT |
| speech-tokenizer decoder | ONNX | ORT |
| speaker encoder | ONNX | ORT |
| ASR sibling models | GGML | `whisper.cpp` |

This is the cleanest way to preserve the "three engines" rule while acknowledging that a single speech bundle may be multi-artifact and multi-runtime internally.

## Recommended Architecture for Motlie

### Recommendation

Motlie should pursue a `llama.cpp + ORT` hybrid Qwen3-TTS backend, but only with a Motlie-controlled decomposition.

The recommended architecture is:

1. `llama.cpp` owns the largest causal talker stage.
2. ORT owns the smaller non-LLM stages:
   - code predictor
   - speech-tokenizer decoder
   - optional speaker encoder
3. `libs/model` exposes one `SpeechModel` capability over the combined pipeline.
4. `libs/models` exposes one curated bundle that hides the internal asset split.

### What Motlie Should Not Do

Motlie should not:

- treat public `qwen3-tts.cpp` GGUFs as drop-in `llama.cpp` checkpoints
- add `qwen3-tts.cpp` as a fourth production engine
- bind directly to `qwen3-tts.cpp` while its license status remains unclear
- assume the post-LM stage is a generic mel-vocoder pair when the actual upstream tokenizer decoder is more specific than that

### Two Viable Implementation Shapes

#### Shape A: Talker-only GGUF export

- export the official talker weights into a `llama.cpp`-compatible GGUF
- keep code predictor and decoder as ONNX
- best match for Motlie's long-term engine-minimization goal

Pros:

- best reuse of current Motlie runtime stack
- smallest new surface area in Rust
- keeps the big model on the mature `llama.cpp` path

Cons:

- requires custom export work
- may require Motlie-side prefill/token handling for TTS-specific IDs

#### Shape B: Minimal `llama.cpp` architecture extension

- add `qwen3_tts_talker` support upstream or in a maintained fork
- still keep code predictor / decoder in ORT

Pros:

- closer to the real upstream talker architecture
- avoids forcing the public talker into an overly generic text-only mold

Cons:

- this is a larger maintenance burden
- it starts to blur the line between "reuse `llama.cpp`" and "carry a custom speech architecture in `llama.cpp`"

## Shared GGUF / GGML Substrate Design

Motlie already has:

- `motlie-model-ort` as a small ORT helper layer
- backend-local GGUF / GGML resolution code in:
  - `libs/model/backends/llama_cpp/src/common.rs`
  - `libs/model/backends/whisper_cpp/src/common.rs`
- `libs/models::resolve_hf_gguf_snapshot(...)`

The next clean step is a shared GGUF / GGML substrate parallel to `motlie-model-ort`.

### Proposed Shared Responsibilities

A small common GGUF / GGML helper layer should own:

- local snapshot validation
- artifact selection by suffix / exact filename
- checkpoint metadata probing:
  - architecture
  - quantization label
  - file inventory
- normalized error reporting for missing or ambiguous local artifacts

### What It Should Not Own

It should not own:

- execution
- sampling
- backend-specific tokenization
- backend-specific scheduler or CUDA wiring

Those remain in:

- `motlie-model-llama-cpp`
- `motlie-model-whisper-cpp`
- any future hybrid speech backend

### Why This Matters

This gives Motlie one checkpoint store for GGUF / GGML-family artifacts even though different engines consume them differently.

Concrete answer to the checkpoint-store question:

- yes, Motlie can and should have one local artifact cache for:
  - `.gguf` files used by `llama.cpp`
  - `.gguf` files used by experimental TTS tooling
  - `.bin` / ggml files used by `whisper.cpp`
- no, that does not imply one file works in every engine
- the shared layer is storage and validation, not execution compatibility

## Curated Bundle Design Impact

For a future hybrid Qwen3-TTS bundle, `libs/models` should expose one curated selector even if the backend loads several assets.

Example shape:

- selector: `tts:qwen3/tts_12hz_0_6b_hybrid`
- user-facing family: still one `SpeechModel`
- internal assets:
  - one or more GGUF files
  - one or more ONNX files
  - tokenizer metadata and sidecars as required

This keeps the Motlie UX aligned with existing curated bundles:

- callers choose a logical model
- they do not manually compose engines
- feature flags gate the bundle and optional accelerators

## Backend Candidate Comparison

### Summary Table

| Option | Runtime count impact | Checkpoint fit | Technical maturity | Licensing posture | Current rank |
|--------|----------------------|----------------|--------------------|------------------|--------------|
| `qwen3_tts_rs` | adds a dedicated `tch` / libtorch runtime | direct fit for official Qwen3-TTS safetensors checkpoints | medium; real crate, active enough repo, successful local speech generation on real weights | strongest of the immediate alternatives; crate declares Apache-2.0 | `#1` near-term prototype candidate |
| `F5-TTS via ONNX` | reuses existing ORT engine | unrelated checkpoint family but fits Motlie's ORT substrate cleanly | medium from repo signals, but not yet locally validated in this design pass | promising if upstream ONNX artifacts and licenses hold up | `#2` |
| `qwen3-tts.cpp` | adds a 4th engine | strong for today's public community GGUFs | low-to-medium; young repo, 29 commits, 5 contributors, real C API but custom runtime | currently weak because inspected repo snapshot did not present a clear license | `#3` fallback evidence source, not preferred engine |
| `fish-speech.rs` | adds or substitutes a separate TTS runtime | unrelated checkpoint family | medium; older but still single-maintainer | code Apache-2.0, common weights non-commercial | `#4` |
| `index-tts-rust` | unknown | unknown | weak; no convincing maintained Rust implementation identified in this pass | unknown | `#5` watch-only |
| `llama.cpp` multi-stage hybrid | stays at 3 engines total | best long-term fit if Motlie controls talker export and lets ORT handle smaller stages | medium, because export / compatibility work is still needed | strongest long-term architecture if built around official Qwen code plus Motlie-owned export tooling | strategic architecture target, not the easiest next prototype |

### `qwen3_tts_rs`

Strengths:

- Rust-native implementation with no C++ FFI boundary in Motlie
- direct fit for official Qwen3-TTS checkpoint layout rather than community GGUF repackaging
- already demonstrated real speech generation on this host
- supports voice cloning and a streaming API mode upstream

Weaknesses:

- adds a fourth runtime family to Motlie unless it is treated as a temporary evaluation-only path
- depends on `tch` / libtorch rather than reusing Motlie's existing `llama.cpp` or ORT substrates
- source build environment still has native dependency friction on this host

Interpretation:

- best immediate way to learn the real Qwen3-TTS runtime behavior in Rust
- not the cleanest final architecture if Motlie wants to keep the engine set minimal

### `F5-TTS via ONNX`

Strengths:

- strongest reuse story with Motlie's current ORT substrate
- likely the simplest path to a curated TTS bundle if upstream artifacts are stable
- avoids adding a brand-new runtime boundary

Weaknesses:

- not the Qwen3-TTS family
- quality, streaming behavior, and cloning ergonomics still need local validation

Interpretation:

- best backup plan if Qwen3-specific runtime reuse becomes too expensive
- should remain ahead of any new C++ GGUF runtime adoption

### `llama.cpp` Multi-Stage Hybrid

Strengths:

- best alignment with Motlie's "few engines, many bundles" philosophy
- reuses existing `llama_cpp.rs` bindings and GGUF resolution patterns
- keeps the largest and most expensive stage on the engine Motlie already trusts for GGUF LMs
- lets ORT do what it already does well: run smaller structured graphs

Weaknesses:

- the public Qwen3-TTS GGUFs are not ready-made for stock `llama.cpp`
- Motlie needs export or compatibility work first
- the official post-LM stages are not as generic as "mel decoder + HiFi-GAN"

### `qwen3-tts.cpp`

Strengths:

- proves that a full Qwen3-TTS GGUF runtime is feasible
- uses GGUF rather than ONNX for the large model path
- exposes a real C API suitable for Rust FFI in principle

Weaknesses:

- custom GGML runtime, not `llama.cpp`
- custom architecture `qwen3-tts`
- unclear repository licensing in the inspected snapshot
- young ecosystem and limited public maintenance signal

Interpretation:

- good evidence source
- poor production dependency candidate

### `fish-speech.rs`

Strengths:

- Rust-native implementation
- cleaner language integration story than C++ FFI
- repository license is explicit

Weaknesses:

- different model family than Qwen3-TTS
- common official model weights are non-commercial
- does not help the `llama.cpp` reuse goal

Interpretation:

- if Motlie only cared about a Rust-native open-code runtime, `fish-speech.rs` would still be worth watching
- for this specific "reuse `llama.cpp` + ORT" goal, it is the wrong direction

## Recommendation

### Primary Recommendation

Motlie should update the evaluation order immediately:

1. prototype `qwen3_tts_rs` first
2. evaluate `F5-TTS` via ORT second
3. keep `qwen3-tts.cpp` only as a lower-priority fallback and evidence source

Motlie should still not replace the long-term architecture target with `qwen3-tts.cpp`.

Instead Motlie should pursue a hybrid backend design:

- `llama.cpp` for the large talker stage
- ORT for code predictor, tokenizer decoder, and optional speaker encoder

That leaves two truths that both need to be documented:

- near-term prototype priority: `qwen3_tts_rs`
- long-term engine-minimizing architecture target: `llama.cpp` + ORT

### Practical Answer to the Priority Question

Can Motlie load today's public Qwen3-TTS GGUF into stock `llama.cpp` and directly generate speech tokens?

Not as-is.

What Motlie can do next without adding a fourth engine is:

1. treat the current public GGUF ecosystem as proof that the talker weights can be represented in tensor-container form
2. define a Motlie-owned export path for the talker stage that targets `llama.cpp`
3. keep the smaller decoder stages in ORT

That remains the only path in the current evidence set that both:

- respects the user's engine-minimization constraint
- fits Motlie's existing architecture

## Risks and Open Questions

1. The talker export path may require non-trivial handling for Qwen3-TTS-specific prompt layout, special tokens, and multi-codebook semantics.
2. The official Hugging Face Qwen3-TTS model repos were not anonymously queryable through the HF API during this research pass, so exact official artifact inventory should be revalidated during implementation with authenticated access if needed.
3. The public community GGUF repositories are useful packaging references, but Motlie should avoid assuming they are semantically identical or stable enough for curated production use.
4. If Motlie later decides voice cloning is in-scope for the first hybrid slice, the speaker encoder should stay in ORT rather than forcing that path into `llama.cpp`.
5. A future upstream `llama.cpp` architecture addition could simplify this entire design, but the current design should not depend on that happening.

## References

- Official Qwen3-TTS repo: <https://github.com/QwenLM/Qwen3-TTS>
- Official Qwen3-TTS GitHub metadata checked 2026-04-15: <https://github.com/QwenLM/Qwen3-TTS>
- `predict-woo/qwen3-tts.cpp`: <https://github.com/predict-woo/qwen3-tts.cpp>
- `second-state/qwen3_tts_rs`: <https://github.com/second-state/qwen3_tts_rs>
- `femelo/py-qwen3-tts-cpp`: <https://github.com/femelo/py-qwen3-tts-cpp>
- `DakeQQ/F5-TTS-ONNX`: <https://github.com/DakeQQ/F5-TTS-ONNX>
- `EndlessReform/fish-speech.rs`: <https://github.com/EndlessReform/fish-speech.rs>
- `Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF`: <https://huggingface.co/Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF>
- `mradermacher/Qwen3-1.7B-Multilingual-TTS-GGUF`: <https://huggingface.co/mradermacher/Qwen3-1.7B-Multilingual-TTS-GGUF>
- `offbeatengineer/qwen3-tts-gguf`: <https://huggingface.co/offbeatengineer/qwen3-tts-gguf>
- `OpenVoiceOS/qwen3-tts-0.6b-f16`: <https://huggingface.co/OpenVoiceOS/qwen3-tts-0.6b-f16>
- Existing Motlie checkpoint/runtime separation design: [DESIGN_CHECKPOINT_SEPARATION.md](../../model/docs/DESIGN_CHECKPOINT_SEPARATION.md)
