# TTS Model Support Design

## Status: Implemented (Phase 1 Piper Slice)

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-15 | @codex-tts | Added a reproducible Qwen3-TTS ONNX export runbook and checked in the export script under `libs/models/scripts/`. Documented the current adapter-graph workaround and the gap between the official Qwen runtime and the current Rust backend contract. |
| 2026-04-15 | @cld-review-models | Implemented Phase 2 Qwen3-TTS vertical slice. Runtime boundary: in-process ONNX via `motlie-model-ort` with pre-exported model components (encoder, decoder, vocoder). Upstream safetensors must be exported to ONNX offline before the curated bundle can start. Added vocabulary-based tokenizer, proper Hann-windowed DFT mel spectrogram for reference-audio conditioning, and shape-preserving tensor pipeline between ONNX stages. |
| 2026-04-14 | @codex-tts | Addressed PR #179 review R1 by fixing the local-only startup path to reuse the same Piper artifact validation as curated checkpoints, documenting the batch-then-chunk `open_stream()` behavior and Piper's runtime rejection of `SpeechParams.seed`, and recording the current `ort` RC dependency/runtime constraint explicitly. |
| 2026-04-14 | @codex-tts | Implemented the Phase 1 Piper slice with additive `SpeechModel` / `SpeechStream` contracts, a shared `motlie-model-ort` ONNX helper reused by the sherpa backend, a `motlie-model-piper` backend using eSpeak-ng phonemization plus Piper sidecar parsing, the curated `piper_en_us_ljspeech_medium` bundle, and the `models_tts_v0_1` example/validation path. |
| 2026-04-14 | @codex-tts | Addressed R1 review by defining the Qwen3-TTS phase-2 slice, adding the shared ONNX Runtime refactor target with the ASR sherpa-onnx path, removing the duplicate speaker-selection knob from `SpeechParams`, tightening `SpeechStream` state semantics, and switching the planned example naming to a capability-specific path that does not collide with ASR `v0.5`. |
| 2026-04-13 | @codex-tts | Initial brownfield design for a text-to-speech vertical slice in the Motlie model stack. Evaluates local-first TTS candidates, recommends a Piper ONNX bundle as the first implementation, and defines a streamed PCM output contract that mirrors the ASR PCM input shape. |

This document defines the design for adding text-to-speech (TTS) support to the existing `libs/model` and `libs/models` architecture.

Assumption for this draft: this is brownfield product work. Motlie already has stable model contracts, curated bundle registration, feature-gated bundle selection, and artifact download infrastructure. The TTS work should extend those seams additively. If the product context is later confirmed as greenfield, the migration section can be reduced, but the recommended architecture is unchanged.

The design is intentionally narrow. It focuses on one end-to-end vertical slice from curated artifact download to streamed PCM output, with adapters above the model layer for `.wav`, local playback, and telephony/websocket transports.

The roadmap beyond that first slice is also explicit: if the Piper slice proves out, the next end-to-end family to add should be Qwen3-TTS rather than leaving the phase-2 story implicit.

The exact re-export flow for the current Qwen3-TTS ONNX artifacts lives in
`docs/EXPORT_QWEN3_TTS_ONNX.md`. That runbook is the source of truth for the
Python environment, Hugging Face download path, emitted artifact sizes, and the
current adapter-graph workaround used to satisfy the Rust backend contract.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Research Summary](#research-summary)
- [Recommended Vertical Slice](#recommended-vertical-slice)
- [Architecture](#architecture)
- [Streaming PCM API Contract](#streaming-pcm-api-contract)
- [Core Contract Changes in `libs/model`](#core-contract-changes-in-libsmodel)
- [Generic Backend Design](#generic-backend-design)
- [Curated Bundle Design in `libs/models`](#curated-bundle-design-in-libsmodels)
- [Feature Flag Design](#feature-flag-design)
- [Output Adapter Boundary](#output-adapter-boundary)
- [Distribution Considerations](#distribution-considerations)
- [Migration and Compatibility Strategy](#migration-and-compatibility-strategy)
- [API Sketch](#api-sketch)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [References](#references)

---

## Overview

### Problem Statement

Motlie currently has no first-class speech synthesis capability. The existing model stack already separates:

- stable capability contracts in `libs/model`
- generic runtime adapters in `libs/model/backends/*`
- curated bundle registration and artifact download in `libs/models`

TTS should follow the same architecture as ASR, but in reverse.

ASR normalizes many upstream inputs into streamed PCM chunks in. TTS should normalize many downstream consumers into streamed PCM chunks out.

The required TTS surface is:

- real-time local generation
- CPU-first execution on macOS and Linux
- optional CUDA acceleration where it is actually beneficial
- `.wav` file output
- streamed output suitable for immediate playback or telephony transport adapters
- future room for speaker selection and reference-audio conditioning without overfitting v1 to cloning-only models
- compatibility with the current curated bundle and feature-gated selector patterns

### Solution

Add a new speech synthesis capability to `libs/model`, implement a new TTS backend crate for the first runtime family, and expose one curated TTS bundle from `libs/models`.

Recommended first slice:

1. `libs/model`
   Add `SpeechModel` / `SpeechStream` traits and PCM-output request/response types.
2. `libs/model/backends/piper`
   Provide a generic backend adapter over Piper-format ONNX voices, implemented on top of ONNX Runtime.
3. `libs/models`
   Add a `tts/` namespace with one curated bundle file, one direct enum family, one selector family, and bundle-local artifact resolution/download rules.

This preserves the existing layering:

- `libs/model` owns the stable contract
- backend crates own runtime integration and model-family specifics
- `libs/models` owns curated identity, artifact download rules, selectors, and feature flags

## Goals and Non-Goals

### Goals

- Add a first-class TTS capability to the Motlie model contract
- Mirror the ASR architectural pattern instead of inventing a parallel subsystem
- Standardize on streamed PCM output so `.wav`, PulseAudio/ALSA, and telephony transports all consume the same contract
- Prefer CPU-viable defaults on macOS and Linux
- Keep CUDA optional and backend-local
- Ship one curated vertical slice end to end before broadening model coverage
- Keep future room for speaker IDs and reference-audio conditioning
- Reuse the existing `Catalog`, `ModelSelector`, artifact download, and per-bundle feature conventions

### Non-Goals

- Broad model-family coverage in the first cut
- Best-in-class zero-shot voice cloning in the first cut
- A telephony RTP/websocket server inside `libs/model` or `libs/models`
- Browser/mobile deployment in the first cut
- Generic user-supplied TTS checkpoint loading in the first cut
- SSML, prosody markup, word timing, or phoneme-level callbacks in the first cut

## Research Summary

### Candidate Comparison

The table below focuses on local inference fit for Motlie rather than absolute research quality. Latency entries use upstream claims where available; otherwise they state the best stable inference from upstream packaging/runtime evidence.

| Candidate | Representative Size | Latency / Realtime Evidence | CPU Fit on macOS/Linux | CUDA Fit | Voice Cloning | Streaming Capability | Rust Integration Path | License | Fit for Motlie |
|-----------|---------------------|-----------------------------|------------------------|----------|---------------|----------------------|-----------------------|---------|----------------|
| Piper TTS | `en_US-ljspeech-medium.onnx` is about `63.5 MB`; sidecar config is a few KB | Strong. Upstream positions Piper as fast/local, optimized for Raspberry Pi 4, and exposes raw streaming synthesis APIs and CLI flows. | Strong. This is still the cleanest CPU-first candidate in the set. | Good. Upstream supports `onnxruntime-gpu` with `--cuda`, but GPU is optional. | Limited. Some voices are multi-speaker, but this is not a zero-shot cloning system. | Strong. Raw audio can be produced incrementally; `.wav` output is already a first-class path. | Strong. ONNX already fits `CheckpointFormat::Onnx`; backend can target `BackendKind::Ort` with Rust ORT bindings. | Code MIT; voice licenses must be reviewed per `MODEL_CARD`; upstream repo archived on 2025-10-06 and points to GPL successor. | Recommended v1 despite upstream maintenance risk because it best matches the architecture and CPU requirement. |
| Fish Speech 1.5 + `fish-speech.rs` | Fish Speech 1.5 weights are about `1.28 GB`; the Rust server compiles to a static binary of about `15 MB` | Good. `fish-speech.rs` advertises streaming audio and `.wav`; official Fish Audio S2 Pro reports about `100 ms` TTFA and `RTF 0.195` on one H200, but that is a different, newer runtime family. | Medium. `fish-speech.rs` says CPU is supported on Linux and macOS, but the official flagship evidence is GPU-first and the Rust runtime currently targets Fish 1.5 and below, not the latest S2 line. | Strong. Official Fish Audio S2 Pro explicitly publishes high streaming performance on NVIDIA GPUs; the Rust runtime also has CUDA and Metal features. | Strong. Official Fish Audio supports rapid voice cloning from short references; `fish-speech.rs` exposes temporary and persisted cloned voices. | Strong. Both official Fish Audio and `fish-speech.rs` expose streaming/server paths. | Medium. This is the best Rust-native candidate, but the Rust engine is third-party and tied to older Fish checkpoints while the official project has moved to S2 Pro. | Code Apache-2.0 or research license depending on repo; weights are non-commercial (`CC-BY-NC-SA-4.0` / Fish Audio Research License). | Strong technically, but blocked for a general Motlie bundle by weight-license restrictions and ecosystem split. |
| Qwen3-TTS | `Qwen3-TTS-12Hz-0.6B-Base` tree is about `2.52 GB`; `model.safetensors` is about `1.83 GB` | Strong. Official repo claims end-to-end streaming latency as low as `97 ms` and supports streaming/non-streaming in one model family. | Medium-to-weak for v1. The smallest public model is still much larger than Piper and the official quickstart/examples are CUDA-first PyTorch flows. | Strong. Official examples load with `cuda:0`, `bfloat16`, and `flash_attention_2`; this is a good GB10 candidate. | Strong. Officially supports 3-second voice cloning, custom voices, and voice design. | Strong. Streaming is a first-class feature in the official repo and API docs. | Weak-to-medium. Open license is excellent, but the current path is Python package + Transformers/vLLM, not a Rust-native backend. | Apache-2.0 | Strongest phase-2 candidate for quality and feature breadth, but not the simplest v1 backend or artifact story. |
| Coqui XTTS v2 | Hugging Face repo is about `2.09 GB`; `model.pth` alone is about `1.87 GB` | Strong upstream claim: XTTS can stream with sub-`200 ms` latency. | Weak for Motlie v1. Coqui's own streaming server docs say CPU is not recommended. | Good. Common deployment path is CUDA/PyTorch. | Strong. This is the main attraction. | Good. Official streaming server exists. | Weak-to-medium. Would require a new PyTorch/Python-serving boundary or heavyweight FFI; current Motlie backends are not set up for that. | Coqui Public Model License for weights; code in repo is permissive, but weight license is not as operationally simple as MIT/Apache. | Still viable as a cloning-focused backend family, but Qwen3-TTS now looks like the better open phase-2 candidate. |
| Bark | `bark-small` checkpoint family is roughly `1.5-1.7 GB` class | Weak for real-time speech calls. Bark is expressive but not shaped around low-latency speech streaming. | Weak. Heavy autoregressive pipeline and multi-stage generation hurt CPU viability. | Medium. GPU helps, but that does not solve product-fit issues. | Weak. Voice presets exist, but it is not the right cloning/runtime story for this product. | Weak. No strong official streaming path. | Weak. PyTorch-heavy, no clean Rust-native backend path. | MIT | Rejected for v1. Interesting research model, poor fit for telephony-style realtime output. |
| StyleTTS 2 | LibriTTS checkpoint is about `771 MB`; common ONNX conversions are secondary/community artifacts | Medium. Excellent quality, but the official runtime story is notebook/PyTorch-centric. | Medium on paper, but packaging is messy and the official import path depends on a GPL-licensed package for inference. | Good for research GPUs. | Strong. Zero-shot speaker adaptation is a headline capability. | Weak-to-medium. Official repo points to a GPL fork for an experimental streaming API. | Weak. Rust-native path is poor unless Motlie adopts an unsupported/community ONNX conversion. | Code MIT; pretrained-model use terms are narrower than pure MIT; practical inference packaging drifts into GPL forks. | Rejected for v1 due licensing and runtime-integration friction. |
| F5-TTS | `F5TTS_Base` weights are about `1.35 GB` | Strong GPU evidence: upstream reports about `253 ms` average latency on a single L20 with Triton/TensorRT-LLM; also supports chunk inference. | Weak for required CPU-first deployment. Upstream installation supports Apple Silicon, but benchmark guidance is GPU-centered. | Strong. This is one of the better CUDA bonus candidates. | Strong. Multi-speaker/style conditioning is supported. | Good. Chunk inference exists. | Weak-to-medium. Official stack is Python/Torch/Triton; ONNX path is community-maintained, not the primary runtime. | Code MIT, weights `CC-BY-NC-4.0` | Rejected for v1 because the weight license is non-commercial and CPU evidence is weak. |
| Parler-TTS Mini | `880M` params; `model.safetensors` is about `3.75 GB` for Mini v1.1 | Medium. Upstream emphasizes faster generation, `torch.compile`, SDPA/FA2, and provides a streaming guide. | Weak-to-medium. Apple Silicon CPU install exists, but model size is still far too large for a CPU-first Motlie slice. | Good. CUDA and flash-attention paths are first-class. | Weak. Speaker/style prompting exists, but not reference-audio cloning. | Medium. Upstream documents streaming optimization. | Weak. Transformer/PyTorch stack would require a new backend substrate not yet present in Motlie. | Apache-2.0 | Good open research option, poor fit for a first local CPU bundle. |
| Chroma 1.0 | `Chroma-4B` files total about `14.3 GB` across three safetensors shards; gated model | Medium-to-strong for its intended problem. Officially positioned as a real-time spoken dialogue model. | Weak for Motlie v1. Weight size and multimodal dialogue framing make CPU deployment implausible. | Good. `transformers` + `device_map="auto"` is the published path. | Strong. Officially supports reference-audio voice cloning. | Strong in the speech-to-speech sense. | Weak for Motlie TTS. It is a custom-code `transformers` model that expects audio input and multimodal dialogue state, not a simple text-to-PCM backend. | Apache-2.0, but gated behind contact-info sharing on Hugging Face. | Not a good fit for the current TTS scope. Better viewed as a future speech-to-speech/dialogue stack, not a `SpeechRequest { text }` bundle. |
| Kokoro-82M | `82M` params; Hugging Face tree shows `kokoro-v0_19.onnx` about `346 MB` and `.pth` about `327 MB` | Good qualitative evidence. Upstream positions Kokoro as significantly faster and more cost-efficient than larger models and exposes a generator-style pipeline. | Good. This remains the strongest non-Piper CPU candidate. | Medium. Apple Silicon/MPS support exists; CUDA story is not the main attraction. | Weak. This is primarily a compact high-quality voice model, not a cloning-first stack. | Medium. Pipeline yields generated chunks/segments, but transport-focused streaming is less mature than Piper. | Medium. There is an ONNX artifact in the model tree, but the official library path is still Python-first; Rust integration is less stable than Piper. | Apache-2.0 | Best alternate v1 candidate if Piper maintenance risk becomes unacceptable. |
| PersonaPlex | `personaplex-7b-v1` tree is about `17.1 GB`; `model.safetensors` is about `16.7 GB`; gated model | Strong for full-duplex speech dialogue. Official model card reports real-time use and evaluates on turn-taking latency on A100. | Weak for Motlie v1. Even NVIDIA's own docs center GPU deployment and CPU offload, not CPU-native realtime use. | Strong. This is explicitly an NVIDIA CUDA-oriented deployment story and a plausible GB10 research target. | Medium. Voice conditioning is supported, but the model is persona-conditioned speech dialogue rather than standalone TTS. | Strong in the speech-to-speech sense. | Weak for current scope. Official code is a Python/Moshi server with audio input and voice/text prompting; it is not a drop-in text-to-speech backend. | NVIDIA Open Model License, gated access, commercial use permitted. | Strong future research candidate for duplex voice agents, but out of scope for Motlie's first TTS bundle because it is speech-to-speech rather than text-to-speech. |

### Why Piper Is Still the Best First Slice

The first Motlie TTS slice should optimize for architecture fit, operability, and CPU viability, not for maximum cloning quality.

Piper is the best first slice because it aligns with the highest-priority constraints:

- explicit local-first CPU design
- ONNX artifacts that already fit `CheckpointFormat::Onnx`
- straightforward `.wav` and raw audio streaming paths
- easy separation between model output and higher-level playback/transport adapters
- no requirement to embed a Python service or introduce PyTorch as a foundational runtime in `libs/model`

Piper is not the best candidate for zero-shot cloning. It is still the best Motlie v1 candidate because voice cloning is explicitly desirable but not mandatory, while CPU viability and clean bundle/backend integration are mandatory.

### Ranking After the Additional Candidates

The extra candidates change the phase-2 roadmap more than they change the v1 decision.

Updated ranking for Motlie:

1. Piper
   Best first vertical slice for CPU-first local inference, artifact simplicity, and a clean Rust/ONNX integration path.
2. Qwen3-TTS
   Best phase-2 candidate for high-quality open-license streaming TTS with strong cloning and voice-design features, especially if GB10/CUDA becomes a meaningful target.
3. Kokoro-82M
   Best alternate CPU-oriented v1 if Piper maintenance risk outweighs its current integration advantage.
4. Fish Speech
   Technically strong and the best Rust-native story, but the weight-license posture and split between official/new models and third-party/older Rust runtime make it hard to curate cleanly in Motlie.
5. XTTS v2
   Still interesting for cloning, but now less attractive than Qwen3-TTS because the open-license and model-family story are weaker.

Chroma and PersonaPlex do not belong in the same shortlist for this v1 because they are full speech-to-speech dialogue systems rather than pure text-to-speech bundles.

### Major Risks and Constraints

#### Piper maintenance risk

The original `rhasspy/piper` repository was archived on 2025-10-06 and points new development to `OHF-Voice/piper1-gpl`. That does not invalidate Piper voices or the ONNX format, but it does mean Motlie should avoid coupling itself to the archived C++ runtime or the GPL successor.

Recommended mitigation:

- treat Piper as a voice/checkpoint format, not as a library dependency
- implement Motlie's backend directly against ONNX Runtime
- keep phonemization and sidecar-config parsing inside Motlie's backend crate

This preserves the curated artifact format while avoiding the upstream packaging/licensing churn.

#### Voice license variance

Piper voice availability is attractive, but individual voices may carry different dataset/model-card restrictions. The first curated bundle should use a voice whose `MODEL_CARD` terms are operationally acceptable for the intended distribution.

Recommended design stance:

- the design recommends the Piper runtime family
- the exact first curated voice remains a curation decision to confirm during implementation
- `en_US-ljspeech-medium` is the provisional reference voice because it is compact and tied to a published upstream path in `rhasspy/piper-voices`

#### Why Qwen3-TTS is not v1

Qwen3-TTS is the most serious challenger to Piper after this additional research. It is still not the right first slice because:

- the smallest official model is still much larger than Piper
- the published integration path is Python package + Transformers/vLLM, not a Rust-native runtime
- the official examples are CUDA-first and tuned around `flash_attention_2`
- Motlie would need a substantially heavier backend substrate before Qwen3-TTS becomes a clean curated bundle family

Inference from the official sources:

- if Motlie later prioritizes voice quality, voice design, and 3-second cloning over minimal artifact/runtime complexity, Qwen3-TTS should likely be the next family evaluated after Piper
- if Motlie later wants to exploit GB10/CUDA specifically, Qwen3-TTS is a better fit than Piper

#### Why XTTS v2 is not v1

XTTS v2 remains a plausible cloning-focused follow-on, but it is no longer the leading phase-2 recommendation after adding Qwen3-TTS to the evaluation set. It is not v1 because:

- the official fast path is CUDA/PyTorch, not CPU-first
- the official streaming server says CPU is not recommended
- the weight format does not fit the current `libs/model` backend substrate
- a Python-serving backend would be a materially different operational model than the existing Motlie backends

## Recommended Vertical Slice

### Curated Model Choice

Recommended first bundle:

- logical model family: Piper
- runtime/backend: `motlie-model-piper` on top of ONNX Runtime
- checkpoint format: ONNX
- provisional curated voice: `en_US-ljspeech-medium`
- capability surface: streamed speech synthesis only
- primary deployment targets: Linux and macOS CPU
- optional acceleration target: CUDA builds that enable the ORT CUDA provider

Current implementation note:

- the v1 backend preserves the streamed PCM contract but synthesizes the utterance during `open_stream()` and then exposes buffered `S16Le` PCM through `next_chunk()`
- this keeps the public stream contract stable while the first Piper slice proves artifact resolution, phonemization, ONNX inference, and sink integration end to end
- the v1 Piper backend supports `SpeechParams.speaking_rate`, rejects `SpeechParams.seed`, accepts `VoiceConditioning::SpeakerId` only for multi-speaker voices, and rejects reference-audio conditioning

### Why `en_US-ljspeech-medium`

This voice is a good benchmark-shaped first artifact because it is:

- compact at about `63.5 MB`
- mono voice output with a simple artifact story
- already published with the `.onnx` + `.onnx.json` pair that Piper expects
- small enough that bundle download and local-only startup remain operationally cheap

The voice choice is intentionally not overfit into the contract. If curation later prefers a different Piper voice with cleaner license terms, the same API and backend design still hold.

### What Is Intentionally Deferred

- zero-shot cloning bundles
- multi-speaker bundle families
- SSML and timing marks
- vendor-specific telephony codecs in the core model crates
- TTS families requiring non-ONNX runtimes, such as XTTS v2 or F5-TTS (note: Qwen3-TTS is now implemented via ONNX export in Phase 2)

## Architecture

### Crate Layout

```text
libs/model/
  src/
    lib.rs
    speech.rs

libs/model/backends/piper/
  Cargo.toml
  src/
    lib.rs
    common.rs
    speech.rs
    phonemize.rs

libs/models/
  src/
    tts/
      mod.rs
      piper_en_us_ljspeech_medium.rs
```

### High-Level Data Flow

1. Caller chooses a curated selector such as `tts:piper/en_us_ljspeech_medium`.
2. `libs/models` resolves the curated descriptor and artifact rules.
3. On `start()`, the bundle resolves local artifacts or downloads them through the existing curated artifact path.
4. The bundle passes the resolved local `.onnx` checkpoint path to the generic Piper backend adapter.
5. The backend loads the ONNX session, the sidecar config, and phonemization resources.
6. Caller opens a speech stream with a text request.
7. The backend emits PCM chunks through `SpeechStream`.
8. Higher layers choose an adapter:
   `.wav` writer,
   PulseAudio/ALSA sink,
   telephony websocket/RTP adapter,
   or any other consumer of PCM chunks.

### Boundary Rule

The model-layer capability should not accept or own:

- filesystem writers
- PulseAudio/ALSA devices
- RTP sockets
- websocket sessions

Those belong above the model crates.

The TTS capability should only own:

- text input
- voice/synthesis parameters
- streamed PCM chunk output
- audio-format metadata

## Streaming PCM API Contract

### Design Principle

ASR converges multiple input sources into `PcmChunk` input. TTS should converge multiple output consumers onto the same `PcmChunk` output shape.

The stream contract should therefore be:

- transport-neutral
- chunk-oriented
- stateful per stream
- able to emit partial output before the full utterance completes

### Proposed Core Types

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcmEncoding {
    S16Le,
    F32Le,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioSpec {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub encoding: PcmEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcmChunk {
    pub data: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechParams {
    pub speaking_rate: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VoiceConditioning {
    SpeakerId(u32),
    ReferenceAudio {
        audio_spec: AudioSpec,
        pcm: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechRequest {
    pub text: String,
    pub params: SpeechParams,
    pub conditioning: Option<VoiceConditioning>,
}
```

### Stream Traits

```rust
#[async_trait]
pub trait SpeechModel: Send + Sync {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError>;
}

#[async_trait]
pub trait SpeechStream: Send {
    fn audio_spec(&self) -> &AudioSpec;
    async fn next_chunk(&mut self) -> Result<Option<PcmChunk>, ModelError>;
    async fn finish(self: Box<Self>) -> Result<(), ModelError>;
}
```

### Semantics

- `SpeechModel` is shareable and stateless at the contract boundary.
- `SpeechStream` is stateful and owned by one task; it should be `Send`, not `Sync`, matching the ASR stream-ownership design.
- `SpeechRequest.conditioning` is the only source of truth for fixed speaker selection or reference-audio cloning.
- `VoiceConditioning::SpeakerId` and `VoiceConditioning::ReferenceAudio` are mutually exclusive by construction.
- `SpeechRequest.text` must contain at least one non-whitespace character. Empty or whitespace-only text returns `ModelError::InvalidConfiguration`.
- `open_stream()` may front-load synthesis work and return a stream that later drains buffered PCM; callers should rely on chunk ordering and `end_of_stream`, not on token-by-token generation assumptions.
- `next_chunk()` returns `Ok(Some(chunk))` while output is still available.
- `next_chunk()` returns `Ok(None)` only after the stream is exhausted.
- `chunk.sequence` must be monotonic and start at `0` for the first emitted chunk.
- all non-final chunks must have `end_of_stream = false`.
- the final emitted chunk must set `end_of_stream = true`.
- after the final chunk has been emitted, all subsequent `next_chunk()` calls must return `Ok(None)` idempotently.
- `finish(self: Box<Self>)` is an early-termination hook that releases backend resources; it does not promise to synthesize or flush any unread trailing audio. Callers that need the full utterance must continue polling `next_chunk()` until `Ok(None)` before calling `finish()`.

### Why This Shape

- `.wav` output can collect chunks and write one file
- PulseAudio/ALSA can play chunks immediately
- websocket/RTP adapters can re-frame the same PCM into transport-sized frames
- future cloning models can reuse the request shape without replacing the stream contract

## Core Contract Changes in `libs/model`

### Additive Capability Extension

Add:

- `CapabilityKind::Speech`
- `CapabilityDescriptor::speech_stream()`
- `BundleHandle::speech() -> Result<&dyn SpeechModel, ModelError>`
- `SpeechModel`
- `SpeechStream`
- `SpeechRequest`, `SpeechParams`, `VoiceConditioning`, `AudioSpec`, `PcmEncoding`, `PcmChunk`

Reuse:

- `ContentKind::Text` for input
- `ContentKind::Audio` for output
- `InteractionStyle::Streaming`

### Proposed Metadata Behavior

Recommended descriptor:

```rust
pub fn speech_stream() -> Self {
    Self::new(
        CapabilityKind::Speech,
        "Streaming text-to-speech synthesis with PCM audio output.",
        vec![ContentKind::Text],
        vec![ContentKind::Audio],
        InteractionStyle::Streaming,
    )
}
```

### Eval and Family Additions

Add:

- `EvalTrack::Speech`
- `BundleFamily::Piper`

Do not add bundle families for every TTS candidate in v1. Only the curated first slice should become a first-class family.

## Generic Backend Design

### Backend Choice

Recommended crate:

- crate name: `motlie-model-piper`
- path: `libs/model/backends/piper`
- reported backend kind: `BackendKind::Ort`

Rationale:

- `BackendKind::Ort` already exists
- the actual generic runtime is ONNX Runtime
- Piper is the model-family-specific logic layered on top of ORT

### Runtime Responsibilities

The backend crate should own:

- loading the `.onnx` checkpoint
- parsing the adjacent `.onnx.json` sidecar
- text normalization and phonemization
- converting backend outputs into Motlie `PcmChunk`
- optional ORT CUDA provider wiring

The backend crate should not own:

- model download policy
- Hugging Face cache layout rules
- telephony packetization
- `.wav` writing

### Artifact Layout

Piper bundles need both:

- the model checkpoint: `en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx`
- the sidecar config: `en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json`

That fits the current artifact system because `download_checkpoint_artifacts_with_options()` matches repo-relative filenames, not just basenames.

### Phonemization

This is the main backend-specific complication.

Design recommendation:

- keep phonemization inside `motlie-model-piper`
- depend on a local phonemization library or FFI boundary appropriate for the target platforms
- do not shell out to a `piper` executable as the long-term architecture

Why:

- Motlie backends are in-process libraries today
- process management would complicate streaming, error handling, and deployment
- a direct ORT backend keeps the resulting crate aligned with the rest of the stack

### CUDA Support

CUDA is a bonus, not a public-contract fork.

The backend should:

- compile without CUDA by default
- optionally enable the ORT CUDA provider behind a Cargo feature
- continue to expose the same `SpeechModel` / `SpeechStream` contract in both modes

### Cross-Capability ONNX Runtime Refactor Target

The Piper backend should not become a dead-end ORT wrapper.

The ASR roadmap already points at a `sherpa-onnx`-based backend family for phase 2. That means Motlie is on track to have at least two ONNX-driven speech capabilities:

- TTS Piper
- ASR sherpa-onnx

Design requirement:

- the first Piper implementation may start as `motlie-model-piper`, but it should keep its ORT session/provider/config scaffolding factored so it can later move into a shared ONNX helper layer rather than being duplicated again for ASR
- the current implementation still depends on `ort = 2.0.0-rc.12`; shared-session inference remains serialized because this release exposes mutable `Session::run(...)`

Recommended future split:

- capability-neutral ORT bootstrap and provider/config helpers in a shared ONNX support module or crate
- capability-specific graph/session logic in `motlie-model-piper` and the future ASR `motlie-model-sherpa-onnx` backend

This keeps the v1 TTS slice pragmatic without baking duplicated ORT initialization, provider selection, or session tuning paths into multiple speech backends.

## Curated Bundle Design in `libs/models`

### New Namespace

Add:

```text
libs/models/src/tts/
  mod.rs
  piper_en_us_ljspeech_medium.rs
```

Recommended direct module path:

```rust
motlie_models::tts::piper_en_us_ljspeech_medium::descriptor()
motlie_models::tts::piper_en_us_ljspeech_medium::bundle()
```

### Selector and Enum

Add:

- `TtsModels`
- `ModelSelector::Tts(TtsModels)`
- selector string `tts:piper/en_us_ljspeech_medium`

Recommended curated bundle id:

- `piper_en_us_ljspeech_medium`

### Bundle Descriptor Shape

Recommended curated descriptor:

- family: `BundleFamily::Piper`
- backend: `BackendKind::Ort`
- capabilities: `Capabilities::new(vec![CapabilityDescriptor::speech_stream()])`
- eval tracks: `vec![EvalTrack::Speech]`
- platform constraints: Linux and macOS
- build constraints: feature-gated backend, optional CUDA feature

### Artifact Declaration

The curated bundle should use exact repo-relative include rules:

```rust
include: vec![
    ArtifactRule::Exact("en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx"),
    ArtifactRule::Exact("en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json"),
]
```

This avoids the ambiguity that would come from suffix-only matching inside the shared `rhasspy/piper-voices` repository.

### Local Artifact Resolution

The local resolver should:

1. resolve the Hugging Face snapshot root for `rhasspy/piper-voices`
2. return the concrete `.onnx` file path under that snapshot
3. let the backend derive the sidecar `.onnx.json` path from the resolved checkpoint path

## Feature Flag Design

Recommended initial features in `libs/models/Cargo.toml`:

```toml
[features]
model-piper-en-us-ljspeech-medium = ["dep:motlie-model-piper"]
piper-cuda = ["dep:motlie-model-piper", "motlie-model-piper/cuda"]
```

Recommendations:

- keep the TTS bundle out of `default` features in the first PR
- gate `TtsModels`, selector parsing, and catalog registration together
- follow the existing `ModelUnavailable` behavior for known-but-disabled selectors

Possible follow-up profile features:

```toml
profile-voice-local = ["model-piper-en-us-ljspeech-medium"]
profile-voice-dgx = ["model-piper-en-us-ljspeech-medium", "piper-cuda"]
```

For the Qwen3-TTS phase-2 slice, use separate feature gates rather than overloading the Piper ones:

```toml
model-qwen3-tts-0_6b = ["dep:motlie-model-qwen3-tts"]
qwen3-tts-cuda = ["dep:motlie-model-qwen3-tts", "motlie-model-qwen3-tts/cuda"]
```

## Output Adapter Boundary

### Principle

`libs/model` and `libs/models` should end at PCM chunks.

All format conversion and transport adaptation should sit above them.

### Adapter Families

#### `.wav` file sink

- collects the stream
- writes headers plus PCM frames
- best first example path because it is deterministic and easy to validate

#### PulseAudio / ALSA sink

- consumes `AudioSpec` plus PCM chunks
- writes to local playback buffers
- no model-crate changes required beyond the PCM stream contract

#### Telephony websocket / RTP sink

- consumes PCM chunks
- performs resampling and codec conversion as needed by the transport
- packetizes into transport-sized frames
- owns websocket session lifecycle or RTP socket lifecycle

### Why This Boundary Matters

Telephony adapters will often need:

- resampling to transport sample rates
- channel normalization
- codec transforms such as mu-law
- chunk/frame re-packetization

Those are transport responsibilities, not model responsibilities.

Keeping the boundary at PCM chunks means:

- the same model stream can drive a `.wav` file, local speaker output, or a websocket sender
- transport-specific failures do not contaminate the stable model API
- Motlie does not have to encode vendor-specific telephony behavior into the model crates

## Distribution Considerations

### Format Choice

Piper's ONNX format is the strongest distribution fit for Motlie today because:

- `CheckpointFormat::Onnx` already exists
- artifacts are small relative to the other serious candidates
- local-only startup and fetch-on-demand remain reasonable
- the format works on CPU-first deployments

### Why Not GGUF for v1

GGUF is currently a poor fit for TTS v1:

- the strongest TTS candidates are not natively distributed in GGUF
- the telephony/output problem is not helped by GGUF
- forcing a conversion pipeline would add moving parts without improving product fit

### Artifact Size Comparison

Approximate footprint pressure from the evaluated candidates:

- Piper voice: about `63 MB`
- Kokoro-82M ONNX: about `346 MB`
- StyleTTS 2 LibriTTS: about `771 MB`
- Fish Speech 1.5: about `1.28 GB`
- F5-TTS Base: about `1.35 GB`
- Qwen3-TTS 0.6B Base tree: about `2.52 GB`
- XTTS v2: about `2.09 GB`
- Parler-TTS Mini v1.1: about `3.75 GB`
- Chroma-4B: about `14.3 GB`
- PersonaPlex-7B: about `17.1 GB`

That gap matters for:

- artifact download time
- CI cache pressure
- local-first operator experience
- bundled distribution options

### Recommended Bundle Download Policy

- support both `LocalOnly` and `AllowFetch`
- lock the curated bundle to explicit repo-relative artifact paths
- prefer exact include rules over suffix rules
- document the artifact root and local snapshot expectations in the example README

### Phase-2 Qwen3-TTS Vertical Slice

If Motlie adds a second TTS family after Piper, the explicit next vertical slice should be Qwen3-TTS.

Recommended phase-2 shape:

- contract surface: reuse the same `SpeechRequest` / `SpeechStream` API
- runtime/backend boundary: a new `motlie-model-qwen3-tts` backend crate using in-process ONNX Runtime via the shared `motlie-model-ort` helper. The upstream model (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`) ships as safetensors; an offline ONNX export step produces the required `encoder.onnx`, `decoder.onnx`, `vocoder.onnx`, `config.json`, and `vocab.json` components. The curated bundle descriptor references these ONNX artifacts, not the original safetensors.
- curated bundle: start with `Qwen3-TTS-12Hz-0.6B-Base`
- selector shape: `tts:qwen/qwen3_tts_12hz_0_6b`
- feature flags:
  `model-qwen3-tts-0_6b`,
  `qwen3-tts-cuda`
- validation target:
  artifact resolution,
  streamed synthesis,
  voice-cloning request path via `VoiceConditioning::ReferenceAudio`,
  `.wav` output parity with the Piper example,
  and one CUDA-oriented runtime smoke test on GB10-class hardware when available

The purpose of that phase-2 slice is different from Piper:

- Piper proves the small CPU-first local path
- Qwen3-TTS proves the higher-quality cloning-oriented path without changing the public TTS contract

### Qwen3-TTS ONNX Export Procedure

The curated bundle expects pre-exported ONNX model components. The upstream
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` model ships as safetensors; the following
offline export step produces the required artifacts:

1. Install the official `qwen-tts` Python package and `torch`, `onnx`, `onnxruntime`.
2. Load the model from the Hugging Face repo.
3. Export the three components separately:
   - `encoder.onnx` — the text/phoneme encoder (input: `[batch, seq_len]` int64 token IDs; output: `[batch, seq_len, hidden_dim]` float32)
   - `decoder.onnx` — the flow-matching mel decoder (input: encoder hidden states + optional reference mel; output: `[batch, mel_channels, mel_frames]` float32)
   - `vocoder.onnx` — the BigVGAN-derived vocoder (input: `[batch, mel_channels, mel_frames]` float32; output: `[batch, 1, audio_samples]` float32)
4. Export `config.json` with `sample_rate`, `hop_length`, `mel_channels`, and `fft_size`.
5. Flatten the BPE tokenizer into `vocab.json`:
   The upstream repo ships `vocab.json`, `merges.txt`, and `tokenizer_config.json`
   as separate BPE tokenizer components. The export step must merge these into a
   single flat `{ token: id }` JSON mapping that contains all subword tokens
   (including multi-character entries from BPE merges) plus the special tokens
   `<bos>`, `<eos>`, `<unk>`. The `motlie-model-qwen3-tts` backend uses greedy
   longest-match tokenization against this flattened vocabulary — it does NOT
   apply BPE merge rules at runtime. The flattened vocab must therefore contain
   every token the encoder expects, at the correct ID.
6. Place all five files in a directory that the artifact resolver can discover.

**Important:** The curated `vocab.json` is a custom export artifact, not the
upstream `vocab.json` from the HuggingFace model tree. The upstream file contains
only the base vocabulary without merged subword entries. The export step must
produce a flattened vocabulary that includes all BPE-merged tokens.

Note: the tensor shapes listed above are the expected ONNX input/output signatures.
The `motlie-model-qwen3-tts` backend preserves ORT-reported tensor shapes between
stages rather than assuming fixed dimensions.

## Migration and Compatibility Strategy

This is additive brownfield work.

Recommended compatibility strategy:

- extend `BundleHandle` with `speech()`
- update existing backends so `speech()` returns `UnsupportedCapability(CapabilityKind::Speech)`
- keep existing chat/completion/embedding behaviors unchanged
- add capability discovery and selector parsing in a strictly additive way

There is no need for a migration path in application code beyond:

- opting into the new Cargo features
- selecting a TTS bundle
- consuming the new speech capability

## API Sketch

### Core usage

```rust
use motlie_model::{ArtifactPolicy, SpeechRequest, SpeechParams};
use motlie_models::{Catalog, ModelSelector};

let selector: ModelSelector = "tts:piper/en_us_ljspeech_medium".parse()?;
let bundle = selector.bundle();
let handle = bundle
    .start(motlie_model::StartOptions {
        artifact_policy: Some(ArtifactPolicy::AllowFetch { root: None }),
        ..Default::default()
    })
    .await?;

let speech = handle.speech()?;
let mut stream = speech
    .open_stream(SpeechRequest {
        text: "Hello from Motlie.".into(),
        params: SpeechParams::default(),
        conditioning: None,
    })
    .await?;

let spec = stream.audio_spec().clone();
let mut chunks = Vec::new();
while let Some(chunk) = stream.next_chunk().await? {
    chunks.push(chunk);
}
```

### Output adapter sketch

```rust
async fn write_wav_from_stream(
    mut stream: Box<dyn motlie_model::SpeechStream>,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let spec = stream.audio_spec().clone();
    let mut writer = wav_sink::open(path, &spec)?;

    while let Some(chunk) = stream.next_chunk().await? {
        writer.write_pcm(&chunk.data)?;
        if chunk.end_of_stream {
            break;
        }
    }

    writer.finish()?;
    Ok(())
}
```

## Alternatives Considered

### Qwen3-TTS as v1

Pros:

- Apache-2.0
- official streaming support with published low-latency claims
- strong open voice cloning and voice-design support
- better long-term quality ceiling than Piper
- strong fit for GB10/CUDA follow-on work

Cons:

- much heavier artifacts than Piper even for the `0.6B` line
- official integration path is Python-first, not Rust-first
- published examples are tuned for CUDA + FlashAttention
- not a natural `BackendKind::Ort` / `CheckpointFormat::Onnx` fit

Decision:

- best phase-2 TTS family after Piper, but too heavy for the first CPU-first Rust bundle

### Fish Speech as v1

Pros:

- excellent official quality/cloning story
- strongest Rust-native path among the newer candidates because `fish-speech.rs` exists
- streaming and `.wav` are both already exposed in the Rust server

Cons:

- weight license is non-commercial
- the Rust engine is third-party and currently targets Fish 1.5 and below, while the official project has moved to S2 Pro
- official flagship performance story is GPU/SGLang-oriented, not the simple CPU-first path Motlie wants for v1

Decision:

- technically compelling, but not suitable as the first curated Motlie bundle under the current product and licensing constraints

### XTTS v2 as v1

Pros:

- strong cloning capability
- official streaming story
- good CUDA upside

Cons:

- CPU is explicitly not recommended by the official streaming server
- much larger artifacts
- would pull Motlie toward a Python/Torch-serving backend model
- less comfortable weight-license posture than Apache/MIT

Decision:

- keep as a secondary cloning candidate behind Qwen3-TTS, not v1

### Kokoro as v1

Pros:

- compact compared with most modern expressive TTS models
- promising CPU-first profile
- Apache-2.0

Cons:

- official integration path is still Python-first
- runtime/format story is less stable for a Rust backend than Piper
- weaker transport-streaming story than Piper

Decision:

- best fallback if Piper ecosystem churn makes direct ORT support unattractive

### Parler-TTS Mini as v1

Pros:

- Apache-2.0
- high-quality controllable synthesis
- open training/inference stack

Cons:

- still much too large for the first CPU-first curated slice
- PyTorch-heavy runtime
- not a cloning-first model anyway

Decision:

- not the best use of the first TTS integration slot

### Chroma or PersonaPlex as v1

Pros:

- both are modern, low-latency voice-interaction systems
- both support conditioning/persona control beyond plain TTS
- PersonaPlex in particular is a plausible NVIDIA GPU research target

Cons:

- both are fundamentally speech-to-speech conversational models, not text-to-speech bundles
- both are much larger than the CPU-first TTS candidates
- both use Python/custom runtime stacks far outside Motlie's current backend substrate
- both are gated models, which is awkward for a curated fetch/download workflow

Decision:

- out of scope for the first TTS design; reconsider later as separate duplex voice-agent work

## Testing Scope for PLAN

PLAN should make the following verification concrete:

- capability discovery in `libs/model`
- unsupported-capability behavior in existing backends
- ORT backend startup with missing/wrong artifacts
- local-only artifact resolution for the curated Piper voice
- chunked speech synthesis over the new stream contract
- `.wav` end-to-end example from fetch to output file
- optional local playback example
- optional CUDA compilation and startup checks when the feature is enabled

## References

- ASR design template in-repo:
  `libs/models/docs/DESIGN_ASR.md`
- Motlie curated bundle architecture in-repo:
  `libs/models/docs/DESIGN.md`
- Motlie core contract crate in-repo:
  `libs/model/src/lib.rs`
- Coqui TTS repository:
  https://github.com/coqui-ai/TTS
- XTTS v2 model card:
  https://huggingface.co/coqui/XTTS-v2
- Piper repository archive:
  https://github.com/rhasspy/piper
- Piper voice repository:
  https://huggingface.co/rhasspy/piper-voices
- Bark model card:
  https://huggingface.co/suno/bark-small
- StyleTTS 2 repository:
  https://github.com/yl4579/StyleTTS2
- StyleTTS 2 LibriTTS checkpoint:
  https://huggingface.co/yl4579/StyleTTS2-LibriTTS
- F5-TTS repository:
  https://github.com/SWivid/F5-TTS
- F5-TTS weights:
  https://huggingface.co/SWivid/F5-TTS
- Parler-TTS repository:
  https://github.com/huggingface/parler-tts
- Parler-TTS Mini v1.1 weights:
  https://huggingface.co/parler-tts/parler-tts-mini-v1.1
- Kokoro repository:
  https://github.com/hexgrad/kokoro
- Kokoro-82M model tree:
  https://huggingface.co/hexgrad/Kokoro-82M
- Fish Speech official repository:
  https://github.com/fishaudio/fish-speech
- Fish Speech 1.5 model card:
  https://huggingface.co/fishaudio/fish-speech-1.5
- `fish-speech.rs` Rust runtime:
  https://github.com/EndlessReform/fish-speech.rs
- Qwen3-TTS official repository:
  https://github.com/QwenLM/Qwen3-TTS
- Qwen3-TTS 0.6B Base model card:
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- FlashLabs Chroma model card:
  https://huggingface.co/FlashLabs/Chroma-4B
- PersonaPlex official repository:
  https://github.com/NVIDIA/personaplex
- PersonaPlex model card:
  https://huggingface.co/nvidia/personaplex-7b-v1
