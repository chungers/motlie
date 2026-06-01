# ASR Phase 3 Design

## Changelog
| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-01 14:30 PDT | @codex-191-asr | Folded in issue #191 refinement: batched Whisper as complementary final-pass evaluation, telephony 8 kHz robustness, same-ONNX-Runtime preference, and DGX CUDA placement. |
| 2026-04-17 | @codex-asr | Renamed the shipped Moonshine example path from `v0.7` to `asr_moonshine`. |
| 2026-04-16 | @codex-asr | Added the Phase 3 decision record: sherpa-onnx remains the primary telephony-grade streaming backend, while Moonshine is the secondary batch/offline backend. Documented the measured latency/accuracy tradeoff and the implementation constraint that Moonshine currently runs CPU-only in Motlie because incremental CUDA chunking is unstable. |
| 2026-04-16 | @codex-asr | Corrected the implementation note after the Moonshine backend switched from finish-only buffering to true chunk-driven inference. Moonshine still remains the secondary backend because the measured chunk latency is too high for telephony, not because the integration is batch-only. |

## Decision

- `sherpa-onnx` remains the primary ASR backend for telephony and real-time streaming.
- `Moonshine Streaming` remains the primary in-scope `transcribe-rs` streaming candidate, but not the live baseline until it can meet telephony chunk latency and stability requirements.
- VAD-gated or utterance-batched Whisper is in scope only as a complementary final-pass / hybrid path. It is not the live engine, and live Whisper micro-chunking remains excluded.
- Nemotron / Parakeet / Canary are DGX batched-GPU investigations, not live-streaming selection candidates unless a Rust-native streaming path exists.

## Data

| Backend | Mode | Latency | WER | Streaming viability | Telephony 8 kHz axis | Integration / role | Notes |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| sherpa-onnx | CPU chunked streaming | 6.6 ms/chunk | 0.296 | Yes | Must be measured on Telnyx-style 8 kHz mu-law upsampled to 16 kHz | Live baseline; same-ORT CPU gateway fit | Telephony-grade incremental decode |
| Moonshine | CPU chunked streaming | 450 ms/chunk | 0.000-0.063 | No for telephony | Must be measured on the same narrowband corpus before any accuracy claim | Primary `transcribe-rs` streaming candidate; same-ORT fit | Strong accuracy, but chunk latency is too high |
| Moonshine | CUDA incremental chunks | Crash | n/a | No | Same narrowband axis applies after CUDA stability exists | DGX investigation only | Whole-file path works; chunked incremental path is unstable |
| whisper.cpp / Whisper | CUDA or CPU utterance batch | 13.7 s/file measured for rolling-window batch | 0.441 measured for rolling-window batch | No for live partials | Important final-pass candidate on narrowband utterances | Complementary final-pass / hybrid; `whisper.cpp` is separate from ORT | Batch-oriented fallback; measure WER uplift vs added utterance latency |
| faster-whisper / CTranslate2 | CUDA utterance batch | tbd | tbd | No for live partials | Must include narrowband robustness | Separate CTranslate2 toolchain | Evaluate only if accuracy/latency beats same-ORT options enough to justify runtime cost |
| Nemotron / Parakeet / Canary | DGX GPU batch or service | tbd | tbd | Out unless Rust-native streaming exists | Strong telephony claims must be validated on Motlie PCMU front-end | NeMo/Riva/NIM service boundary likely | Batched final-pass or appserver-side investigation |

## Rationale

- `sherpa-onnx` is about 68x faster per chunk than Moonshine on CPU.
- Moonshine accuracy is attractive, but that does not offset the chunk-latency gap for telephony workloads.
- Moonshine therefore fits best as the secondary chunk-capable backend that still shares the same PCM chunk contract as the primary backend, but is used for non-telephony workloads because its chunk latency is much higher.
- Current Moonshine CUDA incremental behavior is unstable, so the Motlie integration keeps the backend CPU-only for now.
- Telnyx media arrives as 8 kHz mu-law narrowband audio that is decoded and upsampled to 16 kHz. This front-end and model-domain mismatch is now a first-class accuracy axis for every candidate; generic wideband WER alone is not sufficient.
- Simulated-streaming Whisper should be evaluated as a final-pass quality upgrade after VAD endpointing, LocalAgreement, or Simul-Whisper style stabilization. The acceptance metric is WER uplift over the live sherpa transcript versus the added finalization latency. Per-frame or micro-chunked Whisper remains excluded for live partials because it worsens both latency and WER.
- The gateway now prefers a single statically linked CPU ONNX Runtime path through `ort` download-binaries, with no source build and no manual `ORT_LIB_PATH`. Prefer candidates that can run on that same ORT. `sherpa-onnx` can host non-streaming Whisper on ONNX Runtime; `faster-whisper` / CTranslate2 and `whisper.cpp` bring separate runtime stacks.

## CUDA / GPU Findings On DGX Spark

Verified by @codex-191-asr at 2026-06-01 14:30 PDT on this DGX host:

- `nvidia-smi` reports one `NVIDIA GB10`, driver `580.159.03`, CUDA `13.0`, and no active GPU processes.
- `nvidia-smi -L` reports `GPU 0: NVIDIA GB10`.
- `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv` returns `NVIDIA GB10, 580.159.03, [N/A], [N/A], [N/A]`; `nvidia-smi -q -d MEMORY` also reports FB and BAR1 memory as `N/A` / not supported on this unified-memory device.
- Host memory reports `MemTotal: 127600748 kB` and `MemAvailable: 112503684 kB` (`free -h`: `121Gi` total, `107Gi` available). Treat that as the on-box unified-memory availability signal rather than a discrete VRAM number from SMI.
- `uname -m` is `aarch64`; `nvcc --version` reports CUDA compilation tools `13.0`, `V13.0.88`.
- CUDA dynamic libraries are present under `/usr/local/cuda/targets/sbsa-linux/lib`, including `libcudart.so.13` and `libcublas.so.13`; no `libcudnn*.so*` files were found in that directory. The earlier temporary CUDA ONNX Runtime shared library path `/tmp/onnxruntime-cuda/build/Linux-sm121/Release/libonnxruntime.so` is not present on this run.

CUDA-accelerated ASR paths to keep in the evaluation set:

- `faster-whisper` / CTranslate2 CUDA for batched per-utterance final pass.
- `whisper.cpp` CUDA for GGML/GGUF-style Whisper final pass, already a separate toolchain from ONNX Runtime.
- NVIDIA Parakeet, Canary, and Nemotron through NeMo / Riva / NIM-style service boundaries for DGX-hosted batch or appserver-side final pass.

Sub-100 ms GPU ASR latency is feasible when a GPU service is available, but it should not change the gateway live-engine decision by itself. The key architectural tension is that ONNX Runtime CUDA execution provider use requires CUDA provider/runtime libraries and dynamic linking, while the gateway preferred deployment path is static CPU ONNX Runtime through `ort` download-binaries. Therefore CUDA ASR belongs in a DGX-hosted batched final-pass or appserver-side role. The CPU-portable gateway should keep the sherpa-onnx streaming baseline, with Moonshine Streaming as the in-scope `transcribe-rs` streaming candidate.

## Integration Shape

- Backend crate: `libs/model/backends/moonshine/`
- Runtime substrate: ONNX Runtime via `transcribe-rs` Moonshine streaming runtime
- Contract: existing `TranscriptionModel` / `TranscriptionStream`
- Curated bundle: `libs/models/src/asr/moonshine_streaming_en.rs`
- Example: `libs/models/examples/asr_moonshine/main.rs`
- Gateway integration preference: shared static CPU ONNX Runtime via `ort` download-binaries; keep CUDA ORT and CTranslate2/NeMo/Riva toolchains outside the CPU-portable gateway live path.
- Hybrid final-pass contract: live partials/finals continue to come from the streaming engine; a VAD-gated batched final pass may replace or annotate the final transcript only after measuring WER uplift against added latency.

## Contract Semantics

- The Moonshine backend uses the shared PCM chunk API so `.wav` files and websocket-style streams stay on one contract surface.
- The current integration advances the Moonshine inference state on every `push_chunk()`.
- When `emit_partials = true`, `push_chunk()` may emit a single interim transcript segment representing the current best full-text hypothesis.
- `finish()` only flushes deferred normalization state and performs the final decode pass, returning a committed transcript segment.
- Moonshine is still documented as the non-telephony backend because the measured per-chunk latency remains far above sherpa-onnx even though the integration itself is truly chunk-driven.
