# ASR Phase 3 Design

## Changelog
| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-04 23:54 PDT | @codex-369-rv | Recorded the PR #393 / #396 unified-ORT fix: upstream Sherpa keeps `OnlineRecognizer`, but `sherpa-onnx-sys` is patched to exclude its bundled `libonnxruntime.a` and link the single workspace `ort-sys` static ONNX Runtime used by Moonshine. |
| 2026-06-04 22:22 PDT | @codex-369-rv | Clarified the Moonshine interpretation from PR #393: Moonshine is competitive with kroko on PM/orchestration WER, but remains non-default for live telephony because CPU replay runs near or slower than real time with little streaming headroom. |
| 2026-06-04 22:15 PDT | @codex-369-rv | Updated the ASR A/B matrix after PR #393 retargeted the Moonshine rechunk fix plus #376 backend hardening onto `feature/telnyx-voice`: Moonshine now produces real transcripts instead of canned output, with call-center ~31% / 30% and PM ~16% / 14% WER. |
| 2026-06-03 20:49 PDT | @codex-369-rv | Recorded the Telnyx live-model policy after the padded A/B runs: `kroko-2025` is the balanced live default across call-center and PM/technical corpora, while `sherpa-2023` remains the call-center-only profile. |
| 2026-06-03 00:02 PDT | @codex-191-impl | Fixed the measured A/B repro pointer: the golden corpus manifests, `golden-tts` WAV generator, `asr-golden-ab` harness, and playbook live on the `feature/telnyx-voice` harness branch, not in a nonexistent `bins/telnyx-gateway/corpus/` README in this `feature/models` docs PR. |
| 2026-06-02 21:45 PDT | @codex-191-impl | Finalized the Moonshine A/B cell after a clean #376 rebuild: still ~100% WER on call-center and PM corpora with canned transcripts, so the current Motlie Moonshine backend is disqualified / pending repair for M1.5 selection. |
| 2026-06-02 PDT | @codex-191-impl | Added measured A/B results: validated Qwen3-TTS golden A/B over the Telnyx L16/PCMU round-trip (call-center + PM corpora). sherpa-2023 10.5% (call-center best), kroko-2025 14.0% (PM best), Whisper batch/digit-weak, Moonshine ~100% after clean #376 rebuild. Fixed a tail-flush harness artifact (800 ms trailing-silence pad, PR #378) that had inflated all streaming WER 14-24 pts. Hotword bias warranted only for the technical lexicon. Full data on #371. |
| 2026-06-01 22:49 PDT | @codex-191-asr | Addressed PR #369 review round 1 by making upstream-Sherpa and static-ORT claims conditional on PR #368 merging into `feature/models`, correcting streaming contract names, and framing sub-100 ms GPU latency as feasibility only. |
| 2026-06-01 16:03 PDT | @codex-191-asr | Recorded the planned Phase 3 assumption from PR #368 commit `a5c27cd4` on `feature/telnyx-voice`: once that work merges into `feature/models`, the live Sherpa baseline becomes the upstream `sherpa-onnx` crate with its own bundled native archive and internal ONNX Runtime, separate from Motlie shared `ort` runtime. |
| 2026-06-01 14:30 PDT | @codex-191-asr | Folded in issue #191 refinement: batched Whisper as complementary final-pass evaluation, telephony 8 kHz robustness, same-ONNX-Runtime preference, and DGX CUDA placement. |
| 2026-04-17 | @codex-asr | Renamed the shipped Moonshine example path from `v0.7` to `asr_moonshine`. |
| 2026-04-16 | @codex-asr | Added the Phase 3 decision record: sherpa-onnx remains the primary telephony-grade streaming backend, while Moonshine is the secondary batch/offline backend. Documented the measured latency/accuracy tradeoff and the implementation constraint that Moonshine currently runs CPU-only in Motlie because incremental CUDA chunking is unstable. |
| 2026-04-16 | @codex-asr | Corrected the implementation note after the Moonshine backend switched from finish-only buffering to true chunk-driven inference. Moonshine still remains the secondary backend because the measured chunk latency is too high for telephony, not because the integration is batch-only. |

## Decision

- `sherpa-onnx` remains the primary ASR family for telephony and real-time streaming. This PR targets `feature/models`, where the current backend still uses the shared `motlie-model-ort` path; the Phase 3 baseline assumes PR #368 / commit `a5c27cd4` from `feature/telnyx-voice` is merged into `feature/models`, after which the canonical baseline becomes the official upstream `sherpa-onnx` Rust crate `OnlineRecognizer` / `OnlineStream` path.
- The Telnyx gateway live ASR default is `kroko-2025`, selected as the balanced profile after the post-#378 padded A/B runs: it is near target on call-center utterances and materially better on PM/technical utterances. `sherpa-2023` remains the recommended profile for call-center-only deployments.
- `Moonshine Streaming` remains the primary in-scope `transcribe-rs` streaming candidate, but not the live baseline until it can meet telephony chunk latency and stability requirements.
- VAD-gated or utterance-batched Whisper is in scope only as a complementary final-pass / hybrid path. It is not the live engine, and live Whisper micro-chunking remains excluded.
- Nemotron / Parakeet / Canary are DGX batched-GPU investigations, not live-streaming selection candidates unless a Rust-native streaming path exists.

Convergence note: PR #369 remains targeted at `feature/models`. All upstream-Sherpa and static-ORT `download-binaries` statements below are forward-looking assumptions from PR #368 / commit `a5c27cd4` plus the PR #393 unified-ORT fix on `feature/telnyx-voice`; they apply to `feature/models` only after that backend, ORT policy, and native-link convergence work merges there.

## Data

| Backend | Mode | Latency | WER | Streaming viability | Telephony 8 kHz axis | Integration / role | Notes |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| sherpa-onnx today / upstream sherpa-onnx after #368 | CPU chunked streaming | 6.6 ms/chunk historical baseline; remeasure after #368 convergence on Telnyx PCMU | 0.296 historical baseline; remeasure after #368 convergence on narrowband | Yes | Must be measured on Telnyx-style 8 kHz mu-law upsampled to 16 kHz | Current `feature/models`: shared `motlie-model-ort` decoder. Planned after #368/#393 convergence: upstream `sherpa-onnx` `OnlineRecognizer` with `sherpa-onnx-sys` patched so the workspace `ort-sys` static archive is the single ONNX Runtime. | PR #368 on `feature/telnyx-voice` replaced the custom decoder and improved live output from one-word/repeated artifacts to multi-word transcripts; PR #393 keeps that upstream recognizer while removing the duplicate bundled ORT. |
| Moonshine | CPU chunked streaming | 450 ms/chunk | 0.000-0.063 | No for telephony | Must be measured on the same narrowband corpus before any accuracy claim | Primary `transcribe-rs` streaming candidate; shared Motlie `ort` runtime | Strong accuracy, but chunk latency is too high |
| Moonshine | CUDA incremental chunks | Crash | n/a | No | Same narrowband axis applies after CUDA stability exists | DGX investigation only | Whole-file path works; chunked incremental path is unstable |
| whisper.cpp / Whisper | CUDA or CPU utterance batch | 13.7 s/file measured for rolling-window batch | 0.441 measured for rolling-window batch | No for live partials | Important final-pass candidate on narrowband utterances | Complementary final-pass / hybrid; `whisper.cpp` is separate from ORT | Batch-oriented fallback; measure WER uplift vs added utterance latency |
| faster-whisper / CTranslate2 | CUDA utterance batch | tbd | tbd | No for live partials | Must include narrowband robustness | Separate CTranslate2 toolchain | Evaluate only if accuracy/latency beats same-ORT options enough to justify runtime cost |
| Nemotron / Parakeet / Canary | DGX GPU batch or service | tbd | tbd | Out unless Rust-native streaming exists | Strong telephony claims must be validated on Motlie PCMU front-end | NeMo/Riva/NIM service boundary likely | Batched final-pass or appserver-side investigation |

## Measured A/B Results

### PR #393 validated WER + latency (800 ms trailing-silence pad)

| Backend | Call-center WER (L16/PCMU) | PM/orchestration WER (L16/PCMU) | Avg wall latency | Role |
| --- | ---: | ---: | ---: | --- |
| sherpa-2023 zipformer | 11.2% / 11.3% | 18.9% / 19.4% | ~1075-1163 ms | Call-center-only profile; lowest call-center WER |
| sherpa kroko-2025 zipformer | 15.2% / 15.2% | 15.8% / 15.6% | ~954-1034 ms | Balanced Telnyx live default; best mixed-domain Sherpa profile |
| Moonshine streaming | 31.2% / 30.2% | 15.8% / 14.2% | ~4014-5018 ms | PM WER is competitive with kroko; CPU real-time headroom is the blocker |
| whisper-base.en (batch) | 29.7% / 30.2% | 16.5% / 16.5% | ~2319-2322 ms | Batch/final-pass only; weak on digit-heavy categories |

Source: offline Qwen3-TTS golden A/B harness on PR #393 after the unified-ORT fix (`bins/telnyx-gateway` `golden-tts` + `asr-golden-ab`; codecs `libs/voice/src/codec`). Two 72-sample corpora (call-center, PM/orchestration), each run through the Telnyx L16-16k and PCMU-8k (8 kHz mu-law decoded and resampled to 16 kHz) round-trip. Full per-category tables and run artifacts are on PR #393.

The final PR #393 validation ran the all-in-one `golden-ab` executable with Sherpa, Moonshine, Whisper, and Qwen linked together. Earlier #393 heads aborted with `free(): invalid pointer` when Sherpa's bundled static ONNX Runtime and the workspace `ort-sys` static ONNX Runtime were both linked into the same process. PR #393 patches `sherpa-onnx-sys` to filter out Sherpa's bundled `libonnxruntime.a`; upstream Sherpa still uses `OnlineRecognizer`, but the one process-wide static ORT archive comes from workspace `ort-sys`. Both corpora populated all 8 backend/codec cells with no skipped cells and no abort.

PM/orchestration `kroko-2025` is sensitive to regenerated Qwen3-TTS WAV provenance. On the fresh PR #393 PM WAVs, it measures `15.8% / 15.8%`; on the earlier local PM WAV directory from the previous validation, the same PR #393 head measures `14.0% / 14.0%` with the same manifest, chunking, and trailing-silence pad. This is audio-provenance drift, not a code-path regression.

Latency interpretation: Moonshine is not worse than kroko on PM/orchestration accuracy. On the fresh PR #393 PM WAVs, Moonshine scores `15.8% / 14.2%` versus kroko's `15.8% / 15.6%`. The live-default concern is CPU streaming headroom. The replay harness feeds chunks as fast as possible and records offline wall time; kroko processes roughly `3.0 s` PM samples in about `0.95 s` (`~0.32x` real-time factor), while Moonshine processes the same average audio in about `4.0 s` (`~1.34x` real-time factor) plus a `~257 ms` final flush. On call-center audio, kroko is about `0.31x` real time while Moonshine is `1.49-1.51x`. That means Moonshine can produce good transcripts, but on the current CPU path it risks backlog, delayed partials, and poor headroom for live Telnyx streaming.

Repro locations are on the `feature/telnyx-voice` harness branch / PR #377 and PR #378, not introduced by this `feature/models` docs PR:

- Golden corpus manifests: `bins/telnyx-gateway/corpus/qwen3-call-center-golden.json` and `bins/telnyx-gateway/corpus/qwen3-pm-orchestration-golden.json`.
- WAV generation utility: `cargo run -p motlie-telnyx-gateway --features golden-ab -- golden-tts ...`, implemented in `bins/telnyx-gateway/src/golden_ab.rs` and wired through `bins/telnyx-gateway/src/cli.rs`.
- Standalone A/B utility: `cargo run -p motlie-telnyx-gateway --features golden-ab -- asr-golden-ab ...`.
- Playbook doc: `bins/telnyx-gateway/docs/ASR_GOLDEN_AB.md`.
- Generated WAV/run artifacts are local, untracked `/tmp` outputs, currently `/tmp/motlie-qwen3-call-center-golden` and `/tmp/motlie-pm-golden`; Telnyx round-trip audit WAVs are under each corpus output's `asr-inputs/` directory.

### Methodology finding

The first A/B run reported inflated WER (sherpa 24%, kroko 38%) because the golden WAVs ended abruptly (~92 ms mean trailing silence), starving the streaming decoders' final-chunk flush. The deletion penalty scaled with each model's right-context, punishing kroko-2025 (larger lookahead) hardest. This was a harness artifact, not model accuracy. The fix is to feed an 800 ms trailing-silence pad through `ingest()` before `finish()` (PR #378, merged to `feature/telnyx-voice`). All numbers above are post-fix. Live telephony already carries trailing audio, so this only affected offline replay fidelity.

### Insights

- "Newer is better" is domain-dependent: sherpa-2023 wins call-center (~11%), while kroko-2025 remains the better mixed-domain Sherpa profile (`15.8% / 15.8%` on fresh PM WAVs and `14.0% / 14.0%` on the earlier PM WAV set). Both remain below the ~20% call-center target; the digit-handling tuning track is deprioritized for call-center.
- Hotword or contextual bias is warranted only for a technical/PM lexicon, not call-center: sherpa hits 30.6% on `technical_term` (zipformer, mu-law, ONNX, endpointing) versus 7-11% on call-center words. Spoken-number normalization (`metric_readout` 36-44%) is a second lever.
- Whisper stays a complementary word-heavy final-pass only: unfit for live where IDs/numbers occur (digit strings 47-70%) and batch-only at ~2.2 s.
- Moonshine is repaired as a valid ASR backend in PR #393 and is competitive with kroko on PM/orchestration WER, but its CPU real-time factor is above `1.0x` on both corpora, so it is not the live telephony default without further runtime work.
- Codec L16 versus PCMU differs <2 pts everywhere; model and trailing-silence/endpointing handling are the dominant levers, not the codec.

## Recommendation

- Use `kroko-2025` as the Telnyx gateway live default for mixed production traffic because it balances call-center and PM/technical accuracy at about 14% WER on both corpora.
- Use `sherpa-2023` for call-center-only deployments where the narrower corpus matters more than PM/technical robustness.
- Keep runtime selection between `kroko-2025` and `sherpa-2023` as a first-class gateway control so operators and agents can alternate backends between calls during A/B validation.
- Keep Whisper, faster-whisper, Parakeet, Canary, and Nemotron as batched final-pass or appserver-side investigations; none replace the live streaming default.
- Keep hotword/context-bias work separate from raw backend selection. The measured need is technical/PM lexicon support, not a general call-center fix.

### Moonshine status

PR #393 combines the Telnyx rechunker with #376's Moonshine backend hardening and fixes the prior canned-output failure. Moonshine now emits real transcripts through the same `StreamingTranscriber` contract and scores `31.2% / 30.2%` on call-center and `15.8% / 14.2%` on PM/orchestration (`L16-16k / PCMU-8k`). The remaining blocker is not transcript validity or PM accuracy but live CPU headroom: wall averages are roughly `4.0-5.0 s/sample`, corresponding to about `1.34x` real-time factor on PM and `1.49-1.51x` on call-center. Moonshine therefore remains a non-default candidate for offline/final-pass or future optimization rather than the live telephony baseline.

## Rationale

- The historical Sherpa streaming baseline was about 68x faster per chunk than Moonshine on CPU. The #369 document treats PR #368 as the convergence path that would make upstream `sherpa-onnx` the canonical baseline after it merges into `feature/models`; until then, `feature/models` still carries the shared `motlie-model-ort` decoder.
- Moonshine accuracy is attractive, but that does not offset the chunk-latency gap for telephony workloads.
- Moonshine therefore fits best as the secondary chunk-capable backend that still shares the same PCM chunk contract as the primary backend, but is used for non-telephony workloads because its chunk latency is much higher.
- Current Moonshine CUDA incremental behavior is unstable, so the Motlie integration keeps the backend CPU-only for now.
- Telnyx media arrives as 8 kHz mu-law narrowband audio that is decoded and upsampled to 16 kHz. This front-end and model-domain mismatch is now a first-class accuracy axis for every candidate; generic wideband WER alone is not sufficient.
- Simulated-streaming Whisper should be evaluated as a final-pass quality upgrade after VAD endpointing, LocalAgreement, or Simul-Whisper style stabilization. The acceptance metric is WER uplift over the live streaming transcript versus the added finalization latency. Per-frame or micro-chunked Whisper remains excluded for live partials because it worsens both latency and WER.
- The desired post-convergence integration preference is one static CPU ONNX Runtime per process. Motlie-owned ORT backends such as Moonshine and Piper use the shared workspace `ort` / `download-binaries` path after the #368 ORT policy is merged into `feature/models`. The live Sherpa backend remains the upstream `sherpa-onnx` `OnlineRecognizer` path, but PR #393 patches `sherpa-onnx-sys` so its bundled `libonnxruntime.a` is not linked; Sherpa resolves ONNX Runtime symbols from the same workspace `ort-sys` static archive as Moonshine. `faster-whisper` / CTranslate2 and `whisper.cpp` remain additional runtime stacks.

## CUDA / GPU Findings On DGX Spark

Verified by @codex-191-asr at 2026-06-01 14:30 PDT and refreshed at 2026-06-01 16:03 PDT on this DGX host:

- `nvidia-smi` reports one `NVIDIA GB10`, driver `580.159.03`, CUDA `13.0`, and no active GPU processes.
- `nvidia-smi -L` reports `GPU 0: NVIDIA GB10`.
- `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv` returns `NVIDIA GB10, 580.159.03, [N/A], [N/A], [N/A]`; `nvidia-smi -q -d MEMORY` also reports FB and BAR1 memory as `N/A` / not supported on this unified-memory device.
- Host memory reports `MemTotal: 127600748 kB` and `MemAvailable: 112503684 kB` (`free -h`: `121Gi` total, `107Gi` available). Treat that as the on-box unified-memory availability signal rather than a discrete VRAM number from SMI.
- `uname -m` is `aarch64`; `nvcc --version` reports CUDA compilation tools `13.0`, `V13.0.88`.
- CUDA dynamic libraries are present under `/usr/local/cuda/targets/sbsa-linux/lib`, including `libcudart.so.13` and `libcublas.so.13`; no `libcudnn*.so*` files were found in that directory. The earlier temporary CUDA ONNX Runtime shared library path `/tmp/onnxruntime-cuda/build/Linux-sm121/Release/libonnxruntime.so` is not present on this run.
- PR #368 commit `a5c27cd4` on `feature/telnyx-voice` would change `feature/models` after merge by making `libs/model/backends/sherpa_onnx` depend on `sherpa-onnx = "1.13.2"` and removing the former `motlie-model-ort` / `ort` / `ndarray` custom decoder path.
- `cargo info sherpa-onnx@1.13.2` and `cargo info sherpa-onnx-sys@1.13.2` show only `default = [static]`, `static`, and `shared` features; no published `cuda` feature exists in this crate version. PR #368 keeps the Motlie wrapper `cuda` feature as `cuda = []` with the note that upstream currently publishes CPU static archives through the Rust crate.
- The upstream `sherpa-onnx-sys 1.13.2` build script normally links `onnxruntime` inside the Sherpa native archive set and selects archives such as `sherpa-onnx-v1.13.2-linux-aarch64-static-lib.tar.bz2`. PR #393 keeps those upstream Sherpa archives for `OnlineRecognizer`, but patches the sys crate so the bundled `libonnxruntime.a` is not linked; workspace `ort-sys` supplies the single static ONNX Runtime.

CUDA-accelerated ASR paths to keep in the evaluation set:

- `faster-whisper` / CTranslate2 CUDA for batched per-utterance final pass.
- `whisper.cpp` CUDA for GGML/GGUF-style Whisper final pass, already a separate toolchain from ONNX Runtime.
- NVIDIA Parakeet, Canary, and Nemotron through NeMo / Riva / NIM-style service boundaries for DGX-hosted batch or appserver-side final pass.

Sub-100 ms GPU ASR latency is a feasibility claim for GPU-backed ASR services, not a measurement produced by this PR. For the post-#368/#393 Motlie ORT lane, CUDA still means provider/runtime libraries outside the static CPU `ort/download-binaries` policy. The planned upstream Sherpa lane uses the same single static CPU ORT archive in the gateway; `sherpa-onnx 1.13.2` currently exposes CPU static/shared archive modes, not a Motlie `ort/cuda` execution-provider switch. A future Sherpa GPU path would need to be evaluated as a Sherpa-owned native archive, shared build, or service boundary. Therefore the conclusion still holds: CUDA ASR belongs in a DGX-hosted batched final-pass or appserver-side role, while the CPU-portable gateway keeps the streaming Sherpa baseline and Moonshine Streaming as the in-scope `transcribe-rs` streaming candidate after the #368/#393 convergence work lands.

## Integration Shape

- Live baseline backend crate: `libs/model/backends/sherpa_onnx/`
- Current `feature/models` live runtime substrate: shared Motlie ORT through the existing Sherpa custom decoder.
- Planned live runtime substrate after PR #368/#393 merges into `feature/models`: upstream `sherpa-onnx` Rust crate and `OnlineRecognizer`, statically linked through downloaded prebuilt Sherpa native archives while excluding Sherpa's bundled `libonnxruntime.a`; workspace `ort-sys` provides the single static ONNX Runtime archive.
- Live baseline contract: existing `StreamingTranscriber` / `TranscriptionSession`, adapted over upstream `OnlineRecognizer` / `OnlineStream` only after the #368 convergence work lands.
- Moonshine candidate backend crate: `libs/model/backends/moonshine/`
- Moonshine runtime substrate after the #368 ORT policy lands: shared Motlie ONNX Runtime via `transcribe-rs` Moonshine streaming runtime and workspace `ort` download-binaries.
- Moonshine contract: existing `StreamingTranscriber` / `TranscriptionSession`
- Moonshine curated bundle: `libs/models/src/asr/moonshine_streaming_en.rs`
- Moonshine example: `libs/models/examples/asr_moonshine/main.rs`
- Gateway integration preference after convergence: upstream Sherpa `OnlineRecognizer` for live streaming, with `sherpa-onnx-sys` patched to share the workspace static CPU ORT used by Moonshine/Piper/future Motlie-owned ORT backends; keep CUDA ORT and CTranslate2/NeMo/Riva toolchains outside the CPU-portable gateway live path.
- Hybrid final-pass contract: live partials/finals continue to come from the streaming engine; a VAD-gated batched final pass may replace or annotate the final transcript only after measuring WER uplift against added latency.

## Contract Semantics

- The Moonshine backend uses the shared PCM chunk API so `.wav` files and websocket-style streams stay on one contract surface.
- The current integration advances the Moonshine inference state on every `push_chunk()`.
- When `emit_partials = true`, `push_chunk()` may emit a single interim transcript segment representing the current best full-text hypothesis.
- `finish()` only flushes deferred normalization state and performs the final decode pass, returning a committed transcript segment.
- Moonshine is still documented as the non-telephony backend because the measured per-chunk latency remains far above sherpa-onnx even though the integration itself is truly chunk-driven.
