# Curated Model Eval — Cross-Platform Results

> Collated + synthesized by @ops48-orchestrator 2026-06-09 PDT from per-platform runs posted to
> [Discussion #404](https://github.com/chungers/motlie/discussions/404). This is the **committed,
> published summary** for issue #399; raw per-run JSONL stays under gitignored `evals/reports/`.

## Headline

**Every curated bundle that actually ran produced an `overall: pass` record (behavior + perf + resource).**
Six distinct bundles passed across platforms; the embeddings similarity-gap is **consistent across x86
and CUDA (0.6088)** — a clean cross-platform correctness signal. The remaining bundles did not *fail*;
they were **blocked** by missing cached artifacts, HF-gated downloads, GGUF toolchain blockers, slow-on-CPU
chat, or scenarios not yet wired into the runner. See Caveats.

## Cross-platform matrix

Legend: ✅ pass · ⛔ blocked (reason) · — scenario not wired into runner on the tested head.

| Bundle | Scenario | x86 CPU (amd1) | CUDA / GB10 (dgx) | Metal (mac1) |
|---|---|---|---|---|
| `embeddinggemma_300m` | `embeddings_similarity` | ⛔ HF 401 (gated dl) | ⛔ missing artifact | ✅ pass — 768d, gap 0.7840 |
| `qwen3_embedding_06b` | `embeddings_similarity` | ✅ pass — 1024d, gap 0.608794 | ✅ pass — 1024d, gap 0.608786 | — |
| `sherpa_onnx_streaming_zipformer_en` | `asr_short_transcription` | ✅ pass — RTF 0.117 | ✅ pass — RTF 0.098–0.108 | — |
| `whisper_base_en` | `asr_short_transcription` | ✅ pass — RTF 0.477 | ⛔ missing artifact | — |
| `moonshine_streaming_en` | `asr_short_transcription` | ✅ pass — RTF 0.627 | ⛔ missing artifact | — |
| `piper_en_us_ljspeech_medium` | `tts_synthesis_smoke` | ✅ pass — RTF 0.297 | ✅ pass — RTF 0.145–0.147 | — |
| `qwen3_4b` | `chat_smoke` | ⛔ >37m on CPU, interrupted | ⛔ missing artifact | — |
| `gemma4_e2b` | `chat_smoke` | ⛔ >17m on CPU, interrupted | ⛔ missing artifact | — |
| `gemma4_e4b` | `chat_smoke` | ⛔ not run (after E2B overrun) | ⛔ missing artifact | — |
| GGUF chat family¹ | chat | ⛔ `llama-cpp-sys` bindgen `stdbool.h` | ⛔ same `stdbool.h` blocker | — |
| `qwen3_tts_cpp_0_6b` | tts | ⛔ missing native submodule² | (not attempted) | — |

¹ `qwen3_4b_gguf`, `qwen3_6_27b_gguf`, `gemma4_e2b_gguf`, `gemma4_e4b_gguf`, `gemma4_12b_gguf`, `gemma4_12b_qat_q4_0_gguf`.
² `git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp`.

## Per-platform detail

### x86 CPU — amd1, profile `local-cpu-x86_64`, `gpu_backend=unavailable`, swap delta 0 (head `0f86c577`)
| Bundle | Scenario | Backend | Key metrics |
|---|---|---|---|
| `qwen3_embedding_06b` | embeddings_similarity | MistralRs | startup 51.787s; latencies 3.012/7.705/9.798s; 1024d; gap 0.608794; RSS 2.70 GB |
| `whisper_base_en` | asr_short_transcription | WhisperCpp | startup 0.105s; txn 2.376s / 4.985s audio; RTF 0.477; 86 chars; RSS 190 MB |
| `sherpa_onnx_streaming_zipformer_en` | asr_short_transcription | SherpaOnnx | startup 0.843s; txn 0.585s; RTF 0.117; 86 chars; 15 segs; RSS 191 MB |
| `moonshine_streaming_en` | asr_short_transcription | Ort | startup 0.983s; txn 3.127s; RTF 0.627; 86 chars; RSS 441 MB |
| `piper_en_us_ljspeech_medium` | tts_synthesis_smoke | Ort | startup 1.021s; synth 0.334s; audio 1.126s; RTF 0.297; 24832 samp @ 22050 Hz; RSS 158 MB |

### CUDA / GB10 — dgx, profile `dgx-spark`, `gpu_backend=nvidia` (NVIDIA GB10, CUDA 13.0, driver 580.159.03), swap delta 0 (head `0f86c577`)
Union of two runs (impl `spark-2f6e`; cuda-rv `dgx-gb10` with staged `ARTIFACT_ROOT` cache):
| Bundle | Scenario | Backend | Key metrics |
|---|---|---|---|
| `qwen3_embedding_06b` | embeddings_similarity | MistralRs | startup 46.247s; latencies 2158/2095/4346 ms; 1024d; 0.581 vec/s; gap 0.608786; RSS 824 MB |
| `sherpa_onnx_streaming_zipformer_en` | asr_short_transcription | SherpaOnnx | startup 576–632 ms; txn 489–538 ms; RTF 0.098–0.108; 94 chars; 15 segs; RSS 353 MB |
| `piper_en_us_ljspeech_medium` | tts_synthesis_smoke | Ort | startup 747–762 ms; synth 161–162 ms; RTF 0.145–0.147; ~24.3–24.6k samp; RSS 318 MB |

### Metal — mac1, profile `apple-metal`, arm64 / 64 GB, rustc 1.92.0 (head `194acc22`)
Only the embeddings exemplar bundle is wired into `evals run` at this head; chat/asr/tts/perf have `ScenarioKind` stubs but no runner yet.
| Bundle | Scenario | Backend | Key metrics |
|---|---|---|---|
| `embeddinggemma_300m` | embeddings_similarity | MistralRs (safetensors f32) | 768d; similar 0.8794 > dissimilar 0.0954, gap 0.7840; startup 5837 ms; latencies 675/1381/1883 ms; 1.31 vec/s; RSS 1.86 GB |

Gate: `cargo test -p evals` 14/14 ✅; `clippy -D warnings` failed-as-shipped on an unused `std::fs` import → **fixed and committed** (`160f5315` "Gate Linux fs import for Metal"). GPU/accelerator capture on Metal is still `unavailable` (agreed post-batch fast-follow); macOS swap is correctly reported as `unavailable` (the earlier swap false-FAIL is fixed).

## Caveats (read before interpreting)

1. **Heads differ slightly** — x86 + CUDA ran `0f86c577`, Metal ran `194acc22`; branch is now `160f5315`. Not a single-head batch.
2. **Canonical-command coordination gap** — amd1 (x86) did not pick up the impl's `#404` canonical batch command in time and used per-bundle `evals/README` commands (`--download-artifacts` where cache was missing), so methodology is *not byte-identical* across platforms.
3. **`evals report` CLI is not functional** at these heads (it bails on `report`); these tables were hand-built from JSONL + row logs.
4. **Coverage is a curated smoke pass**, not exhaustive — and the runner wires only embeddings/asr/tts scenarios on these heads (chat/perf partially stubbed).
5. **Raw JSONL is gitignored** under `evals/reports/...`; this file is the committed published summary.

## Gap-closure next steps (to make the matrix fully green)

- Stage the missing HF artifact caches on dgx (embeddinggemma, qwen3_4b, gemma4_e2b/e4b, whisper, moonshine) and rerun — these are cache-not-code blocks.
- Provide HF access/token for `google/embeddinggemma-300m` on amd1 (401).
- Fix the `llama-cpp-sys` bindgen `stdbool.h` toolchain path to unblock the GGUF chat family on Linux.
- Add a reduced-token CPU chat profile so `qwen3_4b`/`gemma4` chat smokes complete in reasonable time on x86.
- Init the `qwen3-tts.cpp` submodule before TTS-cpp evals.
- Wire chat/asr/tts/perf scenarios into the Metal runner; land the GPU-capture fast-follow (CUDA + Metal platform records).
- Make `evals report` functional so this collation can be generated, not hand-built.
