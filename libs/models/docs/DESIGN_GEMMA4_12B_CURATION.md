# Gemma 4 12B Curation Design

## Status: Draft

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-06 10:28 PDT | @gemma4-cdx | Recorded DGX GB10 validation from codex-398-dgx-rv: standard GGUF Q4_K_M and QAT Q4_0 pass live tool-use at ~24/22 gen-tps, while full safetensors on mistral.rs builds and loads but has a real generation/tool-use defect and is not M1 live-accepted yet. |
| 2026-06-06 00:51 PDT | @gemma4-cdx | Addressed second-reviewer L1/M2: GGUF cache tests now use collision-resistant temp directories, local-only GGUF resolution validates exact variant filenames, and generic logical-model resolution documents that QAT requires an explicit selector/bundle id. |
| 2026-06-05 22:29 PDT | @gemma4-cdx | Clarified David's curation direction: safetensors, standard GGUF, and QAT GGUF are all curated Gemma 4 12B variants selected by platform/performance fit, not a priority hierarchy. |
| 2026-06-05 16:59 PDT | @gemma4-cdx | Fixed and validated Gemma 4 12B GGUF live tool-use: default tool demo now disables thinking for the 12B GGUF path, llama.cpp parser strips empty Gemma thought-channel markers, and default Q4 tool demo completed all tool calls plus final answer locally. |
| 2026-06-05 16:36 PDT | @gemma4-cdx | Completed corrected GGUF artifact download, tightened artifact rules to exact Q4/Q8 root files after an MTP suffix-match bug, and ran Q4 startup/warmup successfully on the local CPU host. |
| 2026-06-05 16:11 PDT | @gemma4-cdx | Investigated and fixed GGUF build status: local host needed CMake plus bindgen include paths; corrected GGUF-only cfg wiring, and GGUF lib/example/bench checks now pass with the local toolchain workaround. |
| 2026-06-04 23:08 PDT | @gemma4-cdx | Added verification trace: safetensors library, multimodal example, and benchmark compile; GGUF library and example checks were blocked locally by `llama-cpp-sys-2` missing `stdbool.h` before the 2026-06-05 host/toolchain investigation. |
| 2026-06-04 22:58 PDT | @gemma4-cdx | Killed the no-timeout local Q8 ISQ warmup after about 26 minutes without startup completion; documented memory/timeout findings and expanded M1 to include a llama.cpp/GGUF curated variant alongside safetensors/Mistral for platform/performance selection. |
| 2026-06-04 20:25 PDT | @gemma4-cdx | Added M1 full-artifact validation: official safetensors downloaded, `mistral.rs` pinned to upstream revision `47ec459c`, and CPU-only startup was stopped after heavy swap pressure. |
| 2026-06-04 16:08 PDT | @gemma4-cdx | Opened and linked milestone issues M1 #389, M2 #390, and M3 #391 under parent issue #388. |
| 2026-06-04 16:02 PDT | @gemma4-cdx | Applied David's ASR direction: ASR is a later required milestone, not a stretch goal, and may be gated on PR #387 merge. |
| 2026-06-04 15:57 PDT | @gemma4-cdx | Applied David's direction: full-precision safetensors support is required, GGUF is also needed for platform/performance coverage when full precision is too slow or operationally heavy, and `mistralrs` upgrade is required. |
| 2026-06-04 15:55 PDT | @gemma4-cdx | Updated recommendation after David feedback: make official safetensors on `mistral.rs` the full-model variant for DGX/full-model hosts, with full precision as the default and ISQ as explicit override. |
| 2026-06-04 15:36 PDT | @gemma4-cdx | Initial greenfield design proposal for curating Gemma 4 12B into Motlie for chat, tool use, and ASR feasibility review. |

## Scope

This is greenfield curation work for the Gemma 4 12B multimodal model. It
records the design, M1 implementation direction, and local validation findings
for adding curated 12B bundles. It evaluates how honestly those bundles can
advertise three requested capabilities:

- `Chat`
- `ToolUse`
- later-milestone `Transcription` through Gemma 4 audio understanding

The proposal reuses existing Motlie patterns:

- curated bundle descriptors in `libs/models`
- `mistral.rs` safetensors multimodal bundles used by Gemma 4 E2B/E4B
- `llama.cpp` GGUF text bundles used by Gemma 4 E2B/E4B
- the merged Motlie tool contract in `libs/model/src/tool.rs`
- caller-owned tool execution through `libs/models/src/tool_registry.rs`
- typed ASR traits in `StreamingTranscriber` and `TranscriptionSession`

## Research Snapshot

External research date: 2026-06-04.

Gemma 4 12B was announced on 2026-06-03. Google describes it as a dense,
encoder-free multimodal model, the first mid-sized Gemma model with native audio
input, sized for local use on 16 GB VRAM or unified memory, and released under
Apache 2.0.

Hugging Face availability:

| Artifact | Repo | License | Notes |
|----------|------|---------|-------|
| Instruction tuned safetensors | `google/gemma-4-12B-it` | Apache 2.0 | Official model card, `Any-to-Any`, `Transformers`, `Safetensors` tags. |
| Base safetensors | `google/gemma-4-12B` | Apache 2.0 | Base model. Not the preferred chat/tool curation target. |
| GGUF, ggml-org | `ggml-org/gemma-4-12B-it-GGUF` | inherits upstream | Standard Q4_K_M, Q8_0, BF16 files; llama.cpp examples. |
| GGUF, Unsloth | `unsloth/gemma-4-12b-it-GGUF` | inherits upstream | Existing Motlie Gemma GGUF source pattern; many quants plus text/image/audio llama.cpp instructions. |

Published model traits:

- Inputs: text, image, audio, and video.
- Output: text only.
- Context: up to 256K tokens for the 12B model.
- Audio limit: maximum 30 seconds per audio input.
- Recommended multimodal order: image before text, audio after text.
- Recommended sampling: `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- Audio prompt guidance includes an ASR prompt that requests transcription only,
  no newlines, and digit formatting.

Quantization and footprint:

| Source | Quant | Published file size | Curation note |
|--------|-------|---------------------|---------------|
| `ggml-org/gemma-4-12B-it-GGUF` | Q4_K_M | 7.38 GB | Simple, official ggml-org distribution. |
| `ggml-org/gemma-4-12B-it-GGUF` | Q8_0 | 12.7 GB | Likely higher quality, tighter on 16 GB hosts once KV cache and multimodal projector are included. |
| `ggml-org/gemma-4-12B-it-GGUF` | BF16 | 23.8 GB | Not a 16 GB target. |
| `unsloth/gemma-4-12b-it-GGUF` | Q4_K_M | 7.26 GB | Matches existing Motlie Gemma GGUF source style and is the selected standard GGUF default for local/CPU-constrained profiles. |
| `unsloth/gemma-4-12b-it-GGUF` | Q8_0 | 12.5 GB | Optional quality tier. |
| `unsloth/gemma-4-12b-it-GGUF` | BF16 | 23.4 GB | Available, but not included in the Motlie text-only GGUF artifact rules. |
| `unsloth/gemma-4-12b-it-GGUF` | UD-Q4_K_XL | 7.37 GB | Attractive, but Motlie's `QuantizationBits` mapper currently targets standard GGUF suffixes. |

Runtime status:

- Transformers can load the official model through `AutoProcessor` and
  `AutoModelForMultimodalLM`, including audio examples.
- Recent `mistral.rs` upstream docs advertise Gemma 4 full multimodality:
  text, image, video, and audio input. Motlie currently locks `mistralrs` at
  `0.8.1`; upstream docs call out `0.8.2` as current.
- Recent llama.cpp GGUF repos advertise text, image, and audio use through
  `llama-server`, with automatic `mmproj` download for `-hf`.
- Motlie's current `motlie-model-llama-cpp` wrapper is text-only: it flattens
  `ChatMessage` content to text and rejects `ContentPart::Image` /
  `ContentPart::ImageUrl`.
- Motlie's current `motlie-model-mistral` multimodal wrapper accepts text and
  inline image bytes. It imports upstream audio/video request types but currently
  sends empty `audios` and `videos` arrays because `ContentPart` has no audio or
  video variants.

M1 validation update, 2026-06-04 20:25 PDT by @gemma4-cdx:

- Official `google/gemma-4-12B-it` artifacts downloaded successfully through Motlie's curated downloader without a Hugging Face token on this host.
- Cached HF snapshot revision: `5926caa4ec0cac5cbfadaf4077420520de1d5205`.
- Cached files: `chat_template.jinja`, `config.json`, `generation_config.json`, `model.safetensors`, `processor_config.json`, `tokenizer.json`, and `tokenizer_config.json`.
- `model.safetensors` size: 23,919,549,408 bytes. Full cache size: about 23 GB.
- `mistral.rs` tag `v0.8.3` was insufficient for the official 12B config: it failed auto-detection on `Gemma4UnifiedForConditionalGeneration`, then failed the older `vision_tower` tensor layout when forced to Gemma4.
- Upstream `mistral.rs` master revision `47ec459cbd6d5b0d6c9035bb79d8cf1e37ee14a0` contains the required Gemma4-unified architecture mapping and `vision_embedder.patch_dense` support. M1 therefore pins that exact revision until an upstream tag includes the same support.
- Local CPU-only host was not a valid full-precision performance target: AMD Ryzen 7 7730U, 28 GiB RAM, no `nvidia-smi`. Full-precision startup was stopped after 17m35s before generation, with peak RSS about 18.6 GiB and heavy swap pressure.
- Required next validation target remains DGX Spark or another full-model CUDA/unified-memory host. Later local diagnostics validated GGUF as a local/CPU-friendly curated variant; it complements, but does not replace, full-model DGX validation.


M1 local runtime update, 2026-06-04 22:58 PDT by @gemma4-cdx:

- The five minute limit was an explicit shell diagnostic (`timeout 300s`) around
  benchmark commands; it is not a Motlie runtime timeout.
- Host facts: AMD Ryzen 7 7730U, CPU-only (`nvidia-smi` absent), 28 GiB usable
  RAM, and 8 GiB swap. `/proc/meminfo` showed `MemTotal=29664092 kB` (28.3 GiB)
  and `CommitLimit=23220648 kB` (about 22.1 GiB) under Linux's default
  overcommit heuristic.
- Official safetensors header inspection found 677 BF16 tensors, a tensor
  payload of 23,919,460,448 bytes (22.28 GiB), and a largest tensor of about
  1920 MiB for `model.language_model.embed_tokens.weight` with shape
  `[262144, 3840]`.
- Full precision `bench_chat --model=gemma4-12b --precision=f32 --iterations=0`
  under the 300 second diagnostic did not reach startup completion. Peak RSS was
  11,443,644 KB, CPU was about 99%, and the process recorded 245 major faults.
- A full precision prompt run was manually stopped after 17m35s before
  generation. Peak RSS was 19,457,244 KB (about 18.6 GiB) with about 5.7 GiB of
  swap pressure observed during the run.
- Q8 ISQ `bench_chat --model=gemma4-12b --precision=q8 --iterations=0` under the
  300 second diagnostic also did not reach startup completion. Peak RSS was
  9,048,928 KB, CPU was about 129%, and no major faults were recorded.
- The no-timeout Q8 ISQ warmup run was killed at David's request after about 26
  minutes. The command used `/usr/bin/time -v` around
  `target/debug/examples/bench_chat --model=gemma4-12b --precision=q8
  --iterations=0`; PIDs 3288032 and 3288043 were killed. It had not printed
  `startup-ms`, had not reached warmup generation, and had not surfaced a
  backend error.
- Interpretation: this local 28 GiB CPU-only host is not a useful validation
  target for official safetensors. Even when ISQ is requested, the backend must
  read, materialize, and traverse the 22.28 GiB BF16 artifact before or while
  quantizing. Q8 lowers the final resident shape but does not avoid the local
  CPU-side load/conversion cost.
- Decision: keep `google/gemma4_12b` safetensors/Mistral as the full-model
  curated variant for DGX Spark or another full-model CUDA/unified-memory host,
  and add `google/gemma4_12b_gguf` llama.cpp/GGUF as a separate curated variant
  for hosts where GGUF gives the best startup, memory, or latency fit.
- The GGUF variant follows the existing Gemma GGUF pattern: source
  `unsloth/gemma-4-12b-it-GGUF`, bundle id `gemma4_12b_gguf`, Q4_K_M default,
  Q8_0 supported, and no BF16 or `mmproj-F16.gguf` in the initial artifact rules.
- Current capability boundary: Motlie's `llama.cpp` wrapper is text-only, so the
  GGUF variants can cover chat, completion, and tool-use smoke, but they cannot
  validate image/audio/video or the later ASR milestone.
- Initial local GGUF verification blocker: `llama-cpp-sys-2 v0.1.146`
  bindgen failed before Rust type-checking because clang could not find the C
  standard header `stdbool.h`. The 2026-06-05 follow-up worked around that with
  GCC include paths, supplied local CMake, fixed two Motlie cfg bugs, and got the
  GGUF compile checks passing.

M1 implementation snapshot, 2026-06-04 22:58 PDT by @gemma4-cdx:

- Safetensors full-model variant retained: feature `model-gemma4-12b`, selector
  `google/gemma4_12b`, bundle id `gemma4_12b`, backend `BackendKind::MistralRs`,
  source `google/gemma-4-12B-it`, full precision by default with explicit ISQ
  Q4/Q8 overrides.
- GGUF variant added in parallel: feature `model-gemma4-12b-gguf`, selector
  `google/gemma4_12b_gguf`, bundle id `gemma4_12b_gguf`, backend
  `BackendKind::LlamaCpp`, source `unsloth/gemma-4-12b-it-GGUF`, Q4_K_M default
  and Q8_0 supported.
- Example surface updated to allow `chat_gguf_gwen3_gemma4` and `bench_chat` to
  select the new GGUF variant by flag, matching the existing E2B/E4B GGUF
  examples.


M1 verification update, 2026-06-04 23:08 PDT by @gemma4-cdx:

- Full-repo `cargo fmt --check` is blocked by an unrelated pre-existing module
  resolution issue: rustfmt tries to load missing
  `examples/vector2/app/benchmark.rs`.
- Scoped formatting succeeded with `rustfmt --edition 2024` on the touched Rust
  files: `libs/model/backends/llama_cpp/src/text.rs`,
  `libs/models/src/chat/gemma4_12b_gguf.rs`, `libs/models/src/chat/mod.rs`,
  `libs/models/src/lib.rs`, `libs/models/examples/chat_gguf_gwen3_gemma4/main.rs`,
  and `libs/models/examples/bench_chat.rs`.
- `cargo check -p motlie-models --no-default-features --features model-gemma4-12b
  --lib` passed.
- `cargo check -p motlie-models --no-default-features --features model-gemma4-12b
  --example chat_multimodal_gemma4` passed.
- `bench_chat` initially still had `required-features = ["model-qwen3-4b"]` in
  `libs/models/Cargo.toml`, which prevented a 12B-only benchmark build. That
  hard gate was removed so the benchmark can follow its runtime `--model` flag.
- `cargo check -p motlie-models --no-default-features --features model-gemma4-12b
  --example bench_chat` passed after removing that qwen-only gate.
- Initial `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --lib` was blocked locally before Motlie code was
  type-checked. `llama-cpp-sys-2 v0.1.146` bindgen failed with
  `ggml.h:211:10: fatal error: 'stdbool.h' file not found`.
- Initial `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --example chat_gguf_gwen3_gemma4` was blocked by the same
  `llama-cpp-sys-2` / `stdbool.h` C toolchain issue. The 2026-06-05 follow-up
  supplied a local CMake binary, passed GCC include paths to bindgen, fixed two
  GGUF-only cfg bugs, and got the GGUF library, example, and benchmark checks
  passing.
- No GGUF runtime smoke or artifact download had been run at this point because
  the local llama.cpp build failed before the example binary could be produced.


M1 GGUF build fix update, 2026-06-05 16:11 PDT by @gemma4-cdx:

- Status correction: GGUF was not fundamentally broken. It had two host setup
  blockers and two Motlie cfg wiring bugs that only became visible after the host
  blockers were bypassed.
- Host blocker 1: `clang` was not on `PATH`, clang resource headers were not
  installed, and bindgen could not find `stdbool.h`. GCC 13 did have the needed
  headers under `/usr/lib/gcc/x86_64-linux-gnu/13/include`.
- Host blocker 2: `cmake` was not installed. `sudo apt-get install` was not
  available in this session because sudo requires a password.
- Non-root host workaround used for verification: downloaded Ubuntu `cmake`,
  `cmake-data`, `librhash0`, and `libjsoncpp25` packages, extracted them under
  `/tmp/motlie-cmake`, and ran cargo with:

```sh
PATH=/tmp/motlie-cmake/usr/bin:$PATH LD_LIBRARY_PATH=/tmp/motlie-cmake/usr/lib/x86_64-linux-gnu BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include -isystem /usr/include/x86_64-linux-gnu -isystem /usr/include'   cargo check -p motlie-models --no-default-features --features model-gemma4-12b-gguf --lib
```

- Motlie bug 1 fixed: `model-gemma4-12b-gguf` was accidentally included in the
  `MistralMultimodalHandle` import cfg, causing a GGUF-only build to require the
  Mistral backend dependency. The GGUF feature now pulls only `LlamaCppTextHandle`.
- Motlie bug 2 fixed: `chat_gguf_gwen3_gemma4` included
  `model-gemma4-12b-gguf` in the no-feature stub cfg but omitted it from the
  real `gguf_example` module cfg. A 12B-only example build therefore compiled
  neither module. The real module cfg now includes the 12B GGUF feature.
- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --lib` passed with the local toolchain workaround.
- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --example chat_gguf_gwen3_gemma4` passed with the local
  toolchain workaround.
- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --example bench_chat` passed with the local toolchain
  workaround.
- Recommended durable host fix: install normal system prerequisites instead of
  using the `/tmp` workaround, for example CMake plus clang/libclang resource
  headers. On this Ubuntu host that means packages such as `cmake`, `clang`, and
  `libclang-common-18-dev` / `libclang-18-dev`, subject to the final base image
  policy.
- Runtime status: compile is fixed. Artifact download and live GGUF chat/tool
  smoke have not yet run in this follow-up.


M1 GGUF artifact and startup update, 2026-06-05 16:36 PDT by @gemma4-cdx:

- During the first real GGUF download attempt, the suffix artifact rule
  `ArtifactRule::Suffix("-Q8_0.gguf")` matched Unsloth's new
  `MTP/gemma-4-12B-it-MTP-Q8_0.gguf` drafter file. That is not part of the
  Motlie text-only GGUF bundle.
- The active downloader was stopped, and `gemma4_12b_gguf` artifact rules were
  changed from broad suffixes to exact root filenames:
  `gemma-4-12b-it-Q4_K_M.gguf` and `gemma-4-12b-it-Q8_0.gguf`.
- The descriptor test now explicitly rejects
  `MTP/gemma-4-12B-it-MTP-Q8_0.gguf`, `gemma-4-12b-it-BF16.gguf`, and
  `mmproj-F16.gguf`.
- Corrected curated download completed with exactly two reported files:
  `gemma-4-12b-it-Q4_K_M.gguf` and `gemma-4-12b-it-Q8_0.gguf` from snapshot
  `3f09de26549e6d7ea54f1b83755149f840fcd333`.
- A stale MTP blob remains in the local HF cache from the interrupted first
  download, but the corrected artifact rules no longer request it and the
  llama.cpp loader searches the snapshot root non-recursively for the requested
  quantization file.
- Q4 startup benchmark command passed with local CMake and bindgen include paths:

```sh
PATH=/tmp/motlie-cmake/usr/bin:$PATH LD_LIBRARY_PATH=/tmp/motlie-cmake/usr/lib/x86_64-linux-gnu BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include -isystem /usr/include/x86_64-linux-gnu -isystem /usr/include'   /usr/bin/time -v cargo run -p motlie-models --no-default-features --features model-gemma4-12b-gguf   --example bench_chat -- --model=gemma4-12b-gguf --precision=q4 --iterations=0 'Say OK.'
```

- Q4 runtime result on this CPU-only host: startup 14,517 ms, warmup one-word
  response 15,389 ms, final RSS 11,284.8 MiB, peak RSS 22,055.1 MiB, 0 swaps,
  and about 2 generated words/sec reported by the benchmark metric snapshot.
- `bench_chat` was corrected to label GGUF runs with `quantization_label_gguf`
  instead of the safetensors/ISQ label, and it now accepts `--precision=f16` as
  the no-quantization GGUF spelling.
- Superseded remaining validation note: the 2026-06-05 16:59 PDT follow-up ran
  the live GGUF tool example and fixed the tool-demo thinking/parser issues. Broad
  performance evals and DGX/full-safetensors validation still remain.

M1 GGUF live tool-use update, 2026-06-05 16:59 PDT by @gemma4-cdx:

- Plain GGUF startup/chat was working after the build/artifact fixes. The
  remaining live issue was tool behavior with Gemma 4 12B's recommended
  `ThinkingMode::Auto` in the GGUF example.
- Failure mode reproduced locally: `chat_gguf_gwen3_gemma4 --tool-demo-only`
  with 12B GGUF Q4 and `ThinkingMode::Auto` spent the 192-token tool-demo budget
  on visible reasoning, returned `tool-call-count: 0`, and failed with
  `model did not call get_weather` after an 87,471.66 ms tool-generation turn.
- Fix 1: the GGUF example now keeps normal chat on the selected model's
  recommended thinking mode, but defaults the tool-demo thinking mode to
  `ThinkingMode::Disabled` for `ChatModels::Gemma4_12B_Gguf` unless the caller
  explicitly passes `--thinking=`.
- Fix 2: the llama.cpp OpenAI-compatible response parser now strips Gemma 4's
  empty `<|channel>thought
<channel|>` marker from parsed response content. Two
  backend parser tests cover bare tool-call content and final-answer content.
- Verified default command, with no explicit `--thinking=off`:

```sh
PATH=/tmp/motlie-cmake/usr/bin:$PATH LD_LIBRARY_PATH=/tmp/motlie-cmake/usr/lib/x86_64-linux-gnu BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include -isystem /usr/include/x86_64-linux-gnu -isystem /usr/include'   cargo run -p motlie-models --no-default-features --features model-gemma4-12b-gguf --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_12b_gguf --precision=q4 --tool-demo-only 'What is Rust? Then calculate the average temperature for Seattle, Portland, and San Francisco.'
```

- Verified output shape: `thinking: Auto`, `tool-demo-thinking: Disabled`, Q4
  selected as `GGUF Q4_K_M`, startup 6,961 ms, startup RSS 12,440.9 MiB.
- Verified tool sequence: `get_weather` for Seattle, Portland, and San
  Francisco, then `evaluate_math_expression` with expression
  `(72.0 + 68.0 + 64.0) / 3.0`, then clean final response:
  `The average current temperature for Seattle, Portland, and San Francisco is
  68.0 degrees Fahrenheit.`
- Tool loop latency on this CPU-only host was slow but deterministic enough for a
  smoke: round latencies were about 46.5s, 52.1s, 57.4s, 66.7s, and final 71.4s.
  Runtime metrics after the tool demo reported 5 requests, 2,602 prompt tokens,
  139 generated tokens, final RSS 12,483.0 MiB, and peak resident bytes
  24,553,984,000 with no process failure.
- Validation commands passing after the fix:
  `cargo test -p motlie-model-llama-cpp openai_response_json -- --nocapture`,
  `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-gguf --example chat_gguf_gwen3_gemma4`, and the live Q4
  `--tool-demo-only` command above.
- Remaining M1 work is now performance/eval reporting plus DGX/full-safetensors
  validation, not basic GGUF viability.

References:

- Google launch blog:
  https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/
- Google developer guide:
  https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
- Official HF model:
  https://huggingface.co/google/gemma-4-12B-it
- ggml-org GGUF:
  https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF
- Unsloth GGUF:
  https://huggingface.co/unsloth/gemma-4-12b-it-GGUF
- mistral.rs supported models:
  https://ericlbuehler.github.io/mistral.rs/reference/supported-models/

## Motlie Fit

### Chat

Chat is feasible.

Safetensors can reuse the `MistralMultimodalSpec` and
`MistralMultimodalHandle` pattern from `gemma4_e2b.rs` and `gemma4_e4b.rs`.
The initial Motlie capability should be text plus image chat only, because
Motlie does not yet model audio/video chat content.

GGUF can reuse the `LlamaCppTextSpec` and `LlamaCppTextHandle` pattern from
`gemma4_e2b_gguf.rs` and `gemma4_e4b_gguf.rs`. The current wrapper is
text-only even if the GGUF artifact itself is multimodal, so the GGUF variants can
cover text chat, completion, and tool-use examples but cannot validate image,
audio, video, or ASR.

### Tool Use

Tool use is feasible, but must remain smoke-gated per backend and artifact.

The public API should stay exactly the existing Motlie tool contract:

- callers pass `ToolSpec` and optional `ToolChoice` on `ChatRequest`
- the model returns structured `ToolCall` values on `ChatResponse`
- callers execute tools themselves through `ToolList` or their own dispatcher
- tool-result turns are replayed as `ChatRole::Tool`

Gemma 4 E2B/E4B already validate this pattern in Motlie. Gemma 4 12B should
treat tool use as accepted only after a multi-round tool loop passes on the
selected 12B artifact. Do not infer 12B capability from E2B/E4B alone, even
when the descriptor follows existing GGUF capability patterns.

### ASR

ASR is in scope as a later milestone, not a stretch goal. It can be gated on
PR #387 (`Integrate feature/models (models=SoT model stack)`) because that PR
sets the model stack baseline this work should build on.

Gemma 4 12B can process audio and the official model card includes ASR prompt
guidance. That is enough to justify a required follow-on milestone after the
chat/tool curation slice. It is not enough to immediately advertise the same
capability as `moonshine_streaming_en`, `sherpa_onnx_streaming_en`, or
`whisper_base_en`.

Reasons:

- Motlie's ASR contract is typed PCM streaming through
  `StreamingTranscriber::open_session`, `TranscriptionSession::ingest`, and
  `finish`.
- Gemma 4 audio ASR is prompt-based chat over bounded audio clips, with a
  published 30 second maximum audio input.
- The model does not natively emit Motlie `TranscriptSegment` timings.
- Partial transcript delivery would be synthetic, not model-native.
- A chat-generation model is likely higher latency and higher memory than
  dedicated ASR backends for ordinary transcription.
- `BundleHandle` currently exposes chat, completion, and embeddings accessors;
  existing ASR bundles use typed `start_typed` paths outside that generic handle
  surface.

Milestone recommendation: do not advertise `CapabilityKind::Transcription` in
the first chat/tool slice. After PR #387 merges, evaluate and implement the ASR
path through audio-chat or a batch/final-only wrapper once audio content plumbing
exists. Advertise `Transcription` only after the selected wrapper satisfies the
Motlie ASR contract honestly.

## Approach Options

### Option A: GGUF Curated Chat And Tool Variant

M1 now includes `google/gemma4_12b_gguf` as a curated GGUF variant for
platform/performance profiles where GGUF startup, memory footprint, or latency
is the better fit. The local CPU-only host could not finish official-safetensors
startup or Q8 ISQ warmup in a useful time window, while GGUF later passed local
chat/tool smoke. This does not replace full-model DGX validation; it gives
Motlie another curated 12B variant to select when the host profile calls for it.
The variant uses standard GGUF quantization from
`unsloth/gemma-4-12b-it-GGUF` to match existing Gemma GGUF curation.

Proposed descriptor:

| Field | Value |
|-------|-------|
| Selector | `google/gemma4_12b_gguf` |
| Bundle id | `gemma4_12b_gguf` |
| Display name | `Gemma 4 12B-it (GGUF/llama.cpp)` |
| Backend | `BackendKind::LlamaCpp` |
| Format | `CheckpointFormat::Gguf` |
| Feature | `model-gemma4-12b-gguf` |
| Source | `unsloth/gemma-4-12b-it-GGUF` |
| Default quant | Q4_K_M for 16 GB target; Q8_0 optional |
| Capabilities | `Chat` + `Completion` + `ToolUse`; live 12B smoke still required |

Pros:

- Smallest implementation surface.
- Reuses existing `LlamaCppTextSpec` and artifact resolver patterns.
- Best fit on hosts where full precision is too slow or too heavy for the target latency/resource budget.
- Fast startup compared with safetensors plus runtime ISQ.
- Adds a deployable profile for smaller or latency-sensitive deployments.

Cons:

- Motlie would not expose the model's image/audio/video inputs through this
  backend at first.
- ASR cannot be tested through Motlie's llama.cpp wrapper without new
  multimodal/audio support.
- GGUF source choice needs review: `ggml-org` is simpler; Unsloth matches
  current Motlie Gemma pattern and has richer quant/audio instructions.

### Option B: Official Safetensors Full-Model Variant

Add `google/gemma4_12b` using `google/gemma-4-12B-it` and the existing
`mistral.rs` multimodal backend. This should be selected when the target
deployment hosts can run the full model, such as DGX Spark or David's larger
local hosts, and when the workload benefits from the official safetensors
artifact and future multimodal expansion.

Proposed descriptor:

| Field | Value |
|-------|-------|
| Selector | `google/gemma4_12b` |
| Bundle id | `gemma4_12b` |
| Display name | `Gemma 4 12B-it` |
| Backend | `BackendKind::MistralRs` |
| Format | `CheckpointFormat::Safetensors` |
| Feature | `model-gemma4-12b` |
| Source | `google/gemma-4-12B-it` |
| Default quant | Full precision (`None`); ISQ Q4/Q8 explicit overrides only |
| Capabilities | `Chat` + `Vision`; add `ToolUse` after smoke |

Pros:

- Uses official Google weights directly.
- Closest to current Gemma 4 E2B/E4B multimodal curation.
- Best stepping stone toward image and later audio/video support inside Motlie.
- Lets Motlie keep the existing `MistralMultimodalAdapter` layering.
- Preserves the full model by default. In Motlie terms, safetensors is not
  inherently quantized: `StartOptions { quantization: None }` resolves to full
  precision when the bundle spec uses
  `QuantizationSupport::without_recommended([Four, Eight])`.
- Better reflects the available host fleet if DGX Spark or equivalent machines
  are expected validation targets.

Cons:

- Larger artifact download and higher memory footprint than GGUF.
- If callers request ISQ Q4/Q8, startup pays the runtime quantization cost.
- Motlie currently supports only text/image content on this path.
- Audio/video would still require `ContentPart::Audio` / `ContentPart::Video`
  design and backend mapping.
- Requires upgrading or pinning the locked `mistralrs 0.8.1` dependency to a
  compatible upstream revision. Current M1 work pins revision `47ec459c` until
  a tagged release includes the same Gemma 4 unified support.

### Option C: True Multimodal GGUF Path

Add or adapt a llama.cpp multimodal path that can send OpenAI-compatible
`image_url` and `input_audio` content blocks to recent llama.cpp, then curate
the 12B GGUF as a multimodal bundle.

Pros:

- Most aligned with the 16 GB local target and the new 12B audio story.
- Recent GGUF repos claim stock llama.cpp can run text, image, and audio with
  automatic `mmproj` handling.
- Could support the ASR milestone without paying safetensors startup cost.

Cons:

- Current Rust `llama-cpp-2` wrapper in Motlie is text-only.
- The upstream multimodal API surface may require `libmtmd` bindings or an
  external `llama-server` integration, both larger than normal curation.
- Would be a backend design project, not a simple curated bundle addition.
- `Transcription` still needs a wrapper that maps chat output into
  `TranscriptionUpdate` without real timestamps or partials.

## Recommendation

Use a variant-based M1 plan:

1. Curate official safetensors on `mistral.rs` as the full-model variant for
   DGX Spark or another full-model CUDA/unified-memory host. This is the path
   that can eventually exercise Gemma 4 12B's broader multimodal surface. It
   defaults to full precision; ISQ Q4/Q8 remain explicit operator overrides.
2. Curate the `google/gemma4_12b_gguf` llama.cpp/GGUF variant now. Local testing
   showed the 28 GiB CPU-only host could not reach safetensors startup or Q8 ISQ
   warmup completion in a practical window. The GGUF variant should use Q4_K_M
   by default and support Q8_0.
3. Keep capability claims artifact-specific. Safetensors can represent the
   full-model direction but still needs DGX chat, image, tool-loop, and
   performance validation. GGUF can cover text chat, completion, and tool-use
   examples through Motlie's current text-only llama.cpp wrapper, but it cannot
   answer ASR feasibility.
4. Treat ASR as a later required milestone gated on PR #387 merge, not as a
   stretch goal. The likely first shape is buffered/final-only audio-chat over
   bounded clips, but it must not advertise `CapabilityKind::Transcription`
   until the wrapper satisfies Motlie's typed ASR contract honestly.
5. Keep dedicated ASR models as the production recommendation for transcription
   unless Gemma 4 12B beats them on a Motlie validation set for David's target
   workflow.

This means M1 should land the selected 12B variants with explicit host/profile
selection guidance: safetensors for full-model hosts, standard GGUF for local or
resource-sensitive chat/tool profiles, and QAT Q4_0 GGUF where that profile wins
on measured startup, latency, memory, or quality. M2 owns the ASR
research and Telnyx audio mapping. M3 owns the final advertised-capability
acceptance gate with examples and eval results.

Variant selection note, 2026-06-06 00:51 PDT by @gemma4-cdx: the generic
`Catalog::resolve_model(gemma4_12b, LlamaCpp, Gguf)` path has only logical model,
backend, and checkpoint-format dimensions, so it cannot distinguish standard
Unsloth GGUF from Google QAT Q4_0 GGUF. It resolves to the standard GGUF variant
by registration order. Callers that need QAT must use the exact selector or
bundle id: `google/gemma4_12b_qat_q4_0_gguf` /
`gemma4_12b_qat_q4_0_gguf`. Runtime local-only cache resolution is now guarded
by exact root GGUF filenames for the selected variant, so a QAT-only cache is not
accepted for the standard GGUF variant and a standard-only cache is not accepted
for the QAT variant.

DGX validation update, 2026-06-06 10:28 PDT by @gemma4-cdx: codex-398-dgx-rv validated PR #398 head `b257d7bb` on DGX GB10 with CUDA/flash-attn. The two llama.cpp GGUF live paths passed the weather/math tool-use smoke: standard Unsloth Q4_K_M completed the exact five-step sequence at about 24 generated tokens/sec, and Google QAT Q4_0 completed the same sequence at about 22 generated tokens/sec; both returned the correct 68F final answer. The official full safetensors variant `google/gemma4_12b` is GPU-runnable but not live-accepted: it built and loaded, then `bench_chat` reported `mistralrs-gen-tps: 0`, about 6% GPU utilization, 81s startup, and 246s warmup. The live tool-use smoke failed in round 3 with no tool call and a garbled text answer (`.The average temperature for Seattle is 23.0.`), then failed the harness with `Error: model did not call evaluate_math_expression`. This is a real mistral.rs Gemma4 full-generation defect, not a hardware-capacity limit. Until fixed, GGUF Q4_K_M and QAT Q4_0 are the working M1 live chat/tool-use paths; the full safetensors variant remains build/load traceability plus future multimodal/ASR direction, blocked for live chat/tool-use acceptance.

Initial investigation note, 2026-06-06 10:28 PDT by @gemma4-cdx: Motlie already sends tool-bearing mistral.rs chat requests through the shared template-compatible adapter, includes tools in `send_chat_request`, replays assistant tool calls and tool results, and defaults tool requests to `enable_thinking = Some(false)` unless the caller opts in. The observed failure occurs after successful model load and partial generation, so the next code investigation should focus on the mistral.rs `MultimodalModelBuilder`/Gemma4 loader and generation path pinned at revision `47ec459cbd6d5b0d6c9035bb79d8cf1e37ee14a0`, plus regression smoke for E2B/E4B under the same CUDA/flash-attn builder swap.

## Tracking Issues

Parent issue: #388

- M1: #389 - Full-precision safetensors chat/tool curation and performance evaluation.
- M2: #390 - ASR audio slicing research and Telnyx audio mapping, gated on PR #387.
- M3: #391 - Full acceptance gate for all advertised capabilities and examples.

## Proposed Curation Plan

### Phase 0: Review This Design

- Confirm the host profile where Option B should be selected, starting with DGX Spark / full-model hosts.
- Upgrade `mistralrs` before the 12B safetensors implementation and resolve any
  adapter/API fallout.
- Review the GGUF variant profile after the safetensors startup diagnostics.
- Confirm the standard GGUF source choice of `unsloth/gemma-4-12b-it-GGUF`, which
  matches the existing Gemma GGUF bundles.
- Confirm target validation host and minimum acceptable latency for full
  precision safetensors.
- Track ASR as a later milestone gated on PR #387 merge; decide whether the
  milestone target is batch/final-only transcription first or full
  `StreamingTranscriber` support.

### Phase 1: Descriptor And Spec Design

Safetensors full-model variant:

- Upgrade `mistralrs` from the locked `0.8.1` version to the compatible current
  line needed for Gemma 4 12B, then rerun existing Gemma 4 E2B/E4B chat/tool
  tests to catch regressions.
- Add `model-gemma4-12b` feature.
- Add `ChatModels::Gemma4_12B`.
- Add `MistralMultimodalSpec::gemma4_12b`.
- Add `libs/models/src/chat/gemma4_12b.rs`.
- Include official sidecars: chat template, tokenizer, processor or
  preprocessor config, safetensors shards and index.
- Use full precision as the default by declaring supported ISQ Q4/Q8 without a
  recommended quantization. Operators can still request Q4/Q8 through
  `StartOptions`.
- Keep capabilities at `Chat` + `Vision` until tool smoke passes.

GGUF variant path, active for M1 local validation:

- Add `model-gemma4-12b-gguf` feature.
- Add `ChatModels::Gemma4_12B_Gguf`.
- Add `LlamaCppTextSpec::gemma4_12b`.
- Add `libs/models/src/chat/gemma4_12b_gguf.rs`.
- Use Q4_K_M recommended, Q8_0 supported.
- Follow existing Gemma GGUF capability patterns for `Chat`, `Completion`, and
  `ToolUse`; mark live 12B smoke and performance results as the M1 acceptance
  gate.

### Phase 2: Tool Use Validation

- Reuse `libs/models/examples/chat_tool_binding`.
- Extend `chat_multimodal_gemma4` for the safetensors full-model variant, or add a
  12B-specific example if the current E2B example remains intentionally narrow.
- Extend `chat_gguf_gwen3_gemma4` for the active GGUF variant selectors.
- Run the existing weather/math multi-round tool loop.
- Verify:
  - tool definitions reach the template
  - assistant tool calls parse into `ChatResponse.tool_calls`
  - tool-result turns replay with call id and name
  - thinking defaults do not hide tool calls in reasoning content
  - final answer is produced after tool results

Only then count `ToolUse` as accepted for that exact backend/artifact in M1/M3 results.

### Phase 3: Chat Validation

- Text-only chat smoke.
- Multi-turn chat smoke.
- System prompt and recommended sampling smoke.
- For safetensors only: image plus text smoke.
- Record startup latency, resident memory, generation tok/s, and any backend
  warnings.

Target host baseline:

- Minimum: 16 GB VRAM or unified memory, because that is Google's published
  local target.
- Preferred Motlie host for reproducible validation: David-provided host, or
  DGX Spark GB10 if matching prior backend docs.
- Do not infer 12B latency from E2B/E4B. Use published file sizes only for
  memory feasibility until live smoke exists.

### Phase 4: ASR Later Milestone

Start after PR #387 merges and the chat/tool curation variants have a stable backend baseline.

Possible evaluation paths:

- Safetensors audio-chat path: add `ContentPart::Audio` and map it through
  `mistral.rs` `AudioInput`.
- GGUF audio-chat path: add a new llama.cpp multimodal/server path that sends
  OpenAI-compatible `input_audio`.
- ASR wrapper: buffer up to 30 seconds of 16 kHz mono PCM, prompt
  Gemma 4 12B with the official ASR instruction, and return one final
  `TranscriptionUpdate`.

Validation should compare against `moonshine_streaming_en`,
`sherpa_onnx_streaming_en`, and `whisper_base_en` on the same WAV set:

- word error rate or manually checked transcript correctness
- first-token and final latency
- memory footprint
- behavior on 5s, 15s, and 30s clips
- noisy speech and digits
- whether the model adds prose despite "transcription only" instructions

ASR acceptance bar:

- If Gemma only returns final text with no timings, advertise at most a
  batch/final-only capability, and make that delivery shape explicit.
- If latency or accuracy is worse than existing ASR bundles, keep it as a demo
  only and do not add an ASR selector.
- Do not advertise `Transcription` unless the selected wrapper satisfies the
  typed contract without misleading partials or timestamps.

## Open Questions For David

- Which exact full-model host should be the validation target: DGX Spark GB10
  or another CUDA/unified-memory host?
- Should the first safetensors bundle default to full precision with no
  recommended quantization, as proposed, or should Motlie still recommend ISQ
  for operators who do not specify `StartOptions.quantization`?
- Given the DGX GB10 result that full safetensors builds and loads but fails live generation/tool-use on the mistral.rs path, should `gemma4_12b` remain in #398 as a build/load-only full-model variant pending backend fix, or should it be held out of M1 until live chat/tool-use passes?
- What platform/profile thresholds should drive selection between safetensors,
  standard GGUF, and QAT GGUF: startup time, generation latency, memory, quality,
  or host coverage?
- Please review the M1 standard GGUF source choice of `unsloth/gemma-4-12b-it-GGUF`
  over `ggml-org/gemma-4-12B-it-GGUF` for consistency with the existing Gemma
  GGUF bundles.
- For the ASR milestone, should the first accepted shape be final-only 30
  second batch transcription, or should it require real streaming transcription?

## QAT Q4_0 GGUF Variant Update

@gemma4-cdx 2026-06-05 17:45 PDT: issue #397 adds a Google-published QAT
Q4_0 GGUF variant in parallel with the existing M1 variants. Bundle id
`gemma4_12b_qat_q4_0_gguf`, selector
`google/gemma4_12b_qat_q4_0_gguf`, feature
`model-gemma4-12b-qat-q4-0-gguf`, backend `llama.cpp`, source repo
`google/gemma-4-12B-it-qat-q4_0-gguf`, curated file
`gemma-4-12b-it-qat-q4_0.gguf`. The repo also publishes
`mmproj-gemma-4-12b-it-qat-q4_0.gguf`; Motlie does not include it yet because
the current `llama.cpp` wrapper is text-only, so this variant advertises chat,
completion, and tool use only.

Recommendation after review: try `llama.cpp` first for this Q4_0 artifact. The
published Q4_0 checkpoint is already GGUF and loads directly through the
existing GGUF artifact/cache path. The Mistral backend remains the right path for
full safetensors and future QAT-unquantized experiments, but not for consuming
this GGUF file directly.

Validation on the local CPU host:

- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-qat-q4-0-gguf --lib` passed.
- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-qat-q4-0-gguf --example chat_gguf_gwen3_gemma4` passed.
- `cargo check -p motlie-models --no-default-features --features
  model-gemma4-12b-qat-q4-0-gguf --example bench_chat` passed.
- `motlie-models-download gemma4_12b_qat_q4_0_gguf` downloaded one file from
  snapshot `f6e7774e6148da3b7f201e42ba37cf084c1db35f`.
- `bench_chat --model=gemma4-12b-qat-q4-0-gguf --precision=q4 --iterations=1`
  loaded file type `Q4_0`, file size 6.48 GiB, startup 4.946s, RSS after
  startup 12.6 GiB, one-word warmup 38.8s, measured one-word request 43.6s,
  final RSS 12.5 GiB, peak RSS 18.7 GiB.
- `chat_gguf_gwen3_gemma4 --chat=google/gemma4_12b_qat_q4_0_gguf
  --tool-demo-only` passed with `tool-demo-thinking: Disabled`: Seattle,
  Portland, San Francisco weather calls plus math average call, final response
  "The average current temperature for Seattle, Portland, and San Francisco is
  68.0 degrees Fahrenheit.", startup 5.2s, final RSS 12.4 GiB, peak resident
  bytes 24.45 GB.

Workspace note: `cargo fmt --check` remains blocked by unrelated workspace
state (`examples/vector2/app/benchmark.rs` missing) and reports formatting drift
in unrelated files, so formatting validation was scoped to compile checks for
the touched targets.
