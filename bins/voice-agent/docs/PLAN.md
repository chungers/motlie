# Plan: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial implementation plan for repo-local voice agent skills. |
| 2026-04-22 | @codex-tts | Implemented the shared runtime and `voice-*` skills, then validated them with `bash -n`, optimized build-or-reuse, and a Piper -> Whisper WAV smoke round trip. |
| 2026-04-22 | @codex-tts | Added first-run interactive endpoint bootstrap, qwen3 reference-voice support, and the initial `jarvis.wav` reference asset. |
| 2026-04-23 | @codex-tts | Added the typed `bins/voice-agent` orchestrator, switched the skill wrappers to prefer installed platform binaries under each namespaced `voice/<skill>/bin/`, simplified the runtime to heuristic-first `release` and CUDA auto-preference, and fixed `motlie-voice` to decode the 32-bit integer PCM WAV format seen from `motliehost` SoX captures. |
| 2026-04-23 | @codex-tts | Removed the old setup/config runtime path, made local audio the runtime default with explicit `ssh:<host>` for remote endpoints, and documented that the agent must ask the human when the endpoint location is ambiguous. |
| 2026-04-23 | @codex-tts | Clarified that the skill discovers backend and endpoint details progressively through the conversation with the human rather than expecting predeclared config. |
| 2026-04-23 | @codex-tts | Added the consolidated skills README with example human prompts, example agent responses, operational QA patterns, and build/source-missing responses. |

## Phase 1. Shared Runtime

- [x] 1.1 Add `bins/voice-agent` as the typed orchestration layer with explicit
  backend and endpoint contracts plus heuristic build/runtime selection. See
  [`DESIGN.md`](./DESIGN.md#typed-runtime).
- [x] 1.2 Remove the old shell config/setup runtime path so the skill has one
  supported execution model. See
  [`DESIGN.md`](./DESIGN.md#endpoint-model).

## Phase 2. Voice Actions

- [x] 2.1 Add qwen3 reference-voice support via `--voice` and
  `--reference-audio`. See
  [`DESIGN.md`](./DESIGN.md#reference-voices).

## Phase 3. Skill Wrappers

- [x] 3.1 Add `voice/speak` skill with wrapper script and concise usage
  guidance.
- [x] 3.2 Add `voice/listen` skill with wrapper script and concise usage
  guidance.
- [x] 3.3 Add `voice/turn` skill with wrapper script and concise usage
  guidance.
- [x] 3.4 Switch the wrappers to build `voice-agent` in `release` and execute
  the built binary directly. See
  [`DESIGN.md`](./DESIGN.md#typed-runtime).
- [x] 3.5 Install platform-scoped `voice-agent` binaries under each
  `.agents/skills/voice/<skill>/bin/` directory and prefer those installed
  binaries at runtime. See [`DESIGN.md`](./DESIGN.md#typed-runtime).
- [x] 3.6 Prefer the most optimized installed binary flavor at runtime:
  `-cuda` first on CUDA-ready hosts, then `-cpu`, with legacy names as
  fallback. See
  [`DESIGN.md`](./DESIGN.md#typed-runtime).
- [x] 3.7 Make the skill docs explicit that the agent asks the human when
  local vs remote audio routing is ambiguous, and then passes `ssh:<host>` for
  remote endpoints. See
  [`DESIGN.md`](./DESIGN.md#endpoint-model).
- [x] 3.8 Make the skill docs explicit that backend and endpoint details are
  discovered progressively through the conversation with the human. See
  [`DESIGN.md`](./DESIGN.md#endpoint-model).

## Phase 4. Validation

- [x] 4.1 `bash -n` all new shell scripts.
- [x] 4.2 Smoke-test build-or-reuse on at least one TTS backend and one ASR
  backend.
- [x] 4.3 Smoke-test one real speech round trip:
  - `voice_speak.sh --backend piper --wav /tmp/voice-skill.wav`
  - `voice_listen.sh --backend whisper --wav /tmp/voice-skill.wav`
- [x] 4.4 If CUDA is available, confirm `ensure_examples.sh` selects the CUDA
  feature set on this host.
- [x] 4.5 Validate the live `motliehost` capture path after extending the WAV
  decoder to accept SoX/macOS 32-bit integer PCM.
- [x] 4.6 Validate the no-config path:
  - local default when no endpoint is provided
  - explicit remote with `ssh:<host>`

Validation note:

- 2026-04-22 @codex-tts -- `voice/speak` generated `/tmp/voice-skill.wav` as
  valid `22050 Hz` mono PCM WAV from the Piper backend, and `voice/listen`
  transcribed it through Whisper as: `A low from the voice, speak skill.`
- 2026-04-22 @codex-tts -- `ensure_examples.sh piper whisper` selected the
  CUDA feature set automatically on this host:
  `model-piper-en-us-ljspeech-medium,model-whisper-base-en,piper-cuda,whisper-cpp-cuda`.
- 2026-04-23 @codex-tts -- `cargo clippy -p voice-agent --all-targets -- -D warnings`
  passed after the typed orchestrator landed.
- 2026-04-23 @codex-tts -- `cargo test -p motlie-voice` and
  `cargo clippy -p motlie-voice --all-targets -- -D warnings` passed after
  adding 32-bit integer PCM decode support for live Motlie SoX captures.
- 2026-04-23 @codex-tts -- `cargo run -p voice-agent -- listen --backend whisper --wav /tmp/motliehost-debug.wav --quiet`
  transcribed the captured Motlie debug WAV as `you`, and the live
  `voice/listen` skill path returned the same transcript from a fresh
  `motliehost` SSH capture.
- 2026-04-23 @codex-tts -- the thin skill wrappers now install and execute
  `voice-agent-<os>-<arch>-<profile>-<flavor>` from the matching
  `.agents/skills/voice/<skill>/bin/`, preferring `-cuda` on CUDA-ready hosts and matching the intended
  distribution layout for prebuilt skill binaries.
- 2026-04-23 @codex-tts -- after the simplification, both
  `./.agents/skills/voice/listen/scripts/run.sh --backend whisper --wav /tmp/motliehost-debug.wav --quiet`
  and `./.agents/skills/voice/listen/scripts/run.sh --backend whisper --endpoint ssh:motliehost --seconds 5 --quiet`
  ran without artifact-root or endpoint env overrides and returned `you`.
