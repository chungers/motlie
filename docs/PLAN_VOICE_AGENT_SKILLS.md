# Plan: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial implementation plan for repo-local voice agent skills. |
| 2026-04-22 | @codex-tts | Implemented the shared runtime and `voice-*` skills, then validated them with `bash -n`, optimized build-or-reuse, and a Piper -> Whisper WAV smoke round trip. |
| 2026-04-22 | @codex-tts | Added first-run interactive endpoint bootstrap, qwen3 reference-voice support, and the initial `jarvis.wav` reference asset. |
| 2026-04-23 | @codex-tts | Added the typed `bins/voice-agent` orchestrator, switched the skill wrappers to prefer installed platform binaries under `.agents/skills/bin/`, and fixed `motlie-voice` to decode the 32-bit integer PCM WAV format seen from `motliehost` SoX captures. |

## Phase 1. Shared Runtime

- [x] 1.1 Add `.agents/voice/voice.env.example` with endpoint, artifact-root,
  and acceleration defaults. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#endpoint-model).
- [x] 1.2 Add `.agents/voice/scripts/common.sh` with shared helpers for repo
  discovery, env loading, endpoint lookup, and backend metadata. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#shared-runtime).
- [x] 1.3 Add `.agents/voice/scripts/ensure_examples.sh` to build or reuse
  optimized release binaries and enable CUDA when available. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#build-policy).
- [x] 1.4 Add interactive bootstrap for missing endpoint config and persist it
  to `.agents/voice/voice.env`. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#endpoint-model).
- [x] 1.5 Add `bins/voice-agent` as the typed orchestration layer with explicit
  backend, endpoint, build-profile, and acceleration contracts. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#typed-runtime).

## Phase 2. Voice Actions

- [x] 2.1 Add `.agents/voice/scripts/voice_speak.sh` for TTS playback or WAV
  file output. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#shared-runtime).
- [x] 2.2 Add `.agents/voice/scripts/voice_listen.sh` for local/remote mic
  capture into the ASR examples. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#shared-runtime).
- [x] 2.3 Add `.agents/voice/scripts/voice_turn.sh` to compose the two
  actions into one turn. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#solution).
- [x] 2.4 Add qwen3 reference-voice support via `--voice` and
  `--reference-audio`. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#reference-voices).

## Phase 3. Skill Wrappers

- [x] 3.1 Add `voice-speak` skill with wrapper script and concise usage
  guidance.
- [x] 3.2 Add `voice-listen` skill with wrapper script and concise usage
  guidance.
- [x] 3.3 Add `voice-turn` skill with wrapper script and concise usage
  guidance.
- [x] 3.4 Switch the wrappers to build `voice-agent` in the selected profile
  and execute the built binary directly. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#typed-runtime).
- [x] 3.5 Install platform-scoped `voice-agent` binaries under
  `.agents/skills/bin/` and prefer those installed binaries at runtime. See
  [`DESIGN_VOICE_AGENT_SKILLS.md`](./DESIGN_VOICE_AGENT_SKILLS.md#typed-runtime).

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

Validation note:

- 2026-04-22 @codex-tts -- `voice-speak` generated `/tmp/voice-skill.wav` as
  valid `22050 Hz` mono PCM WAV from the Piper backend, and `voice-listen`
  transcribed it through Whisper as: `A low from the voice, speak skill.`
- 2026-04-22 @codex-tts -- `ensure_examples.sh piper whisper` selected the
  auto-CUDA feature set on this host:
  `model-piper-en-us-ljspeech-medium,model-whisper-base-en,piper-cuda,whisper-cpp-cuda`.
- 2026-04-23 @codex-tts -- `cargo clippy -p voice-agent --all-targets -- -D warnings`
  passed after the typed orchestrator landed.
- 2026-04-23 @codex-tts -- `cargo test -p motlie-voice` and
  `cargo clippy -p motlie-voice --all-targets -- -D warnings` passed after
  adding 32-bit integer PCM decode support for live Motlie SoX captures.
- 2026-04-23 @codex-tts -- `cargo run -p voice-agent -- listen --backend whisper --wav /tmp/motliehost-debug.wav --quiet`
  transcribed the captured Motlie debug WAV as `you`, and the live
  `voice-listen` skill path returned the same transcript from a fresh
  `motliehost` SSH capture.
- 2026-04-23 @codex-tts -- the thin skill wrappers now install and execute
  `voice-agent-<os>-<arch>-<profile>` from `.agents/skills/bin/`, matching the
  intended distribution layout for prebuilt skill binaries.
