# Plan: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial implementation plan for repo-local voice agent skills. |
| 2026-04-22 | @codex-tts | Implemented the shared runtime and `voice-*` skills, then validated them with `bash -n`, optimized build-or-reuse, and a Piper -> Whisper WAV smoke round trip. |

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

## Phase 3. Skill Wrappers

- [x] 3.1 Add `voice-speak` skill with wrapper script and concise usage
  guidance.
- [x] 3.2 Add `voice-listen` skill with wrapper script and concise usage
  guidance.
- [x] 3.3 Add `voice-turn` skill with wrapper script and concise usage
  guidance.

## Phase 4. Validation

- [x] 4.1 `bash -n` all new shell scripts.
- [x] 4.2 Smoke-test build-or-reuse on at least one TTS backend and one ASR
  backend.
- [x] 4.3 Smoke-test one real speech round trip:
  - `voice_speak.sh --backend piper --wav /tmp/voice-skill.wav`
  - `voice_listen.sh --backend whisper --wav /tmp/voice-skill.wav`
- [x] 4.4 If CUDA is available, confirm `ensure_examples.sh` selects the CUDA
  feature set on this host.

Validation note:

- 2026-04-22 @codex-tts -- `voice-speak` generated `/tmp/voice-skill.wav` as
  valid `22050 Hz` mono PCM WAV from the Piper backend, and `voice-listen`
  transcribed it through Whisper as: `A low from the voice, speak skill.`
- 2026-04-22 @codex-tts -- `ensure_examples.sh piper whisper` selected the
  auto-CUDA feature set on this host:
  `model-piper-en-us-ljspeech-medium,model-whisper-base-en,piper-cuda,whisper-cpp-cuda`.
