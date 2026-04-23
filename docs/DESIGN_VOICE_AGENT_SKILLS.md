# Design: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial design for repo-local voice agent skills built on the existing TTS/ASR examples. |
| 2026-04-22 | @codex-tts | Renamed the skill surface to `voice-speak`, `voice-listen`, and `voice-turn` and implemented the shared shell runtime under `.agents/voice/`. |
| 2026-04-22 | @codex-tts | Added first-run interactive endpoint bootstrap and qwen3 reference-voice support via `--voice` / `--reference-audio`. |
| 2026-04-23 | @codex-tts | Switched the primary orchestration path to a typed `bins/voice-agent` CLI, kept thin shell wrappers for skill entrypoints, installed platform-scoped binaries under `.agents/skills/bin/`, and extended the shared WAV decode path to accept the macOS/SoX 32-bit PCM capture format seen on `motliehost`. |

## Problem

The repo now has working command-line TTS and ASR example binaries, but agents
still need to know the exact build flags, artifact roots, SSH playback/capture
commands, and host-specific optimizations to use them safely. That makes voice
interaction fragile and repetitive.

We need repo-local skills and helper scripts that let an agent:

- synthesize speech to a human with the supported TTS backends
- capture human speech from a local or remote endpoint into the supported ASR backends
- compose those pieces into a single voice-turn interaction
- build or reuse optimized release binaries for the platform the agent is
  running on, including CUDA when supported

## Non-Goals

- Full-duplex live voice conversation
- Telephony/RTP/WebSocket transport
- New model backends
- Changes to the typed TTS/ASR library contracts

## Requirements

### Functional

1. Agents can select a supported TTS backend:
   - `tts_piper`
   - `tts_qwen3_tts_cpp`
2. Agents can select a supported ASR backend:
   - `asr_whisper`
   - `asr_sherpa_onnx`
   - `asr_moonshine`
3. Skills can play audio either:
   - locally on the current host
   - remotely over SSH on a named endpoint such as `motliehost`
4. Skills can capture microphone WAV input either:
   - locally on the current host
   - remotely over SSH on a named endpoint
5. Skills can build missing binaries in `release` mode and prefer CUDA when
   the current host supports it.
6. Skills can be used by both Codex and Claude from a shared repo-local
   directory layout.
7. `voice-speak` and `voice-turn` support qwen3 voice cloning through either:
   - a named alias such as `--voice jarvis`
   - a direct file path via `--reference-audio /path/to/file.wav`
8. If endpoint config is missing and the shell is interactive, the runtime
   prompts once and persists the answers into `.agents/voice/voice.env`.

### Non-Functional

1. Shared orchestration logic should live in one place.
2. Skills should stay thin and declarative.
3. Failure modes should be explicit:
   - missing artifact roots
   - missing submodule
   - missing remote playback/capture command
   - unsupported backend name
4. The first slice must be testable with `bash -n` and at least one real smoke
   path.

## Solution

### Layout

```text
.agents/
├── skills/
│   ├── voice-speak/
│   ├── voice-listen/
│   ├── voice-turn/
│   └── bin/
├── voice/
│   ├── voice.env.example
│   └── scripts/
└── ...
bins/
└── voice-agent/
```

The typed orchestration lives under `bins/voice-agent/`. Each skill only contains:

- `SKILL.md`
- a tiny `scripts/run.sh` wrapper
- optional references

### Typed Runtime

`bins/voice-agent` owns the typed contracts for:

- backend selection
- endpoint resolution
- config loading and persistence
- artifact-root defaults
- optimized build selection
- qwen reference-voice resolution

The thin shell wrappers call:

```text
.agents/voice/scripts/run_voice_agent.sh
```

That helper:

- prefers an installed platform binary under `.agents/skills/bin/`
- installs `voice-agent` into `.agents/skills/bin/voice-agent-<os>-<arch>-<profile>` when missing
- builds `voice-agent` in `release` mode by default when installation is needed
- respects `MOTLIE_VOICE_BUILD_PROFILE=debug`
- executes the installed binary directly rather than using `cargo run`

Within the typed runtime:

- `speak` owns text input, playback, and `.wav` output
- `listen` owns microphone capture or `.wav` input and transcript emission
- `turn` composes the two into one request/response action
- `setup` handles first-run interactive config bootstrap

The typed runtime also owns:

- build-or-reuse of `motlie-models` example binaries
- CUDA feature selection when available
- qwen3-tts.cpp submodule initialization when needed

### Endpoint Model

Endpoint config is env-based so the shell runtime does not depend on `jq` or a
JSON parser. The config file is:

```text
.agents/voice/voice.env
```

Example endpoint variables:

```bash
MOTLIE_VOICE_PLAYBACK_ENDPOINT=motliehost
MOTLIE_VOICE_CAPTURE_ENDPOINT=motliehost

MOTLIE_ENDPOINT_MOTLIEHOST_KIND=ssh
MOTLIE_ENDPOINT_MOTLIEHOST_SSH_TARGET=motliehost
MOTLIE_ENDPOINT_MOTLIEHOST_PLAY_CMD=/opt/homebrew/bin/play -t wav -
MOTLIE_ENDPOINT_MOTLIEHOST_RECORD_CMD=/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -
```

This keeps the endpoint contract simple and editable in a terminal.

When `voice.env` is missing or an endpoint field is not yet defined, the typed
runtime can ask for the missing values interactively and append them to
`.agents/voice/voice.env`. That gives the agent a one-time bootstrap path
without relying on external memory.

### Reference Voices

The first slice keeps reference voices under:

```text
artifacts/voice-references/
```

Alias resolution is intentionally simple:

- `--voice jarvis` -> `artifacts/voice-references/jarvis.wav`
- `--voice "voice of jarvis"` normalizes to the same alias
- `--reference-audio /path/to/file.wav` bypasses alias lookup

Only `qwen3cpp` supports reference voices. Piper rejects `--voice` and
`--reference-audio` explicitly.

### Build Policy

Default build profile:

- `release`

Acceleration policy:

- `auto` by default
- use CUDA feature flags when `nvidia-smi` is available and succeeds
- otherwise fall back to CPU release

That preserves the repo’s current optimized path without hardcoding “always
CUDA” or “always CPU”.

## Alternatives Considered

### 1. Hardcode raw shell commands only in `SKILL.md`

Pros:

- fastest to write

Cons:

- duplicates build logic
- impossible to keep endpoint config in sync
- poor failure handling

### 2. Add a dedicated Rust binary for agent voice I/O

Pros:

- stronger typing
- explicit backend/endpoint contracts
- easier validation than stringly-typed shell dispatch

Cons:

- still requires thin shell wrappers for skill entrypoints
- larger implementation surface than shell-only bootstrap

This second option is now the chosen design. The example binaries remain the
speech engines, while `voice-agent` is the typed orchestration layer above
them.
