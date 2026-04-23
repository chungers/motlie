# Design: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial design for repo-local voice agent skills built on the existing TTS/ASR examples. |
| 2026-04-22 | @codex-tts | Renamed the skill surface to `voice-speak`, `voice-listen`, and `voice-turn` and implemented the shared shell runtime under `.agents/voice/`. |

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

### Non-Functional

1. Shared shell logic should live in one place.
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
│   └── voice-turn/
└── voice/
    ├── voice.env.example
    └── scripts/
```

The shared runtime lives under `.agents/voice/`. Each skill only contains:

- `SKILL.md`
- a tiny `scripts/run.sh` wrapper
- optional references

### Shared Runtime

`.agents/voice/scripts/common.sh` owns:

- repo-root discovery
- endpoint config loading
- artifact-root defaults
- optimized build selection
- backend-to-example mapping

`.agents/voice/scripts/ensure_examples.sh` owns:

- build-or-reuse of `motlie-models` example binaries
- `release` by default
- CUDA feature selection when available on the current host
- qwen3-tts.cpp submodule initialization when needed

`.agents/voice/scripts/voice_speak.sh` owns:

- text input from `--text` or stdin
- local playback, remote SSH playback, or direct `.wav` file output

`.agents/voice/scripts/voice_listen.sh` owns:

- microphone capture from a local or remote endpoint
- optional fixed-duration capture window
- optional `--wav` input bypass for testing

`.agents/voice/scripts/voice_turn.sh` composes:

- TTS prompt to the playback endpoint
- ASR capture from the capture endpoint

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

Cons:

- larger implementation surface
- duplicates the purpose of the existing example binaries
- slower path for initial testing

### 3. Shared shell runtime plus thin skills

Pros:

- reuses the existing example binaries directly
- easy to inspect and test
- keeps skills concise

Cons:

- shell scripts are less structured than a dedicated Rust tool

This third option is the chosen design for the first slice.
