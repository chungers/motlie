# Design: Voice Agent Skills

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-22 | @codex-tts | Initial design for repo-local voice agent skills built on the existing TTS/ASR examples. |
| 2026-04-22 | @codex-tts | Renamed the skill surface to the `voice/` namespace with `speak`, `listen`, and `turn` subskills. |
| 2026-04-22 | @codex-tts | Added first-run interactive endpoint bootstrap and qwen3 reference-voice support via `--voice` / `--reference-audio`. |
| 2026-04-23 | @codex-tts | Switched the primary orchestration path to a typed `bins/voice-agent` CLI, kept thin shell wrappers for skill entrypoints, installed platform-scoped binaries under each namespaced voice skill `bin/` directory, made the runtime heuristic-first (`release`, CUDA auto-preference, heuristic artifact lookup), and extended the shared WAV decode path to accept the macOS/SoX 32-bit PCM capture format seen on `motliehost`. |
| 2026-04-23 | @codex-tts | Simplified the contract further: local audio by default, explicit `ssh:<host>` for remote endpoints, no setup/config path for normal use, and agent-facing docs that require asking the human when local vs remote is ambiguous. |
| 2026-04-23 | @codex-tts | Clarified that the skill discovers runtime details progressively through the conversation with the human rather than loading a predeclared config upfront. |
| 2026-04-23 | @codex-tts | Added a conversational playbook with example human prompts, example agent responses, and operational answer patterns for `voice/speak`, `voice/listen`, and `voice/turn`. |
| 2026-04-24 | @codex-tts | Hardened the repo-present bootstrap path: generic Linux ORT discovery, no build-host path fallback in the binary, a single repo-present build now seeds all three subskill `bin/` directories, and Piper stays on CPU ORT to avoid the reproducible CUDA shutdown abort seen in live smoke validation. |

## Problem

The repo now has working command-line TTS and ASR example binaries, but agents
still need to know the exact SSH playback/capture commands and host-specific
optimizations to use them safely. That makes voice
interaction fragile and repetitive.

We need repo-local skills and helper scripts that let an agent:

- synthesize speech to a human with the supported TTS backends
- capture human speech from a local or remote endpoint into the supported ASR backends
- compose those pieces into a single `voice/turn` interaction
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
   - remotely over SSH via an explicit `ssh:<host>` endpoint
4. Skills can capture microphone WAV input either:
   - locally on the current host
   - remotely over SSH via an explicit `ssh:<host>` endpoint
5. Skills can build missing binaries in `release` mode and prefer CUDA when
   the current host supports it, without requiring build-profile or
   acceleration env vars.
6. Skills can be used by both Codex and Claude from a shared repo-local
   directory layout.
7. `voice/speak` and `voice/turn` support qwen3 voice cloning through either:
   - a named alias such as `--voice jarvis`
   - a direct file path via `--reference-audio /path/to/file.wav`
8. The agent skill should ask the human when it is unclear whether playback or
   capture should be local or remote.

### Non-Functional

1. Shared orchestration logic should live in one place.
2. Skills should stay thin and declarative.
3. Failure modes should be explicit:
   - missing artifact roots
   - missing submodule
   - missing remote playback/capture command
   - unsupported backend name
4. The user should not need to set env vars for normal use.
5. The first slice must be testable with `bash -n` and at least one real smoke
   path.
6. Runtime details should be discovered progressively through the conversation:
   backend choice, local vs remote, and remote SSH host only when needed.

## Solution

### Layout

```text
.agents/
├── skills/
│   └── voice/
│       ├── README.md
│       ├── common/
│       ├── speak/
│       ├── listen/
│       └── turn/
└── ...
bins/
└── voice-agent/
```

The typed orchestration lives under `bins/voice-agent/`. Each namespaced
subskill only contains:

- `SKILL.md`
- a tiny `scripts/run.sh` wrapper
- optional references

### Typed Runtime

`bins/voice-agent` owns the typed contracts for:

- backend selection
- endpoint resolution
- artifact-root defaults
- optimized build selection
- qwen reference-voice resolution

The thin shell wrappers call:

```text
.agents/skills/voice/common/run_voice_agent.sh
```

That helper:

- prefers an installed platform binary under the subskill-local `bin/`
- installs `voice-agent` into `.agents/skills/voice/<skill>/bin/voice-agent-<os>-<arch>-<profile>-<flavor>` when missing
- seeds all three `voice/{speak,listen,turn}/bin/` directories from a single repo-present build so later skill invocations reuse the installed binary instead of rebuilding
- builds `voice-agent` in `release` mode by default when installation is needed
- executes the installed binary directly rather than using `cargo run`
- chooses the most optimized installed flavor available at runtime:
  - prefer `-cuda` on CUDA-ready hosts
  - otherwise prefer `-cpu`
  - accept older unsuffixed names as a fallback

Within the typed runtime:

- `speak` owns text input, playback, and `.wav` output
- `listen` owns microphone capture or `.wav` input and transcript emission
- `turn` composes the two into one request/response action

The typed runtime also owns:

- direct invocation of the typed Motlie TTS/ASR backends without shelling out to repo example binaries
- CUDA feature selection when available
- qwen3-tts.cpp runtime sidecar handling for installed skill binaries; the current all-backends voice-agent binary links those sidecars eagerly at process startup
- a shared curated artifact cache rooted at `.agents/skills/voice/artifacts/hf-cache/`
- first-use artifact download when the selected backend is missing from that cache
- explicit failure when captured audio is effectively silent, so the agent can guide the human toward the right microphone device or permission fix

### Endpoint Model

The runtime is heuristic-first:

- no endpoint argument means local device
- `--endpoint ssh:<host>` means remote device over SSH
- a bare hostname is also accepted as shorthand for `ssh:<host>`

Local command heuristics:

- playback:
  - `/opt/homebrew/bin/play -t wav -`
  - `play -t wav -`
  - `ffplay -autoexit -nodisp -i pipe:0`
- record:
  - `/opt/homebrew/bin/rec -q -t wav -`
  - `rec -q -t wav -`

Remote command heuristics:

- over SSH, the runtime probes for `play`/`ffplay` and `rec`
- no preconfigured remote command file is required

Agent-facing rule:

- discover runtime details progressively through the conversation
- if the human did not make local vs remote clear, the skill asks first
- once the human says remote, the agent uses `ssh:<host>`

### Reference Voices

The first slice keeps reference voices under:

```text
.agents/skills/voice/speak/references/voices/
```

Alias resolution is intentionally simple:

- `--voice jarvis` -> `.agents/skills/voice/speak/references/voices/jarvis.wav`
- `--voice "voice of jarvis"` normalizes to the same alias
- `--reference-audio /path/to/file.wav` bypasses alias lookup

Only `qwen3cpp` supports reference voices. Piper rejects `--voice` and
`--reference-audio` explicitly.

### Build Policy

Default build/runtime policy:

- always use `release` for skill-managed binaries
- prefer CUDA when `nvidia-smi` is available and succeeds
- otherwise fall back to CPU

That preserves the repo’s optimized path without asking the skill user to tune
build flags manually.

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
