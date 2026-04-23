---
name: voice-turn
description: Run a single spoken interaction turn with Motlie voice tooling. Use when the agent should speak a prompt to the human and then capture a spoken reply through the configured playback and capture endpoints.
---

# Voice Turn

Use this skill for a single spoken turn:

1. speak a prompt
2. capture the response
3. return the transcript

Default behavior:

- prefers an installed platform binary from `.agents/skills/bin/`
- builds and installs the most optimized host binary in `release` mode when missing
- `voice-agent` then builds or reuses the optimized speech example binaries
- prefers CUDA automatically on the current host when available

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

Agent decision rule:

- discover runtime details progressively through the conversation with the human
- if the user says `say with qwen3`, use `--tts-backend qwen3cpp`; otherwise default to `piper`
- if the user says `listen with sherpa` or `listen with moonshine`, use that ASR backend; otherwise default to `whisper`
- if local playback/capture commands exist, try local first
- ask the human whether speak/listen actually worked locally
- if local playback or capture did not work, ask the human whether to use a remote SSH host
- when remote is requested, pass `--playback-endpoint ssh:<host>` and/or `--capture-endpoint ssh:<host>`
- if the wrapper says it is building the optimized binary, tell the human to wait

If the human asks operational questions during the turn:

- "how do I hear you?" means explain the `voice-speak` path:
  - local speaker by default
  - remote playback over SSH with `ssh:<host>` if needed
  - Homebrew `sox` on macOS provides `play`
- "how do you hear me?" means explain the `voice-listen` path:
  - local mic by default
  - remote mic over SSH with `ssh:<host>` if needed
  - Homebrew `sox` on macOS provides `rec`

Example:

```bash
.agents/skills/voice-turn/scripts/run.sh \
  --tts-backend piper \
  --asr-backend whisper \
  --prompt "Please say your status update after the tone." \
  --seconds 8
```

```bash
.agents/skills/voice-turn/scripts/run.sh \
  --tts-backend qwen3cpp \
  --asr-backend whisper \
  --voice jarvis \
  --prompt "Please give me your status update after the tone." \
  --seconds 8
```

Use this skill for turn-based voice interactions, not full-duplex streaming.
