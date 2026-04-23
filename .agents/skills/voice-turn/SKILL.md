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
- builds and installs the typed `voice-agent` binary in `release` mode when missing
- `voice-agent` then builds or reuses the optimized speech example binaries

Shared config lives in:

- `.agents/voice/voice.env.example`
- `.agents/voice/voice.env`

If endpoint config is missing and the script is running interactively, the voice
runtime prompts once for the missing values and stores them in
`.agents/voice/voice.env`.

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

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
