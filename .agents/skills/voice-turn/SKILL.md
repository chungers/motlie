---
name: voice-turn
description: Run a single spoken interaction turn with Motlie voice tooling. Use when the agent should speak a prompt to the human and then capture a spoken reply through the configured playback and capture endpoints.
---

# Voice Turn

Use this skill for a single spoken turn:

1. speak a prompt
2. capture the response
3. return the transcript

Shared config lives in:

- `.agents/voice/voice.env.example`
- `.agents/voice/voice.env`

Main entrypoint:

- `.agents/voice/scripts/voice_turn.sh`

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

Use this skill for turn-based voice interactions, not full-duplex streaming.
