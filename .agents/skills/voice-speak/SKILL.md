---
name: voice-speak
description: Speak text aloud to a human with Motlie TTS using the repo-local voice runtime. Use when the agent should convert text into audible speech through Piper or qwen3-tts.cpp, either on the local host or over SSH to a named playback endpoint.
---

# Voice Speak

Use this skill when spoken output is the goal.

Default behavior:

- prefers an installed platform binary from `.agents/skills/bin/`
- builds and installs the typed `voice-agent` binary in `release` mode when missing
- `voice-agent` then builds or reuses the optimized release example binary
- prefers CUDA on the current host when available
- sends WAV output to the configured playback endpoint
- supports qwen3 voice cloning via `--voice <alias>` or `--reference-audio <path>`

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

Examples:

```bash
.agents/skills/voice-speak/scripts/run.sh --backend piper --text "Hello from Motlie."
```

```bash
printf '%s\n' "Hello from qwen3-tts.cpp." \
| .agents/skills/voice-speak/scripts/run.sh --backend qwen3cpp
```

```bash
.agents/skills/voice-speak/scripts/run.sh \
  --backend qwen3cpp \
  --voice jarvis \
  --text "Nothing new shipping."
```

If you need a file instead of live playback:

```bash
.agents/skills/voice-speak/scripts/run.sh \
  --backend piper \
  --text "Hello from Motlie." \
  --wav /tmp/motlie-voice.wav
```
