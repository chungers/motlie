---
name: voice-listen
description: Capture human speech into Motlie ASR using the repo-local voice runtime. Use when the agent should record audio from a local or remote microphone endpoint and transcribe it with Whisper, sherpa-onnx, or Moonshine.
---

# Voice Listen

Use this skill when the agent needs spoken input from a person.

Default behavior:

- builds or reuses the optimized release ASR example binary
- prefers CUDA on the current host when available
- captures audio from the configured endpoint using `rec -t wav -`
- writes the transcript to stdout

Shared config lives in:

- `.agents/voice/voice.env.example`
- `.agents/voice/voice.env`

Main entrypoint:

- `.agents/voice/scripts/voice_listen.sh`

Thin wrapper:

- `scripts/run.sh`

Examples:

```bash
.agents/skills/voice-listen/scripts/run.sh --backend whisper --seconds 8
```

```bash
.agents/skills/voice-listen/scripts/run.sh --backend sherpa --seconds 8 --partials
```

For testing from an existing WAV instead of a microphone:

```bash
.agents/skills/voice-listen/scripts/run.sh --backend whisper --wav /tmp/motlie-voice.wav
```
