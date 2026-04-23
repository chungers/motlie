---
name: voice-listen
description: Capture human speech into Motlie ASR using the repo-local voice runtime. Use when the agent should record audio from a local or remote microphone endpoint and transcribe it with Whisper, sherpa-onnx, or Moonshine.
---

# Voice Listen

Use this skill when the agent needs spoken input from a person.

Default behavior:

- prefers an installed platform binary from `.agents/skills/bin/`
- builds and installs the most optimized host binary in `release` mode when missing
- `voice-agent` then builds or reuses the optimized release ASR example binary
- prefers CUDA automatically on the current host when available
- captures audio from the local microphone by default
- writes the transcript to stdout

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

Agent decision rule:

- discover runtime details progressively through the conversation with the human
- if the user says `listen with sherpa` or `listen with moonshine`, use that backend
- otherwise default to `--backend whisper`
- if a local recording command exists, try local first
- after local capture, ask the human whether the skill actually heard/captured them
- if local capture did not work, ask:
  - `Should I listen on a remote host over SSH? If so, what host should I use?`
- when the user says remote, pass `--endpoint ssh:<host>`
- if the wrapper says it is building the optimized binary, tell the human to wait

If the human asks "how do you hear me?" answer in this shape:

- by default I try the local microphone on the machine running the skill
- if the microphone is on another machine, I can capture over SSH with `--endpoint ssh:<host>`
- on macOS the simplest remote capture path is Homebrew `sox`, which provides `rec`
- install command on macOS:
  - `brew install sox`
- the remote capture command the runtime expects is effectively:
  - `/opt/homebrew/bin/rec -q -t wav -`
- if local capture did not work, ask whether they are on a remote machine and what SSH host to use

Examples:

```bash
.agents/skills/voice-listen/scripts/run.sh --backend whisper --seconds 8
```

```bash
.agents/skills/voice-listen/scripts/run.sh \
  --backend whisper \
  --endpoint ssh:motliehost \
  --seconds 8
```

```bash
.agents/skills/voice-listen/scripts/run.sh --backend sherpa --seconds 8 --partials
```

For testing from an existing WAV instead of a microphone:

```bash
.agents/skills/voice-listen/scripts/run.sh --backend whisper --wav /tmp/motlie-voice.wav
```
