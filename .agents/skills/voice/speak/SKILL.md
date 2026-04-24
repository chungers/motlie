---
name: voice/speak
description: Speak text aloud to a human with Motlie TTS using the repo-local voice runtime. Use when the agent should convert text into audible speech through Piper or qwen3-tts.cpp, either on the local host or over SSH to a named playback endpoint.
---

# Voice Speak

Use this skill when spoken output is the goal.

Default behavior:

- prefers an installed platform binary from `.agents/skills/voice/speak/bin/`
- builds and installs the most optimized host binary in `release` mode when missing
- `voice-agent` runs the typed Motlie TTS backends directly
- bootstraps missing model weights into `.agents/skills/voice/artifacts/hf-cache/`
- repopulates shared ORT runtime sidecars into `.agents/skills/voice/lib/<os>-<arch>/` when the full repo is present and ONNX Runtime is already installed on the host
- prefers CUDA automatically on the current host when available
- sends WAV output to the local playback device by default
- supports qwen3 voice cloning via `--voice <alias>` or `--reference-audio <path>`

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

Prerequisites:

- macOS playback host:
  - `brew install sox`
- build/run host for ONNX-backed `voice-agent`:
  - macOS: `brew install onnxruntime`
  - Linux: provide ONNX Runtime shared libraries, or build ONNX Runtime from source with `--build_shared_lib`

Agent decision rule:

- discover runtime details progressively through the conversation with the human
- if the user says "say with qwen3", use `--backend qwen3cpp`
- otherwise default to `--backend piper`
- if a local playback command exists, try local first
- after local playback, ask the human whether they heard it
- if local playback did not work, ask:
  - `Should I send audio to a remote host over SSH? If so, what host should I use?`
- when the user says remote, pass `--endpoint ssh:<host>`
- if the wrapper says it is building the optimized binary, tell the human to wait
- if the wrapper reports missing ONNX Runtime, tell the human to install it on the build/run host and then retry
- if the wrapper reports missing `play` on the Mac, tell the human to install `sox` there and then retry

If the human asks "how do I hear you?" answer in this shape:

- by default I try the local playback device on the machine running the skill
- if local playback is not what they want, I can send WAV audio over SSH with `--endpoint ssh:<host>`
- on macOS the simplest remote playback path is Homebrew `sox`, which provides `play`
- install command on macOS:
  - `brew install sox`
- if I need to build or run the ONNX-backed voice runtime on a Mac host, I also need:
  - `brew install onnxruntime`
- the remote playback command the runtime expects is effectively:
  - `/opt/homebrew/bin/play -t wav -`
- if local playback did not work, ask whether they want remote playback and what SSH host to use

Examples:

```bash
.agents/skills/voice/speak/scripts/run.sh --backend piper --text "Hello from Motlie."
```

```bash
.agents/skills/voice/speak/scripts/run.sh \
  --backend piper \
  --endpoint ssh:motliehost \
  --text "Hello from Motlie."
```

```bash
printf '%s\n' "Hello from qwen3-tts.cpp." \
| .agents/skills/voice/speak/scripts/run.sh --backend qwen3cpp
```

```bash
.agents/skills/voice/speak/scripts/run.sh \
  --backend qwen3cpp \
  --voice jarvis \
  --text "Nothing new shipping."
```

If you need a file instead of live playback:

```bash
.agents/skills/voice/speak/scripts/run.sh \
  --backend piper \
  --text "Hello from Motlie." \
  --wav /tmp/motlie-voice.wav
```
