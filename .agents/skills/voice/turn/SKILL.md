---
name: voice/turn
description: Run a single spoken interaction turn with Motlie voice tooling. Use when the agent should speak a prompt to the human and then capture a spoken reply through the configured playback and capture endpoints.
---

# Voice Turn

Use this skill for a single spoken turn:

1. speak a prompt
2. capture the response
3. return the transcript

Default behavior:

- prefers an installed platform binary from `.agents/skills/voice/turn/bin/`
- builds and installs the most optimized host binary in `release` mode when missing
- `voice-agent` runs the typed Motlie TTS and ASR backends directly
- bootstraps missing model weights into `.agents/skills/voice/artifacts/hf-cache/`
- repopulates shared ORT runtime sidecars into `.agents/skills/voice/lib/<os>-<arch>/` when the full repo is present and ONNX Runtime is already installed on the host
- prefers CUDA automatically on the current host when available

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

Prerequisites:

- macOS playback or capture host:
  - `brew install sox`
- build/run host for ONNX-backed `voice-agent`:
  - macOS: `brew install onnxruntime`
  - Linux: provide ONNX Runtime shared libraries, or build ONNX Runtime from source with `--build_shared_lib`

Agent decision rule:

- discover runtime details progressively through the conversation with the human
- if the user says `say with qwen3`, use `--tts-backend qwen3cpp`; otherwise default to `piper`
- after TTS backend selection, rely on the curated capability metadata to determine whether the chosen model is buffered or streaming speech
- treat `piper` and `qwen3cpp` as buffered TTS backends; they produce complete audio before the runtime writes it to playback sinks
- if the user says `listen with sherpa` or `listen with moonshine`, use that ASR backend; otherwise default to `whisper`
- after ASR backend selection, rely on the curated capability metadata to determine whether the chosen model is batch or streaming ASR
- treat `whisper` as batch/final-only ASR and `sherpa`/`moonshine` as streaming ASR backends
- for remote Mac push capture, keep ASR behavior consistent across `whisper`, `sherpa`, and `moonshine`: wait for EOF from Ctrl-C on the Mac and then print one final transcript for the listen half of the turn
- if local playback/capture commands exist, try local first
- ask the human whether speak/listen actually worked locally
- if local playback or capture did not work, ask the human whether to use a remote SSH host
- when remote is requested, pass `--playback-endpoint ssh:<host>` and/or `--capture-endpoint ssh:<host>`
- if the reply microphone is remote on a Mac, prompt for the local Mac Terminal/iTerm2 push flow for the listen half of the turn
- when the human has to run a command, give one short, copy-pasteable command first
- for the remote Mac push helper, only emit validated shell-safe SSH targets and FIFO paths; fail clearly instead of printing a risky command
- only give the longer compatibility fallback if the short command fails
- infer the SSH destination for that push flow from the current host name first and only ask the human if it is unclear or likely unreachable from the Mac
- if the listen side reports silent capture, tell the human the capture host likely has the wrong input device selected or is missing microphone permission
- on macOS, explain that SSH/non-interactive capture may not share Terminal/iTerm2 microphone permission, so reverse tunnels alone do not solve it
- if that happens, switch to a human-run local Mac `rec` command that records locally on the Mac until Ctrl-C, then streams the recorded WAV over SSH into a stable waiting FIFO for the listen half of the turn; the agent should kill any stale listener on that FIFO before starting a new one, run the waiting listen side itself in one live polled session instead of a detached background job, show the human the short Mac command first, poll until EOF arrives, and only offer the fallback command if the short one fails
- if the wrapper says it is building the optimized binary, tell the human to wait
- if the wrapper reports missing dependencies like ONNX Runtime or `sox`, tell the human exactly what to install and then retry

If the human asks operational questions during the turn:

- "how do I hear you?" means explain the `voice/speak` path:
  - local speaker by default
  - remote playback over SSH with `ssh:<host>` if needed
  - Homebrew `sox` on macOS provides `play`
  - if I need to build or run the ONNX-backed voice runtime on a Mac host, I also need `brew install onnxruntime`
- "how do you hear me?" means explain the `voice/listen` path:
  - local mic by default
  - remote mic over SSH with `ssh:<host>` if needed
  - Homebrew `sox` on macOS provides `rec`
  - if I need to build or run the ONNX-backed voice runtime on a Mac host, I also need `brew install onnxruntime`

Example:

```bash
.agents/skills/voice/turn/scripts/run.sh \
  --tts-backend piper \
  --asr-backend whisper \
  --prompt "Please say your status update after the tone." \
  --seconds 8
```

```bash
.agents/skills/voice/turn/scripts/run.sh \
  --tts-backend qwen3cpp \
  --asr-backend whisper \
  --voice jarvis \
  --prompt "Please give me your status update after the tone." \
  --seconds 8
```

Use this skill for turn-based voice interactions, not full-duplex streaming. Even when `sherpa` or `moonshine` are selected, the default turn flow still waits for one final transcript before continuing.
