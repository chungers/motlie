---
name: voice/listen
description: Capture human speech into Motlie ASR using the repo-local voice runtime. Use when the agent should record audio from a local or remote microphone endpoint and transcribe it with Whisper, sherpa-onnx, or Moonshine.
---

# Voice Listen

Use this skill when the agent needs spoken input from a person.

Default behavior:

- prefers an installed platform binary from `.agents/skills/voice/listen/bin/`
- builds and installs the most optimized host binary in `release` mode when missing
- `voice-agent` runs the typed Motlie ASR backends directly
- bootstraps missing model weights into `.agents/skills/voice/artifacts/hf-cache/`
- repopulates shared ORT runtime sidecars into `.agents/skills/voice/lib/<os>-<arch>/` when the full repo is present and ONNX Runtime is already installed on the host
- prefers CUDA automatically on the current host when available
- captures audio from the local microphone by default
- writes the transcript to stdout

Typed orchestrator:

- `bins/voice-agent`

Thin wrapper:

- `scripts/run.sh`

Prerequisites:

- macOS capture host:
  - `brew install sox`
- build/run host for ONNX-backed `voice-agent`:
  - macOS: `brew install onnxruntime`
  - Linux: provide ONNX Runtime shared libraries, or build ONNX Runtime from source with `--build_shared_lib`

Agent decision rule:

- discover runtime details progressively through the conversation with the human
- if the user says `listen with sherpa` or `listen with moonshine`, use that backend
- otherwise default to `--backend whisper`
- for remote Mac push capture, keep backend behavior consistent across `whisper`, `sherpa`, and `moonshine`: wait for EOF from Ctrl-C on the Mac and then print one final transcript
- if a local recording command exists, try local first
- after local capture, ask the human whether the skill actually heard/captured them
- if local capture did not work, ask:
  - `Should I listen on a remote host over SSH? If so, what host should I use?`
- when the user says remote, pass `--endpoint ssh:<host>`
- for the remote Mac push helper, only emit validated shell-safe SSH targets and FIFO paths; fail clearly instead of printing a risky command
- if the microphone is remote on a Mac, prompt for the local Mac Terminal/iTerm2 push flow instead of assuming SSH capture will work
- when the human has to run a command, give one short, copy-pasteable command first
- only give the longer compatibility fallback if the short command fails
- infer the SSH destination for that push flow from the current host name first and only ask the human if it is unclear or likely unreachable from the Mac
- if the wrapper says it is building the optimized binary, tell the human to wait
- if the wrapper reports missing ONNX Runtime, tell the human to install it on the build/run host and then retry
- if the wrapper reports missing `rec` on the Mac, tell the human to install `sox` there and then retry

If the human asks "how do you hear me?" answer in this shape:

- by default I try the local microphone on the machine running the skill
- if the microphone is on another machine, I can capture over SSH with `--endpoint ssh:<host>`
- on macOS the simplest remote capture path is Homebrew `sox`, which provides `rec`
- install command on macOS:
  - `brew install sox`
- if I need to build or run the ONNX-backed voice runtime on a Mac host, I also need:
  - `brew install onnxruntime`
- the remote capture command the runtime expects is effectively:
  - `/opt/homebrew/bin/rec -q -t wav -`
- if local capture did not work, ask whether they are on a remote machine and what SSH host to use
- if the runtime reports silent capture, tell the human the capture host likely has the wrong input device selected or is missing microphone permission
- on macOS, explain that SSH/non-interactive capture may not share Terminal/iTerm2 microphone permission, so reverse tunnels alone do not solve it
- if the microphone is remote on a Mac, or if SSH capture on macOS is silent, ask the human to use the local Mac Terminal/iTerm2 push flow
- use `scripts/prepare_remote_push.sh` to generate the stable FIFO path plus the short Mac command the human should run
- `prepare_remote_push.sh` should kill any stale listener still attached to the fixed FIFO before starting a new one
- the short Mac command records locally until `Ctrl-C`, then streams the recorded WAV over SSH; it does not force sample-rate or channel flags because the device may ignore them anyway
- if the short command fails because `rec` is not at `/opt/homebrew/bin/rec`, then offer the generated fallback command
- run the agent listen side yourself; do not ask the human to run an agent-side listen command
- for the remote Mac push flow, poll until EOF from Ctrl-C arrives, then print one final transcript
- after the listen returns or fails, clean up the FIFO path before the next run

Examples:

```bash
.agents/skills/voice/listen/scripts/run.sh --backend whisper --seconds 8
```

```bash
.agents/skills/voice/listen/scripts/run.sh   --backend whisper   --endpoint ssh:motliehost   --seconds 8
```

```bash
.agents/skills/voice/listen/scripts/run.sh --backend sherpa --seconds 8
```

For testing from an existing WAV instead of a microphone:

```bash
.agents/skills/voice/listen/scripts/run.sh --backend whisper --wav /tmp/motlie-voice.wav
```

macOS SSH-capture fallback:

1. The agent runs:

```bash
.agents/skills/voice/listen/scripts/prepare_remote_push.sh --ssh-target dchung@spark-2f6e
```

2. The agent starts the printed `AGENT_LISTEN_CMD` locally in one live polled session, not as a detached background job, and waits on the FIFO.
3. The human runs the printed short `HUMAN_MAC_CMD` locally in Terminal or iTerm2 on the Mac.
4. Only if that short command fails should the agent provide `HUMAN_MAC_FALLBACK_CMD`.
5. The human presses `Ctrl-C` on the Mac when done recording, which ends the local recording. The command then streams the recorded WAV and sends EOF when that stream finishes.
6. The agent keeps the listen side running and polls for EOF instead of assuming the user is done early.
7. When the stream closes, the waiting listener prints one final transcript and the wrapper removes the FIFO path.
