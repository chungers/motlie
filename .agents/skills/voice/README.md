# Voice Skills README

This README is the conversational playbook for the repo-local voice skills:

- `voice/speak`
- `voice/listen`
- `voice/turn`

## Runtime Rules

- prefer shipped platform binaries under each namespaced skill directory:
  - `.agents/skills/voice/speak/bin/`
  - `.agents/skills/voice/listen/bin/`
  - `.agents/skills/voice/turn/bin/`
- one repo-present build should seed all three subskill `bin/` directories; later skill calls should reuse those installed binaries instead of rebuilding
- if no binary is available for this host, build the most optimized one for the host
- the build always uses `release`
- prefer CUDA when the host supports it
- keep Piper on CPU even in CUDA builds for now; the ONNX Runtime CUDA path is unstable on shutdown in the current validation environment
- if the selected backend weights are missing, download them into the shared skill cache:
  - `.agents/skills/voice/artifacts/hf-cache/`
- if the full `motlie` repo is present, repo-based builds should repopulate the skill runtime sidecars:
  - `.agents/skills/voice/lib/<os>-<arch>/` for ONNX Runtime
  - `.agents/skills/voice/{speak,listen,turn}/bin/` for `voice-agent-*` and `libqwen3tts.so*`
- use local audio devices first when they are available
- if local audio does not work, ask the human whether they are remote and what SSH host to use
- if capture succeeds technically but the runtime reports silent audio, treat that as a microphone/device/permission problem on the capture host
- on macOS, reverse tunnels do not fix microphone TCC; if the human is remote on a Mac, prefer the local Mac Terminal/iTerm2 push flow over SSH capture
- if SSH capture on macOS is silent, switch to the local Mac Terminal/iTerm2 push flow automatically
- when preparing that push flow, infer the SSH destination from the current host name first and only ask the human if that destination is unclear or not reachable from the Mac
- when the human has to run a command, prefer one short, copy-pasteable command line first
- keep the longer compatibility fallback command in reserve and only show it if the short command fails
- discover runtime details progressively through the conversation with the human

## Prerequisites

The current `voice-agent` binary is built with all curated voice backends enabled, so the build/run host needs:

- ONNX Runtime shared libraries for the ONNX-backed backends:
  - `piper`
  - `moonshine`
  - `sherpa`
- `qwen3-tts.cpp` runtime sidecars for the current all-backends voice-agent binary:
  - installed into the voice subskill `bin/` directories after a repo-present build

What the human should install:

- macOS playback or capture host:
  - `brew install sox`
- macOS build/run host for ONNX-backed voice-agent:
  - `brew install onnxruntime`
- Linux build/run host for ONNX-backed voice-agent:
  - provide `libonnxruntime.so` and `libonnxruntime.so.1`
  - for CUDA-backed ONNX paths also provide:
    - `libonnxruntime_providers_shared.so`
    - `libonnxruntime_providers_cuda.so`
  - if distro packages are unavailable, build ONNX Runtime from source with `--build_shared_lib`

## Bootstrapping Rules

- treat build and model download as normal first-run bootstrap work
- if the host binary is missing, tell the human you need to build it and ask them to wait
- if the selected model weights are missing, tell the human you need to download them and ask them to wait
- if the ONNX Runtime shared libraries are missing on the build/run host, stop and tell the human what to install before retrying
- if both binary build and model download are needed, tell the human both steps are part of bootstrap:
  - build the optimized binary for this host
  - download the selected backend weights into `.agents/skills/voice/artifacts/hf-cache/`
- if neither a shipped binary nor repo source is present, tell the human exactly that
- after bootstrap, continue with the original speak/listen/turn request instead of stopping

## Backend Defaults

- TTS default: `piper`
- TTS override: if the human says `say with qwen3`, use `qwen3cpp`
- ASR default: `whisper`
- ASR override: if the human says `listen with sherpa` or `listen with moonshine`, use that backend
- for remote Mac push capture, all ASR backends use the same interaction contract: the human presses `Ctrl-C`, the stream closes, and the agent prints one final transcript after `EOF`

## Endpoint Defaults

- default playback: local audio device
- default capture: local microphone
- remote endpoint form: `ssh:<host>`

## Example Human Prompts And Agent Responses

### Speak

Human prompt:

`Say hello to me.`

Agent interpretation:

- use `voice/speak`
- backend: `piper`
- endpoint: local

Example agent response:

`I’m going to try the local speaker first with Piper.`

Human prompt:

`Say this with qwen3: Nothing new shipping.`

Agent interpretation:

- use `voice/speak`
- backend: `qwen3cpp`
- endpoint: local

Example agent response:

`I’m going to say that locally with qwen3-tts.cpp.`

Human prompt:

`Play it on motliehost.`

Agent interpretation:

- use `voice/speak`
- endpoint: `ssh:motliehost`

Example agent response:

`I’ll send the audio to motliehost over SSH.`

Bootstrap example when ORT is missing:

`I need ONNX Runtime on this host before I can build the voice skill. On macOS, install it with 'brew install onnxruntime'. On Linux, I need the ONNX Runtime shared libraries available on this host. After that, I can retry automatically.`

Bootstrap example when sox is missing on the Mac:

`I can send or capture audio on the Mac, but I need sox there first. Please install it with 'brew install sox', then I can retry.`

### Listen

Human prompt:

`Listen to me.`

Agent interpretation:

- use `voice/listen`
- backend: `whisper`
- endpoint: local

Example agent response:

`I’m going to try the local microphone first with Whisper.`

Human prompt:

`Listen with sherpa.`

Agent interpretation:

- use `voice/listen`
- backend: `sherpa`
- endpoint: local

Example agent response:

`I’m going to listen locally with sherpa.`

Human prompt:

`My mic is on motliehost.`

Agent interpretation:

- use `voice/listen`
- endpoint: `ssh:motliehost`

Example agent response:

`I’ll listen from motliehost over SSH.`

Bootstrap example when Whisper weights are missing:

`I need to bootstrap listening on this host first by downloading the Whisper weights into the shared voice artifacts cache. Please wait a moment.`

Bootstrap example when ORT is missing:

`I need ONNX Runtime on this host before I can build or run the ONNX-backed listening backends. Please install it, then I can retry automatically.`

### Turn

Human prompt:

`Ask me for my status and listen to the reply.`

Agent interpretation:

- use `voice/turn`
- TTS backend: `piper`
- ASR backend: `whisper`
- playback: local
- capture: local

Example agent response:

`I’ll use a local voice turn with Piper for speaking and Whisper for listening.`

Human prompt:

`Use qwen3 to ask me and listen with moonshine.`

Agent interpretation:

- use `voice/turn`
- TTS backend: `qwen3cpp`
- ASR backend: `moonshine`

Example agent response:

`I’ll use qwen3 for speech and Moonshine for listening.`

## Progressive Discovery Examples

### Local Or Remote Not Clear

Human prompt:

`Say something to me.`

Good agent follow-up:

`I detected a local playback command on this host. I can try the local speaker first, or send audio to a remote host over SSH. Which do you want?`

If local did not work:

`Local playback may not be the right device. Are you on a remote machine? If so, what SSH host should I use?`

### Listen Fallback

Human prompt:

`Listen to me.`

Good agent follow-up:

`I detected a local recording command on this host. I can try the local microphone first, or listen from a remote host over SSH. Which should I use?`

If the human says their microphone is remote on a Mac, good agent follow-up:

`Because your microphone is on a Mac, the most reliable path is for me to listen here while you run one short capture command locally in Terminal or iTerm2 on the Mac. I will infer the SSH target from the host I am running on and give you the command; if that host name looks wrong, I will ask you before proceeding.`

If local did not work:

`I may not be listening to the right local microphone. Are you speaking from a remote machine? If so, what SSH host should I use?`

If the runtime reports silent capture:

`I captured audio, but it was effectively silent. That usually means the wrong input device is selected or microphone permission is missing on the capture host. Should we check the device or switch to another host?`

On macOS over SSH, good agent follow-up:

`This Mac microphone is remote, and SSH capture on macOS often does not share Terminal/iTerm2 microphone permission. I will listen here and give you one short command to run locally on the Mac.`

If the agent is using the remote Mac push flow, good agent behavior:

`I am listening now. Speak on the Mac and press Ctrl-C when you are done recording. After Ctrl-C, the Mac command will stream the recorded WAV to me. I will wait for EOF, then print the final transcript and clean up this listen session before the next one.`

Agent-side helper:

```bash
.agents/skills/voice/listen/scripts/prepare_remote_push.sh --ssh-target dchung@spark-2f6e
```

That prepares a stable FIFO path, one agent-only listen command, one short human-facing Mac command that records locally until Ctrl-C and then streams the recorded WAV, and one longer fallback command. The agent should run the listen side itself in one live polled session, not as a detached background job, kill any stale listener still attached to the fixed FIFO before starting a new one, show the human the short command first, poll until EOF arrives, and only offer the fallback command if the short one fails. After the listen returns or fails, clean up the FIFO path before the next run. For `whisper`, `sherpa`, and `moonshine`, the remote-push behavior is the same: wait for EOF from Ctrl-C on the Mac and then print one final transcript.

## Operational Questions

### If The Human Asks “How Do I Hear You?”

Good agent answer:

`By default I try the local speaker on the machine running the skill. If that’s not the right place, I can stream WAV audio over SSH to another host with ssh:<host>. On a Mac, the simplest setup is Homebrew sox, which provides play. Install it with 'brew install sox'. If I need to build or run the ONNX-backed voice runtime on a Mac host, I also need 'brew install onnxruntime'.`

### If The Human Asks “How Do You Hear Me?”

Good agent answer:

`By default I try the local microphone on the machine running the skill. If your microphone is on another host, I can capture WAV over SSH with ssh:<host>. On a Mac, the simplest setup is Homebrew sox, which provides rec. Install it with 'brew install sox'. If I need to build or run the ONNX-backed voice runtime on a Mac host, I also need 'brew install onnxruntime'. If your microphone is remote on a Mac, I will usually listen here and ask you to run one short rec command locally in Terminal or iTerm2 so the capture uses the app that has microphone permission. I do not force sample-rate or channel flags there because the Mac input device may ignore them anyway. If that short command fails, I will then give you a longer compatibility fallback.`
