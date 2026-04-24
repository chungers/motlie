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
- if no binary is available for this host, build the most optimized one for the host
- if the selected backend weights are missing, download them into the shared skill cache:
  - `.agents/skills/voice/artifacts/hf-cache/`
- always use `release`
- prefer CUDA when the host supports it
- use local audio devices first when they are available
- if local audio does not work, ask the human whether they are remote and what SSH host to use
- discover runtime details progressively through the conversation with the human

## Bootstrapping Rules

- treat build and model download as normal first-run bootstrap work
- if the host binary is missing, tell the human you need to build it and ask them to wait
- if the selected model weights are missing, tell the human you need to download them and ask them to wait
- if both are missing, tell the human both steps are part of bootstrap:
  - build the optimized binary for this host
  - download the selected backend weights into `.agents/skills/voice/artifacts/hf-cache/`
- if neither a shipped binary nor repo source is present, tell the human exactly that
- after bootstrap, continue with the original speak/listen/turn request instead of stopping

## Backend Defaults

- TTS default: `piper`
- TTS override: if the human says `say with qwen3`, use `qwen3cpp`
- ASR default: `whisper`
- ASR override: if the human says `listen with sherpa` or `listen with moonshine`, use that backend

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

Bootstrap example:

Human prompt:

`Say hello to me.`

Good agent response when the binary and/or models are missing:

`I need to bootstrap the voice skill on this host first by building the optimized binary and downloading the Piper weights into the shared voice artifacts cache. Please wait a moment.`

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

Bootstrap example:

Human prompt:

`Listen to me.`

Good agent response when Whisper is not cached yet:

`I need to bootstrap listening on this host first by downloading the Whisper weights into the shared voice artifacts cache. Please wait a moment.`

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

If local did not work:

`I may not be listening to the right local microphone. Are you speaking from a remote machine? If so, what SSH host should I use?`

## Operational Questions

### If The Human Asks “How Do I Hear You?”

Good agent answer:

`By default I try the local speaker on the machine running the skill. If that’s not the right place, I can stream WAV audio over SSH to another host with ssh:<host>. On a Mac, the simplest setup is Homebrew sox, which provides play. Install it with 'brew install sox'.`

### If The Human Asks “How Do You Hear Me?”

Good agent answer:

`By default I try the local microphone on the machine running the skill. If your microphone is on another host, I can capture WAV over SSH with ssh:<host>. On a Mac, the simplest setup is Homebrew sox, which provides rec. Install it with 'brew install sox'.`

## Build-In-Progress Response

If the wrapper has to build a host binary, tell the human to wait.

Good agent response:

`I need to build the optimized voice binary for this host first. Please wait a moment.`

If weights are also likely missing:

`I need to bootstrap the voice skill on this host first. That usually means building the optimized binary and downloading the selected model weights into .agents/skills/voice/artifacts/hf-cache/. Please wait a moment.`

## Source-Missing Response

If no shipped binary exists and the repo source is not present on this host, tell the human exactly that.

Good agent response:

`There isn’t a shipped voice binary for this host under the namespaced skill bin directory, and the source tree is not available here to build one. I need either the prebuilt platform binary in .agents/skills/voice/<skill>/bin/ or the repo source on this machine.`
