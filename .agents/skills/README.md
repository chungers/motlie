# Voice Skills README

This README is the conversational playbook for the repo-local voice skills:

- `voice-speak`
- `voice-listen`
- `voice-turn`

## Runtime Rules

- prefer shipped platform binaries under `.agents/skills/bin/`
- if no binary is available for this host, build the most optimized one for the host
- always use `release`
- prefer CUDA when the host supports it
- use local audio devices first when they are available
- if local audio does not work, ask the human whether they are remote and what SSH host to use
- discover runtime details progressively through the conversation with the human

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

- use `voice-speak`
- backend: `piper`
- endpoint: local

Example agent response:

`I’m going to try the local speaker first with Piper.`

Human prompt:

`Say this with qwen3: Nothing new shipping.`

Agent interpretation:

- use `voice-speak`
- backend: `qwen3cpp`
- endpoint: local

Example agent response:

`I’m going to say that locally with qwen3-tts.cpp.`

Human prompt:

`Play it on motliehost.`

Agent interpretation:

- use `voice-speak`
- endpoint: `ssh:motliehost`

Example agent response:

`I’ll send the audio to motliehost over SSH.`

### Listen

Human prompt:

`Listen to me.`

Agent interpretation:

- use `voice-listen`
- backend: `whisper`
- endpoint: local

Example agent response:

`I’m going to try the local microphone first with Whisper.`

Human prompt:

`Listen with sherpa.`

Agent interpretation:

- use `voice-listen`
- backend: `sherpa`
- endpoint: local

Example agent response:

`I’m going to listen locally with sherpa.`

Human prompt:

`My mic is on motliehost.`

Agent interpretation:

- use `voice-listen`
- endpoint: `ssh:motliehost`

Example agent response:

`I’ll listen from motliehost over SSH.`

### Turn

Human prompt:

`Ask me for my status and listen to the reply.`

Agent interpretation:

- use `voice-turn`
- TTS backend: `piper`
- ASR backend: `whisper`
- playback: local
- capture: local

Example agent response:

`I’ll use a local voice turn with Piper for speaking and Whisper for listening.`

Human prompt:

`Use qwen3 to ask me and listen with moonshine.`

Agent interpretation:

- use `voice-turn`
- TTS backend: `qwen3cpp`
- ASR backend: `moonshine`

Example agent response:

`I’ll use qwen3 for speech and Moonshine for listening.`

## Progressive Discovery Examples

### Local Or Remote Not Clear

Human prompt:

`Say something to me.`

Good agent follow-up:

`I can try the local speaker first. If you want remote playback instead, tell me the SSH host.`

If local did not work:

`Local playback may not be the right device. Are you on a remote machine? If so, what SSH host should I use?`

### Listen Fallback

Human prompt:

`Listen to me.`

Good agent follow-up:

`I’ll try the local microphone first. Tell me if it doesn’t hear you, and I can switch to a remote SSH host.`

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

## Source-Missing Response

If no shipped binary exists and the repo source is not present on this host, tell the human exactly that.

Good agent response:

`There isn’t a shipped voice binary for this host, and the source tree is not available here to build one. I need either the prebuilt platform binary in .agents/skills/bin/ or the repo source on this machine.`
