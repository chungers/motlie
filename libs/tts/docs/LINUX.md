# Linux TTS with Remote Audio Streaming

This document covers running the MCP TTS server on a headless Linux host with audio playback on a remote macOS client over SSH.

## Overview

When Claude Code runs on a headless Linux server (no speakers), we need to stream TTS audio back to the client machine. This document compares two approaches and provides implementation guidance.

```
┌─────────────────────────────────────────────────────────────┐
│ macOS Client (your laptop)                                  │
│                                                             │
│  Terminal (SSH) ◄──────────────────────┐                   │
│                                        │                    │
│  Audio Listener ◄──── SSH Tunnel ◄─────┼──┐                │
│       │                                │  │                 │
│       ▼                                │  │                 │
│  Speakers                              │  │                 │
└────────────────────────────────────────┼──┼─────────────────┘
                                         │  │
┌────────────────────────────────────────┼──┼─────────────────┐
│ Linux Host (headless)                  │  │                 │
│                                        │  │                 │
│  Claude Code ◄──► MCP TTS Server ──────┘  │                 │
│                        │                  │                 │
│                   piper ──► netcat ───────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Approach Comparison

| Aspect | nc + SSH + sox | PulseAudio Forwarding |
|--------|----------------|----------------------|
| **Latency** | **~20-50ms** | ~50-150ms |
| **macOS setup** | **Simple (one command)** | Complex (run PA daemon) |
| **Linux setup** | None | None (paplay works) |
| **MCP implementation** | Slightly more | Simpler |
| **Protocol overhead** | **None (raw bytes)** | Buffering, negotiation |
| **All-app audio** | No (per-command) | Yes |
| **Debugging** | Easy (check nc) | `pactl`, `pavucontrol` |

**Recommendation**: Use **nc + SSH + sox** for MCP TTS because:
- Lower latency (TTS should feel responsive)
- Simpler macOS setup (no daemon required)
- Only TTS needs remote audio (not all Linux apps)
- Easy to understand and debug

## nc + SSH + sox Approach

### Prerequisites

**macOS client:**
```bash
brew install sox    # For `play` command
# Or use built-in `afplay` for WAV files
```

**Linux host:**
```bash
# Install piper (neural TTS)
pip install piper-tts

# Download a voice model
piper --update-voices
# Or manually download from https://github.com/rhasspy/piper/releases

# netcat is usually pre-installed (nc or ncat)
```

### Setup

#### Step 1: Start Audio Listener on macOS

For continuous raw PCM stream:
```bash
# Listen on port 12345, play raw audio as it arrives
nc -l 12345 | sox -t raw -r 22050 -b 16 -c 1 -e signed - -d
```

Or for WAV files (one at a time):
```bash
while true; do nc -l 12345 | afplay -; done
```

#### Step 2: SSH to Linux with Reverse Tunnel

```bash
ssh -R 12345:localhost:12345 user@linux-host
```

This creates a tunnel where `localhost:12345` on Linux forwards to `localhost:12345` on your Mac.

#### Step 3: Test TTS

On Linux:
```bash
echo "Hello from Linux" | piper --model en_US-lessac-medium --output-raw | nc localhost 12345
```

You should hear the audio on your Mac.

### Helper Scripts

#### macOS: `~/bin/audio-listener.sh`

```bash
#!/bin/bash
# Persistent audio listener for remote TTS
# Usage: audio-listener.sh [port]

PORT=${1:-12345}
echo "Listening for audio on port $PORT..."
echo "Press Ctrl+C to stop"

while true; do
    nc -l $PORT | sox -t raw -r 22050 -b 16 -c 1 -e signed - -d 2>/dev/null
done
```

#### macOS: `~/bin/ssh-audio.sh`

```bash
#!/bin/bash
# SSH with audio tunnel
# Usage: ssh-audio.sh user@host

PORT=${AUDIO_PORT:-12345}

# Start listener in background
~/bin/audio-listener.sh $PORT &
LISTENER_PID=$!
trap "kill $LISTENER_PID 2>/dev/null" EXIT

# SSH with reverse tunnel
ssh -R $PORT:localhost:$PORT "$@"
```

Usage:
```bash
ssh-audio.sh user@linux-host
```

## Bidirectional Audio: STT + TTS

For voice-driven workflows (speak on Mac → transcribe on Linux → respond via TTS):

```
┌─────────────────────────────────────────────────────────────┐
│ macOS Client                                                │
│                                                             │
│  Microphone ──► rec ──► nc localhost 12346 ────────┐       │
│                                          (to Linux) │       │
│                                                     │       │
│  Speakers ◄── sox ◄── nc -l 12345 ◄────────────────┼──┐    │
│                                       (from Linux)  │  │    │
└─────────────────────────────────────────────────────┼──┼────┘
                                                      │  │
                 SSH -R 12345:... -L 12346:...        │  │
                                                      │  │
┌─────────────────────────────────────────────────────┼──┼────┐
│ Linux Host                                          │  │    │
│                                                     ▼  │    │
│  nc -l 12346 ──► whisper ──► LLM ──► piper ──► nc ─────┘    │
│                                                             │
│  (mic input)    (STT)      (process)  (TTS)   (audio out)  │
└─────────────────────────────────────────────────────────────┘
```

### SSH Command for Bidirectional Tunnels

```bash
ssh -R 12345:localhost:12345 \
    -L 12346:localhost:12346 \
    user@linux-host
```

- `-R 12345`: Linux → Mac (TTS audio output)
- `-L 12346`: Mac → Linux (microphone input)

### macOS: Capture and Send Microphone

```bash
# Using sox (recommended)
rec -t raw -r 16000 -b 16 -c 1 -e signed - 2>/dev/null | nc localhost 12346

# Using ffmpeg
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - 2>/dev/null | nc localhost 12346
```

### Linux: Voice Command Pipeline

Simple test (echo audio back):
```bash
nc -l 12346 | nc localhost 12345
```

With STT and TTS:
```bash
# Receive audio, transcribe, speak response
nc -l 12346 > /tmp/input.wav
whisper /tmp/input.wav --model tiny --output_format txt
# Process text, generate response...
echo "I heard you say something" | piper --model en_US-lessac-medium --output-raw | nc localhost 12345
```

### Bidirectional Helper Scripts

#### macOS: `~/bin/voice-session.sh`

```bash
#!/bin/bash
# Full bidirectional voice session
# Usage: voice-session.sh user@host

TTS_PORT=${TTS_PORT:-12345}
MIC_PORT=${MIC_PORT:-12346}

# Start TTS listener (Linux → Mac speakers)
(while true; do
    nc -l $TTS_PORT | sox -t raw -r 22050 -b 16 -c 1 -e signed - -d 2>/dev/null
done) &
TTS_PID=$!

# Start mic sender (Mac mic → Linux) - press Enter to start recording
echo "Audio session ready. TTS on port $TTS_PORT, Mic on port $MIC_PORT"
echo "Press Ctrl+C to stop"

trap "kill $TTS_PID 2>/dev/null" EXIT

# SSH with both tunnels
ssh -R $TTS_PORT:localhost:$TTS_PORT \
    -L $MIC_PORT:localhost:$MIC_PORT \
    "$@"
```

## MCP TTS Server Implementation Notes

### Detecting Remote SSH Session

```rust
fn is_remote_session() -> bool {
    std::env::var("SSH_CLIENT").is_ok() ||
    std::env::var("SSH_TTY").is_ok()
}
```

### Audio Output Strategy

```rust
enum AudioOutput {
    /// macOS: use `say` command with persistent worker
    LocalMacOs,

    /// Linux with local audio: use piper + paplay
    LocalLinux,

    /// Linux over SSH: stream to netcat tunnel
    RemoteNetcat { port: u16 },
}

fn detect_audio_output() -> AudioOutput {
    if cfg!(target_os = "macos") {
        AudioOutput::LocalMacOs
    } else if is_remote_session() {
        // Check for tunnel port in env or config
        let port = std::env::var("TTS_AUDIO_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(12345);
        AudioOutput::RemoteNetcat { port }
    } else {
        AudioOutput::LocalLinux
    }
}
```

### Streaming to Netcat

```rust
async fn say_via_netcat(&self, phrase: &str, port: u16) -> Result<()> {
    let cmd = format!(
        "echo {} | piper --model {} --output-raw | nc localhost {}",
        shell_escape(phrase),
        self.model,
        port
    );

    Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .spawn()?
        .wait()
        .await?;

    Ok(())
}
```

## PulseAudio Forwarding (Alternative)

If you need **all audio** from Linux to play on Mac (not just TTS), use PulseAudio forwarding.

### How It Works

Modern Linux uses PipeWire with PulseAudio compatibility:

```
Linux App ──► PulseAudio API ──► pipewire-pulse ──► PipeWire
                                      │
                                      ▼
                              PULSE_SERVER env
                                      │
                                      ▼
                              TCP to macOS PA server
```

### Setup

**macOS:**
```bash
# Install PulseAudio
brew install pulseaudio

# Run PA server accepting network connections
pulseaudio --load="module-native-protocol-tcp auth-anonymous=1" --exit-idle-time=-1
```

**SSH with tunnel:**
```bash
ssh -R 24713:localhost:4713 user@linux-host
```

**Linux:**
```bash
export PULSE_SERVER=tcp:localhost:24713

# Now all audio apps play on Mac
paplay /usr/share/sounds/sound.wav
piper --output-raw | paplay --raw --rate=22050
mpv video.mp4  # audio goes to Mac
```

### Why nc + sox is Better for TTS

| Aspect | nc + sox | PulseAudio |
|--------|----------|------------|
| Latency | 20-50ms | 50-150ms |
| macOS daemon | Not needed | Required |
| Scope | Just TTS | All audio |
| Complexity | Low | Medium |
| Buffering | None | Yes (can cause delays) |

For MCP TTS, we only need to stream speech audio, not all system sounds. The simpler nc + sox approach gives lower latency and easier setup.

## Troubleshooting

### No audio on macOS

1. Check listener is running:
   ```bash
   lsof -i :12345
   ```

2. Test tunnel from Linux:
   ```bash
   echo "test" | nc -v localhost 12345
   ```

3. Check sox is receiving:
   ```bash
   nc -l 12345 | sox -t raw -r 22050 -b 16 -c 1 -e signed - -d -V
   ```

### Audio is garbled

Sample rate mismatch. Ensure piper output rate matches sox input:
```bash
# Piper default is 22050 Hz
# sox must match:
sox -t raw -r 22050 -b 16 -c 1 -e signed - -d
```

### Connection refused

SSH tunnel not established. Verify with:
```bash
# On Linux
nc -zv localhost 12345
```

If it fails, reconnect SSH with `-R 12345:localhost:12345`.

### Latency is high

1. Use raw PCM instead of WAV (no header buffering)
2. Reduce network buffer: `nc -l 12345 | stdbuf -i0 -o0 sox ...`
3. Check network latency: `ping mac-client`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_AUDIO_PORT` | 12345 | Port for netcat audio tunnel |
| `TTS_MODEL` | en_US-lessac-medium | Piper voice model |
| `PULSE_SERVER` | (none) | PulseAudio server address |

## References

- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) - Fast local neural TTS
- [Piper Voice Models](https://rhasspy.github.io/piper-samples/) - Available voices
- [SoX Documentation](http://sox.sourceforge.net/sox.html) - Swiss army knife of audio
