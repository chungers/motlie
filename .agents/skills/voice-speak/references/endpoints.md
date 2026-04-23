Use `.agents/voice/voice.env` to choose:

- playback endpoint
- artifact roots
- build profile
- acceleration policy (`auto`, `cpu`, `cuda`)

For remote macOS playback:

- `KIND=ssh`
- `SSH_TARGET=motliehost`
- `PLAY_CMD=/opt/homebrew/bin/play -t wav -`
