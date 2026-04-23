Use `.agents/voice/voice.env` to choose:

- capture endpoint
- artifact roots
- build profile
- acceleration policy (`auto`, `cpu`, `cuda`)

For remote macOS capture:

- `KIND=ssh`
- `SSH_TARGET=motliehost`
- `RECORD_CMD=/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -`

If `--seconds` is omitted, stop capture with:

- `Ctrl-C`
- ending the SSH session
- interrupting the remote `rec` process
