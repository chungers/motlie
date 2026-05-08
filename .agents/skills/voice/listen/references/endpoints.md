Default:

- local microphone

If the human says the microphone is remote, use:

- `--endpoint ssh:<host>`

Examples:

- `--endpoint ssh:motliehost`
- `--endpoint ssh:macmini`

macOS remote capture setup:

- install `sox`:
  - `brew install sox`
- expected remote capture command:
  - `/opt/homebrew/bin/rec -q -t wav -`

If `--seconds` is omitted, stop capture with:

- `Ctrl-C`
- ending the SSH session
- interrupting the remote `rec` process
