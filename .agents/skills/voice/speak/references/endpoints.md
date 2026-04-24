Default:

- local playback device

If the human says the speaker is remote, use:

- `--endpoint ssh:<host>`

Examples:

- `--endpoint ssh:motliehost`
- `--endpoint ssh:macmini`

macOS remote playback setup:

- install `sox`:
  - `brew install sox`
- expected remote playback command:
  - `/opt/homebrew/bin/play -t wav -`
