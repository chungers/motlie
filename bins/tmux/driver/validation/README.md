# tmux-driver validation artifacts

These files were captured from a real end-to-end smoke test of:
- binary: `motlie-tmux-driver`
- frontend: line REPL hosted inside local tmux
- remote backend: `ssh://dchung@motliehost?identity-file=/home/dchung/.ssh/motliehost`
- date: `2026-04-11`

## Asciicast

A fresh driver-owned asciicast implementation now lives under:
- `motlie_driver::term::asciicast`

This branch does not depend on `libs/vmm` for recording.

Saved replay artifact:
- [`tmux-driver-validation.cast`](./tmux-driver-validation.cast)

It was recorded from the top-level tmux driver using:
- `--record-asciicast <path>`

The pane captures remain alongside the cast because they are easy to diff and
review directly in GitHub.

## Artifact map

- [`00-initial-prompt.txt`](./00-initial-prompt.txt)
  - initial successful connect and prompt
- [`01-targets.txt`](./01-targets.txt)
  - baseline session list showing existing sessions, including `jarvis`
- [`02-create-send-capture.txt`](./02-create-send-capture.txt)
  - create test session, send command, and capture output
- [`03-monitor-follow.txt`](./03-monitor-follow.txt)
  - attached monitor path showing new remote output arriving live
- [`04-mirror-history.txt`](./04-mirror-history.txt)
  - retained local history readback using `mirror history --limit 10`
- [`05-cleanup-targets.txt`](./05-cleanup-targets.txt)
  - cleanup proof showing the temporary test session was removed
- [`06-writer-pane.txt`](./06-writer-pane.txt)
  - second driver instance used to inject output into the monitored session
- [`tmux-driver-validation.cast`](./tmux-driver-validation.cast)
  - asciicast v3 replay artifact from the validated plain REPL session

## Validation summary

The smoke test covered:
- connect
- `targets`
- `create`
- `send`
- `capture`
- attached `monitor`
- `Ctrl-C` stop behavior for live follow
- `mirror history`
- `kill`
- post-cleanup `targets`

Constraint honored:
- the existing `jarvis` session was not touched

Current limitation:
- the saved cast is a driver-owned command-flow/session artifact, not a frame-accurate
  alternate-screen replay of every TUI draw
