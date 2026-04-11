# tmux-driver validation artifacts

These files were captured from a real end-to-end smoke test of:
- binary: `motlie-tmux-driver`
- frontend: line REPL hosted inside local tmux
- remote backend: `ssh://dchung@motliehost?identity-file=/home/dchung/.ssh/motliehost`
- date: `2026-04-11`

## Why pane captures instead of asciicast

`asciinema` is not installed on the validation host (`spark-2f6e`), so a proper
`.cast` file was not recorded in this run.

The saved artifacts here are plain text pane captures from the tmux-hosted REPL
sessions used during validation.

## Artifact map

- [`00-initial-prompt.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/00-initial-prompt.txt)
  - initial successful connect and prompt
- [`01-targets.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/01-targets.txt)
  - baseline session list showing existing sessions, including `jarvis`
- [`02-create-send-capture.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/02-create-send-capture.txt)
  - create test session, send command, and capture output
- [`03-monitor-follow.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/03-monitor-follow.txt)
  - attached monitor path showing new remote output arriving live
- [`04-mirror-history.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/04-mirror-history.txt)
  - retained local history readback using `mirror history --limit 10`
- [`05-cleanup-targets.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/05-cleanup-targets.txt)
  - cleanup proof showing the temporary test session was removed
- [`06-writer-pane.txt`](/home/dchung/cdx-repl/motlie/bins/tmux/driver/validation/06-writer-pane.txt)
  - second driver instance used to inject output into the monitored session

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
