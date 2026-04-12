# tmux-driver

Top-level tmux operator entrypoint built on `motlie-driver`.

This binary exists outside `motlie-tmux` on purpose:
- `motlie-driver` depends on `motlie-tmux`
- putting the driver binary inside the `motlie-tmux` package would create a package cycle

## Relationship To The Legacy Example

The historical tmux REPL remains at:
- `libs/tmux/examples/repl/main.rs`

That example is kept for brownfield reference and historical context.
New driver-backed work should target this package instead.

## Features

- `commands-tmux`
  - enables the tmux command bindings from `motlie-driver`
- `repl`
  - enables the plain line REPL frontend
- `tui`
  - enables the split-screen ratatui frontend

Defaults:
- `repl`
- `tui`
- `commands-tmux`

## Build

Plain tmux REPL only:

```bash
cargo run -p motlie-tmux-driver --no-default-features --features repl,commands-tmux -- ssh://localhost
```

Split-screen TUI only:

```bash
cargo run -p motlie-tmux-driver --no-default-features --features tui,commands-tmux -- --tui ssh://localhost
```

Both frontends:

```bash
cargo run -p motlie-tmux-driver -- ssh://localhost
```

Start directly in TUI:

```bash
cargo run -p motlie-tmux-driver -- --tui ssh://localhost
```

Record a plain REPL session to asciicast v3:

```bash
cargo run -p motlie-tmux-driver -- \
  --record-asciicast bins/tmux/driver/validation/tmux-driver-validation.cast \
  ssh://localhost
```

## Behavior

- `tui on` switches from the plain REPL into the split-screen TUI
- `tui off` returns from the split-screen TUI back to the plain REPL
- `monitor start <session>` attaches live monitoring until it is stopped or replaced
- `monitor stop` stops the active watch or stream
- the tmux REPL and TUI both use the shared library frontend in `motlie_driver::tmux_frontend`
- split-screen TUI uses the same shared `TmuxState` mirror/watch/stream state as the line REPL

## Validation

The driver has been smoke-tested end to end against:
- host: `ssh://dchung@motliehost?identity-file=/home/dchung/.ssh/motliehost`
- local frontend host: `spark-2f6e`

The validation flow was:
1. Build the top-level driver:
   `cargo build -p motlie-tmux-driver`
2. Launch the driver inside a local tmux session so `reedline` has a real terminal.
3. Connect to `motliehost` and confirm existing sessions with `targets`.
4. Create an isolated test session:
   `create codex-e2e-smoke-20260411a`
5. Send and capture shell output:
   `send codex-e2e-smoke-20260411a echo hello-from-validation`
   `capture codex-e2e-smoke-20260411a 20`
6. Start attached monitoring:
   `monitor start codex-e2e-smoke-20260411a`
7. Use a second driver instance to inject new output into the same remote session.
8. Stop live follow with `Ctrl-C`.
9. Read retained local history:
   `mirror history --limit 10`
10. Clean up the test session:
    `kill codex-e2e-smoke-20260411a`
    `targets`

The historical `jarvis` session was not modified during validation.

Saved validation artifacts are under:
- [`validation/README.md`](./validation/README.md)
- [`00-initial-prompt.txt`](./validation/00-initial-prompt.txt)
- [`01-targets.txt`](./validation/01-targets.txt)
- [`02-create-send-capture.txt`](./validation/02-create-send-capture.txt)
- [`03-monitor-follow.txt`](./validation/03-monitor-follow.txt)
- [`04-mirror-history.txt`](./validation/04-mirror-history.txt)
- [`05-cleanup-targets.txt`](./validation/05-cleanup-targets.txt)
- [`06-writer-pane.txt`](./validation/06-writer-pane.txt)

Environment note:
- a fresh driver-owned asciicast writer now exists in `motlie_driver::term::asciicast`
- the saved validation cast is:
  [`tmux-driver-validation.cast`](./validation/tmux-driver-validation.cast)
- the pane captures remain useful as plain-text proof alongside the `.cast`
- current recording is command-flow oriented; it is not yet a frame-accurate full-screen TUI replay
