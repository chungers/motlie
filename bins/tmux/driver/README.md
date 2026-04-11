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

## Behavior

- `tui on` switches from the plain REPL into the split-screen TUI
- `tui off` returns from the split-screen TUI back to the plain REPL
- plain `monitor <session>` uses an application-layer live render loop owned by this binary
- split-screen TUI uses the shared `TmuxState` mirror/watch/stream state
