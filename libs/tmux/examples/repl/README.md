# Legacy tmux REPL

This directory contains the original brownfield tmux REPL prototype:
- `main.rs`
- `tui_mirror.rs`

It is kept for historical context and comparison while the new driver-backed tmux
entrypoint evolves.

New work should target the top-level driver package instead:
- `bins/tmux/driver`

That package is built on `motlie-driver` and is the intended path for:
- shared command-engine integration
- shared static and dynamic completion
- shared REPL and TUI frontends
- future selective feature-gated assembly

Do not add new product behavior here unless the driver-backed path is being kept in
sync intentionally.
