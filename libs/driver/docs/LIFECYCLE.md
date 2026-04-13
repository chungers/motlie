# Driver Lifecycle Notes

## Scope

This document describes the lifecycle semantics that are actually implemented in the current
driver slice.

Current implemented resource family:
- tmux

Future resource families such as VMM/VNET/VFS are intentionally omitted here until their
driver adapters exist on `main`.

## Layers

### Raw resource lifecycle

Owned by the domain crate.

For tmux, `motlie-tmux` owns:
- host connection lifecycle
- session/window/pane creation and kill
- monitor/watch handles
- file transfer operations

### Driver session lifecycle

Owned by `TmuxState`.

The driver decides:
- which sessions it considers locally owned
- whether a watch/stream is currently active
- what mirror content is retained locally
- what the frontend should render from that local state

## tmux Resource Categories

### Host connection

Type:
- `HostHandle`

Driver semantics:
- created when `TmuxState::connect(uri)` runs
- long-lived for the whole driver session
- not exposed as a named child resource

Cleanup:
- implicit when the driver process exits and the handle drops

### Owned sessions

Tracked by:
- `TmuxState::owned_sessions`

Driver semantics:
- sessions created through `create` are marked locally owned
- killing a target/session removes the owned mark

Cleanup:
- explicit `kill` when the operator requests it
- the current tmux driver does not auto-kill all owned sessions on exit

That is deliberate for the tmux slice:
- a tmux operator shell may create sessions meant to outlive the shell itself

### Active watch

Type:
- `SessionWatchHandle`

Driver semantics:
- at most one active watch at a time
- starting a new watch replaces any existing watch/stream
- `monitor stop` tears it down
- leaving the tmux TUI also tears down managed watch/stream state

Cleanup:
- explicit via `shutdown_managed_state()`

### Active stream

Type:
- driver-local `ManagedStream`

Driver semantics:
- at most one active stream at a time
- polling-driven mirror updates
- tracks consecutive failure count and last error
- updates the mirror label when repeated polling errors happen

Cleanup:
- explicit via `shutdown_managed_state()`

### Mirror history

Type:
- `HistoryBuffer<TmuxHistoryEntry>`

Driver semantics:
- local sidecar object, not a remote tmux resource
- bounded FIFO
- fed by monitor/stream/capture/history activity
- consumed by:
  - plain REPL live-follow
  - `mirror history`
  - future pagination/RPC work

Cleanup:
- local only
- cleared by `mirror clear`
- otherwise dropped with the driver session

## Frontend Semantics

### Plain REPL

Implemented in:
- `driver::tmux_frontend::run_tmux_repl(...)`

Lifecycle semantics:
- command execution runs through `CommandEngine`
- if live follow becomes active, the frontend pages through retained mirror history
- `Ctrl-C` stops live follow and returns to the prompt

### Split-screen TUI

Implemented in:
- `driver::tmux_frontend::run_tmux_tui(...)`

Lifecycle semantics:
- same `CommandEngine` and `TmuxState` as the line REPL
- mirrors the same active watch/stream state
- `tui off` returns to the line REPL
- leaving the TUI tears down managed watch/stream state

## What Is Not Modeled Yet

Not implemented in the current driver lifecycle model:
- imported resources
- remote proxy resources
- generic owned/imported/ephemeral enums
- generalized detach/rehydrate flows across process boundaries

Those remain future design topics. The current tmux slice is intentionally:
- local
- in-process
- single-session-state
- explicit about only the resources it really manages today
