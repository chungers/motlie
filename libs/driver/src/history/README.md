# `motlie_driver::history`

`motlie_driver::history` is a generic bounded event/history buffer for data that
the command driver wants to retain locally as session sidecar state.

## Why This Exists

The immediate vertical slice is tmux watch/stream state.

The tmux driver needs more than a single mutable `mirror_text` string:
- a live watch or stream should accumulate local session history
- callers should be able to page through that history instead of only reading the
  latest frame
- different frontends should be able to build their own representations over the
  same underlying retained history

That is useful for:
- plain REPL tail/follow behavior
- split-screen TUI scrollback
- future RPC/admin APIs that want cursor-based pagination

## Why This Does Not Live Under `term`

Terminal output is one consumer of history, but not the only one.

This module is intentionally generic so that the driver can also retain:
- normalized tracing/log records
- lifecycle/status events
- resource-specific sidecar state that is not terminal text

That leaves room for future tees such as:
- feeding selected tracing/OTel events into a driver-owned history buffer
- correlating terminal output history with lifecycle or telemetry events

If the module lived under `driver::term`, it would overfit the first tmux use case.

## Ownership Model

History buffers are driver-session sidecars.

That means the ownership of a history buffer is local to the command engine
session, even when the underlying resource is:
- `Owned`
- `Imported`
- `Ephemeral`

Example:
- a tmux watch handle may be ephemeral
- the watched tmux session may be imported
- the local history buffer for that watch is still owned by the current driver
  session

## First Vertical Slice

The first implementation target is:
- tmux watch/stream/capture/history output

That slice should prove:
- bounded retention
- sequence/cursor-based reads
- frontend-independent access to retained history

Only after that should richer adapters be added, such as:
- terminal-specific helpers
- tracing/log tee support
- serialization for remote admin surfaces
