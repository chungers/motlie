# tmux Driver Integration

## Current State

This document records the tmux integration that now exists in `motlie-driver`.

Implemented:
- `driver::commands::tmux` as the single-host tmux command family
- `driver::commands::tmux_app` as the multi-host / namespaced outer command family
- shared tmux REPL and TUI frontends in `motlie_driver::tmux_frontend`
- semantic resolution in the core driver via `CommandSet::Resolved`, `resolve_command`, and static `execute`
- opt-in multi-host mode in `bins/tmux/driver` with `connect <ssh-uri> as <alias>`

The driver now supports both:
- single-host mode: one `TmuxState`, namespace-less behavior
- multi-host mode: one `TmuxAppState` with multiple named `TmuxState` connections

## Brownfield Constraint

This integration remains brownfield with respect to `motlie-tmux`.

That means:
1. public tmux APIs already used by examples remain stable
2. driver-facing tmux improvements are additive
3. the driver consumes `motlie-tmux`; it does not move ownership semantics into it

The examples and driver still rely on the stable tmux public surface around:
- `HostHandle`
- `Target`
- monitor/session watch APIs
- capture / send / transfer operations

## What Lives Where

### `motlie-tmux` owns

- lifecycle and transport logic
- host/session/window/pane semantics
- monitor supervision
- output bus and history machinery
- capture and transfer operations
- additive helper APIs such as:
  - `watch_session(...)`
  - `snapshot_targets()`
  - `list_target_strings()`
  - `resolve_target_str(...)`

### `motlie-driver::commands::tmux` owns

- the typed tmux command schema
- single-host tmux driver state (`TmuxState`)
- dynamic completion over sessions/targets/stream modes
- retained local mirror history
- REPL/TUI-facing command outputs and effects

### `motlie-driver::commands::tmux_app` owns

- app-level multi-host commands:
  - `connect`
  - `disconnect`
  - `use`
  - `connections`
- the multi-host app state (`TmuxAppState`)
- semantic resolution from raw names into `(alias, command)` pairs
- namespaced completion over `alias/...`
- current-alias fallback when bare names are allowed

## Semantic Resolution Model

The tmux integration is the first real proof of the new core driver resolution stage.

Flow:
1. `clap` parses the command syntax
2. `TmuxCommand` or `TmuxAppCommand` is created from matches
3. `resolve_command(...)` performs sync in-memory scope resolution
4. async execution runs against the selected `TmuxState`

Single-host tmux keeps identity resolution:
- parsed command == resolved command
- no namespace layer

Multi-host tmux adds a real resolution layer:
- `alias/<session>` and `alias/<target>` resolve to one selected `TmuxState`
- bare names resolve against `current` when allowed
- explicit alias always wins over `current`

## Multi-host State Shape

The binary-level multi-host tmux context is:

```rust
pub struct TmuxAppState {
    connections: BTreeMap<String, TmuxState>,
    current: Option<String>,
}
```

Where:
- `connections` maps alias -> connected tmux host state
- `current` is the default alias selected by `use <alias>`

`disconnect <alias>` behavior:
- remove the alias from `connections`
- call `shutdown_managed_state()` on that alias before dropping it
- clear `current` if it pointed at the removed alias

Process-exit behavior in multi-host mode:
- the top-level tmux driver now calls `shutdown_all_managed_state()` before exit
- this ensures non-current aliases do not retain driver-managed watch/stream state after the session ends

## Command Resolution Rules

### App-level commands

These do not require `current`:
- `connect <ssh-uri> as <alias>`
- `disconnect <alias>`
- `use <alias>`
- `connections`

### Commands that accept `alias/<session>`

These can resolve explicitly by alias:
- `new-window <session> <name> ...`
- `monitor start <session> [seconds]`
- `history <session> [session...]`

### Commands that accept `alias/<target>`

These can resolve explicitly by alias:
- `split-pane <target> ...`
- `kill <target>`
- `send <target> <text...>`
- `keys <target> <keys...>`
- `capture <target> <lines>`
- `stream <target> ...`

### Commands that require `current`

These do not take an explicit qualified resource argument and therefore require
`use <alias>` first in multi-host mode:
- `create <name> ...`
- `targets`
- `mirror history ...`
- `mirror clear`
- `tui on`
- `tui off`
- `monitor stop`
- `upload ...`
- `download ...`

Important rule:
- multi-host `targets` does **not** implicitly fan out across every alias
- if `current` is unset, bare `targets` returns `MissingCurrentScope`

### `history` normalization rule

One `history` command is constrained to one alias.

Rules:
- `history alpha/demo demo` resolves both names against `alpha`
- `history alpha/demo beta/build` is rejected
- if no session argument has an explicit alias, bare names resolve against `current`

## Completion Model

Completion follows the same rules as resolution.

In single-host mode:
- completion is driven from the one `TmuxState`

In multi-host mode:
- app-level commands complete aliases directly
- if the user already typed `alias/...`, completion stays inside that alias
- if `current` is set, bare completion resolves against the current alias
- if `current` is not set, qualifying completions are returned as `alias/...`

## Frontend Integration

The shared tmux frontends in `motlie_driver::tmux_frontend` now work against both:
- `CommandEngine<TmuxState, TmuxCommand>`
- `CommandEngine<TmuxAppState, TmuxAppCommand>`

That reuse is enabled by the `TmuxFrontendState` trait.

The top-level binary stays an assembly layer:
- single-host mode wires `TmuxState + TmuxCommand`
- `--multi-host` wires `TmuxAppState + TmuxAppCommand`
- REPL/TUI behavior stays shared above that

## Current Verification Scope

What this integration now proves:
- the core driver resolution stage is compositional
- a simple adapter can keep identity resolution
- a larger app-level command family can add namespace-aware resolution without changing the shared frontend code
- tmux is a working proof that future namespace-aware adapters, including VMM, do not need to bolt scope parsing into every `execute()` path

## Non-goals

This integration still does not:
1. move resource lifecycle/business logic into `motlie-driver`
2. make `motlie-tmux` depend on `motlie-driver`
3. generalize a cross-subsystem orchestration layer yet
4. add imported/remote proxy lifecycles beyond the current in-process tmux driver session
