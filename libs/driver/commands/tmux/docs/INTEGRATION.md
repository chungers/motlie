# tmux Driver Integration

## Scope

This document records how the first `motlie-driver` vertical slice should
integrate with `motlie-tmux`.

The first slice is:

- REPL-owned
- in-process
- tmux resources only
- feature-equivalent with `libs/tmux/examples/repl`
- TUI-capable, including the current split-screen mirror flow

The goal is to build `driver::commands::tmux`, `driver::repl`, and
`driver::tui` on top of the tmux crate without breaking the current examples.

## Brownfield Constraint

This step is brownfield.

That means:

1. public tmux APIs already used by examples should remain stable
2. driver-facing tmux improvements should be additive
3. examples should not need to be rewritten just to keep compiling

The examples currently depend directly on:

- `HostHandle::create_session`
- `HostHandle::session`
- `HostHandle::target`
- `Target::new_window`
- `Target::split_pane`
- `Target::kill`
- `Target::send_text`
- `Target::send_keys`
- `Target::sample_text`
- `Target::capture_all`
- `HostHandle::output_bus`
- `HostHandle::start_monitoring_session`
- `OutputBus::subscribe`

Those are the APIs that should not be churned for the driver work.

## What The Driver Needs

The driver layer needs higher-level helpers around three recurring tmux tasks:

1. session watch lifecycle
2. reusable target discovery
3. canonical target-string resolution

The current examples implement those ad hoc.

### Session watch lifecycle

Both the plain REPL and the TUI manually compose:

1. `output_bus.subscribe(...)`
2. `start_monitoring_session(...)`
3. `Subscription::history(...)`
4. teardown with `unsubscribe`, `join`, and `shutdown`

The driver wants one additive owned handle for that lifecycle.

The tmux-side additive surface is:

```rust
pub struct SessionWatchOptions {
    pub queue_capacity: usize,
    pub history: HistoryOptions,
}

pub struct SessionWatchHandle { ... }

impl HostHandle {
    pub async fn watch_session(
        &self,
        session_name: &str,
        opts: &SessionWatchOptions,
    ) -> Result<SessionWatchHandle>;
}
```

This does not replace the lower-level monitoring APIs. It simply packages the
existing pattern into one driver-friendly handle.

### Target discovery

The current examples rebuild the session/window/pane tree inline for `targets`
rendering and completion-oriented logic.

The driver wants one reusable discovery snapshot so `targets`, tab completion,
and TUI navigation can share the same source.

The additive tmux-side surface is:

```rust
pub struct SessionTargetTree { ... }
pub struct WindowTargetTree { ... }
pub struct PaneTargetTree { ... }

impl HostHandle {
    pub async fn snapshot_targets(&self) -> Result<Vec<SessionTargetTree>>;
    pub async fn list_target_strings(&self) -> Result<Vec<String>>;
}
```

### Target-string resolution

The current REPL and TUI both duplicate:

1. `TargetSpec::parse(...)`
2. `HostHandle::target(...)`

The driver wants a single helper:

```rust
impl HostHandle {
    pub async fn resolve_target_str(&self, target_str: &str) -> Result<Option<Target>>;
}
```

## What Stays In tmux vs What Moves To driver

### `motlie-tmux` keeps

- lifecycle and transport logic
- target/session/window/pane semantics
- monitor supervision
- output bus and history machinery
- capture and transfer operations

### `motlie-driver::commands::tmux` owns

- command names and clap schema
- driver session state
- owned/imported/ephemeral semantics
- command sequencing and orchestration
- dynamic completion from live driver state plus tmux discovery
- frontend-facing effects for REPL/TUI

## First Vertical Slice State Model

For the first slice, the driver should treat tmux state roughly as:

```rust
pub struct TmuxState {
    pub host: HostHandle,
    pub owned_sessions: HashSet<String>,
    pub active_watch: Option<SessionWatchHandle>,
    pub active_stream: Option<ManagedStream>,
}
```

Important:

- `owned_sessions` are the only tmux entities destroyed by default on driver close
- watch/stream state is ephemeral child state
- discovered targets are usable immediately, but not automatically owned

## Required Driver Work After tmux Helpers Exist

Once the additive tmux helpers are in place, the driver work becomes:

1. define `TmuxCommand` as one typed family
2. cover the union of plain REPL and TUI commands
3. route both `driver::repl` and `driver::tui` through the same command engine
4. replace binary-local parser logic with `clap`
5. replace duplicated watch/discovery code with tmux helper APIs

## Non-Goals For This Step

1. do not move ownership semantics into `motlie-tmux`
2. do not change the signatures of the existing example-facing APIs
3. do not block driver work on a deep tmux crate redesign

The tmux crate only needs additive public lifecycle helpers to support the new
driver architecture cleanly.
