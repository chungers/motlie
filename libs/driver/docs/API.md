# motlie-driver API

## Status

Implemented surface for the current driver feasibility slice.

This file documents the public API shape that exists in `libs/driver` today and the contract
used by the tmux adapter. It does not describe speculative VMM/VNET/VFS adapter code that is
not on `main`.

## Core Types

### Output and effects

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEffect {
    ExitShell,
    EnterTui,
    ExitTui,
}

#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub lines: Vec<String>,
    pub effects: Vec<CommandEffect>,
}
```

`CommandOutput` is the shared frontend payload:
- `lines` are plain rendered text lines
- `effects` are frontend control signals

Current effect semantics:
- `ExitShell`
  - frontend should terminate
- `EnterTui`
  - tmux line REPL should switch into the tmux split-screen TUI
- `ExitTui`
  - tmux TUI should return to the line REPL

### Completion

```rust
pub struct CompletionRequest<'a> {
    pub command_path: &'a [&'a str],
    pub arg_id: Option<&'a str>,
    pub prefix: &'a str,
}

pub struct CompletionCandidate {
    pub value: String,
    pub help: Option<String>,
}
```

`command_path` is the resolved subcommand path relative to the command-family root.

Example:
- input: `stream demo --mo`
- request path: `["stream"]`
- prefix: `--mo`

### Errors

Library code uses `DriverError` and `DriverResult<T>`, not `anyhow`.

The error type currently covers:
- invalid shell quoting
- invalid help topics
- invalid history capacity
- IO / clap / task join failures
- tmux and regex errors when the tmux adapter is enabled
- not-found and invalid-argument cases
- asciicast persistence failures

## CommandSet Contract

Current implemented contract:

```rust
#[async_trait::async_trait]
pub trait CommandSet<C>: Sized {
    type CompletionContext: Send + 'static;

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;
    fn help(topic: &[String]) -> Option<String> { None }
    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> { Vec::new() }
    async fn execute(self, context: &mut C) -> DriverResult<CommandOutput>;
}
```

Planned extension for semantic resolution:

```rust
#[async_trait::async_trait]
pub trait CommandSet<C>: Sized {
    type CompletionContext: Send + 'static;
    type Resolved;

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;
    fn help(topic: &[String]) -> Option<String> { None }
    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> { Vec::new() }
    fn resolve(self, context: &C) -> DriverResult<Self::Resolved>;
    async fn execute(
        resolved: Self::Resolved,
        context: &mut C,
    ) -> DriverResult<CommandOutput>;
}
```

Planned compositional rule:
- simple adapters use identity resolution
- `type Resolved = Self`
- `resolve(self, _) -> Ok(self)`
- namespace-aware adapters return a distinct resolved type when needed

Responsibilities:

| Method | Purpose |
|---|---|
| `root_command()` | Build the static `clap` command tree for this family |
| `from_matches()` | Convert parsed `ArgMatches` into the typed command enum |
| `completion_context()` | Produce a read-only sync snapshot for completion |
| `help()` | Optional rich help topics that override plain `clap` help |
| `complete()` | Adapter-owned dynamic completion |
| `execute()` | Run the typed command against mutable session state |

## CommandEngine Contract

`CommandEngine<C, S>` is generic over context and command family.

Current execution flow:
- tokenize shell input
- parse with `clap`
- build typed parsed command
- execute typed parsed command

Planned execution flow after semantic-resolution support:
- tokenize shell input
- parse with `clap`
- build typed parsed command
- resolve parsed names/references against read-only context
- execute resolved command against mutable context

Current stable methods:
- `new(context)`
- `context()`
- `context_mut()`
- `completion_context()`
- `run_line(&str)`
- `run_argv(&[String])`
- `complete(line, cursor)`

Current built-ins:
- `help`
- `quit`
- `exit`

Important behavior:
- `run_line()` accepts REPL-style input without requiring the command-family root name
- `run_argv()` also accepts already-prefixed argv like `["tmux", "targets"]`


## Planned Generic Naming Helpers

This is proposed driver-core support, not implemented surface yet.

```rust
pub struct QualifiedName<'a> {
    pub scope: Option<&'a str>,
    pub value: &'a str,
}

pub struct ResolvedName {
    pub scope: String,
    pub value: String,
}

pub fn parse_qualified_name(raw: &str) -> QualifiedName<'_>;
```

Planned generic resolution traits may look like:

```rust
pub trait ResolveName<K> {
    type Resolved;

    fn resolve_name(&self, kind: K, raw: &str) -> DriverResult<Self::Resolved>;
}
```

This support is intended for:
- tmux connection aliases like `alias/target`
- future VMM guest or namespace-qualified names
- app-level command sets that compose multiple scoped adapters

## History API

The retained-history utility is generic:

```rust
pub struct HistoryBuffer<T> { /* bounded FIFO */ }

impl<T> HistoryBuffer<T> {
    pub fn new(capacity: NonZeroUsize) -> Self;
    pub fn try_with_capacity(capacity: usize) -> DriverResult<Self>;
    pub fn capacity(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn push(&mut self, item: T) -> u64;
    pub fn clear(&mut self);
    pub fn latest(&self) -> Option<&HistoryRecord<T>>;
    pub fn oldest_seq(&self) -> Option<u64>;
    pub fn newest_seq(&self) -> Option<u64>;
}

impl<T: Clone> HistoryBuffer<T> {
    pub fn page_after(&self, after: Option<u64>, limit: usize) -> HistoryPage<T>;
}
```

Design points:
- zero capacity is rejected by type/result, not by panic
- history is local to the driver session
- sequence ids are monotonic and page-friendly

## tmux Adapter Contract

The first real adapter is `driver::commands::tmux`.

### Types

Main public types:
- `TmuxState`
- `TmuxCommand`
- `TmuxMirrorSnapshot`
- `TmuxHistoryEntry`

### TmuxState responsibilities

`TmuxState` owns:
- a connected `HostHandle`
- discovered sessions and targets for completion
- a set of owned sessions created by this driver session
- active watch/stream state
- the current mirror snapshot
- retained mirror history

Key methods:
- `connect(uri)`
- `refresh_discovery()`
- `refresh_mirror()`
- `mirror_snapshot()`
- `mirror_history_page(after, limit)`
- `has_live_follow()`
- `shutdown_managed_state()`

### TmuxCommand surface

Current command family:
- `create`
- `new-window`
- `split-pane`
- `kill`
- `mirror history`
- `mirror clear`
- `tui on`
- `tui off`
- `targets`
- `send`
- `keys`
- `capture`
- `monitor start`
- `monitor stop`
- `history`
- `stream`
- `upload`
- `download`

### tmux completion

Current dynamic completion coverage:
- sessions for `new-window`, `monitor start`, `history`
- targets for `split-pane`, `kill`, `send`, `keys`, `capture`, `stream`
- stream mode values

### tmux shared frontends

The real tmux product frontends are library functions:
- `driver::tmux_frontend::run_tmux_repl(...)`
- `driver::tmux_frontend::run_tmux_tui(...)`

These are shared by:
- the top-level tmux driver binary
- the driver examples

That is the supported shape today; there is no longer a separate generic `TuiFrontend`
abstraction in the library.

## Adapter Guidance For Future Resource Crates

For future VMM/VNET/VFS adapters, the rule is:
- resource crates expose lifecycle APIs
- `motlie-driver` adapters consume those APIs and implement `CommandSet<C>`

They should follow the tmux pattern:
- typed command enum
- typed session state
- explicit dynamic completion snapshot
- help topics close to the adapter
- no `anyhow` in the library layer
