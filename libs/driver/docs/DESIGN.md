# motlie-driver Design

## Status

Implemented feasibility slice on `main`:
- `libs/driver` exists and builds
- `CommandEngine<C, S>` is the generic execution core
- `CommandSet<C>` is the typed command-family contract
- `driver::commands::tmux` is the first real adapter
- shared tmux REPL/TUI frontends live in `driver::tmux_frontend`
- top-level app assembly lives in `bins/tmux/driver`

This document describes the code that exists today. Future adapters such as
VMM/VNET/VFS are design targets only and are not scaffolded as code on `main`.


## New Problem: Semantic Name Resolution

The current feasibility slice parses command syntax correctly, but it assumes
that the parsed command payload is already semantically executable.

That assumption holds for the current single-host tmux slice because:
- one `CommandEngine<TmuxState, TmuxCommand>` owns one `TmuxState`
- one `TmuxState` owns one `HostHandle`
- target strings like `demo:0.1` can be resolved directly against that one host

It will not hold once the driver surface grows.

Examples:
- tmux multi-host mode:
  - `connect ssh://dchung@motliehost?identity-file=... as dchung-motlie`
  - `send dchung-motlie/demo:0.1 echo hello`
  - `capture demo:0.1 50`
    - uses the current selected connection when no explicit alias is given
- future VMM mode:
  - `exec alice uname -a`
  - `shutdown cluster-a/alice`
  - `pty sandbox/main`

These are not parsing problems. They are semantic resolution problems:
- choose the correct namespace / scope
- apply current/default scope when the user omits one
- validate the reference against session state
- convert a parsed command into a resolved command before execution

If every adapter solves this ad hoc inside `execute()`, the driver surface will
drift:
- different separators and namespace rules
- different current-scope behavior
- duplicated string-splitting logic
- inconsistent error messages
- inconsistent completion behavior

The driver crate should own this stage once, compositionally.

## Crate Boundaries

### `libs/driver`

Owns:
- command parsing and dispatch
- completion analysis and merge
- driver-local history buffers
- frontend-neutral command output/effects
- tmux adapter and tmux-specific shared frontends
- terminal-side utilities such as asciicast recording

Does not own:
- tmux lifecycle semantics
- VM/network/filesystem business logic
- remote process/resource discovery beyond what adapters call through public APIs

### Resource crates

Resource crates continue to own their public lifecycle APIs.

Examples:
- `motlie-tmux` owns host/session/target/monitor operations
- future `motlie-vmm` should own `prepare/boot/ready/exec/shutdown`
- future `motlie-vnet` should own network backend lifecycle

`motlie-driver` consumes those APIs. It does not redefine them.

### Top-level applications

Applications assemble:
- a concrete context
- a concrete `CommandSet`
- one or more frontends
- feature gates for which adapters/frontends are compiled

Current real application:
- `bins/tmux/driver`

## Feature Topology

Current `motlie-driver` features:

- `repl`
  - enables `reedline` integration and the tmux line REPL path
- `tui`
  - enables `ratatui + crossterm` integration and the tmux split-screen UI
- `commands-tmux`
  - enables the tmux command adapter and shared tmux frontends
- `term-vt100`
  - reserved for future terminal-emulation work
- `term-shadow`
  - reserved for future terminal-shadow work

Important current constraint:
- future adapter feature names for VMM/VNET/VFS are documented work items, not implemented modules on `main`

## Core Runtime Model

The generic driver surface is intentionally small:

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

pub struct CommandEngine<C, S> {
    /* context + typed command family */
}
```

`CommandEngine` owns:
- the mutable session context `C`
- `run_line()` and `run_argv()`
- built-in `help`
- built-in `quit` / `exit`
- static + dynamic completion merge

It does not own:
- editor loops
- TUI rendering
- subsystem business logic


## Planned Semantic Resolution Stage

### Current flow

Today the engine flow is:

1. tokenize shell input
2. parse with `clap`
3. convert `ArgMatches` into the typed command
4. execute the typed command against mutable context

That is sufficient only when:
- parsed names need no semantic interpretation beyond `clap`
- command execution can directly consume the parsed strings

### Proposed flow

The planned driver flow is:

1. tokenize shell input
2. parse with `clap`
3. convert `ArgMatches` into the typed parsed command
4. resolve parsed names/references against read-only context
5. execute the resolved command against mutable context

The key design rule is:
- `clap` owns syntax
- the driver owns semantic resolution
- adapters own resource-specific execution

### Compositional requirement

This stage must be optional-by-default.

If an adapter does not need semantic resolution, the design should collapse back
to today's namespace-less behavior with no extra ceremony.

That means:
- parsed command type may equal resolved command type
- `resolve()` may be identity
- simple adapters pay no extra complexity tax

The design must support three cases cleanly:

1. identity / namespace-less adapters
   - current single-host tmux slice
2. scoped adapters
   - tmux multi-host aliases like `alias/target`
   - future VMM guest namespaces like `cluster/guest`
3. composed app command sets
   - app-level connect/select/disconnect plus subsystem commands

### Proposed trait shape

One viable evolution is:

```rust
#[async_trait::async_trait]
pub trait CommandSet<C>: Sized {
    type CompletionContext: Send + 'static;
    type Resolved;

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;

    fn help(topic: &[String]) -> Option<String> {
        None
    }

    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        Vec::new()
    }

    fn resolve(self, context: &C) -> DriverResult<Self::Resolved>;

    async fn execute(
        resolved: Self::Resolved,
        context: &mut C,
    ) -> DriverResult<CommandOutput>;
}
```

Then the engine becomes:

```rust
let parsed = S::from_matches(&matches)?;
let resolved = parsed.resolve(&self.context)?;
S::execute(resolved, &mut self.context).await
```

Identity adapters use:
- `type Resolved = Self`
- `resolve(self, _) -> Ok(self)`

This keeps the design compositional.

### Alternative trait split

If the single-trait shape becomes awkward in Rust, the same idea can be modeled
as two traits:
- parsed command trait
- executable resolved command trait

The important requirement is not the exact trait spelling. It is the presence of
a first-class semantic resolution stage between parse and execute.

## Planned Naming / Resolution Support In `libs/driver`

The core driver crate should grow small generic utilities for scoped names.

Planned examples:

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

And a context-facing resolution helper pattern:

```rust
pub trait ResolveName<K> {
    type Resolved;

    fn resolve_name(&self, kind: K, raw: &str) -> DriverResult<Self::Resolved>;
}
```

This is intentionally generic:
- tmux can resolve connection alias + target/session
- VMM can resolve namespace + guest / PTY / handle
- future adapters can reuse the same structure

The driver should also own the generic error vocabulary for this stage:
- malformed qualified name
- missing current scope
- unknown scope
- ambiguous name
- resource not found in scope


### Shared namespace, adapter-relative resolution

A namespace label must not imply one universal object.

The design must support the same scope label existing across multiple adapters at
once.

Example:
- `alice` may name a tmux session scope
- `alice` may also name a vnet allocation
- `alice` may also name a vfs mount set
- later, `alice` may also name a VMM guest or PTY scope

That means resolution must stay relative to command context and resource kind,
not just the raw namespace string.

The intended model is:
- the application owns a shared namespace vocabulary
- each adapter owns its own resource registry within that vocabulary
- commands resolve names relative to adapter/resource kind
- composed commands may intentionally coordinate several resource kinds under
  the same namespace label

So `alice` is a shared scope label, not "the one thing called alice".

This is important for future composed commands such as:
- `workspace create alice`
- `workspace destroy alice`
- `workspace status alice`

Those commands should be able to orchestrate multiple adapter-owned resources
under the same namespace without introducing global name ambiguity.

## Completion Impact

Completion should continue to work compositionally with the new stage.

Current completion remains valid:
- static completion from `clap`
- dynamic completion from adapter snapshots

Planned additions:
- current-scope completion for bare names
- explicit scoped-name completion for prefixes like `alias/...`
- scope-alias completion for app-level commands such as:
  - `connect`
  - `disconnect`
  - `use`
  - `connections`

The completion rule should match the resolution rule:
- if the user provides an explicit scope, complete inside that scope
- otherwise use current/default scope when one exists

## Verification Slice: tmux Multi-host Namespaced Mode

The proposed proving slice for this design is the top-level tmux driver binary.

Add an opt-in mode that turns the current single-host tmux binary into a
multi-host session manager.

### User-facing shape

When enabled, the binary should support app-level commands such as:

```text
connect ssh://dchung@motliehost?identity-file=... as dchung-motlie
connect ssh://dchung@otherhost?identity-file=... as staging
use dchung-motlie
connections
disconnect staging
```

After that, tmux entity references should support:

```text
targets
send dchung-motlie/demo:0.1 echo hello
capture dchung-motlie/demo 50
kill staging/build:0.1
```

And when a current connection is selected:

```text
use dchung-motlie
send demo:0.1 echo hello
capture demo 50
```

### Planned context shape

The current one-host `TmuxState` would become a lower-level connection state.

The binary-level app context would look more like:

```rust
pub struct TmuxAppState {
    connections: BTreeMap<String, TmuxState>,
    current: Option<String>,
}
```

And the app-level command family would compose:
- connection-management commands
- tmux operational commands

### Why this slice is a good proof

It exercises all of the new generic concerns:
- optional namespace resolution
- current/default scope behavior
- dynamic completion over scoped names
- one command family reused inside a larger app-level command surface

And it does so without requiring VMM to land first.

## clap Integration

`clap` is the source of truth for static command structure.

The driver uses it for:
- command parsing
- subcommand discovery
- flag completion
- help rendering fallback

The driver adds:
- root-name prefixing for REPL-style input
- completion analysis for the current command path / active arg
- built-in command handling before command-family dispatch

## Completion Architecture

Completion has two layers.

### Static completion

Derived from the `clap::Command` tree:
- subcommands
- long flags
- short flags

### Dynamic completion

Provided by the adapter through:
- `CommandSet::completion_context(&C)`
- `CommandSet::complete(request, &CompletionContext)`

This split matters:
- execution owns `&mut C`
- completion needs a sync, read-only snapshot

The associated `CompletionContext` type on `CommandSet` is the actual mechanism used in code
to solve that borrowing boundary.

## Help Architecture

`help` is engine-owned.

Its content is composed from the same command family that drives parsing:

1. `CommandEngine` checks whether the input starts with `help`
2. it asks `CommandSet::help(topic)` for an adapter-specific rich help topic
3. if no rich topic exists, it renders `clap` help from the composed command tree

That means:
- disabled command families contribute no help
- rich help stays near the adapter that owns the command semantics
- static command help never drifts away from the `clap` schema

Current real example:
- `driver::commands::tmux::help()` provides richer topics for `stream`, `monitor`,
  `mirror`, `capture`, `history`, and `tui`

## tmux Vertical Slice

The first full slice is tmux-only.

### Shared state

`TmuxState` owns:
- connected `HostHandle`
- discovered session names
- discovered target names
- owned session names created by this driver session
- at most one active watch
- at most one active stream
- current mirror snapshot
- driver-local retained mirror history

This is intentionally a narrow slice:
- one active watch/stream at a time
- no attach/import persistence yet
- no generalized resource-mode metadata yet


The planned namespaced multi-host mode above is a future proof slice on top of
the current single-host implementation, not a statement that it already exists
on `main`.

### Shared frontends

The tmux product frontends are shared library code:
- `driver::tmux_frontend::run_tmux_repl(...)`
- `driver::tmux_frontend::run_tmux_tui(...)`

That code is consumed by:
- `bins/tmux/driver`
- `libs/driver/examples/tmux_repl.rs`
- `libs/driver/examples/tmux_tui.rs`

This replaced the earlier duplication between:
- bin-local `plain.rs` / `ui.rs`
- example-local `common/tmux_plain.rs` / `common/tmux_ui.rs`

## History Buffer Design

`driver::history` is generic and driver-owned.

It is not tmux-specific and not terminal-specific.

Current types:

```rust
pub struct HistoryRecord<T> {
    pub seq: u64,
    pub at: SystemTime,
    pub item: T,
}

pub struct HistoryPage<T> {
    pub items: Vec<HistoryRecord<T>>,
    pub next_after: Option<u64>,
    pub oldest_available: Option<u64>,
    pub newest_available: Option<u64>,
}

pub struct HistoryBuffer<T> { /* bounded FIFO */ }
```

Current real use:
- tmux watch/stream/capture/history commands retain local mirror history
- plain REPL follow mode pages forward over that retained history
- `mirror history --after ... --limit ...` exposes the retained local history

Motivation:
- the retained buffer is a better primitive than periodic stdout dumps
- it can later support RPC/admin pagination
- it can later support teeing driver-owned tracing/log events into another buffer

## Asciicast Recording

`driver::term::asciicast` is a driver-owned recorder.

Current real behavior:
- the tmux driver binary can record REPL and TUI command-flow events
- plain REPL live-follow output is recorded
- startup TUI mode records entry/exit and command/output flow

Current limitation:
- this is not a full alternate-screen frame recorder
- it does not yet serialize every rendered TUI frame as a visual replay stream
- it is currently best understood as a driver-owned terminal/session artifact format

That is why the implementation includes an explicit comment that the event timestamps are
delta-based and are not aiming for strict asciinema compatibility.

## Top-level tmux driver binary

`bins/tmux/driver` exists outside `motlie-tmux` on purpose:
- `motlie-driver` depends on `motlie-tmux`
- placing the driver binary in the `motlie-tmux` package would create a package cycle

The binary is intentionally assembly-only:
- parse top-level args
- connect `TmuxState`
- choose REPL or TUI
- optionally create an asciicast recorder

## What Is Deliberately Not Implemented Yet

Not implemented on `main`:
- VMM/VNET/VFS command adapters
- generic resource-mode enums like `Owned | Imported | Ephemeral`
- generic remote-proxy lifecycle support
- generic shared TUI abstraction for non-tmux consumers
- full-screen frame-accurate TUI capture

Those remain valid future directions, but they are not part of the implemented contract of
this PR and should not be documented as if they already exist.
