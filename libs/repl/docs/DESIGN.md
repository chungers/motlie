# Unified REPL Engine Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-10 | @codex-repl: Refine the design from REPL-first to command-engine-first. Lock subsystem vs command-engine boundaries, add explicit owned/imported/ephemeral semantics, document cleanup expectations, and reference the new [`LIFECYCLE.md`](./LIFECYCLE.md) resource inventory. | Goals, Requirements, Chosen Solution, Architecture, Risks, Open Questions, Summary |
| 2026-04-10 | @codex-repl: Expand the command-engine and REPL relationship with a concrete code walkthrough. Replace the vague handler sketch with subsystem-facing traits that avoid exposing `Pin<Box<dyn Future>>` in the public integration surface, and reference the new [`API.md`](./API.md) contract document. | Command Registration Model, Completion Model, API Ergonomics |
| 2026-04-10 | @codex-repl: Initial DESIGN for `libs/repl`, derived from issue #150 and reconciled against the current repo snapshot. Greenfield product direction; migration and backward-compatibility concerns are out of scope. | All |

This document defines the design for a new `libs/repl` crate that provides a unified,
composable command engine plus an interactive REPL frontend for Motlie. It is derived from
GitHub issue #150, the current checkout of `main`, and a follow-up lifecycle review of the
active VMM worktree used as a design input.

## Repo Reality Check

Issue #150 is directionally sound, but several assumptions in the issue do not match the
current checkout and must be treated as design inputs:

1. `libs/vmm` does not exist in this repo snapshot, so VMM adoption is a future consumer,
   not an implementation-phase dependency.
2. The `motlie` binary is declared from the workspace root `Cargo.toml`; there is no
   standalone `bins/motlie/Cargo.toml`.
3. `motlie-tmux` is not currently wired into the root `motlie` binary as an optional
   dependency or feature.
4. The only current ad-hoc REPL in-tree is `libs/tmux/examples/repl/main.rs`, which uses
   `rustyline` plus manual string parsing and command dispatch.
5. Existing `motlie` CLI commands (`db`, `fulltext`) are modeled as clap derive types plus
   typed handlers, not as reusable runtime `clap::Command` registration functions.

This DESIGN treats the work as a greenfield product direction within the current repo
snapshot. Existing code is relevant as reference and reuse material, but migration and
backward-compatibility concerns are out of scope for this design.

## Problem

Motlie has started to accumulate interactive command surfaces, but they do not share a
common engine or command model.

Today:

1. `bins/motlie/src/main.rs` exposes one-shot CLI commands through clap derive types.
2. `libs/tmux/examples/repl/main.rs` exposes a separate interactive shell with manual
   parsing, manual help text, and no shared command-definition path with the main CLI.
3. There is no standard way for subsystem crates to contribute interactive commands,
   dynamic completion, or stateful REPL context to the main binary.

If Motlie continues with subsystem-specific shells, several costs grow linearly:

1. Duplicate parsing and dispatch logic.
2. Divergent UX across command surfaces.
3. No reusable dynamic completion model for live state such as session names or targets.
4. More rework each time a subsystem later needs both one-shot CLI and REPL forms.

## Goals

1. Define one reusable command engine in `libs/repl`.
2. Allow subsystem crates to register commands into that engine without depending on each
   other.
3. Support an interactive shell with line editing, history, help, and tab completion.
4. Support future non-REPL frontends such as TUI or socket-driven command ingress without
   redesigning command dispatch or state ownership.
5. Support dynamic completion for context-dependent values such as tmux sessions and target
   specs.
6. Define a clean product direction that can coexist with current one-shot CLI behavior while
   the new REPL surfaces are built.
7. Reuse existing clap command definitions where practical instead of forcing every command
   to be rewritten by hand.
8. Make engine-local management semantics explicit so the command engine can own resources it
   created, import references to already-running resources, and later detach cleanly.

## Non-Goals

1. Implement a remote or networked command transport in the first design slice.
2. Redesign the `db` and `fulltext` UX in this phase.
3. Make REPL mode the default behavior of `motlie` when no subcommand is supplied in the
   first integration slice.
4. Introduce `libs/vmm` work in this repo snapshot.
5. Replace tmux TUI behavior; the REPL must coexist with the current `tui on` flow.

## Requirements

### Functional Requirements

1. The binary must be able to assemble REPL commands from feature-enabled subsystem crates.
2. A command definition must be usable in interactive mode and, where practical, reusable
   from one-shot CLI wiring.
3. The engine must support mutable session context owned by the command engine instance.
4. The engine must support async handlers because tmux and future subsystem operations are
   async.
5. The engine must provide built-in `help` and `quit` behavior.
6. The engine must support static completion from clap metadata and dynamic completion from
   live subsystem state.
7. The engine must surface structured command errors without panicking the session.
8. The first concrete adopter must be the tmux example REPL.
9. The command engine must expose frontend-neutral command execution so the same registry and
   state model can be used by interactive REPL, TUI, or future socket/message frontends.
10. The command engine must support attaching to already-running resources and representing
    whether each named resource is `Owned`, `Imported`, or `Ephemeral`.
11. The command engine must provide an explicit close/shutdown path that calls subsystem APIs
    for owned resources rather than silently abandoning live state.

### Non-Functional Requirements

1. The REPL boundary must stay lightweight and composable; subsystem crates should register
   commands, not own the event loop.
2. Dynamic dispatch is acceptable only on the low-frequency registration and dispatch path;
   hot library paths remain statically dispatched.
3. The design must preserve feature-gated binary assembly and avoid cross-crate dependency
   cycles.
4. The design must be testable at unit level and at integration/example level.
5. The first integration must be additive and keep current `motlie` CLI behavior available.
6. Cleanup semantics must be explicit and truthful: best-effort `Drop` may exist as a safety
   net, but the primary contract is explicit close/shutdown through subsystem APIs.

## Existing State

### Main CLI

`bins/motlie/src/main.rs` is a clap derive CLI with top-level subcommands `Info`, `Db`, and
`Fulltext`. The implementation is typed and direct. It does not expose a reusable REPL
registration layer today.

### Tmux REPL

`libs/tmux/examples/repl/main.rs` is already a useful proof that Motlie wants a stateful
interactive shell. It currently provides commands for:

1. Session and target lifecycle: `create`, `new-window`, `split-pane`, `kill`, `targets`
2. Input and capture: `send`, `keys`, `capture`, `monitor`
3. Mode transitions: `tui on`, `tui off`
4. File transfer: `upload`, `download`

The command surface is real, but the implementation is tightly coupled to one binary-local
loop, one prompt, and one parsing style.

## Alternatives Considered

### Alternative A: Keep per-subsystem REPLs and only share helper utilities

This would extract tokenization, help rendering, and maybe history setup into a helper
module, while each subsystem continues to own its own loop.

Pros:

1. Lowest short-term change risk.
2. Minimal new abstraction.
3. No need to reconcile clap derive commands with interactive registration.

Cons:

1. Fails the "one engine" goal.
2. Leaves dynamic completion fragmented and inconsistent.
3. Preserves duplicated command-dispatch structure in every subsystem.
4. Makes eventual `motlie` binary composition harder, not easier.

Verdict: rejected. It reduces local duplication but does not solve the product problem.

### Alternative B: Use `reedline-repl-rs` directly

This would adopt `reedline-repl-rs` as the core engine and adapt Motlie commands to its API.

Pros:

1. Small amount of custom code.
2. Mature line-editing experience via `reedline`.
3. Built-in bridge from clap metadata to REPL commands.

Cons:

1. It does not cleanly expose dynamic completion hooks, which issue #150 correctly calls
   out as essential for live state.
2. Motlie would still need local extension or a fork for completion providers.
3. Fork maintenance would move the critical path outside the repo.
4. The crate API is not designed around Motlie's specific reuse needs, especially typed clap
   derive commands and incremental adoption by existing binaries.

Verdict: rejected. The missing dynamic completion extension point is a hard blocker.

### Alternative C: Build `libs/repl` directly on `reedline` plus `clap`

This creates a thin Motlie-owned command engine on top of clap metadata, with a `reedline`
interactive shell frontend and a small registration API for commands, handlers, and
completion providers.

Pros:

1. Solves dynamic completion directly.
2. Keeps the line-editing UX on a mature dependency.
3. Lets Motlie choose a registration API that supports both runtime `clap::Command` values
   and existing derive-based command types.
4. Fits the repo's layering: subsystem crates export commands, binaries compose them.
5. Keeps the product direction and implementation fully inside the workspace.

Cons:

1. More initial engineering work than using an off-the-shelf wrapper.
2. Motlie owns the completion logic and dispatch glue.
3. Requires a clear implementation boundary so that the first slice does not sprawl.

Verdict: chosen.

## Chosen Solution

Create a new `libs/repl` crate that owns:

1. A frontend-neutral command engine that owns the mutable session context.
2. A command registry based on clap metadata.
3. Async command dispatch into named, mutable session state.
4. Static and dynamic completion.
5. An interactive `reedline` shell frontend layered on top of that engine.
6. Minimal built-in commands such as `help` and `quit`.

Subsystem crates register commands and completion providers into a command engine builder.
The main binary decides which subsystems are enabled, which context fields exist, and which
frontend starts the engine.

The first concrete adopter is the tmux example REPL. Integration into the main `motlie`
binary is a later implementation slice behind an explicit feature gate and an explicit subcommand
such as `motlie repl` or `motlie tmux repl`.

## Architecture

### Crate Layout

The new crate is expected to look roughly like this:

```text
libs/repl/
  Cargo.toml
  src/
    lib.rs
    repl.rs
    command.rs
    completion.rs
    tokenize.rs
    error.rs
  docs/
    API.md
    DESIGN.md
    LIFECYCLE.md
    PLAN.md
```

### Core Types

```rust
pub struct CommandEngine<C> {
    context: C,
    commands: Vec<RegisteredCommand<C>>,
}

pub struct InteractiveShell<C> {
    engine: CommandEngine<C>,
    prompt: String,
    name: String,
}

pub struct RegisteredCommand<C> {
    pub command: clap::Command,
    pub handler: CommandHandler<C>,
    pub completions: Vec<DynamicCompletionBinding<C>>,
}

pub enum ResourceMode {
    Owned,
    Imported,
    Ephemeral,
}

pub enum ResourceLocality {
    InProcess,
    RemoteProxy,
}
```

The registration path may use type erasure for handlers and completion callbacks because
registration is dynamic by nature. This is acceptable at the REPL boundary because command
dispatch is user-driven and low frequency. The underlying subsystem libraries remain regular,
typed Rust APIs.

### Command Registration Model

The engine must support two registration styles:

1. Raw clap command registration for subsystem-local REPL commands.
2. Typed registration for existing clap derive commands so current `db` and `fulltext`
   command structs can be reused rather than rewritten.

The public integration surface should not force subsystem crates to write or even see
`Pin<Box<dyn Future<...>>>`. That is an acceptable internal implementation detail for erased
dispatch, but it is not a good subsystem-facing API.

The intended public shape is trait-based:

```rust
pub struct CommandOutput {
    pub lines: Vec<String>,
}

pub struct CompletionRequest<'a> {
    pub command_path: &'a [&'a str],
    pub arg_id: Option<&'a str>,
    pub prefix: &'a str,
}

pub struct CompletionCandidate {
    pub value: String,
    pub help: Option<String>,
}

#[async_trait::async_trait]
pub trait TypedCommand<C>:
    clap::CommandFactory + clap::FromArgMatches + Send + 'static
{
    const PATH: &'static [&'static str];

    fn complete(_request: CompletionRequest<'_>, _context: &C) -> Vec<CompletionCandidate> {
        Vec::new()
    }

    async fn execute(self, context: &mut C) -> anyhow::Result<CommandOutput>;
}

pub trait CommandModule<C> {
    fn register(self, engine: &mut CommandEngine<C>);
}

impl<C> CommandEngine<C> {
    pub fn add_typed<T>(&mut self) -> &mut Self
    where
        T: TypedCommand<C>;

    pub fn add_spec<S>(&mut self, spec: S) -> &mut Self
    where
        S: Send + Sync + 'static;

    pub async fn run_line(&mut self, line: &str) -> anyhow::Result<CommandOutput>;
    pub async fn run_argv(&mut self, argv: &[String]) -> anyhow::Result<CommandOutput>;
    pub fn complete(&self, line: &str, cursor: usize) -> Vec<CompletionCandidate>;
}
```

The exact trait names may change during implementation, but the design intent is stable:
subsystem authors should implement typed commands and registration modules, not hand-roll
erased futures.

In implementation terms, the registration API should land on `CommandEngine<C>`, while the
interactive shell is just one consumer layered over the same engine.

The preferred registration style for subsystem crates is therefore:

1. Define clap-derived command structs.
2. Implement `TypedCommand<C>` for those structs.
3. Group them behind a `CommandModule<C>` registration function.
4. Let the engine internally erase those concrete command types into its runtime registry.

Raw command registration still exists as an escape hatch for hand-built command surfaces, but
typed command registration is the design center.

### Detailed Walkthrough

The concrete relationship between REPL and command engine should look like this:

1. A subsystem such as VMM defines typed clap commands.
2. Each command implements `TypedCommand<C>` with execution and optional dynamic completion.
3. A subsystem registration module adds those commands to `CommandEngine<C>`.
4. The engine builds a composite clap tree and owns the mutable session context.
5. `InteractiveShell<C>` wraps a `reedline::Reedline` instance and forwards input and
   completion requests into the engine.

The code below is intentionally longer than the library example because it walks through the
actual boundary in one place.

#### 1. Subsystem command definitions

```rust
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
pub struct VmmRoot {
    #[command(subcommand)]
    pub command: VmmSubcommand,
}

#[derive(Subcommand)]
pub enum VmmSubcommand {
    Boot(BootGuest),
    Exec(ExecGuest),
    Shutdown(ShutdownGuest),
}

#[derive(Parser)]
pub struct BootGuest {
    #[arg(value_name = "NAME")]
    pub name: String,

    #[arg(long)]
    pub image: PathBuf,
}

#[derive(Parser)]
pub struct ExecGuest {
    #[arg(value_name = "NAME")]
    pub guest: String,

    #[arg(trailing_var_arg = true, required = true)]
    pub argv: Vec<String>,
}

#[derive(Parser)]
pub struct ShutdownGuest {
    #[arg(value_name = "NAME")]
    pub guest: String,
}
```

Each command is just a normal clap type. There is no REPL-specific parser model here.

#### 2. Typed command implementation

```rust
#[async_trait::async_trait]
impl TypedCommand<AppContext> for BootGuest {
    const PATH: &'static [&'static str] = &["vmm", "boot"];

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        let prepared = motlie_vmm::orchestrator::prepare(
            ctx.vmm.prepare_request(&self.name, &self.image)?,
        )
        .await?;

        let vm = motlie_vmm::orchestrator::boot(prepared, ctx.vmm.services()).await?;

        ctx.vmm.guests.insert(
            self.name.clone(),
            ManagedResource {
                mode: ResourceMode::Owned,
                locality: ResourceLocality::InProcess,
                value: vm,
            },
        );

        Ok(CommandOutput {
            lines: vec![format!("booted {}", self.name)],
        })
    }
}

#[async_trait::async_trait]
impl TypedCommand<AppContext> for ExecGuest {
    const PATH: &'static [&'static str] = &["vmm", "exec"];

    fn complete(
        request: CompletionRequest<'_>,
        ctx: &AppContext,
    ) -> Vec<CompletionCandidate> {
        if request.arg_id != Some("guest") {
            return Vec::new();
        }

        ctx.vmm
            .guests
            .keys()
            .filter(|name| name.starts_with(request.prefix))
            .map(|name| CompletionCandidate {
                value: name.clone(),
                help: Some("registered VM".into()),
            })
            .collect()
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        let vm = ctx
            .vmm
            .guests
            .get_mut(&self.guest)
            .ok_or_else(|| anyhow::anyhow!("unknown guest {}", self.guest))?;

        let output = vm.value.exec(self.argv).await?;

        Ok(CommandOutput {
            lines: vec![output.stdout],
        })
    }
}

#[async_trait::async_trait]
impl TypedCommand<AppContext> for ShutdownGuest {
    const PATH: &'static [&'static str] = &["vmm", "shutdown"];

    fn complete(
        request: CompletionRequest<'_>,
        ctx: &AppContext,
    ) -> Vec<CompletionCandidate> {
        if request.arg_id != Some("guest") {
            return Vec::new();
        }

        ctx.vmm
            .guests
            .keys()
            .filter(|name| name.starts_with(request.prefix))
            .map(|name| CompletionCandidate {
                value: name.clone(),
                help: Some("registered VM".into()),
            })
            .collect()
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        let vm = ctx
            .vmm
            .guests
            .remove(&self.guest)
            .ok_or_else(|| anyhow::anyhow!("unknown guest {}", self.guest))?;

        vm.value.shutdown().await?;

        Ok(CommandOutput {
            lines: vec![format!("stopped {}", self.guest)],
        })
    }
}
```

This is the key boundary:

1. clap parsing is handled by derive-generated code.
2. VMM lifecycle work is still done by `motlie_vmm`.
3. The command engine only manages the named registry and command dispatch.
4. Dynamic completion reads the same registry through `&AppContext`.

#### 3. Subsystem registration module

```rust
pub struct VmmModule;

impl CommandModule<AppContext> for VmmModule {
    fn register(self, engine: &mut CommandEngine<AppContext>) {
        engine.add_typed::<BootGuest>();
        engine.add_typed::<ExecGuest>();
        engine.add_typed::<ShutdownGuest>();
    }
}
```

This is all the REPL crate should need from a subsystem integration module.

#### 4. Command-engine execution path

```rust
impl<C> CommandEngine<C> {
    pub async fn run_line(&mut self, line: &str) -> anyhow::Result<CommandOutput> {
        let argv = shlex::split(line)
            .ok_or_else(|| anyhow::anyhow!("invalid shell quoting"))?;
        self.run_argv(&argv).await
    }

    pub async fn run_argv(&mut self, argv: &[String]) -> anyhow::Result<CommandOutput> {
        let root = self.build_root_command();
        let matches = root.try_get_matches_from(argv)?;
        let dispatch = self.resolve_dispatch(&matches)?;
        dispatch.run(&matches, &mut self.context).await
    }
}
```

At this layer, the engine is:

1. building the aggregate clap tree,
2. parsing the incoming argv,
3. selecting the registered command implementation, and
4. executing it against the mutable session context.

The engine does not perform guest orchestration itself. It only routes to the concrete typed
command implementation.

#### 5. Static plus dynamic completion

```rust
impl<C> CommandEngine<C> {
    pub fn complete(&self, line: &str, cursor: usize) -> Vec<CompletionCandidate> {
        let partial = &line[..cursor];
        let tokens = tokenize_partial(partial);
        let request = self.analyze_completion(&tokens);

        let mut out = self.static_clap_completion(&request);
        out.extend(self.dynamic_completion(&request));
        dedup_candidates(out)
    }

    fn dynamic_completion(
        &self,
        request: &CompletionRequest<'_>,
    ) -> Vec<CompletionCandidate> {
        self.lookup_registered_command(request.command_path)
            .map(|command| command.complete(request.clone(), &self.context))
            .unwrap_or_default()
    }
}
```

Static completion comes from clap metadata:

1. command names
2. subcommands
3. flags
4. value enums or static possible values

Dynamic completion comes from the typed command implementation or adapter layer:

1. guest names
2. tmux targets
3. monitor aliases
4. imported remote resource identifiers

The engine merges those two sources before returning them to the frontend.

#### 6. Reedline frontend

```rust
pub struct EngineCompleter<'a, C> {
    pub engine: &'a CommandEngine<C>,
}

impl<'a, C> reedline::Completer for EngineCompleter<'a, C> {
    fn complete(&mut self, line: &str, cursor: usize) -> Vec<reedline::Suggestion> {
        self.engine
            .complete(line, cursor)
            .into_iter()
            .map(|candidate| reedline::Suggestion {
                value: candidate.value,
                description: candidate.help,
                style: None,
                extra: None,
                span: reedline::Span::new(0, cursor),
                append_whitespace: true,
            })
            .collect()
    }
}

impl<C> InteractiveShell<C> {
    pub async fn run(&mut self) -> anyhow::Result<()> {
        let mut editor = reedline::Reedline::create()
            .with_completer(Box::new(EngineCompleter { engine: &self.engine }));

        loop {
            match editor.read_line(&reedline::DefaultPrompt::default())? {
                reedline::Signal::Success(line) => {
                    let output = self.engine.run_line(&line).await?;
                    for line in output.lines {
                        println!("{line}");
                    }
                }
                reedline::Signal::CtrlC | reedline::Signal::CtrlD => break,
            }
        }

        Ok(())
    }
}
```

This makes the frontend boundary concrete:

1. `reedline` owns the terminal editor loop.
2. `InteractiveShell` adapts `reedline` signals into engine calls.
3. `CommandEngine` owns parsing, dispatch, registry lookup, and completion.
4. Subsystem crates only contribute command types and lifecycle-aware command handlers.

The architectural consequence is simple: if later a TUI or admin socket wants the same
command surface, it talks to `CommandEngine`, not to `reedline`.

### Session State Model

The command engine owns a single mutable context value for the lifetime of the session:

```rust
pub struct AppContext {
    #[cfg(feature = "tmux")]
    pub tmux: TmuxSessionState,

    #[cfg(feature = "vmm")]
    pub vmm: VmmSessionState,
}

pub struct TmuxSessionState {
    pub hosts: HashMap<String, ManagedResource<motlie_tmux::HostHandle>>,
    pub monitors: HashMap<String, ManagedResource<motlie_tmux::SessionMonitorHandle>>,
}

pub struct VmmSessionState {
    pub guests: HashMap<String, ManagedResource<motlie_vmm::orchestrator::VmHandle>>,
    pub ptys: HashMap<String, ManagedResource<motlie_vmm::ssh::GuestPtySession>>,
}

pub struct ManagedResource<T> {
    pub mode: ResourceMode,
    pub locality: ResourceLocality,
    pub value: T,
}
```

Subsystem crates do not depend on each other. They only require access to the portions of
context they own. The binary remains the composition root.

The crucial boundary is:

1. Subsystem crates own business logic, validation, and real lifecycle semantics.
2. The command engine owns session-local names, aliases, references, and management metadata.
3. Frontends such as REPL, TUI, or future socket ingress call into the command engine; they
   do not own resource state directly.

### Boundaries and Responsibilities

The command engine is intentionally not a replacement for subsystem orchestration APIs.

Subsystem crates own:

1. Resource creation and destruction semantics.
2. Validation, readiness, and runtime correctness.
3. Type-specific handles and operations such as `VmHandle::ready()`,
   `VmHandle::shutdown()`, `Target::kill()`, or `SessionMonitorHandle::shutdown()`.

The command engine owns:

1. The session registry of named live objects and aliases.
2. Resolution from command arguments such as `demo`, `alice`, or `logs-monitor` to typed
   handles stored in context.
3. Engine-local management policy for those named objects:
   `Owned` means the engine is responsible for explicit cleanup, `Imported` means the engine
   attached to an externally owned resource and should detach without destroying it by
   default, and `Ephemeral` means the object is a short-lived child attachment that should be
   cleaned up locally and recreated as needed.
4. Frontend-neutral command execution and completion over the session registry.

This boundary is what allows a temporary admin shell to attach to already-running resources,
rehydrate its local state, and later detach without implicitly killing externally owned
resources.

### Managed Resource Adapters

The command engine should not require `vmm`, `tmux`, `vnet`, or `vfs` crate types to
implement command-engine traits directly. That would invert the dependency direction.

Instead, the command engine owns a small adapter layer around raw handles or remote proxies.
Those adapters are where engine-local semantics live:

1. `ResourceMode` such as `Owned`, `Imported`, or `Ephemeral`.
2. `ResourceLocality` such as `InProcess` or `RemoteProxy`.
3. Cleanup policy: destroy, detach, or local child close.
4. Rehydrate/import policy and completion-token generation.
5. Invalidation rules, for example a VM restart invalidating guest PTY sessions.

Ownership is therefore not a property of `VmHandle` or `Target` alone. It is a property of
the command engine's relationship to that resource in a given session.

The adapter shape can remain small. For example:

```rust
pub trait ManagedResourceAdapter {
    fn mode(&self) -> ResourceMode;
    fn locality(&self) -> ResourceLocality;
    fn completion_tokens(&self) -> Vec<String>;
}

pub trait ManagedResourceLifecycle {
    async fn close(self) -> anyhow::Result<()>;
    async fn detach(self) -> anyhow::Result<()>;
}
```

The final trait split may differ, but the design requirement is stable: the command-engine
crate owns the adapter contracts, while subsystem crates remain independent and are wrapped
by engine-local integration types.

The intended contract families are:

| Contract | Responsibility | Applies to | Notes |
|----------|----------------|------------|-------|
| `ManagedResourceAdapter` | Surface engine-local metadata such as `ResourceMode`, `ResourceLocality`, stable naming, and completion tokens. | All named managed resources. | This is the minimal contract the registry needs for lookup and completion. |
| `ManagedResourceLifecycle` | Define what engine `close()` and `detach()` mean for the adapted resource. | Resources that may outlive one command. | `close()` destroys only when the engine truly owns the resource; `detach()` drops engine-local attachment state. |
| Import / rehydrate adapter | Rebuild a managed entry from runtime artifacts, discovery results, or RPC connection metadata. | `Imported` resources only. | This is expected to be crate-specific and strongest for tmux first. |
| Remote proxy adapter | Normalize RPC-backed resources into the same command surface as local in-process handles. | `RemoteProxy` locality. | This is how a temporary admin process can manage a long-running host process without owning its resources directly. |
| Invalidation policy | Decide when a managed record becomes stale and what children must be dropped. | Parent-child resources such as VM to PTY or host to monitor. | Example: VM restart invalidates `GuestPtySession` children. |

The exact trait names may change, but all five responsibilities need an explicit home in the
implementation.

### Completion Model

Completion has two sources:

1. Static clap metadata such as command names, subcommands, flags, and value enums.
2. Dynamic providers that return live values for a given command and argument id.

Dynamic providers must not require `&mut C` because completion runs while the line editor is
active. Instead, providers read a snapshot-oriented view or shared state owned by the
context, for example `Arc<RwLock<HashSet<String>>>` or the command engine's named resource
registry.

```rust
pub trait CompletionProvider<C>: Send + Sync + 'static {
    fn complete(&self, context: &C, prefix: &str) -> Vec<String>;
}
```

The final implementation may use a read-only accessor or snapshot function instead of
borrowing the full context directly, but the design requirement is the same: completion must
observe live state without taking the mutable execution borrow needed for handlers.

### Lifecycle and Rehydration Model

Lifecycle expectations are documented in detail in [`LIFECYCLE.md`](./LIFECYCLE.md). The
high-level command-engine contract is:

1. Frontends may create owned resources during a session.
2. Frontends may also import references to resources that predate the engine process.
3. The engine must keep enough typed state to support command resolution and completion.
4. The engine must expose explicit shutdown, for example `close(self)`, that attempts cleanup
   for `Owned` resources and local child cleanup for `Ephemeral` resources by calling the
   correct subsystem APIs.
5. Dropping the engine may perform best-effort fallback, but explicit close is the primary
   contract.

This distinction matters for resources such as tmux targets or guest PTY sessions:

1. A tmux `HostHandle` or `Target` may be rehydrated by rediscovery on a host the engine did
   not create.
2. A guest PTY session is short-lived and should be recreated after VM restart or control
   plane loss.
3. A `VmHandle` is a long-lived control object, but rehydration of already-running VMs is a
   future adapter problem rather than a responsibility of the command engine core.

## Data Flow

### Command Execution

1. The binary creates `AppContext`.
2. The binary creates `CommandEngine<AppContext>`.
3. Feature-enabled subsystem crates register commands and dynamic completion providers.
4. A frontend such as the interactive shell passes one line or argv vector to the engine.
5. The engine tokenizes shell-like input when needed.
6. The engine identifies the command root and runs clap parsing.
7. The typed or raw handler executes against the mutable context.
8. The handler returns displayable output or a structured error.
9. The frontend renders the result and continues.

### Tab Completion

1. `reedline` asks the completer for suggestions at the current cursor position.
2. The completer tokenizes the partial input and identifies command plus argument position.
3. Static clap suggestions are gathered first.
4. Matching dynamic providers are queried against live context state.
5. Suggestions are merged, de-duplicated, and returned to the editor.

## API Ergonomics

### Library Example

```rust
use clap::{Arg, Command};
use motlie_repl::{CommandEngine, InteractiveShell};

struct AppContext {
    tmux: TmuxSessionState,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = AppContext {
        tmux: TmuxSessionState::default(),
    };

    let mut engine = CommandEngine::new(ctx);

    engine.add_command(
        Command::new("ping").arg(Arg::new("value").required(true)),
        |matches, _ctx| async move {
            let value = matches
                .get_one::<String>("value")
                .expect("clap requires 'value'");
            Ok(Some(format!("pong: {value}")))
        },
    );

    #[cfg(feature = "tmux")]
    motlie_tmux::commands::register(&mut engine);

    let mut shell = InteractiveShell::new(engine)
        .with_name("motlie")
        .with_prompt("motlie> ");
    shell.run().await
}
```

### CLI Example

The first binary integration should be explicit rather than implicit:

```text
$ cargo run --bin motlie --features tmux -- repl
motlie> tmux targets
motlie> tmux create demo --size 200x50
motlie> tmux send demo echo hello
motlie> quit
```

This preserves current non-interactive behavior and avoids surprising users who already use
`motlie info`, `motlie db`, and `motlie fulltext`.

## Delivery Strategy

Because this is a greenfield product direction, the work should be delivered in focused
phases rather than framed as migration or compatibility work.

### Stage 1: New crate, no product behavior change

1. Add `libs/repl`.
2. Implement the command engine and tests.
3. Keep existing `motlie` CLI and tmux example untouched.

### Stage 2: Tmux example adoption

1. Move tmux command definitions and handlers out of `examples/repl/main.rs` into a reusable
   registration module.
2. Keep the example binary as a thin bootstrap over the new engine and interactive shell.
3. Preserve existing command names and `tui on` behavior.

### Stage 3: Main binary integration

1. Add optional `motlie-tmux` dependency and root feature flag wiring.
2. Add an explicit REPL entrypoint to `motlie`.
3. Keep current one-shot `db` and `fulltext` flows intact.

### Stage 4: Wider subsystem adoption

1. Reuse derive-based clap definitions where reasonable.
2. Add future subsystem adapters such as VMM only when those crates exist in-tree.

## Risks and Mitigations

### Risk 1: Command-definition reuse is harder than the issue suggests

Current `db` and `fulltext` commands are derive-based typed structs, while tmux REPL commands
are binary-local and manually parsed. The design addresses this by supporting both typed and
raw registration forms.

### Risk 2: Dynamic completion can create borrowing and state-lifetime problems

The design avoids completion-time mutable borrows by requiring read-oriented providers or
snapshot-backed shared state.

### Risk 3: The command engine accidentally absorbs subsystem business logic

The boundary is explicit: subsystem crates own real lifecycle behavior; the command engine
only owns session registry state, engine-local management metadata, and dispatch/completion
glue.

### Risk 4: Scope expands to an entire product-shell rewrite

The delivery is explicitly staged: first the engine, then the tmux example, then optional
main-binary integration.

### Risk 5: Attach/detach semantics become destructive or ambiguous

The design requires per-resource management metadata so a temporary admin session can detach
from imported resources without killing them, while still cleaning up resources it created
itself and closing ephemeral child attachments.

### Risk 6: Issue #150 assumes a VMM adopter that is not in this repo

The design treats VMM as future scope and removes it from the first implementation plan.

## Dependency Inventory

The intended dependency set is:

1. `reedline` for line editing, history, and editor integration.
2. `clap` for command metadata and parsing.
3. `shlex` or an equivalent small tokenizer for shell-like line splitting.
4. `thiserror` for typed library errors.
5. `tokio` for async execution at the call boundary already used across the repo.

## Open Questions

1. Should the first binary integration be `motlie repl` or `motlie tmux repl`?
2. Should built-in REPL commands live in `libs/repl` only, or may subsystem crates override
   names like `help` through namespacing?
3. Should typed registration use clap's existing `CommandFactory` and `FromArgMatches`
   directly, or should `libs/repl` wrap them behind a Motlie-specific trait for a cleaner API?
4. Should subsystem commands be required to live under explicit roots such as `tmux ...` and
   future `vmm ...`, or may flat top-level command names exist?
5. What is the first generic rehydration contract the engine should support: tmux-only attach,
   or a broader named-resource import surface for future VMM adapters?

## Summary

The proposal is feasible, but only if it is corrected to the actual repo state. The
appropriate first step is a thin Motlie-owned `libs/repl` crate on top of clap plus a
`reedline` shell frontend, with:

1. Brownfield-safe staged adoption.
2. Explicit support for dynamic completion.
3. A registration model that supports both raw clap commands and existing derive-based CLI
   types.
4. Explicit engine-local management semantics separated from subsystem business logic.
5. Attach/import/detach semantics for named live resources.
6. Tmux as the first real adopter.

This is the aligned solution for subsequent planning.
