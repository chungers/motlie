# Driver Design

## Status: Draft

## Change Log

| Date | Change |
|------|--------|
| 2026-04-10 | Rewrite the design around `libs/driver` as the primary crate. `repl` is now a frontend layer, command/resource adapters live under `driver::commands`, and feature-gated assembly happens at the application root. |

## Purpose

This document defines the architecture for `motlie-driver`.

`motlie-driver` is not "the REPL crate". It is the command/runtime layer that sits between:

- domain crates such as `vmm`, `vnet`, `vfs`, and `tmux`
- frontend layers such as `repl`, `tui`, and future terminal-buffer modules
- user applications that selectively assemble those pieces with Cargo features

The design goal is to keep lifecycle/business logic in the resource crates while moving
command adaptation, command composition, runtime state, and operator-facing orchestration
into one reusable driver layer.

## Problem

The old `libs/repl` framing was too narrow.

It captured the need for an interactive shell, but it mixed three separate concerns:

1. command parsing and execution
2. runtime state and named resource management
3. terminal UI concerns such as line editing

That shape breaks down once Motlie needs more than one frontend or more than one subsystem.

Examples:

- a one-shot CLI wants the same command tree but not a REPL loop
- a `ratatui` application wants the same command engine but not `reedline`
- a temporary admin session may want to attach to running resources and detach later
- composed commands may need to orchestrate `vmm + vnet + vfs + tmux` in one transaction

The design therefore moves from "REPL-first" to "driver-first".

## Goals

1. Make `motlie-driver` the stable home for command execution, completion, and runtime state.
2. Keep `driver::repl` as a thin `reedline` frontend over the generic engine.
3. Allow sibling frontends such as `driver::tui` and terminal-buffer helpers under `driver::term`.
4. Keep resource lifecycle and business logic in `vmm`, `vnet`, `vfs`, and `tmux`.
5. Put all command adaptation into `driver::commands::*`.
6. Support typed runtime state so commands can create resources in one step and reuse them in later steps.
7. Support static completion from `clap` and dynamic completion from live named state.
8. Support feature-gated assembly so applications only build the frontends and adapters they enable.
9. Make room for higher-level composed commands that orchestrate multiple resources in one operation.

## Non-Goals

1. Make attach/import/remote management fully implemented in the first slice.
2. Force resource crates to implement driver traits directly.
3. Make `reedline` the center of the architecture.
4. Finalize every command name or user-facing workflow in this document.

## Repo Reality Check

The current repo state matters:

1. The workspace root owns application assembly in the root [`Cargo.toml`](/home/dchung/cdx-repl/motlie/Cargo.toml).
2. The new crate scaffold lives at [`libs/driver`](/home/dchung/cdx-repl/motlie/libs/driver).
3. `motlie-driver` already has modules for:
   - `engine`
   - `completion`
   - `clap`
   - `repl`
   - `tui`
   - `term`
   - `commands::{vmm,vnet,vfs,tmux}`
4. The current scaffold is intentionally minimal. This document defines how that scaffold should grow.

## Chosen Architecture

The crate hierarchy is:

```text
motlie-driver
  engine.rs        // generic command execution and mutable runtime context
  clap.rs          // clap tree composition and parse helpers
  completion.rs    // static + dynamic completion contracts
  repl.rs          // reedline frontend
  tui.rs           // ratatui/crossterm frontend
  term.rs          // terminal buffer / emulation adapters
  commands/
    vmm.rs         // VMM command/resource adapters
    vnet.rs        // VNET command/resource adapters
    vfs.rs         // VFS command/resource adapters
    tmux.rs        // tmux command/resource adapters
```

The dependency direction is:

```text
resource crates -> consumed by driver::commands::*
driver core     -> consumed by repl / tui / term frontends
applications    -> compose driver features + command adapters
```

The critical rule is:

- resource crates must not depend on `motlie-driver`
- `motlie-driver` may depend on resource crates only through feature-gated adapter modules

## Boundaries and Responsibilities

### Domain crates

`vmm`, `vnet`, `vfs`, and `tmux` own:

- lifecycle APIs
- cleanup semantics
- business logic
- transport and protocol details
- resource-specific invariants

They do not own:

- `clap` command composition for operator workflows
- REPL loops
- driver-side naming/registry semantics
- cross-resource orchestration policy

### `motlie-driver`

`motlie-driver` owns:

- command parsing and dispatch
- typed session state
- live resource naming and lookup
- static and dynamic completion plumbing
- frontend-neutral execution
- frontend integration for REPL/TUI/terminal buffers
- orchestration commands spanning multiple resources

### `driver::commands::*`

Each adapter module owns:

- the command family shape for one subsystem
- mapping between command names and resource-crate APIs
- management semantics such as `Owned`, `Imported`, or `Ephemeral`
- registry naming and lookup policy
- completion from driver-owned runtime state
- attach/detach/import wrappers when supported

This is also the right place for composed commands.

Example:

- `driver::commands::vmm` can expose low-level `vmm boot`
- `driver::commands::workflow` can expose higher-level `workspace up`

That keeps cross-resource orchestration out of the resource crates.

## Feature Model

Applications should selectively assemble the driver through feature flags.

The current scaffold already uses the right shape:

```toml
[features]
default = []
driver = ["dep:motlie-driver"]

driver-repl = ["driver", "motlie-driver/repl"]
driver-tui = ["driver", "motlie-driver/tui"]

driver-term-vt100 = ["driver", "motlie-driver/term-vt100"]
driver-term-shadow = ["driver", "motlie-driver/term-shadow"]

driver-commands-vmm = ["driver", "motlie-driver/commands-vmm"]
driver-commands-vnet = ["driver", "motlie-driver/commands-vnet"]
driver-commands-vfs = ["driver", "motlie-driver/commands-vfs"]
driver-commands-tmux = ["driver", "motlie-driver/commands-tmux"]
```

Semantics:

1. If a frontend feature is disabled, its module is not compiled into the app.
2. If a command-adapter feature is disabled, that command family does not exist.
3. Disabled subsystems contribute no `clap` schema and no completion candidates.
4. The application root is the composition point that chooses the aggregate command set.

Important for the current `main` branch:

1. These command-adapter features do not imply that all corresponding resource crates are
   already present in the workspace.
2. On `main`, modules such as `driver::commands::vmm`, `driver::commands::vnet`, and
   `driver::commands::vfs` are scaffolding placeholders only.
3. The branch must not add workspace members or Cargo dependencies for crates that do not
   exist on `main` yet.
4. Wiring an adapter module to a concrete crate dependency happens only when that resource
   crate actually lands in the workspace.

This is the right model for specialized binaries:

- headless CLI: `driver + driver-commands-vmm`
- REPL host: `driver + driver-repl + driver-commands-vmm`
- tmux operator shell: `driver + driver-repl + driver-commands-tmux`
- future TUI admin app: `driver + driver-tui + driver-commands-vmm + driver-commands-tmux`

## Core Types

The current scaffold already points to the intended API:

```rust
pub struct CommandOutput {
    pub lines: Vec<String>,
}

#[async_trait::async_trait]
pub trait CommandSet<C>: Sized {
    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self>;
    fn complete(request: CompletionRequest<'_>, context: &C) -> Vec<CompletionCandidate>;
    async fn execute(self, context: &mut C) -> anyhow::Result<CommandOutput>;
}

pub struct CommandEngine<C, S> {
    context: C,
    _commands: PhantomData<S>,
}
```

This is intentionally typed.

The default path should not force subsystem adapters to hand-write boxed futures or dynamic
handler registries. The application already knows its command universe at compile time, so
the aggregate command set should stay typed.

## Runtime State Model

The command engine owns a mutable context value for the lifetime of the session:

```rust
pub struct AppContext {
    pub vmm: VmmSessionState,
    pub tmux: TmuxSessionState,
    pub workflows: WorkflowState,
}
```

Subsystem state stays typed:

```rust
pub struct ManagedVm {
    pub mode: ResourceMode,
    pub locality: ResourceLocality,
    pub handle: VmHandle,
}

pub struct VmmSessionState {
    guests: HashMap<String, ManagedVm>,
}

impl VmmSessionState {
    pub fn insert_guest(&mut self, name: String, vm: ManagedVm) -> anyhow::Result<()>;
    pub fn guest(&self, name: &str) -> anyhow::Result<&ManagedVm>;
    pub fn guest_mut(&mut self, name: &str) -> anyhow::Result<&mut ManagedVm>;
    pub fn guest_names(&self) -> impl Iterator<Item = &str>;
}
```

This gives the driver the same practical value people often want from a generic REPL
`Context` object, but with explicit types and subsystem boundaries.

## Ownership and Locality

The same raw resource may appear in different driver modes:

```rust
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

Interpretation:

- `Owned`: the driver created it and should clean it up explicitly
- `Imported`: the driver attached to it and should detach by default
- `Ephemeral`: short-lived child object such as a PTY or monitor
- `InProcess`: the driver holds the real handle locally
- `RemoteProxy`: the driver holds an RPC-backed proxy

Important: these are driver semantics, not domain-crate semantics.

The same `VmHandle` may be:

- `Owned + InProcess` in a local REPL host
- `Imported + InProcess` in a local attach flow
- `Imported + RemoteProxy` in a remote admin session

## Relationship Between Driver and REPL

`driver::repl` is not the engine. It is a frontend adapter.

The runtime split is:

```text
CommandEngine<C, S>
  owns parsing, dispatch, completion, and mutable context

ReplFrontend<C, S>
  owns prompt, history, editor loop, and rendering via reedline
```

The same engine can back:

- a REPL loop
- a one-shot CLI
- a TUI event handler
- a socket or admin-session ingress later

## Code Walkthrough

### 1. Define typed commands in an adapter module

`driver::commands::vmm` should define the operator-facing command family, not the `vmm` crate.

```rust
#[derive(clap::Args)]
pub struct BootGuest {
    pub name: String,
}

#[derive(clap::Args)]
pub struct ExecGuest {
    pub name: String,
    #[arg(trailing_var_arg = true)]
    pub argv: Vec<String>,
}

pub enum VmmCommand {
    Boot(BootGuest),
    Exec(ExecGuest),
}
```

### 2. Implement `CommandSet<AppContext>` for that family

```rust
#[async_trait::async_trait]
impl CommandSet<AppContext> for VmmCommand {
    fn root_command() -> clap::Command {
        clap::Command::new("vmm")
            .subcommand(BootGuest::command().name("boot"))
            .subcommand(ExecGuest::command().name("exec"))
    }

    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self> {
        match matches.subcommand() {
            Some(("boot", sub)) => Ok(Self::Boot(BootGuest::from_arg_matches(sub)?)),
            Some(("exec", sub)) => Ok(Self::Exec(ExecGuest::from_arg_matches(sub)?)),
            _ => anyhow::bail!("unsupported vmm command"),
        }
    }

    fn complete(request: CompletionRequest<'_>, ctx: &AppContext) -> Vec<CompletionCandidate> {
        if request.command_path == ["vmm", "exec"] && request.arg_id == Some("name") {
            return ctx
                .vmm
                .guest_names()
                .filter(|name| name.starts_with(request.prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect();
        }

        Vec::new()
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        match self {
            Self::Boot(cmd) => {
                let vm = motlie_vmm::boot(/* ... */).await?;
                ctx.vmm.insert_guest(
                    cmd.name.clone(),
                    ManagedVm::owned_local(vm),
                )?;
                Ok(CommandOutput::line(format!("booted {}", cmd.name)))
            }
            Self::Exec(cmd) => {
                let vm = ctx.vmm.guest(&cmd.name)?;
                let status = vm.handle.exec(&cmd.argv).await?;
                Ok(CommandOutput::line(format!("status={status}")))
            }
        }
    }
}
```

The driver adapter owns:

- command names
- registry naming
- completion behavior
- ownership semantics

The `vmm` crate still owns `boot`, `exec`, `open_pty`, `shutdown`, and related lifecycle rules.

### 3. Compose subsystem command families at the app root

The application chooses which command families exist.

```rust
pub enum AppCommand {
    #[cfg(feature = "driver-commands-vmm")]
    Vmm(VmmCommand),

    #[cfg(feature = "driver-commands-tmux")]
    Tmux(TmuxCommand),
}

#[async_trait::async_trait]
impl CommandSet<AppContext> for AppCommand {
    fn root_command() -> clap::Command {
        let mut root = clap::Command::new("motlie");

        #[cfg(feature = "driver-commands-vmm")]
        {
            root = root.subcommand(VmmCommand::root_command());
        }

        #[cfg(feature = "driver-commands-tmux")]
        {
            root = root.subcommand(TmuxCommand::root_command());
        }

        root
    }

    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self> {
        match matches.subcommand() {
            #[cfg(feature = "driver-commands-vmm")]
            Some(("vmm", sub)) => Ok(Self::Vmm(VmmCommand::from_matches(sub)?)),

            #[cfg(feature = "driver-commands-tmux")]
            Some(("tmux", sub)) => Ok(Self::Tmux(TmuxCommand::from_matches(sub)?)),

            _ => anyhow::bail!("unknown command"),
        }
    }

    fn complete(request: CompletionRequest<'_>, ctx: &AppContext) -> Vec<CompletionCandidate> {
        let mut out = Vec::new();

        #[cfg(feature = "driver-commands-vmm")]
        out.extend(VmmCommand::complete(request, ctx));

        #[cfg(feature = "driver-commands-tmux")]
        out.extend(TmuxCommand::complete(request, ctx));

        out
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        match self {
            #[cfg(feature = "driver-commands-vmm")]
            Self::Vmm(cmd) => cmd.execute(ctx).await,

            #[cfg(feature = "driver-commands-tmux")]
            Self::Tmux(cmd) => cmd.execute(ctx).await,
        }
    }
}
```

### 4. The engine runs typed commands

This is the generic core:

```rust
let mut engine = CommandEngine::<AppContext, AppCommand>::new(app_context);
engine.run_line("vmm boot alice").await?;
engine.run_line("vmm exec alice uname -a").await?;
```

The engine is responsible for:

- tokenizing shell-like input
- calling `clap`
- building the typed command enum
- executing it against mutable context

### 5. `clap` provides static structure

Static completion comes from the `clap::Command` tree:

- root command names
- subsystem names such as `vmm` and `tmux`
- subcommands such as `boot`, `exec`, `shutdown`
- flags and known static choices

This is why `clap` belongs in the driver layer, not inside `driver::repl`.

One-shot CLI, REPL, TUI, and future socket ingress all need the same command schema.

### 6. The driver adds dynamic completion

Dynamic completion uses live runtime state from the engine context.

Example:

```rust
fn complete(request: CompletionRequest<'_>, ctx: &AppContext) -> Vec<CompletionCandidate> {
    if request.command_path == ["tmux", "send"] && request.arg_id == Some("target") {
        return ctx
            .tmux
            .target_names()
            .filter(|name| name.starts_with(request.prefix))
            .map(|name| CompletionCandidate::new(name))
            .collect();
    }

    Vec::new()
}
```

That is the right split:

- `clap` describes the shape of commands
- the driver resolves which argument is being completed
- the adapter asks typed runtime state for live values

### 7. `driver::repl` integrates `reedline`

The REPL layer stays thin:

```rust
let engine = CommandEngine::<AppContext, AppCommand>::new(context);
let mut repl = ReplFrontend::new(engine)
    .with_name("motlie")
    .with_prompt("motlie> ");

repl.run().await?;
```

`driver::repl` should do only REPL-specific work:

- prompt management
- history
- editor loop
- mapping `reedline` completion requests into `CommandEngine::complete`
- rendering command output and errors

It should not own subsystem lifecycle logic.

## Static and Dynamic Completion Design

The completion flow should be:

1. `reedline` asks the REPL frontend for completions at a cursor position.
2. `driver::repl` forwards that request to `CommandEngine::complete`.
3. `CommandEngine` determines the current token span and command path.
4. The driver walks the `clap` tree for static candidates.
5. The active adapter module contributes dynamic candidates from typed runtime state.
6. The engine merges, deduplicates, and sorts the results.

This keeps the expensive or tricky logic in one place.

## Composed Commands

A major benefit of `driver::commands::*` is that the layer can define commands spanning
multiple resource crates.

Example shape:

```rust
pub enum WorkspaceCommand {
    Up(WorkspaceUp),
    Down(WorkspaceDown),
}
```

`workspace up demo` can:

1. allocate a network
2. boot a VM
3. attach a filesystem backing
4. create a tmux session
5. register all resulting resources under one typed record
6. roll back partial state if step 4 fails

That orchestration belongs in the driver layer because it is:

- higher-level than any individual resource crate
- still tied to operator workflows and command runtime state

## First Implementation Slice

The first implementation slice should stay strict:

- `Owned + InProcess` resources only
- typed command families only
- typed session registries only
- REPL frontend only
- no remote transport yet

This is enough to validate:

- driver/runtime boundaries
- command composition
- resource naming
- explicit cleanup
- static and dynamic completion

Attach/import/remote management stay documented design targets, but not the first code path.

## Risks

### Risk: driver becomes a dumping ground

Mitigation:

- keep business logic in resource crates
- keep frontend code in `repl` or `tui`
- keep subsystem-specific command logic in `driver::commands::*`

### Risk: dynamic completion leaks mutable state concerns

Mitigation:

- completion must use read-oriented access to typed session state
- short-lived snapshot extraction is preferred for complex frontends

### Risk: command composition becomes too dynamic

Mitigation:

- keep the default path typed through `CommandSet<C>`
- use dynamic dispatch only if a later frontend genuinely requires it

## Summary

The chosen design is:

1. rename `libs/repl` to `libs/driver`
2. make `motlie-driver` the generic command/runtime layer
3. keep `driver::repl` as a thin `reedline` frontend
4. put subsystem and orchestration adapters under `driver::commands::*`
5. assemble applications through feature-gated driver modules

That architecture keeps the hard lifecycle logic where it belongs, while giving Motlie one
clean place to build REPLs, TUIs, one-shot CLIs, and future admin-session frontends on top
of the same runtime.
