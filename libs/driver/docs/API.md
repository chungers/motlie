# Driver API Contract

## Status: Draft

## Change Log

| Date | Change |
|------|--------|
| 2026-04-10 | Rewrite the subsystem-facing contract for `motlie-driver`. The API is now driver-first, with adapter modules under `driver::commands::*` and frontend layers such as `driver::repl` consuming the generic engine. |

## Scope

[`DESIGN.md`](./DESIGN.md) explains the architecture. This document explains what the driver
expects from:

- applications composing command families
- adapter modules such as `driver::commands::vmm`
- domain crates such as `vmm`

The core rule is simple:

- domain crates do not implement driver contracts directly
- driver adapter modules consume domain-crate public APIs and implement driver semantics

For the current `main` branch, that also means:

- adapter modules may exist as placeholders before the corresponding resource crate lands
- placeholder modules must not force nonexistent workspace members or Cargo dependencies
- docs may describe future adapters such as VMM, but the scaffold on `main` must remain
  dependency-safe

## Core Driver Surface

The intended generic surface is:

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
pub trait CommandSet<C>: Sized {
    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self>;
    fn complete(request: CompletionRequest<'_>, context: &C) -> Vec<CompletionCandidate>;
    async fn execute(self, context: &mut C) -> anyhow::Result<CommandOutput>;
}

pub struct CommandEngine<C, S> {
    /* generic runtime state */
}
```

This contract is intentionally typed and frontend-neutral.

## What An Application Must Provide

The application root owns composition.

It must provide:

| Capability | Required | Notes |
|------------|----------|-------|
| Aggregate context type such as `AppContext` | Yes | Holds typed subsystem registries and any app-level orchestration state. |
| Aggregate command enum such as `AppCommand` | Yes | Composes the enabled command families into one typed surface. |
| Feature-gated command-family inclusion | Yes | Disabled subsystems must disappear from build, parsing, and completion. |
| Frontend assembly | Yes | Example: choose `ReplFrontend` or `TuiFrontend` and pass in a `CommandEngine`. |

## What A Driver Adapter Module Must Provide

Each adapter module under `driver::commands::*` is responsible for one command family.

For example, `driver::commands::vmm` should provide:

| Capability | Required | Purpose |
|------------|----------|---------|
| Typed command structs or subcommand structs | Yes | Reuse `clap` declarative schema. |
| Typed family enum such as `VmmCommand` | Yes | Gives the family one typed execution surface. |
| `CommandSet<C>` implementation | Yes | Parsing, execution, and completion for that family. |
| Registry mapping policy | Yes | Decide how names map to live resources. |
| Ownership/locality semantics | Yes | Mark handles as `Owned`, `Imported`, or `Ephemeral`; `InProcess` or `RemoteProxy`. |
| Cleanup semantics | Yes | Define what engine `close()` and `detach()` mean for that family. |
| Invalidation rules | Yes | Explain how child handles become stale after restart/disconnect. |
| Import/rehydrate support | Optional | Needed only if the subsystem supports attach flows. |
| Remote proxy support | Optional | Needed only if the subsystem supports admin-session management from another process. |

## Management Semantics

The driver layer should use explicit metadata around raw handles:

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

pub struct ManagedResource<T> {
    pub mode: ResourceMode,
    pub locality: ResourceLocality,
    pub value: T,
}
```

The adapter layer defines the semantics, not the raw resource type.

That matters because the same `VmHandle` may be:

- created locally by the REPL host
- imported later by a local attach flow
- represented remotely by an admin proxy

## Typed Registry Contract

The command engine should never have to downcast erased resources just to execute the next
command. Registries should stay typed.

Example:

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

That gives the driver:

1. deterministic name lookup
2. dynamic completion from live state
3. explicit cleanup on close
4. a clean place for attach/import semantics later

## `clap` and Completion Contract

The split is:

- `clap` defines static structure
- the adapter defines runtime completion
- the engine merges both

Adapter responsibilities:

| Completion concern | Who owns it |
|--------------------|-------------|
| Root commands, subcommands, flags | `clap` tree built by `root_command()` |
| Static argument choices | `clap` metadata |
| Live names such as guest aliases or tmux targets | adapter `complete(...)` implementation |
| Candidate merge, de-duplication, sorting | driver core |
| Mapping editor/TUI completion events into `CompletionRequest` | frontend layer |

## VMM Contract

`vmm` is the clearest example because it already has a strong lifecycle surface.

### What the raw `vmm` crate should continue to own

The raw crate owns lifecycle and business logic:

- `prepare(...)`
- `boot(...)`
- `ready()`
- `exec(...)`
- `open_pty(...)`
- `shutdown(...)`

The driver must consume those APIs. It should not reimplement their semantics.

### What `driver::commands::vmm` must add

The VMM adapter owns:

- user-facing command names such as `vmm boot`
- naming of live VMs in the session registry
- dynamic completion for guest names
- close/detach behavior at driver shutdown
- child-resource invalidation rules for PTYs and similar attachments
- import/remote proxy wrappers if and when those scenarios are implemented

### VMM resource contract by type

| Resource | Raw crate lifecycle | Driver adapter responsibility |
|----------|---------------------|-------------------------------|
| `VmHandle` | `ready()`, `exec()`, `open_pty()`, `shutdown()` | Stable registry identity, `Owned/Imported` semantics, guest-name completion, close/detach policy, optional import/proxy adapter |
| `GuestPtySession` | `send()`, `send_line()`, `resize()`, `transcript()`, `close()` | Parent linkage to named VM, `Ephemeral` classification, optional naming, invalidation on VM restart or close |
| `GuestFsHandle` / VFS backing | provision/readiness/shutdown through VMM internals | Usually subordinate to VM lifecycle, not a top-level user-facing driver resource |
| `GuestBridgeHandle` / SSH control plane | readiness, `exec`, `open_pty`, `shutdown` | Usually subordinate to VM lifecycle, not a top-level driver root |

### VMM contract by scenario

| Resource | `repl owns` | `repl attaches` | `repl manages remotely` |
|----------|-------------|-----------------|-------------------------|
| `VmHandle` | Store as `Owned + InProcess`; engine close calls `shutdown().await`. | Store as `Imported + InProcess`; engine close detaches by default. | Store as `Imported + RemoteProxy`; engine close drops proxy/detaches by default. |
| `GuestPtySession` | Store as `Ephemeral + InProcess`; engine close calls `close().await`. | Usually recreate from imported VM instead of importing PTY state. | Store as `Ephemeral + RemoteProxy`; engine close closes the remote child session only. |

### VMM adapter example

```rust
pub enum VmmCommand {
    Boot(BootGuest),
    Exec(ExecGuest),
    Shutdown(ShutdownGuest),
}

#[async_trait::async_trait]
impl CommandSet<AppContext> for VmmCommand {
    fn root_command() -> clap::Command {
        clap::Command::new("vmm")
            .subcommand(BootGuest::command().name("boot"))
            .subcommand(ExecGuest::command().name("exec"))
            .subcommand(ShutdownGuest::command().name("shutdown"))
    }

    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self> {
        match matches.subcommand() {
            Some(("boot", sub)) => Ok(Self::Boot(BootGuest::from_arg_matches(sub)?)),
            Some(("exec", sub)) => Ok(Self::Exec(ExecGuest::from_arg_matches(sub)?)),
            Some(("shutdown", sub)) => Ok(Self::Shutdown(ShutdownGuest::from_arg_matches(sub)?)),
            _ => anyhow::bail!("unknown vmm command"),
        }
    }

    fn complete(request: CompletionRequest<'_>, ctx: &AppContext) -> Vec<CompletionCandidate> {
        match (request.command_path, request.arg_id) {
            (["vmm", "exec"], Some("name")) | (["vmm", "shutdown"], Some("name")) => ctx
                .vmm
                .guest_names()
                .filter(|name| name.starts_with(request.prefix))
                .map(CompletionCandidate::new)
                .collect(),
            _ => Vec::new(),
        }
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        match self {
            Self::Boot(cmd) => {
                let handle = motlie_vmm::boot(/* ... */).await?;
                ctx.vmm.insert_guest(cmd.name.clone(), ManagedVm::owned_local(handle))?;
                Ok(CommandOutput::line(format!("booted {}", cmd.name)))
            }
            Self::Exec(cmd) => {
                let vm = ctx.vmm.guest(&cmd.name)?;
                vm.handle.exec(&cmd.argv).await?;
                Ok(CommandOutput::default())
            }
            Self::Shutdown(cmd) => {
                let mut vm = ctx.vmm.remove_guest(&cmd.name)?;
                vm.handle.shutdown().await?;
                Ok(CommandOutput::line(format!("stopped {}", cmd.name)))
            }
        }
    }
}
```

### VMM checklist

The VMM adapter is sufficiently defined when these questions have concrete answers:

| Question | Expected answer |
|----------|-----------------|
| How is a VM named in the driver? | By a stable alias chosen at create/import time. |
| How does `vmm exec <name>` resolve its target? | Through the typed VMM registry. |
| Where do guest-name completion candidates come from? | From the typed VMM registry. |
| What does engine close do to an owned VM? | `shutdown().await` through the VMM API. |
| What does engine close do to an imported VM? | Detach local state only by default. |
| What invalidates a PTY child? | PTY close, VM restart, guest shutdown, or control-plane loss. |

## VNET, VFS, and tmux Expectations

The same pattern applies to the other adapter modules:

| Adapter module | Main driver-facing resource expectation |
|----------------|-----------------------------------------|
| `driver::commands::vnet` | Treat `VnetHandle` as a strong first-class managed resource when used standalone; treat it as subordinate when managed through VMM. |
| `driver::commands::vfs` | Usually wrap VFS serving in a higher-level owned record rather than managing raw `FsServer` directly. |
| `driver::commands::tmux` | Distinguish remote resource references such as `Target` from local child monitors and buses; only destroy tmux entities the session actually owns. |

Detailed lifecycle inventory remains in [`LIFECYCLE.md`](./LIFECYCLE.md).
