# Unified REPL Engine API Contract

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-10 | @codex-repl: Add the feature-gating contract for subsystem registration so optional crates can be selectively compiled and included in the command surface. | Principles, Subsystem-Facing Surface, What A Subsystem Must Provide |
| 2026-04-10 | @codex-repl: Initial API contract for subsystem integration into `libs/repl`. Documents the subsystem-facing command-registration model, managed-resource adapter responsibilities, and a concrete VMM integration contract across owned, attached, and remote-management scenarios. | All |

This document complements [`DESIGN.md`](./DESIGN.md) and [`LIFECYCLE.md`](./LIFECYCLE.md).
`DESIGN.md` explains the architecture. This document explains the subsystem-facing API
contract.

## Principles

1. Subsystem crates own lifecycle and business logic.
2. The command engine owns command registration, session registry state, completion wiring,
   and frontend-neutral dispatch.
3. Subsystem authors should not have to hand-write `Pin<Box<dyn Future<...>>>` or dynamic
   handler registries to register a command.
4. Raw handle types such as `VmHandle` should not be forced to implement REPL traits
   directly.
5. Integration should happen through typed command enums plus typed adapter records.
6. Optional subsystems must disappear cleanly from both the build graph and the command
   surface when their Cargo feature is disabled.
7. The first vertical slice is `Owned + InProcess` only; attach and remote-management
   scenarios are documented now but are not part of the first implementation milestone.

## Subsystem-Facing Surface

The intended public integration surface is:

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
```

Subsystem command enums are expected to be feature-gated by the composing binary or
integration crate:

```rust
#[cfg(feature = "repl-vmm")]
pub enum VmmCommand {
    Boot(BootGuest),
    Exec(ExecGuest),
    Shutdown(ShutdownGuest),
}

#[cfg(feature = "repl-vmm")]
pub enum AppCommand {
    Vmm(VmmCommand),
}
```

If the feature is disabled:

1. the subsystem dependency is not linked,
2. the subsystem command enum does not compile,
3. the aggregate clap tree does not contain that command family, and
4. neither static nor dynamic completion exposes that subsystem's values.

The intended managed-resource surface is:

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

The phase-1 engine should not need to erase these concrete command types into dynamic
dispatch objects. The binary knows the full feature-enabled command universe at compile time,
so the command set should stay typed.

## Typed Session Registry Contract

The command engine stores resources created by one command so that later commands can access
them through typed subsystem state.

The intended pattern is:

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

This is the mechanism for command sequencing:

1. one command creates a resource and inserts it under a stable key,
2. later commands fetch it through a typed accessor,
3. the registry remains typed throughout, and
4. any reflection or generic listing surface is produced from explicit metadata views rather
   than erased object storage.

## What A Subsystem Must Provide

For a subsystem to integrate cleanly, it should provide the following through either its own
crate or a thin integration module:

| Capability | Required | Purpose |
|------------|----------|---------|
| Typed clap command structs | Yes | Reuse the same command schema for REPL and other frontends. |
| Typed subsystem command enum | Yes | Group related commands into a compile-time command family such as `VmmCommand`. |
| `CommandSet<C>` implementation | Yes | Define parsing, execution, and optional dynamic completion for the command family. |
| App-level command composition | Yes at the binary root | Compose enabled subsystem enums into one aggregate command set. |
| Cargo feature gate for the command family | Yes for optional subsystems | Allows the root binary to compile and include the subsystem selectively. |
| Named managed-resource adapters | Yes for live resources | Tell the engine whether a resource is `Owned`, `Imported`, or `Ephemeral`, and whether it is local or remote. |
| Cleanup semantics | Yes for live resources | Define what engine `close()` and `detach()` mean truthfully. |
| Import / rehydrate support | Optional, scenario-dependent | Required only if the subsystem supports attaching to already-running resources. |
| Remote proxy support | Optional, scenario-dependent | Required only if the subsystem wants remote admin-session management. |
| Invalidation rules | Yes for parent/child resources | Ensure stale children are dropped after restart, disconnect, or shutdown. |

## VMM Integration Contract

VMM is a good example because it has both durable resources such as `VmHandle` and ephemeral
children such as `GuestPtySession`.

The raw `motlie_vmm` crate should continue to own:

1. `prepare(...)`
2. `boot(...)`
3. `ready()`
4. `exec()`
5. `open_pty()`
6. `shutdown()`

The REPL integration layer should own:

1. command-family composition under paths such as `vmm boot`, `vmm exec`, `vmm shutdown`
2. registry naming such as `alice`, `demo`, or `ci-runner`
3. `ResourceMode` and `ResourceLocality`
4. completion from the named registry
5. close vs detach semantics at engine shutdown

### VMM By Resource

| Resource | What the raw crate already provides | What the REPL integration layer must add |
|----------|-------------------------------------|------------------------------------------|
| `VmHandle` | `ready()`, `exec()`, `open_pty()`, `shutdown()` | Stable registry name, mode/locality metadata, completion identity, import/remote adapter if supported, and close/detach policy. |
| `GuestPtySession` | `send()`, `send_line()`, `resize()`, `transcript()`, `close()` | Parent linkage to a named VM, ephemeral lifecycle classification, completion if named, and invalidation on VM restart or control-plane loss. |
| `GuestFsHandle` / `MotlieVfsHandle` | provision / readiness / shutdown through VMM backing | Usually no top-level registration; treat as subordinate VM backing unless intentionally surfaced later. |
| `GuestBridgeHandle` / `MotlieSshProxyHandle` | control-plane readiness / exec / open_pty / shutdown | Usually subordinate to VM lifecycle; remote admin sessions should route through the VM integration layer instead of exposing this as a separate root. |

### VMM Scenario Contract

| Resource | Scenario | Engine mode / locality | What the VMM integration must provide | Engine shutdown behavior |
|----------|----------|------------------------|---------------------------------------|--------------------------|
| `VmHandle` | `repl owns` | `Owned + InProcess` | Boot command, registry insert by name, guest-name completion, and close path calling `shutdown().await`. | Destroy the VM through VMM APIs. |
| `VmHandle` | `repl attaches` | `Imported + InProcess` | Discovery or import adapter that can rebuild a usable local handle from runtime artifacts or discovery state. | Detach only; do not destroy the VM. |
| `VmHandle` | `repl manages remotely` | `Imported + RemoteProxy` | Remote client or proxy exposing the VM control surface needed by the command engine. | Drop proxy / detach session only by default. |
| `GuestPtySession` | `repl owns` | `Ephemeral + InProcess` | Open-PTY command, optional naming, parent VM link, and explicit `close().await` path. | Close the PTY session. |
| `GuestPtySession` | `repl attaches` | Normally unsupported as first-class import | Recreate from imported VM rather than importing transcript or thread state. | Close recreated local PTY only. |
| `GuestPtySession` | `repl manages remotely` | `Ephemeral + RemoteProxy` | Remote PTY token or proxy API, plus close semantics scoped to the child session. | Close the remote PTY child only. |

### VMM Command Set Example

The integration layer is expected to look roughly like this:

```rust
pub enum VmmCommand {
    Boot(BootGuest),
    Exec(ExecGuest),
    OpenGuestPty(OpenGuestPty),
    Shutdown(ShutdownGuest),
}

#[async_trait::async_trait]
impl CommandSet<AppContext> for VmmCommand {
    fn root_command() -> clap::Command {
        clap::Command::new("vmm")
            .subcommand(BootGuest::command().name("boot"))
            .subcommand(ExecGuest::command().name("exec"))
            .subcommand(OpenGuestPty::command().name("pty-open"))
            .subcommand(ShutdownGuest::command().name("shutdown"))
    }

    fn from_matches(matches: &clap::ArgMatches) -> anyhow::Result<Self> {
        # todo!()
    }

    fn complete(request: CompletionRequest<'_>, ctx: &AppContext) -> Vec<CompletionCandidate> {
        # todo!()
    }

    async fn execute(self, ctx: &mut AppContext) -> anyhow::Result<CommandOutput> {
        # todo!()
    }
}
```

### VMM Contract Checklist

The VMM integration is complete enough for the REPL/command engine when the answer is clear
for each row below:

| Question | Expected answer |
|----------|-----------------|
| How are VMs named in the engine registry? | A stable alias chosen at boot/import time. |
| How does `vmm exec <name>` resolve the target VM? | Through the engine registry, not ad hoc global lookup. |
| How are guest names completed? | From the engine registry through `CommandSet::complete`. |
| What does engine `close()` do for an owned VM? | Call `VmHandle::shutdown().await`. |
| What does engine `close()` do for an imported VM? | Detach registry state only unless the user explicitly requested destructive cleanup. |
| What invalidates named PTYs? | PTY close, VM restart, guest shutdown, or control-plane disconnect. |
| How does remote management differ from local attach? | The adapter wraps a remote proxy and uses `RemoteProxy` locality; engine semantics stay the same. |

## Relationship To LIFECYCLE

[`LIFECYCLE.md`](./LIFECYCLE.md) documents what resources exist and how they behave.
This document explains what an integrating subsystem must provide so those resources can be
used by the command engine truthfully.
