# Unified REPL Engine API Contract

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-10 | @codex-repl: Initial API contract for subsystem integration into `libs/repl`. Documents the subsystem-facing command-registration model, managed-resource adapter responsibilities, and a concrete VMM integration contract across owned, attached, and remote-management scenarios. | All |

This document complements [`DESIGN.md`](./DESIGN.md) and [`LIFECYCLE.md`](./LIFECYCLE.md).
`DESIGN.md` explains the architecture. This document explains the subsystem-facing API
contract.

## Principles

1. Subsystem crates own lifecycle and business logic.
2. The command engine owns command registration, session registry state, completion wiring,
   and frontend-neutral dispatch.
3. Subsystem authors should not have to hand-write `Pin<Box<dyn Future<...>>>` to register a
   command.
4. Raw handle types such as `VmHandle` should not be forced to implement REPL traits
   directly.
5. Integration should happen through typed commands plus adapter records.

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
```

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

The engine may internally erase these concrete command types into dynamic dispatch objects,
but subsystem crates should only need to implement typed commands and adapter records.

## What A Subsystem Must Provide

For a subsystem to integrate cleanly, it should provide the following through either its own
crate or a thin integration module:

| Capability | Required | Purpose |
|------------|----------|---------|
| Typed clap command structs | Yes | Reuse the same command schema for REPL and other frontends. |
| `TypedCommand<C>` implementations | Yes | Define execution logic and optional dynamic completion. |
| A registration module or function | Yes | Add all subsystem commands to the command engine. |
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

1. command registration under paths such as `vmm boot`, `vmm exec`, `vmm shutdown`
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
pub struct VmmModule;

impl CommandModule<AppContext> for VmmModule {
    fn register(self, engine: &mut CommandEngine<AppContext>) {
        engine.add_typed::<BootGuest>();
        engine.add_typed::<ExecGuest>();
        engine.add_typed::<OpenGuestPty>();
        engine.add_typed::<ShutdownGuest>();
    }
}
```

And one command implementation should look like this:

```rust
#[derive(clap::Parser)]
pub struct OpenGuestPty {
    #[arg(value_name = "NAME")]
    pub guest: String,
}

#[async_trait::async_trait]
impl TypedCommand<AppContext> for OpenGuestPty {
    const PATH: &'static [&'static str] = &["vmm", "pty-open"];

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
        let pty = ctx
            .vmm
            .guests
            .get_mut(&self.guest)
            .ok_or_else(|| anyhow::anyhow!("unknown guest {}", self.guest))?
            .value
            .open_pty(Default::default())
            .await?;

        ctx.vmm.ptys.insert(
            format!("{}:pty", self.guest),
            ManagedResource {
                mode: ResourceMode::Ephemeral,
                locality: ResourceLocality::InProcess,
                value: pty,
            },
        );

        Ok(CommandOutput {
            lines: vec![format!("opened pty for {}", self.guest)],
        })
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
| How are guest names completed? | From the engine registry through `TypedCommand::complete`. |
| What does engine `close()` do for an owned VM? | Call `VmHandle::shutdown().await`. |
| What does engine `close()` do for an imported VM? | Detach registry state only unless the user explicitly requested destructive cleanup. |
| What invalidates named PTYs? | PTY close, VM restart, guest shutdown, or control-plane disconnect. |
| How does remote management differ from local attach? | The adapter wraps a remote proxy and uses `RemoteProxy` locality; engine semantics stay the same. |

## Relationship To LIFECYCLE

[`LIFECYCLE.md`](./LIFECYCLE.md) documents what resources exist and how they behave.
This document explains what an integrating subsystem must provide so those resources can be
used by the command engine truthfully.
