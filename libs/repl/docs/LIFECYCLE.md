# Command Engine Lifecycle Inventory

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-10 | @codex-repl | Initial lifecycle inventory for command-engine design. Documents current resource types across `vmm`, `vnet`, `vfs`, and `tmux`, their public lifecycle APIs, engine-local management semantics, cleanup expectations, and attach/rehydrate implications for a temporary admin shell or command engine. |

This document records the resource lifecycle inputs for the command engine design in
[`DESIGN.md`](./DESIGN.md).

Scope notes:

1. This is a design input document, not an implementation artifact list.
2. The `tmux` review is from the current `main` snapshot.
3. The `vmm`, `vnet`, and `vfs` review is based on the actively reviewed `feature/vmm`
   worktree as of 2026-04-10; some of those types are future adopters rather than current
   `main` dependencies.
4. The first implementation slice still targets only the `Owned + InProcess` scenario; the
   imported and remote-management scenarios below are future-facing design targets.

## Engine-Local Management Modes

The command engine must distinguish between resource lifecycle owned by the subsystem crate
and management semantics owned by the engine session. The same raw type may appear in
different engine modes depending on how the session acquired it.

| Mode | Meaning in engine state | Engine `close()` behavior |
|------|-------------------------|---------------------------|
| `Owned` | The engine session created the resource or explicitly claimed cleanup responsibility for it. | Attempt explicit subsystem cleanup through the crate API. |
| `Imported` | The resource predated the engine session or was discovered/rehydrated from outside the engine. | Detach local references and stop local helper tasks; do not destroy the underlying external resource by default. |
| `Ephemeral` | The resource is a short-lived child attachment layered on a longer-lived parent resource. | Close the local child object unconditionally and be prepared to recreate it. |

The engine should also record locality because the same management mode may exist either as a
live in-process handle or as an RPC-backed proxy imported from another process:

| Locality | Meaning |
|----------|---------|
| `InProcess` | The engine holds the real crate handle in the current process. |
| `RemoteProxy` | The engine holds a proxy/reference to a resource managed by another process. |

This distinction is what allows a temporary admin shell to attach to already-running
resources and later disappear without leaving them in an accidental or inconsistent state.

## Crate Summary

| Crate | Main live resources | Explicit cleanup today | Rehydrate today | Notes |
|-------|---------------------|------------------------|-----------------|-------|
| `vmm` | `VmHandle`, `GuestPtySession` | Yes | Partial / future adapter needed | Best overall lifecycle surface; explicit shutdown is first-class. |
| `vnet` | `VnetHandle` | Yes | No direct rehydrate API | Strong standalone handle with explicit shutdown plus best-effort `Drop`. |
| `vfs` | `FsServer`, `MountHandles` | Weak / mixed | Limited | More service-oriented than handle-oriented; caller often owns transport/listener policy. |
| `tmux` | `HostHandle`, `Target`, monitor handles, `Fleet`, `OutputBus` | Mixed | Good for host/session rediscovery | Resource references are easy to rediscover; top-level host handle has no explicit shutdown. |

## VMM

| Resource | Create / Acquire API | Steady-State API | Cleanup API | Engine Role | Duration / Invalidation | Rehydrate Notes |
|----------|----------------------|------------------|-------------|-------------|-------------------------|-----------------|
| `GuestSpec`, `PrepareRequest`, `PreparedGuest` | `prepare(...)` and spec construction | Data only | None | Ephemeral config / launch material | Short-lived setup artifacts | Recomputed, not rehydrated |
| `VmHandle` | `boot(prepared, services)` | `ready()`, `exec()`, `open_pty()`, `observability()` | `shutdown().await` | Primary named VM control object | Medium / long-lived; invalid after shutdown or guest exit | No first-class generic rehydrate API today; future adapter would need runtime-path discovery and handle reconstruction |
| `GuestPtySession` | `VmHandle::open_pty(...)` | `send()`, `send_line()`, `resize()`, `read_for()`, `read_until_contains()`, `transcript()` | `close().await` | Short-lived named child object | Short-lived; invalid after PTY close, VM restart, or control-plane loss | Do not rehydrate; recreate from a live `VmHandle` |
| `GuestFsHandle` / `MotlieVfsHandle` | `GuestFsHandle::provision(...)` or backing provision | `wait_until_ready()` / `wait_ready()`, mount introspection | `shutdown()` | Usually subordinate to `VmHandle`, not directly named in engine state | Long-lived while guest backing is active | Normally managed through `VmHandle::shutdown()` |
| `GuestBridgeHandle` / `MotlieSshProxyHandle` | `spawn_guest_ssh_bridge(...)` or backing provision | `wait_ready()`, `exec()`, `open_pty()` | `shutdown()` | Usually subordinate to `VmHandle` | Long-lived while guest control plane is active | Normally managed through `VmHandle::shutdown()` |
| `MotlieVnetHandle` | backing `provision(...)` | no user API beyond health through parent | `shutdown()` | Usually subordinate to `VmHandle` | Long-lived while guest networking is active | Normally managed through `VmHandle::shutdown()` |

VMM command-engine semantics:

1. `VmHandle` is the named durable resource.
2. `GuestPtySession` is a short-lived attachment layered on top of a VM.
3. A VM restart should invalidate all child PTY sessions and any cached control-plane
   assumptions.
4. Attach-to-existing-VM is desirable, but it is not yet a generic API surface in the crate.

## VNET

| Resource | Create / Acquire API | Steady-State API | Cleanup API | Engine Role | Duration / Invalidation | Rehydrate Notes |
|----------|----------------------|------------------|-------------|-------------|-------------------------|-----------------|
| `VnetConfig` | `VnetConfig::builder()` | Data only | None | Ephemeral config | Short-lived setup input | Recomputed, not rehydrated |
| `VnetBackend` | `VnetBackend::new(config)` | `start()` | None | Construction helper | Short-lived setup helper | Recomputed, not rehydrated |
| `VnetHandle` | `VnetBackend::start()` | `is_alive()` | `shutdown()` plus best-effort `Drop` | Named standalone network backend if exposed directly; otherwise subordinate to VMM | Long-lived while backend threads and socket remain live | No direct rehydrate API today |

VNET command-engine semantics:

1. If used under VMM, treat `VnetHandle` as subordinate to the VM lifecycle.
2. If exposed standalone later, it can be a first-class named engine resource because it has
   a strong explicit shutdown path.

## VFS

| Resource | Create / Acquire API | Steady-State API | Cleanup API | Engine Role | Duration / Invalidation | Rehydrate Notes |
|----------|----------------------|------------------|-------------|-------------|-------------------------|-----------------|
| `FsServer` | `FsServer::builder().build()` | `handle_op()`, `add_mount()`, `remove_mount()`, `subscribe_events()` | None | Long-lived in-process service object; usually wrapped by a listener/task | Long-lived while process-local serving remains active | Caller can rebuild or rediscover surrounding transport, but there is no unified rehydrate API |
| `VsockConnectionHandler` | `VsockConnectionHandler::new(server, tag)` | `serve(stream)` | Connection ends on stream close | Per-connection helper, not usually a named engine resource | Short-lived per transport connection | Recreated per connection |
| `GuestMountRunner` | `GuestMountRunner::new(specs)` | `mount_all()` | None | Guest-side bootstrap helper | Short-lived setup helper | Recomputed, not rehydrated |
| `MountHandles` | `mount_all()` / `mount_all_stub()` | `join_all()` | None beyond external unmount and thread exit | Usually not a command-engine-managed resource | Long-lived until mount threads exit | Not suitable as a strong attach/detach resource on its own |

VFS command-engine semantics:

1. `FsServer` is more of a local service than a traditional handle.
2. On the host side, command-engine ownership should usually sit above VFS in a wrapper that
   also owns the listener/task lifecycle.
3. In VMM usage, VFS should normally be treated as subordinate VM backing rather than a
   top-level engine resource.

## TMUX

| Resource | Create / Acquire API | Steady-State API | Cleanup API | Engine Role | Duration / Invalidation | Rehydrate Notes |
|----------|----------------------|------------------|-------------|-------------|-------------------------|-----------------|
| `HostHandle` | `SshConfig::connect()`, `HostHandle::local()`, `HostHandle::new()` | discovery, create session, target resolution, transfer, monitoring startup | No explicit `shutdown()` | Named root attachment to a tmux-capable host | Long-lived process-local capability object | Easy to recreate from saved connection config |
| `Target` | `HostHandle::create_session()`, `session()`, `target()` | navigation, I/O, capture, `kill()`, `rename()`, `start_monitoring()`, `start_exec()` | `kill()` destroys the remote tmux entity when appropriate | Primary named tmux resource reference | Medium-lived; invalidated by remote rename/kill/session changes | Good rediscovery surface through host/session/window/pane queries |
| `SessionMonitorHandle` | `HostHandle::start_monitoring_session()` or `Target::start_monitoring()` | `health()`, `is_active()` | `shutdown().await` | Named local monitor attachment | Medium-lived local task; invalidated by session disappearance or explicit shutdown | Recreate after rediscovering the target |
| `MonitorHandle` | `HostHandle::start_monitoring()` | `get()`, per-session inspection | `shutdown().await`, `stop_session().await` | Aggregate monitor owner | Medium-lived local task set | Recreate from host/session discovery |
| `ExecHandle` | `Target::start_exec()` | `status()`, `wait().await` | Consumed by completion / task end | Short-lived child object | Short-lived until command completion or discontinuity | Do not rehydrate; recreate |
| `Fleet` | `Fleet::new()`, `register()` | host registration, monitoring, routing | `shutdown()` | Long-lived local aggregation object | Process-local; not a remote resource | Reconstructable from saved host aliases/config |
| `OutputBus` | `OutputBus::new()` | `subscribe()`, `publish()`, `publish_discontinuity()` | `shutdown()` | Local fan-out bus | Process-local and frontend-local | Reconstructable |

TMUX command-engine semantics:

1. `HostHandle` and `Target` are ideal attach/detach resources because they represent remote
   state that may predate the engine.
2. `Target` is a reference, not proof of ownership. The engine should only call destructive
   lifecycle operations such as `kill()` when the command session actually owns that target.
3. Monitor handles and exec handles are local child resources and should always be cleaned up
   on engine close.

## Duration Classes For Command-Engine State

| Class | Examples | Engine Behavior |
|-------|----------|-----------------|
| Session-root, long-lived | `VmHandle`, standalone `VnetHandle`, `HostHandle`, `Fleet` | Name them, expose them in completion, and track engine mode explicitly |
| Remote resource reference | tmux `Target` | Rehydrate by rediscovery when possible; avoid destructive cleanup unless owned |
| Child attachment, short-lived | `GuestPtySession`, tmux `SessionMonitorHandle`, tmux `ExecHandle` | Treat as ephemeral children of a longer-lived parent; recreate after parent restart/discontinuity |
| Local service / coordination | `FsServer`, `OutputBus` | Usually wrap in a higher-level owned record with explicit engine cleanup policy |
| Ephemeral setup input | `GuestSpec`, `PreparedGuest`, `VnetConfig` | Do not keep as named live resources unless needed for replay/relaunch |

## Attach, Rehydrate, and Detach

The desired user story is:

1. A host or subsystem may already be running.
2. A temporary command engine can attach to it, build a local registry, and offer commands and
   completion.
3. The engine may later detach without killing externally owned resources.

Current practical status by crate:

| Crate | Attach / Rehydrate Today | Main Gap |
|-------|--------------------------|----------|
| `tmux` | Strong: reconnect to host, rediscover sessions/windows/panes, recreate monitor handles | Need engine policy for what it owns vs what it merely references |
| `vmm` | Weak / future: `VmHandle` creation is explicit, but generic attach to already-running VMs is not yet a stable crate API | Need a discovery/import adapter that can rebuild typed control objects from runtime artifacts |
| `vnet` | Weak: no direct import/rebuild API for an already-running backend | Usually subordinate to VMM rather than independently attached |
| `vfs` | Weak: no top-level serving/import handle; more service-oriented than attach-oriented | Needs wrapper-level lifecycle if exposed directly |

Detach semantics for the command engine should therefore be:

1. Stop local helper tasks unconditionally.
2. Close local ephemeral child sessions unconditionally.
3. Destroy only `Owned` top-level resources.
4. Drop `Imported` references without destructive side effects.

In other words, the engine's close behavior is defined by the adapter/registry entry, not by
the raw resource type alone.

## Design Implications

The command engine should model named resource records roughly like this:

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

And the command engine should define adapter traits around subsystem types rather than making
subsystem crates implement engine traits directly. That adapter layer is where close,
detach, import, completion, and invalidation semantics belong.

That gives the engine enough information to:

1. power dynamic completion from live state,
2. route commands from REPL, TUI, or future socket frontends,
3. attach to existing systems temporarily, and
4. clean up truthfully when the engine closes.

## Adapter Scenario Matrix

The tables below describe how the command-engine adapter should treat the major non-tmux
resources under the three named scenarios:

1. `repl owns`: the command engine created the resource in-process.
2. `repl attaches`: the command engine starts later and imports an already-running resource
   into local in-process state.
3. `repl manages remotely`: the command engine is not the host process and instead operates
   through an RPC or proxy handle exposed by another process.

These are adapter semantics, not changes to the underlying crate APIs.

### VMM Adapter Matrix

| Resource | Repl owns | Repl attaches | Repl manages remotely | Adapter expectation |
|----------|-----------|---------------|------------------------|---------------------|
| `VmHandle` | Store as `Owned + InProcess`; completion exposes engine name plus any stable guest identifier; engine `close()` calls `shutdown().await`. | Future import path: reconstruct or wrap a discovered running VM as `Imported + InProcess`; engine `close()` detaches only. | Wrap a host-process RPC capability as `Imported + RemoteProxy`; commands map to remote `ready/exec/open_pty/shutdown` policy, but default engine close detaches the proxy rather than shutting down the VM. | This is the primary durable VMM resource and the root parent for child sessions/backings. |
| `GuestPtySession` | Store as `Ephemeral + InProcess`; engine `close()` always calls `close().await`; invalidate on PTY close, VM restart, or control-plane loss. | Usually do not attach PTYs directly; recreate from an attached `VmHandle` instead of trying to import transcript state. | Wrap the remote host's PTY/session token as `Ephemeral + RemoteProxy`; engine `close()` closes the remote child session only. | PTY sessions are child attachments, not durable imported resources. |
| `GuestFsHandle` / `MotlieVfsHandle` | Normally not named separately; treat as subordinate to an owned `VmHandle` and let VM shutdown own cleanup. | If surfaced at all, import only as subordinate state beneath an imported VM; do not make it a first-class top-level record in the first slice. | Remote management should stay behind the parent VM proxy; do not expose a separate first-class remote resource unless the crate grows that API explicitly. | Better modeled as VM backing than as a standalone command-engine root. |
| `GuestBridgeHandle` / `MotlieSshProxyHandle` | Same as guest FS backing: subordinate to the VM and cleaned up via VM shutdown by default. | Attach only through the VM import path; child control-plane state should be rediscovered from the parent. | Remote management should use the parent VM proxy or a higher-level control-plane proxy, not a separate engine root. | This is a control-plane backing, not the main user-facing durable resource. |
| `MotlieVnetHandle` | Usually subordinate to the owned VM; engine should not name it separately unless standalone VNET management is added later. | Attach only through VM import for now. | Remote management should remain behind the VM proxy. | Treat as subordinate network backing in the first design slice. |

### VNET Adapter Matrix

| Resource | Repl owns | Repl attaches | Repl manages remotely | Adapter expectation |
|----------|-----------|---------------|------------------------|---------------------|
| `VnetHandle` | Store as `Owned + InProcess`; engine `close()` calls `shutdown()`; completion can expose the engine alias and health-oriented status if useful. | No direct crate API exists today; any attach flow would be adapter-driven from runtime artifacts and should be `Imported + InProcess` with detach-only close semantics. | Represent the remote backend as `Imported + RemoteProxy`; map health and shutdown-capable commands through RPC, but default engine close detaches the proxy rather than destroying the network. | Good standalone managed resource if VNET is exposed directly; otherwise subordinate to VMM. |
| `VnetBackend` | Treat as setup-only and not a named managed resource. | Do not attach. | Do not proxy directly. | Construction helper only. |
| `VnetConfig` | Treat as replayable setup input, not a live managed resource. | Recompute from saved config if needed. | Serialize as configuration metadata only, not as a proxy-managed resource. | Useful for relaunch, not for runtime management. |

### VFS Adapter Matrix

| Resource | Repl owns | Repl attaches | Repl manages remotely | Adapter expectation |
|----------|-----------|---------------|------------------------|---------------------|
| `FsServer` | Wrap in a higher-level owned record that also owns listener/task lifecycle; store as `Owned + InProcess`; engine `close()` stops the wrapper, not just the raw server object. | Weak attach story today; only attach if there is a wrapper that can rediscover the transport/listener and recreate a coherent managed record as `Imported + InProcess`. | Represent as `Imported + RemoteProxy` behind a host admin API that exposes mount/event operations; engine close detaches the proxy only. | Raw `FsServer` is service state, so the adapter should manage a wrapper rather than the bare type. |
| `VsockConnectionHandler` | Do not store as a named managed resource; treat as ephemeral per-connection execution state owned by the wrapper/server. | Do not attach directly. | Do not proxy directly. | Per-connection helper, not a registry root. |
| `GuestMountRunner` | Treat as setup helper only; do not expose as a managed resource. | Do not attach. | Do not proxy. | Bootstrap-only helper. |
| `MountHandles` | Usually avoid naming directly; if exposed, treat as `Ephemeral + InProcess` child state tied to the surrounding guest mount session and close by unmounting via the higher-level owner. | Weak attach story; prefer rebuilding from the higher-level guest or VM state rather than importing thread handles. | Remote management should occur through the higher-level guest/VM/admin API, not by proxying raw mount threads. | Not a strong first-class engine resource on its own. |

## Contract Coverage

The adapter layer is considered sufficiently documented when each managed resource has an
answer for these questions:

| Question | Why it matters |
|----------|----------------|
| What mode does the engine assign: `Owned`, `Imported`, or `Ephemeral`? | Determines cleanup truthfulness and detach behavior. |
| What locality does the engine assign: `InProcess` or `RemoteProxy`? | Determines whether the adapter wraps a raw handle or an RPC capability. |
| What is the stable completion identity? | Lets REPL, TUI, and socket frontends resolve the same named object. |
| What does engine `close()` do? | Prevents accidental destruction of imported resources and leaks of owned ones. |
| What does engine `detach()` do? | Defines temporary admin-session semantics. |
| What invalidates the record? | Prevents stale child resources after restart or disconnect. |
| Can the resource be imported or rehydrated at all? | Avoids promising unsupported attach flows. |
