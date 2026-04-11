# Driver Lifecycle Inventory

## Change Log

| Date | Change |
|------|--------|
| 2026-04-10 | Rename the lifecycle inventory to the driver architecture. Clarify that ownership semantics belong to driver adapters, not raw resource crates, and keep the scenario tables aligned with `repl owns`, `repl attaches`, and `repl manages remotely`. |

This document records the resource-lifecycle inputs for the driver design in
[`DESIGN.md`](./DESIGN.md).

Scope notes:

1. This is a design-input document, not an implementation checklist.
2. `tmux` is reviewed against the current `main` worktree.
3. `vmm`, `vnet`, and `vfs` reflect the reviewed `feature/vmm` worktree as design input.
4. The first implementation slice remains `Owned + InProcess` only. Attach and remote
   management are documented here so the adapter contracts are honest from the start.

## Driver-Local Management Semantics

The driver must distinguish raw resource lifecycle from driver-session semantics.

| Mode | Meaning in driver state | Default engine-close behavior |
|------|-------------------------|-------------------------------|
| `Owned` | The session created the resource or explicitly claimed cleanup responsibility. | Call explicit subsystem cleanup APIs. |
| `Imported` | The resource predated the session or was rehydrated from elsewhere. | Detach local state only by default. |
| `Ephemeral` | A short-lived child attachment layered on a longer-lived parent. | Close the child handle unconditionally. |

The driver should also track locality:

| Locality | Meaning |
|----------|---------|
| `InProcess` | The driver holds the real handle in the current process. |
| `RemoteProxy` | The driver holds a proxy or RPC capability managed elsewhere. |

This is how a temporary admin shell can attach to already-running resources and later
disappear without destroying them accidentally.

## Crate Summary

| Crate | Main live resources | Explicit cleanup today | Attach/import story today | Driver implication |
|-------|---------------------|------------------------|---------------------------|--------------------|
| `vmm` | `VmHandle`, `GuestPtySession` | Strong | Partial / future adapter work | Best lifecycle surface; explicit shutdown should remain the primary contract. |
| `vnet` | `VnetHandle` | Strong | Weak | Good standalone managed resource if exposed directly. |
| `vfs` | `FsServer`, guest mount helpers | Mixed | Weak | Usually better managed through a wrapper or through VMM. |
| `tmux` | `HostHandle`, `Target`, monitor handles, `Fleet`, `OutputBus` | Mixed | Strong for rediscovery | Good attach/detach surface; be careful with ownership vs reference semantics. |

## Resource Inventory

### VMM

| Resource | Create / acquire | Main operations | Cleanup | Driver role |
|----------|------------------|-----------------|---------|-------------|
| `VmHandle` | `boot(...)` | `ready()`, `exec()`, `open_pty()`, observability | `shutdown().await` | Primary durable named VM resource |
| `GuestPtySession` | `VmHandle::open_pty(...)` | send/resize/read/transcript | `close().await` | Short-lived named child attachment |
| `GuestFsHandle` / VFS backing | provision/backing setup | readiness through VMM | `shutdown()` | Usually subordinate to VM lifecycle |
| `GuestBridgeHandle` / SSH backing | control-plane setup | `exec()`, `open_pty()`, readiness | `shutdown()` | Usually subordinate to VM lifecycle |
| VNET backing under VMM | network provision | health via parent | `shutdown()` | Usually subordinate to VM lifecycle |

### VNET

| Resource | Create / acquire | Main operations | Cleanup | Driver role |
|----------|------------------|-----------------|---------|-------------|
| `VnetHandle` | `VnetBackend::start()` | `is_alive()` | `shutdown()` plus best-effort `Drop` | Durable standalone network resource if exposed directly |
| `VnetBackend` | `VnetBackend::new(config)` | `start()` | none | Setup helper only |
| `VnetConfig` | builder/config | data only | none | Replayable setup input |

### VFS

| Resource | Create / acquire | Main operations | Cleanup | Driver role |
|----------|------------------|-----------------|---------|-------------|
| `FsServer` | builder/build | mount add/remove, `handle_op()`, events | none | Long-lived service object; usually wrapped |
| `VsockConnectionHandler` | `new(server, tag)` | `serve(stream)` | connection close | Per-connection helper only |
| `GuestMountRunner` | `new(specs)` | `mount_all()` | none | Setup helper only |
| `MountHandles` | `mount_all()` | `join_all()` | external unmount / thread exit | Usually not a first-class driver root |

### tmux

| Resource | Create / acquire | Main operations | Cleanup | Driver role |
|----------|------------------|-----------------|---------|-------------|
| `HostHandle` | SSH/local constructors | discovery, target lookup, monitoring setup, session creation | no explicit top-level shutdown | Durable host capability |
| `Target` | create, lookup, navigation | I/O, capture, `kill()`, `rename()`, `start_exec()` | `kill()` when destructive | Primary remote resource reference |
| `SessionMonitorHandle` | monitoring startup | health/is-active | `shutdown().await` | Local child attachment |
| `MonitorHandle` | aggregate monitoring startup | inspection/routing | `shutdown().await` | Local child aggregation object |
| `ExecHandle` | `Target::start_exec()` | status/wait | task end | Short-lived child object |
| `Fleet` | local builder/register | host routing / monitoring | `shutdown()` | Local coordination object |
| `OutputBus` | local builder | subscribe/publish | `shutdown()` | Local coordination object |

## Scenario Matrices

The tables below describe adapter behavior, not changes required in the raw crates.

### VMM

| Resource | `repl owns` | `repl attaches` | `repl manages remotely` |
|----------|-------------|-----------------|-------------------------|
| `VmHandle` | `Owned + InProcess`; registry name is driver-owned; engine close calls `shutdown().await`. | `Imported + InProcess`; attach requires future discovery/import adapter; engine close detaches only. | `Imported + RemoteProxy`; default engine close drops proxy/detaches only. |
| `GuestPtySession` | `Ephemeral + InProcess`; always close explicitly on engine close. | Usually recreate from imported VM, not imported directly. | `Ephemeral + RemoteProxy`; close the child proxy only. |
| VFS/network/control-plane backings | Usually subordinate to owned VM. | Usually subordinate to imported VM. | Usually subordinate to remote VM proxy. |

Important VMM semantic difference:

- VMs and their backings are long-lived roots
- PTYs are short-lived child attachments
- VM restart should invalidate PTYs and similar child state

### VNET

| Resource | `repl owns` | `repl attaches` | `repl manages remotely` |
|----------|-------------|-----------------|-------------------------|
| `VnetHandle` | `Owned + InProcess`; explicit `shutdown()` on close. | Future `Imported + InProcess` attach path if runtime discovery exists; detach by default. | `Imported + RemoteProxy`; proxy detach by default, remote destroy only via explicit command. |
| Setup/config types | Replayable inputs only. | Rebuilt from saved metadata. | Serialized metadata only, not live managed handles. |

### VFS

| Resource | `repl owns` | `repl attaches` | `repl manages remotely` |
|----------|-------------|-----------------|-------------------------|
| `FsServer` | Usually wrap it in a higher-level owned record that also owns listener/task lifecycle. | Weak today; only sensible through a wrapper that can rediscover transport state. | `Imported + RemoteProxy` only through a higher-level host/admin API. |
| `VsockConnectionHandler` | Ephemeral child state only. | Do not import directly. | Do not proxy directly. |
| `MountHandles` | Usually child state beneath a higher-level owner. | Prefer rebuild from higher-level state. | Manage through higher-level proxy, not raw thread handles. |

### tmux

| Resource | `repl owns` | `repl attaches` | `repl manages remotely` |
|----------|-------------|-----------------|-------------------------|
| `HostHandle` | `Owned + InProcess` only if the session owns connection policy/resources; otherwise often `Imported + InProcess`. | `Imported + InProcess`; easy to recreate from saved URI/SSH config. | `Imported + RemoteProxy` if a host process exposes admin RPC. |
| `Target` | May be `Owned` only when the driver created the tmux entity and claims cleanup responsibility. | Often `Imported + InProcess`; rediscover from host state. | `Imported + RemoteProxy`; destructive operations require explicit ownership policy. |
| Monitor / exec handles | Usually `Ephemeral + InProcess`. | Recreate from rediscovered host/target. | Usually `Ephemeral + RemoteProxy`. |

Important tmux semantic difference:

- many tmux handles are references to remote state, not proof of ownership
- only resources the session truly owns should be destroyed on default close

## Attach, Rehydrate, and Detach

The desired operator story is:

1. a host process or subsystem may already be running
2. a temporary driver session attaches to it
3. the session builds a local registry for commands and completion
4. the session later detaches without destroying imported resources

Current status by crate:

| Crate | Attach potential today | Main gap |
|-------|------------------------|----------|
| `tmux` | Strong | Need explicit driver ownership policy for what is only referenced vs what is owned. |
| `vmm` | Future-facing | Needs stable import/discovery API for already-running VMs. |
| `vnet` | Weak | No clear import/rebuild API today. |
| `vfs` | Weak | More service-oriented than attach-oriented. |

Driver close semantics should therefore be:

1. close local child attachments
2. stop local helper tasks
3. destroy only `Owned` top-level resources
4. drop `Imported` references cleanly

## Contract Coverage Checklist

The lifecycle contract is sufficiently documented when each managed resource has a clear
answer for these questions:

| Question | Why it matters |
|----------|----------------|
| What driver mode does it use? | Determines cleanup truthfulness. |
| What locality does it use? | Determines real handle vs proxy semantics. |
| What is the stable completion identity? | Keeps REPL/TUI/socket frontends aligned. |
| What does engine close do? | Avoids leaks and accidental destruction. |
| What does detach do? | Enables temporary admin sessions. |
| What invalidates the record? | Prevents stale child handles after restart/disconnect. |
| Can it be imported at all? | Avoids promising unsupported flows. |
