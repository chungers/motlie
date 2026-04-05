# motlie-vmm: Reusable VM Orchestration Extracted from Proven Examples

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-04 | @claude-vmm | Fold SSH proxy as programmatic guest control plane into DESIGN; add FR/NFR sections; update status to reflect v1.2 validation; resolve open questions |
| 2026-04-03 | @codex | Initial DESIGN for `libs/vmm`: capture the post-`v1.2` extraction target for reusable VM orchestration code |

## Status

The `v1.2` proving ground (`libs/vfs/examples/v1.2`) has validated the core
orchestration flow end to end: guest launch composition, dual-network
(TAP admin + vhost-user egress), cloud-init rendering, VFS mount wiring,
and deterministic shutdown. The validated runbook is in
`libs/vfs/examples/v1.2/README.md`.

Extraction into `libs/vmm` is now the active next step, with one major new
component: an SSH proxy (russh) that replaces TAP-based ingress and doubles
as the programmatic guest control plane for automated validation.

## Problem Statement

The current example lineage (`v1`, `v1.1`, and planned `v1.2`) contains
growing amounts of host-side VM orchestration logic:

- Cloud Hypervisor argument construction
- guest-specific socket, CID, IP, and runtime-path allocation
- cloud-init asset generation
- runtime overlay assembly
- coordinated startup/shutdown of helper services such as `motlie-vfs` and
  `motlie-vnet`
- launch-mode composition such as admin ingress vs egress networking

That logic is useful beyond any single example, but extracting it too early
creates two risks:

1. we freeze the wrong API before `v1.2` proves the workflow
2. we blur crate boundaries by pushing VM orchestration into `motlie-vfs` or
   `motlie-vnet`

We need a dedicated home for reusable VM orchestration once the `v1.2` flow is
validated, while keeping the early experimentation in examples where it is easy
to change.

## Goals

- Provide a dedicated library for host-side VM orchestration and launch
  composition.
- Extract reusable code only after it is proven in `examples/v1.2`.
- Keep device-specific logic in the device crates:
  - `motlie-vfs` owns filesystem serving and guest mount transport
  - `motlie-vnet` owns outbound guest networking and the vhost-user-net backend
- Centralize Cloud Hypervisor launch construction and per-guest runtime layout.
- Centralize guest boot asset generation such as cloud-init and generated
  `mounts.yaml`.
- Make multi-guest orchestration and per-guest lifecycle management reusable.
- Provide a host-side SSH proxy (russh) that replaces TAP-based ingress,
  completing the fully-userspace stack and enabling programmatic guest
  command execution for automated testing.
- Let future examples become thin wiring layers over library code rather than
  carrying large shell-script control planes.

## Non-Goals

- Defining a general-purpose hypervisor abstraction across VMMs.
- Replacing `cloud-hypervisor` with a pluggable backend in v1.
- Moving `motlie-vfs` protocol logic into `libs/vmm`.
- Moving `motlie-vnet` backend logic into `libs/vmm`.
- Freezing the final public Rust API before `v1.2` validates the operator flow.
- Eliminating example scripts immediately; scripts may remain as thin wrappers.

## Functional Requirements

### FR-1: Guest Lifecycle Orchestration

The library must support prepare → launch → running → shutdown → cleanup
for one or more guest VMs. Each lifecycle transition must be explicit and
deterministic. The orchestrator manages per-guest state and coordinates
subsystem startup/teardown order.

### FR-2: Launch Composition

Compose Cloud Hypervisor launch arguments from typed guest, network, and
subsystem configuration. This includes:

- admin ingress vs egress network selection
- per-mode CH device arguments (vhost-user sockets, shared memory, TAP)
- NIC MAC assignment and guest-visible route ownership
- shared-memory region sizing for vhost-user devices

### FR-3: Boot Asset Generation

Render guest boot assets from typed state rather than string templates:

- cloud-init `user-data` (user account, SSH keys, boot services)
- cloud-init `meta-data` (instance-id, hostname)
- optional `network-config` (egress NIC DHCP matching on stable MAC)
- generated `mounts.yaml` (VFS mount tags → guest paths)

### FR-4: Subsystem Wiring

Start and stop `motlie-vnet` and coordinate `motlie-vfs` host-side
provisioning with guest launch. Cleanup order must be deterministic:
guests shut down before subsystem backends are torn down.

### FR-5: Runtime Layout

Deterministic per-guest directory structure:

- runtime directory naming and lifecycle
- overlay image paths
- cloud-init seed paths
- per-guest vsock, API, and vnet sockets
- serial/log output locations

### FR-6: SSH Proxy — Host-Side Ingress and Programmatic Control Plane

An in-process SSH server (russh) that serves two roles:

1. **User-facing SSH ingress.** Replaces TAP-based admin SSH path.
   Users connect via `ssh -p <port> <guest>@localhost`. The proxy
   extracts the username as the guest identity, ensures the VM is
   running, signs an ephemeral CA cert (Ed25519, 60s TTL), and bridges
   the external SSH channel to the guest's stock `openssh-server`.

2. **Programmatic guest command execution.** The host orchestrator
   (or an automated test harness) uses the same russh client path to
   open SSH channels and execute commands inside the guest without
   human intervention:

   ```rust
   // Host-side automated validation — no human, no PTY needed:
   let output = orchestrator.exec(&guest, "curl -s -o /dev/null -w '%{http_code}' https://example.com").await?;
   assert_eq!(output.stdout.trim(), "200");

   let output = orchestrator.exec(&guest, "readlink ~/.codex").await?;
   assert_eq!(output.stdout.trim(), "/agent-state/codex");
   ```

   This enables fully agent-driven testing of guest networking,
   filesystem behavior, and subsystem integration — no human in the
   loop, suitable for CI.

**Auth model:**

- **Inbound (client → russh):** Localhost trust. russh binds
  `127.0.0.1` only. Both `auth_none` and `auth_publickey` accept
  unconditionally — the username is the only extracted value. If you
  can reach localhost, you already own the host-side credential and
  workspace directories. The SSH protocol is used as a session
  transport and identity carrier, not as an authentication gate.

- **Outbound (russh → guest sshd):** CA-based ephemeral certs. The
  daemon holds a user CA keypair in memory. On each connection it
  signs a throwaway Ed25519 cert with `principal=<username>` and
  60-second TTL. The guest image has the CA public key baked into
  `/etc/ssh/ca/user_ca.pub` and each VM's
  `/etc/ssh/auth_principals/root` contains only that VM's username.
  Even with a valid cert for "bob", you cannot reach alice's VM.

- **Scope limitation:** The localhost-only trust model is appropriate
  for single-user development hosts. Multi-tenant or network-exposed
  deployments would require real inbound authentication (e.g.
  publickey verification against an authorized keys source). That is
  out of scope for v1.

### FR-7: Automated Guest Validation

The orchestrator must expose a command-execution interface (`exec`) that
captures stdout, stderr, and exit code from commands run inside a guest.
This is the primitive that enables:

- CI-driven regression testing of the full guest stack
- Automated runbook execution (every step in the v1.2 validated
  runbook becomes a programmatic assertion)
- Agent-driven guest provisioning without SSH shell sessions

Every step in the current v1.2 manual runbook maps directly:

| Manual step | Programmatic equivalent |
|---|---|
| `ssh alice@192.168.249.2` then `ip route` | `orchestrator.exec(&alice, "ip route")` |
| `curl -I https://example.com` | `orchestrator.exec(&alice, "curl ...")` → assert exit 0 |
| `readlink ~/.codex` | `orchestrator.exec(&alice, "readlink ~/.codex")` → assert `/agent-state/codex` |
| `sudo apt-get update` | `orchestrator.exec(&alice, "sudo apt-get update")` → assert exit 0 |

## Non-Functional Requirements

### NFR-1: Fully Userspace Operation

With the SSH proxy replacing TAP-based ingress, the entire host-side
stack runs without elevated privileges:

| Layer | Mechanism | Privilege |
|---|---|---|
| SSH ingress | russh on `127.0.0.1:<port>` | None (unprivileged port) |
| VM execution | KVM via `/dev/kvm` | `kvm` group membership |
| Guest networking (egress) | motlie-vnet (vhost-user + libslirp) | None — userspace sockets |
| Guest filesystems | motlie-vfs over vsock | None |
| Host↔guest transport | `/dev/vhost-vsock` | `kvm` group membership |
| Guest networking (ingress) | SSH proxy (replaces TAP) | None |

No `CAP_NET_ADMIN`, no `sudo`, no `setcap`, no `iptables`, no TAP
devices, no network namespaces, no host network interface modifications.

### NFR-2: No Guest-Side Dependencies for Control

Programmatic `exec` uses standard SSH `channel_exec` — the guest needs
only stock `openssh-server`, not a custom control agent. This keeps the
guest image generic and avoids coupling the test harness to a bespoke
guest-side protocol.

### NFR-3: Cloud Hypervisor-Specific in v1

`libs/vmm` targets Cloud Hypervisor directly in v1. No hypervisor
abstraction trait, no pluggable backend. Internal module boundaries
should be clean enough that a future backend could be introduced, but
v1 does not design for that or expose seams for it.

<!-- @claude-vmm 2026-04-04 — Resolves open question "CH-specific or CH-first
     with seams?" The non-goals already exclude pluggable backends in v1.
     A trait adds complexity with zero current consumers. -->

## Design Principles

### 1. Prove First, Extract Second

`examples/v1.2` is the place to prove:

- dual-network launch composition
- guest runtime overlay layout
- cloud-init rendering rules
- `motlie-vfs` + `motlie-vnet` + Cloud Hypervisor composed startup
- operational shutdown and cleanup behavior

Only code that is stable after that validation should move into `libs/vmm`.

### 2. Keep Ownership Boundaries Sharp

`libs/vmm` should own orchestration, not device behavior.

- `motlie-vfs` remains the filesystem subsystem
- `motlie-vnet` remains the networking subsystem
- `libs/vmm` composes subsystems into a guest launch/runtime

This avoids turning `motlie-vfs` or `motlie-vnet` into grab-bag crates for
unrelated orchestration concerns.

### 3. Prefer Typed Host-Side Models Over Shell State

If `v1.2` proves that a piece of shell glue is fundamental to the runtime
contract, it should move into typed Rust structures:

- guest identity
- runtime directories
- socket allocation
- NIC configuration
- cloud-init fragments
- launch-time artifact selection

The example can still expose an operator-friendly CLI, but the hard parts
should not remain encoded as ad hoc environment variables and string assembly.

## Proposed Responsibility Split

### `motlie-vfs`

Owns:

- `FsServer`
- guest mount protocol
- guest mounter binaries
- host REPL commands that are intrinsically about mounted filesystems

Does not own:

- Cloud Hypervisor command construction
- network topology selection
- cloud-init rendering as a general facility

### `motlie-vnet`

Owns:

- libslirp wrapper
- vhost-user-net backend
- outbound DHCP/DNS/internet egress mechanics
- optional hostfwd helper behavior

Does not own:

- guest persona or cloud-init rendering
- VM overlay layout
- VMM lifecycle orchestration

### `libs/vmm`

Owns:

- guest launch configuration
- per-guest runtime directory layout
- generated cloud-init / boot assets
- CH command-line/device composition
- orchestration of optional subsystems (`motlie-vfs`, `motlie-vnet`)
- deterministic startup/shutdown and cleanup coordination
- SSH proxy (russh): user-facing ingress and programmatic guest exec
- SSH CA: in-memory keypair, ephemeral cert signing for guest auth
- automated guest validation primitives (`exec` → stdout/stderr/exit)

## General Layout

The exact module split should follow what `v1.2` proves, but the expected shape
is:

```text
libs/vmm/
  docs/
    DESIGN.md
  src/
    lib.rs
    guest.rs        # typed guest identity/config model
    runtime.rs      # runtime dir layout, sockets, overlays, temp assets
    cloud_init.rs   # render user-data/meta-data/network-config
    network.rs      # admin-net / egress-net composition model
    launcher.rs     # CH argument construction and process launch
    ssh.rs          # russh server + client, channel bridging, exec
    ca.rs           # SSH CA keypair, ephemeral cert signing
    orchestrator.rs # high-level "prepare, start, stop, exec" flow
```

This is a target shape, not a commitment to specific filenames.

`ssh.rs` and `ca.rs` are the new modules supporting FR-6/FR-7. The SSH
proxy is an orchestration concern — it composes guest identity (from
`guest.rs`), VM lifecycle (from `orchestrator.rs`), and CA signing (from
`ca.rs`) into a single ingress/control-plane endpoint.

## Candidate Reusable Pieces to Extract from `v1.2`

### 1. Launch Composition

Expected extraction:

- admin ingress vs egress network selection
- per-mode CH args
- shared-memory requirements for vhost-user devices
- NIC MAC assignment and guest-visible route ownership

This is orchestration logic and belongs in `libs/vmm`, not in `motlie-vnet`.

### 2. Guest Boot Asset Generation

Expected extraction:

- cloud-init `user-data`
- cloud-init `meta-data`
- optional `network-config`
- generated `mounts.yaml`

The critical rule is that `libs/vmm` should generate these assets from typed
guest/runtime state rather than from example-specific string templates once the
behavior is proven.

### 3. Runtime Layout and Artifact Assembly

Expected extraction:

- runtime directory naming
- overlay image paths
- cloud-init seed paths
- per-guest sockets and API sockets
- launch-helper log/serial locations

This logic is already shaping into a reusable contract in the example lineage.

### 4. Subsystem Lifecycle Wiring

Expected extraction:

- start/stop `motlie-vnet`
- coordinate `motlie-vfs` host-side provisioning with guest launch
- ensure cleanup order is deterministic on shutdown

This code should become reusable once the composed `v1.2` flow is stable.

### 5. SSH Proxy and Programmatic Control Plane

This is a **new component**, not an extraction from v1.2. The design
originates from `docs/motlie-vmm.md` (§8 Daemon, §9 SSH Server).

The SSH proxy replaces TAP-based admin ingress with a fully-userspace
path and simultaneously provides the host with programmatic command
execution inside guests. The data flow:

```
 User / test harness
        │
        ▼
 russh server (127.0.0.1:2222)
        │ extract username → guest identity
        ▼
 orchestrator.ensure_vm(guest)
        │ boot VM if needed
        ▼
 ca.sign_ephemeral(guest)
        │ Ed25519, principal=guest, TTL=60s
        ▼
 russh::client::connect(vm_ip:22, cert)
        │ guest sshd validates CA + principal
        ▼
 channel bridge (interactive)    ─or─    channel exec (programmatic)
   pty_request + shell_request              exec("command") → ExecOutput
   data ↔ data bidirectional                stdout, stderr, exit_code
```

For interactive sessions, the proxy bridges PTY, data, and
window-change events bidirectionally. For programmatic exec, it opens
a non-PTY channel, sends the command, and collects output — this is the
primitive that FR-7 requires for automated validation.

The SSH proxy is the component that **eliminates TAP** from the ingress
path. Once it lands, the only kernel interfaces the full stack requires
are `/dev/kvm` and `/dev/vhost-vsock` — both accessible via `kvm` group
membership, no capabilities or root needed.

## API Direction

The API should be small and orchestration-centered. These shapes are
derived from the proven v1.2 patterns (`GuestConfig`, `GuestRuntime`,
`AdminNet`/`EgressNet`, `render_cloud_init`, `render_launch_script`,
`ensure_vnet_backend`, `shutdown_guest`) and the SSH proxy design from
`docs/motlie-vmm.md`.

### Core types

```rust
/// Guest identity and configuration (from v1.2 GuestConfig/GuestIdentity).
pub struct GuestSpec {
    pub name: String,
    pub user: String,
    pub mounts: Vec<MountSpec>,
    pub admin_net: AdminNet,
    pub egress_net: EgressNet,
}

/// Per-guest runtime state produced by prepare().
pub struct LaunchArtifacts {
    pub runtime_dir: PathBuf,
    pub cloud_init_dir: PathBuf,
    pub overlay_path: PathBuf,
    pub vnet_socket: Option<PathBuf>,
    pub api_socket: PathBuf,
}

/// Handle to a running guest VM.
pub struct VmHandle {
    pub name: String,
    pub pid: Option<u32>,
    pub guest_ip: Ipv4Addr,
    // ... internal state
}

/// Output from a programmatic command execution.
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: u32,
}
```

### Orchestration

```rust
pub struct VmOrchestrator { /* ... */ }

impl VmOrchestrator {
    /// Render cloud-init, create runtime dirs, start vnet backend.
    pub async fn prepare(&self, guest: &GuestSpec) -> Result<LaunchArtifacts>;

    /// Launch CH with the prepared artifacts. Returns a handle.
    pub async fn launch(&self, guest: &GuestSpec, artifacts: &LaunchArtifacts) -> Result<VmHandle>;

    /// Deterministic shutdown: API → SIGTERM → SIGKILL fallback.
    pub async fn shutdown(&self, handle: &VmHandle) -> Result<()>;

    /// Execute a command inside the guest via SSH exec channel.
    /// No PTY, no human — captures stdout/stderr/exit_code.
    pub async fn exec(&self, handle: &VmHandle, command: &str) -> Result<ExecOutput>;
}
```

### Usage: automated guest validation (FR-7)

```rust
let orch = VmOrchestrator::new(config)?;
let spec = GuestSpec {
    name: "alice".into(),
    user: "alice".into(),
    mounts: vec![
        MountSpec { tag: "alice-home".into(), guest_path: "/home/alice".into(), .. },
        MountSpec { tag: "alice-agent-state".into(), guest_path: "/agent-state".into(), .. },
    ],
    admin_net: AdminNet::None,       // no TAP — SSH proxy handles ingress
    egress_net: EgressNet::VhostUser,
};

let artifacts = orch.prepare(&spec).await?;
let handle = orch.launch(&spec, &artifacts).await?;

// Verify egress networking
let out = orch.exec(&handle, "curl -s -o /dev/null -w '%{http_code}' https://example.com").await?;
assert_eq!(out.exit_code, 0);
assert_eq!(out.stdout.trim(), "200");

// Verify VFS mount wiring
let out = orch.exec(&handle, "readlink ~/.codex").await?;
assert_eq!(out.stdout.trim(), "/agent-state/codex");

// Verify DNS resolution through motlie-vnet
let out = orch.exec(&handle, "nslookup example.com 10.0.2.3").await?;
assert_eq!(out.exit_code, 0);

orch.shutdown(&handle).await?;
```

### Usage: interactive SSH ingress (FR-6)

```rust
// The SSH proxy runs as a background task inside the orchestrator.
// Users connect from outside:
//   ssh -p 2222 alice@localhost
//
// The proxy:
//   1. Extracts "alice" as guest identity
//   2. Calls orchestrator.ensure_vm("alice") — boots if needed
//   3. Signs ephemeral cert (Ed25519, principal="alice", 60s TTL)
//   4. Opens russh::client to guest sshd at vm_ip:22
//   5. Bridges PTY, data, window-change bidirectionally
//
// The user lands in a shell inside the guest VM.
```

## Relationship to `examples/v1.2`

`examples/v1.2` should initially own the fluid implementation details.

Once validated, the extraction sequence should be:

1. move pure rendering/layout logic into `libs/vmm`
2. move CH launch construction into `libs/vmm`
3. leave example-specific topology and sample operator UX in `examples/v1.2`
4. make `examples/v1.2` consume `libs/vmm` instead of owning the orchestration

This keeps the example as a demo/runbook while shrinking its bespoke logic.

## Validation Gates Before Extraction

Before code moves from `v1.2` into `libs/vmm`, the `v1.2` flow should prove:

- guest boots reliably with generated launch assets
- admin ingress and egress networking compose correctly
- `motlie-vfs` mounts still work in the composed launch
- guest has outbound internet access through `motlie-vnet`
- startup/shutdown sequencing is deterministic
- Alice/Bob multi-guest flows do not rely on accidental path or ordering hacks

If any of those remain unstable, the code should stay in the example until the
behavior settles.

## Resolved Questions

<!-- @claude-vmm 2026-04-04 — Resolved during design session. -->

**Q: Should `libs/vmm` be Cloud Hypervisor-specific in v1, or CH-first with
internal seams?**
A: CH-specific. See NFR-3. No abstraction trait in v1 — no consumers to
justify the complexity. Clean module boundaries are sufficient.

**Q: How much of the current REPL launch flow should remain example-only?**
A: The REPL is a consumer of `libs/vmm`, not part of it. The boundary is
clear: `libs/vmm` owns `prepare` / `launch` / `shutdown` / `exec`.
The REPL owns the interactive command loop and operator UX. Examples
become thin wiring over library calls.

**Q: Should cloud-init generation live as a standalone submodule?**
A: Implementation detail of `prepare()`. In v1.2, cloud-init rendering
is tightly coupled to guest config (name, user, mounts, network mode).
Exposing it as a separate public API adds surface with no current
consumer. Keep it in `cloud_init.rs` as an internal module.

**Q: How should guest runtime artifact cleanup be exposed?**
A: Explicit `shutdown()` with deterministic fallback (API → SIGTERM →
SIGKILL), as already proven in v1.2's `shutdown_guest()`. Best-effort
`Drop` on `VmHandle` as a safety net, but callers should not rely on it.

## Open Questions

- Should the SSH proxy listen on a fixed port (2222) or let the caller
  choose? The product doc (`docs/motlie-vmm.md`) uses 2222 as default.
  The library should probably accept a `SocketAddr` in config.
- How should the orchestrator handle concurrent `exec` calls to the same
  guest? Open multiple SSH channels on one connection, or one connection
  per exec? Performance vs complexity tradeoff.
- Should `exec` have a timeout parameter, or should the caller wrap it?
