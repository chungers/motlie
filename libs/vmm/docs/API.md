# motlie-vmm API Review

This document is the review surface for the evolving `libs/vmm` API during the
`v1.4` extraction line.

Rules for this document:

- record the intended library-owned API before or alongside extraction work
- show small usage examples for review
- distinguish:
  - current exported API
  - planned near-term API
  - intentionally provisional API

`v1.3` remains the behavioral comparison baseline. This file is for reviewing
the new library surface that `v1.4` is building.

## Current Exported Surface

As of the current `v1.4` planning checkpoint, [lib.rs](/tmp/vmm-v1.4/libs/vmm/src/lib.rs) exports:

- `artifacts`
- `ca`
- `network`
- `network_alloc`
- `ssh`
- `spec`

These are not yet the final shape of the reusable orchestration surface. They
are the starting point.

## Phase 1 Review Surface

Phase 1 is `Typed Spec and Network Extraction`.

The intent is to move pure configuration and allocation policy out of the
example harness and into small, typed library modules.

The first reviewable modules should be:

- `spec.rs`
- `network.rs`
- `network_alloc.rs`

### Current `spec.rs`

This module should hold typed guest/runtime inputs rather than lifecycle
behavior.

Current implemented types:

```rust
pub struct GuestSpec {
    pub name: String,
    pub socket_path: String,
    pub mounts: Vec<GuestMountSpec>,
    pub identity: Option<GuestIdentity>,
}

pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: Option<std::path::PathBuf>,
    pub host_path: std::path::PathBuf,
}

pub struct GuestIdentity {
    pub uid: u32,
    pub gid: u32,
}

pub struct RuntimeNamespace {
    pub prefix: String,
    pub temp_root: std::path::PathBuf,
}

pub struct GuestRuntimePaths {
    pub runtime_dir: std::path::PathBuf,
    pub launch_dir: std::path::PathBuf,
    pub cloud_init_dir: std::path::PathBuf,
    pub api_socket: std::path::PathBuf,
    pub vsock_socket: std::path::PathBuf,
    pub vnet_socket: std::path::PathBuf,
    pub serial_log: std::path::PathBuf,
    pub launch_log: std::path::PathBuf,
}
```

Review intent:

- keep this module free of process launching and side effects
- make namespace-sensitive runtime path derivation typed and explicit
- ensure the types are valid for both `v1.3` comparison and `v1.4` namespace

Example usage:

```rust
let guest = GuestSpec {
    name: "alice".to_string(),
    socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
    mounts: vec![
        GuestMountSpec {
            tag: "alice-home".to_string(),
            guest_path: Some("/home/alice".into()),
            host_path: "/tmp/motlie-vmm-v14-demo/alice-home".into(),
        },
    ],
    identity: Some(GuestIdentity { uid: 1000, gid: 1000 }),
};
```

Namespace-aware runtime path derivation now exists and is intended to replace
ad hoc string assembly in the harness:

```rust
let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp")?;
let paths = GuestRuntimePaths::for_guest(&namespace, "alice")?;

assert_eq!(paths.api_socket, std::path::PathBuf::from("/tmp/motlie-vmm-v14-alice-api.sock"));
```

### Current `network.rs`

This module should hold network mode selection and validation, not per-guest
allocation state.

Current implemented types:

```rust
pub enum AdminNetMode {
    None,
    Tap,
}

pub enum EgressNetMode {
    None,
    Tap,
    VhostUser,
}

pub struct NetworkModes {
    pub admin: AdminNetMode,
    pub egress: EgressNetMode,
}
```

Current implemented helpers:

```rust
pub fn validate_network_modes(modes: NetworkModes) -> Result<(), NetworkModeError>;
```

Review intent:

- keep network mode policy separate from identity allocation
- allow later rootless-only harness flows to remain explicit
- make invalid combinations a typed error rather than a late harness failure

Example usage:

```rust
let modes = NetworkModes {
    admin: AdminNetMode::None,
    egress: EgressNetMode::VhostUser,
};

validate_network_modes(&modes)?;
```

### Current `network_alloc.rs`

Current scaffold:

- [network_alloc.rs](/tmp/vmm-v1.4/libs/vmm/src/network_alloc.rs)

Current exported types:

- `AdminIpv4Pair`
- `EgressIpv4Layout`
- `GuestNetAssignment`
- `GuestNetAllocatorConfig`
- `GuestNetAllocatorError`
- `GuestNetAllocator`

Current review points:

- this allocator should remain library-owned
- guest assignments should be stable for the lifetime of the harness process
- exhaustion must be a typed error
- `v1.4` should not silently saturate into collisions

Current example usage:

```rust
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};

let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());

let alice = alloc.ensure("alice")?;
let bob = alloc.ensure("bob")?;

assert_ne!(alice.cid, bob.cid);
assert_ne!(alice.admin_ipv4.guest, bob.admin_ipv4.guest);
```

### Planned Phase 1 Integration Shape

By the end of Phase 1, the `v1.4` harness should be able to do something like:

```rust
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes, validate_network_modes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::spec::{GuestIdentity, GuestSpec};

let modes = NetworkModes {
    admin: AdminNetMode::None,
    egress: EgressNetMode::VhostUser,
};
validate_network_modes(&modes)?;

let guest = GuestSpec {
    name: "alice".to_string(),
    socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
    mounts: vec![],
    identity: Some(GuestIdentity { uid: 1000, gid: 1000 }),
};

let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
let net = alloc.ensure(&guest.name)?;
```

That is the first API slice to review before lifecycle/orchestration extraction
begins.

## Phase 2 Review Surface

Phase 2 is `Launch Artifact and Runtime Layout Extraction`.

The owning module for this phase is:

- `artifacts.rs`

This module owns pure rendering/generation of boot/runtime artifacts. It should
not own process spawning or shutdown behavior.

### Current `artifacts.rs`

Current implemented types:

```rust
pub struct CloudInitArtifacts {
    pub meta_data: String,
    pub user_data: String,
    pub mounts_yaml: String,
}

pub struct LaunchArtifactRenderConfig<'a> {
    pub guest: &'a GuestSpec,
    pub runtime_paths: &'a GuestRuntimePaths,
    pub network_modes: NetworkModes,
    pub net_assignment: &'a GuestNetAssignment,
    pub base_dir: &'a std::path::Path,
    pub ssh_ca_pubkey: Option<&'a str>,
}
```

### Planned software composition extension

Yes, Phase 2 should grow a typed way to request guest software such as `vim`,
`gh`, or similar packages.

The important design constraint is:

- express **software intent** in the API
- do not hardcode the implementation to "always apt install at boot"

That keeps the same API usable for:

1. boot-time cloud-init package installation during development
2. later baked-image / union-binary composition for distributable artifacts

Planned types:

```rust
pub struct SoftwareProfile {
    pub packages: Vec<String>,
}
```

Likely API direction:

```rust
pub fn render_cloud_init_with_software(
    guest: &GuestSpec,
    software: &SoftwareProfile,
) -> Result<String, ArtifactError>;
```

Or, if we want the software request to travel with the guest config:

```rust
pub struct GuestSpec {
    // existing fields...
    pub software: SoftwareProfile,
}
```

Recommended first implementation:

- start with package names only
- render them into cloud-init package installation for the development path
- keep the type declarative so the later union-binary phase can interpret the
  same `SoftwareProfile` as "bake these into the image"

Example usage:

```rust
let software = SoftwareProfile {
    packages: vec!["vim".to_string(), "gh".to_string()],
};
```

Review intent:

- avoid scattering package lists through shell templates
- keep software customization typed and reviewable
- preserve a path from "boot-time install" to "baked image composition"

Current implemented helpers:

```rust
pub fn render_mounts_yaml(guest: &GuestSpec) -> Result<String, ArtifactError>;
pub fn render_cloud_init(guest: &GuestSpec) -> Result<String, ArtifactError>;
pub fn render_meta_data(guest_name: &str) -> String;
pub fn render_cloud_init_artifacts(guest: &GuestSpec) -> Result<CloudInitArtifacts, ArtifactError>;
pub fn render_launch_script(cfg: &LaunchArtifactRenderConfig<'_>) -> Result<String, ArtifactError>;
```

Review intent:

- keep artifact rendering pure and testable
- keep namespace-sensitive paths sourced from `spec.rs`
- keep network/device identity sourced from `network.rs` and `network_alloc.rs`
- make this module the stabilization point for later image/union-binary work
- allow software customization to be expressed declaratively so later phases can
  choose between boot-time install and baked-image composition

Example usage:

```rust
use motlie_vmm::artifacts::{render_launch_script, LaunchArtifactRenderConfig};
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};

let script = render_launch_script(&LaunchArtifactRenderConfig {
    guest: &guest,
    runtime_paths: &paths,
    network_modes: NetworkModes {
        admin: AdminNetMode::None,
        egress: EgressNetMode::VhostUser,
    },
    net_assignment: &net,
    base_dir: std::path::Path::new("/tmp/vmm-v1.4/libs/vmm/examples/v1.4"),
    ssh_ca_pubkey: Some("ssh-ed25519 AAAA-test"),
})?;
```

This is the point where image/build artifact handling becomes stable enough for
the later embedded-image / union-binary prototype phase, even though the
programmatic harness phase is still needed before that prototype becomes
pleasant to iterate on.

## Phase 3 Review Surface

Phase 3 is `Orchestrator and Blocking Readiness`.

The owning module for this phase should be:

- `orchestrator.rs`

This module is where `libs/vmm` becomes a real lifecycle owner instead of just
types plus renderers.

### Planned `orchestrator.rs`

Planned types:

```rust
pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
}

pub struct VmHandle {
    pub guest_name: String,
    pub pid: Option<u32>,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
}

pub enum ReadinessStage {
    LaunchSpawned,
    ApiSocketReady,
    GuestFsConnected,
    SshBridgeReady,
    ExecReady,
}

pub struct ReadinessPolicy {
    pub api_socket_timeout: std::time::Duration,
    pub guestfs_timeout: std::time::Duration,
    pub ssh_bridge_timeout: std::time::Duration,
    pub exec_ready_timeout: std::time::Duration,
}

pub struct ShutdownReport {
    pub pid: Option<u32>,
    pub api_attempted: bool,
    pub forced: Option<&'static str>,
}
```

Planned helpers:

```rust
pub fn prepare(...) -> Result<PreparedGuest, OrchestratorError>;
pub fn launch(prepared: PreparedGuest) -> Result<VmHandle, OrchestratorError>;
pub async fn launch_and_wait(
    prepared: PreparedGuest,
    policy: &ReadinessPolicy,
) -> Result<VmHandle, OrchestratorError>;
pub async fn wait_until_ready(
    handle: &VmHandle,
    policy: &ReadinessPolicy,
) -> Result<(), OrchestratorError>;
```

Review intent:

- move lifecycle control into the library
- make readiness explicit and stage-based
- return typed errors instead of REPL-oriented status text
- establish the substrate that the later programmatic harness phase will call

Example usage:

```rust
let prepared = orchestrator::prepare(/* guest + modes + namespace + alloc */)?;

let handle = orchestrator::launch_and_wait(
    prepared,
    &ReadinessPolicy {
        api_socket_timeout: std::time::Duration::from_secs(10),
        guestfs_timeout: std::time::Duration::from_secs(15),
        ssh_bridge_timeout: std::time::Duration::from_secs(15),
        exec_ready_timeout: std::time::Duration::from_secs(20),
    },
)
.await?;
```

This is the first phase where the library should be able to say not just "the
launch script was spawned", but "the guest is actually ready to use" and, if
not, which readiness stage failed.

## Provisional API Notes

The following names are not yet fixed:

- `GuestSpec`
- `GuestMountSpec`
- `GuestIdentity`
- `RuntimeNamespace`
- `GuestRuntimePaths`
- `CloudInitArtifacts`
- `LaunchArtifactRenderConfig`
- `PreparedGuest`
- `VmHandle`
- `ReadinessStage`
- `ReadinessPolicy`
- `ShutdownReport`
- `AdminNetMode`
- `EgressNetMode`
- `NetworkModes`
- `validate_network_modes`

The important thing for review right now is the separation of concerns:

- `spec.rs` owns typed guest/runtime inputs
- `network.rs` owns mode selection and validation
- `network_alloc.rs` owns stable per-guest allocation
- `artifacts.rs` owns rendered boot/runtime artifacts
- `orchestrator.rs` will own lifecycle control and blocking readiness

## Review Questions

As Phase 1 lands, this document should answer:

1. Are the extracted types small, typed, and side-effect-free?
2. Are namespace-sensitive runtime paths explicit enough?
3. Is network mode policy separated cleanly from allocation state?
4. Is the allocator contract clear about stability and exhaustion?
5. Is `artifacts.rs` cleanly separated from lifecycle/process execution?
6. Is the planned readiness model explicit enough for agents and future CI?
7. Is the API surface small enough that later phases can layer on top without
   forcing a rewrite?
