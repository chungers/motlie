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

Important:

- this document distinguishes between current exported code and the reviewed
  API direction
- some current Phase 1 code still uses provisional names such as
  `GuestIdentity`
- later `v1.4` phases should follow the reviewed naming and binding model in
  this document even where the code has not been renamed yet

Additional reviewed rule:

- the public `v1.4` API should be layered so it can be translated almost
  mechanically into Cloud Hypervisor's internal `VmConfig`
- guest OS user, SSH access, software, and mounts stay above the CH adapter
- CH-shaped boot inputs should be modeled separately as resources, storage, and
  boot artifacts

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

These are the currently exported Phase 1 code shapes. They are intentionally
provisional and should be refined before later phases depend on them too
heavily.

### Reviewed Phase 1 Spec Shape

The preferred reviewed shape is:

```rust
pub struct GuestSpec {
    pub guest_id: String,
    pub hostname: String,
    pub socket_path: String,
    pub user: GuestUser,
    pub ssh: GuestSshAccess,
    pub mounts: Vec<GuestMountSpec>,
    pub software: SoftwareProfile,
    pub resources: GuestResources,
    pub storage: GuestStorage,
    pub boot: BootArtifacts,
}

pub struct GuestUser {
    pub name: String,
    pub uid: u32,
    pub gid: u32,
    pub home: std::path::PathBuf,
}

pub struct GuestSshAccess {
    pub principal: String,
    pub login_user: String,
}

pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: Option<std::path::PathBuf>,
    pub host_path: std::path::PathBuf,
}

pub struct SoftwareProfile {
    pub packages: Vec<String>,
}

pub struct GuestResources {
    pub boot_vcpus: u8,
    pub memory_mib: u32,
    pub max_vcpus: Option<u8>,
}

pub struct GuestStorage {
    pub overlay_size: String,
}

pub struct BootArtifacts {
    pub kernel: std::path::PathBuf,
    pub initramfs: Option<std::path::PathBuf>,
    pub firmware: Option<std::path::PathBuf>,
    pub cmdline: Option<String>,
}
```

Review intent:

- keep this module free of process launching and side effects
- make namespace-sensitive runtime path derivation typed and explicit
- ensure the types are valid for both `v1.3` comparison and `v1.4` namespace
- keep guest OS account modeling (`GuestUser`) separate from SSH routing/auth
  policy (`GuestSshAccess`)
- keep VM identity (`guest_id`) separate from guest-visible hostname
- keep CH-shaped boot inputs (`GuestResources`, `GuestStorage`,
  `BootArtifacts`) separate from top-layer guest intent such as user/auth,
  mounts, and software

Reviewed example usage:

```rust
let guest = GuestSpec {
    guest_id: "alice".to_string(),
    hostname: "motlie-alice".to_string(),
    socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
    user: GuestUser {
        name: "alice".to_string(),
        uid: 1000,
        gid: 1000,
        home: "/home/alice".into(),
    },
    ssh: GuestSshAccess {
        principal: "alice".to_string(),
        login_user: "alice".to_string(),
    },
    mounts: vec![
        GuestMountSpec {
            tag: "alice-home".to_string(),
            guest_path: Some("/home/alice".into()),
            host_path: "/tmp/motlie-vmm-v14-demo/alice-home".into(),
        },
    ],
    software: SoftwareProfile {
        packages: vec!["vim".to_string(), "gh".to_string()],
    },
    resources: GuestResources {
        boot_vcpus: 2,
        memory_mib: 512,
        max_vcpus: None,
    },
    storage: GuestStorage {
        overlay_size: "2G".to_string(),
    },
    boot: BootArtifacts {
        kernel: "/tmp/vmm-v1.4/libs/vmm/examples/v1.4/artifacts/base/Image".into(),
        initramfs: None,
        firmware: None,
        cmdline: None,
    },
};
```

### Reviewed Layering

The intended layering is:

- top layer: Motlie guest intent
  - `guest_id`
  - `hostname`
  - `GuestUser`
  - `GuestSshAccess`
  - mounts
  - software
- middle layer: bootable VM shape
  - `GuestResources`
  - `GuestStorage`
  - `BootArtifacts`
  - network mode and allocated identities
- bottom layer: Cloud Hypervisor realization
  - `VmConfig`
  - VMM thread start
  - event monitor thread start

This is the intended review boundary for later phases.

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

### Reviewed SSH/CA Binding Shape

The binding between guest OS user state and SSH principal policy should be
explicit in the API.

Reviewed types:

```rust
pub struct IssuedGuestSshCredentials {
    pub principal: String,
    pub login_user: String,
    pub private_key_openssh: String,
    pub certificate_openssh: String,
}
```

Reviewed CA surface:

```rust
impl SshCa {
    pub fn issue_guest_ssh_credentials(
        &self,
        user: &GuestUser,
        access: &GuestSshAccess,
    ) -> Result<IssuedGuestSshCredentials, CaError>;
}
```

This should be the explicit API point where:

- `GuestUser` binds the guest OS account information
- `GuestSshAccess` binds the SSH principal/login-user policy
- the CA returns the resulting ephemeral credential material

### Reviewed Phase 1 Integration Shape

By the end of Phase 1, the `v1.4` harness should be able to do something like:

```rust
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes, validate_network_modes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::spec::{
    BootArtifacts, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, SoftwareProfile,
};

let modes = NetworkModes {
    admin: AdminNetMode::None,
    egress: EgressNetMode::VhostUser,
};
validate_network_modes(&modes)?;

let guest = GuestSpec {
    guest_id: "alice".to_string(),
    hostname: "motlie-alice".to_string(),
    socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".to_string(),
    mounts: vec![],
    user: GuestUser {
        name: "alice".to_string(),
        uid: 1000,
        gid: 1000,
        home: "/home/alice".into(),
    },
    ssh: GuestSshAccess {
        principal: "alice".to_string(),
        login_user: "alice".to_string(),
    },
    software: SoftwareProfile {
        packages: vec!["vim".to_string()],
    },
    resources: GuestResources {
        boot_vcpus: 2,
        memory_mib: 512,
        max_vcpus: None,
    },
    storage: GuestStorage {
        overlay_size: "2G".to_string(),
    },
    boot: BootArtifacts {
        kernel: "/tmp/vmm-v1.4/libs/vmm/examples/v1.4/artifacts/base/Image".into(),
        initramfs: None,
        firmware: None,
        cmdline: None,
    },
};

let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
let net = alloc.ensure(&guest.guest_id)?;
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

### Reviewed software composition extension

Yes, Phase 2 should grow a typed way to request guest software such as `vim`,
`gh`, or similar packages.

The important design constraint is:

- express **software intent** in the API
- do not hardcode the implementation to "always apt install at boot"

That keeps the same API usable for:

1. boot-time cloud-init package installation during development
2. later baked-image / union-binary composition for distributable artifacts

Reviewed types:

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

Reviewed boundary note:

- `artifacts.rs` should render concrete guest boot/runtime files from reviewed
  inputs such as `GuestUser`, `GuestSshAccess`, `SoftwareProfile`,
  `GuestStorage`, and `BootArtifacts`
- `BootArtifacts` is the declarative input model
- rendered cloud-init, mounts YAML, and launch files remain implementation
  outputs below that API layer

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
- `backend.rs`

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
    pub guest_id: String,
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

pub enum BackendKind {
    ChShell,
    ChForkExec,
    ChVmmThread,
    Vz,
}

pub struct VmBackendCapabilities {
    pub same_process_vmm: bool,
    pub supports_api_socket: bool,
    pub supports_event_monitor: bool,
    pub supports_fd_handoff: bool,
    pub supports_memfd_boot_artifacts: bool,
    pub supports_guest_metrics: bool,
}

pub trait VmBackend {
    type Handle;
    type Error;

    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> VmBackendCapabilities;
    fn boot(&self, prepared: &PreparedGuest) -> Result<Self::Handle, Self::Error>;
    fn shutdown(&self, handle: &Self::Handle) -> Result<(), Self::Error>;
}

pub struct ChShellBackend;
```

Reviewed top-level orchestrator surface:

```rust
pub fn prepare(...) -> Result<PreparedGuest, OrchestratorError>;
pub fn boot(prepared: PreparedGuest) -> Result<VmHandle, OrchestratorError>;
```

Reviewed handle surface:

```rust
impl VmHandle {
    pub async fn ready(
        &self,
        policy: &ReadinessPolicy,
    ) -> Result<(), OrchestratorError>;

    pub async fn shutdown(&self) -> Result<ShutdownReport, OrchestratorError>;
}
```

Review intent:

- move lifecycle control into the library
- make readiness explicit and stage-based
- return typed errors instead of REPL-oriented status text
- establish the substrate that the later programmatic harness phase will call
- keep backend dispatch enum-based rather than dynamically discovered
- keep the backend trait narrow: boot, shutdown, capabilities
- start with `ChShellBackend`, which is effectively the current `v1.3` model
- keep `exec`, readiness, validation, SSH CA, and guestfs orchestration above
  the backend layer

Example usage:

```rust
let prepared = orchestrator::prepare(/* guest + modes + namespace + alloc */)?;

let handle = orchestrator::boot(prepared)?;

handle.ready(&ReadinessPolicy {
    api_socket_timeout: std::time::Duration::from_secs(10),
    guestfs_timeout: std::time::Duration::from_secs(15),
    ssh_bridge_timeout: std::time::Duration::from_secs(15),
    exec_ready_timeout: std::time::Duration::from_secs(20),
}).await?;
```

Reviewed backend intent:

```rust
let backend = BackendKind::ChShell;
// internally:
// match backend {
//   BackendKind::ChShell => ChShellBackend::boot(...),
//   BackendKind::ChForkExec => ...,
//   BackendKind::ChVmmThread => ...,
//   BackendKind::Vz => ...,
// }
```

Important reviewed rule:

- enum dispatch is preferred here because all supported backends are known and
  implemented in-tree
- do not introduce plugin-style or runtime-loaded backend discovery
- even `ch_shell`, `ch_fork_exec`, and `ch_vmm_thread` count as separate
  backends because their lifecycle semantics and isolation properties differ

This is the first phase where the library should be able to say not just "the
launch script was spawned", but "the guest is actually ready to use" and, if
not, which readiness stage failed.

## Provisional API Notes

The following names are not yet fixed in code, but the reviewed direction is:

- `GuestSpec`
- `GuestMountSpec`
- `GuestUser`
- `GuestSshAccess`
- `IssuedGuestSshCredentials`
- `GuestStorage`
- `BootArtifacts`
- `RuntimeNamespace`
- `GuestRuntimePaths`
- `SoftwareProfile`
- `GuestResources`
- `CloudInitArtifacts`
- `LaunchArtifactRenderConfig`
- `PreparedGuest`
- `VmHandle`
- `ReadinessStage`
- `ReadinessPolicy`
- `ShutdownReport`
- `BackendKind`
- `VmBackendCapabilities`
- `VmBackend`
- `ChShellBackend`
- `AdminNetMode`
- `EgressNetMode`
- `NetworkModes`
- `validate_network_modes`

The important thing for review right now is the separation of concerns:

- `spec.rs` owns typed guest/runtime inputs
- `network.rs` owns mode selection and validation
- `network_alloc.rs` owns stable per-guest allocation
- `artifacts.rs` owns rendered boot/runtime artifacts
- `backend.rs` will own backend-specific boot/shutdown execution
- `orchestrator.rs` will own lifecycle control and blocking readiness

## Review Questions

As Phase 1 lands, this document should answer:

1. Are the extracted types small, typed, and side-effect-free?
2. Are namespace-sensitive runtime paths explicit enough?
3. Is network mode policy separated cleanly from allocation state?
4. Is the allocator contract clear about stability and exhaustion?
5. Is `artifacts.rs` cleanly separated from lifecycle/process execution?
6. Is the explicit `GuestUser` + `GuestSshAccess` -> CA-issued credentials
   binding clear enough?
7. Is the planned `boot()` + `VmHandle::ready(...)` model explicit enough for
   agents and future CI?
8. Is the split between top-layer guest intent and CH-shaped boot inputs
   (`GuestResources`, `GuestStorage`, `BootArtifacts`) clear enough?
9. Is the API surface small enough that later phases can layer on top without
   forcing a rewrite?
