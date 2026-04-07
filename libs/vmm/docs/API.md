# motlie-vmm API

This document is the canonical reviewed API target for the `v1.4` extraction
line.

Rules for this document:

- describe the desired library surface, not competing "current vs reviewed"
  shapes
- keep the public model small and layered
- update implementation status as code converges
- keep `v1.3` as the behavioral comparison baseline

## Implementation Status

High-level status:

- [x] Phase 1 modules added in code:
  - [x] `spec.rs`
  - [x] `network.rs`
  - [x] `network_alloc.rs`
- [x] Phase 2 module added in code:
  - [x] `artifacts.rs`
- [x] Phase 1 reviewed naming fully implemented in code
- [x] Phase 2 reviewed API fully implemented in code
- [x] Phase 3 modules added in code:
  - [x] `orchestrator.rs`
  - [x] `backend.rs`

Phase 1 convergence:

- [x] `GuestSpec.guest_id`
- [x] `GuestSpec.hostname`
- [x] `GuestSpec.user`
- [x] `GuestSpec.ssh`
- [x] `GuestSpec.software`
- [x] `GuestSpec.resources`
- [x] `GuestSpec.storage`
- [x] `GuestSpec.boot`
- [x] `GuestUser`
- [x] `GuestSshAccess`
- [x] `GuestResources`
- [x] `GuestStorage`
- [x] `BootArtifacts`
- [x] `RuntimeNamespace`
- [x] `GuestRuntimePaths`
- [x] `SshCa::issue_guest_ssh_credentials(...)`

Phase 2 convergence:

- [x] `CloudInitArtifacts`
- [x] `LaunchArtifactRenderConfig`
- [x] `render_mounts_yaml(...)`
- [x] `render_cloud_init(...)`
- [x] `render_cloud_init_artifacts(...)`
- [x] `render_launch_script(...)`
- [x] reviewed software/storage/boot-artifact inputs fully reflected in render APIs

Phase 3 initial implementation:

- [x] `PrepareRequest`
- [x] `PreparedGuest`
- [x] `VmHandle`
- [x] `ReadinessStage`
- [x] `ReadinessPolicy`
- [x] `ShutdownReport`
- [x] `BackendKind`
- [x] `VmBackendCapabilities`
- [x] `VmBackend`
- [x] `ChShellBackend`
- [x] `prepare(...)`
- [x] `boot(...)`
- [x] `VmHandle::ready(...)`
- [x] `VmHandle::shutdown(...)`
- [ ] guestfs / SSH bridge / exec-ready readiness gates beyond API socket

## Layering

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
- bottom layer: backend realization
  - Cloud Hypervisor shell launch
  - Cloud Hypervisor fork/exec launch
  - Cloud Hypervisor in-process VMM thread launch
  - future `vz`

Important reviewed rule:

- guest OS user, SSH access, software, and mounts stay above the backend
- CH-shaped boot inputs are modeled separately as resources, storage, and boot
  artifacts
- the design should translate almost mechanically into Cloud Hypervisor's
  internal `VmConfig`

## Phase 1 Surface

### `spec.rs`

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

- `guest_id` is the stable logical VM key
- `hostname` is guest-visible and must stay separate from `guest_id`
- `GuestUser` models the in-guest OS account
- `GuestSshAccess` models SSH routing/auth policy
- `GuestResources`, `GuestStorage`, and `BootArtifacts` are the CH-shaped boot
  inputs

Example:

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
    mounts: vec![],
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

### `network.rs`

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

pub fn validate_network_modes(
    modes: &NetworkModes,
) -> Result<(), NetworkModeError>;
```

### `network_alloc.rs`

```rust
pub struct AdminIpv4Pair { /* ... */ }
pub struct EgressIpv4Layout { /* ... */ }
pub struct GuestNetAssignment { /* ... */ }
pub struct GuestNetAllocatorConfig { /* ... */ }
pub struct GuestNetAllocator { /* ... */ }
```

Contract:

- allocator is library-owned
- assignments are stable for the life of the harness process
- exhaustion is a typed error
- no silent collision-prone saturation

### SSH / CA Binding

The binding between guest OS user state and SSH principal policy must be
explicit.

```rust
pub struct IssuedGuestSshCredentials {
    pub principal: String,
    pub login_user: String,
    pub private_key_openssh: String,
    pub certificate_openssh: String,
}

impl SshCa {
    pub fn issue_guest_ssh_credentials(
        &self,
        user: &GuestUser,
        access: &GuestSshAccess,
    ) -> Result<IssuedGuestSshCredentials, CaError>;
}
```

Binding rule:

- `GuestUser` names the in-guest OS account
- `GuestSshAccess` names the SSH principal and login user policy
- `issue_guest_ssh_credentials(...)` is the explicit binding point
- current code rejects a `login_user` that does not match `GuestUser.name`

## Phase 2 Surface

### `artifacts.rs`

This module owns pure rendering/generation of boot/runtime artifacts.

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

pub fn render_mounts_yaml(guest: &GuestSpec) -> Result<String, ArtifactError>;
pub fn render_cloud_init(guest: &GuestSpec) -> Result<String, ArtifactError>;
pub fn render_meta_data(guest_id: &str, hostname: &str) -> String;
pub fn render_cloud_init_artifacts(
    guest: &GuestSpec,
) -> Result<CloudInitArtifacts, ArtifactError>;
pub fn render_launch_script(
    cfg: &LaunchArtifactRenderConfig<'_>,
) -> Result<String, ArtifactError>;
```

Reviewed boundary:

- `BootArtifacts` is the declarative input model
- `artifacts.rs` produces concrete rendered files and paths
- software packages render into cloud-init `packages:`
- storage renders into shell-launch overlay sizing
- boot artifacts render into backend-consumable launch-script variables
- process spawning and shutdown stay out of this module

## Phase 3 Surface

### `backend.rs`

```rust
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
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> VmBackendCapabilities;
    fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError>;
    fn shutdown(&self, handle: &BackendHandle) -> Result<(), BackendError>;
}

pub enum BackendHandle {
    ChShell(ChShellHandle),
}

pub struct ChShellBackend;
```

Rules:

- enum dispatch, not dynamic dispatch
- all backends are known and implemented in-tree
- `ChShellBackend` is the first implementation and should preserve current
  `v1.3` shell/CLI behavior
- readiness, SSH exec, validation, SSH CA, and guestfs semantics stay above the
  backend layer

### `orchestrator.rs`

```rust
pub struct PrepareRequest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub network_modes: NetworkModes,
    pub backend_kind: BackendKind,
    pub base_dir: std::path::PathBuf,
    pub ssh_ca_pubkey: Option<String>,
}

pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
    pub network_modes: NetworkModes,
    pub backend_kind: BackendKind,
    pub base_dir: std::path::PathBuf,
}

pub struct VmHandle {
    pub guest_id: String,
    pub pid: Option<u32>,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
    pub backend_kind: BackendKind,
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

pub fn prepare(
    req: PrepareRequest,
    allocator: &mut GuestNetAllocator,
) -> Result<PreparedGuest, OrchestratorError>;
pub fn boot(prepared: PreparedGuest) -> Result<VmHandle, OrchestratorError>;

impl VmHandle {
    pub async fn ready(
        &self,
        policy: &ReadinessPolicy,
    ) -> Result<(), OrchestratorError>;

    pub async fn shutdown(&self) -> Result<ShutdownReport, OrchestratorError>;
}
```

Example:

```rust
let prepared = orchestrator::prepare(req, &mut allocator)?;
let handle = orchestrator::boot(prepared)?;

handle.ready(&ReadinessPolicy {
    api_socket_timeout: std::time::Duration::from_secs(10),
    guestfs_timeout: std::time::Duration::from_secs(15),
    ssh_bridge_timeout: std::time::Duration::from_secs(15),
    exec_ready_timeout: std::time::Duration::from_secs(20),
}).await?;
```

Current implementation note:

- `VmHandle::ready(...)` currently waits only for the API socket stage
- the later guestfs / SSH bridge / exec-ready stages remain Phase 3 follow-up

## Backend Transition Path

The reviewed path is:

1. `ChShellBackend`
2. `ChForkExecBackend`
3. `ChVmmThreadBackend`
4. future `VzBackend`

The API above should remain stable across those backend changes.

## Review Questions

1. Are the extracted types small, typed, and side-effect-free?
2. Are namespace-sensitive runtime paths explicit enough?
3. Is network mode policy separated cleanly from allocation state?
4. Is the allocator contract clear about stability and exhaustion?
5. Is `artifacts.rs` cleanly separated from lifecycle/process execution?
6. Is the explicit `GuestUser` + `GuestSshAccess` -> CA-issued credentials binding clear enough?
7. Is the planned `boot()` + `VmHandle::ready(...)` model explicit enough?
8. Is the split between top-layer guest intent and CH-shaped boot inputs clear enough?
9. Is the backend seam narrow enough while still being future-proof for `Vz`?
