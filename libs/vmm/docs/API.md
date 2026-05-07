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

Changelog:

- 2026-05-07 | @vmm-cdx | tighten resolver provenance so single-image manifests are rejected until config blob inspection can verify the requested platform
- 2026-05-07 | @vmm-cdx | document that resolver live tests are a PR sub-gate and v1.5 acceptance requires v1.4/v1.45 functional parity through the unified v1.5 harness/image-builder/OCI flow
- 2026-05-07 | @vmm-cdx | add the first OCI Registry v2 resolver API for image reference parsing, manifest/index digest resolution, platform manifest selection, and bearer-token auth
- 2026-05-07 | @vmm-cdx | tighten the guest-image contract so `sha256` digests must be full-length and validation records embed the typed profile instead of a freeform profile name
- 2026-05-07 | @vmm-cdx | add the first guest-image OCI contract surface in `image.rs` for source digests, selected platform, import profile, and emitted artifact validation records
- 2026-05-04 | @codex-vz | add `VzUserspaceEgress` to the reviewed runtime model so VZ egress is lifecycle-owned by VMM like CH Motlie VNET while the VZ runner remains a hypervisor adapter
- 2026-04-08 | @codex | add `wait_egress_ready` as a first-class harness/scenario readiness primitive so saved validations and manual certification can block on DNS + outbound HTTPS readiness instead of one opportunistic probe
- 2026-04-08 | @codex | address PR 140 review drift: remove the dead `VmBackend` / `BackendSet` transitional story, update `GuestSpec` / `PreparedGuest` / shutdown snippets to match code, and record typed `OverlaySize`, namespace-sensitive socket paths, and shutdown-cleanup failure reporting
- 2026-04-08 | @codex | add PTY asciicast export as the portable replay artifact beside canonical transcript NDJSON + VTE screen JSON, and add a Rust-native static SVG export for GitHub-friendly snapshot embedding; PNG/GIF generation remains out of scope for `v1.4`
- 2026-04-08 | @codex | make the harness terminal-state engine switchable; `shadow` is now the default PTY/TUI backend, `vt100` remains as an explicit fallback, and result artifacts record the terminal backend used

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
  - [x] `backend/mod.rs`
  - [x] `backend/ch/shell.rs`
- [x] Phase 4 module added in code:
  - [x] `observability.rs`
- [x] initial `examples/v1.4/repl_host_v1_4` exists and compiles against the
      library surface
- [x] `libs/vmm/src/guestfs.rs` exists and is used by the `v1.4` harness
- [x] initial `examples/v1.4/harness_v1_4` exists and runs against the library
      surface
- [x] backend hierarchy is now explicitly split into:
  - [x] `backend::ch`
  - [x] placeholder `backend::motlie`
  - [x] placeholder `backend::vz`
- [ ] next guest-shape convergence:
  - [x] `Runtime`
  - [x] `HypervisorBacking`
  - [x] `FilesystemBacking`
  - [x] `NetworkBacking`
  - [x] `ControlPlaneBacking`
  - [ ] `VmSpec`
  - [ ] simple CH “hello world” example over the same lifecycle API
- [x] first guest-image OCI contract module added in code:
  - [x] `image.rs`
  - [x] typed source reference and immutable digest metadata
  - [x] typed selected OCI platform
  - [x] typed import profile metadata for `ubuntu-systemd`
  - [x] typed validation records for backend-emitted artifacts
  - [x] typed image reference parsing for Docker/OCI-style refs
  - [x] OCI/Docker manifest index resolution through Registry v2
  - [x] selected platform-manifest digest extraction

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
- [x] `RuntimeNamespace::root_from_env_or_temp()`
- [x] `RuntimeNamespace::for_process(...)`
- [x] `RuntimeNamespace::guest_vsock_port_socket(...)`
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
- [x] `BackendHandle`
- [x] `ChShellBackend`
- [x] `prepare(...)`
- [x] `boot(...)`
- [x] `LifecycleServices`
- [x] `VmHandle::ready(...)`
- [x] `VmHandle::exec(...)`
- [x] `VmHandle::open_pty(...)`
- [x] `VmHandle::shutdown(...)`
- [x] guestfs / SSH bridge / exec-ready readiness gates beyond API socket
- [x] `boot(...)` provisions guestfs and optional rootless `vhost-user` egress
- [x] `boot(...)` can spawn the guest SSH bridge through lifecycle services
- [x] backend shutdown now tracks the spawned child process directly and exits
      cleanly through CH API shutdown or `SIGTERM` before falling back to
      `SIGKILL`
- [ ] next structural convergence:
  - [x] `Runtime`
  - [x] `HypervisorBacking`
  - [x] `FilesystemBacking`
  - [x] `NetworkBacking`
  - [x] `ControlPlaneBacking`
  - [x] PTY/session control through `VmHandle`
  - [ ] `VmSpec`
  - [ ] simple CH “hello world” example using the same lifecycle API
  - [ ] harness interactive mode replaces the standalone `repl_host_v1_4`
  - [x] harness script/scenario format for action/expectation pairs

Phase 4 initial implementation:

- [x] `VmHandle::observability()`
- [x] `VmObservability`
- [x] `VmRuntimePaths`
- [x] `FilesystemObservability`
- [x] `NetworkObservability`
- [x] `ControlPlaneObservability`
- [x] `VmRunBundle`
- [x] `VmCapturePaths`
- [x] `VmRunArtifact`
- [x] `VmHostMount`
- [x] library-owned runtime/log/socket visibility for the live guest handle
- [x] typed run-bundle metadata for later debug collation

Phase 5 first slice:

- [x] harness `--result-json <path>`
- [x] machine-readable `smoke` scenario result
- [x] JSON includes named checks plus `VmObservability`
- [x] structured status/error classification for agents and CI
- [x] PTY scenario result hardening
- [x] reusable PTY transcript artifact capture
- [x] VTE/rendered terminal state
- [x] Rust-native static SVG export from rendered screen state
- [x] timed VTE screen assertions for alternate-screen TUIs via `pty_expect_screen`
- [x] first built-in network readiness primitive via `wait_egress_ready`
- [x] first built-in scenario readiness primitive via `wait_package_manager_quiescent`
- [x] asciicast replay/export
- [x] switchable terminal-state backend in the harness (`shadow` default, `vt100` fallback)
- [ ] PNG/GIF/movie export

## Layering

The intended layering is:

- top layer: guest intent
  - `GuestUser`
  - `GuestSshAccess`
  - `SoftwareProfile`
- portable VM slice
  - `VmSpec`
  - `GuestResources`
  - `GuestStorage`
  - `BootArtifacts`
- runtime composition
  - `Runtime`
  - `HypervisorBacking`
  - `FilesystemBacking`
  - `NetworkBacking`
  - `ControlPlaneBacking`
- interactive control plane
  - `VmHandle::exec(...)`
  - `VmHandle::open_pty(...)`
  - `GuestPtySession`
- observability
  - `VmHandle::observability()`
  - `VmObservability`
- backend realization
  - `backend::ch::*`
  - `backend::motlie::*`
  - `backend::vz::*`

## Desired Usage Outcomes

The API is aiming to support two equally explicit usage patterns:

1. Motlie-backed guests
   - guest backing providers such as Motlie guestfs, Motlie userspace vnet, and
     the SSH proxy/control plane are composed through the library lifecycle API
   - this is the path validated by the `v1.4` rootless harness

2. Simple standard CH guests
   - the same lifecycle API can boot a guest with ordinary hypervisor-managed
     resources only
   - no Motlie guestfs backing
   - no Motlie userspace vnet backing
   - this should be represented first by a small Cloud Hypervisor “hello world”
     example over the same API surface
   - the same slice should later map cleanly to `backend::vz::*`

Important reviewed rule:

- guest OS user, SSH access, software, and mounts stay above the backend
- CH-shaped boot inputs are modeled separately as resources, storage, and boot
  artifacts
- the design should translate almost mechanically into Cloud Hypervisor's
  internal `VmConfig`
- the harness becomes the future primary driver over this API; the standalone
  REPL is transitional only

## Reviewed Next Surface

```rust
pub struct GuestSpec {
    pub vm: VmSpec,
    pub user: GuestUser,
    pub ssh: GuestSshAccess,
    pub software: SoftwareProfile,
    pub runtime: Runtime,
}

pub struct VmSpec {
    pub guest_id: String,
    pub hostname: String,
    pub resources: GuestResources,
    pub storage: GuestStorage,
    pub boot: BootArtifacts,
}

pub struct Runtime {
    pub hypervisor: HypervisorBacking,
    pub filesystem: FilesystemBacking,
    pub network: NetworkBacking,
    pub control_plane: ControlPlaneBacking,
}

pub enum HypervisorBacking {
    CloudHypervisorShell,
    CloudHypervisorForkExec,
    CloudHypervisorVmmThread,
    AppleVirtualization,
}

pub enum FilesystemBacking {
    HypervisorManaged,
    MotlieVfs,
}

pub enum NetworkBacking {
    None,
    HypervisorManaged,
    VzUserspaceEgress,
    MotlieVnet,
    HypervisorManagedPlusMotlieVnet,
}

pub enum ControlPlaneBacking {
    None,
    MotlieSshProxy,
}
```

Reviewed interactive/session surface:

```rust
impl VmHandle {
    pub async fn exec(
        &self,
        command: &str,
        timeout: std::time::Duration,
    ) -> Result<ExecOutput, OrchestratorError>;

    pub async fn open_pty(
        &self,
        request: PtyRequest,
        timeout: std::time::Duration,
    ) -> Result<GuestPtySession, OrchestratorError>;

    pub fn observability(&self) -> VmObservability;
}

pub struct PtyRequest {
    pub term: String,
    pub col_width: u32,
    pub row_height: u32,
    pub pix_width: u32,
    pub pix_height: u32,
    pub command: Option<String>,
}

impl GuestPtySession {
    pub async fn send(&self, data: &[u8]) -> Result<(), SshProxyError>;
    pub async fn send_line(&self, line: &str) -> Result<(), SshProxyError>;
    pub async fn resize(
        &self,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    ) -> Result<(), SshProxyError>;
    pub async fn read_for(
        &self,
        timeout: std::time::Duration,
    ) -> Result<PtyRead, SshProxyError>;
    pub async fn read_until_contains(
        &self,
        needle: &str,
        timeout: std::time::Duration,
    ) -> Result<PtyRead, SshProxyError>;
    pub fn transcript(&self) -> Result<Vec<PtyTranscriptEvent>, SshProxyError>;
    pub async fn close(&self) -> Result<(), SshProxyError>;
}

pub struct VmObservability {
    pub guest_id: String,
    pub pid: Option<u32>,
    pub namespace_prefix: String,
    pub temp_root: std::path::PathBuf,
    pub guest_socket_path: std::path::PathBuf,
    pub runtime_paths: VmRuntimePaths,
    pub filesystem: FilesystemObservability,
    pub network: NetworkObservability,
    pub control_plane: ControlPlaneObservability,
    pub run_bundle: VmRunBundle,
}
```

Reviewed harness direction:

- `examples/v1.4/harness/main.rs` is the future primary driver
- the harness should support:
  - named scenarios such as `smoke`, `multiguest`, and `pty`
  - interactive/manual mode for humans and coding agents
  - transcript and log capture for PTY sessions and VM launch artifacts
  - rendered terminal-state capture alongside raw PTY transcript capture
  - asciicast export for portable replay in terminal or web viewers
  - switchable terminal backends for PTY/TUI validation:
    `--terminal-backend shadow|vt100`
- `examples/v1.4/repl_host.rs` remains useful during the transition, but it
  should not accumulate unique lifecycle/control-plane logic

Reviewed intent:

- `VmSpec` is the simple vertical slice for ordinary hypervisor guests
- `Runtime` describes the composed runtime/backing story for guest-visible
  capabilities
- Motlie guestfs and Motlie userspace vnet are not “sidecars”; they are actual
  guest backing providers
- the simple CH case is:
  - `hypervisor = CloudHypervisorShell`
  - `filesystem = HypervisorManaged`
  - `network = HypervisorManaged`
  - `control_plane = None`

## Current Implemented Guest Shape

### `spec.rs`

```rust
pub struct GuestSpec {
    pub guest_id: String,
    pub hostname: String,
    pub socket_path: std::path::PathBuf,
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
    pub overlay_size: OverlaySize,
}

pub struct OverlaySize(String);

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
        overlay_size: OverlaySize::new("2G").unwrap(),
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
pub struct Ipv4Subnet { /* ... */ }
pub struct Ipv4SubnetPool { /* ... */ }
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
- capacity is computed from subnet pools, CID headroom, and MAC headroom
- config is inspectable and suitable for harness UX

Reviewed shape:

```rust
impl Ipv4Subnet {
    pub fn new(
        network: std::net::Ipv4Addr,
        prefix_len: u8,
    ) -> Result<Self, GuestNetAllocatorError>;
}

impl std::str::FromStr for Ipv4Subnet { /* parses 10.0.0.0/8 style input */ }

impl Ipv4SubnetPool {
    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError>;
    pub fn subnet_for_slot(
        &self,
        slot: u32,
    ) -> Result<Ipv4Subnet, GuestNetAllocatorError>;
}

impl GuestNetAllocatorConfig {
    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError>;
    pub fn validate(&self) -> Result<(), GuestNetAllocatorError>;
    pub fn with_max_guests(self, max_guests: u32) -> Self;
}

impl GuestNetAllocator {
    pub fn new(
        config: GuestNetAllocatorConfig,
    ) -> Result<Self, GuestNetAllocatorError>;
    pub fn config(&self) -> &GuestNetAllocatorConfig;
    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError>;
    pub fn next_slot(&self) -> u32;
    pub fn remaining_capacity(&self) -> Result<u32, GuestNetAllocatorError>;
    pub fn assignments(
        &self,
    ) -> &std::collections::BTreeMap<String, GuestNetAssignment>;
    pub fn get(&self, guest_name: &str) -> Option<&GuestNetAssignment>;
    pub fn ensure(
        &mut self,
        guest_name: &str,
    ) -> Result<&GuestNetAssignment, GuestNetAllocatorError>;
}
```

Default reviewed policy:

- admin pool: `172.20.0.0/16` split into guest `/30`s
- egress pool: `10.0.0.0/8` split into guest `/24`s
- `cid = first_cid + slot`
- admin and egress MAC addresses are derived from the slot, not truncated to
  one byte

### `image.rs`

The guest-image surface is the first host-runtime API for the OCI import
roadmap. It records source identity and validation metadata, and can resolve
OCI/Docker manifest indexes to immutable digests. It does not yet unpack rootfs
layers or emit CH/VZ artifacts.

```rust
use motlie_vmm::backend::BackendKind;
use motlie_vmm::image::{
    EmittedArtifactDigest, ExternalOciSource, GuestImageProfile,
    GuestImageValidationRecord, OciDigest, OciImageReference, OciPlatform,
    OciRegistryClient,
};

let source = ExternalOciSource::ubuntu_systemd(
    OciPlatform::linux_amd64(),
    OciDigest::new("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")?,
    OciDigest::new("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")?,
);

let profile = GuestImageProfile::ubuntu_systemd(source.clone());
profile.validate()?;

let record = GuestImageValidationRecord {
    profile: profile.clone(),
    contract_version: "v1.5".to_string(),
    backend_kind: BackendKind::ChShell,
    emitted_artifacts: vec![EmittedArtifactDigest {
        label: "rootfs".to_string(),
        path: "artifacts/base/rootfs.squashfs".into(),
        digest: OciDigest::new("sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")?,
    }],
};
record.validate()?;

let image_ref: OciImageReference = "docker.io/library/ubuntu:24.04".parse()?;
assert_eq!(image_ref.normalized(), "docker.io/library/ubuntu:24.04");

let resolver = OciRegistryClient::new();
let resolved_source = resolver
    .resolve_ubuntu_systemd_source(OciPlatform::linux_amd64())
    .await?;
resolved_source.validate()?;
```

The helper `OciPlatform::default_for_v1_5_validation_backend(...)` returns only
the current validation-lab default: CH maps to `linux/amd64` for DGX validation
and VZ maps to `linux/arm64` for Apple Silicon validation. It is not a backend
invariant; callers should pass an explicit `OciPlatform` when the host or guest
architecture differs.

Validation rules intentionally reject short fake `sha256` values and enforce
the current `ubuntu-systemd` profile/source coherence:
`GuestImageValidationRecord` embeds `GuestImageProfile`, and the embedded
profile must validate before emitted artifact digests are accepted.

Resolver behavior:

- parses Docker-style references such as `ubuntu:24.04`,
  `docker.io/library/ubuntu:24.04`, and `registry:5000/team/repo@sha256:...`
- normalizes Docker Hub official images to `docker.io/library/<name>:<tag>`
- uses Registry v2 manifest requests with OCI and Docker media-type accept
  headers
- handles Bearer auth challenges for public Docker Hub-style registries
- computes the returned manifest/index body digest locally and checks
  `Docker-Content-Digest` when the registry provides it
- selects the requested `OciPlatform` descriptor digest from an OCI image index
  or Docker manifest list
- rejects single-image manifests until config blob inspection verifies the
  manifest's actual platform; unknown JSON without descriptors is rejected

Resolver validation:

```bash
cargo test -p motlie-vmm image --lib
cargo test -p motlie-vmm resolves_ubuntu_systemd_source_from_registry --lib -- --ignored
```

The ignored live-registry test is not part of default unit tests because it
depends on external registry availability, DNS, and rate limits. It is still a
required PR acceptance step for changes that affect source resolution.

v1.5 acceptance is broader than this resolver API. The final v1.5 surface must
prove v1.4 CH and v1.45 VZ guest-visible functional parity through the unified
v1.5 harness, unified image builder, and OCI-derived guest image/profile flow.
That parity includes multi-guest isolation, SSH auto-provisioning, VFS,
VNET/egress, sudo/apt readiness, PTY/TUI operation, Codex/Claude startup
checks, and reproducible run artifacts.

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

### `backend/mod.rs`

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

pub enum BackendHandle {
    ChShell(ChShellHandle),
}

impl ChShellBackend {
    pub fn kind(&self) -> BackendKind;
    pub fn capabilities(&self) -> VmBackendCapabilities;
    pub fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError>;
    pub fn shutdown(
        &self,
        handle: &BackendHandle,
    ) -> Result<BackendShutdownOutcome, BackendError>;
}
```

Rules:

- enum dispatch, not dynamic dispatch
- all backends are known and implemented in-tree
- `backend/mod.rs` defines the generic contract only
- backend-specific realization lives under `backend/`
- `backend/ch/shell.rs` is the first implementation and preserves current
  `v1.3` shell/CLI behavior
- generic orchestrator code should depend on an injected `Runtime`, not import
  `backend::ch::*`, `backend::vz::*`, or `backend::motlie::*` directly
- readiness, SSH exec, validation, SSH CA, and guestfs semantics stay above the
  backend layer
- current code now follows this rule for the active `v1.4` path:
  - VM boot/shutdown is injected through `Runtime.hypervisor`
  - Motlie filesystem/network/control-plane wiring is injected through the same
    `Runtime` composition
- the earlier transitional `VmBackend` / `BackendSet` scaffold has now been
  removed so the code matches the reviewed enum-dispatch design directly

### `orchestrator.rs`

```rust
pub struct PrepareRequest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub backend_kind: BackendKind,
    pub network_modes: NetworkModes,
    pub base_dir: std::path::PathBuf,
    pub ssh_ca_pubkey: Option<String>,
}

pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub runtime_paths: GuestRuntimePaths,
    pub guest_socket_path: std::path::PathBuf,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
    pub network_modes: NetworkModes,
    pub base_dir: std::path::PathBuf,
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

pub struct LifecycleServices {
    pub runtime: std::sync::Arc<Runtime>,
}

pub fn prepare(
    req: PrepareRequest,
    allocator: &mut GuestNetAllocator,
) -> Result<PreparedGuest, OrchestratorError>;
pub async fn boot(
    prepared: PreparedGuest,
    services: LifecycleServices,
) -> Result<VmHandle, OrchestratorError>;

impl VmHandle {
    pub async fn ready(
        &self,
        policy: &ReadinessPolicy,
    ) -> Result<(), OrchestratorError>;

    pub async fn exec(
        &self,
        command: &str,
        timeout: std::time::Duration,
    ) -> Result<crate::ssh::ExecOutput, OrchestratorError>;

    pub async fn shutdown(&self) -> Result<ShutdownReport, OrchestratorError>;
}
```

Example:

```rust
let prepared = orchestrator::prepare(req, &mut allocator)?;
let handle = orchestrator::boot(
    prepared,
    LifecycleServices {
        runtime: std::sync::Arc::new(Runtime {
            hypervisor: HypervisorBacking::CloudHypervisorShell,
            filesystem: FilesystemBacking::MotlieVfs,
            network: NetworkBacking::MotlieVnet,
            control_plane: ControlPlaneBacking::MotlieSshProxy,
        }),
    },
).await?;

handle.ready(&ReadinessPolicy {
    api_socket_timeout: std::time::Duration::from_secs(10),
    guestfs_timeout: std::time::Duration::from_secs(15),
    ssh_bridge_timeout: std::time::Duration::from_secs(15),
    exec_ready_timeout: std::time::Duration::from_secs(20),
}).await?;

let out = handle
    .exec("/bin/echo hello", std::time::Duration::from_secs(10))
    .await?;
```

Current implementation note:

- `VmHandle::ready(...)` now covers API socket, guestfs, SSH bridge, and a
  simple exec-ready probe
- backend shutdown now owns the real child process state rather than polling
  `/proc`, so readiness and shutdown no longer treat exited CH processes as
  still alive zombies
- current code now injects the active hypervisor plus Motlie guest-backing
  providers through `Runtime`
- the next API convergence is guest-shape cleanup:
  - introduce `VmSpec`
  - move user-facing runtime composition out of the current direct
    `GuestSpec { mounts, ... }` shape
  - add the simple Cloud Hypervisor “hello world” example

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
