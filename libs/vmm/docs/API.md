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

- `ca`
- `network_alloc`
- `ssh`

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

### Planned `spec.rs`

This module should hold typed guest/runtime inputs rather than lifecycle
behavior.

Planned types:

```rust
pub struct GuestSpec {
    pub name: String,
    pub uid: u32,
    pub gid: u32,
    pub mounts: Vec<GuestMountSpec>,
}

pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: std::path::PathBuf,
    pub host_path: std::path::PathBuf,
}

pub struct GuestRuntimePaths {
    pub runtime_dir: std::path::PathBuf,
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
    uid: 1000,
    gid: 1000,
    mounts: vec![
        GuestMountSpec {
            tag: "alice-home".to_string(),
            guest_path: "/home/alice".into(),
            host_path: "/tmp/motlie-vmm-v14-demo/alice-home".into(),
        },
    ],
};
```

### Planned `network.rs`

This module should hold network mode selection and validation, not per-guest
allocation state.

Planned types:

```rust
pub enum AdminNetMode {
    None,
    Tap,
}

pub enum EgressNetMode {
    None,
    VhostUser,
}

pub struct NetworkModes {
    pub admin: AdminNetMode,
    pub egress: EgressNetMode,
}
```

Planned helpers:

```rust
pub fn validate_network_modes(modes: &NetworkModes) -> Result<(), NetworkModeError>;
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
use motlie_vmm::spec::GuestSpec;

let modes = NetworkModes {
    admin: AdminNetMode::None,
    egress: EgressNetMode::VhostUser,
};
validate_network_modes(&modes)?;

let guest = GuestSpec {
    name: "alice".to_string(),
    uid: 1000,
    gid: 1000,
    mounts: vec![],
};

let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
let net = alloc.ensure(&guest.name)?;
```

That is the first API slice to review before lifecycle/orchestration extraction
begins.

## Provisional API Notes

The following names are not yet fixed:

- `GuestSpec`
- `GuestMountSpec`
- `GuestRuntimePaths`
- `AdminNetMode`
- `EgressNetMode`
- `NetworkModes`
- `validate_network_modes`

The important thing for review right now is the separation of concerns:

- `spec.rs` owns typed guest/runtime inputs
- `network.rs` owns mode selection and validation
- `network_alloc.rs` owns stable per-guest allocation

## Review Questions

As Phase 1 lands, this document should answer:

1. Are the extracted types small, typed, and side-effect-free?
2. Are namespace-sensitive runtime paths explicit enough?
3. Is network mode policy separated cleanly from allocation state?
4. Is the allocator contract clear about stability and exhaustion?
5. Is the API surface small enough that later phases can layer on top without
   forcing a rewrite?
