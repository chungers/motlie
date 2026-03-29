//! GuestMountRunner and GuestMountSpec: guest-side mount orchestration over public APIs.
//!
//! This is the public guest-facing orchestration layer above `FuseClient`.
//! Both `src/bin/motlie-vfs-guest.rs` and future `motlie-vmm-guest` binaries
//! should call into this module rather than reimplementing mount loops.
