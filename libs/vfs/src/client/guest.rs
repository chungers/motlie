//! GuestMountRunner and GuestMountSpec: guest-side mount orchestration over public APIs.
//!
//! This is the public guest-facing orchestration layer above `FuseClient`.
//! Both `src/bin/motlie-vfs-guest.rs` and future `motlie-vmm-guest` binaries
//! should call into this module rather than reimplementing mount loops.

// Placeholder — GuestMountRunner depends on FuseClient which requires
// the full fuser callback mapping. For the vertical slice, the transport
// layer is testable end-to-end without requiring a real FUSE mount.
