//! Guest-side FUSE client and mount orchestration.
//!
//! The `guest` module provides `GuestMountRunner` and `GuestMountSpec` — these
//! are transport-independent and compile on any platform.
//!
//! The `fuse` module provides `FuseClient` which implements `fuser::Filesystem`
//! and requires the `fuser` crate (Linux only in v1).

pub mod guest;

#[cfg(feature = "client")]
pub mod fuse;
