//! Guest-side FUSE client and mount orchestration.
//!
//! The `guest` module provides `GuestMountRunner` and `GuestMountSpec` — these
//! are transport-independent and compile on any platform.
//!
//! The `fuse` module provides `FuseClient` which implements `fuser::Filesystem`
//! and requires the `fuser` crate. In v1 this real mount path is Linux-only;
//! macOS host tests exercise the transport-independent guest runner instead.

pub mod guest;

#[cfg(all(feature = "fuser-client", target_os = "linux"))]
pub mod fuse;

#[cfg(all(feature = "local-mount", target_os = "linux"))]
pub mod local;
