//! motlie-vfs: Layered guest filesystem composition.
//!
//! A transport-agnostic FUSE server with in-memory overlay for runtime
//! filesystem composition in VM guests and standalone mounts.

pub mod core;

#[cfg(feature = "vsock")]
pub mod vsock;

#[cfg(feature = "client")]
pub mod client;
