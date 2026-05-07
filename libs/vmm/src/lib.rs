//! motlie-vmm: Reusable VM orchestration with SSH proxy and programmatic control plane.
//!
//! This crate provides host-side VM orchestration extracted from the proven
//! `v1.2` example lineage, plus a new SSH proxy (russh) that replaces
//! TAP-based ingress and enables programmatic guest command execution.
//!
//! See `docs/DESIGN.md` for architecture and requirements.
//! See `examples/v1.4/HARNESS.md` for the repeatable `v1.4` smoke-test
//! runbook covering the programmatic harness, migrated REPL, multi-guest VFS,
//! Motlie `vnet`, and real interactive SSH validation paths.

#[cfg(feature = "host-runtime")]
pub mod artifacts;
#[cfg(feature = "host-runtime")]
pub mod backend;
#[cfg(feature = "host-runtime")]
pub mod ca;
#[cfg(feature = "guest-vfs")]
pub mod guest;
#[cfg(feature = "host-runtime")]
pub mod guestfs;
#[cfg(feature = "host-runtime")]
pub mod image;
#[cfg(feature = "host-runtime")]
pub mod network;
#[cfg(feature = "host-runtime")]
pub mod network_alloc;
#[cfg(feature = "host-runtime")]
pub mod observability;
#[cfg(feature = "host-runtime")]
pub mod orchestrator;
#[cfg(feature = "host-runtime")]
pub mod provisioning;
#[cfg(feature = "host-runtime")]
pub mod runtime;
#[cfg(feature = "host-runtime")]
pub mod spec;
#[cfg(feature = "host-runtime")]
pub mod ssh;
