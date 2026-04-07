//! motlie-vmm: Reusable VM orchestration with SSH proxy and programmatic control plane.
//!
//! This crate provides host-side VM orchestration extracted from the proven
//! `v1.2` example lineage, plus a new SSH proxy (russh) that replaces
//! TAP-based ingress and enables programmatic guest command execution.
//!
//! See `docs/DESIGN.md` for architecture and requirements.

pub mod ca;
pub mod artifacts;
pub mod network;
pub mod network_alloc;
pub mod ssh;
pub mod spec;
