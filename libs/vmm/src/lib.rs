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

pub mod ca;
pub mod artifacts;
pub mod backend;
pub mod guestfs;
pub mod network;
pub mod network_alloc;
pub mod observability;
pub mod orchestrator;
pub mod runtime;
pub mod ssh;
pub mod spec;
