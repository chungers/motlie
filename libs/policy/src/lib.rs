//! motlie-policy: Shared detection primitives for policy implementations.
//!
//! This crate provides reusable analysis functions for `motlie-vfs` and
//! `motlie-vnet` policy implementations. It contains no policy logic —
//! only the building blocks that policies compose.
//!
//! # Design principle
//!
//! Detection primitives are stack-agnostic. The same `shannon_entropy()`
//! function is used by vnet's DNS exfiltration detector and could be used
//! by a vfs policy detecting obfuscated filenames. Neither vfs nor vnet
//! depends on the other — both depend on this crate.

pub mod entropy;
pub mod domain;
// Future modules:
// pub mod ratio;    — byte ratio analysis (outbound/inbound)
// pub mod duration; — flow duration / low-and-slow detection
// pub mod beacon;   — heartbeat / C2 beacon periodicity detection
// pub mod first_seen; — novel destination tracking
