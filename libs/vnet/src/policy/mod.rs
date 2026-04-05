//! Network policy detection primitives and analysis utilities.
//!
//! This module provides reusable building blocks for `EgressPolicy`
//! implementations: entropy analysis for DNS exfiltration detection,
//! domain name utilities for category matching, and documented patterns
//! for byte ratio, flow duration, beacon, and first-seen detectors.

pub mod entropy;
pub mod domain;
// Future modules (documented in README with feasibility assessment):
// pub mod ratio;      — outbound/inbound byte ratio analysis
// pub mod duration;   — flow duration / low-and-slow leak detection
// pub mod beacon;     — heartbeat / C2 beacon periodicity detection
// pub mod first_seen; — novel destination tracking
