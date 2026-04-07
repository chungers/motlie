//! Motlie-specific guest backing providers such as guestfs, userspace vnet,
//! and the SSH proxy/control-plane integration.
//!
//! The current `v1.4` code still exposes these pieces through top-level
//! modules while the API shape converges. This module is the intended vertical
//! slice home for that functionality.
