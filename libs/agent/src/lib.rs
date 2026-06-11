//! Agent interaction policy and voice protocol contracts.
//!
//! The default `channel` feature provides managed prompt delivery through a
//! process-local [`Channel`] keyed by a stable tmux session identity. The
//! `voice` modules are serde-only protocol contracts and remain available when
//! default features are disabled.

pub mod voice;

#[cfg(feature = "channel")]
mod channel;

#[cfg(feature = "channel")]
pub use channel::*;
