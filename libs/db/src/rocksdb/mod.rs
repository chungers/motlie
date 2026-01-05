//! Common RocksDB storage infrastructure.
//!
//! This module provides generic storage infrastructure that eliminates boilerplate
//! across subsystems (graph, vector) while allowing subsystem-specific behavior.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     StorageSubsystem Trait                          │
//! │  Defines: column families, cache type, prewarm logic                │
//! └─────────────────────────────────────────────────────────────────────┘
//!                              ▲
//!                              │ implements
//!              ┌───────────────┼───────────────┐
//!              │               │               │
//! ┌────────────┴───┐   ┌───────┴───────┐   ┌───┴────────────┐
//! │ graph::Subsys  │   │ vector::Subsys│   │ (future)       │
//! │                │   │               │   │                │
//! │ - NameCache    │   │ - Registry    │   │                │
//! │ - prewarm_names│   │ - prewarm_emb │   │                │
//! └────────────────┘   └───────────────┘   └────────────────┘
//!         │                    │
//!         ▼                    ▼
//! ┌────────────────┐   ┌───────────────┐
//! │ Storage<Graph> │   │ Storage<Vec>  │   ← Generic Storage<S>
//! └────────────────┘   └───────────────┘
//! ```
//!
//! # Usage Patterns
//!
//! ## Standalone (single subsystem)
//!
//! ```ignore
//! let mut storage = graph::Storage::readonly(path);
//! storage.ready()?;
//! let cache = storage.cache();
//! ```
//!
//! ## Shared (multiple subsystems via StorageBuilder)
//!
//! ```ignore
//! // Create components and get cache references before boxing
//! let graph_component = graph::component();
//! let name_cache = graph_component.cache().clone();
//!
//! let shared = StorageBuilder::new(path)
//!     .with_component(Box::new(graph_component))
//!     .with_component(Box::new(vector::component()))
//!     .build()?;
//! ```
//!
//! See `README.md` in this directory for detailed architecture documentation.

mod cf_traits;
mod config;
mod handle;
mod storage;
mod subsystem;

// Re-exports
pub use cf_traits::{
    prewarm_cf, ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde, HotColumnFamilyRecord,
    MutationCodec,
};
pub use config::BlockCacheConfig;
pub use handle::{DatabaseHandle, StorageMode, StorageOptions};
pub use storage::Storage;
pub use subsystem::{ComponentWrapper, DbAccess, StorageSubsystem};
