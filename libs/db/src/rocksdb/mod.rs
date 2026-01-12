//! Common RocksDB storage infrastructure.
//!
//! This module provides generic storage infrastructure that eliminates boilerplate
//! across subsystems (graph, vector) while allowing subsystem-specific behavior.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     RocksdbSubsystem Trait                           │
//! │  Extends: SubsystemProvider<TransactionDB> with CF management       │
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
//! let graph_subsystem = graph::Subsystem::new();
//! let name_cache = graph_subsystem.cache().clone();
//!
//! let shared = StorageBuilder::new(path)
//!     .with_rocksdb(Box::new(graph_subsystem))
//!     .with_rocksdb(Box::new(vector::Subsystem::new()))
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
pub use subsystem::{DbAccess, RocksdbSubsystem, StorageSubsystem};
