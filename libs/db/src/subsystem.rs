//! Base subsystem provider trait for modular storage initialization.
//!
//! This module defines the `SubsystemProvider` trait that serves as the foundation
//! for all storage subsystems (fulltext, vector, graph). It extends `SubsystemInfo`
//! from `motlie_core::telemetry` with lifecycle hooks.
//!
//! ## Design Rationale
//!
//! - `SubsystemInfo` provides identity and observability (name, info_lines)
//! - `SubsystemProvider<B>` adds lifecycle hooks (on_ready, on_shutdown)
//! - Backend-specific traits (`FulltextSubsystem`, `RocksdbSubsystem`) extend this
//!
//! ## Trait Hierarchy
//!
//! ```text
//! SubsystemInfo (motlie_core)
//!     │
//!     └── SubsystemProvider<B> (this module)
//!             │
//!             ├── FulltextSubsystem (fulltext module, B = tantivy::Index)
//!             │
//!             └── RocksdbSubsystem (rocksdb module, B = TransactionDB)
//! ```

use anyhow::Result;
use motlie_core::telemetry::SubsystemInfo;

/// Base trait for all storage subsystems.
///
/// Extends [`SubsystemInfo`] with lifecycle hooks for initialization and shutdown.
/// Generic over backend type `B` (e.g., `tantivy::Index`, `TransactionDB`).
///
/// # Lifecycle
///
/// 1. Subsystem is created with configuration
/// 2. Backend is opened/initialized
/// 3. `on_ready(backend)` is called for subsystem-specific initialization
/// 4. ... application runs ...
/// 5. `on_shutdown()` is called before backend is closed
///
/// # Example
///
/// ```ignore
/// use motlie_core::telemetry::SubsystemInfo;
/// use motlie_db::SubsystemProvider;
///
/// struct MySubsystem { /* ... */ }
///
/// impl SubsystemInfo for MySubsystem {
///     fn name(&self) -> &'static str { "my-subsystem" }
///     fn info_lines(&self) -> Vec<(&'static str, String)> { vec![] }
/// }
///
/// impl SubsystemProvider<MyBackend> for MySubsystem {
///     fn on_ready(&self, backend: &MyBackend) -> anyhow::Result<()> {
///         // Pre-warm caches, validate schema, etc.
///         Ok(())
///     }
///
///     fn on_shutdown(&self) -> anyhow::Result<()> {
///         // Flush caches, close resources, etc.
///         Ok(())
///     }
/// }
/// ```
pub trait SubsystemProvider<B>: SubsystemInfo + Send + Sync {
    /// Called after the backend is ready for subsystem-specific initialization.
    ///
    /// Use this to:
    /// - Pre-warm caches from stored data
    /// - Validate schema compatibility
    /// - Initialize readers/writers
    ///
    /// # Default
    ///
    /// Default implementation is a no-op, returning `Ok(())`.
    fn on_ready(&self, backend: &B) -> Result<()> {
        let _ = backend;
        Ok(())
    }

    /// Called before shutdown for graceful cleanup.
    ///
    /// Use this to:
    /// - Flush pending writes
    /// - Persist in-memory state
    /// - Close open resources
    ///
    /// # Default
    ///
    /// Default implementation is a no-op, returning `Ok(())`.
    fn on_shutdown(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;

    struct MockSubsystem {
        name: &'static str,
    }

    impl SubsystemInfo for MockSubsystem {
        fn name(&self) -> &'static str {
            self.name
        }

        fn info_lines(&self) -> Vec<(&'static str, String)> {
            vec![("Test", "value".to_string())]
        }
    }

    impl SubsystemProvider<MockBackend> for MockSubsystem {}

    #[test]
    fn test_default_on_ready() {
        let subsystem = MockSubsystem { name: "test" };
        let backend = MockBackend;
        assert!(subsystem.on_ready(&backend).is_ok());
    }

    #[test]
    fn test_default_on_shutdown() {
        let subsystem = MockSubsystem { name: "test" };
        assert!(subsystem.on_shutdown().is_ok());
    }

    #[test]
    fn test_subsystem_info_methods() {
        let subsystem = MockSubsystem { name: "test" };
        assert_eq!(subsystem.name(), "test");
        assert_eq!(subsystem.info_lines().len(), 1);
    }
}
