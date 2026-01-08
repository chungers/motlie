//! Database handle types for RocksDB storage.
//!
//! Provides abstractions for different database access modes:
//! - `DatabaseHandle`: Enum wrapping read-only, read-write, and secondary DB instances
//! - `StorageMode`: Configuration for how the database should be opened
//! - `StorageOptions`: Default RocksDB options per access mode

use std::path::PathBuf;

use rocksdb::{Options, TransactionDB, DB};

// ============================================================================
// DatabaseHandle
// ============================================================================

/// Handle for either a read-only DB or read-write TransactionDB.
///
/// This enum abstracts over the different RocksDB instance types:
/// - `ReadOnly`: Uses `DB` opened in read-only mode
/// - `ReadWrite`: Uses `TransactionDB` for ACID transactions
/// - `Secondary`: Uses `DB` as a secondary instance that follows a primary
pub enum DatabaseHandle {
    /// Read-only database access
    ReadOnly(DB),
    /// Read-write access with transaction support
    ReadWrite(TransactionDB),
    /// Secondary instance that follows a primary database
    Secondary(DB),
}

impl DatabaseHandle {
    /// Get TransactionDB reference if in ReadWrite mode.
    pub fn as_transaction_db(&self) -> Option<&TransactionDB> {
        match self {
            DatabaseHandle::ReadWrite(txn_db) => Some(txn_db),
            DatabaseHandle::ReadOnly(_) | DatabaseHandle::Secondary(_) => None,
        }
    }

    /// Get DB reference if in ReadOnly or Secondary mode.
    pub fn as_db(&self) -> Option<&DB> {
        match self {
            DatabaseHandle::ReadOnly(db) | DatabaseHandle::Secondary(db) => Some(db),
            DatabaseHandle::ReadWrite(_) => None,
        }
    }

    /// Check if this is a read-write handle with transaction support.
    pub fn is_read_write(&self) -> bool {
        matches!(self, DatabaseHandle::ReadWrite(_))
    }

    /// Check if this is a secondary instance.
    pub fn is_secondary(&self) -> bool {
        matches!(self, DatabaseHandle::Secondary(_))
    }
}

// ============================================================================
// StorageMode
// ============================================================================

/// Storage access mode configuration.
///
/// Determines how the database is opened and what operations are permitted.
pub enum StorageMode {
    /// Read-only access - multiple instances can open simultaneously
    ReadOnly,
    /// Read-write access with exclusive TransactionDB
    ReadWrite,
    /// Secondary instance following a primary database
    Secondary {
        /// Path for secondary's MANIFEST (must differ from primary)
        secondary_path: PathBuf,
    },
}

// ============================================================================
// StorageOptions
// ============================================================================

/// Default RocksDB options factory for each storage mode.
pub struct StorageOptions;

impl StorageOptions {
    /// Default options for read-write mode.
    ///
    /// Settings:
    /// - `error_if_exists`: false (allow opening existing DBs)
    /// - `create_if_missing`: true (create new DBs)
    /// - `create_missing_column_families`: true (auto-create CFs)
    /// - Parallelism: uses all available CPUs
    /// - Write buffer: 128MB with up to 4 buffers before stalling
    pub fn default_for_readwrite() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);

        // Parallelism: use available CPU cores for background jobs
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get() as i32)
            .unwrap_or(4);
        options.increase_parallelism(num_cpus);
        options.set_max_background_jobs(num_cpus.min(8));

        // Write buffer tuning for bulk insert performance
        // Larger buffers = fewer flushes during index build
        options.set_write_buffer_size(128 * 1024 * 1024); // 128MB per buffer
        options.set_max_write_buffer_number(4); // Up to 4 buffers before stalling

        // Optimize for point lookups (most HNSW operations)
        options.set_advise_random_on_open(true);

        options
    }

    /// Default options for read-only mode.
    ///
    /// Settings:
    /// - `error_if_exists`: false
    /// - `create_if_missing`: false (DB must exist)
    /// - `create_missing_column_families`: false
    pub fn default_for_readonly() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(false);
        options.create_missing_column_families(false);
        options
    }

    /// Default options for secondary mode.
    ///
    /// Key requirement: `max_open_files = -1` (must keep all file descriptors open)
    pub fn default_for_secondary() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(false);
        options.create_missing_column_families(false);
        options.set_max_open_files(-1); // Required for secondary instances
        options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_options_readwrite() {
        let opts = StorageOptions::default_for_readwrite();
        // Options are created successfully
        drop(opts);
    }

    #[test]
    fn test_storage_options_readonly() {
        let opts = StorageOptions::default_for_readonly();
        drop(opts);
    }

    #[test]
    fn test_storage_options_secondary() {
        let opts = StorageOptions::default_for_secondary();
        drop(opts);
    }
}
