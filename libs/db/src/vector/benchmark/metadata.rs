//! Benchmark metadata for incremental index builds
//!
//! This module provides persistence of benchmark state to enable:
//! - Incremental index builds (add vectors to existing index)
//! - Checkpointing (resume after interruption)
//! - Configuration validation (prevent mismatches)
//! - Ground truth caching (avoid O(n^2) recomputation)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Persisted benchmark metadata for incremental builds.
///
/// Stored as JSON alongside the database to track:
/// - How many vectors are indexed
/// - Configuration parameters (must match for incremental builds)
/// - Random seeds for reproducibility
///
/// # Example
///
/// ```ignore
/// use motlie_db::vector::benchmark::BenchmarkMetadata;
///
/// // Create new metadata
/// let mut meta = BenchmarkMetadata::new(128, "cosine", true, 4, true, 16, 200, "random");
///
/// // After indexing
/// meta.num_vectors = 10000;
/// meta.checkpoint(&db_path)?;
///
/// // On next run, load and validate
/// let loaded = BenchmarkMetadata::load(&db_path)?;
/// loaded.validate_config(128, "cosine", true, 4, true, 16, 200, "random")?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Number of vectors currently indexed
    pub num_vectors: usize,

    /// Vector dimensionality
    pub dim: usize,

    /// Distance metric: "l2" or "cosine"
    pub distance: String,

    /// RaBitQ enabled
    pub rabitq_enabled: bool,

    /// RaBitQ bits per dimension (1, 2, or 4)
    pub rabitq_bits: u8,

    /// ADC mode enabled
    pub adc_enabled: bool,

    /// Random seed for reproducible vector generation
    pub vector_seed: u64,

    /// Random seed for query generation (separate from vectors)
    pub query_seed: u64,

    /// HNSW M parameter
    pub hnsw_m: usize,

    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,

    /// Dataset name (random, sift10k, laion, etc.)
    pub dataset: String,

    /// Timestamp of last update (ISO 8601)
    pub last_updated: String,

    /// Version of metadata format (for future migrations)
    pub version: u32,
}

impl BenchmarkMetadata {
    /// Current metadata format version
    pub const CURRENT_VERSION: u32 = 1;

    /// Metadata filename
    pub const FILENAME: &'static str = "benchmark_metadata.json";

    /// Create new metadata with default seeds
    pub fn new(
        dim: usize,
        distance: &str,
        rabitq_enabled: bool,
        rabitq_bits: u8,
        adc_enabled: bool,
        hnsw_m: usize,
        hnsw_ef_construction: usize,
        dataset: &str,
    ) -> Self {
        Self {
            num_vectors: 0,
            dim,
            distance: distance.to_string(),
            rabitq_enabled,
            rabitq_bits,
            adc_enabled,
            vector_seed: 42,      // Fixed seed for reproducibility
            query_seed: 12345,    // Different seed for queries
            hnsw_m,
            hnsw_ef_construction,
            dataset: dataset.to_string(),
            last_updated: chrono_now(),
            version: Self::CURRENT_VERSION,
        }
    }

    /// Get metadata file path for a database
    pub fn path(db_path: &Path) -> PathBuf {
        db_path.join(Self::FILENAME)
    }

    /// Check if metadata exists for a database
    pub fn exists(db_path: &Path) -> bool {
        Self::path(db_path).exists()
    }

    /// Load metadata from database directory
    pub fn load(db_path: &Path) -> Result<Self> {
        let path = Self::path(db_path);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read metadata from {}", path.display()))?;
        let metadata: Self = serde_json::from_str(&content)
            .with_context(|| "Failed to parse metadata JSON")?;
        Ok(metadata)
    }

    /// Save metadata to database directory
    pub fn save(&self, db_path: &Path) -> Result<()> {
        let path = Self::path(db_path);
        let content = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize metadata")?;
        fs::write(&path, content)
            .with_context(|| format!("Failed to write metadata to {}", path.display()))?;
        Ok(())
    }

    /// Update timestamp and save
    pub fn checkpoint(&mut self, db_path: &Path) -> Result<()> {
        self.last_updated = chrono_now();
        self.save(db_path)
    }

    /// Validate that configuration matches for incremental build
    pub fn validate_config(
        &self,
        dim: usize,
        distance: &str,
        rabitq_enabled: bool,
        rabitq_bits: u8,
        adc_enabled: bool,
        hnsw_m: usize,
        hnsw_ef_construction: usize,
        dataset: &str,
    ) -> Result<()> {
        let mut errors = Vec::new();

        if self.dim != dim {
            errors.push(format!("dim mismatch: {} vs {}", self.dim, dim));
        }
        if self.distance != distance {
            errors.push(format!("distance mismatch: {} vs {}", self.distance, distance));
        }
        if self.rabitq_enabled != rabitq_enabled {
            errors.push(format!(
                "rabitq_enabled mismatch: {} vs {}",
                self.rabitq_enabled, rabitq_enabled
            ));
        }
        if self.rabitq_bits != rabitq_bits {
            errors.push(format!(
                "rabitq_bits mismatch: {} vs {}",
                self.rabitq_bits, rabitq_bits
            ));
        }
        if self.adc_enabled != adc_enabled {
            errors.push(format!(
                "adc_enabled mismatch: {} vs {}",
                self.adc_enabled, adc_enabled
            ));
        }
        if self.hnsw_m != hnsw_m {
            errors.push(format!("hnsw_m mismatch: {} vs {}", self.hnsw_m, hnsw_m));
        }
        if self.hnsw_ef_construction != hnsw_ef_construction {
            errors.push(format!(
                "hnsw_ef_construction mismatch: {} vs {}",
                self.hnsw_ef_construction, hnsw_ef_construction
            ));
        }
        if self.dataset != dataset {
            errors.push(format!("dataset mismatch: {} vs {}", self.dataset, dataset));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "Configuration mismatch for incremental build:\n  {}",
                errors.join("\n  ")
            )
        }
    }
}

/// Cached ground truth for a specific index size and k value.
///
/// Ground truth computation is O(n * m) where n=num_vectors, m=num_queries.
/// Caching avoids recomputation when only running queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthCache {
    /// Number of vectors in index when computed
    pub num_vectors: usize,

    /// Number of queries
    pub num_queries: usize,

    /// k value used
    pub k: usize,

    /// Distance metric used
    pub distance: String,

    /// Ground truth results: query_idx -> [(distance, vec_id), ...]
    pub results: Vec<Vec<(f32, u32)>>,

    /// Timestamp of computation
    pub computed_at: String,
}

impl GroundTruthCache {
    /// Get cache filename for given parameters
    pub fn filename(num_vectors: usize, num_queries: usize, k: usize, distance: &str) -> String {
        format!(
            "ground_truth_{}v_{}q_k{}_{}.json",
            num_vectors, num_queries, k, distance
        )
    }

    /// Get cache file path
    pub fn path(db_path: &Path, num_vectors: usize, num_queries: usize, k: usize, distance: &str) -> PathBuf {
        db_path.join(Self::filename(num_vectors, num_queries, k, distance))
    }

    /// Check if cache exists
    pub fn exists(db_path: &Path, num_vectors: usize, num_queries: usize, k: usize, distance: &str) -> bool {
        Self::path(db_path, num_vectors, num_queries, k, distance).exists()
    }

    /// Load from cache
    pub fn load(db_path: &Path, num_vectors: usize, num_queries: usize, k: usize, distance: &str) -> Result<Self> {
        let path = Self::path(db_path, num_vectors, num_queries, k, distance);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read ground truth cache from {}", path.display()))?;
        let cache: Self = serde_json::from_str(&content)
            .with_context(|| "Failed to parse ground truth cache JSON")?;

        // Validate cache matches requested parameters
        if cache.num_vectors != num_vectors || cache.num_queries != num_queries || cache.k != k {
            anyhow::bail!("Ground truth cache parameters mismatch");
        }

        Ok(cache)
    }

    /// Save to cache
    pub fn save(&self, db_path: &Path) -> Result<()> {
        let path = Self::path(db_path, self.num_vectors, self.num_queries, self.k, &self.distance);
        let content = serde_json::to_string(self)
            .with_context(|| "Failed to serialize ground truth cache")?;
        fs::write(&path, content)
            .with_context(|| format!("Failed to write ground truth cache to {}", path.display()))?;
        Ok(())
    }

    /// Create new cache from computed ground truth
    pub fn new(
        num_vectors: usize,
        num_queries: usize,
        k: usize,
        distance: &str,
        results: Vec<Vec<(f32, u32)>>,
    ) -> Self {
        Self {
            num_vectors,
            num_queries,
            k,
            distance: distance.to_string(),
            results,
            computed_at: chrono_now(),
        }
    }
}

/// Get current timestamp in ISO 8601 format (without external chrono dependency)
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();

    // Simple ISO 8601 format (UTC)
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate year/month/day from days since epoch (1970-01-01)
    let mut days = days_since_epoch as i64;
    let mut year = 1970i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let leap = is_leap_year(year);
    let month_days = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &md in &month_days {
        if days < md as i64 {
            break;
        }
        days -= md as i64;
        month += 1;
    }
    let day = days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_metadata_roundtrip() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let mut metadata = BenchmarkMetadata::new(
            128,
            "cosine",
            true,
            4,
            true,
            16,
            200,
            "random",
        );
        metadata.num_vectors = 10000;

        // Save and load
        metadata.save(db_path).unwrap();
        let loaded = BenchmarkMetadata::load(db_path).unwrap();

        assert_eq!(loaded.num_vectors, 10000);
        assert_eq!(loaded.dim, 128);
        assert_eq!(loaded.distance, "cosine");
        assert!(loaded.rabitq_enabled);
        assert_eq!(loaded.rabitq_bits, 4);
        assert!(loaded.adc_enabled);
    }

    #[test]
    fn test_config_validation() {
        let metadata = BenchmarkMetadata::new(128, "cosine", true, 4, true, 16, 200, "random");

        // Same config should pass
        metadata
            .validate_config(128, "cosine", true, 4, true, 16, 200, "random")
            .unwrap();

        // Different dim should fail
        assert!(metadata
            .validate_config(256, "cosine", true, 4, true, 16, 200, "random")
            .is_err());

        // Different bits should fail
        assert!(metadata
            .validate_config(128, "cosine", true, 2, true, 16, 200, "random")
            .is_err());
    }

    #[test]
    fn test_ground_truth_cache() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let results = vec![
            vec![(0.1, 1), (0.2, 2), (0.3, 3)],
            vec![(0.15, 4), (0.25, 5), (0.35, 6)],
        ];

        let cache = GroundTruthCache::new(1000, 2, 3, "cosine", results.clone());
        cache.save(db_path).unwrap();

        let loaded = GroundTruthCache::load(db_path, 1000, 2, 3, "cosine").unwrap();
        assert_eq!(loaded.results.len(), 2);
        assert_eq!(loaded.results[0].len(), 3);
    }

    #[test]
    fn test_chrono_now() {
        let ts = chrono_now();
        // Should be in ISO 8601 format
        assert!(ts.contains("T"));
        assert!(ts.ends_with("Z"));
        // Year should be reasonable
        assert!(ts.starts_with("20"));
    }
}
