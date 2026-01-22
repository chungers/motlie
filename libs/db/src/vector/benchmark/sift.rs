//! SIFT dataset loading for vector benchmarks.
//!
//! Supports loading SIFT1M and SIFT10K datasets in fvecs/ivecs format.
//! These are industry-standard benchmarks for ANN algorithms.
//!
//! ## File Formats
//!
//! ### fvecs (float vectors)
//! Each vector: 4 bytes (dim as i32) + dim * 4 bytes (f32 values)
//!
//! ### ivecs (integer vectors)
//! Each vector: 4 bytes (dim as i32) + dim * 4 bytes (i32 values)
//! Used for ground truth neighbor indices.
//!
//! ## Sources
//! - Original: http://corpus-texmex.irisa.fr/
//! - HuggingFace: https://huggingface.co/datasets/qbo-odp/sift1m

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

use crate::vector::Distance;

/// SIFT dataset URLs from HuggingFace.
const SIFT_BASE_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs";
const SIFT_QUERY_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs";
const SIFT_GT_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs";

/// Default embedding dimensions for SIFT (128D).
pub const SIFT_EMBEDDING_DIM: usize = 128;

/// SIFT dataset with base and query vectors plus ground truth.
#[derive(Debug, Clone)]
pub struct SiftDataset {
    /// Base vectors for indexing.
    pub base_vectors: Vec<Vec<f32>>,
    /// Query vectors for search evaluation.
    pub query_vectors: Vec<Vec<f32>>,
    /// Pre-computed ground truth for the full 1M dataset.
    /// For subsets, ground truth is recomputed via brute force.
    ground_truth_full: Vec<Vec<i32>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl SiftDataset {
    /// Download SIFT dataset if not already present.
    pub fn download(data_dir: &Path) -> Result<()> {
        let base_path = data_dir.join("sift_base.fvecs");
        let query_path = data_dir.join("sift_query.fvecs");
        let gt_path = data_dir.join("sift_groundtruth.ivecs");

        if !base_path.exists() {
            println!("Downloading SIFT base vectors (~500MB)...");
            download_file(SIFT_BASE_URL, &base_path)?;
        } else {
            println!("SIFT base vectors already exist: {:?}", base_path);
        }

        if !query_path.exists() {
            println!("Downloading SIFT query vectors (~5MB)...");
            download_file(SIFT_QUERY_URL, &query_path)?;
        } else {
            println!("SIFT query vectors already exist: {:?}", query_path);
        }

        if !gt_path.exists() {
            println!("Downloading SIFT ground truth (~4MB)...");
            download_file(SIFT_GT_URL, &gt_path)?;
        } else {
            println!("SIFT ground truth already exist: {:?}", gt_path);
        }

        Ok(())
    }

    /// Load SIFT dataset from data directory.
    ///
    /// # Arguments
    ///
    /// * `data_dir` - Directory containing sift_*.fvecs/ivecs files
    /// * `max_base` - Maximum number of base vectors to load
    /// * `max_queries` - Maximum number of query vectors to load
    pub fn load(data_dir: &Path, max_base: usize, max_queries: usize) -> Result<Self> {
        let base_path = data_dir.join("sift_base.fvecs");
        let query_path = data_dir.join("sift_query.fvecs");
        let gt_path = data_dir.join("sift_groundtruth.ivecs");

        if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
            anyhow::bail!(
                "SIFT dataset not found. Run with --download first.\n\
                 Expected: {:?}, {:?}, {:?}",
                base_path,
                query_path,
                gt_path
            );
        }

        println!("Loading SIFT base vectors (max {})...", max_base);
        let base_vectors = read_fvecs_limited(&base_path, max_base)?;
        println!("  Loaded {} base vectors ({}D)", base_vectors.len(), base_vectors[0].len());

        println!("Loading SIFT query vectors (max {})...", max_queries);
        let query_vectors = read_fvecs_limited(&query_path, max_queries)?;
        println!("  Loaded {} query vectors", query_vectors.len());

        println!("Loading SIFT ground truth...");
        let ground_truth_full = read_ivecs_limited(&gt_path, max_queries)?;
        println!("  Loaded ground truth for {} queries", ground_truth_full.len());

        Ok(Self {
            dim: base_vectors.first().map(|v| v.len()).unwrap_or(SIFT_EMBEDDING_DIM),
            base_vectors,
            query_vectors,
            ground_truth_full,
        })
    }

    /// Get a subset of the dataset for benchmarking.
    ///
    /// For small subsets (< 100K base vectors), ground truth is recomputed
    /// via brute force since the pre-computed indices reference the full 1M.
    pub fn subset(&self, num_vectors: usize, num_queries: usize) -> SiftSubset {
        let num_vectors = num_vectors.min(self.base_vectors.len());
        let num_queries = num_queries.min(self.query_vectors.len());

        let db_vectors: Vec<Vec<f32>> = self.base_vectors[..num_vectors].to_vec();
        let queries: Vec<Vec<f32>> = self.query_vectors[..num_queries].to_vec();

        SiftSubset {
            db_vectors,
            queries,
            dim: self.dim,
        }
    }

    /// Number of loaded base vectors.
    pub fn len(&self) -> usize {
        self.base_vectors.len()
    }

    /// Whether dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.base_vectors.is_empty()
    }
}

/// A subset of SIFT data for benchmarking.
#[derive(Debug, Clone)]
pub struct SiftSubset {
    /// Database vectors (base vectors).
    pub db_vectors: Vec<Vec<f32>>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl SiftSubset {
    /// Compute brute-force ground truth for Recall@k calculation using L2 distance.
    ///
    /// Returns top-k indices for each query using exact search.
    pub fn compute_ground_truth_topk(&self, k: usize) -> Vec<Vec<usize>> {
        self.compute_ground_truth_topk_with_distance(k, Distance::L2)
    }

    /// Compute brute-force ground truth with specified distance metric.
    pub fn compute_ground_truth_topk_with_distance(
        &self,
        k: usize,
        distance: Distance,
    ) -> Vec<Vec<usize>> {
        println!("Computing brute-force ground truth (k={}, {:?})...", k, distance);

        let mut results = Vec::with_capacity(self.queries.len());

        for (qi, query) in self.queries.iter().enumerate() {
            // Compute distances to all database vectors
            let mut distances: Vec<(usize, f32)> = self
                .db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, distance.compute(query, v)))
                .collect();

            // Sort by distance (ascending)
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Take top-k
            let topk: Vec<usize> = distances.iter().take(k).map(|(i, _)| *i).collect();
            results.push(topk);

            if (qi + 1) % 100 == 0 || qi + 1 == self.queries.len() {
                println!("  Computed {}/{} queries", qi + 1, self.queries.len());
            }
        }

        results
    }

    /// Number of database vectors.
    pub fn num_vectors(&self) -> usize {
        self.db_vectors.len()
    }

    /// Number of queries.
    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }
}

// ============================================================================
// fvecs/ivecs File Format Readers
// ============================================================================

/// Read all vectors from fvecs format file.
///
/// Format: Each vector is stored as:
/// - 4 bytes: dimension (i32, little-endian)
/// - dim * 4 bytes: vector components (f32, little-endian)
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open fvecs file: {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut bytes_read = 0u64;

    while bytes_read < file_size {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        reader
            .read_exact(&mut dim_buf)
            .with_context(|| "Failed to read dimension")?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        bytes_read += 4;

        // Read vector components
        let mut vector = vec![0.0f32; dim];
        for v in vector.iter_mut() {
            let mut val_buf = [0u8; 4];
            reader
                .read_exact(&mut val_buf)
                .with_context(|| "Failed to read vector component")?;
            *v = f32::from_le_bytes(val_buf);
        }
        bytes_read += (dim * 4) as u64;

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from fvecs format file with limit.
pub fn read_fvecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open fvecs file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();

    while vectors.len() < limit {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector components
        let mut vector = vec![0.0f32; dim];
        for v in vector.iter_mut() {
            let mut val_buf = [0u8; 4];
            reader
                .read_exact(&mut val_buf)
                .with_context(|| "Failed to read vector component")?;
            *v = f32::from_le_bytes(val_buf);
        }

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read all integer vectors from ivecs format file (for ground truth).
///
/// Format: Each vector is stored as:
/// - 4 bytes: dimension (i32, little-endian)
/// - dim * 4 bytes: vector components (i32, little-endian)
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open ivecs file: {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut bytes_read = 0u64;

    while bytes_read < file_size {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        reader
            .read_exact(&mut dim_buf)
            .with_context(|| "Failed to read dimension")?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        bytes_read += 4;

        // Read vector components
        let mut vector = vec![0i32; dim];
        for v in vector.iter_mut() {
            let mut val_buf = [0u8; 4];
            reader
                .read_exact(&mut val_buf)
                .with_context(|| "Failed to read vector component")?;
            *v = i32::from_le_bytes(val_buf);
        }
        bytes_read += (dim * 4) as u64;

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read integer vectors from ivecs format file with limit.
pub fn read_ivecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<i32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open ivecs file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();

    while vectors.len() < limit {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector components
        let mut vector = vec![0i32; dim];
        for v in vector.iter_mut() {
            let mut val_buf = [0u8; 4];
            reader
                .read_exact(&mut val_buf)
                .with_context(|| "Failed to read vector component")?;
            *v = i32::from_le_bytes(val_buf);
        }

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Download a file from URL to local path using curl.
fn download_file(url: &str, path: &Path) -> Result<()> {
    use std::process::Command;

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let status = Command::new("curl")
        .args(["-L", "-o", path.to_str().unwrap(), url])
        .status()
        .context("Failed to run curl")?;

    if !status.success() {
        anyhow::bail!("Download failed with status: {}", status);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_read_fvecs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fvecs");

        // Write test fvecs file: 2 vectors of dim 3
        let mut file = File::create(&path).unwrap();
        // Vector 1: dim=3, values=[1.0, 2.0, 3.0]
        file.write_all(&3i32.to_le_bytes()).unwrap();
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&3.0f32.to_le_bytes()).unwrap();
        // Vector 2: dim=3, values=[4.0, 5.0, 6.0]
        file.write_all(&3i32.to_le_bytes()).unwrap();
        file.write_all(&4.0f32.to_le_bytes()).unwrap();
        file.write_all(&5.0f32.to_le_bytes()).unwrap();
        file.write_all(&6.0f32.to_le_bytes()).unwrap();

        let vectors = read_fvecs(&path).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vectors[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_fvecs_limited() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fvecs");

        // Write 3 vectors
        let mut file = File::create(&path).unwrap();
        for i in 0..3 {
            file.write_all(&2i32.to_le_bytes()).unwrap();
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
            file.write_all(&((i + 1) as f32).to_le_bytes()).unwrap();
        }

        // Read only 2
        let vectors = read_fvecs_limited(&path, 2).unwrap();
        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_read_ivecs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ivecs");

        // Write test ivecs file
        let mut file = File::create(&path).unwrap();
        file.write_all(&3i32.to_le_bytes()).unwrap();
        file.write_all(&10i32.to_le_bytes()).unwrap();
        file.write_all(&20i32.to_le_bytes()).unwrap();
        file.write_all(&30i32.to_le_bytes()).unwrap();

        let vectors = read_ivecs(&path).unwrap();
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_sift_subset_ground_truth() {
        // Create a small synthetic subset
        let subset = SiftSubset {
            db_vectors: vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            queries: vec![vec![0.1, 0.1]],
            dim: 2,
        };

        let gt = subset.compute_ground_truth_topk(2);
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].len(), 2);
        // Closest to [0.1, 0.1] should be [0,0] then [1,0] or [0,1]
        assert_eq!(gt[0][0], 0); // [0,0] is closest
    }
}
