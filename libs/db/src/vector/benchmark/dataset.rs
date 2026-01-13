//! Dataset loading for vector benchmarks.
//!
//! Supports loading LAION-400M CLIP embeddings and other NPY-format datasets.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::vector::Distance;

/// LAION-400M embedding URLs (CLIP ViT-B-32, 512D, float16).
const IMG_EMB_URL: &str =
    "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_0.npy";
const TEXT_EMB_URL: &str =
    "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_0.npy";

/// Default embedding dimensions for LAION-CLIP (ViT-B-32).
pub const LAION_EMBEDDING_DIM: usize = 512;

/// Default maximum vectors to load.
pub const DEFAULT_MAX_VECTORS: usize = 200_000;

/// Dataset configuration.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Maximum number of vectors to load.
    pub max_vectors: usize,
    /// Expected embedding dimension.
    pub dim: usize,
    /// Whether to print progress messages.
    pub verbose: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_vectors: DEFAULT_MAX_VECTORS,
            dim: LAION_EMBEDDING_DIM,
            verbose: true,
        }
    }
}

impl DatasetConfig {
    /// Create config for a specific dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set maximum vectors to load.
    pub fn with_max_vectors(mut self, max: usize) -> Self {
        self.max_vectors = max;
        self
    }

    /// Enable/disable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// LAION dataset with image (database) and text (query) embeddings.
#[derive(Debug, Clone)]
pub struct LaionDataset {
    /// Image embeddings used as database vectors.
    pub image_embeddings: Vec<Vec<f32>>,
    /// Text embeddings used as query vectors.
    pub text_embeddings: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl LaionDataset {
    /// Download LAION embeddings if not already present.
    pub fn download(data_dir: &Path) -> Result<()> {
        let img_path = data_dir.join("img_emb_0.npy");
        let text_path = data_dir.join("text_emb_0.npy");

        if !img_path.exists() {
            println!("Downloading image embeddings (~1GB)...");
            download_file(IMG_EMB_URL, &img_path)?;
        } else {
            println!("Image embeddings already exist: {:?}", img_path);
        }

        if !text_path.exists() {
            println!("Downloading text embeddings (~1GB)...");
            download_file(TEXT_EMB_URL, &text_path)?;
        } else {
            println!("Text embeddings already exist: {:?}", text_path);
        }

        Ok(())
    }

    /// Load LAION dataset from data directory.
    pub fn load(data_dir: &Path, max_vectors: usize) -> Result<Self> {
        Self::load_with_config(data_dir, DatasetConfig::default().with_max_vectors(max_vectors))
    }

    /// Load LAION dataset with custom configuration.
    pub fn load_with_config(data_dir: &Path, config: DatasetConfig) -> Result<Self> {
        let img_path = data_dir.join("img_emb_0.npy");
        let text_path = data_dir.join("text_emb_0.npy");

        if !img_path.exists() || !text_path.exists() {
            anyhow::bail!(
                "LAION embeddings not found. Run with --download first.\n\
                 Expected: {:?} and {:?}",
                img_path,
                text_path
            );
        }

        let loader = NpyLoader::new().with_verbose(config.verbose);

        let image_embeddings = loader.load(&img_path, config.max_vectors, Some(config.dim))?;
        let text_embeddings = loader.load(&text_path, config.max_vectors, Some(config.dim))?;

        Ok(Self {
            image_embeddings,
            text_embeddings,
            dim: config.dim,
        })
    }

    /// Get a subset of the dataset for benchmarking.
    ///
    /// Uses text embeddings as queries - each text embedding has its
    /// corresponding image embedding at the same index as ground truth.
    pub fn subset(&self, num_vectors: usize, num_queries: usize) -> LaionSubset {
        let db_vectors: Vec<Vec<f32>> = self.image_embeddings[..num_vectors].to_vec();

        // Use text embeddings as queries - sample evenly
        let query_indices: Vec<usize> = (0..num_vectors)
            .step_by(num_vectors / num_queries)
            .take(num_queries)
            .collect();

        let queries: Vec<Vec<f32>> = query_indices
            .iter()
            .map(|&i| self.text_embeddings[i].clone())
            .collect();

        // Ground truth: each query's matching image is at the same index
        let ground_truth: Vec<usize> = query_indices;

        LaionSubset {
            db_vectors,
            queries,
            ground_truth,
            dim: self.dim,
        }
    }

    /// Number of loaded embeddings.
    pub fn len(&self) -> usize {
        self.image_embeddings.len()
    }

    /// Whether dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.image_embeddings.is_empty()
    }
}

/// A subset of LAION data for benchmarking.
#[derive(Debug, Clone)]
pub struct LaionSubset {
    /// Database vectors (image embeddings).
    pub db_vectors: Vec<Vec<f32>>,
    /// Query vectors (text embeddings).
    pub queries: Vec<Vec<f32>>,
    /// Index of the ground truth image for each query.
    pub ground_truth: Vec<usize>,
    /// Embedding dimension.
    pub dim: usize,
}

impl LaionSubset {
    /// Compute brute-force ground truth for Recall@k calculation.
    ///
    /// Returns top-k indices for each query using exact search.
    pub fn compute_ground_truth_topk(&self, k: usize) -> Vec<Vec<usize>> {
        self.compute_ground_truth_topk_with_distance(k, Distance::Cosine)
    }

    /// Compute brute-force ground truth with specified distance metric.
    pub fn compute_ground_truth_topk_with_distance(
        &self,
        k: usize,
        distance: Distance,
    ) -> Vec<Vec<usize>> {
        println!("Computing brute-force ground truth (k={})...", k);

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

            if (qi + 1) % 100 == 0 {
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

/// NPY file loader supporting float16 and float32 formats.
#[derive(Debug, Clone, Default)]
pub struct NpyLoader {
    verbose: bool,
}

impl NpyLoader {
    /// Create a new NPY loader.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Load embeddings from NPY file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to NPY file
    /// * `max_count` - Maximum number of vectors to load
    /// * `expected_dim` - Expected dimension (None to accept any)
    pub fn load(
        &self,
        path: &Path,
        max_count: usize,
        expected_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if self.verbose {
            println!("Loading embeddings from {:?}...", path);
        }

        let file = File::open(path).context("Failed to open NPY file")?;
        let mut reader = BufReader::new(file);

        // Read and parse NPY header
        let (shape, dtype) = self.parse_header(&mut reader)?;

        let num_vectors = shape[0].min(max_count);
        let dim = shape[1];

        if let Some(expected) = expected_dim {
            if dim != expected {
                anyhow::bail!("Expected {}D embeddings, got {}D", expected, dim);
            }
        }

        // Detect dtype
        let is_float16 = dtype.contains("f2") || dtype.contains("float16");
        let is_float32 = dtype.contains("f4") || dtype.contains("float32");

        if !is_float16 && !is_float32 {
            anyhow::bail!(
                "Unsupported dtype: {}. Expected float16 (<f2) or float32 (<f4)",
                dtype
            );
        }

        if self.verbose {
            println!(
                "  Shape: {:?}, dtype: {}, loading first {} vectors",
                shape, dtype, num_vectors
            );
        }

        // Read embeddings based on dtype
        let mut embeddings = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let mut vec = vec![0.0f32; dim];
            if is_float16 {
                for v in vec.iter_mut().take(dim) {
                    let bits = reader.read_u16::<LittleEndian>()?;
                    *v = f16::from_bits(bits).to_f32();
                }
            } else {
                for v in vec.iter_mut().take(dim) {
                    *v = reader.read_f32::<LittleEndian>()?;
                }
            }
            embeddings.push(vec);

            if self.verbose && (i + 1) % 50000 == 0 {
                println!("  Loaded {}/{} vectors", i + 1, num_vectors);
            }
        }

        if self.verbose {
            println!("  Loaded {} vectors ({}D)", embeddings.len(), dim);
        }

        Ok(embeddings)
    }

    /// Parse NPY file header to get shape and dtype.
    fn parse_header<R: Read>(&self, reader: &mut R) -> Result<(Vec<usize>, String)> {
        // Read magic number
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != b"\x93NUMPY" {
            anyhow::bail!("Invalid NPY magic number");
        }

        // Read version
        let major = reader.read_u8()?;
        let minor = reader.read_u8()?;

        // Read header length
        let header_len = if major == 1 {
            reader.read_u16::<LittleEndian>()? as usize
        } else {
            reader.read_u32::<LittleEndian>()? as usize
        };

        // Read header string
        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let header = String::from_utf8_lossy(&header_bytes);

        if self.verbose {
            println!("  NPY v{}.{}, header: {}", major, minor, header.trim());
        }

        let shape = parse_shape_from_header(&header)?;
        let dtype = parse_dtype_from_header(&header)?;

        Ok((shape, dtype))
    }
}

/// Extract shape tuple from NPY header string.
fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    let shape_start = header.find("'shape':").context("No shape in header")?;
    let paren_start = header[shape_start..].find('(').context("No shape tuple")?;
    let paren_end = header[shape_start..].find(')').context("No shape end")?;

    let shape_str = &header[shape_start + paren_start + 1..shape_start + paren_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    Ok(dims)
}

/// Extract dtype from NPY header string.
fn parse_dtype_from_header(header: &str) -> Result<String> {
    let descr_start = header.find("'descr':").context("No descr in header")?;
    let quote_start = header[descr_start + 8..]
        .find('\'')
        .context("No dtype quote")?;
    let quote_end = header[descr_start + 8 + quote_start + 1..]
        .find('\'')
        .context("No dtype end")?;

    let dtype = &header[descr_start + 9 + quote_start..descr_start + 9 + quote_start + quote_end];
    Ok(dtype.to_string())
}

/// Download a file from URL to local path using curl.
fn download_file(url: &str, path: &Path) -> Result<()> {
    use std::process::Command;

    let status = Command::new("curl")
        .args(["-L", "-o", path.to_str().unwrap(), url])
        .status()
        .context("Failed to run curl")?;

    if !status.success() {
        anyhow::bail!("Download failed with status: {}", status);
    }

    Ok(())
}

// ============================================================================
// GIST-960 Dataset
// ============================================================================

/// GIST dataset dimensions.
pub const GIST_EMBEDDING_DIM: usize = 960;

/// GIST dataset size constraints.
pub const GIST_BASE_VECTORS: usize = 1_000_000;
pub const GIST_QUERIES: usize = 1_000;

/// GIST dataset URL (texmex corpus).
const GIST_BASE_URL: &str = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz";

/// GIST-960 dataset with high-dimensional descriptors.
///
/// GIST is a high-dimensional (960D) dataset of image descriptors.
/// Unlike SIFT, these are unnormalized L2 vectors - useful for testing
/// RaBitQ's behavior on non-unit vectors.
///
/// ## Files
/// - `gist_base.fvecs`: 1M base vectors × 960D
/// - `gist_query.fvecs`: 1K query vectors × 960D
/// - `gist_groundtruth.ivecs`: 1K × 100 ground truth indices
///
/// ## Source
/// - http://corpus-texmex.irisa.fr/
#[derive(Debug, Clone)]
pub struct GistDataset {
    /// Base vectors for indexing.
    pub base_vectors: Vec<Vec<f32>>,
    /// Query vectors for search evaluation.
    pub query_vectors: Vec<Vec<f32>>,
    /// Pre-computed ground truth (indices into base_vectors).
    pub ground_truth_full: Vec<Vec<usize>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl GistDataset {
    /// Load GIST dataset from data directory.
    ///
    /// # Arguments
    ///
    /// * `data_dir` - Directory containing gist_*.fvecs/ivecs files
    /// * `max_base` - Maximum number of base vectors to load
    /// * `max_queries` - Maximum number of query vectors to load
    pub fn load(data_dir: &Path, max_base: usize, max_queries: usize) -> Result<Self> {
        let base_path = data_dir.join("gist_base.fvecs");
        let query_path = data_dir.join("gist_query.fvecs");
        let gt_path = data_dir.join("gist_groundtruth.ivecs");

        if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
            anyhow::bail!(
                "GIST dataset not found. Please download from {}\n\
                 Expected: {:?}, {:?}, {:?}",
                GIST_BASE_URL,
                base_path,
                query_path,
                gt_path
            );
        }

        println!("Loading GIST base vectors (max {})...", max_base);
        let base_vectors = super::sift::read_fvecs_limited(&base_path, max_base)?;
        println!(
            "  Loaded {} base vectors ({}D)",
            base_vectors.len(),
            base_vectors.first().map(|v| v.len()).unwrap_or(0)
        );

        println!("Loading GIST query vectors (max {})...", max_queries);
        let query_vectors = super::sift::read_fvecs_limited(&query_path, max_queries)?;
        println!("  Loaded {} query vectors", query_vectors.len());

        println!("Loading GIST ground truth...");
        let gt_raw = super::sift::read_ivecs_limited(&gt_path, max_queries)?;
        // Convert i32 to usize
        let ground_truth_full: Vec<Vec<usize>> = gt_raw
            .into_iter()
            .map(|v| v.into_iter().map(|i| i as usize).collect())
            .collect();
        println!(
            "  Loaded ground truth for {} queries",
            ground_truth_full.len()
        );

        Ok(Self {
            dim: base_vectors
                .first()
                .map(|v| v.len())
                .unwrap_or(GIST_EMBEDDING_DIM),
            base_vectors,
            query_vectors,
            ground_truth_full,
        })
    }

    /// Get a subset of the dataset for benchmarking.
    ///
    /// For small subsets (< full 1M), ground truth is recomputed
    /// via brute force since the pre-computed indices reference the full 1M.
    pub fn subset(&self, num_vectors: usize, num_queries: usize) -> GistSubset {
        let num_vectors = num_vectors.min(self.base_vectors.len());
        let num_queries = num_queries.min(self.query_vectors.len());

        let db_vectors: Vec<Vec<f32>> = self.base_vectors[..num_vectors].to_vec();
        let queries: Vec<Vec<f32>> = self.query_vectors[..num_queries].to_vec();

        // For subsets smaller than full dataset, we can't use pre-computed ground truth
        // since indices may reference vectors not in our subset
        let ground_truth: Option<Vec<Vec<usize>>> = if num_vectors >= self.base_vectors.len() {
            // Can use pre-computed ground truth
            Some(self.ground_truth_full[..num_queries].to_vec())
        } else {
            // Will need to recompute
            None
        };

        GistSubset {
            db_vectors,
            queries,
            precomputed_ground_truth: ground_truth,
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

/// A subset of GIST data for benchmarking.
#[derive(Debug, Clone)]
pub struct GistSubset {
    /// Database vectors (base vectors).
    pub db_vectors: Vec<Vec<f32>>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// Pre-computed ground truth (if using full dataset).
    precomputed_ground_truth: Option<Vec<Vec<usize>>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl GistSubset {
    /// Compute brute-force ground truth for Recall@k calculation using L2 distance.
    ///
    /// Returns top-k indices for each query using exact search.
    /// If pre-computed ground truth exists (full dataset), uses that instead.
    pub fn compute_ground_truth_topk(&self, k: usize) -> Vec<Vec<usize>> {
        // Use pre-computed if available and sufficient
        if let Some(ref gt) = self.precomputed_ground_truth {
            if gt.first().map(|v| v.len()).unwrap_or(0) >= k {
                println!("Using pre-computed ground truth (k={})...", k);
                return gt.iter().map(|v| v[..k].to_vec()).collect();
            }
        }

        // Otherwise compute brute-force
        self.compute_ground_truth_topk_with_distance(k, Distance::L2)
    }

    /// Compute brute-force ground truth with specified distance metric.
    pub fn compute_ground_truth_topk_with_distance(
        &self,
        k: usize,
        distance: Distance,
    ) -> Vec<Vec<usize>> {
        println!(
            "Computing brute-force ground truth (k={}, {:?})...",
            k, distance
        );

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
// Helper Functions
// ============================================================================

/// Normalize a vector to unit length.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Normalize all vectors in a collection.
pub fn normalize_all(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vectors.iter().map(|v| normalize(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, v);
    }

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert_eq!(config.dim, LAION_EMBEDDING_DIM);
        assert_eq!(config.max_vectors, DEFAULT_MAX_VECTORS);
        assert!(config.verbose);
    }

    #[test]
    fn test_dataset_config_builder() {
        let config = DatasetConfig::default()
            .with_dim(768)
            .with_max_vectors(1000)
            .with_verbose(false);
        assert_eq!(config.dim, 768);
        assert_eq!(config.max_vectors, 1000);
        assert!(!config.verbose);
    }
}
