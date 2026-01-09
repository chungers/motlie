//! LAION-400M CLIP embedding loader
//!
//! Downloads and loads pre-computed CLIP ViT-B-32 embeddings from LAION-400M.
//! - Image embeddings (512D): used as database vectors
//! - Text embeddings (512D): used as query vectors

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// LAION-400M embedding URLs
const IMG_EMB_URL: &str = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_0.npy";
const TEXT_EMB_URL: &str = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_0.npy";

/// Embedding dimensions (CLIP ViT-B-32)
pub const EMBEDDING_DIM: usize = 512;

/// Maximum vectors to load (first file has 1M, we only need 200K)
pub const MAX_VECTORS: usize = 200_000;

/// Download LAION embeddings if not already present
pub fn download_laion_embeddings(data_dir: &Path) -> Result<()> {
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

/// Download a file from URL to local path
fn download_file(url: &str, path: &Path) -> Result<()> {
    use std::process::Command;

    // Use curl or wget for large file downloads
    let status = Command::new("curl")
        .args(["-L", "-o", path.to_str().unwrap(), url])
        .status()
        .context("Failed to run curl")?;

    if !status.success() {
        anyhow::bail!("Download failed with status: {}", status);
    }

    Ok(())
}

/// Load embeddings from NPY file
///
/// NPY format for float32 arrays:
/// - Magic: \x93NUMPY
/// - Version: 1 byte major, 1 byte minor
/// - Header length: 2 bytes (v1) or 4 bytes (v2+)
/// - Header: Python dict string describing shape/dtype
/// - Data: raw binary float32 values
pub fn load_npy_embeddings(path: &Path, max_count: usize) -> Result<Vec<Vec<f32>>> {
    println!("Loading embeddings from {:?}...", path);

    let file = File::open(path).context("Failed to open NPY file")?;
    let mut reader = BufReader::new(file);

    // Read and parse NPY header
    let (shape, _dtype) = parse_npy_header(&mut reader)?;

    let num_vectors = shape[0].min(max_count);
    let dim = shape[1];

    if dim != EMBEDDING_DIM {
        anyhow::bail!("Expected {}D embeddings, got {}D", EMBEDDING_DIM, dim);
    }

    println!("  Shape: {:?}, loading first {} vectors", shape, num_vectors);

    // Read embeddings
    let mut embeddings = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let mut vec = vec![0.0f32; dim];
        for j in 0..dim {
            vec[j] = reader.read_f32::<LittleEndian>()?;
        }
        embeddings.push(vec);

        if (i + 1) % 50000 == 0 {
            println!("  Loaded {}/{} vectors", i + 1, num_vectors);
        }
    }

    println!("  Loaded {} vectors ({}D)", embeddings.len(), dim);
    Ok(embeddings)
}

/// Parse NPY file header to get shape and dtype
fn parse_npy_header<R: Read>(reader: &mut R) -> Result<(Vec<usize>, String)> {
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

    println!("  NPY v{}.{}, header: {}", major, minor, header.trim());

    // Parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (1000000, 512), }"
    let shape = parse_shape_from_header(&header)?;
    let dtype = parse_dtype_from_header(&header)?;

    Ok((shape, dtype))
}

/// Extract shape tuple from NPY header string
fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (N, M)
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

/// Extract dtype from NPY header string
fn parse_dtype_from_header(header: &str) -> Result<String> {
    // Find 'descr': '<f4' or similar
    let descr_start = header.find("'descr':").context("No descr in header")?;
    let quote_start = header[descr_start + 8..].find('\'').context("No dtype quote")?;
    let quote_end = header[descr_start + 8 + quote_start + 1..]
        .find('\'')
        .context("No dtype end")?;

    let dtype = &header[descr_start + 9 + quote_start..descr_start + 9 + quote_start + quote_end];
    Ok(dtype.to_string())
}

/// LAION dataset with image (database) and text (query) embeddings
pub struct LaionDataset {
    pub image_embeddings: Vec<Vec<f32>>,
    pub text_embeddings: Vec<Vec<f32>>,
}

impl LaionDataset {
    /// Load LAION dataset from data directory
    pub fn load(data_dir: &Path, max_vectors: usize) -> Result<Self> {
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

        let image_embeddings = load_npy_embeddings(&img_path, max_vectors)?;
        let text_embeddings = load_npy_embeddings(&text_path, max_vectors)?;

        Ok(Self {
            image_embeddings,
            text_embeddings,
        })
    }

    /// Get a subset of the dataset
    pub fn subset(&self, num_vectors: usize, num_queries: usize) -> LaionSubset {
        let db_vectors: Vec<Vec<f32>> = self.image_embeddings[..num_vectors].to_vec();

        // Use text embeddings as queries - each text embedding has its
        // corresponding image embedding at the same index as ground truth
        let query_indices: Vec<usize> = (0..num_vectors)
            .step_by(num_vectors / num_queries)
            .take(num_queries)
            .collect();

        let queries: Vec<Vec<f32>> = query_indices
            .iter()
            .map(|&i| self.text_embeddings[i].clone())
            .collect();

        // Ground truth: each query's matching image is at the same index
        let ground_truth: Vec<usize> = query_indices.clone();

        LaionSubset {
            db_vectors,
            queries,
            ground_truth,
        }
    }
}

/// A subset of LAION data for benchmarking
pub struct LaionSubset {
    pub db_vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    /// Index of the ground truth image for each query
    pub ground_truth: Vec<usize>,
}

impl LaionSubset {
    /// Compute brute-force ground truth for Recall@k calculation
    /// Returns top-k indices for each query using exact search
    pub fn compute_ground_truth_topk(&self, k: usize) -> Vec<Vec<usize>> {
        println!("Computing brute-force ground truth (k={})...", k);

        let mut results = Vec::with_capacity(self.queries.len());

        for (qi, query) in self.queries.iter().enumerate() {
            // Compute distances to all database vectors
            let mut distances: Vec<(usize, f32)> = self
                .db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance(query, v)))
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
}

/// Compute cosine distance between two vectors
/// Returns 1 - cosine_similarity (so 0 = identical, 2 = opposite)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Normalize a vector to unit length
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Normalize all vectors in a collection
pub fn normalize_all(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vectors.iter().map(|v| normalize(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &d) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }
}
