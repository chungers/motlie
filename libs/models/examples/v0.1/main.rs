use anyhow::{bail, Context, Result};
use motlie_model::{ArtifactPolicy, EmbeddingRequest, StartOptions};
use motlie_models::{
    default_artifact_root, download_bundle_artifacts, embeddings::EmbeddingModels, ModelSelector,
};
use std::time::Instant;

const SIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const SIMILAR_B: &str = "An orange kitten is napping on a sofa.";
const DISSIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const DISSIMILAR_B: &str = "Quarterly revenue increased by twelve percent year over year.";

#[tokio::main]
async fn main() -> Result<()> {
    let mut download_artifacts = false;
    let mut embedding_selector = None;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if let Some(selector) = arg.strip_prefix("--embedding=") {
            embedding_selector = Some(selector.to_owned());
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    if input.trim().is_empty() {
        bail!(
            "usage: cargo run -p motlie-models --example models_v0_1 -- [--download-artifacts] [--embedding=google/embeddinggemma_300m] <text to embed>"
        );
    }

    let (selector_label, bundle_id, descriptor, bundle, path_kind) = if let Some(selector) =
        embedding_selector
    {
        let model_selector: ModelSelector = format!("embedding:{selector}")
            .parse()
            .with_context(|| format!("failed to parse model selector `embedding:{selector}`"))?;
        (
            model_selector.to_string(),
            model_selector.bundle_id(),
            model_selector.descriptor(),
            model_selector.bundle(),
            "selector",
        )
    } else {
        let model = EmbeddingModels::GoogleGemma300m;
        (
            model.to_string(),
            model.bundle_id(),
            model.descriptor(),
            model.bundle(),
            "direct-enum",
        )
    };

    let artifact_root = default_artifact_root();

    println!("bundle-selector: {selector_label}");
    println!("resolution-path: {path_kind}");
    println!("bundle-id: {}", bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());

    if download_artifacts {
        let catalog = motlie_models::Catalog::with_defaults();
        let summary = download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
            .with_context(|| format!("failed to download curated artifacts for `{bundle_id}`"))?;
        println!("downloaded-files: {}", summary.downloaded.len());
    } else {
        println!("downloaded-files: skipped (using existing local artifacts only)");
    }

    println!("display-name: {}", descriptor.display_name);
    println!("family: {:?}", descriptor.family);
    println!("backend: {:?}", descriptor.backend);
    println!("capabilities:");
    for capability in descriptor.capability_descriptors() {
        println!(
            "  - kind={:?} input={:?} output={:?} interaction={:?} summary={}",
            capability.kind,
            capability.inputs,
            capability.outputs,
            capability.interaction,
            capability.summary
        );
    }

    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            ..Default::default()
        })
        .await
        .context("bundle startup should succeed from local artifacts")?;
    let embeddings = handle
        .embeddings()
        .context("embeddinggemma bundle should expose embeddings")?;
    let custom_started_at = Instant::now();
    let custom_response = embeddings
        .embed(EmbeddingRequest {
            inputs: vec![input.clone()],
        })
        .await
        .context("embedding generation should succeed")?;
    let custom_latency = custom_started_at.elapsed();

    let vector = custom_response
        .vectors
        .into_iter()
        .next()
        .context("expected exactly one embedding vector")?;

    println!("input: {input}");
    println!("embedding-dim: {}", vector.len());
    println!(
        "embedding-head: {:?}",
        vector.iter().take(8).copied().collect::<Vec<_>>()
    );
    println!(
        "embedding-latency-ms: {:.2}",
        custom_latency.as_secs_f64() * 1000.0
    );

    run_pair_demo(
        embeddings,
        "similar",
        SIMILAR_A,
        SIMILAR_B,
        "expected to have a relatively high cosine similarity",
    )
    .await?;
    run_pair_demo(
        embeddings,
        "dissimilar",
        DISSIMILAR_A,
        DISSIMILAR_B,
        "expected to have a noticeably lower cosine similarity",
    )
    .await?;

    handle
        .shutdown()
        .await
        .context("bundle shutdown should succeed")?;

    Ok(())
}

async fn run_pair_demo(
    embeddings: &dyn motlie_model::EmbeddingModel,
    label: &str,
    first: &str,
    second: &str,
    expectation: &str,
) -> Result<()> {
    let started_at = Instant::now();
    let response = embeddings
        .embed(EmbeddingRequest {
            inputs: vec![first.to_owned(), second.to_owned()],
        })
        .await
        .with_context(|| format!("embedding generation should succeed for the {label} pair"))?;
    let latency = started_at.elapsed();

    let mut vectors = response.vectors.into_iter();
    let first_vector = vectors
        .next()
        .with_context(|| format!("expected first vector for the {label} pair"))?;
    let second_vector = vectors
        .next()
        .with_context(|| format!("expected second vector for the {label} pair"))?;
    let cosine = cosine_similarity(&first_vector, &second_vector)
        .with_context(|| format!("cosine similarity should be computable for the {label} pair"))?;

    println!("{label}-pair:");
    println!("  text-a: {first}");
    println!("  text-b: {second}");
    println!(
        "  vector-a-head: {:?}",
        first_vector.iter().take(8).copied().collect::<Vec<_>>()
    );
    println!(
        "  vector-b-head: {:?}",
        second_vector.iter().take(8).copied().collect::<Vec<_>>()
    );
    println!("  cosine-similarity: {:.6}", cosine);
    println!("  latency-ms: {:.2}", latency.as_secs_f64() * 1000.0);
    println!("  note: {expectation}");

    Ok(())
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "cosine similarity requires vectors with the same dimension: {} != {}",
            left.len(),
            right.len()
        );
    }
    if left.is_empty() {
        bail!("cosine similarity requires non-empty vectors");
    }

    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        bail!("cosine similarity requires non-zero vector norms");
    }

    Ok(dot / (left_norm * right_norm))
}
