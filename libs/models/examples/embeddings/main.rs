use anyhow::{Context, Result, bail, ensure};
use motlie_model::{
    ArtifactPolicy, BundleHandle, EmbeddingModel, EmbeddingRequest, QuantizationBits, StartOptions,
};
use motlie_models::{ModelSelector, default_artifact_root, download_bundle_artifacts};
use std::time::Instant;

#[path = "../support.rs"]
mod support;

const SIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const SIMILAR_B: &str = "An orange kitten is napping on a sofa.";
const DISSIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const DISSIMILAR_B: &str = "Quarterly revenue increased by twelve percent year over year.";
const DEFAULT_EMBEDDING_SELECTOR: &str = motlie_models::embeddings::GOOGLE_GEMMA_300M_SELECTOR;

#[tokio::main]
async fn main() -> Result<()> {
    let mut download_artifacts = false;
    let mut embedding_selector = None;
    let mut precision = None;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if let Some(selector) = arg.strip_prefix("--embedding=") {
            embedding_selector = Some(selector.to_owned());
        } else if let Some(p) = arg.strip_prefix("--precision=") {
            precision = Some(p.to_owned());
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    let selector = embedding_selector.unwrap_or_else(|| DEFAULT_EMBEDDING_SELECTOR.to_owned());
    if input.trim().is_empty() {
        bail!(
            "usage: cargo run -p motlie-models --no-default-features --features 'model-google-gemma-300m model-qwen3-embedding-06b' --example embeddings -- [--embedding=google/embeddinggemma_300m|qwen/qwen3_embedding_06b] [--download-artifacts] [--precision=q4|q8|f32] <text to embed>"
        );
    }

    let quantization = match precision.as_deref() {
        Some("q4") => Some(QuantizationBits::Four),
        Some("q8") => Some(QuantizationBits::Eight),
        Some("f32") | None => None,
        Some(other) => bail!("unknown precision `{other}` — use q4, q8, or f32"),
    };

    let model_selector: ModelSelector = format!("embedding:{selector}")
        .parse()
        .with_context(|| format!("failed to parse model selector `embedding:{selector}`"))?;
    let selector_label = model_selector.to_string();
    let bundle_id = model_selector.bundle_id();
    let descriptor = model_selector.descriptor();
    let bundle = model_selector.bundle()?;

    let artifact_root = default_artifact_root();
    let catalog = motlie_models::Catalog::with_defaults();

    println!("catalog-entry-count: {}", catalog.len());
    println!(
        "available-embedding-selectors: {}",
        available_embedding_selectors().join(", ")
    );
    println!("default-embedding-selector: {DEFAULT_EMBEDDING_SELECTOR}");
    ensure!(
        catalog.len() == 2,
        "embeddings must be built with exactly the two curated embedding bundle features enabled"
    );

    println!("bundle-selector: {selector_label}");
    println!(
        "resolution-path: {}",
        if selector == DEFAULT_EMBEDDING_SELECTOR {
            "default-or-selector"
        } else {
            "selector"
        }
    );
    println!("bundle-id: {}", bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    support::print_process_snapshot("process-before-start", &support::current_process_snapshot());
    println!(
        "quantization: {}",
        match quantization {
            Some(QuantizationBits::Four) => "ISQ Q4",
            Some(QuantizationBits::Eight) => "ISQ Q8",
            None => "F32 (none)",
        }
    );

    if download_artifacts {
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

    println!("starting bundle...");
    let startup_sampler = support::StartupSampler::spawn("startup");
    let startup_at = Instant::now();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            quantization,
            ..Default::default()
        })
        .await
        .context("bundle startup should succeed from local artifacts")?;
    let startup_elapsed = startup_at.elapsed();
    let startup_stats = startup_sampler.finish().await;
    println!(
        "startup-latency-ms: {:.0} ({:.1}s)",
        startup_elapsed.as_secs_f64() * 1000.0,
        startup_elapsed.as_secs_f64()
    );
    support::print_startup_stats(&startup_stats);
    support::print_process_snapshot("process-after-start", &support::current_process_snapshot());
    support::print_model_metrics("model-metrics-after-start", handle.metric_snapshot());

    let embeddings = handle
        .embeddings()
        .context("embedding bundle should expose embeddings")?;
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
    support::print_process_snapshot(
        "process-after-custom-embed",
        &support::current_process_snapshot(),
    );
    support::print_model_metrics("model-metrics-after-custom-embed", handle.metric_snapshot());

    run_pair_demo(
        embeddings,
        "similar",
        SIMILAR_A,
        SIMILAR_B,
        "expected to have a relatively high cosine similarity",
    )
    .await?;
    support::print_model_metrics("model-metrics-after-similar-pair", handle.metric_snapshot());
    run_pair_demo(
        embeddings,
        "dissimilar",
        DISSIMILAR_A,
        DISSIMILAR_B,
        "expected to have a noticeably lower cosine similarity",
    )
    .await?;
    support::print_model_metrics(
        "model-metrics-after-dissimilar-pair",
        handle.metric_snapshot(),
    );

    handle
        .shutdown()
        .await
        .context("bundle shutdown should succeed")?;
    support::print_process_snapshot(
        "process-after-shutdown",
        &support::current_process_snapshot(),
    );

    Ok(())
}

fn available_embedding_selectors() -> Vec<&'static str> {
    vec![
        motlie_models::embeddings::GOOGLE_GEMMA_300M_SELECTOR,
        motlie_models::embeddings::QWEN3_EMBEDDING_06B_SELECTOR,
    ]
}

async fn run_pair_demo<E: motlie_model::EmbeddingModel + ?Sized>(
    embeddings: &E,
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
    support::print_process_snapshot(
        &format!("process-after-{label}-pair"),
        &support::current_process_snapshot(),
    );
    // Pair runs share the same long-lived handle; metrics are printed by the caller after each top-level step.

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
