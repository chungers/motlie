use anyhow::{bail, Context, Result};
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, EmbeddingModel, EmbeddingRequest, QuantizationScheme,
    StartOptions,
};
use motlie_models::{
    default_artifact_root, download_bundle_artifacts, quantization_label_isq, BundleDescriptor,
    Catalog, CuratedBundle,
};
use std::time::Instant;

use crate::support;

const SIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const SIMILAR_B: &str = "An orange kitten is napping on a sofa.";
const DISSIMILAR_A: &str = "A small orange cat is sleeping on the couch.";
const DISSIMILAR_B: &str = "Quarterly revenue increased by twelve percent year over year.";
const DEFAULT_EMBEDDING_SELECTOR: &str = motlie_models::embeddings::GOOGLE_GEMMA_300M_SELECTOR;

#[derive(Debug)]
struct ExampleArgs {
    selection: Option<Selection>,
    download_artifacts: bool,
    precision: Option<String>,
    input: String,
}

#[derive(Debug)]
enum Selection {
    Bundle(String),
    Selector(String),
    LegacyEmbeddingSelector(String),
}

struct ResolvedSelection {
    selector_label: String,
    resolution_path: &'static str,
    bundle_id: BundleId,
    descriptor: BundleDescriptor,
    bundle: CuratedBundle,
}

pub async fn run(target_name: &'static str) -> Result<()> {
    let args = parse_args(target_name)?;
    let artifact_root = default_artifact_root();
    let catalog = Catalog::with_defaults();
    let resolved = resolve_selection(args.selection, &catalog)?;
    let quantization = parse_quantization(args.precision.as_deref(), &resolved.bundle_id)?;

    println!("example-target: {}", target_name);
    println!("catalog-entry-count: {}", catalog.len());
    println!(
        "available-embedding-selectors: {}",
        available_embedding_selectors().join(", ")
    );
    println!("default-embedding-selector: {DEFAULT_EMBEDDING_SELECTOR}");
    println!("bundle-selector: {}", resolved.selector_label);
    println!("resolution-path: {}", resolved.resolution_path);
    println!("bundle-id: {}", resolved.bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    support::print_process_snapshot("process-before-start", &support::current_process_snapshot());
    println!("quantization: {}", quantization_label_isq(quantization));

    if args.download_artifacts {
        let summary = download_bundle_artifacts(&catalog, &resolved.bundle_id, &artifact_root)
            .with_context(|| {
                format!(
                    "failed to download curated artifacts for `{}`",
                    resolved.bundle_id
                )
            })?;
        println!("downloaded-files: {}", summary.downloaded.len());
    } else {
        println!("downloaded-files: skipped (using existing local artifacts only)");
    }

    println!("display-name: {}", resolved.descriptor.display_name);
    println!("family: {:?}", resolved.descriptor.family);
    println!("backend: {:?}", resolved.descriptor.backend);
    println!("capabilities:");
    for capability in resolved.descriptor.capability_descriptors() {
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
    let handle = resolved
        .bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            quantization_scheme: quantization,
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
            inputs: vec![args.input.clone()],
        })
        .await
        .context("embedding generation should succeed")?;
    let custom_latency = custom_started_at.elapsed();

    let vector = custom_response
        .vectors
        .into_iter()
        .next()
        .context("expected exactly one embedding vector")?;

    println!("input: {}", args.input);
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

fn parse_args(target_name: &'static str) -> Result<ExampleArgs> {
    let mut download_artifacts = false;
    let mut selection = None;
    let mut precision = None;
    let mut input_parts = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if arg == "--bundle" {
            selection = set_selection(
                selection,
                Selection::Bundle(next_value(&mut args, "--bundle")?),
            )?;
        } else if let Some(bundle) = arg.strip_prefix("--bundle=") {
            selection = set_selection(selection, Selection::Bundle(bundle.to_owned()))?;
        } else if arg == "--selector" {
            selection = set_selection(
                selection,
                Selection::Selector(next_value(&mut args, "--selector")?),
            )?;
        } else if let Some(selector) = arg.strip_prefix("--selector=") {
            selection = set_selection(selection, Selection::Selector(selector.to_owned()))?;
        } else if arg == "--embedding" {
            selection = set_selection(
                selection,
                Selection::LegacyEmbeddingSelector(next_value(&mut args, "--embedding")?),
            )?;
        } else if let Some(selector) = arg.strip_prefix("--embedding=") {
            selection = set_selection(
                selection,
                Selection::LegacyEmbeddingSelector(selector.to_owned()),
            )?;
        } else if arg == "--precision" {
            precision = Some(next_value(&mut args, "--precision")?);
        } else if let Some(value) = arg.strip_prefix("--precision=") {
            precision = Some(value.to_owned());
        } else if arg == "--help" || arg == "-h" {
            println!("{}", usage(target_name));
            std::process::exit(0);
        } else if arg.starts_with("--") {
            bail!("unknown option `{arg}`\n{}", usage(target_name));
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    if input.trim().is_empty() {
        bail!("{}", usage(target_name));
    }

    Ok(ExampleArgs {
        selection,
        download_artifacts,
        precision,
        input,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn set_selection(current: Option<Selection>, next: Selection) -> Result<Option<Selection>> {
    if current.is_some() {
        bail!("choose only one of --bundle, --selector, or --embedding");
    }
    Ok(Some(next))
}

fn parse_quantization(
    precision: Option<&str>,
    bundle_id: &BundleId,
) -> Result<Option<QuantizationScheme>> {
    match bundle_id.as_str() {
        "embeddinggemma_300m" => match precision {
            None | Some("default") | Some("fp32") | Some("f32") => {
                Ok(Some(QuantizationScheme::Fp32))
            }
            Some("q4") | Some("q8") => bail!(
                "precision `{}` is not supported by embeddinggemma_300m; use fp32",
                precision.unwrap()
            ),
            Some(other) => bail!("unknown precision `{other}` - use fp32"),
        },
        "qwen3_embedding_06b" => match precision {
            None | Some("default") | Some("bf16") => Ok(Some(QuantizationScheme::Bf16)),
            Some("q8") => Ok(Some(QuantizationScheme::IsqQ8)),
            Some("q4") => {
                bail!("precision `q4` is not supported by qwen3_embedding_06b; use bf16 or q8")
            }
            Some(other) => bail!("unknown precision `{other}` - use bf16 or q8"),
        },
        other => bail!("no embedding precision mapping registered for bundle `{other}`"),
    }
}

fn resolve_selection(selection: Option<Selection>, catalog: &Catalog) -> Result<ResolvedSelection> {
    match selection {
        Some(Selection::Bundle(raw_id)) => {
            let bundle_id = BundleId::new(raw_id);
            let descriptor = catalog
                .bundle(&bundle_id)
                .with_context(|| format!("unknown compiled bundle `{bundle_id}`"))?
                .clone();
            let bundle = catalog
                .instantiate(&bundle_id)
                .with_context(|| format!("bundle `{bundle_id}` is not available in this build"))?;
            let selector_label = embedding_selector_for_bundle(&bundle_id)
                .map(|selector| format!("embedding:{selector}"))
                .unwrap_or_else(|| format!("bundle:{}", bundle_id.as_str()));

            Ok(ResolvedSelection {
                selector_label,
                resolution_path: "bundle",
                bundle_id,
                descriptor,
                bundle,
            })
        }
        Some(Selection::Selector(raw_selector)) => resolve_selector(&raw_selector, "selector"),
        Some(Selection::LegacyEmbeddingSelector(raw_selector)) => {
            resolve_selector(&raw_selector, "legacy-embedding")
        }
        None => resolve_selector(DEFAULT_EMBEDDING_SELECTOR, "default"),
    }
}

fn resolve_selector(
    raw_selector: &str,
    resolution_path: &'static str,
) -> Result<ResolvedSelection> {
    let selector = raw_selector
        .strip_prefix("embedding:")
        .unwrap_or(raw_selector);
    let model: motlie_models::embeddings::EmbeddingModels = selector
        .parse()
        .with_context(|| format!("failed to parse embedding selector `{raw_selector}`"))?;
    let bundle_id = model.bundle_id();

    Ok(ResolvedSelection {
        selector_label: format!("embedding:{selector}"),
        resolution_path,
        bundle_id,
        descriptor: model.descriptor(),
        bundle: model.bundle(),
    })
}

fn embedding_selector_for_bundle(bundle_id: &BundleId) -> Option<&'static str> {
    match bundle_id.as_str() {
        "embeddinggemma_300m" => Some(motlie_models::embeddings::GOOGLE_GEMMA_300M_SELECTOR),
        "qwen3_embedding_06b" => Some(motlie_models::embeddings::QWEN3_EMBEDDING_06B_SELECTOR),
        _ => None,
    }
}

fn available_embedding_selectors() -> Vec<&'static str> {
    vec![
        motlie_models::embeddings::GOOGLE_GEMMA_300M_SELECTOR,
        motlie_models::embeddings::QWEN3_EMBEDDING_06B_SELECTOR,
    ]
}

async fn run_pair_demo<E: EmbeddingModel + ?Sized>(
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

fn usage(target_name: &str) -> String {
    format!(
        "usage: cargo run -p motlie-models --no-default-features --features 'model-google-gemma-300m model-qwen3-embedding-06b' --example {} -- [--bundle embeddinggemma_300m|qwen3_embedding_06b | --selector embedding:google/embeddinggemma_300m|embedding:qwen/qwen3_embedding_06b | --embedding=google/embeddinggemma_300m|qwen/qwen3_embedding_06b] [--download-artifacts] [--precision=fp32|bf16|q8] <text to embed>",
        target_name
    )
}
