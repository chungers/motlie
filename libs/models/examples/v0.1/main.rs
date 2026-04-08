use anyhow::{bail, Context, Result};
use motlie_model::{ArtifactPolicy, BundleId, EmbeddingRequest, StartOptions};
use motlie_models::{default_artifact_root, download_bundle_artifacts, Catalog};

const BUNDLE_ID: &str = "embeddinggemma_300m";

#[tokio::main]
async fn main() -> Result<()> {
    let input = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if input.trim().is_empty() {
        bail!("usage: cargo run -p motlie-models --example models_v0_1 -- <text to embed>");
    }

    let catalog = Catalog::with_defaults();
    let bundle_id = BundleId::new(BUNDLE_ID);
    let artifact_root = default_artifact_root();

    println!("bundle: {BUNDLE_ID}");
    println!("artifact-root: {}", artifact_root.display());

    let summary = download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
        .with_context(|| format!("failed to download curated artifacts for `{bundle_id}`"))?;
    println!("downloaded-files: {}", summary.downloaded.len());

    let descriptor = catalog
        .bundle(&bundle_id)
        .context("bundle descriptor should exist in the default catalog")?;
    println!("display-name: {}", descriptor.display_name);
    println!("family: {:?}", descriptor.family);
    println!("backend: {:?}", descriptor.backend);
    println!("packaging: {:?}", descriptor.packaging);
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

    let bundle = catalog
        .instantiate(&bundle_id)
        .context("bundle should be constructible from the default catalog")?;
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            ..Default::default()
        })
        .await
        .context("bundle startup should succeed after curated artifact download")?;
    let embeddings = handle
        .embeddings()
        .context("embeddinggemma bundle should expose embeddings")?;
    let response = embeddings
        .embed(EmbeddingRequest {
            inputs: vec![input.clone()],
        })
        .await
        .context("embedding generation should succeed")?;

    let vector = response
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

    handle
        .shutdown()
        .await
        .context("bundle shutdown should succeed")?;

    Ok(())
}
