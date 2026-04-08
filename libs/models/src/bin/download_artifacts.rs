use anyhow::{Context, Result};
use motlie_model::BundleId;
use motlie_models::{default_artifact_root, download_bundle_artifacts, Catalog};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let artifact_root = default_artifact_root();
    let catalog = Catalog::with_defaults();

    if args.is_empty() {
        args = catalog
            .bundles()
            .map(|bundle| bundle.id.as_str().to_owned())
            .collect();
    }

    for raw_id in args {
        let bundle_id = BundleId::new(raw_id);
        let summary = download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
            .with_context(|| format!("failed to download artifacts for `{bundle_id}`"))?;

        println!(
            "bundle={} files={} root={}",
            summary.bundle_id,
            summary.downloaded.len(),
            artifact_root.display()
        );
        for path in summary.downloaded {
            println!("  {}", path.display());
        }
    }

    Ok(())
}
