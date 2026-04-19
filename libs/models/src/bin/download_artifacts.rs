use motlie_model::BundleId;
use motlie_models::{
    ArtifactDownloadOptions, Catalog, default_artifact_root, download_bundle_artifacts_with_options,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).peekable();
    let artifact_root = default_artifact_root();
    let catalog = Catalog::with_defaults();
    let mut bundle_args = Vec::new();
    let mut download_options = ArtifactDownloadOptions::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--hf-token-env" => {
                let var_name = args
                    .next()
                    .ok_or_else(|| "expected env var name after `--hf-token-env`".to_string())?;
                let token = std::env::var(&var_name).map_err(|err| {
                    format!("failed to read Hugging Face token from env var `{var_name}`: {err}")
                })?;
                download_options.hf_token = Some(token);
            }
            "--hf-token-file" => {
                let path = args
                    .next()
                    .ok_or_else(|| "expected file path after `--hf-token-file`".to_string())?;
                let token = std::fs::read_to_string(&path).map_err(|err| {
                    format!("failed to read Hugging Face token file `{path}`: {err}")
                })?;
                download_options.hf_token = Some(token.trim().to_string());
            }
            _ => bundle_args.push(arg),
        }
    }

    if bundle_args.is_empty() {
        bundle_args = catalog
            .bundles()
            .map(|bundle| bundle.id.as_str().to_owned())
            .collect();
    }

    for raw_id in bundle_args {
        let bundle_id = BundleId::new(raw_id);
        let summary = download_bundle_artifacts_with_options(
            &catalog,
            &bundle_id,
            &artifact_root,
            &download_options,
        )
        .map_err(|err| err.to_string())?;

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
