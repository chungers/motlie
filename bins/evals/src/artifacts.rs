use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use motlie_models::artifact_manifest::{
    check_curated_artifacts, curated_artifact_entries, fetch_metadata_for_entries,
    render_preflight_report, render_provenance_markdown, sync_missing_curated_artifacts,
};

pub fn run_preflight(args: &[String]) -> Result<()> {
    let mut rewritten = Vec::with_capacity(args.len() + 1);
    rewritten.push("check".to_owned());
    rewritten.extend(args.iter().cloned());
    run_artifacts(&rewritten)
}

pub fn run_artifacts(args: &[String]) -> Result<()> {
    let (subcommand, rest) = args
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("evals artifacts requires a subcommand"))?;
    match subcommand.as_str() {
        "check" => run_check(rest),
        "sync" | "download" => run_sync(rest),
        "provenance" => run_provenance(rest),
        "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => bail!("unknown evals artifacts subcommand `{other}`"),
    }
}

fn run_check(args: &[String]) -> Result<()> {
    let options = ArtifactOptions::parse(args)?;
    let catalog = motlie_models::Catalog::with_defaults();
    let entries = curated_artifact_entries(&catalog).with_context(incomplete_catalog_context)?;
    let metadata = options.metadata(&entries);
    let report = check_curated_artifacts(&catalog, &options.artifact_root, &metadata)
        .with_context(|| {
            format!(
                "failed to inspect artifact cache `{}`",
                options.artifact_root.display()
            )
        })?;

    print!("{}", render_preflight_report(&report));
    ensure!(
        report.is_complete(),
        "artifact preflight failed: {}/{} bundles complete, {} missing requirement(s)",
        report.complete_bundle_count(),
        report.bundles.len(),
        report.missing_rule_count()
    );
    Ok(())
}

fn run_sync(args: &[String]) -> Result<()> {
    let options = ArtifactOptions::parse(args)?;
    let catalog = motlie_models::Catalog::with_defaults();
    curated_artifact_entries(&catalog).with_context(incomplete_catalog_context)?;
    let downloaded =
        sync_missing_curated_artifacts(&catalog, &options.artifact_root, options.hf_token())
            .with_context(|| {
                format!(
                    "failed to sync artifacts into `{}`",
                    options.artifact_root.display()
                )
            })?;

    if downloaded.is_empty() {
        println!("artifact sync: no missing bundles");
    } else {
        println!(
            "artifact sync: downloaded/verified {} bundle(s)",
            downloaded.len()
        );
        for summary in &downloaded {
            println!(
                "  {}\t{} file(s)",
                summary.bundle_id.as_str(),
                summary.downloaded.len()
            );
        }
    }

    let entries = curated_artifact_entries(&catalog).with_context(incomplete_catalog_context)?;
    let metadata = options.metadata(&entries);
    let report = check_curated_artifacts(&catalog, &options.artifact_root, &metadata)
        .with_context(|| {
            format!(
                "failed to inspect artifact cache `{}`",
                options.artifact_root.display()
            )
        })?;
    print!("{}", render_preflight_report(&report));
    ensure!(
        report.is_complete(),
        "artifact sync finished with missing artifacts: {} missing requirement(s)",
        report.missing_rule_count()
    );
    Ok(())
}

fn run_provenance(args: &[String]) -> Result<()> {
    let options = ArtifactOptions::parse(args)?;
    let catalog = motlie_models::Catalog::with_defaults();
    let entries = curated_artifact_entries(&catalog).with_context(incomplete_catalog_context)?;
    let rendered = render_provenance_markdown(&entries);

    match (&options.output, options.check) {
        (Some(path), true) => {
            let existing = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read `{}`", path.display()))?;
            ensure!(
                existing == rendered,
                "artifact provenance is stale; regenerate with `evals artifacts provenance --output {}`",
                path.display()
            );
            println!("artifact provenance is current: {}", path.display());
        }
        (Some(path), false) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create `{}`", parent.display()))?;
            }
            std::fs::write(path, rendered)
                .with_context(|| format!("failed to write `{}`", path.display()))?;
            println!("wrote artifact provenance: {}", path.display());
        }
        (None, true) => bail!("--check requires --output <path>"),
        (None, false) => print!("{rendered}"),
    }
    Ok(())
}

#[derive(Debug)]
struct ArtifactOptions {
    artifact_root: PathBuf,
    hf_token_env: String,
    offline: bool,
    output: Option<PathBuf>,
    check: bool,
}

impl ArtifactOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut artifact_root = motlie_models::default_artifact_root();
        let mut hf_token_env = "HF_TOKEN".to_owned();
        let mut offline = false;
        let mut output = None;
        let mut check = false;

        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--artifact-root" => {
                    artifact_root = PathBuf::from(take_value(args, &mut index, "--artifact-root")?);
                }
                "--hf-token-env" => {
                    hf_token_env = take_value(args, &mut index, "--hf-token-env")?;
                }
                "--offline" => {
                    offline = true;
                }
                "--output" => {
                    output = Some(PathBuf::from(take_value(args, &mut index, "--output")?));
                }
                "--check" => {
                    check = true;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => bail!("unknown evals artifacts option `{other}`"),
            }
            index += 1;
        }

        Ok(Self {
            artifact_root,
            hf_token_env,
            offline,
            output,
            check,
        })
    }

    fn hf_token(&self) -> Option<String> {
        env::var(&self.hf_token_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    fn metadata(
        &self,
        entries: &[motlie_models::artifact_manifest::ArtifactBundleEntry],
    ) -> BTreeMap<
        String,
        std::result::Result<motlie_models::artifact_manifest::HfRepoMetadata, String>,
    > {
        if self.offline {
            BTreeMap::new()
        } else {
            fetch_metadata_for_entries(entries, self.hf_token())
        }
    }
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .with_context(|| format!("{flag} requires a value"))
}

fn incomplete_catalog_context() -> String {
    "artifact preflight requires a descriptor-complete build; rerun with `--features all-curated`"
        .to_owned()
}

fn print_usage() {
    println!("usage:");
    println!("  evals preflight [--artifact-root PATH] [--hf-token-env NAME] [--offline]");
    println!("  evals artifacts check [--artifact-root PATH] [--hf-token-env NAME] [--offline]");
    println!("  evals artifacts sync [--artifact-root PATH] [--hf-token-env NAME] [--offline]");
    println!(
        "  evals artifacts provenance [--output PATH] [--check] [--hf-token-env NAME] [--offline]"
    );
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "all-curated")]
    use std::path::PathBuf;

    #[cfg(feature = "all-curated")]
    use motlie_models::artifact_manifest::{curated_artifact_entries, render_provenance_markdown};

    #[cfg(feature = "all-curated")]
    #[test]
    fn artifact_provenance_doc_matches_registry() {
        let catalog = motlie_models::Catalog::with_defaults();
        let entries =
            curated_artifact_entries(&catalog).expect("all curated descriptors should compile");
        let rendered = render_provenance_markdown(&entries);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .map(PathBuf::from)
            .expect("repo root should exist")
            .join("evals/artifacts/provenance.md");
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read `{}`: {err}", path.display()));

        assert_eq!(
            committed, rendered,
            "artifact provenance is stale; regenerate with `evals artifacts provenance --output evals/artifacts/provenance.md`"
        );
    }
}
