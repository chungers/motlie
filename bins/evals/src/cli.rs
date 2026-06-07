use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::metrics::MetricsSampler;
use crate::platform::PlatformCollector;
use crate::report::OutputSink;
use crate::runner::asr::AsrRunner;
use crate::runner::chat::ChatRunner;
use crate::runner::embeddings::EmbeddingSimilarityRunner;
use crate::runner::perf::PerfRunner;
use crate::runner::tts::TtsRunner;
use crate::runner::{BundleSelection, ProfileSelection, RunContext, RuntimeFlags, ScenarioRunner};
use crate::scenario::{self, CapabilityName};

pub async fn run(args: impl IntoIterator<Item = String>) -> Result<()> {
    let command_line = args.into_iter().collect::<Vec<_>>();
    let args = command_line.iter().skip(1).cloned().collect::<Vec<_>>();

    match args.as_slice() {
        [] => {
            print_usage();
            Ok(())
        }
        [single] if single == "--help" || single == "-h" => {
            print_usage();
            Ok(())
        }
        [command, subject] if command == "list" && subject == "scenarios" => {
            list_scenarios(default_eval_root())
        }
        [command, subject, flag, root]
            if command == "list" && subject == "scenarios" && flag == "--root" =>
        {
            list_scenarios(PathBuf::from(root))
        }
        [command, subject] if command == "list" && subject == "bundles" => list_bundles(),
        [command, rest @ ..] if command == "run" => run_scenario(command_line, rest).await,
        [command, ..] if command == "matrix" || command == "report" => {
            bail!("`evals {command}` is planned after the embeddings runner pattern is reviewed")
        }
        _ => {
            print_usage();
            bail!("unknown evals command")
        }
    }
}

async fn run_scenario(command_line: Vec<String>, args: &[String]) -> Result<()> {
    let options = RunOptions::parse(args)?;
    let scenario = scenario::load_scenario(&options.eval_root, &options.scenario)
        .with_context(|| format!("failed to load scenario `{}`", options.scenario))?;
    let output_sink = options
        .jsonl
        .map(OutputSink::JsonlFile)
        .unwrap_or(OutputSink::Stdout);
    let context = RunContext {
        scenario,
        bundle_selection: BundleSelection {
            bundle_id: options.bundle,
            selector: options.selector,
        },
        profile: ProfileSelection {
            name: options.profile,
        },
        artifact_root: options.artifact_root,
        runtime_flags: RuntimeFlags {
            command_line,
            download_artifacts: options.download_artifacts,
            precision: options.precision,
            quiet_backend_logs: options.quiet_backend_logs,
        },
        platform_collector: PlatformCollector::new(options.profile_for_collector),
        metrics_sampler: MetricsSampler::new(),
        output_sink,
    };

    match context.scenario.capability() {
        CapabilityName::Embeddings => {
            EmbeddingSimilarityRunner.run(context).await?;
        }
        CapabilityName::Chat => {
            ChatRunner.run(context).await?;
        }
        CapabilityName::Perf => {
            PerfRunner.run(context).await?;
        }
        CapabilityName::Asr => {
            AsrRunner.run(context).await?;
        }
        CapabilityName::Tts => {
            TtsRunner.run(context).await?;
        }
    }
    Ok(())
}

fn list_bundles() -> Result<()> {
    let catalog = motlie_models::Catalog::with_defaults();
    for descriptor in catalog.bundles() {
        println!(
            "{}\t{:?}\t{:?}",
            descriptor.id.as_str(),
            descriptor.backend,
            descriptor
                .capabilities
                .descriptors()
                .iter()
                .map(|capability| format!("{:?}", capability.kind))
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn list_scenarios(root: PathBuf) -> Result<()> {
    let scenarios = scenario::list_scenarios(&root)
        .with_context(|| format!("failed to list scenarios under `{}`", root.display()))?;
    for scenario in scenarios {
        println!("{}", scenario.id);
    }
    Ok(())
}

#[derive(Debug)]
struct RunOptions {
    bundle: String,
    selector: Option<String>,
    scenario: String,
    profile: String,
    profile_for_collector: String,
    eval_root: PathBuf,
    artifact_root: PathBuf,
    jsonl: Option<PathBuf>,
    download_artifacts: bool,
    precision: Option<String>,
    quiet_backend_logs: bool,
}

impl RunOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut bundle = None;
        let mut selector = None;
        let mut scenario = None;
        let mut profile = None;
        let mut eval_root = default_eval_root();
        let mut artifact_root = motlie_models::default_artifact_root();
        let mut jsonl = None;
        let mut download_artifacts = false;
        let mut precision = None;
        let mut quiet_backend_logs = false;

        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--bundle" => {
                    bundle = Some(take_value(args, &mut index, "--bundle")?);
                }
                "--selector" => {
                    selector = Some(take_value(args, &mut index, "--selector")?);
                }
                "--scenario" => {
                    scenario = Some(take_value(args, &mut index, "--scenario")?);
                }
                "--profile" => {
                    profile = Some(take_value(args, &mut index, "--profile")?);
                }
                "--root" => {
                    eval_root = PathBuf::from(take_value(args, &mut index, "--root")?);
                }
                "--artifact-root" => {
                    artifact_root = PathBuf::from(take_value(args, &mut index, "--artifact-root")?);
                }
                "--jsonl" => {
                    jsonl = Some(PathBuf::from(take_value(args, &mut index, "--jsonl")?));
                }
                "--precision" => {
                    precision = Some(take_value(args, &mut index, "--precision")?);
                }
                "--download-artifacts" => {
                    download_artifacts = true;
                }
                "--quiet-backend-logs" => {
                    quiet_backend_logs = true;
                }
                other => bail!("unknown evals run option `{other}`"),
            }
            index += 1;
        }

        let profile = profile.unwrap_or_else(default_profile);
        Ok(Self {
            bundle: bundle.context("evals run requires --bundle <bundle_id>")?,
            selector,
            scenario: scenario.context("evals run requires --scenario <scenario_id>")?,
            profile_for_collector: profile.clone(),
            profile,
            eval_root,
            artifact_root,
            jsonl,
            download_artifacts,
            precision,
            quiet_backend_logs,
        })
    }
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .with_context(|| format!("{flag} requires a value"))
}

fn default_eval_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("evals")
}

fn default_profile() -> String {
    match std::env::consts::ARCH {
        "aarch64" if std::env::consts::OS == "macos" => "apple-metal".to_owned(),
        "aarch64" => "local-cpu-aarch64".to_owned(),
        "x86_64" => "local-cpu-x86_64".to_owned(),
        other => format!("local-cpu-{other}"),
    }
}

fn print_usage() {
    println!("usage:");
    println!("  evals list scenarios [--root PATH]");
    println!("  evals list bundles");
    println!("  evals run --bundle <bundle_id> --scenario <scenario_id> [--profile NAME] [--artifact-root PATH] [--jsonl PATH]");
    println!("  evals matrix");
    println!("  evals report --input <jsonl> --format markdown");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_eval_root_points_to_repo_evals_dir() {
        let root = default_eval_root();
        assert!(root.ends_with("evals"));
    }

    #[tokio::test]
    async fn help_returns_ok() {
        run(["evals".to_owned(), "--help".to_owned()])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn unknown_command_returns_error() {
        let error = run(["evals".to_owned(), "nope".to_owned()])
            .await
            .expect_err("unknown command should fail");
        assert!(error.to_string().contains("unknown evals command"));
    }

    #[test]
    fn parses_run_options() {
        let options = RunOptions::parse(&[
            "--bundle".to_owned(),
            "embeddinggemma_300m".to_owned(),
            "--scenario".to_owned(),
            "embeddings_similarity".to_owned(),
            "--profile".to_owned(),
            "local-cpu-x86_64".to_owned(),
        ])
        .unwrap();

        assert_eq!(options.bundle, "embeddinggemma_300m");
        assert_eq!(options.scenario, "embeddings_similarity");
        assert_eq!(options.profile, "local-cpu-x86_64");
    }
}
