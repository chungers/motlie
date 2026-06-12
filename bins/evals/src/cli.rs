use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};

use crate::accelerator;
use crate::metrics::MetricsSampler;
use crate::platform::PlatformCollector;
use crate::report::{self, OutputSink};
use crate::result::{
    AcceleratorClass, ApplicabilityDecision, ChildBuildSection, CoverageSection, EvalDepth,
    TerminalOutcome,
};
use crate::runner::asr::AsrRunner;
use crate::runner::chat::ChatRunner;
use crate::runner::embeddings::EmbeddingSimilarityRunner;
use crate::runner::perf::PerfRunner;
use crate::runner::tts::TtsRunner;
use crate::runner::{BundleSelection, ProfileSelection, RunContext, RuntimeFlags, ScenarioRunner};
use crate::scenario::{self, AudioIterationOverrides, CapabilityName};

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
        [command, rest @ ..] if command == "matrix" => {
            crate::driver::run_matrix(command_line, rest).await
        }
        [command, rest @ ..] if command == "provision" => {
            crate::driver::run_provision(command_line, rest).await
        }
        [command, rest @ ..] if command == "report" => report::run_report(rest),
        _ => {
            print_usage();
            bail!("unknown evals command")
        }
    }
}

async fn run_scenario(command_line: Vec<String>, args: &[String]) -> Result<()> {
    let options = RunOptions::parse(args)?;
    let mut scenario = scenario::load_scenario(&options.eval_root, &options.scenario)
        .with_context(|| format!("failed to load scenario `{}`", options.scenario))?;
    scenario.apply_audio_iteration_overrides(options.audio_iteration_overrides);
    let output_sink = options
        .jsonl
        .clone()
        .map(OutputSink::JsonlFile)
        .unwrap_or(OutputSink::Stdout);
    let platform_collector = PlatformCollector::new(options.profile_for_collector.clone());
    let platform = platform_collector.collect();
    let accelerator = options
        .requested_accelerator
        .map(|requested| accelerator::resolve(requested, &platform, None, None))
        .or_else(|| {
            Some(accelerator::resolve_for_profile(
                &options.profile,
                &platform,
            ))
        });
    let coverage = options.coverage(&scenario, accelerator.as_ref(), &platform);
    let child_build = options.child_build();
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
            artifact_quantization: options.artifact_quantization,
            quiet_backend_logs: options.quiet_backend_logs,
            run_id: options.run_id,
        },
        coverage,
        accelerator,
        child_build,
        platform_collector,
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
        CapabilityName::ToolUse => {
            crate::runner::tool_use::ToolUseRunner.run(context).await?;
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
    run_id: Option<String>,
    snapshot_id: Option<String>,
    cell_id: Option<String>,
    depth: Option<EvalDepth>,
    checkpoint_format: Option<String>,
    artifact_quantization: Option<String>,
    model_family: Option<String>,
    backend: Option<String>,
    requested_accelerator: Option<AcceleratorClass>,
    child_build_log: Option<String>,
    child_build_status: Option<i32>,
    child_build_duration_ms: Option<u64>,
    audio_iteration_overrides: AudioIterationOverrides,
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
        let mut run_id = None;
        let mut snapshot_id = None;
        let mut cell_id = None;
        let mut depth = None;
        let mut checkpoint_format = None;
        let mut artifact_quantization = None;
        let mut model_family = None;
        let mut backend = None;
        let mut requested_accelerator = None;
        let mut child_build_log = None;
        let mut child_build_status = None;
        let mut child_build_duration_ms = None;
        let mut audio_iteration_overrides = AudioIterationOverrides::default();
        let mut cold = false;

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
                "--run-id" => {
                    run_id = Some(take_value(args, &mut index, "--run-id")?);
                }
                "--snapshot-id" => {
                    snapshot_id = Some(take_value(args, &mut index, "--snapshot-id")?);
                }
                "--cell-id" => {
                    cell_id = Some(take_value(args, &mut index, "--cell-id")?);
                }
                "--depth" => {
                    depth = Some(parse_depth(&take_value(args, &mut index, "--depth")?)?);
                }
                "--checkpoint-format" => {
                    checkpoint_format = Some(take_value(args, &mut index, "--checkpoint-format")?);
                }
                "--artifact-quantization" => {
                    artifact_quantization =
                        Some(take_value(args, &mut index, "--artifact-quantization")?);
                }
                "--model-family" => {
                    model_family = Some(take_value(args, &mut index, "--model-family")?);
                }
                "--backend" => {
                    backend = Some(take_value(args, &mut index, "--backend")?);
                }
                "--requested-accelerator" => {
                    requested_accelerator = Some(parse_accelerator(&take_value(
                        args,
                        &mut index,
                        "--requested-accelerator",
                    )?)?);
                }
                "--child-build-log" => {
                    child_build_log = Some(take_value(args, &mut index, "--child-build-log")?);
                }
                "--child-build-status" => {
                    child_build_status = Some(
                        take_value(args, &mut index, "--child-build-status")?
                            .parse::<i32>()
                            .context("--child-build-status must be an integer")?,
                    );
                }
                "--child-build-duration-ms" => {
                    child_build_duration_ms = Some(
                        take_value(args, &mut index, "--child-build-duration-ms")?
                            .parse::<u64>()
                            .context("--child-build-duration-ms must be an integer")?,
                    );
                }
                "--warmup-iterations" => {
                    ensure!(!cold, "--warmup-iterations cannot be combined with --cold");
                    audio_iteration_overrides.warmup_iterations =
                        Some(take_u64(args, &mut index, "--warmup-iterations")?);
                }
                "--iterations" => {
                    ensure!(!cold, "--iterations cannot be combined with --cold");
                    audio_iteration_overrides.iterations =
                        Some(take_u64(args, &mut index, "--iterations")?);
                }
                "--cold" => {
                    ensure!(
                        audio_iteration_overrides.is_empty(),
                        "--cold cannot be combined with --iterations or --warmup-iterations"
                    );
                    cold = true;
                    audio_iteration_overrides = AudioIterationOverrides::cold();
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
            run_id,
            snapshot_id,
            cell_id,
            depth,
            checkpoint_format,
            artifact_quantization,
            model_family,
            backend,
            requested_accelerator,
            child_build_log,
            child_build_status,
            child_build_duration_ms,
            audio_iteration_overrides,
        })
    }

    fn child_build(&self) -> Option<ChildBuildSection> {
        (self.child_build_log.is_some()
            || self.child_build_status.is_some()
            || self.child_build_duration_ms.is_some())
        .then(|| ChildBuildSection {
            command: Vec::new(),
            status: self.child_build_status,
            duration_ms: self.child_build_duration_ms,
            log_path: self.child_build_log.clone(),
        })
    }

    fn coverage(
        &self,
        scenario: &scenario::Scenario,
        accelerator: Option<&crate::result::AcceleratorSection>,
        platform: &crate::platform::PlatformSnapshot,
    ) -> Option<CoverageSection> {
        let snapshot_id = self.snapshot_id.clone()?;
        let host_id = platform
            .host_id
            .clone()
            .or_else(|| platform.hostname.clone())
            .unwrap_or_else(|| "unknown".to_owned());
        let host_slug = platform
            .host_slug
            .clone()
            .unwrap_or_else(|| crate::platform::sanitize_slug(&host_id));
        let requested = accelerator
            .map(|accelerator| accelerator.requested_class)
            .unwrap_or(AcceleratorClass::Any);
        let resolved = accelerator
            .map(|accelerator| accelerator.resolved_class)
            .unwrap_or(AcceleratorClass::Unavailable);
        let depth = self.depth.unwrap_or(scenario.depth);
        let checkpoint_format = self
            .checkpoint_format
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        let artifact_quantization = self
            .artifact_quantization
            .clone()
            .unwrap_or_else(|| "default".to_owned());
        let backend = self.backend.clone().unwrap_or_else(|| "unknown".to_owned());
        let mut grouping_keys = std::collections::BTreeMap::new();
        grouping_keys.insert("bundle".to_owned(), self.bundle.clone());
        grouping_keys.insert(
            "capability".to_owned(),
            scenario.capability().as_str().to_owned(),
        );
        grouping_keys.insert("depth".to_owned(), depth.as_str().to_owned());
        grouping_keys.insert("backend".to_owned(), backend.clone());
        grouping_keys.insert("checkpoint_format".to_owned(), checkpoint_format.clone());
        grouping_keys.insert("quantization".to_owned(), artifact_quantization.clone());
        grouping_keys.insert("profile".to_owned(), self.profile.clone());

        Some(CoverageSection {
            snapshot_id,
            cell_id: self
                .cell_id
                .clone()
                .unwrap_or_else(|| format!("{}__{}", self.bundle, scenario.id)),
            depth,
            capability: scenario.capability().as_str().to_owned(),
            scenario_id: scenario.id.clone(),
            bundle_id: self.bundle.clone(),
            model_family: self
                .model_family
                .clone()
                .unwrap_or_else(|| "unknown".to_owned()),
            checkpoint_format,
            quantization: artifact_quantization,
            backend,
            profile: self.profile.clone(),
            host_id,
            host_slug,
            arch: std::env::consts::ARCH.to_owned(),
            requested_accelerator: requested,
            resolved_accelerator: resolved,
            applicability: ApplicabilityDecision::Applicable,
            terminal_outcome: TerminalOutcome::Blocked,
            reason: None,
            grouping_keys,
        })
    }
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .with_context(|| format!("{flag} requires a value"))
}

fn take_u64(args: &[String], index: &mut usize, flag: &str) -> Result<u64> {
    take_value(args, index, flag)?
        .parse::<u64>()
        .with_context(|| format!("{flag} must be an unsigned integer"))
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
    let platform = PlatformCollector::new("auto").collect();
    accelerator::default_profile_for_platform(&platform)
}

fn parse_depth(raw: &str) -> Result<EvalDepth> {
    match raw {
        "smoke" => Ok(EvalDepth::Smoke),
        "enriched" => Ok(EvalDepth::Enriched),
        other => bail!("unknown eval depth `{other}`"),
    }
}

fn parse_accelerator(raw: &str) -> Result<AcceleratorClass> {
    match raw {
        "cpu" => Ok(AcceleratorClass::Cpu),
        "cuda" => Ok(AcceleratorClass::Cuda),
        "metal" => Ok(AcceleratorClass::Metal),
        "any" => Ok(AcceleratorClass::Any),
        "unavailable" => Ok(AcceleratorClass::Unavailable),
        other => bail!("unknown accelerator `{other}`"),
    }
}

fn print_usage() {
    println!("usage:");
    println!("  evals list scenarios [--root PATH]");
    println!("  evals list bundles");
    println!("  evals run --bundle <bundle_id> --scenario <scenario_id> [--profile NAME] [--artifact-root PATH] [--jsonl PATH] [--warmup-iterations N | --cold]");
    println!("  evals matrix --snapshot <path> [--profile NAME] [--artifact-root PATH] [--warmup-iterations N | --cold]");
    println!("  evals provision --snapshot <path> [--artifact-root PATH]");
    println!("  evals report --input <jsonl> --format markdown");
    println!("  evals report --aggregate <glob-or-path> --output <path> [--snapshot <path>] [--allow-invalid-records]");
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

    #[test]
    fn parses_audio_run_iteration_overrides() {
        let warm = RunOptions::parse(&[
            "--bundle".to_owned(),
            "piper_en_us_ljspeech_medium".to_owned(),
            "--scenario".to_owned(),
            "tts_synthesis_smoke".to_owned(),
            "--warmup-iterations".to_owned(),
            "2".to_owned(),
        ])
        .unwrap();
        let cold = RunOptions::parse(&[
            "--bundle".to_owned(),
            "piper_en_us_ljspeech_medium".to_owned(),
            "--scenario".to_owned(),
            "tts_synthesis_smoke".to_owned(),
            "--cold".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            warm.audio_iteration_overrides,
            AudioIterationOverrides {
                iterations: None,
                warmup_iterations: Some(2)
            }
        );
        assert_eq!(
            cold.audio_iteration_overrides,
            AudioIterationOverrides::cold()
        );
        assert!(RunOptions::parse(&[
            "--bundle".to_owned(),
            "piper_en_us_ljspeech_medium".to_owned(),
            "--scenario".to_owned(),
            "tts_synthesis_smoke".to_owned(),
            "--cold".to_owned(),
            "--warmup-iterations".to_owned(),
            "1".to_owned(),
        ])
        .is_err());
    }
}
