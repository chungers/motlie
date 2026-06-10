use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};

use crate::accelerator;
use crate::metrics::{PerformanceMetrics, ResourceMetrics};
use crate::platform::{sanitize_slug, PlatformCollector, PlatformSnapshot};
use crate::report::OutputSink;
use crate::result::{
    AcceleratorClass, AcceleratorSection, AcceptanceSection, AcceptanceStatus,
    ApplicabilityDecision, CoverageSection, IdentitySection, OutcomeReason, ProfileSection,
    ResultRecord, RuntimeSection, SelectionSection, TerminalOutcome, RESULT_SCHEMA_VERSION,
};
use crate::snapshot::{load_snapshot, EvalSnapshot, SnapshotCell};

pub async fn run_matrix(command_line: Vec<String>, args: &[String]) -> Result<()> {
    let options = MatrixOptions::parse(args)?;
    let snapshot = load_snapshot(&options.snapshot)?;
    let base_collector = PlatformCollector::new("auto");
    let base_platform = base_collector.collect();
    let profile = options
        .profile
        .clone()
        .unwrap_or_else(|| accelerator::default_profile_for_platform(&base_platform));
    let platform_collector = PlatformCollector::new(profile.clone());
    let platform = platform_collector.collect();
    let requested_for_profile = accelerator::requested_for_profile(&profile);
    let accelerator = accelerator::resolve_for_profile(&profile, &platform);
    let run_id = make_run_id(&snapshot.id, &platform, accelerator.resolved_class);
    let results_dir = options.results_root.join(&snapshot.id).join(&run_id);
    fs::create_dir_all(results_dir.join("logs"))
        .with_context(|| format!("failed to create `{}`", results_dir.display()))?;
    let jsonl_path = results_dir.join("results.jsonl");
    let sink = OutputSink::JsonlFile(jsonl_path.clone());

    write_run_manifest(
        &results_dir,
        &snapshot,
        &profile,
        &platform,
        &run_id,
        &command_line,
    )?;

    let mut emitted_pre_run = 0_u64;
    let mut launched = 0_u64;
    for cell in &snapshot.cells {
        let requested = cell.requested_accelerator.unwrap_or(requested_for_profile);
        let cell_accelerator = accelerator::resolve(requested, &platform, None, None);
        let coverage = coverage_for_cell(
            &snapshot,
            cell,
            &profile,
            &platform,
            &cell_accelerator,
            ApplicabilityDecision::Applicable,
            TerminalOutcome::Blocked,
            None,
        );

        if !cell.applies_to_profile(&profile) {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Skipped,
                    Some(OutcomeReason::ProfileNotApplicable),
                ),
                AcceptanceStatus::Skipped,
                Some("profile not applicable to snapshot cell".to_owned()),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        if requires_hf_token(cell) && std::env::var_os("HF_TOKEN").is_none() {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Blocked,
                    Some(OutcomeReason::HfTokenMissing),
                ),
                AcceptanceStatus::Blocked,
                Some(
                    "artifact provisioning requires HF_TOKEN env; token value is not logged"
                        .to_owned(),
                ),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        if is_cpu_profile(&profile)
            && !cell.budgets.cpu_depth_allowed.is_empty()
            && !cell.budgets.cpu_depth_allowed.contains(&cell.depth)
        {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Skipped,
                    Some(OutcomeReason::CpuProfileNotPractical),
                ),
                AcceptanceStatus::Skipped,
                Some("cell depth is not practical for this CPU profile".to_owned()),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        if matches!(requested, AcceleratorClass::Cuda | AcceleratorClass::Metal)
            && cell_accelerator.resolved_class != requested
        {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Skipped,
                    Some(OutcomeReason::AcceleratorUnavailable),
                ),
                AcceptanceStatus::Skipped,
                Some(format!(
                    "requested accelerator {} resolved as {}",
                    requested.as_str(),
                    cell_accelerator.resolved_class.as_str()
                )),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        if is_unverified_metal_gguf_cell(cell, &profile, requested) {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Blocked,
                    Some(OutcomeReason::GgufMetalUnverified),
                ),
                AcceptanceStatus::Blocked,
                Some(
                    "Metal GGUF cell is missing the apple-metal profile feature marker needed to verify Apple clang, Metal backend flags, shader build, and runtime loading"
                        .to_owned(),
                ),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        if options.dry_run {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(TerminalOutcome::Skipped, Some(OutcomeReason::DryRun)),
                AcceptanceStatus::Skipped,
                Some("matrix dry run: child execution skipped".to_owned()),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
            continue;
        }

        launched += 1;
        let child = run_child_cell(
            cell,
            &snapshot,
            &options,
            &profile,
            &run_id,
            &jsonl_path,
            &results_dir,
            requested,
        )?;
        if !child.success {
            let reason = classify_child_failure(cell, requested, &child.log_path);
            let child_coverage = coverage_for_cell(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                ApplicabilityDecision::BlockedPreRun,
                TerminalOutcome::Blocked,
                Some(reason.clone()),
            );
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                child_coverage,
                AcceptanceStatus::Blocked,
                Some(format!(
                    "child cargo invocation failed; see {}",
                    child.log_path.display()
                )),
                &run_id,
                &command_line,
            ))?;
            emitted_pre_run += 1;
        }
    }

    write_summary(&results_dir, &snapshot, &profile, launched, emitted_pre_run)?;
    println!("matrix-results-dir: {}", results_dir.display());
    println!("matrix-results-jsonl: {}", jsonl_path.display());
    Ok(())
}

pub async fn run_provision(command_line: Vec<String>, args: &[String]) -> Result<()> {
    let options = ProvisionOptions::parse(args)?;
    let snapshot = load_snapshot(&options.snapshot)?;
    let hf_token_present = std::env::var_os("HF_TOKEN").is_some();
    let gated_cells = snapshot
        .cells
        .iter()
        .filter(|cell| cell.artifact.requires_hf_token)
        .count();
    let missing_token_cells = if hf_token_present { 0 } else { gated_cells };

    println!("provision-snapshot: {}", snapshot.id);
    println!("artifact-root: {}", options.artifact_root.display());
    println!("hf_token_present: {hf_token_present}");
    println!("gated-artifact-cells: {gated_cells}");
    println!("missing-token-cells: {missing_token_cells}");
    println!("command-args: {}", command_line.len());

    if missing_token_cells > 0 {
        bail!("{missing_token_cells} snapshot cells require HF_TOKEN; token value was not logged");
    }
    Ok(())
}

#[derive(Debug)]
struct MatrixOptions {
    snapshot: PathBuf,
    profile: Option<String>,
    eval_root: PathBuf,
    artifact_root: PathBuf,
    results_root: PathBuf,
    dry_run: bool,
}

impl MatrixOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut snapshot = None;
        let mut profile = None;
        let mut eval_root = repo_root().join("evals");
        let mut artifact_root = motlie_models::default_artifact_root();
        let mut results_root = repo_root().join("evals/results");
        let mut dry_run = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--snapshot" => snapshot = Some(take_value(args, &mut index, "--snapshot")?),
                "--profile" => profile = Some(take_value(args, &mut index, "--profile")?),
                "--root" => eval_root = PathBuf::from(take_value(args, &mut index, "--root")?),
                "--artifact-root" => {
                    artifact_root = PathBuf::from(take_value(args, &mut index, "--artifact-root")?)
                }
                "--results-root" => {
                    results_root = PathBuf::from(take_value(args, &mut index, "--results-root")?)
                }
                "--dry-run" => dry_run = true,
                other => bail!("unknown evals matrix option `{other}`"),
            }
            index += 1;
        }
        Ok(Self {
            snapshot: PathBuf::from(snapshot.context("evals matrix requires --snapshot <path>")?),
            profile,
            eval_root,
            artifact_root,
            results_root,
            dry_run,
        })
    }
}

#[derive(Debug)]
struct ProvisionOptions {
    snapshot: PathBuf,
    artifact_root: PathBuf,
}

impl ProvisionOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut snapshot = None;
        let mut artifact_root = motlie_models::default_artifact_root();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--snapshot" => snapshot = Some(take_value(args, &mut index, "--snapshot")?),
                "--artifact-root" => {
                    artifact_root = PathBuf::from(take_value(args, &mut index, "--artifact-root")?)
                }
                other => bail!("unknown evals provision option `{other}`"),
            }
            index += 1;
        }
        Ok(Self {
            snapshot: PathBuf::from(
                snapshot.context("evals provision requires --snapshot <path>")?,
            ),
            artifact_root,
        })
    }
}

struct ChildOutcome {
    success: bool,
    log_path: PathBuf,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct GgufBindgenEnv {
    args: String,
    repo_wired: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_child_cell(
    cell: &SnapshotCell,
    snapshot: &EvalSnapshot,
    options: &MatrixOptions,
    profile: &str,
    run_id: &str,
    jsonl_path: &Path,
    results_dir: &Path,
    requested: AcceleratorClass,
) -> Result<ChildOutcome> {
    let log_path = results_dir.join("logs").join(format!("{}.log", cell.id));
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open child log `{}`", log_path.display()))?;
    let stderr = log.try_clone()?;
    let gguf_bindgen_env = gguf_bindgen_env(cell);

    let mut command = Command::new("cargo");
    command.current_dir(repo_root());
    command.args(["run", "-p", "evals", "--no-default-features"]);
    let features = cell.features_for_profile(profile);
    if !features.is_empty() {
        command.arg("--features").arg(features.join(" "));
    }
    command.arg("--").arg("run");
    command.args(["--bundle", &cell.bundle_id]);
    if let Some(selector) = &cell.selector {
        command.args(["--selector", selector]);
    }
    command.args(["--scenario", &cell.scenario]);
    command.args(["--profile", profile]);
    command.args(["--root", &options.eval_root.display().to_string()]);
    command.args([
        "--artifact-root",
        &options.artifact_root.display().to_string(),
    ]);
    command.args(["--jsonl", &jsonl_path.display().to_string()]);
    command.args(["--run-id", run_id]);
    command.args(["--snapshot-id", &snapshot.id]);
    command.args(["--cell-id", &cell.id]);
    command.args(["--depth", cell.depth.as_str()]);
    command.args(["--checkpoint-format", &cell.checkpoint_format]);
    command.args(["--artifact-quantization", &cell.quantization]);
    command.args(["--model-family", &cell.model_family]);
    command.args(["--backend", &cell.backend]);
    command.args(["--requested-accelerator", requested.as_str()]);
    command.arg("--quiet-backend-logs");
    if let Some(bindgen_env) = &gguf_bindgen_env {
        command.env("BINDGEN_EXTRA_CLANG_ARGS", &bindgen_env.args);
        command.env(
            "MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED",
            bindgen_env.repo_wired.to_string(),
        );
    } else if is_gguf_cell(cell) {
        command.env("MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED", "false");
    }
    command.stdout(Stdio::from(log));
    command.stderr(Stdio::from(stderr));

    let started_at = Instant::now();
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn child cargo invocation for snapshot cell `{}`",
            cell.id
        )
    })?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(ChildOutcome {
                success: status.success(),
                log_path,
            });
        }
        if started_at.elapsed() > Duration::from_secs(cell.budgets.wall_time_secs()) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(ChildOutcome {
                success: false,
                log_path,
            });
        }
        thread::sleep(Duration::from_millis(250));
    }
}

#[allow(clippy::too_many_arguments)]
fn pre_run_record(
    snapshot: &EvalSnapshot,
    cell: &SnapshotCell,
    profile: &str,
    platform: &PlatformSnapshot,
    accelerator: &AcceleratorSection,
    coverage: CoverageSection,
    status: AcceptanceStatus,
    failure_reason: Option<String>,
    run_id: &str,
    command_line: &[String],
) -> ResultRecord {
    let host_id = platform
        .host_id
        .clone()
        .or_else(|| platform.hostname.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let host_slug = platform
        .host_slug
        .clone()
        .unwrap_or_else(|| sanitize_slug(&host_id));
    let platform = platform.clone();
    let mut env = BTreeMap::new();
    env.insert(
        "HF_TOKEN_PRESENT".to_owned(),
        Some(std::env::var_os("HF_TOKEN").is_some().to_string()),
    );

    ResultRecord {
        schema_version: RESULT_SCHEMA_VERSION,
        identity: IdentitySection {
            run_id: run_id.to_owned(),
            git_sha: git_output(["rev-parse", "HEAD"]),
            git_branch: git_output(["branch", "--show-current"]),
            command_line: command_line.to_vec(),
            host_id: Some(host_id),
            host_slug: Some(host_slug),
        },
        coverage,
        selection: SelectionSection {
            bundle_id: cell.bundle_id.clone(),
            selector: cell.selector.clone(),
            backend: Some(cell.backend.clone()),
            checkpoint_format: Some(cell.checkpoint_format.clone()),
            artifact_quantization: Some(cell.quantization.clone()),
            artifact_snapshot: snapshot.git_sha.clone(),
            artifact_patterns: cell.artifact.patterns.clone(),
            artifact_files: Vec::new(),
            scenario: cell.scenario.clone(),
            capability: cell.capability.as_str().to_owned(),
        },
        profile: ProfileSection {
            name: profile.to_owned(),
        },
        platform,
        accelerator: accelerator.clone(),
        runtime: RuntimeSection {
            cargo_features: cell.features_for_profile(profile),
            build_profile: None,
            quantization: Some(cell.quantization.clone()),
            runtime_precision: None,
            artifact_root: String::new(),
            download_artifacts: false,
            context_length: None,
            gpu_layers: None,
            child_build: None,
            budgets: budget_map(cell),
            env,
        },
        performance: PerformanceMetrics::default(),
        resources: ResourceMetrics::default(),
        acceptance: AcceptanceSection {
            behavior_status: status.clone(),
            performance_status: AcceptanceStatus::NotMeasured,
            resource_status: AcceptanceStatus::NotMeasured,
            accelerator_status: accelerator::evaluate_use(accelerator),
            overall_status: status,
            failure_reason,
            assertions: Vec::new(),
            gates: Vec::new(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn coverage_for_cell(
    snapshot: &EvalSnapshot,
    cell: &SnapshotCell,
    profile: &str,
    platform: &PlatformSnapshot,
    accelerator: &AcceleratorSection,
    applicability: ApplicabilityDecision,
    terminal_outcome: TerminalOutcome,
    reason: Option<OutcomeReason>,
) -> CoverageSection {
    let host_id = platform
        .host_id
        .clone()
        .or_else(|| platform.hostname.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let host_slug = platform
        .host_slug
        .clone()
        .unwrap_or_else(|| sanitize_slug(&host_id));
    let mut grouping_keys = BTreeMap::new();
    grouping_keys.insert("bundle".to_owned(), cell.bundle_id.clone());
    grouping_keys.insert("capability".to_owned(), cell.capability.as_str().to_owned());
    grouping_keys.insert("depth".to_owned(), cell.depth.as_str().to_owned());
    grouping_keys.insert("backend".to_owned(), cell.backend.clone());
    grouping_keys.insert(
        "checkpoint_format".to_owned(),
        cell.checkpoint_format.clone(),
    );
    grouping_keys.insert("quantization".to_owned(), cell.quantization.clone());
    grouping_keys.insert("profile".to_owned(), profile.to_owned());

    CoverageSection {
        snapshot_id: snapshot.id.clone(),
        cell_id: cell.id.clone(),
        depth: cell.depth,
        capability: cell.capability.as_str().to_owned(),
        scenario_id: cell.scenario.clone(),
        bundle_id: cell.bundle_id.clone(),
        model_family: cell.model_family.clone(),
        checkpoint_format: cell.checkpoint_format.clone(),
        quantization: cell.quantization.clone(),
        backend: cell.backend.clone(),
        profile: profile.to_owned(),
        host_id,
        host_slug,
        arch: std::env::consts::ARCH.to_owned(),
        requested_accelerator: accelerator.requested_class,
        resolved_accelerator: accelerator.resolved_class,
        applicability,
        terminal_outcome,
        reason,
        grouping_keys,
    }
}

fn write_run_manifest(
    results_dir: &Path,
    snapshot: &EvalSnapshot,
    profile: &str,
    platform: &PlatformSnapshot,
    run_id: &str,
    command_line: &[String],
) -> Result<()> {
    let manifest = format!(
        "snapshot_id = {:?}\nrun_id = {:?}\nprofile = {:?}\nhost = {:?}\narch = {:?}\ncommand_arg_count = {}\nhf_token_present = {}\n",
        snapshot.id,
        run_id,
        profile,
        platform.host_slug.as_deref().unwrap_or("unknown"),
        platform.arch.as_deref().unwrap_or("unknown"),
        command_line.len(),
        std::env::var_os("HF_TOKEN").is_some()
    );
    fs::write(results_dir.join("run-manifest.toml"), manifest)?;
    Ok(())
}

fn write_summary(
    results_dir: &Path,
    snapshot: &EvalSnapshot,
    profile: &str,
    launched: u64,
    pre_run: u64,
) -> Result<()> {
    let summary = format!(
        "# Eval Matrix Run\n\n- snapshot: `{}`\n- profile: `{}`\n- launched cells: `{}`\n- pre-run records: `{}`\n",
        snapshot.id, profile, launched, pre_run
    );
    fs::write(results_dir.join("summary.md"), summary)?;
    Ok(())
}

fn budget_map(cell: &SnapshotCell) -> BTreeMap<String, String> {
    let mut budgets = BTreeMap::new();
    if let Some(value) = cell.budgets.max_wall_time_secs {
        budgets.insert("max_wall_time_secs".to_owned(), value.to_string());
    }
    if let Some(value) = cell.budgets.max_artifact_bytes {
        budgets.insert("max_artifact_bytes".to_owned(), value.to_string());
    }
    if let Some(value) = cell.budgets.max_rss_bytes {
        budgets.insert("max_rss_bytes".to_owned(), value.to_string());
    }
    budgets
}

fn classify_child_failure(
    cell: &SnapshotCell,
    requested: AcceleratorClass,
    log_path: &Path,
) -> OutcomeReason {
    let log = fs::read_to_string(log_path)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if requested == AcceleratorClass::Metal
        && is_gguf_cell(cell)
        && (log.contains("metal") || log.contains("apple clang") || log.contains("shader"))
    {
        return OutcomeReason::GgufMetalUnverified;
    }
    if log.contains("could not compile") || log.contains("failed to run custom build command") {
        if log.contains("stdbool.h") || log.contains("llama") || log.contains("gguf") {
            OutcomeReason::GgufToolchainFailed
        } else {
            OutcomeReason::FeatureBuildFailed
        }
    } else if log.contains("timed out") {
        OutcomeReason::RuntimeBudgetExceeded
    } else {
        OutcomeReason::ChildRunFailed
    }
}

fn is_unverified_metal_gguf_cell(
    cell: &SnapshotCell,
    profile: &str,
    requested: AcceleratorClass,
) -> bool {
    requested == AcceleratorClass::Metal
        && is_gguf_cell(cell)
        && !cell
            .features_for_profile(profile)
            .iter()
            .any(|feature| feature == "metal")
}

fn gguf_bindgen_env(cell: &SnapshotCell) -> Option<GgufBindgenEnv> {
    if !is_gguf_cell(cell) || std::env::consts::OS != "linux" {
        return None;
    }

    let repo_include = repo_root().join("tools/clang-compat/include");
    let repo_include = repo_include
        .join("stdbool.h")
        .is_file()
        .then_some(repo_include);
    let compiler_include = compiler_builtin_include_dir();

    let repo_wired = repo_include.is_some() && compiler_include.is_some();
    let args = merge_bindgen_args(
        std::env::var("BINDGEN_EXTRA_CLANG_ARGS").ok().as_deref(),
        repo_include.into_iter().chain(compiler_include),
    );

    (!args.is_empty()).then_some(GgufBindgenEnv { args, repo_wired })
}

fn is_gguf_cell(cell: &SnapshotCell) -> bool {
    cell.checkpoint_format.eq_ignore_ascii_case("gguf")
        || cell
            .features_for_profile("")
            .iter()
            .any(|feature| feature.contains("gguf"))
}

fn compiler_builtin_include_dir() -> Option<PathBuf> {
    let output = Command::new("cc")
        .arg("-print-file-name=include")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(value.trim());
    if path.as_os_str().is_empty() || path == Path::new("include") {
        return None;
    }
    (path.join("stddef.h").is_file() && path.join("stdbool.h").is_file()).then_some(path)
}

fn merge_bindgen_args(
    existing: Option<&str>,
    include_dirs: impl IntoIterator<Item = PathBuf>,
) -> String {
    let mut args = existing
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .into_iter()
        .collect::<Vec<_>>();
    args.extend(
        include_dirs
            .into_iter()
            .map(|path| format!("-I{}", path.display())),
    );
    args.join(" ")
}

fn requires_hf_token(cell: &SnapshotCell) -> bool {
    cell.artifact.requires_hf_token
}

fn is_cpu_profile(profile: &str) -> bool {
    profile.contains("cpu")
}

fn make_run_id(
    snapshot_id: &str,
    platform: &PlatformSnapshot,
    accelerator: AcceleratorClass,
) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let host = platform
        .host_slug
        .clone()
        .or_else(|| platform.hostname.as_deref().map(sanitize_slug))
        .unwrap_or_else(|| "unknown".to_owned());
    let git = git_output(["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "nogit".to_owned());
    let arch = platform.arch.as_deref().unwrap_or(std::env::consts::ARCH);
    let pid = std::process::id();
    format!(
        "{}-{}-{}-{}-{}-{}-{}",
        sanitize_slug(snapshot_id),
        timestamp,
        pid,
        git,
        host,
        arch,
        accelerator.as_str()
    )
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("bins/evals should live two levels below the repo root")
        .to_path_buf()
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .with_context(|| format!("{flag} requires a value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::GpuSnapshot;
    use crate::result::EvalDepth;
    use crate::snapshot::EvalSnapshot;

    fn test_snapshot_cell() -> SnapshotCell {
        SnapshotCell {
            id: "cell".to_owned(),
            bundle_id: "bundle".to_owned(),
            scenario: "chat_smoke".to_owned(),
            capability: crate::scenario::CapabilityName::Chat,
            depth: EvalDepth::Smoke,
            model_family: "family".to_owned(),
            checkpoint_format: "hf_safetensors".to_owned(),
            quantization: "default".to_owned(),
            backend: "mistralrs".to_owned(),
            selector: None,
            features: Vec::new(),
            profile_features: BTreeMap::new(),
            profiles: Vec::new(),
            requested_accelerator: None,
            artifact: Default::default(),
            budgets: Default::default(),
        }
    }

    #[test]
    fn run_id_includes_portable_host_and_accelerator() {
        let platform = PlatformSnapshot {
            os: Some("linux".to_owned()),
            arch: Some("x86_64".to_owned()),
            target_triple: None,
            hostname: Some("DGX Spark".to_owned()),
            host_id: Some("DGX Spark".to_owned()),
            host_slug: Some("dgx-spark".to_owned()),
            total_memory_bytes: None,
            available_memory_bytes: None,
            total_swap_bytes: None,
            free_swap_bytes: None,
            gpu_backend: Some("nvidia".to_owned()),
            gpus: vec![GpuSnapshot {
                index: Some(0),
                backend: Some("nvidia".to_owned()),
                id: Some("GPU-1".to_owned()),
                model: Some("NVIDIA GB10".to_owned()),
                total_memory_bytes: None,
                free_memory_bytes: None,
                unified_memory: Some(false),
                recommended_max_working_set_size_bytes: None,
            }],
            accelerator_metadata: BTreeMap::new(),
            unavailable: Vec::new(),
        };

        let run_id = make_run_id("curated-v2-smoke", &platform, AcceleratorClass::Cuda);

        assert!(run_id.contains("curated-v2-smoke"));
        assert!(run_id.contains("dgx-spark"));
        assert!(run_id.ends_with("cuda"));
    }

    #[test]
    fn gguf_cells_are_detected_by_checkpoint_format() {
        let mut cell = test_snapshot_cell();
        cell.checkpoint_format = "gguf".to_owned();

        assert!(is_gguf_cell(&cell));

        cell.checkpoint_format = "hf_safetensors".to_owned();

        assert!(!is_gguf_cell(&cell));
    }

    #[test]
    fn bindgen_args_preserve_existing_and_append_include_dirs() {
        let merged = merge_bindgen_args(
            Some("--target=aarch64-unknown-linux-gnu"),
            [
                PathBuf::from("/repo/tools/clang-compat/include"),
                PathBuf::from("/usr/lib/gcc/include"),
            ],
        );

        assert_eq!(
            merged,
            "--target=aarch64-unknown-linux-gnu -I/repo/tools/clang-compat/include -I/usr/lib/gcc/include"
        );
    }

    #[test]
    fn metal_gguf_cells_require_metal_profile_feature_marker() {
        let mut cell = test_snapshot_cell();
        cell.checkpoint_format = "gguf".to_owned();
        cell.profile_features.insert(
            "apple-metal".to_owned(),
            vec!["accelerate".to_owned(), "metal".to_owned()],
        );

        assert!(!is_unverified_metal_gguf_cell(
            &cell,
            "apple-metal",
            AcceleratorClass::Metal
        ));

        cell.profile_features
            .insert("apple-metal".to_owned(), vec!["accelerate".to_owned()]);

        assert!(is_unverified_metal_gguf_cell(
            &cell,
            "apple-metal",
            AcceleratorClass::Metal
        ));
    }

    #[test]
    fn profile_filter_emits_not_applicable() {
        let snapshot = EvalSnapshot {
            schema_version: 1,
            id: "snap".to_owned(),
            git_sha: None,
            cells: vec![SnapshotCell {
                id: "cell".to_owned(),
                bundle_id: "bundle".to_owned(),
                scenario: "chat_smoke".to_owned(),
                capability: crate::scenario::CapabilityName::Chat,
                depth: EvalDepth::Smoke,
                model_family: "family".to_owned(),
                checkpoint_format: "hf_safetensors".to_owned(),
                quantization: "default".to_owned(),
                backend: "mistralrs".to_owned(),
                selector: None,
                features: Vec::new(),
                profile_features: BTreeMap::new(),
                profiles: vec!["apple-metal".to_owned()],
                requested_accelerator: Some(AcceleratorClass::Metal),
                artifact: Default::default(),
                budgets: Default::default(),
            }],
        };

        assert!(!snapshot.cells[0].applies_to_profile("local-cpu-x86_64"));
    }
}
