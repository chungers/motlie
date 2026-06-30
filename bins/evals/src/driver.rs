use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use motlie_models::{ArtifactSource, BundleId, Catalog, QuantizationScheme};

use crate::accelerator;
use crate::metrics::{PerformanceMetrics, ResourceMetrics};
use crate::platform::{sanitize_slug, PlatformCollector, PlatformSnapshot};
use crate::report::OutputSink;
use crate::result::{
    AcceleratorClass, AcceleratorSection, AcceptanceSection, AcceptanceStatus,
    ApplicabilityDecision, ChildBuildSection, CoverageSection, IdentitySection, OutcomeReason,
    ProfileSection, ResultRecord, RuntimeSection, SelectionSection, TerminalOutcome,
    RESULT_SCHEMA_VERSION,
};
use crate::scenario::{AudioIterationOverrides, CapabilityName};
use crate::snapshot::{load_snapshot, EvalSnapshot, SnapshotCell};

const CHILD_BUILD_PROFILE: &str = "release";
const QWEN3_TTS_CPP_SUBMODULE_PATH: &str = "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp";
const QWEN3_TTS_CPP_REQUIRED_PATHS: &[&str] = &[
    "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/CMakeLists.txt",
    "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/src/qwen3tts_c_api.cpp",
    "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/src/qwen3tts_c_api.h",
    "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/ggml/CMakeLists.txt",
];

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
        )?;

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

        if requires_hf_token(cell, &options.artifact_root) && std::env::var_os("HF_TOKEN").is_none()
        {
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

        if requires_artifact_cache(cell) && !artifact_cache_satisfies(cell, &options.artifact_root)
        {
            sink.emit(&pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                coverage.with_outcome(
                    TerminalOutcome::Blocked,
                    Some(OutcomeReason::ArtifactMissing),
                ),
                AcceptanceStatus::Blocked,
                Some(format!(
                    "artifact cache preflight missing required artifacts for `{}` under `{}`; run `evals provision` or provide --artifact-root before matrix execution",
                    cell.bundle_id,
                    options.artifact_root.display()
                )),
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

        if let Some(child) = preflight_required_submodules(cell, &profile, &results_dir)? {
            let child_coverage = coverage_for_cell(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                ApplicabilityDecision::BlockedPreRun,
                TerminalOutcome::Blocked,
                child.reason.clone(),
            )?;
            let mut record = pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                child_coverage,
                AcceptanceStatus::Blocked,
                Some(format!(
                    "vendored submodule preflight failed; see {}",
                    child.log_path.display()
                )),
                &run_id,
                &command_line,
            );
            record.runtime.child_build = child.child_build.clone();
            record.runtime.build_profile = Some(CHILD_BUILD_PROFILE.to_owned());
            sink.emit(&record)?;
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
            let reason = child_failure_reason(cell, requested, &child);
            let child_coverage = coverage_for_cell(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                ApplicabilityDecision::BlockedPreRun,
                TerminalOutcome::Blocked,
                Some(reason.clone()),
            )?;
            let mut record = pre_run_record(
                &snapshot,
                cell,
                &profile,
                &platform,
                &cell_accelerator,
                child_coverage,
                AcceptanceStatus::Blocked,
                Some(format!(
                    "child eval invocation failed; see {}",
                    child.log_path.display()
                )),
                &run_id,
                &command_line,
            );
            record.runtime.child_build = child.child_build.clone();
            record.runtime.build_profile = Some(CHILD_BUILD_PROFILE.to_owned());
            sink.emit(&record)?;
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
    let cached_gated_cells = snapshot
        .cells
        .iter()
        .filter(|cell| cell.artifact.requires_hf_token)
        .filter(|cell| artifact_cache_satisfies(cell, &options.artifact_root))
        .count();
    let missing_token_cells = if hf_token_present {
        0
    } else {
        gated_cells.saturating_sub(cached_gated_cells)
    };

    println!("provision-snapshot: {}", snapshot.id);
    println!("artifact-root: {}", options.artifact_root.display());
    println!("hf_token_present: {hf_token_present}");
    println!("gated-artifact-cells: {gated_cells}");
    println!("cached-gated-artifact-cells: {cached_gated_cells}");
    println!("missing-token-cells: {missing_token_cells}");
    println!("command-args: {}", command_line.len());

    if missing_token_cells > 0 {
        bail!("{missing_token_cells} uncached snapshot cells require HF_TOKEN; token value was not logged");
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
    audio_iteration_overrides: AudioIterationOverrides,
    dry_run: bool,
}

impl MatrixOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut snapshot = None;
        let mut profile = None;
        let mut eval_root = repo_root().join("evals");
        let mut artifact_root = motlie_models::default_artifact_root();
        let mut results_root = repo_root().join("evals/results");
        let mut audio_iteration_overrides = AudioIterationOverrides::default();
        let mut cold = false;
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
                "--warmup-iterations" => {
                    ensure!(!cold, "--warmup-iterations cannot be combined with --cold");
                    audio_iteration_overrides.warmup_iterations =
                        Some(take_u64(args, &mut index, "--warmup-iterations")?);
                }
                "--cold" => {
                    ensure!(
                        audio_iteration_overrides.is_empty(),
                        "--cold cannot be combined with --warmup-iterations"
                    );
                    cold = true;
                    audio_iteration_overrides = AudioIterationOverrides::cold();
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
            audio_iteration_overrides,
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
    reason: Option<OutcomeReason>,
    child_build: Option<ChildBuildSection>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct GgufBindgenEnv {
    args: String,
    repo_wired: bool,
}

fn preflight_required_submodules(
    cell: &SnapshotCell,
    profile: &str,
    results_dir: &Path,
) -> Result<Option<ChildOutcome>> {
    if !requires_qwen3_tts_cpp_submodule(cell, profile) {
        return Ok(None);
    }

    let missing = missing_qwen3_tts_cpp_submodule_paths();
    if missing.is_empty() {
        return Ok(None);
    }

    let log_path = results_dir
        .join("logs")
        .join(format!("{}-submodule-preflight.log", cell.id));
    append_child_log(
        &log_path,
        &format!(
            "qwen3-tts.cpp submodule preflight missing before init:\n{}",
            missing
                .iter()
                .map(|path| format!("- {}", path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )?;

    let args = vec![
        "submodule".to_owned(),
        "update".to_owned(),
        "--init".to_owned(),
        "--recursive".to_owned(),
        QWEN3_TTS_CPP_SUBMODULE_PATH.to_owned(),
    ];
    append_child_log(&log_path, &format!("running: git {}", args.join(" ")))?;

    let started_at = Instant::now();
    let mut command = Command::new("git");
    command.current_dir(repo_root()).args(&args);
    attach_child_log(&mut command, &log_path)?;
    let status = command.status().with_context(|| {
        format!(
            "failed to spawn qwen3-tts.cpp submodule init for snapshot cell `{}`",
            cell.id
        )
    })?;
    let child_build = ChildBuildSection {
        command: std::iter::once("git".to_owned())
            .chain(args.clone())
            .collect(),
        status: status.code(),
        duration_ms: Some(duration_ms(started_at.elapsed())),
        log_path: Some(log_path.display().to_string()),
    };

    let remaining = missing_qwen3_tts_cpp_submodule_paths();
    if status.success() && remaining.is_empty() {
        append_child_log(
            &log_path,
            "qwen3-tts.cpp submodule preflight satisfied after init",
        )?;
        return Ok(None);
    }

    append_child_log(
        &log_path,
        &format!(
            "qwen3-tts.cpp submodule preflight still missing after init:\n{}",
            remaining
                .iter()
                .map(|path| format!("- {}", path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )?;

    Ok(Some(ChildOutcome {
        success: false,
        log_path,
        reason: Some(OutcomeReason::SubmoduleMissing),
        child_build: Some(child_build),
    }))
}

fn requires_qwen3_tts_cpp_submodule(cell: &SnapshotCell, profile: &str) -> bool {
    cell.bundle_id == "qwen3_tts_cpp_0_6b"
        || cell
            .features_for_profile(profile)
            .iter()
            .any(|feature| feature == "model-qwen3-tts-cpp")
}

fn missing_qwen3_tts_cpp_submodule_paths() -> Vec<PathBuf> {
    let root = repo_root();
    QWEN3_TTS_CPP_REQUIRED_PATHS
        .iter()
        .map(|relative| root.join(relative))
        .filter(|path| !path.is_file())
        .collect()
}

fn append_child_log(log_path: &Path, message: &str) -> Result<()> {
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open child log `{}`", log_path.display()))?;
    writeln!(log, "{message}")?;
    Ok(())
}

fn child_build_args(features: &[String]) -> Vec<String> {
    let mut build_args = vec![
        "build".to_owned(),
        "--release".to_owned(),
        "-p".to_owned(),
        "evals".to_owned(),
        "--no-default-features".to_owned(),
    ];
    if !features.is_empty() {
        build_args.push("--features".to_owned());
        build_args.push(features.join(" "));
    }
    build_args
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
    let gguf_bindgen_env = gguf_bindgen_env(cell);
    let features = cell.features_for_profile(profile);
    let build_args = child_build_args(&features);

    let build_started_at = Instant::now();
    let mut build_command = Command::new("cargo");
    build_command.current_dir(repo_root()).args(&build_args);
    apply_child_env(&mut build_command, cell, gguf_bindgen_env.as_ref());
    attach_child_log(&mut build_command, &log_path)?;
    let build_status = build_command.status().with_context(|| {
        format!(
            "failed to spawn child cargo build for snapshot cell `{}`",
            cell.id
        )
    })?;
    let child_build = ChildBuildSection {
        command: std::iter::once("cargo".to_owned())
            .chain(build_args.clone())
            .collect(),
        status: build_status.code(),
        duration_ms: Some(duration_ms(build_started_at.elapsed())),
        log_path: Some(log_path.display().to_string()),
    };
    if !build_status.success() {
        let reason = classify_child_failure(cell, requested, &log_path);
        return Ok(ChildOutcome {
            success: false,
            log_path,
            reason: Some(reason),
            child_build: Some(child_build),
        });
    }

    let mut run_command = Command::new(evals_child_binary());
    run_command.current_dir(repo_root());
    run_command.arg("run");
    run_command.args(["--bundle", &cell.bundle_id]);
    if let Some(selector) = &cell.selector {
        run_command.args(["--selector", selector]);
    }
    run_command.args(["--scenario", &cell.scenario]);
    run_command.args(["--profile", profile]);
    run_command.args(["--root", &options.eval_root.display().to_string()]);
    run_command.args([
        "--artifact-root",
        &options.artifact_root.display().to_string(),
    ]);
    run_command.args(["--jsonl", &jsonl_path.display().to_string()]);
    run_command.args(["--run-id", run_id]);
    run_command.args(["--snapshot-id", &snapshot.id]);
    run_command.args(["--cell-id", &cell.id]);
    run_command.args(["--depth", cell.depth.as_str()]);
    let artifact_quantization = cell_artifact_quantization_scheme(cell)?.as_str().to_owned();
    run_command.args(["--checkpoint-format", &cell.checkpoint_format]);
    run_command.args(["--artifact-quantization", &artifact_quantization]);
    if let Some(precision) = cell_runtime_precision(cell)? {
        run_command.arg("--precision").arg(precision);
    }
    run_command.args(["--model-family", &cell.model_family]);
    run_command.args(["--backend", &cell.backend]);
    run_command.args(["--requested-accelerator", requested.as_str()]);
    run_command.args(audio_override_args_for_cell(
        cell,
        options.audio_iteration_overrides,
    ));
    run_command.args(["--child-build-log", &log_path.display().to_string()]);
    if let Some(status) = child_build.status {
        run_command
            .arg("--child-build-status")
            .arg(status.to_string());
    }
    if let Some(duration_ms) = child_build.duration_ms {
        run_command
            .arg("--child-build-duration-ms")
            .arg(duration_ms.to_string());
    }
    run_command.arg("--quiet-backend-logs");
    apply_child_env(&mut run_command, cell, gguf_bindgen_env.as_ref());
    attach_child_log(&mut run_command, &log_path)?;
    #[cfg(unix)]
    {
        run_command.process_group(0);
    }

    let started_at = Instant::now();
    let mut child = run_command.spawn().with_context(|| {
        format!(
            "failed to spawn child eval invocation for snapshot cell `{}`",
            cell.id
        )
    })?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(ChildOutcome {
                success: status.success(),
                log_path,
                reason: None,
                child_build: Some(child_build),
            });
        }
        if started_at.elapsed() > Duration::from_secs(cell.budgets.wall_time_secs()) {
            terminate_child(&mut child);
            return Ok(ChildOutcome {
                success: false,
                log_path,
                reason: Some(OutcomeReason::RuntimeBudgetExceeded),
                child_build: Some(child_build),
            });
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn audio_override_args_for_cell(
    cell: &SnapshotCell,
    overrides: AudioIterationOverrides,
) -> Vec<String> {
    if !is_audio_cell(cell) || overrides.is_empty() {
        return Vec::new();
    }

    let mut args = Vec::new();
    if let Some(warmup_iterations) = overrides.warmup_iterations {
        args.push("--warmup-iterations".to_owned());
        args.push(warmup_iterations.to_string());
    }
    if let Some(iterations) = overrides.iterations {
        args.push("--iterations".to_owned());
        args.push(iterations.to_string());
    }
    args
}

fn is_audio_cell(cell: &SnapshotCell) -> bool {
    matches!(cell.capability, CapabilityName::Asr | CapabilityName::Tts)
}

fn child_target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target"))
}

fn evals_child_binary() -> PathBuf {
    child_target_dir()
        .join(CHILD_BUILD_PROFILE)
        .join(format!("evals{}", std::env::consts::EXE_SUFFIX))
}

fn attach_child_log(command: &mut Command, log_path: &Path) -> Result<()> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open child log `{}`", log_path.display()))?;
    let stderr = log.try_clone()?;
    command.stdout(Stdio::from(log));
    command.stderr(Stdio::from(stderr));
    Ok(())
}

fn apply_child_env(
    command: &mut Command,
    cell: &SnapshotCell,
    gguf_bindgen_env: Option<&GgufBindgenEnv>,
) {
    if let Some(bindgen_env) = gguf_bindgen_env {
        command.env("BINDGEN_EXTRA_CLANG_ARGS", &bindgen_env.args);
        command.env(
            "MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED",
            bindgen_env.repo_wired.to_string(),
        );
    } else if is_gguf_cell(cell) {
        command.env("MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED", "false");
    }

    if is_ort_backed_cell(cell) {
        apply_static_ort_child_env(command);
    }

    if is_qwen3_tts_cpp_cell(cell) {
        apply_qwen3_tts_cpp_runtime_env(command);
    }
}

fn is_qwen3_tts_cpp_cell(cell: &SnapshotCell) -> bool {
    cell.bundle_id == "qwen3_tts_cpp_0_6b"
        || cell.backend.contains("qwen3_tts_cpp")
        || cell
            .features_for_profile("")
            .iter()
            .any(|feature| feature == "model-qwen3-tts-cpp")
}

/// qwen3-tts.cpp links `libqwen3tts.so` dynamically (a deliberate `-Bsymbolic`
/// design that isolates its bundled ggml from co-linked backends). Cargo does
/// not propagate an rpath from the dependency's build script to the child
/// `evals` binary, so the runtime loader cannot find the library on its own.
/// Prepend the built shared-library directory to the loader search path for
/// qwen3-tts child runs so the cell can execute instead of failing at exec
/// with `libqwen3tts.so.0: cannot open shared object file`.
fn apply_qwen3_tts_cpp_runtime_env(command: &mut Command) {
    let Some(lib_dir) = qwen3_tts_cpp_lib_dir() else {
        return;
    };
    let lib_dir = lib_dir.display().to_string();
    let var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let prepended = match std::env::var_os(var) {
        Some(existing) if !existing.is_empty() => {
            format!("{lib_dir}:{}", existing.to_string_lossy())
        }
        _ => lib_dir,
    };
    command.env(var, prepended);
}

/// Locate the directory holding the freshly built `libqwen3tts` shared library
/// under the child build profile's cargo `OUT_DIR`. Mirrors the candidate
/// layout that the qwen3-tts.cpp `build.rs` writes the library into.
fn qwen3_tts_cpp_lib_dir() -> Option<PathBuf> {
    let build_root = child_target_dir().join(CHILD_BUILD_PROFILE).join("build");
    let lib_names: &[&str] = if cfg!(target_os = "macos") {
        &["libqwen3tts.dylib", "libqwen3tts.0.dylib"]
    } else if cfg!(target_os = "windows") {
        &["qwen3tts.dll"]
    } else {
        &["libqwen3tts.so", "libqwen3tts.so.0"]
    };

    let mut newest: Option<(SystemTime, PathBuf)> = None;
    let entries = fs::read_dir(&build_root).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name
            .to_string_lossy()
            .starts_with("motlie-model-qwen3-tts-cpp-")
        {
            continue;
        }
        let out_dir = entry.path().join("out");
        for candidate in [
            out_dir.join("build/vendor-build"),
            out_dir.join("build"),
            out_dir.join("vendor-build"),
            out_dir.join("lib"),
        ] {
            let Some(lib_path) = lib_names
                .iter()
                .map(|lib| candidate.join(lib))
                .find(|path| path.is_file())
            else {
                continue;
            };
            let modified = lib_path
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(UNIX_EPOCH);
            if newest.as_ref().is_none_or(|(best, _)| modified >= *best) {
                newest = Some((modified, candidate));
            }
        }
    }

    newest.map(|(_, dir)| dir)
}

fn apply_static_ort_child_env(command: &mut Command) {
    for var in [
        "ORT_LIB_PATH",
        "ORT_LIB_LOCATION",
        "ORT_PREFER_DYNAMIC_LINK",
        "ORT_SKIP_DOWNLOAD",
        "ORT_OFFLINE",
        "CARGO_NET_OFFLINE",
    ] {
        command.env_remove(var);
    }
    command.env("MOTLIE_ORT_SOURCE", "sherpa-onnx");
    command.env("MOTLIE_ORT_LINK_POLICY", "static");
}

fn terminate_child(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pgid = format!("-{}", child.id());
        let _ = Command::new("kill").args(["-TERM", &pgid]).status();
        thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[allow(clippy::too_many_arguments)]
fn pre_run_record(
    snapshot: &EvalSnapshot,
    cell: &SnapshotCell,
    profile: &str,
    platform: &PlatformSnapshot,
    accelerator: &AcceleratorSection,
    mut coverage: CoverageSection,
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

    let coverage_quantization = coverage.quantization.clone();
    let artifact_quantization = cell_artifact_quantization_scheme(cell)
        .map(|scheme| scheme.as_str().to_owned())
        .unwrap_or_else(|_| coverage_quantization.clone());
    let accelerator = synthetic_pre_run_accelerator(accelerator);
    let accelerator_status = synthetic_pre_run_accelerator_status(&accelerator);
    coverage.requested_accelerator = accelerator.requested_class;
    coverage.resolved_accelerator = accelerator.resolved_class;

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
            artifact_quantization: Some(artifact_quantization),
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
        accelerator,
        runtime: RuntimeSection {
            cargo_features: cell.features_for_profile(profile),
            build_profile: None,
            quantization: Some(coverage_quantization.clone()),
            runtime_precision: Some(coverage_quantization),
            artifact_root: String::new(),
            download_artifacts: false,
            context_length: None,
            gpu_layers: accelerator::runtime_gpu_layers(),
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
            accelerator_status,
            overall_status: status,
            failure_reason,
            assertions: Vec::new(),
            gates: Vec::new(),
        },
    }
}

fn synthetic_pre_run_accelerator(accelerator: &AcceleratorSection) -> AcceleratorSection {
    let mut accelerator = accelerator.clone();
    if accelerator.fallback_reason == Some(OutcomeReason::BackendOffloadUnverified) {
        accelerator.fallback_reason = None;
        accelerator.backend_mode = Some("backend_not_started".to_owned());
        accelerator.offload = None;
        accelerator.use_proof_source = Some("pre_run:backend_not_started".to_owned());
    }
    accelerator
}

fn synthetic_pre_run_accelerator_status(accelerator: &AcceleratorSection) -> AcceptanceStatus {
    if accelerator.use_proof_source.as_deref() == Some("pre_run:backend_not_started") {
        AcceptanceStatus::NotMeasured
    } else {
        accelerator::evaluate_use(accelerator)
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
) -> Result<CoverageSection> {
    let host_id = platform
        .host_id
        .clone()
        .or_else(|| platform.hostname.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let host_slug = platform
        .host_slug
        .clone()
        .unwrap_or_else(|| sanitize_slug(&host_id));
    let quantization = cell_quantization_scheme(cell)?.as_str().to_owned();
    let mut grouping_keys = BTreeMap::new();
    grouping_keys.insert("bundle".to_owned(), cell.bundle_id.clone());
    grouping_keys.insert("capability".to_owned(), cell.capability.as_str().to_owned());
    grouping_keys.insert("depth".to_owned(), cell.depth.as_str().to_owned());
    grouping_keys.insert("backend".to_owned(), cell.backend.clone());
    grouping_keys.insert(
        "checkpoint_format".to_owned(),
        cell.checkpoint_format.clone(),
    );
    grouping_keys.insert("quantization".to_owned(), quantization.clone());
    grouping_keys.insert("profile".to_owned(), profile.to_owned());
    if cell.capability == CapabilityName::Tts {
        grouping_keys.insert(
            "speech_mode".to_owned(),
            tts_speech_mode_for_scenario(&cell.scenario).to_owned(),
        );
    }

    Ok(CoverageSection {
        snapshot_id: snapshot.id.clone(),
        cell_id: cell.id.clone(),
        depth: cell.depth,
        capability: cell.capability.as_str().to_owned(),
        scenario_id: cell.scenario.clone(),
        bundle_id: cell.bundle_id.clone(),
        model_family: cell.model_family.clone(),
        checkpoint_format: cell.checkpoint_format.clone(),
        quantization: quantization.clone(),
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
    })
}

fn tts_speech_mode_for_scenario(scenario_id: &str) -> &'static str {
    match scenario_id {
        "tts_streaming_synthesis" => "streaming",
        _ => "buffered",
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

fn child_failure_reason(
    cell: &SnapshotCell,
    requested: AcceleratorClass,
    child: &ChildOutcome,
) -> OutcomeReason {
    child
        .reason
        .clone()
        .unwrap_or_else(|| classify_child_failure(cell, requested, &child.log_path))
}

fn classify_child_failure(
    cell: &SnapshotCell,
    requested: AcceleratorClass,
    log_path: &Path,
) -> OutcomeReason {
    let log = fs::read_to_string(log_path)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if is_native_link_failure_log(&log) {
        return OutcomeReason::NativeLinkFailed;
    }
    if log.contains("401")
        || log.contains("unauthorized")
        || log.contains("gated repo")
        || log.contains("access to model")
    {
        return OutcomeReason::ArtifactUnauthorized;
    }
    if log.contains("missing artifacts")
        || log.contains("missing artifact")
        || log.contains("no such file or directory")
        || log.contains("not found in local artifact cache")
    {
        return OutcomeReason::ArtifactMissing;
    }
    if log.contains("submodule is missing")
        || log.contains("submodule checkout")
        || log.contains("nested `ggml` submodule is missing")
    {
        return OutcomeReason::SubmoduleMissing;
    }
    if log.contains("undefined symbols") && (log.contains("vdsp") || log.contains("accelerate")) {
        return OutcomeReason::NativeToolchainMissing;
    }
    if requested == AcceleratorClass::Metal
        && is_gguf_cell(cell)
        && (log.contains("failed to compile metal")
            || log.contains("metal shader compile")
            || log.contains("metallib")
            || log.contains("xcrun")
            || log.contains("metal toolchain"))
    {
        return OutcomeReason::GgufMetalUnverified;
    }
    if log.contains("could not compile") || log.contains("failed to run custom build command") {
        if is_native_link_failure_log(&log) {
            OutcomeReason::NativeLinkFailed
        } else if log.contains("stdbool.h") || log.contains("llama") || log.contains("gguf") {
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

fn is_native_link_failure_log(log: &str) -> bool {
    log.contains("ortgetapibase")
        || log.contains("_ortgetapibase")
        || log.contains("undefined symbols")
        || log.contains("undefined reference")
        || log.contains("symbol(s) not found")
        || log.contains("linking with `cc` failed")
        || log.contains("linking with `clang` failed")
        || log.contains("linker command failed")
        || log.contains("ld returned 1 exit status")
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
    let repo_include = repo_compat_include_dir(repo_include);
    build_gguf_bindgen_env(
        std::env::var("BINDGEN_EXTRA_CLANG_ARGS").ok().as_deref(),
        repo_include,
        compiler_builtin_include_dirs(),
    )
}

fn repo_compat_include_dir(path: PathBuf) -> Option<PathBuf> {
    ["stdbool.h", "stddef.h", "stdarg.h"]
        .into_iter()
        .all(|header| path.join(header).is_file())
        .then_some(path)
}

fn build_gguf_bindgen_env(
    existing: Option<&str>,
    repo_include: Option<PathBuf>,
    compiler_includes: Vec<PathBuf>,
) -> Option<GgufBindgenEnv> {
    let repo_wired = repo_include.is_some();
    let args = merge_bindgen_args(existing, compiler_includes.into_iter().chain(repo_include));

    (!args.is_empty()).then_some(GgufBindgenEnv { args, repo_wired })
}

fn is_ort_backed_cell(cell: &SnapshotCell) -> bool {
    cell.checkpoint_format.eq_ignore_ascii_case("onnx")
        || cell.backend.contains("ort")
        || cell.backend.contains("piper")
        || cell.backend.contains("sherpa")
        || cell.features_for_profile("").iter().any(|feature| {
            feature.contains("piper")
                || feature.contains("sherpa-onnx")
                || feature.contains("moonshine")
        })
}

fn is_gguf_cell(cell: &SnapshotCell) -> bool {
    cell.checkpoint_format.eq_ignore_ascii_case("gguf")
        || cell
            .features_for_profile("")
            .iter()
            .any(|feature| feature.contains("gguf"))
}

fn compiler_builtin_include_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(path) = compiler_builtin_include_dir_from_cc() {
        push_unique_compiler_include_dir(&mut dirs, path);
    }
    append_gcc_builtin_include_dirs(&mut dirs);
    dirs
}

fn compiler_builtin_include_dir_from_cc() -> Option<PathBuf> {
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
    is_compiler_builtin_include_dir(&path).then_some(path)
}

fn append_gcc_builtin_include_dirs(dirs: &mut Vec<PathBuf>) {
    let Ok(triples) = fs::read_dir("/usr/lib/gcc") else {
        return;
    };
    for triple in triples.flatten() {
        let Ok(versions) = fs::read_dir(triple.path()) else {
            continue;
        };
        for version in versions.flatten() {
            push_unique_compiler_include_dir(dirs, version.path().join("include"));
        }
    }
}

fn push_unique_compiler_include_dir(dirs: &mut Vec<PathBuf>, path: PathBuf) {
    if is_compiler_builtin_include_dir(&path) && !dirs.iter().any(|existing| existing == &path) {
        dirs.push(path);
    }
}

fn is_compiler_builtin_include_dir(path: &Path) -> bool {
    path.join("stdarg.h").is_file() && path.join("stddef.h").is_file()
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

fn cell_runtime_precision(cell: &SnapshotCell) -> Result<Option<String>> {
    let scheme = cell_quantization_scheme(cell)?;
    if cell.quantization == "default" {
        return Ok(None);
    }
    Ok(Some(scheme.as_str().to_owned()))
}

fn cell_artifact_quantization_scheme(cell: &SnapshotCell) -> Result<QuantizationScheme> {
    let scheme = cell_quantization_scheme(cell)?;
    if matches!(
        scheme,
        QuantizationScheme::IsqQ4 | QuantizationScheme::IsqQ8
    ) {
        default_quantization_scheme(cell)
    } else {
        Ok(scheme)
    }
}

fn cell_quantization_scheme(cell: &SnapshotCell) -> Result<QuantizationScheme> {
    match cell.quantization.as_str() {
        "q4_k_m" | "gguf_q4_k_m" => Ok(QuantizationScheme::GgufQ4_K_M),
        "q4_0" | "gguf_q4_0" => Ok(QuantizationScheme::GgufQ4_0),
        "q5_k_m" | "gguf_q5_k_m" => Ok(QuantizationScheme::GgufQ5_K_M),
        "q8_0" | "gguf_q8_0" => Ok(QuantizationScheme::GgufQ8_0),
        "onnx_int8" => Ok(QuantizationScheme::OnnxInt8),
        "isq_q4" => Ok(QuantizationScheme::IsqQ4),
        "isq_q8" => Ok(QuantizationScheme::IsqQ8),
        "fp32" | "f32" => Ok(QuantizationScheme::Fp32),
        "fp16" | "f16" => Ok(QuantizationScheme::Fp16),
        "bf16" => Ok(QuantizationScheme::Bf16),
        "default" => default_quantization_scheme(cell),
        other => bail!("unknown snapshot quantization `{other}`"),
    }
}

fn default_quantization_scheme(cell: &SnapshotCell) -> Result<QuantizationScheme> {
    match cell.bundle_id.as_str() {
        "qwen3_4b" | "gemma4_e2b" | "gemma4_e4b" | "qwen3_embedding_06b" => {
            Ok(QuantizationScheme::Bf16)
        }
        "embeddinggemma_300m" => Ok(QuantizationScheme::Fp32),
        "whisper_base_en" => Ok(QuantizationScheme::Fp16),
        "sherpa_onnx_streaming_zipformer_en" | "kokoro_82m" => {
            Ok(QuantizationScheme::OnnxInt8)
        }
        "sherpa_onnx_streaming_zipformer_en_kroko_2025"
        | "moonshine_streaming_en"
        | "piper_en_us_ljspeech_medium" => Ok(QuantizationScheme::Fp32),
        other => bail!(
            "snapshot cell `{}` uses default quantization for bundle `{other}` without a bundle-aware QuantizationScheme mapping",
            cell.id
        ),
    }
}

fn requires_hf_token(cell: &SnapshotCell, artifact_root: &Path) -> bool {
    cell.artifact.requires_hf_token && !artifact_cache_satisfies(cell, artifact_root)
}

fn requires_artifact_cache(cell: &SnapshotCell) -> bool {
    !cell.artifact.allow_missing && !cell.artifact.patterns.is_empty()
}

fn artifact_cache_satisfies(cell: &SnapshotCell, artifact_root: &Path) -> bool {
    if cell.artifact.allow_missing {
        return true;
    }
    if cell.artifact.patterns.is_empty() {
        return false;
    }

    let search_root = match bundle_artifact_cache_root(cell, artifact_root) {
        Some(path) => {
            if !path.exists() {
                return false;
            }
            path
        }
        None => artifact_root.to_path_buf(),
    };
    if !search_root.exists() {
        return false;
    }

    let mut patterns = cell.artifact.patterns.clone();
    patterns.sort();
    patterns.dedup();
    patterns
        .iter()
        .all(|pattern| artifact_pattern_exists(&search_root, pattern))
}

fn bundle_artifact_cache_root(cell: &SnapshotCell, artifact_root: &Path) -> Option<PathBuf> {
    if let Some(repo) = curated_hf_repo_for_bundle(&cell.bundle_id) {
        return Some(artifact_root.join(format!("models--{}", repo.replace('/', "--"))));
    }

    let catalog = Catalog::with_defaults();
    let descriptor = catalog.bundle(&BundleId::new(cell.bundle_id.clone()))?;
    let artifacts = descriptor.artifacts.as_ref()?;
    match &artifacts.source {
        ArtifactSource::HuggingFace { repo } => {
            Some(artifact_root.join(format!("models--{}", repo.replace('/', "--"))))
        }
    }
}

fn curated_hf_repo_for_bundle(bundle_id: &str) -> Option<&'static str> {
    match bundle_id {
        "embeddinggemma_300m" => Some("google/embeddinggemma-300m"),
        "qwen3_embedding_06b" => Some("Qwen/Qwen3-Embedding-0.6B"),
        "qwen3_4b" => Some("Qwen/Qwen3-4B"),
        "gemma4_e2b" => Some("google/gemma-4-E2B-it"),
        "gemma4_e4b" => Some("google/gemma-4-E4B-it"),
        "qwen3_4b_gguf" => Some("Qwen/Qwen3-4B-GGUF"),
        "qwen3_6_27b_gguf" => Some("unsloth/Qwen3.6-27B-GGUF"),
        "gemma4_e2b_gguf" => Some("unsloth/gemma-4-E2B-it-GGUF"),
        "gemma4_e4b_gguf" => Some("unsloth/gemma-4-E4B-it-GGUF"),
        "gemma4_12b_gguf" => Some("unsloth/gemma-4-12b-it-GGUF"),
        "gemma4_12b_qat_gguf" => Some("google/gemma-4-12B-it-qat-q4_0-gguf"),
        "whisper_base_en" => Some("ggerganov/whisper.cpp"),
        "moonshine_streaming_en" => Some("UsefulSensors/moonshine-streaming"),
        "sherpa_onnx_streaming_zipformer_en" => {
            Some("csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26")
        }
        "sherpa_onnx_streaming_zipformer_en_kroko_2025" => {
            Some("csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06")
        }
        "piper_en_us_ljspeech_medium" => Some("rhasspy/piper-voices"),
        "qwen3_tts_cpp_0_6b" => Some("koboldcpp/tts"),
        "kokoro_82m" => Some("onnx-community/Kokoro-82M-v1.0-ONNX"),
        _ => None,
    }
}

fn artifact_pattern_exists(root: &Path, pattern: &str) -> bool {
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| wildcard_match(pattern, name))
            {
                return true;
            }
        }
    }
    false
}

fn wildcard_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return pattern == value;
    }

    let mut remainder = value;
    let mut parts = pattern.split('*').peekable();
    if let Some(first) = parts.next() {
        if !first.is_empty() {
            let Some(stripped) = remainder.strip_prefix(first) else {
                return false;
            };
            remainder = stripped;
        }
    }

    while let Some(part) = parts.next() {
        if part.is_empty() {
            continue;
        }
        if parts.peek().is_none() {
            return remainder.ends_with(part);
        }
        let Some(index) = remainder.find(part) else {
            return false;
        };
        remainder = &remainder[index + part.len()..];
    }
    true
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

fn take_u64(args: &[String], index: &mut usize, flag: &str) -> Result<u64> {
    take_value(args, index, flag)?
        .parse::<u64>()
        .with_context(|| format!("{flag} must be an unsigned integer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::GpuSnapshot;
    use crate::result::EvalDepth;
    use crate::snapshot::{load_snapshot, EvalSnapshot};

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
    fn matrix_options_parse_audio_iteration_overrides() {
        let warm = MatrixOptions::parse(&[
            "--snapshot".to_owned(),
            "snapshot.toml".to_owned(),
            "--warmup-iterations".to_owned(),
            "2".to_owned(),
        ])
        .unwrap();
        let cold = MatrixOptions::parse(&[
            "--snapshot".to_owned(),
            "snapshot.toml".to_owned(),
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
        assert!(MatrixOptions::parse(&[
            "--snapshot".to_owned(),
            "snapshot.toml".to_owned(),
            "--cold".to_owned(),
            "--warmup-iterations".to_owned(),
            "1".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn audio_iteration_overrides_forward_only_to_audio_cells() {
        let mut asr = test_snapshot_cell();
        asr.capability = CapabilityName::Asr;
        let mut tts = test_snapshot_cell();
        tts.capability = CapabilityName::Tts;
        let chat = test_snapshot_cell();

        assert_eq!(
            audio_override_args_for_cell(
                &asr,
                AudioIterationOverrides {
                    iterations: None,
                    warmup_iterations: Some(3)
                }
            ),
            vec!["--warmup-iterations".to_owned(), "3".to_owned()]
        );
        assert_eq!(
            audio_override_args_for_cell(&tts, AudioIterationOverrides::cold()),
            vec![
                "--warmup-iterations".to_owned(),
                "0".to_owned(),
                "--iterations".to_owned(),
                "1".to_owned(),
            ]
        );
        assert!(audio_override_args_for_cell(&chat, AudioIterationOverrides::cold()).is_empty());
    }

    #[test]
    fn child_failure_record_does_not_claim_backend_offload_unverified() {
        let snapshot = EvalSnapshot {
            schema_version: 1,
            id: "snap".to_owned(),
            git_sha: None,
            cells: Vec::new(),
        };
        let mut cell = test_snapshot_cell();
        cell.bundle_id = "qwen3_4b".to_owned();
        cell.quantization = "bf16".to_owned();
        let platform = PlatformSnapshot {
            os: Some("linux".to_owned()),
            arch: Some("aarch64".to_owned()),
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
        let speculative = accelerator::resolve(AcceleratorClass::Cuda, &platform, None, None);
        assert_eq!(
            speculative.fallback_reason,
            Some(OutcomeReason::BackendOffloadUnverified)
        );
        let coverage = coverage_for_cell(
            &snapshot,
            &cell,
            "dgx-spark",
            &platform,
            &speculative,
            ApplicabilityDecision::BlockedPreRun,
            TerminalOutcome::Blocked,
            Some(OutcomeReason::NativeLinkFailed),
        )
        .expect("coverage");

        let record = pre_run_record(
            &snapshot,
            &cell,
            "dgx-spark",
            &platform,
            &speculative,
            coverage,
            AcceptanceStatus::Blocked,
            Some("child eval invocation failed".to_owned()),
            "run",
            &[],
        );

        assert_eq!(
            record.coverage.reason,
            Some(OutcomeReason::NativeLinkFailed)
        );
        assert_eq!(record.accelerator.requested_class, AcceleratorClass::Cuda);
        assert_eq!(record.accelerator.resolved_class, AcceleratorClass::Cuda);
        assert_eq!(record.coverage.resolved_accelerator, AcceleratorClass::Cuda);
        assert_eq!(record.accelerator.selected_devices.len(), 1);
        assert_eq!(record.accelerator.fallback_reason, None);
        assert_eq!(
            record.accelerator.backend_mode.as_deref(),
            Some("backend_not_started")
        );
        assert_eq!(
            record.accelerator.use_proof_source.as_deref(),
            Some("pre_run:backend_not_started")
        );
        assert_eq!(
            record.acceptance.accelerator_status,
            AcceptanceStatus::NotMeasured
        );
        assert_eq!(record.acceptance.overall_status, AcceptanceStatus::Blocked);
    }

    #[test]
    fn child_timeout_reason_does_not_depend_on_log_marker() {
        let cell = test_snapshot_cell();
        let log_path = std::env::temp_dir().join(format!(
            "motlie-evals-timeout-test-{}.log",
            std::process::id()
        ));
        std::fs::write(&log_path, "child exited without timeout marker").unwrap();
        let child = ChildOutcome {
            success: false,
            log_path: log_path.clone(),
            reason: Some(OutcomeReason::RuntimeBudgetExceeded),
            child_build: None,
        };

        let reason = child_failure_reason(&cell, AcceleratorClass::Cpu, &child);

        let _ = std::fs::remove_file(log_path);
        assert_eq!(reason, OutcomeReason::RuntimeBudgetExceeded);
    }

    #[test]
    fn ort_link_error_is_native_link_failure_before_auth_heuristics() {
        let cell = test_snapshot_cell();
        let log_path = std::env::temp_dir().join(format!(
            "motlie-evals-ort-link-test-{}.log",
            std::process::id()
        ));
        std::fs::write(
            &log_path,
            r#"Undefined symbols for architecture arm64:
  "_OrtGetApiBase", referenced from:
unauthorized access to model cache"#,
        )
        .unwrap();

        let reason = classify_child_failure(&cell, AcceleratorClass::Cpu, &log_path);

        let _ = std::fs::remove_file(log_path);
        assert_eq!(reason, OutcomeReason::NativeLinkFailed);
    }

    #[test]
    fn onnx_cells_are_ort_backed_for_static_child_policy() {
        let mut cell = test_snapshot_cell();
        cell.checkpoint_format = "onnx".to_owned();
        cell.backend = "piper".to_owned();
        cell.features = vec!["model-piper-en-us-ljspeech-medium".to_owned()];

        assert!(is_ort_backed_cell(&cell));
    }

    #[test]
    fn ort_child_env_scrubs_dynamic_overrides_and_selects_static_source() {
        let mut cell = test_snapshot_cell();
        cell.checkpoint_format = "onnx".to_owned();
        cell.backend = "sherpa_onnx".to_owned();
        let mut command = Command::new("cargo");
        command.env("ORT_LIB_PATH", "/tmp/host-ort");
        command.env("ORT_PREFER_DYNAMIC_LINK", "1");

        apply_child_env(&mut command, &cell, None);

        let env = command.get_envs().collect::<BTreeMap<_, _>>();
        assert_eq!(env.get(std::ffi::OsStr::new("ORT_LIB_PATH")), Some(&None));
        assert_eq!(
            env.get(std::ffi::OsStr::new("ORT_PREFER_DYNAMIC_LINK")),
            Some(&None)
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new("MOTLIE_ORT_SOURCE")),
            Some(&Some(std::ffi::OsStr::new("sherpa-onnx")))
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new("MOTLIE_ORT_LINK_POLICY")),
            Some(&Some(std::ffi::OsStr::new("static")))
        );
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
    fn snapshot_quantization_maps_to_scheme_ids_without_bits_collapse() {
        let mut cell = test_snapshot_cell();

        cell.bundle_id = "qwen3_4b".to_owned();
        cell.quantization = "default".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::Bf16
        );
        assert_eq!(cell_runtime_precision(&cell).unwrap(), None);

        cell.bundle_id = "embeddinggemma_300m".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::Fp32
        );

        cell.quantization = "q4_k_m".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::GgufQ4_K_M
        );
        assert_eq!(
            cell_runtime_precision(&cell).unwrap(),
            Some("gguf_q4_k_m".to_owned())
        );

        cell.quantization = "q4_0".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::GgufQ4_0
        );
        assert_eq!(
            cell_runtime_precision(&cell).unwrap(),
            Some("gguf_q4_0".to_owned())
        );

        cell.quantization = "q5_k_m".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::GgufQ5_K_M
        );

        cell.quantization = "q8_0".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::GgufQ8_0
        );

        cell.bundle_id = "qwen3_4b".to_owned();
        cell.quantization = "isq_q4".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::IsqQ4
        );
        assert_eq!(
            cell_runtime_precision(&cell).unwrap(),
            Some("isq_q4".to_owned())
        );
        assert_eq!(
            cell_artifact_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::Bf16
        );

        cell.quantization = "isq_q8".to_owned();
        assert_eq!(
            cell_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::IsqQ8
        );
        assert_eq!(
            cell_runtime_precision(&cell).unwrap(),
            Some("isq_q8".to_owned())
        );
        assert_eq!(
            cell_artifact_quantization_scheme(&cell).unwrap(),
            QuantizationScheme::Bf16
        );

        cell.quantization = "q6_k".to_owned();
        assert!(cell_quantization_scheme(&cell).is_err());
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
    fn gguf_bindgen_env_is_repo_wired_with_repo_compat_only() {
        let env = build_gguf_bindgen_env(
            None,
            Some(PathBuf::from("/repo/tools/clang-compat/include")),
            Vec::new(),
        )
        .expect("repo compat include should produce bindgen env");

        assert!(env.repo_wired);
        assert_eq!(env.args, "-I/repo/tools/clang-compat/include");
    }

    #[test]
    fn gguf_bindgen_env_keeps_compiler_builtins_before_repo_compat() {
        let env = build_gguf_bindgen_env(
            Some("--target=x86_64-unknown-linux-gnu"),
            Some(PathBuf::from("/repo/tools/clang-compat/include")),
            vec![PathBuf::from("/usr/lib/gcc/x86_64-linux-gnu/13/include")],
        )
        .expect("compiler plus repo includes should produce bindgen env");

        assert!(env.repo_wired);
        assert_eq!(
            env.args,
            "--target=x86_64-unknown-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/13/include -I/repo/tools/clang-compat/include"
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
    fn child_build_args_use_release_profile() {
        let args = child_build_args(&["model-gemma4-e2b".to_owned()]);

        assert_eq!(args[0], "build");
        assert_eq!(args[1], "--release");
        assert!(args.contains(&"--no-default-features".to_owned()));
        assert_eq!(args.last().map(String::as_str), Some("model-gemma4-e2b"));
    }

    #[test]
    fn qwen3_tts_cpp_cells_require_submodule_preflight() {
        let mut cell = test_snapshot_cell();

        assert!(!requires_qwen3_tts_cpp_submodule(&cell, "local-cpu-x86_64"));

        cell.bundle_id = "qwen3_tts_cpp_0_6b".to_owned();
        assert!(requires_qwen3_tts_cpp_submodule(&cell, "local-cpu-x86_64"));

        cell.bundle_id = "bundle".to_owned();
        cell.features.push("model-qwen3-tts-cpp".to_owned());
        assert!(requires_qwen3_tts_cpp_submodule(&cell, "local-cpu-x86_64"));
    }

    #[test]
    fn artifact_cache_preflight_requires_declared_patterns() {
        let mut cell = test_snapshot_cell();

        assert!(!requires_artifact_cache(&cell));

        cell.artifact.patterns = vec!["*.gguf".to_owned()];
        assert!(requires_artifact_cache(&cell));

        cell.artifact.allow_missing = true;
        assert!(!requires_artifact_cache(&cell));
    }

    #[test]
    fn artifact_cache_root_uses_curated_repo_without_feature_enabled() {
        let mut cell = test_snapshot_cell();
        cell.bundle_id = "qwen3_4b".to_owned();

        let root = bundle_artifact_cache_root(&cell, Path::new("/tmp/hf-cache")).unwrap();

        assert_eq!(root, PathBuf::from("/tmp/hf-cache/models--Qwen--Qwen3-4B"));
    }

    #[test]
    fn curated_asr_tts_snapshot_patterns_match_cached_artifact_basenames() {
        fn write_file(path: &Path) {
            std::fs::create_dir_all(path.parent().unwrap()).expect("parent dir should be writable");
            std::fs::write(path, b"stub").expect("artifact stub should be writable");
        }

        let root = std::env::temp_dir().join(format!(
            "motlie-evals-artifact-patterns-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);

        write_file(&root.join("models--ggerganov--whisper.cpp/snapshots/test/ggml-base.en.bin"));
        for name in [
            "frontend.ort",
            "encoder.ort",
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "streaming_config.json",
            "tokenizer.json",
        ] {
            write_file(&root.join(format!(
                "models--UsefulSensors--moonshine-streaming/snapshots/test/onnx/small/{name}"
            )));
        }
        write_file(&root.join("models--koboldcpp--tts/snapshots/test/qwen3-tts-0.6b-q8_0.gguf"));
        write_file(
            &root.join("models--koboldcpp--tts/snapshots/test/qwen3-tts-tokenizer-f16.gguf"),
        );
        write_file(&root.join(
            "models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/onnx/model_quantized.onnx",
        ));
        write_file(
            &root
                .join("models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/tokenizer.json"),
        );
        write_file(&root.join(
            "models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/voices/af_bella.bin",
        ));
        write_file(
            &root.join("models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/model.onnx"),
        );
        write_file(
            &root.join("models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/voices.bin"),
        );
        write_file(
            &root.join("models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/tokens.txt"),
        );
        write_file(&root.join(
            "models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/espeak-ng-data/phondata",
        ));
        write_file(&root.join(
            "models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/test/espeak-ng-data/phontab",
        ));

        let snapshot = load_snapshot(&repo_root().join("evals/snapshots/curated-v2-smoke.toml"))
            .expect("curated snapshot should parse");
        for cell_id in [
            "whisper_base_en__asr_short_transcription__smoke__ggml_default",
            "moonshine_streaming_en__asr_short_transcription__smoke__hf_default",
            "kokoro_82m__tts_synthesis_smoke__smoke__onnx_default",
            "kokoro_82m__tts_streaming_synthesis__smoke__onnx_default",
            "qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0",
        ] {
            let cell = snapshot
                .cells
                .iter()
                .find(|cell| cell.id == cell_id)
                .expect("snapshot cell should exist");
            assert!(
                artifact_cache_satisfies(cell, &root),
                "{cell_id} artifact preflight should match cached filenames"
            );
        }

        let _ = std::fs::remove_dir_all(root);
    }

    // --- Issue #518 structural rules: snapshot/data bundle_id must be a
    // CuratedBundle canonical string, and every variant must have a cell. ---

    #[test]
    fn snapshot_bundle_ids_are_canonical() {
        let snapshot = load_snapshot(&repo_root().join("evals/snapshots/curated-v2-smoke.toml"))
            .expect("curated snapshot should parse");
        for cell in &snapshot.cells {
            assert!(
                motlie_models::CuratedBundle::CANONICAL_IDS.contains(&cell.bundle_id.as_str()),
                "snapshot cell `{}` bundle_id `{}` is not a CuratedBundle canonical id",
                cell.id,
                cell.bundle_id
            );
        }
    }

    #[test]
    fn every_curated_bundle_has_a_snapshot_cell() {
        let snapshot = load_snapshot(&repo_root().join("evals/snapshots/curated-v2-smoke.toml"))
            .expect("curated snapshot should parse");
        let covered: std::collections::BTreeSet<&str> = snapshot
            .cells
            .iter()
            .map(|cell| cell.bundle_id.as_str())
            .collect();
        let missing: Vec<&str> = motlie_models::CuratedBundle::CANONICAL_IDS
            .iter()
            .copied()
            .filter(|id| !covered.contains(id))
            .collect();
        assert!(
            missing.is_empty(),
            "CuratedBundle variants without a curated cell (completeness gap): {missing:?}"
        );
    }

    #[test]
    fn committed_result_bundle_ids_are_canonical() {
        // Data guard: proves committed records already use canonical bundle_ids
        // without editing any data. Walks evals/results/**/results.jsonl.
        let canonical: std::collections::BTreeSet<&str> =
            motlie_models::CuratedBundle::CANONICAL_IDS
                .iter()
                .copied()
                .collect();
        let results_root = repo_root().join("evals/results");
        let mut stack = vec![results_root];
        let mut checked = 0_u64;
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.file_name().and_then(|n| n.to_str()) != Some("results.jsonl") {
                    continue;
                }
                let contents = std::fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("failed to read `{}`: {err}", path.display()));
                for (line_no, line) in contents.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: serde_json::Value =
                        serde_json::from_str(line).unwrap_or_else(|err| {
                            panic!(
                                "invalid jsonl in `{}` line {}: {err}",
                                path.display(),
                                line_no + 1
                            )
                        });
                    let bundle_id = value
                        .get("coverage")
                        .and_then(|c| c.get("bundle_id"))
                        .and_then(|b| b.as_str())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing coverage.bundle_id in `{}` line {}",
                                path.display(),
                                line_no + 1
                            )
                        });
                    assert!(
                        canonical.contains(bundle_id),
                        "committed record `{}` line {} has non-canonical bundle_id `{}`",
                        path.display(),
                        line_no + 1,
                        bundle_id
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "data guard found no committed result records to check"
        );
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
