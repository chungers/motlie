use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::result::{AcceptanceStatus, ResultRecord, TerminalOutcome};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputSink {
    Stdout,
    JsonlFile(PathBuf),
}

impl OutputSink {
    pub fn emit(&self, record: &ResultRecord) -> Result<()> {
        match self {
            Self::Stdout => {
                let mut stdout = io::stdout().lock();
                serde_json::to_writer(&mut stdout, record)?;
                writeln!(stdout)?;
                Ok(())
            }
            Self::JsonlFile(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("failed to create report directory `{}`", parent.display())
                    })?;
                }
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("failed to open `{}`", path.display()))?;
                serde_json::to_writer(&mut file, record)?;
                writeln!(file)?;
                Ok(())
            }
        }
    }
}

pub fn run_report(args: &[String]) -> Result<()> {
    let options = ReportOptions::parse(args)?;
    match options.mode {
        ReportMode::Input { input, format } => {
            if format != "markdown" {
                bail!("unsupported report format `{format}`");
            }
            let records = read_jsonl(&input)?;
            print!("{}", render_records_markdown(&records, &[input]));
            Ok(())
        }
        ReportMode::Aggregate { aggregate, output } => {
            let paths = expand_aggregate_paths(&aggregate)?;
            let mut records = Vec::new();
            for path in &paths {
                records.extend(read_jsonl(path)?);
            }
            let markdown = render_records_markdown(&records, &paths);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create aggregate report dir `{}`",
                        parent.display()
                    )
                })?;
            }
            fs::write(&output, markdown)
                .with_context(|| format!("failed to write `{}`", output.display()))?;
            println!("aggregate-report: {}", output.display());
            Ok(())
        }
    }
}

pub fn summarize_status(records: &[ResultRecord]) -> AcceptanceStatus {
    if records
        .iter()
        .any(|record| record.acceptance.overall_status == AcceptanceStatus::Fail)
    {
        return AcceptanceStatus::Fail;
    }

    if records
        .iter()
        .any(|record| record.acceptance.overall_status == AcceptanceStatus::Blocked)
    {
        return AcceptanceStatus::Blocked;
    }

    if records
        .iter()
        .all(|record| record.acceptance.overall_status == AcceptanceStatus::Pass)
    {
        AcceptanceStatus::Pass
    } else {
        AcceptanceStatus::NotMeasured
    }
}

#[derive(Debug)]
struct ReportOptions {
    mode: ReportMode,
}

#[derive(Debug)]
enum ReportMode {
    Input { input: PathBuf, format: String },
    Aggregate { aggregate: String, output: PathBuf },
}

impl ReportOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut input = None;
        let mut format = "markdown".to_owned();
        let mut aggregate = None;
        let mut output = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--input" => input = Some(PathBuf::from(take_value(args, &mut index, "--input")?)),
                "--format" => format = take_value(args, &mut index, "--format")?,
                "--aggregate" => aggregate = Some(take_value(args, &mut index, "--aggregate")?),
                "--output" => {
                    output = Some(PathBuf::from(take_value(args, &mut index, "--output")?))
                }
                other => bail!("unknown evals report option `{other}`"),
            }
            index += 1;
        }

        let mode = match (input, aggregate, output) {
            (Some(input), None, None) => ReportMode::Input { input, format },
            (None, Some(aggregate), Some(output)) => ReportMode::Aggregate { aggregate, output },
            _ => bail!(
                "evals report requires either --input <jsonl> [--format markdown] or --aggregate <path> --output <path>"
            ),
        };
        Ok(Self { mode })
    }
}

fn render_records_markdown(records: &[ResultRecord], inputs: &[PathBuf]) -> String {
    let mut out = String::new();
    out.push_str("# Curated Eval Coverage Report\n\n");
    out.push_str(&format!("- input files: `{}`\n", inputs.len()));
    out.push_str(&format!("- records: `{}`\n", records.len()));
    out.push_str(&format!(
        "- overall: `{}`\n\n",
        status_label(&summarize_status(records))
    ));

    out.push_str("## Outcome Summary\n\n");
    out.push_str("| outcome | count |\n|---|---:|\n");
    for (outcome, count) in count_by(records, |record| {
        outcome_label(record.coverage.terminal_outcome)
    }) {
        out.push_str(&format!("| `{outcome}` | {count} |\n"));
    }

    out.push_str("\n## Per-Cell Coverage\n\n");
    out.push_str("| cell | bundle | capability | depth | profile | requested | resolved | outcome | reason |\n|---|---|---|---|---|---|---|---|---|\n");
    for record in records {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.coverage.cell_id,
            record.coverage.bundle_id,
            record.coverage.capability,
            record.coverage.depth.as_str(),
            record.coverage.profile,
            record.coverage.requested_accelerator.as_str(),
            record.coverage.resolved_accelerator.as_str(),
            outcome_label(record.coverage.terminal_outcome),
            record
                .coverage
                .reason
                .as_ref()
                .map(|reason| reason.as_str())
                .unwrap_or("none")
        ));
    }

    out.push_str("\n## Model x Capability\n\n");
    render_slice(
        &mut out,
        records,
        "model_family",
        "capability",
        |record| record.coverage.model_family.clone(),
        |record| record.coverage.capability.clone(),
    );

    out.push_str("\n## Model x Quantization x Backend/Profile/Depth\n\n");
    out.push_str("| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |\n|---|---|---|---|---|---:|---:|---:|---:|\n");
    let mut rows: BTreeMap<(String, String, String, String, String), OutcomeCounts> =
        BTreeMap::new();
    for record in records {
        let key = (
            record.coverage.model_family.clone(),
            record.coverage.quantization.clone(),
            record.coverage.backend.clone(),
            record.coverage.profile.clone(),
            record.coverage.depth.as_str().to_owned(),
        );
        rows.entry(key)
            .or_default()
            .add(record.coverage.terminal_outcome);
    }
    for ((model, quant, backend, profile, depth), counts) in rows {
        out.push_str(&format!(
            "| `{model}` | `{quant}` | `{backend}` | `{profile}` | `{depth}` | {} | {} | {} | {} |\n",
            counts.passed, counts.failed, counts.blocked, counts.skipped
        ));
    }

    out.push_str("\n## Requested x Resolved Accelerator\n\n");
    render_slice(
        &mut out,
        records,
        "requested",
        "resolved",
        |record| record.coverage.requested_accelerator.as_str().to_owned(),
        |record| record.coverage.resolved_accelerator.as_str().to_owned(),
    );

    out.push_str("\n## Metric Gaps\n\n");
    out.push_str("| metric | reason | source | count |\n|---|---|---|---:|\n");
    let mut gaps = BTreeMap::new();
    for record in records {
        for gap in record
            .performance
            .unavailable
            .iter()
            .chain(record.resources.unavailable_metrics.iter())
        {
            *gaps
                .entry((
                    gap.metric.clone(),
                    gap.reason.clone(),
                    gap.source.clone().unwrap_or_else(|| "unknown".to_owned()),
                ))
                .or_insert(0_u64) += 1;
        }
    }
    for ((metric, reason, source), count) in gaps {
        out.push_str(&format!(
            "| `{metric}` | `{reason}` | `{source}` | {count} |\n"
        ));
    }

    out.push_str("\n## Inputs\n\n");
    for path in inputs {
        out.push_str(&format!("- `{}`\n", path.display()));
    }

    out
}

fn render_slice<A, B, FA, FB>(
    out: &mut String,
    records: &[ResultRecord],
    left_name: &str,
    right_name: &str,
    left: FA,
    right: FB,
) where
    FA: Fn(&ResultRecord) -> A,
    FB: Fn(&ResultRecord) -> B,
    A: Ord + ToString,
    B: Ord + ToString,
{
    out.push_str(&format!(
        "| {left_name} | {right_name} | passed | failed | blocked | skipped |\n|---|---|---:|---:|---:|---:|\n"
    ));
    let mut rows: BTreeMap<(String, String), OutcomeCounts> = BTreeMap::new();
    for record in records {
        rows.entry((left(record).to_string(), right(record).to_string()))
            .or_default()
            .add(record.coverage.terminal_outcome);
    }
    for ((left, right), counts) in rows {
        out.push_str(&format!(
            "| `{left}` | `{right}` | {} | {} | {} | {} |\n",
            counts.passed, counts.failed, counts.blocked, counts.skipped
        ));
    }
}

#[derive(Default)]
struct OutcomeCounts {
    passed: u64,
    failed: u64,
    blocked: u64,
    skipped: u64,
}

impl OutcomeCounts {
    fn add(&mut self, outcome: TerminalOutcome) {
        match outcome {
            TerminalOutcome::Passed => self.passed += 1,
            TerminalOutcome::Failed => self.failed += 1,
            TerminalOutcome::Blocked => self.blocked += 1,
            TerminalOutcome::Skipped => self.skipped += 1,
        }
    }
}

fn count_by<F>(records: &[ResultRecord], mut key: F) -> BTreeMap<String, u64>
where
    F: FnMut(&ResultRecord) -> &'static str,
{
    let mut counts = BTreeMap::new();
    for record in records {
        *counts.entry(key(record).to_owned()).or_insert(0) += 1;
    }
    counts
}

fn read_jsonl(path: &Path) -> Result<Vec<ResultRecord>> {
    let file = File::open(path).with_context(|| format!("failed to open `{}`", path.display()))?;
    let reader = io::BufReader::new(file);
    let mut records = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read `{}`", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str::<ResultRecord>(&line).with_context(|| {
                format!(
                    "failed to parse `{}` line {} as ResultRecord",
                    path.display(),
                    line_number + 1
                )
            })?,
        );
    }
    Ok(records)
}

fn expand_aggregate_paths(pattern: &str) -> Result<Vec<PathBuf>> {
    let path = PathBuf::from(pattern);
    if !pattern.contains("**") {
        if path.is_dir() {
            let mut paths = Vec::new();
            collect_results_jsonl(&path, &mut paths)?;
            paths.sort();
            return Ok(paths);
        }
        return Ok(vec![path]);
    }

    let prefix = pattern
        .split("**")
        .next()
        .unwrap_or(".")
        .trim_end_matches('/');
    let root = if prefix.is_empty() { "." } else { prefix };
    let mut paths = Vec::new();
    collect_results_jsonl(Path::new(root), &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_results_jsonl(root: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in
        fs::read_dir(root).with_context(|| format!("failed to read `{}`", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_results_jsonl(&path, paths)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("results.jsonl") {
            paths.push(path);
        }
    }
    Ok(())
}

fn outcome_label(outcome: TerminalOutcome) -> &'static str {
    match outcome {
        TerminalOutcome::Passed => "passed",
        TerminalOutcome::Failed => "failed",
        TerminalOutcome::Blocked => "blocked",
        TerminalOutcome::Skipped => "skipped",
    }
}

fn status_label(status: &AcceptanceStatus) -> &'static str {
    match status {
        AcceptanceStatus::Pass => "pass",
        AcceptanceStatus::Fail => "fail",
        AcceptanceStatus::Blocked => "blocked",
        AcceptanceStatus::Skipped => "skipped",
        AcceptanceStatus::NotMeasured => "not_measured",
        AcceptanceStatus::NotApplicable => "not_applicable",
    }
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
    use crate::result::{
        AcceleratorClass, AcceleratorSection, AcceptanceSection, ApplicabilityDecision,
        CoverageSection, EvalDepth, IdentitySection, ProfileSection, RuntimeSection,
        SelectionSection,
    };
    use crate::{metrics::CapabilityPerformanceMetrics, platform::PlatformSnapshot};

    #[test]
    fn aggregate_markdown_includes_quant_and_accelerator_slices() {
        let record = test_record();
        let markdown = render_records_markdown(&[record], &[PathBuf::from("results.jsonl")]);

        assert!(markdown.contains("Model x Quantization"));
        assert!(markdown.contains("Requested x Resolved Accelerator"));
        assert!(markdown.contains("q4_0"));
    }

    fn test_record() -> ResultRecord {
        ResultRecord {
            schema_version: 2,
            identity: IdentitySection {
                run_id: "run".to_owned(),
                git_sha: None,
                git_branch: None,
                command_line: Vec::new(),
                host_id: Some("host".to_owned()),
                host_slug: Some("host".to_owned()),
            },
            coverage: CoverageSection {
                snapshot_id: "snap".to_owned(),
                cell_id: "cell".to_owned(),
                depth: EvalDepth::Smoke,
                capability: "chat".to_owned(),
                scenario_id: "chat_smoke".to_owned(),
                bundle_id: "bundle".to_owned(),
                model_family: "qwen3".to_owned(),
                checkpoint_format: "gguf".to_owned(),
                quantization: "q4_0".to_owned(),
                backend: "llama_cpp".to_owned(),
                profile: "dgx-spark".to_owned(),
                host_id: "host".to_owned(),
                host_slug: "host".to_owned(),
                arch: "aarch64".to_owned(),
                requested_accelerator: AcceleratorClass::Cuda,
                resolved_accelerator: AcceleratorClass::Cuda,
                applicability: ApplicabilityDecision::Applicable,
                terminal_outcome: TerminalOutcome::Passed,
                reason: None,
                grouping_keys: BTreeMap::new(),
            },
            selection: SelectionSection {
                bundle_id: "bundle".to_owned(),
                selector: None,
                backend: Some("llama_cpp".to_owned()),
                checkpoint_format: Some("gguf".to_owned()),
                artifact_quantization: Some("q4_0".to_owned()),
                artifact_snapshot: None,
                artifact_patterns: Vec::new(),
                artifact_files: Vec::new(),
                scenario: "chat_smoke".to_owned(),
                capability: "chat".to_owned(),
            },
            profile: ProfileSection {
                name: "dgx-spark".to_owned(),
            },
            platform: PlatformSnapshot {
                os: Some("linux".to_owned()),
                arch: Some("aarch64".to_owned()),
                target_triple: None,
                hostname: Some("host".to_owned()),
                host_id: Some("host".to_owned()),
                host_slug: Some("host".to_owned()),
                total_memory_bytes: None,
                available_memory_bytes: None,
                total_swap_bytes: None,
                free_swap_bytes: None,
                gpu_backend: Some("nvidia".to_owned()),
                gpus: Vec::new(),
                accelerator_metadata: BTreeMap::new(),
                unavailable: Vec::new(),
            },
            accelerator: AcceleratorSection {
                requested_class: AcceleratorClass::Cuda,
                resolved_class: AcceleratorClass::Cuda,
                selected_devices: Vec::new(),
                backend_mode: Some("cuda".to_owned()),
                offload: Some("gpu_layers=999".to_owned()),
                driver_versions: BTreeMap::new(),
                fallback_reason: None,
                use_proof_source: Some("backend".to_owned()),
            },
            runtime: RuntimeSection {
                cargo_features: Vec::new(),
                build_profile: None,
                quantization: Some("q4_0".to_owned()),
                runtime_precision: None,
                artifact_root: String::new(),
                download_artifacts: false,
                context_length: None,
                gpu_layers: None,
                child_build: None,
                budgets: BTreeMap::new(),
                env: BTreeMap::new(),
            },
            performance: crate::metrics::PerformanceMetrics {
                capability_metrics: CapabilityPerformanceMetrics::NotMeasured,
                ..Default::default()
            },
            resources: Default::default(),
            acceptance: AcceptanceSection {
                behavior_status: AcceptanceStatus::Pass,
                performance_status: AcceptanceStatus::Pass,
                resource_status: AcceptanceStatus::Pass,
                accelerator_status: AcceptanceStatus::Pass,
                overall_status: AcceptanceStatus::Pass,
                failure_reason: None,
                assertions: Vec::new(),
                gates: Vec::new(),
            },
        }
    }
}
