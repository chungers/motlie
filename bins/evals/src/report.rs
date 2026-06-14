use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::metrics::CapabilityPerformanceMetrics;
use crate::result::{
    terminal_outcome, AcceptanceStatus, OutcomeReason, ResultRecord, TerminalOutcome,
    MIN_AGGREGATE_SCHEMA_VERSION, RESULT_SCHEMA_VERSION,
};
use crate::snapshot::{load_snapshot, EvalSnapshot};

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
            print!("{}", render_records_markdown(&records, &[input], None));
            Ok(())
        }
        ReportMode::Aggregate {
            aggregate,
            output,
            allow_invalid_records,
            snapshot,
        } => {
            let paths = expand_aggregate_paths(&aggregate)?;
            let mut records = Vec::new();
            for path in &paths {
                records.extend(read_aggregate_jsonl(path, !allow_invalid_records)?);
            }
            let snapshot = snapshot
                .as_ref()
                .map(|path| load_snapshot(path))
                .transpose()?;
            let markdown = render_records_markdown(&records, &paths, snapshot.as_ref());
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
    Input {
        input: PathBuf,
        format: String,
    },
    Aggregate {
        aggregate: String,
        output: PathBuf,
        allow_invalid_records: bool,
        snapshot: Option<PathBuf>,
    },
}

impl ReportOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut input = None;
        let mut format = "markdown".to_owned();
        let mut aggregate = None;
        let mut output = None;
        let mut allow_invalid_records = false;
        let mut snapshot = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--input" => input = Some(PathBuf::from(take_value(args, &mut index, "--input")?)),
                "--format" => format = take_value(args, &mut index, "--format")?,
                "--aggregate" => aggregate = Some(take_value(args, &mut index, "--aggregate")?),
                "--output" => {
                    output = Some(PathBuf::from(take_value(args, &mut index, "--output")?))
                }
                "--allow-invalid-records" => allow_invalid_records = true,
                "--snapshot" => {
                    snapshot = Some(PathBuf::from(take_value(args, &mut index, "--snapshot")?));
                }
                other => bail!("unknown evals report option `{other}`"),
            }
            index += 1;
        }

        let mode = match (input, aggregate, output) {
            (Some(input), None, None) => ReportMode::Input { input, format },
            (None, Some(aggregate), Some(output)) => ReportMode::Aggregate {
                aggregate,
                output,
                allow_invalid_records,
                snapshot,
            },
            _ => bail!(
                "evals report requires either --input <jsonl> [--format markdown] or --aggregate <path> --output <path>"
            ),
        };
        Ok(Self { mode })
    }
}

fn render_records_markdown(
    records: &[ResultRecord],
    inputs: &[PathBuf],
    snapshot: Option<&EvalSnapshot>,
) -> String {
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

    out.push_str("\n## Platform Notes\n\n");
    render_platform_notes(&mut out, records);

    out.push_str("\n## LLM Accelerator Comparison\n\n");
    render_llm_accelerator_comparison(&mut out, records);

    out.push_str("\n## Per-Cell Coverage\n\n");
    out.push_str("| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for record in records {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.coverage.cell_id,
            record.coverage.host_slug,
            record.coverage.arch,
            record.identity.run_id,
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

    out.push_str("\n## Latency Metrics\n\n");
    render_latency_metrics(&mut out, records);

    out.push_str("\n## Model x Capability\n\n");
    render_slice(
        &mut out,
        records,
        "model_family",
        "capability",
        |record| record.coverage.model_family.clone(),
        |record| record.coverage.capability.clone(),
    );

    out.push_str("\n## Capability x Profile\n\n");
    render_slice(
        &mut out,
        records,
        "capability",
        "profile",
        |record| record.coverage.capability.clone(),
        |record| record.coverage.profile.clone(),
    );

    out.push_str("\n## Capability x Depth\n\n");
    render_slice(
        &mut out,
        records,
        "capability",
        "depth",
        |record| record.coverage.capability.clone(),
        |record| record.coverage.depth.as_str().to_owned(),
    );

    out.push_str("\n## Backend x Profile\n\n");
    render_slice(
        &mut out,
        records,
        "backend",
        "profile",
        |record| record.coverage.backend.clone(),
        |record| record.coverage.profile.clone(),
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

    out.push_str("\n## Blocker Rollups\n\n");
    out.push_str("| reason | profile | count |\n|---|---|---:|\n");
    let mut blockers = BTreeMap::new();
    for record in records {
        if matches!(
            record.coverage.terminal_outcome,
            TerminalOutcome::Blocked | TerminalOutcome::Failed
        ) {
            let reason = record
                .coverage
                .reason
                .as_ref()
                .map(|reason| reason.as_str().to_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            *blockers
                .entry((reason, record.coverage.profile.clone()))
                .or_insert(0_u64) += 1;
        }
    }
    for ((reason, profile), count) in blockers {
        out.push_str(&format!("| `{reason}` | `{profile}` | {count} |\n"));
    }

    out.push_str("\n## Missing Coverage\n\n");
    render_missing_coverage(&mut out, records, snapshot);

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

fn render_platform_notes(out: &mut String, records: &[ResultRecord]) {
    let metal_mistral_mismatch = records.iter().any(|record| {
        record.coverage.profile == "apple-metal"
            && record.coverage.backend == "mistralrs"
            && record.accelerator.fallback_reason == Some(OutcomeReason::AcceleratorMismatch)
    });

    if metal_mistral_mismatch {
        out.push_str(
            "- `apple-metal` mistralrs rows with `accelerator_mismatch` are expected at this head: the curated `metal` profile feature does not currently enable the mistralrs Metal backend, and forced candle-metal probing is blocked by upstream M4 threadgroup-memory limits. These rows are honest CPU-fallback/blocked coverage, not an eval-framework failure.\n",
        );
    } else {
        out.push_str("- No platform-specific caveats detected in the input records.\n");
    }
}

/// LLM bundles run through `llama.cpp`/GGUF or `mistralrs`/HF; everything else
/// (ASR/TTS/embeddings) is excluded from the accelerator comparison.
fn is_llm_record(record: &ResultRecord) -> bool {
    matches!(record.coverage.backend.as_str(), "llama_cpp" | "mistralrs")
        && matches!(
            record.coverage.capability.as_str(),
            "chat" | "perf" | "tool_use"
        )
}

fn backend_family_label(backend: &str) -> &'static str {
    match backend {
        "llama_cpp" => "llama.cpp/GGUF",
        "mistralrs" => "mistralrs/HF",
        other => other_backend_family(other),
    }
}

fn other_backend_family(_backend: &str) -> &'static str {
    "other"
}

/// Accelerator axis for the comparison: the *target* the cell asked for
/// (derived from the profile), so a backend that refuses to forward to
/// CUDA/Metal still shows up under that column as blocked rather than vanishing.
const COMPARISON_ACCELERATORS: [&str; 3] = ["cpu", "cuda", "metal"];

#[derive(Default)]
struct AcceleratorPerf {
    decode_tps: Vec<f64>,
    ttft_ms: Vec<f64>,
}

fn perf_metrics(record: &ResultRecord) -> Option<&crate::metrics::PerfPerformanceMetrics> {
    match &record.performance.capability_metrics {
        CapabilityPerformanceMetrics::Perf(metrics) => Some(metrics),
        _ => None,
    }
}

/// "LLM Accelerator Comparison": a bundle × accelerator throughput/TTFT pivot
/// plus a backend-family viability rollup. Reproduced from records, not
/// hand-authored.
fn render_llm_accelerator_comparison(out: &mut String, records: &[ResultRecord]) {
    let llm_records: Vec<&ResultRecord> = records.iter().filter(|r| is_llm_record(r)).collect();
    if llm_records.is_empty() {
        out.push_str("No LLM (`llama.cpp`/GGUF or `mistralrs`/HF) records in this input set.\n");
        return;
    }

    render_llm_throughput(out, &llm_records);
    out.push_str("\n### Backend-family viability\n\n");
    render_llm_viability(out, &llm_records);
    out.push_str("\n### Build provenance\n\n");
    render_llm_provenance(out, &llm_records);
}

/// Build-SHA provenance per accelerator, from `identity.git_sha` of the records
/// whose numbers feed the tables above (on-target passing cells). Surfaces pin
/// mismatch honestly: an accelerator whose SHA set differs from the others is
/// prior-pin data. Fully data-driven — no pin is hardcoded.
fn render_llm_provenance(out: &mut String, llm_records: &[&ResultRecord]) {
    out.push_str(
        "Distinct build SHAs (`identity.git_sha`) of the on-target passing records backing the \
         numbers above, per accelerator. An accelerator whose SHA set differs from the others is \
         **prior-pin** data (pin mismatch — confirmatory only, not a fresh re-run).\n\n",
    );
    out.push_str("| accelerator | build SHAs |\n|---|---|\n");

    let mut by_accel: BTreeMap<&'static str, std::collections::BTreeSet<String>> = BTreeMap::new();
    for record in llm_records {
        if record.coverage.terminal_outcome != TerminalOutcome::Passed
            || record.coverage.resolved_accelerator != record.coverage.requested_accelerator
        {
            continue;
        }
        let accelerator = record.coverage.requested_accelerator.as_str();
        if !COMPARISON_ACCELERATORS.contains(&accelerator) {
            continue;
        }
        let sha = record
            .identity
            .git_sha
            .as_deref()
            .map(short_sha)
            .unwrap_or_else(|| "unknown".to_owned());
        by_accel.entry(accelerator).or_default().insert(sha);
    }

    if by_accel.is_empty() {
        out.push_str("| `none` | — |\n");
        return;
    }
    for accelerator in COMPARISON_ACCELERATORS {
        let shas = by_accel.get(accelerator);
        let rendered = shas
            .filter(|set| !set.is_empty())
            .map(|set| {
                set.iter()
                    .map(|sha| format!("`{sha}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "—".to_owned());
        out.push_str(&format!("| `{accelerator}` | {rendered} |\n"));
    }
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(8).collect()
}

/// Decode throughput (tok/s) and TTFT (ms), one row per LLM bundle, pivoted by
/// the cpu/cuda/metal target. Numbers come only from passing `perf` cells; a
/// `—` means no passing perf metric for that bundle × accelerator (see the
/// viability rollup below for whether the path was blocked vs simply not run).
fn render_llm_throughput(out: &mut String, llm_records: &[&ResultRecord]) {
    out.push_str(
        "Decode throughput (tok/s) and TTFT (ms) per LLM bundle, by target accelerator. \
         Values are from passing `perf` cells; `—` = no passing perf metric for that pairing.\n\n",
    );
    out.push_str(
        "| bundle | family | cpu tok/s | cpu ttft ms | cuda tok/s | cuda ttft ms | metal tok/s | metal ttft ms |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");

    // (bundle, family) -> accelerator -> accumulated perf metrics.
    let mut rows: BTreeMap<(String, &'static str), BTreeMap<&'static str, AcceleratorPerf>> =
        BTreeMap::new();
    for record in llm_records {
        if record.coverage.terminal_outcome != TerminalOutcome::Passed {
            continue;
        }
        // Only count a cell under an accelerator column if it actually ran there.
        // Blocked/fallback cells leave `resolved_accelerator` echoing the request,
        // so guard on resolved == requested to avoid e.g. a silent cpu-fallback
        // pass landing a cpu-speed number in the cuda column.
        if record.coverage.resolved_accelerator != record.coverage.requested_accelerator {
            continue;
        }
        let Some(metrics) = perf_metrics(record) else {
            continue;
        };
        let accelerator = record.coverage.requested_accelerator.as_str();
        if !COMPARISON_ACCELERATORS.contains(&accelerator) {
            continue;
        }
        let family = backend_family_label(&record.coverage.backend);
        let entry = rows
            .entry((record.coverage.bundle_id.clone(), family))
            .or_default()
            .entry(accelerator)
            .or_default();
        if let Some(tps) = metrics.mean_decode_tokens_per_second {
            entry.decode_tps.push(tps);
        }
        if let Some(ttft) = metrics.mean_ttft_first_token_ms {
            entry.ttft_ms.push(ttft);
        }
    }

    if rows.is_empty() {
        out.push_str("| `none` | `none` | — | — | — | — | — | — |\n");
        return;
    }

    for ((bundle, family), per_accel) in &rows {
        out.push_str(&format!("| `{bundle}` | {family} |"));
        for accelerator in COMPARISON_ACCELERATORS {
            let cell = per_accel.get(accelerator);
            let tps = cell.and_then(|c| mean_f64(&c.decode_tps));
            let ttft = cell.and_then(|c| mean_f64(&c.ttft_ms));
            out.push_str(&format!(
                " {} | {} |",
                format_decimal(tps, 1),
                format_decimal(ttft, 0)
            ));
        }
        out.push('\n');
    }
}

/// Backend-family × accelerator rollup: how many LLM cells resolved to the
/// requested accelerator and how the outcomes split. Surfaces "GGUF runs
/// everywhere; mistralrs does not forward to CUDA/Metal and is unusably slow on
/// CPU".
fn render_llm_viability(out: &mut String, llm_records: &[&ResultRecord]) {
    out.push_str(
        "`on_target` = passed cells that actually resolved to the requested accelerator (a \
         passed cell that silently fell back to CPU is counted in `passed` but not \
         `on_target`). `mean decode tok/s` is averaged over on-target passing `perf` cells.\n\n",
    );
    out.push_str(
        "| family | accelerator | cells | passed | on_target | blocked | failed | mean decode tok/s |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");

    #[derive(Default)]
    struct Viability {
        cells: u64,
        passed: u64,
        on_target: u64,
        blocked: u64,
        failed: u64,
        decode_tps: Vec<f64>,
    }

    let mut rollup: BTreeMap<(&'static str, &'static str), Viability> = BTreeMap::new();
    for record in llm_records {
        let accelerator = record.coverage.requested_accelerator.as_str();
        if !COMPARISON_ACCELERATORS.contains(&accelerator) {
            continue;
        }
        let on_target =
            record.coverage.resolved_accelerator == record.coverage.requested_accelerator;
        let family = backend_family_label(&record.coverage.backend);
        let entry = rollup.entry((family, accelerator)).or_default();
        entry.cells += 1;
        match record.coverage.terminal_outcome {
            TerminalOutcome::Passed => {
                entry.passed += 1;
                if on_target {
                    entry.on_target += 1;
                    if let Some(tps) =
                        perf_metrics(record).and_then(|m| m.mean_decode_tokens_per_second)
                    {
                        entry.decode_tps.push(tps);
                    }
                }
            }
            TerminalOutcome::Blocked => entry.blocked += 1,
            TerminalOutcome::Failed => entry.failed += 1,
            TerminalOutcome::Skipped => {}
        }
    }

    if rollup.is_empty() {
        out.push_str("| `none` | `none` | 0 | 0 | 0 | 0 | 0 | — |\n");
        return;
    }

    for ((family, accelerator), v) in &rollup {
        out.push_str(&format!(
            "| {family} | `{accelerator}` | {} | {} | {} | {} | {} | {} |\n",
            v.cells,
            v.passed,
            v.on_target,
            v.blocked,
            v.failed,
            format_decimal(mean_f64(&v.decode_tps), 1)
        ));
    }
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn format_decimal(value: Option<f64>, places: usize) -> String {
    value
        .map(|value| format!("{value:.places$}"))
        .unwrap_or_else(|| "—".to_owned())
}

fn render_latency_metrics(out: &mut String, records: &[ResultRecord]) {
    out.push_str("| cell | host | capability | ttft_first_token_ms | ttft_first_answer_token_ms | thinking_tokens_to_answer | completion_tokens | mean_ttft_first_token_ms | p95_ttft_first_token_ms | mean_ttft_first_answer_token_ms | p95_ttft_first_answer_token_ms | mean_transcription_latency_ms | p95_transcription_latency_ms | mean_ttfp_first_partial_ms | p95_ttfp_first_partial_ms | mean_synthesis_latency_ms | p95_synthesis_latency_ms | mean_ttfa_first_chunk_ms | p95_ttfa_first_chunk_ms | ttfp_first_partial_ms | ttfa_first_chunk_ms |\n");
    out.push_str(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );

    let mut row_count = 0_u64;
    for record in records {
        if !matches!(
            record.coverage.capability.as_str(),
            "chat" | "perf" | "asr" | "tts"
        ) {
            continue;
        }

        let metrics = latency_metric_cells(&record.performance.capability_metrics);
        if !metrics.has_any_value() {
            continue;
        }
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.coverage.cell_id,
            record.coverage.host_slug,
            record.coverage.capability,
            format_optional_u64(metrics.ttft_first_token_ms),
            format_optional_u64(metrics.ttft_first_answer_token_ms),
            format_optional_u64(metrics.thinking_tokens_to_answer),
            format_optional_u64(metrics.completion_tokens),
            format_optional_f64(metrics.mean_ttft_first_token_ms),
            format_optional_f64(metrics.p95_ttft_first_token_ms),
            format_optional_f64(metrics.mean_ttft_first_answer_token_ms),
            format_optional_f64(metrics.p95_ttft_first_answer_token_ms),
            format_optional_f64(metrics.mean_transcription_latency_ms),
            format_optional_f64(metrics.p95_transcription_latency_ms),
            format_optional_f64(metrics.mean_ttfp_first_partial_ms),
            format_optional_f64(metrics.p95_ttfp_first_partial_ms),
            format_optional_f64(metrics.mean_synthesis_latency_ms),
            format_optional_f64(metrics.p95_synthesis_latency_ms),
            format_optional_f64(metrics.mean_ttfa_first_chunk_ms),
            format_optional_f64(metrics.p95_ttfa_first_chunk_ms),
            format_optional_u64(metrics.ttfp_first_partial_ms),
            format_optional_u64(metrics.ttfa_first_chunk_ms),
        ));
        row_count += 1;
    }

    if row_count == 0 {
        out.push_str("| `none` | `none` | `none` | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |\n");
    }
}

#[derive(Default)]
struct LatencyMetricCells {
    ttft_first_token_ms: Option<u64>,
    ttft_first_answer_token_ms: Option<u64>,
    thinking_tokens_to_answer: Option<u64>,
    completion_tokens: Option<u64>,
    mean_ttft_first_token_ms: Option<f64>,
    p95_ttft_first_token_ms: Option<f64>,
    mean_ttft_first_answer_token_ms: Option<f64>,
    p95_ttft_first_answer_token_ms: Option<f64>,
    mean_transcription_latency_ms: Option<f64>,
    p95_transcription_latency_ms: Option<f64>,
    mean_ttfp_first_partial_ms: Option<f64>,
    p95_ttfp_first_partial_ms: Option<f64>,
    mean_synthesis_latency_ms: Option<f64>,
    p95_synthesis_latency_ms: Option<f64>,
    mean_ttfa_first_chunk_ms: Option<f64>,
    p95_ttfa_first_chunk_ms: Option<f64>,
    ttfp_first_partial_ms: Option<u64>,
    ttfa_first_chunk_ms: Option<u64>,
}

impl LatencyMetricCells {
    fn has_any_value(&self) -> bool {
        self.ttft_first_token_ms.is_some()
            || self.ttft_first_answer_token_ms.is_some()
            || self.thinking_tokens_to_answer.is_some()
            || self.completion_tokens.is_some()
            || self.mean_ttft_first_token_ms.is_some()
            || self.p95_ttft_first_token_ms.is_some()
            || self.mean_ttft_first_answer_token_ms.is_some()
            || self.p95_ttft_first_answer_token_ms.is_some()
            || self.mean_transcription_latency_ms.is_some()
            || self.p95_transcription_latency_ms.is_some()
            || self.mean_ttfp_first_partial_ms.is_some()
            || self.p95_ttfp_first_partial_ms.is_some()
            || self.mean_synthesis_latency_ms.is_some()
            || self.p95_synthesis_latency_ms.is_some()
            || self.mean_ttfa_first_chunk_ms.is_some()
            || self.p95_ttfa_first_chunk_ms.is_some()
            || self.ttfp_first_partial_ms.is_some()
            || self.ttfa_first_chunk_ms.is_some()
    }
}

fn latency_metric_cells(metrics: &CapabilityPerformanceMetrics) -> LatencyMetricCells {
    match metrics {
        CapabilityPerformanceMetrics::Chat(metrics) => LatencyMetricCells {
            ttft_first_token_ms: metrics.ttft_first_token_ms,
            ttft_first_answer_token_ms: metrics.ttft_first_answer_token_ms,
            thinking_tokens_to_answer: metrics.thinking_tokens_to_answer,
            completion_tokens: metrics.completion_tokens,
            ..Default::default()
        },
        CapabilityPerformanceMetrics::Perf(metrics) => LatencyMetricCells {
            mean_ttft_first_token_ms: metrics.mean_ttft_first_token_ms,
            p95_ttft_first_token_ms: metrics.p95_ttft_first_token_ms,
            mean_ttft_first_answer_token_ms: metrics.mean_ttft_first_answer_token_ms,
            p95_ttft_first_answer_token_ms: metrics.p95_ttft_first_answer_token_ms,
            ..Default::default()
        },
        CapabilityPerformanceMetrics::Asr(metrics) => LatencyMetricCells {
            mean_transcription_latency_ms: metrics.mean_transcription_latency_ms,
            p95_transcription_latency_ms: metrics.p95_transcription_latency_ms,
            mean_ttfp_first_partial_ms: metrics.mean_ttfp_first_partial_ms,
            p95_ttfp_first_partial_ms: metrics.p95_ttfp_first_partial_ms,
            ttfp_first_partial_ms: metrics.ttfp_first_partial_ms,
            ..Default::default()
        },
        CapabilityPerformanceMetrics::Tts(metrics) => LatencyMetricCells {
            mean_synthesis_latency_ms: metrics.mean_synthesis_latency_ms,
            p95_synthesis_latency_ms: metrics.p95_synthesis_latency_ms,
            mean_ttfa_first_chunk_ms: metrics.mean_ttfa_first_chunk_ms,
            p95_ttfa_first_chunk_ms: metrics.p95_ttfa_first_chunk_ms,
            ttfa_first_chunk_ms: metrics.ttfa_first_chunk_ms,
            ..Default::default()
        },
        _ => LatencyMetricCells::default(),
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_owned())
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "null".to_owned())
}

fn render_missing_coverage(
    out: &mut String,
    records: &[ResultRecord],
    snapshot: Option<&EvalSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        out.push_str(
            "Snapshot manifest not supplied; pass `--snapshot <path>` to list missing cells.\n",
        );
        return;
    };

    out.push_str(
        "| cell | bundle | capability | depth | profile | reason |\n|---|---|---|---|---|---|\n",
    );
    let observed = records
        .iter()
        .map(|record| {
            (
                record.coverage.cell_id.as_str(),
                record.coverage.profile.as_str(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut missing = 0_u64;
    for cell in &snapshot.cells {
        for profile in &cell.profiles {
            if !observed.contains(&(cell.id.as_str(), profile.as_str())) {
                missing += 1;
                out.push_str(&format!(
                    "| `{}` | `{}` | `{}` | `{}` | `{}` | `no_record` |\n",
                    cell.id,
                    cell.bundle_id,
                    cell.capability.as_str(),
                    cell.depth.as_str(),
                    profile
                ));
            }
        }
    }
    if missing == 0 {
        out.push_str("| `none` | `none` | `none` | `none` | `none` | `complete` |\n");
    }
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
    read_jsonl_with_validation(path, false, true)
}

fn read_aggregate_jsonl(path: &Path, strict: bool) -> Result<Vec<ResultRecord>> {
    read_jsonl_with_validation(path, true, strict)
}

fn read_jsonl_with_validation(
    path: &Path,
    validate_aggregate: bool,
    strict: bool,
) -> Result<Vec<ResultRecord>> {
    let file = File::open(path).with_context(|| format!("failed to open `{}`", path.display()))?;
    let reader = io::BufReader::new(file);
    let mut records = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read `{}`", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<ResultRecord>(&line).with_context(|| {
            format!(
                "failed to parse `{}` line {} as ResultRecord",
                path.display(),
                line_number + 1
            )
        })?;
        if validate_aggregate {
            if let Err(reason) = validate_aggregate_record(&record) {
                let message = format!(
                    "invalid aggregate input `{}` line {}: {}",
                    path.display(),
                    line_number + 1,
                    reason
                );
                if strict {
                    bail!(message);
                }
                eprintln!("aggregate-input-excluded: {message}");
                continue;
            }
        }
        records.push(record);
    }
    Ok(records)
}

fn validate_aggregate_record(record: &ResultRecord) -> std::result::Result<(), String> {
    if !(MIN_AGGREGATE_SCHEMA_VERSION..=RESULT_SCHEMA_VERSION).contains(&record.schema_version) {
        return Err(format!(
            "unsupported schema_version {}; expected {}..={}",
            record.schema_version, MIN_AGGREGATE_SCHEMA_VERSION, RESULT_SCHEMA_VERSION
        ));
    }

    let coverage = &record.coverage;
    for (name, value) in [
        ("snapshot_id", coverage.snapshot_id.as_str()),
        ("cell_id", coverage.cell_id.as_str()),
        ("capability", coverage.capability.as_str()),
        ("scenario_id", coverage.scenario_id.as_str()),
        ("bundle_id", coverage.bundle_id.as_str()),
        ("model_family", coverage.model_family.as_str()),
        ("checkpoint_format", coverage.checkpoint_format.as_str()),
        ("quantization", coverage.quantization.as_str()),
        ("backend", coverage.backend.as_str()),
        ("profile", coverage.profile.as_str()),
        ("host_id", coverage.host_id.as_str()),
        ("host_slug", coverage.host_slug.as_str()),
        ("arch", coverage.arch.as_str()),
    ] {
        if is_missing_coverage_value(value) {
            return Err(format!("coverage.{name} is missing or ad_hoc/unknown"));
        }
    }

    for (key, expected) in [
        ("bundle", coverage.bundle_id.as_str()),
        ("capability", coverage.capability.as_str()),
        ("depth", coverage.depth.as_str()),
        ("backend", coverage.backend.as_str()),
        ("checkpoint_format", coverage.checkpoint_format.as_str()),
        ("quantization", coverage.quantization.as_str()),
        ("profile", coverage.profile.as_str()),
    ] {
        match coverage.grouping_keys.get(key).map(String::as_str) {
            Some(value) if value == expected && !value.trim().is_empty() => {}
            Some(value) => {
                return Err(format!(
                    "coverage.grouping_keys.{key}={value:?} does not match {expected:?}"
                ));
            }
            None => return Err(format!("coverage.grouping_keys.{key} is missing")),
        }
    }

    if coverage.requested_accelerator != record.accelerator.requested_class {
        return Err("coverage/requested accelerator mismatch".to_owned());
    }
    if coverage.resolved_accelerator != record.accelerator.resolved_class {
        return Err("coverage/resolved accelerator mismatch".to_owned());
    }
    if record
        .accelerator
        .use_proof_source
        .as_deref()
        .is_none_or(|source| source.trim().is_empty())
    {
        return Err("accelerator.use_proof_source is missing".to_owned());
    }

    if terminal_outcome(&record.acceptance.overall_status) != coverage.terminal_outcome {
        return Err(
            "coverage terminal_outcome does not match acceptance.overall_status".to_owned(),
        );
    }

    match coverage.terminal_outcome {
        TerminalOutcome::Passed if coverage.reason.is_some() => {
            Err("passed records must not carry a terminal reason".to_owned())
        }
        TerminalOutcome::Failed | TerminalOutcome::Blocked | TerminalOutcome::Skipped
            if coverage.reason.is_none() =>
        {
            Err("non-passed records must carry a terminal reason".to_owned())
        }
        _ => Ok(()),
    }
}

fn is_missing_coverage_value(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty() || trimmed == "ad_hoc" || trimmed == "unknown"
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
    use crate::metrics::{
        AsrPerformanceMetrics, CapabilityPerformanceMetrics, PerfPerformanceMetrics,
        TtsPerformanceMetrics,
    };
    use crate::platform::PlatformSnapshot;
    use crate::result::{
        AcceleratorClass, AcceleratorSection, AcceptanceSection, ApplicabilityDecision,
        CoverageSection, EvalDepth, IdentitySection, ProfileSection, RuntimeSection,
        SelectionSection,
    };

    #[test]
    fn aggregate_markdown_includes_quant_and_accelerator_slices() {
        let record = test_record();
        let markdown = render_records_markdown(&[record], &[PathBuf::from("results.jsonl")], None);

        assert!(markdown.contains("Model x Quantization"));
        assert!(markdown.contains("Requested x Resolved Accelerator"));
        assert!(markdown.contains("q4_0"));
    }

    #[test]
    fn aggregate_markdown_includes_latency_columns() {
        let mut asr_record = test_record();
        asr_record.coverage.capability = "asr".to_owned();
        asr_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Asr(AsrPerformanceMetrics {
                mean_transcription_latency_ms: Some(100.0),
                p95_transcription_latency_ms: Some(120.0),
                mean_ttfp_first_partial_ms: Some(42.0),
                p95_ttfp_first_partial_ms: Some(48.0),
                ..Default::default()
            });

        let mut tts_record = test_record();
        tts_record.coverage.capability = "tts".to_owned();
        tts_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Tts(TtsPerformanceMetrics {
                mean_synthesis_latency_ms: Some(90.0),
                p95_synthesis_latency_ms: Some(110.0),
                mean_ttfa_first_chunk_ms: Some(17.0),
                p95_ttfa_first_chunk_ms: Some(22.0),
                ..Default::default()
            });

        let markdown = render_records_markdown(
            &[asr_record, tts_record],
            &[PathBuf::from("results.jsonl")],
            None,
        );

        assert!(markdown.contains("ttft_first_token_ms"));
        assert!(markdown.contains("mean_ttfp_first_partial_ms"));
        assert!(markdown.contains("p95_ttfp_first_partial_ms"));
        assert!(markdown.contains("mean_ttfa_first_chunk_ms"));
        assert!(markdown.contains("p95_ttfa_first_chunk_ms"));
        assert!(markdown.contains("ttfp_first_partial_ms"));
        assert!(markdown.contains("ttfa_first_chunk_ms"));
        assert!(markdown.contains("| 100.00 | 120.00 | 42.00 | 48.00 |"));
        assert!(markdown.contains("| 90.00 | 110.00 | 17.00 | 22.00 |"));
    }

    #[test]
    fn aggregate_latency_metrics_keeps_v4_single_shot_audio_columns() {
        let mut cold_asr_record = test_record();
        cold_asr_record.schema_version = 4;
        cold_asr_record.coverage.cell_id = "cold_asr".to_owned();
        cold_asr_record.coverage.capability = "asr".to_owned();
        cold_asr_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Asr(AsrPerformanceMetrics {
                ttfp_first_partial_ms: Some(110),
                ..Default::default()
            });

        let mut cold_tts_record = test_record();
        cold_tts_record.schema_version = 4;
        cold_tts_record.coverage.cell_id = "cold_tts".to_owned();
        cold_tts_record.coverage.capability = "tts".to_owned();
        cold_tts_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Tts(TtsPerformanceMetrics {
                ttfa_first_chunk_ms: Some(183),
                ..Default::default()
            });

        let mut markdown = String::new();
        render_latency_metrics(&mut markdown, &[cold_asr_record, cold_tts_record]);

        assert!(markdown.contains("cold_asr"));
        assert!(markdown.contains("cold_tts"));
        assert!(markdown.contains("| 110 | null |"));
        assert!(markdown.contains("| null | 183 |"));
    }

    #[test]
    fn aggregate_latency_metrics_skips_all_null_rows() {
        let mut measured_null_record = test_record();
        measured_null_record.coverage.cell_id = "measured_null".to_owned();
        measured_null_record.coverage.capability = "asr".to_owned();
        measured_null_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Asr(AsrPerformanceMetrics::default());

        let mut unmeasured_record = test_record();
        unmeasured_record.coverage.cell_id = "unmeasured".to_owned();
        unmeasured_record.coverage.capability = "chat".to_owned();
        unmeasured_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::NotMeasured;

        let mut measured_value_record = test_record();
        measured_value_record.coverage.cell_id = "measured_value".to_owned();
        measured_value_record.coverage.capability = "asr".to_owned();
        measured_value_record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Asr(AsrPerformanceMetrics {
                mean_ttfp_first_partial_ms: Some(42.0),
                ..Default::default()
            });

        let mut markdown = String::new();
        render_latency_metrics(
            &mut markdown,
            &[
                measured_null_record,
                unmeasured_record,
                measured_value_record,
            ],
        );

        assert!(!markdown.contains("measured_null"));
        assert!(!markdown.contains("unmeasured"));
        assert!(markdown.contains("measured_value"));
    }

    fn llm_perf_record(
        bundle: &str,
        backend: &str,
        accelerator: AcceleratorClass,
        decode_tps: Option<f64>,
        ttft_ms: Option<f64>,
    ) -> ResultRecord {
        let mut record = test_record();
        record.coverage.bundle_id = bundle.to_owned();
        record.coverage.cell_id = format!("{bundle}__perf");
        record.coverage.capability = "perf".to_owned();
        record.coverage.backend = backend.to_owned();
        record.coverage.requested_accelerator = accelerator;
        record.coverage.resolved_accelerator = accelerator;
        record.performance.capability_metrics =
            CapabilityPerformanceMetrics::Perf(PerfPerformanceMetrics {
                mean_decode_tokens_per_second: decode_tps,
                mean_ttft_first_token_ms: ttft_ms,
                ..Default::default()
            });
        record
    }

    #[test]
    fn llm_accelerator_comparison_pivots_throughput_and_viability() {
        let cpu = llm_perf_record(
            "qwen3_4b_gguf",
            "llama_cpp",
            AcceleratorClass::Cpu,
            Some(11.5),
            Some(540.0),
        );
        let cuda = llm_perf_record(
            "qwen3_4b_gguf",
            "llama_cpp",
            AcceleratorClass::Cuda,
            Some(95.2),
            Some(120.0),
        );

        // mistralrs requested CUDA but did not forward — blocked, no metric.
        let mut mistral_cuda =
            llm_perf_record("qwen3_4b", "mistralrs", AcceleratorClass::Cuda, None, None);
        mistral_cuda.coverage.resolved_accelerator = AcceleratorClass::Cpu;
        mistral_cuda.coverage.terminal_outcome = TerminalOutcome::Blocked;
        mistral_cuda.coverage.reason = Some(OutcomeReason::AcceleratorMismatch);

        let mut markdown = String::new();
        render_llm_accelerator_comparison(&mut markdown, &[cpu, cuda, mistral_cuda]);

        // Throughput pivot: one bundle row carries both cpu and cuda numbers.
        assert!(markdown.contains("`qwen3_4b_gguf` | llama.cpp/GGUF | 11.5 | 540 | 95.2 | 120 |"));
        // Viability rollup: mistralrs/HF on cuda is blocked, never on-target.
        // cols: cells | passed | on_target | blocked | failed | mean tok/s
        assert!(markdown.contains("Backend-family viability"));
        assert!(markdown.contains("| mistralrs/HF | `cuda` | 1 | 0 | 0 | 1 | 0 | — |"));
        assert!(markdown.contains("| llama.cpp/GGUF | `cuda` | 1 | 1 | 1 | 0 | 0 | 95.2 |"));
    }

    #[test]
    fn llm_accelerator_comparison_marks_prior_pin_provenance() {
        let mut cpu = llm_perf_record(
            "qwen3_4b_gguf",
            "llama_cpp",
            AcceleratorClass::Cpu,
            Some(14.3),
            Some(16000.0),
        );
        cpu.identity.git_sha = Some("e8f27b6e12c325f257aefa0e1f4714cce630330f".to_owned());

        // Metal data from a prior #399-cycle pin — different sha.
        let mut metal = llm_perf_record(
            "qwen3_4b_gguf",
            "llama_cpp",
            AcceleratorClass::Metal,
            Some(64.5),
            Some(774.0),
        );
        metal.identity.git_sha = Some("874c9f69abcdef0123456789".to_owned());

        let mut markdown = String::new();
        render_llm_accelerator_comparison(&mut markdown, &[cpu, metal]);

        assert!(markdown.contains("Build provenance"));
        assert!(markdown.contains("| `cpu` | `e8f27b6e` |"));
        assert!(markdown.contains("| `metal` | `874c9f69` |"));
        // cuda had no records -> dash.
        assert!(markdown.contains("| `cuda` | — |"));
    }

    #[test]
    fn llm_accelerator_comparison_handles_no_llm_records() {
        let mut asr = test_record();
        asr.coverage.capability = "asr".to_owned();
        asr.coverage.backend = "whisper".to_owned();

        let mut markdown = String::new();
        render_llm_accelerator_comparison(&mut markdown, &[asr]);

        assert!(markdown.contains("No LLM"));
    }

    #[test]
    fn aggregate_validation_accepts_snapshot_record() {
        let record = test_record();

        validate_aggregate_record(&record).expect("snapshot record should validate");
    }

    #[test]
    fn aggregate_validation_accepts_previous_schema_version() {
        let mut record = test_record();
        record.schema_version = MIN_AGGREGATE_SCHEMA_VERSION;

        validate_aggregate_record(&record).expect("v3 snapshot record should remain valid");
    }

    #[test]
    fn aggregate_validation_rejects_ad_hoc_coverage() {
        let mut record = test_record();
        record.coverage = CoverageSection::default();

        let err = validate_aggregate_record(&record).unwrap_err();

        assert!(err.contains("coverage.snapshot_id"));
    }

    #[test]
    fn aggregate_strict_read_fails_on_invalid_record() {
        let mut record = test_record();
        record.coverage = CoverageSection::default();
        let path = write_temp_record(&record, "strict-invalid");

        let err = read_aggregate_jsonl(&path, true).unwrap_err();

        let _ = std::fs::remove_file(path);
        assert!(err.to_string().contains("invalid aggregate input"));
        assert!(err.to_string().contains("coverage.snapshot_id"));
    }

    #[test]
    fn aggregate_tolerant_read_excludes_invalid_record() {
        let mut record = test_record();
        record.coverage = CoverageSection::default();
        let path = write_temp_record(&record, "tolerant-invalid");

        let records = read_aggregate_jsonl(&path, false).unwrap();

        let _ = std::fs::remove_file(path);
        assert!(records.is_empty());
    }

    #[test]
    fn aggregate_validation_rejects_grouping_key_mismatch() {
        let mut record = test_record();
        record
            .coverage
            .grouping_keys
            .insert("quantization".to_owned(), "default".to_owned());

        let err = validate_aggregate_record(&record).unwrap_err();

        assert!(err.contains("coverage.grouping_keys.quantization"));
    }

    fn write_temp_record(record: &ResultRecord, label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "motlie-evals-report-{label}-{}.jsonl",
            std::process::id()
        ));
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(record).unwrap()),
        )
        .unwrap();
        path
    }

    fn grouping_keys() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("bundle".to_owned(), "bundle".to_owned()),
            ("capability".to_owned(), "chat".to_owned()),
            ("depth".to_owned(), "smoke".to_owned()),
            ("backend".to_owned(), "llama_cpp".to_owned()),
            ("checkpoint_format".to_owned(), "gguf".to_owned()),
            ("quantization".to_owned(), "q4_0".to_owned()),
            ("profile".to_owned(), "dgx-spark".to_owned()),
        ])
    }

    fn test_record() -> ResultRecord {
        ResultRecord {
            schema_version: RESULT_SCHEMA_VERSION,
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
                grouping_keys: grouping_keys(),
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
