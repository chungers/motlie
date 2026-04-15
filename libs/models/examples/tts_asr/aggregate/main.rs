//! Aggregates JSON line results from all 4 TTS↔ASR pipeline runs.
//!
//! Usage:
//!   cat results_*.jsonl | cargo run -p motlie-models --example tts_asr_aggregate
//!
//! Or with files:
//!   cargo run -p motlie-models --example tts_asr_aggregate -- \
//!     --input results_piper_whisper.jsonl \
//!     --input results_piper_sherpa.jsonl \
//!     --input results_qwen3_whisper.jsonl \
//!     --input results_qwen3_sherpa.jsonl

use std::collections::BTreeMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

fn main() -> Result<()> {
    let args = parse_args()?;
    let results = load_results(&args.input_files)?;

    if results.is_empty() {
        eprintln!("No results to aggregate.");
        return Ok(());
    }

    let stats = compute_stats(&results);
    print_report(&stats);

    Ok(())
}

struct Args {
    input_files: Vec<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut input_files = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                input_files.push(PathBuf::from(
                    args.next().context("--input requires a path")?,
                ));
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    Ok(Args { input_files })
}

/// Only the fields the aggregator needs. Extra JSON keys are ignored via
/// `#[serde(deny_unknown_fields)]` being absent (default: allow unknown).
#[derive(Clone, Debug, Deserialize)]
struct PipelineResult {
    pipeline: String,
    category: String,
    wer: f64,
    tts_latency_ms: u64,
    asr_latency_ms: u64,
    total_latency_ms: u64,
}

fn load_results(files: &[PathBuf]) -> Result<Vec<PipelineResult>> {
    let mut results = Vec::new();

    if files.is_empty() {
        // Read from stdin
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.context("failed to read stdin line")?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(result) = serde_json::from_str::<PipelineResult>(line) {
                results.push(result);
            }
        }
    } else {
        for file in files {
            let content =
                std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(result) = serde_json::from_str::<PipelineResult>(line) {
                    results.push(result);
                }
            }
        }
    }

    Ok(results)
}

#[derive(Debug)]
struct PipelineStats {
    pipeline: String,
    sample_count: usize,
    mean_wer: f64,
    median_wer: f64,
    p95_wer: f64,
    max_wer: f64,
    mean_total_latency_ms: f64,
    median_total_latency_ms: f64,
    mean_tts_latency_ms: f64,
    mean_asr_latency_ms: f64,
    category_stats: BTreeMap<String, CategoryStats>,
}

#[derive(Debug)]
struct CategoryStats {
    count: usize,
    mean_wer: f64,
    mean_latency_ms: f64,
}

fn compute_stats(results: &[PipelineResult]) -> Vec<PipelineStats> {
    let mut by_pipeline: BTreeMap<String, Vec<&PipelineResult>> = BTreeMap::new();
    for r in results {
        by_pipeline.entry(r.pipeline.clone()).or_default().push(r);
    }

    by_pipeline
        .into_iter()
        .map(|(pipeline, samples)| {
            let n = samples.len();
            let mut wers: Vec<f64> = samples.iter().map(|s| s.wer).collect();
            wers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mean_wer = wers.iter().sum::<f64>() / n as f64;
            let median_wer = percentile(&wers, 50.0);
            let p95_wer = percentile(&wers, 95.0);
            let max_wer = wers.last().copied().unwrap_or(0.0);

            let total_latencies: Vec<f64> =
                samples.iter().map(|s| s.total_latency_ms as f64).collect();
            let mean_total = total_latencies.iter().sum::<f64>() / n as f64;
            let mut sorted_latencies = total_latencies.clone();
            sorted_latencies
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_total = percentile(&sorted_latencies, 50.0);

            let mean_tts =
                samples.iter().map(|s| s.tts_latency_ms as f64).sum::<f64>() / n as f64;
            let mean_asr =
                samples.iter().map(|s| s.asr_latency_ms as f64).sum::<f64>() / n as f64;

            // Per-category
            let mut by_cat: BTreeMap<String, Vec<&PipelineResult>> = BTreeMap::new();
            for s in &samples {
                by_cat.entry(s.category.clone()).or_default().push(s);
            }

            let category_stats = by_cat
                .into_iter()
                .map(|(cat, cat_samples)| {
                    let cn = cat_samples.len();
                    let cat_mean_wer =
                        cat_samples.iter().map(|s| s.wer).sum::<f64>() / cn as f64;
                    let cat_mean_lat = cat_samples
                        .iter()
                        .map(|s| s.total_latency_ms as f64)
                        .sum::<f64>()
                        / cn as f64;
                    (
                        cat,
                        CategoryStats {
                            count: cn,
                            mean_wer: cat_mean_wer,
                            mean_latency_ms: cat_mean_lat,
                        },
                    )
                })
                .collect();

            PipelineStats {
                pipeline,
                sample_count: n,
                mean_wer,
                median_wer,
                p95_wer,
                max_wer,
                mean_total_latency_ms: mean_total,
                median_total_latency_ms: median_total,
                mean_tts_latency_ms: mean_tts,
                mean_asr_latency_ms: mean_asr,
                category_stats,
            }
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_report(stats: &[PipelineStats]) {
    println!("=== TTS↔ASR Pipeline Comparison Report ===\n");

    // Summary table
    println!("{:<20} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Pipeline", "Samples", "Mean WER", "Med WER", "P95 WER", "Max WER", "Mean Lat(ms)", "Med Lat(ms)");
    println!("{}", "-".repeat(102));

    for s in stats {
        println!(
            "{:<20} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>12.0} {:>12.0}",
            s.pipeline,
            s.sample_count,
            s.mean_wer,
            s.median_wer,
            s.p95_wer,
            s.max_wer,
            s.mean_total_latency_ms,
            s.median_total_latency_ms,
        );
    }

    // Per-category breakdown
    println!("\n=== Per-Category Breakdown ===\n");

    let categories = ["short", "medium", "long", "paragraph"];
    for cat in categories {
        println!("Category: {cat}");
        println!("{:<20} {:>8} {:>10} {:>12}", "Pipeline", "Samples", "Mean WER", "Mean Lat(ms)");
        println!("{}", "-".repeat(54));
        for s in stats {
            if let Some(cs) = s.category_stats.get(cat) {
                println!(
                    "{:<20} {:>8} {:>10.3} {:>12.0}",
                    s.pipeline, cs.count, cs.mean_wer, cs.mean_latency_ms,
                );
            }
        }
        println!();
    }

    // Latency breakdown
    println!("=== Latency Breakdown (TTS vs ASR) ===\n");
    println!("{:<20} {:>12} {:>12} {:>12}", "Pipeline", "TTS(ms)", "ASR(ms)", "Total(ms)");
    println!("{}", "-".repeat(60));
    for s in stats {
        println!(
            "{:<20} {:>12.0} {:>12.0} {:>12.0}",
            s.pipeline, s.mean_tts_latency_ms, s.mean_asr_latency_ms, s.mean_total_latency_ms,
        );
    }

    // Rankings
    println!("\n=== Rankings ===\n");

    if let Some(best_wer) = stats.iter().min_by(|a, b| {
        a.mean_wer.partial_cmp(&b.mean_wer).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("Best WER (accuracy):  {} (mean WER = {:.3})", best_wer.pipeline, best_wer.mean_wer);
    }

    if let Some(best_lat) = stats.iter().min_by(|a, b| {
        a.mean_total_latency_ms.partial_cmp(&b.mean_total_latency_ms).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("Best latency (speed): {} (mean = {:.0}ms)", best_lat.pipeline, best_lat.mean_total_latency_ms);
    }

    println!("\nNote: Rankings are based on aggregate metrics across all categories.");
    println!("For CPU-only scenarios, Piper TTS pipelines are recommended (lightweight, no GPU required).");
    println!("For CUDA scenarios, Qwen3-TTS pipelines may produce higher quality but require ONNX Runtime with CUDA EP.");
}
