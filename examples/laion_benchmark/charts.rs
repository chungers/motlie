//! Chart generation for LAION benchmark results
//!
//! Generates charts matching the article's visualizations:
//! 1. Recall@5 vs Database Size (for different ef_search values)
//! 2. Recall@k vs ef_search
//! 3. Latency vs Database Size
//! 4. Recall vs Latency tradeoff (Pareto curve)

use anyhow::Result;
use plotters::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Chart colors matching common visualization styles
const COLORS: &[RGBColor] = &[
    RGBColor(31, 119, 180),   // Blue
    RGBColor(255, 127, 14),   // Orange
    RGBColor(44, 160, 44),    // Green
    RGBColor(214, 39, 40),    // Red
    RGBColor(148, 103, 189),  // Purple
    RGBColor(140, 86, 75),    // Brown
];

/// Parsed result from CSV
#[derive(Debug, Clone)]
struct ResultRow {
    scale: usize,
    ef_search: usize,
    strategy: String,
    recall_1: f64,
    recall_5: f64,
    recall_10: f64,
    recall_15: f64,
    recall_20: f64,
    latency_avg_ms: f64,
    latency_p50_ms: f64,
    qps: f64,
}

/// Generate all charts from CSV results
pub fn generate_all_charts(results_dir: &PathBuf) -> Result<()> {
    let csv_path = results_dir.join("laion_benchmark_results.csv");

    if !csv_path.exists() {
        anyhow::bail!("Results CSV not found: {:?}\nRun experiments first with --run-all", csv_path);
    }

    let results = load_results_csv(&csv_path)?;
    println!("Loaded {} result rows from CSV", results.len());

    // Generate each chart type
    generate_recall_vs_scale_chart(&results, results_dir)?;
    generate_recall_vs_ef_search_chart(&results, results_dir)?;
    generate_latency_vs_scale_chart(&results, results_dir)?;
    generate_pareto_chart(&results, results_dir)?;

    println!("\nAll charts generated in: {:?}", results_dir);
    Ok(())
}

/// Load results from CSV file
fn load_results_csv(path: &PathBuf) -> Result<Vec<ResultRow>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut results = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 14 {
            continue;
        }

        results.push(ResultRow {
            scale: parts[0].parse().unwrap_or(0),
            ef_search: parts[1].parse().unwrap_or(0),
            strategy: parts[2].to_string(),
            recall_1: parts[3].parse().unwrap_or(0.0),
            recall_5: parts[4].parse().unwrap_or(0.0),
            recall_10: parts[5].parse().unwrap_or(0.0),
            recall_15: parts[6].parse().unwrap_or(0.0),
            recall_20: parts[7].parse().unwrap_or(0.0),
            latency_avg_ms: parts[8].parse().unwrap_or(0.0),
            latency_p50_ms: parts[9].parse().unwrap_or(0.0),
            qps: parts[12].parse().unwrap_or(0.0),
        });
    }

    Ok(results)
}

/// Chart 1: Recall@5 vs Database Size (matching article Fig 1)
fn generate_recall_vs_scale_chart(results: &[ResultRow], results_dir: &PathBuf) -> Result<()> {
    let path = results_dir.join("recall_vs_scale.svg");

    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Recall@5 vs Database Size", ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(40_000f64..210_000f64, 0.0f64..1.0f64)?;

    chart
        .configure_mesh()
        .x_desc("Database Size (vectors)")
        .y_desc("Recall@5")
        .x_label_formatter(&|x| format!("{}K", (*x as usize) / 1000))
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .draw()?;

    // Group by ef_search and strategy
    let ef_values: Vec<usize> = vec![10, 20, 40, 80, 160];

    for (i, &ef) in ef_values.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];

        // HNSW data points for this ef_search
        let hnsw_points: Vec<(f64, f64)> = results
            .iter()
            .filter(|r| r.strategy == "HNSW-Cosine" && r.ef_search == ef)
            .map(|r| (r.scale as f64, r.recall_5))
            .collect();

        if !hnsw_points.is_empty() {
            chart
                .draw_series(LineSeries::new(hnsw_points.clone(), color.stroke_width(2)))?
                .label(format!("HNSW ef={}", ef))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

            chart.draw_series(hnsw_points.iter().map(|&(x, y)| Circle::new((x, y), 4, color.filled())))?;
        }
    }

    // Flat baseline (dashed)
    let flat_points: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.strategy == "Flat")
        .map(|r| (r.scale as f64, r.recall_5))
        .collect();

    if !flat_points.is_empty() {
        chart
            .draw_series(LineSeries::new(flat_points.clone(), BLACK.stroke_width(2)))?
            .label("Flat (brute-force)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

        chart.draw_series(flat_points.iter().map(|&(x, y)| Circle::new((x, y), 4, BLACK.filled())))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: {:?}", path);
    Ok(())
}

/// Chart 2: Recall@k vs ef_search (matching article Fig 2)
fn generate_recall_vs_ef_search_chart(results: &[ResultRow], results_dir: &PathBuf) -> Result<()> {
    let path = results_dir.join("recall_vs_ef_search.svg");

    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Use largest scale for this chart
    let max_scale = results.iter().map(|r| r.scale).max().unwrap_or(200_000);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Recall@k vs ef_search ({}K vectors)", max_scale / 1000), ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..170f64, 0.0f64..1.0f64)?;

    chart
        .configure_mesh()
        .x_desc("ef_search")
        .y_desc("Recall")
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .draw()?;

    let k_values = vec![1, 5, 10, 15, 20];

    for (i, &k) in k_values.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];

        let points: Vec<(f64, f64)> = results
            .iter()
            .filter(|r| r.strategy == "HNSW-Cosine" && r.scale == max_scale)
            .map(|r| {
                let recall = match k {
                    1 => r.recall_1,
                    5 => r.recall_5,
                    10 => r.recall_10,
                    15 => r.recall_15,
                    20 => r.recall_20,
                    _ => 0.0,
                };
                (r.ef_search as f64, recall)
            })
            .collect();

        if !points.is_empty() {
            chart
                .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))?
                .label(format!("Recall@{}", k))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

            chart.draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 4, color.filled())))?;
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: {:?}", path);
    Ok(())
}

/// Chart 3: Latency vs Database Size (matching article Fig 3)
fn generate_latency_vs_scale_chart(results: &[ResultRow], results_dir: &PathBuf) -> Result<()> {
    let path = results_dir.join("latency_vs_scale.svg");

    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max latency for y-axis
    let max_latency = results.iter().map(|r| r.latency_avg_ms).fold(0.0f64, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latency vs Database Size", ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(40_000f64..210_000f64, 0.0f64..max_latency)?;

    chart
        .configure_mesh()
        .x_desc("Database Size (vectors)")
        .y_desc("Latency (ms)")
        .x_label_formatter(&|x| format!("{}K", (*x as usize) / 1000))
        .draw()?;

    // HNSW with ef_search=80 (typical value)
    let hnsw_points: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.strategy == "HNSW-Cosine" && r.ef_search == 80)
        .map(|r| (r.scale as f64, r.latency_avg_ms))
        .collect();

    if !hnsw_points.is_empty() {
        chart
            .draw_series(LineSeries::new(hnsw_points.clone(), COLORS[0].stroke_width(2)))?
            .label("HNSW (ef=80)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], COLORS[0].stroke_width(2)));

        chart.draw_series(hnsw_points.iter().map(|&(x, y)| Circle::new((x, y), 4, COLORS[0].filled())))?;
    }

    // Flat baseline
    let flat_points: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.strategy == "Flat")
        .map(|r| (r.scale as f64, r.latency_avg_ms))
        .collect();

    if !flat_points.is_empty() {
        chart
            .draw_series(LineSeries::new(flat_points.clone(), COLORS[1].stroke_width(2)))?
            .label("Flat (brute-force)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], COLORS[1].stroke_width(2)));

        chart.draw_series(flat_points.iter().map(|&(x, y)| Circle::new((x, y), 4, COLORS[1].filled())))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: {:?}", path);
    Ok(())
}

/// Chart 4: Recall vs Latency tradeoff (Pareto curve)
fn generate_pareto_chart(results: &[ResultRow], results_dir: &PathBuf) -> Result<()> {
    let path = results_dir.join("recall_latency_tradeoff.svg");

    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_latency = results
        .iter()
        .filter(|r| r.strategy != "Flat")
        .map(|r| r.latency_avg_ms)
        .fold(0.0f64, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Recall@10 vs Latency Tradeoff", ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..max_latency, 0.0f64..1.0f64)?;

    chart
        .configure_mesh()
        .x_desc("Latency (ms)")
        .y_desc("Recall@10")
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .draw()?;

    // Group by scale
    let scales: Vec<usize> = vec![50_000, 100_000, 150_000, 200_000];

    for (i, &scale) in scales.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];

        let points: Vec<(f64, f64)> = results
            .iter()
            .filter(|r| r.strategy == "HNSW-Cosine" && r.scale == scale)
            .map(|r| (r.latency_avg_ms, r.recall_10))
            .collect();

        if !points.is_empty() {
            chart
                .draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 5, color.filled())))?
                .label(format!("{}K vectors", scale / 1000))
                .legend(move |(x, y)| Circle::new((x + 10, y), 5, color.filled()));
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: {:?}", path);
    Ok(())
}
