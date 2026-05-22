use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use motlie_model::ModelMetricSnapshot;

#[derive(Clone, Debug, Default)]
pub struct ProcessSnapshot {
    pub pid: Option<u32>,
    pub rss_mib: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct StartupStats {
    pub start_rss_mib: Option<f64>,
    pub peak_rss_mib: Option<f64>,
    pub end_rss_mib: Option<f64>,
}

pub struct StartupSampler {
    stop: Arc<AtomicBool>,
    task: thread::JoinHandle<StartupStats>,
}

pub fn current_process_snapshot() -> ProcessSnapshot {
    use sysinfo::{ProcessesToUpdate, System, get_current_pid};

    let pid = get_current_pid().ok();
    let rss_mib = pid.and_then(|pid| {
        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
        let process = system.process(pid)?;
        Some(process.memory() as f64 / (1024.0 * 1024.0))
    });

    ProcessSnapshot {
        pid: pid.map(|pid| pid.as_u32()),
        rss_mib,
    }
}

pub fn print_process_snapshot(label: &str, snapshot: &ProcessSnapshot) {
    match (snapshot.pid, snapshot.rss_mib) {
        (Some(pid), Some(rss_mib)) => {
            println!("{label}: pid={pid} rss-mib={rss_mib:.1}");
        }
        (Some(pid), None) => {
            println!("{label}: pid={pid} rss-mib=unavailable");
        }
        _ => println!("{label}: unavailable"),
    }
}

pub fn print_model_metrics(label: &str, snapshot: Option<ModelMetricSnapshot>) {
    let Some(snapshot) = snapshot else {
        println!("{label}: unavailable");
        return;
    };

    println!("{label}:");
    if let Some(runtime) = snapshot.runtime {
        println!(
            "  runtime: resident-bytes={:?} peak-resident-bytes={:?} request-count={:?} last-latency={:?} max-latency={:?} avg-latency={:?}",
            runtime.resident_memory,
            runtime.peak_resident_memory,
            runtime.request_count,
            runtime.last_latency,
            runtime.max_latency,
            runtime.avg_latency
        );
    } else {
        println!("  runtime: unavailable");
    }

    if let Some(text) = snapshot.text_generation {
        println!(
            "  text-generation: total-prompt-tokens={:?} total-generated-tokens={:?} total-tokens={:?} avg-prompt-tps={:?} avg-generated-tps={:?}",
            text.total_prompt_tokens,
            text.total_generated_tokens,
            text.total_tokens,
            text.avg_prompt_tokens_per_sec,
            text.avg_generated_tokens_per_sec
        );
    }

    if let Some(embeddings) = snapshot.embeddings {
        println!(
            "  embeddings: request-count={:?} input-count={:?}",
            embeddings.request_count, embeddings.input_count
        );
    }
}

impl StartupSampler {
    pub fn spawn(label: &'static str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_task = Arc::clone(&stop);

        let task = thread::spawn(move || {
            let start = current_process_snapshot().rss_mib;
            let mut peak = start;
            let sample_started_at = Instant::now();

            while !stop_for_task.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(5));
                let current = current_process_snapshot().rss_mib;
                peak = max_rss(peak, current);
                if let Some(rss_mib) = current {
                    println!(
                        "{label}-sample: elapsed-s={:.1} rss-mib={:.1}",
                        sample_started_at.elapsed().as_secs_f64(),
                        rss_mib
                    );
                }
            }

            let end = current_process_snapshot().rss_mib;
            StartupStats {
                start_rss_mib: start,
                peak_rss_mib: max_rss(peak, end),
                end_rss_mib: end,
            }
        });

        Self { stop, task }
    }

    pub async fn finish(self) -> StartupStats {
        self.stop.store(true, Ordering::Relaxed);
        self.task.join().unwrap_or_default()
    }
}

pub fn print_startup_stats(stats: &StartupStats) {
    match (stats.start_rss_mib, stats.peak_rss_mib, stats.end_rss_mib) {
        (Some(start), Some(peak), Some(end)) => {
            println!(
                "startup-memory-mib: start={start:.1} peak={peak:.1} end={end:.1} delta={:.1}",
                end - start
            );
        }
        _ => println!("startup-memory-mib: unavailable"),
    }
}

fn max_rss(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => None,
    }
}
