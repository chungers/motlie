/// Focused chat generation benchmark for CPU vs CUDA comparison.
///
/// Usage:
///   cargo run --release -p motlie-models \
///     --no-default-features --features model-qwen3-4b \
///     --example bench_chat -- [OPTIONS] <prompt or --input=file>
///
/// Options:
///   --iterations=N         Number of measured iterations (default: 5)
///   --precision=q4|q8|f32  Quantization (default: q4)
///   --model=qwen|gemma     Model selection (default: qwen)
///   --input=FILE           Read prompt from file instead of args
///
/// Environment variables:
///   MOTLIE_MODEL_FORCE_CPU=1  — force CPU even if built with CUDA
///   MOTLIE_PAGED_ATTN=1       — enable PagedAttention (CUDA only)
use anyhow::{Context, Result, bail};
use motlie_model::{
    ArtifactPolicy, ChatMessage, ChatModel, ChatRequest, ChatRole, QuantizationBits, StartOptions,
};
use motlie_models::default_artifact_root;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let mut iterations: usize = 5;
    let mut precision = None;
    let mut model_name = "qwen".to_string();
    let mut input_file = None;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if let Some(n) = arg.strip_prefix("--iterations=") {
            iterations = n.parse().context("invalid --iterations")?;
        } else if let Some(p) = arg.strip_prefix("--precision=") {
            precision = Some(p.to_owned());
        } else if let Some(m) = arg.strip_prefix("--model=") {
            model_name = m.to_owned();
        } else if let Some(f) = arg.strip_prefix("--input=") {
            input_file = Some(f.to_owned());
        } else {
            input_parts.push(arg);
        }
    }

    let prompt = if let Some(path) = input_file {
        std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read input file `{path}`"))?
    } else if !input_parts.is_empty() {
        input_parts.join(" ")
    } else {
        "Answer in exactly one sentence: what is 2+2?".to_string()
    };

    let quantization = match precision.as_deref() {
        Some("q4") | None => Some(QuantizationBits::Four),
        Some("q8") => Some(QuantizationBits::Eight),
        Some("f32") => None,
        Some(other) => bail!("unknown precision `{other}`"),
    };

    let artifact_root = default_artifact_root();

    let force_cpu = std::env::var("MOTLIE_MODEL_FORCE_CPU")
        .map(|v| matches!(v.as_str(), "1" | "true"))
        .unwrap_or(false);
    let paged_attn = std::env::var("MOTLIE_PAGED_ATTN")
        .map(|v| matches!(v.as_str(), "1" | "true"))
        .unwrap_or(false);

    let prompt_words = prompt.split_whitespace().count();
    let est_tokens = prompt_words * 13 / 10;

    println!("=== bench_chat ===");
    println!("model: {model_name}");
    println!("quantization: {}", match quantization {
        Some(QuantizationBits::Four) => "ISQ Q4",
        Some(QuantizationBits::Eight) => "ISQ Q8",
        None => "F32",
    });
    println!("iterations: {iterations}");
    println!("force-cpu: {force_cpu}");
    println!("paged-attn: {paged_attn}");
    println!("input-words: {prompt_words}");
    println!("input-est-tokens: ~{est_tokens}");

    let bundle: Box<dyn motlie_model::ModelBundle> = match model_name.as_str() {
        "qwen" => {
            #[cfg(feature = "model-qwen3-4b")]
            {
                motlie_models::chat::qwen3_4b::bundle()
            }
            #[cfg(not(feature = "model-qwen3-4b"))]
            bail!("model-qwen3-4b feature not enabled")
        }
        "gemma" => {
            #[cfg(feature = "model-gemma4-e2b")]
            {
                motlie_models::chat::gemma4_e2b::bundle()
            }
            #[cfg(not(feature = "model-gemma4-e2b"))]
            bail!("model-gemma4-e2b feature not enabled")
        }
        other => bail!("unknown model `{other}` — use qwen or gemma"),
    };

    // Startup
    println!("\n--- startup ---");
    let startup_at = Instant::now();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            quantization,
            ..Default::default()
        })
        .await
        .context("bundle startup failed")?;
    let startup_ms = startup_at.elapsed().as_secs_f64() * 1000.0;
    println!("startup-ms: {startup_ms:.0}");
    println!("rss-mib: {:.1}", current_rss_mib());

    let chat = handle.chat().context("chat capability missing")?;

    // Warmup run
    println!("\n--- warmup ---");
    let warmup = run_one(chat, &prompt).await;
    match &warmup {
        Ok((tokens, ms)) => {
            let tps = *tokens as f64 / (*ms / 1000.0);
            println!("warmup: {tokens} tokens in {ms:.0}ms ({tps:.1} tok/s)");
        }
        Err(e) => {
            println!("warmup: FAILED — {e}");
            println!("\n--- summary ---");
            println!("status: FAILED at warmup");
            println!("error: {e}");
            handle.shutdown().await.ok();
            return Ok(());
        }
    }

    // Measured iterations
    println!("\n--- measured ({iterations} iterations) ---");
    let mut total_tokens: usize = 0;
    let mut total_ms: f64 = 0.0;
    let mut failures = 0;

    for i in 0..iterations {
        match run_one(chat, &prompt).await {
            Ok((tokens, ms)) => {
                let tps = tokens as f64 / (ms / 1000.0);
                println!("  iter {}: {} tokens in {:.0}ms ({:.1} tok/s)", i + 1, tokens, ms, tps);
                total_tokens += tokens;
                total_ms += ms;
            }
            Err(e) => {
                println!("  iter {}: FAILED — {e}", i + 1);
                failures += 1;
            }
        }
    }

    let successful = iterations - failures;
    println!("\n--- summary ---");
    if let Ok((warmup_tok, warmup_ms)) = warmup {
        let warmup_tps = warmup_tok as f64 / (warmup_ms / 1000.0);
        println!("first-run: {warmup_tok} tokens, {warmup_ms:.0}ms, {warmup_tps:.1} tok/s");
    }
    if successful > 0 {
        let avg_tps = total_tokens as f64 / (total_ms / 1000.0);
        let avg_ms = total_ms / successful as f64;
        let avg_tokens = total_tokens as f64 / successful as f64;
        println!("steady-state ({successful}/{iterations} succeeded):");
        println!("  avg-tokens: {avg_tokens:.0}");
        println!("  avg-latency-ms: {avg_ms:.0}");
        println!("  avg-tok/s: {avg_tps:.1}");
        println!("  total-tokens: {total_tokens}");
        println!("  total-ms: {total_ms:.0}");
    }
    if failures > 0 {
        println!("failures: {failures}/{iterations}");
    }
    println!("rss-mib-final: {:.1}", current_rss_mib());

    if let Some(snapshot) = handle.metric_snapshot() {
        if let Some(rt) = &snapshot.runtime {
            println!("  peak-rss-mib: {}", rt.peak_resident_memory.map(|b| format!("{:.1}", b.0 as f64 / 1048576.0)).unwrap_or_else(|| "n/a".into()));
        }
        if let Some(tg) = &snapshot.text_generation {
            println!("  mistralrs-prompt-tps: {}", tg.avg_prompt_tokens_per_sec.map(|t| format!("{}", t.0)).unwrap_or_else(|| "n/a".into()));
            println!("  mistralrs-gen-tps: {}", tg.avg_generated_tokens_per_sec.map(|t| format!("{}", t.0)).unwrap_or_else(|| "n/a".into()));
        }
    }

    handle.shutdown().await.context("shutdown failed")?;
    Ok(())
}

async fn run_one(chat: &dyn ChatModel, prompt: &str) -> Result<(usize, f64)> {
    let started = Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: vec![
                ChatMessage::new(ChatRole::System, "Be concise."),
                ChatMessage::new(ChatRole::User, prompt),
            ],
            ..Default::default()
        })
        .await
        .context("chat generation failed")?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let token_count = response.content.split_whitespace().count();
    Ok((token_count, elapsed_ms))
}

fn current_rss_mib() -> f64 {
    use sysinfo::{ProcessesToUpdate, System, get_current_pid};
    let pid = match get_current_pid() {
        Ok(pid) => pid,
        Err(_) => return 0.0,
    };
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
    system
        .process(pid)
        .map(|p| p.memory() as f64 / 1048576.0)
        .unwrap_or(0.0)
}
