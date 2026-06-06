/// Focused chat generation benchmark for CPU vs CUDA comparison.
///
/// Usage:
///   cargo run --release -p motlie-models \
///     --no-default-features --features model-qwen3-4b \
///     --example bench_chat -- [OPTIONS] <prompt or --input=file>
///
/// Options:
///   --iterations=N         Number of measured iterations (default: 5)
///   --precision=q4|q8|f32|f16
///                          Quantization (default: q4; GGUF uses f16 for no quantization)
///   --model=qwen|gemma|gemma4-12b|gemma4-12b-gguf
///                          Model selection (default: qwen; `gemma` is Gemma 4 E2B)
///   --input=FILE           Read prompt from file instead of args
///
/// Environment variables:
///   MOTLIE_MODEL_FORCE_CPU=1  — force CPU even if built with CUDA
///   MOTLIE_PAGED_ATTN_CONTEXT=N — enable PagedAttention with N-token context budget (CUDA only)
use anyhow::{Context, Result, bail};
use motlie_model::{
    ArtifactPolicy, BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole, QuantizationBits,
    StartOptions,
};
use motlie_models::{
    CuratedBundle, default_artifact_root, quantization_label_gguf, quantization_label_isq,
};
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
        Some("f32") | Some("f16") => None,
        Some(other) => bail!("unknown precision `{other}`"),
    };

    let artifact_root = default_artifact_root();

    let force_cpu = std::env::var("MOTLIE_MODEL_FORCE_CPU")
        .map(|v| matches!(v.as_str(), "1" | "true"))
        .unwrap_or(false);
    let prompt_words = prompt.split_whitespace().count();
    let est_tokens = prompt_words * 13 / 10;

    let quantization_label = if matches!(model_name.as_str(), "gemma4-12b-gguf" | "gemma4_12b_gguf")
    {
        quantization_label_gguf(quantization)
    } else {
        quantization_label_isq(quantization)
    };

    println!("=== bench_chat ===");
    println!("model: {model_name}");
    println!("quantization: {quantization_label}");
    println!("iterations: {iterations}");
    println!("force-cpu: {force_cpu}");
    let pa_context = std::env::var("MOTLIE_PAGED_ATTN_CONTEXT").ok();
    println!(
        "paged-attn-context: {}",
        pa_context.as_deref().unwrap_or("disabled")
    );
    println!("input-words: {prompt_words}");
    println!("input-est-tokens: ~{est_tokens}");

    let bundle: CuratedBundle = match model_name.as_str() {
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
        "gemma4-12b" | "gemma4_12b" => {
            #[cfg(feature = "model-gemma4-12b")]
            {
                motlie_models::chat::gemma4_12b::bundle()
            }
            #[cfg(not(feature = "model-gemma4-12b"))]
            bail!("model-gemma4-12b feature not enabled")
        }
        "gemma4-12b-gguf" | "gemma4_12b_gguf" => {
            #[cfg(feature = "model-gemma4-12b-gguf")]
            {
                motlie_models::chat::gemma4_12b_gguf::bundle()
            }
            #[cfg(not(feature = "model-gemma4-12b-gguf"))]
            bail!("model-gemma4-12b-gguf feature not enabled")
        }
        other => bail!("unknown model `{other}` — use qwen, gemma, gemma4-12b, or gemma4-12b-gguf"),
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
        Ok((words, ms)) => {
            println!("warmup: {words} words in {ms:.0}ms (e2e-latency={ms:.0}ms)");
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
    let mut total_words: usize = 0;
    let mut total_ms: f64 = 0.0;
    let mut failures = 0;

    for i in 0..iterations {
        match run_one(chat, &prompt).await {
            Ok((words, ms)) => {
                println!("  iter {}: {} words in {:.0}ms", i + 1, words, ms);
                total_words += words;
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
    println!("note: per-iteration counts are output *words* (whitespace-split), not model tokens.");
    println!("      latency is end-to-end request time (prefill + full decode), not true TTFT.");
    println!("      real token throughput is reported below from mistralrs internal metrics.");
    if let Ok((warmup_words, warmup_ms)) = warmup {
        println!("first-run: {warmup_words} words, e2e-latency={warmup_ms:.0}ms");
    }
    if successful > 0 {
        let avg_ms = total_ms / successful as f64;
        let avg_words = total_words as f64 / successful as f64;
        println!("steady-state ({successful}/{iterations} succeeded):");
        println!("  avg-words: {avg_words:.0}");
        println!("  avg-latency-ms: {avg_ms:.0}");
        println!("  total-words: {total_words}");
        println!("  total-ms: {total_ms:.0}");
    }
    if failures > 0 {
        println!("failures: {failures}/{iterations}");
    }
    println!("rss-mib-final: {:.1}", current_rss_mib());

    if let Some(snapshot) = handle.metric_snapshot() {
        if let Some(rt) = &snapshot.runtime {
            println!(
                "  peak-rss-mib: {}",
                rt.peak_resident_memory
                    .map(|b| format!("{:.1}", b.0 as f64 / 1048576.0))
                    .unwrap_or_else(|| "n/a".into())
            );
        }
        if let Some(tg) = &snapshot.text_generation {
            println!(
                "  mistralrs-prompt-tps: {}",
                tg.avg_prompt_tokens_per_sec
                    .map(|t| format!("{}", t.0))
                    .unwrap_or_else(|| "n/a".into())
            );
            println!(
                "  mistralrs-gen-tps: {}",
                tg.avg_generated_tokens_per_sec
                    .map(|t| format!("{}", t.0))
                    .unwrap_or_else(|| "n/a".into())
            );
        }
    }

    handle.shutdown().await.context("shutdown failed")?;
    Ok(())
}

async fn run_one<C: ChatModel + ?Sized>(chat: &C, prompt: &str) -> Result<(usize, f64)> {
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
