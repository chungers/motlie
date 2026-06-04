use anyhow::{bail, ensure, Context, Result};
use motlie_model::{
    ArtifactPolicy, BundleHandle, CapabilityKind, ChatMessage, ChatModel, ChatRequest, ChatRole,
    CompletionModel, ContentPart, QuantizationBits, StartOptions,
};
use motlie_models::{chat::ChatModels, default_artifact_root, quantization_label_gguf};
use std::path::Path;
use std::time::Instant;

#[path = "../support.rs"]
mod support;

#[tokio::main]
async fn main() -> Result<()> {
    let mut precision = None;
    let mut image_path = None;
    let mut download_artifacts = false;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if let Some(p) = arg.strip_prefix("--precision=") {
            precision = Some(p.to_owned());
        } else if let Some(path) = arg.strip_prefix("--image=") {
            image_path = Some(path.to_owned());
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    if input.trim().is_empty() {
        bail!(
            "usage: cargo run -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf --example chat_multimodal_qwen3_6_27b -- \
             [--download-artifacts] [--precision=q4|q5|q8|fp8] [--image=/path/to/image] <prompt>"
        );
    }

    let quantization = parse_precision(precision.as_deref())?;

    let model = ChatModels::Qwen3_6_27B_Gguf;
    let selector_label = model.to_string();
    let bundle_id = model.bundle_id();
    let descriptor = model.descriptor();
    let bundle = model.bundle();

    let artifact_root = default_artifact_root();
    let catalog = motlie_models::Catalog::with_defaults();

    println!("catalog-entry-count: {}", catalog.len());
    ensure!(
        catalog.len() == 1,
        "chat_multimodal_qwen3_6_27b must be built with exactly one curated bundle feature enabled"
    );

    println!("backend: llama.cpp (GGUF)");
    println!("bundle-selector: {selector_label}");
    println!("bundle-id: {}", bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    println!("gpu-offload: {}", gpu_offload_summary());
    support::print_process_snapshot("process-before-start", &support::current_process_snapshot());
    println!(
        "requested-precision: {}",
        requested_precision_label(quantization)
    );

    if download_artifacts {
        let summary =
            motlie_models::download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
                .with_context(|| {
                    format!("failed to download curated GGUF artifacts for `{bundle_id}`")
                })?;
        println!("downloaded-files: {}", summary.downloaded.len());
    } else {
        println!("downloaded-files: skipped (using existing local GGUF artifacts only)");
    }

    println!("display-name: {}", descriptor.display_name);
    println!("family: {:?}", descriptor.family);
    println!("backend: {:?}", descriptor.backend);
    println!("capabilities:");
    for capability in descriptor.capability_descriptors() {
        println!(
            "  - kind={:?} input={:?} output={:?} interaction={:?} summary={}",
            capability.kind,
            capability.inputs,
            capability.outputs,
            capability.interaction,
            capability.summary
        );
    }

    println!("starting bundle (loading GGUF weights)...");
    let startup_sampler = support::StartupSampler::spawn("startup");
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
        .context("bundle startup should succeed from pre-downloaded local GGUF artifacts")?;
    let startup_elapsed = startup_at.elapsed();
    let startup_stats = startup_sampler.finish().await;
    println!(
        "startup-latency-ms: {:.0} ({:.1}s)",
        startup_elapsed.as_secs_f64() * 1000.0,
        startup_elapsed.as_secs_f64()
    );
    support::print_startup_stats(&startup_stats);
    support::print_process_snapshot("process-after-start", &support::current_process_snapshot());
    support::print_model_metrics("model-metrics-after-start", handle.metric_snapshot());
    println!(
        "resolved-quantization: {}",
        quantization_label_gguf(handle.descriptor().resolved_quantization)
    );

    let chat = handle
        .chat()
        .context("Qwen3.6 GGUF bundle should expose chat")?;

    println!("\n--- text-only chat ---");
    let started_at = Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: vec![
                ChatMessage::text(ChatRole::System, "Be concise. Answer in one paragraph."),
                ChatMessage::text(ChatRole::User, &input),
            ],
            ..Default::default()
        })
        .await
        .context("text-only chat generation should succeed")?;
    let latency = started_at.elapsed();

    println!("prompt: {input}");
    println!("response: {}", response.content);
    println!("latency-ms: {:.2}", latency.as_secs_f64() * 1000.0);
    support::print_process_snapshot(
        "process-after-text-chat",
        &support::current_process_snapshot(),
    );
    support::print_model_metrics("model-metrics-after-text-chat", handle.metric_snapshot());

    println!("\n--- completion ---");
    let completion = handle
        .completion()
        .context("Qwen3.6 GGUF bundle should expose completion")?;
    let completion_started_at = Instant::now();
    let completion_response = completion
        .complete(motlie_model::CompletionRequest {
            prompt: format!("Complete this sentence: {input}"),
            ..Default::default()
        })
        .await
        .context("completion should succeed")?;
    let completion_latency = completion_started_at.elapsed();

    println!("completion-prompt: Complete this sentence: {input}");
    println!("completion-response: {}", completion_response.content);
    println!(
        "completion-latency-ms: {:.2}",
        completion_latency.as_secs_f64() * 1000.0
    );
    support::print_process_snapshot(
        "process-after-completion",
        &support::current_process_snapshot(),
    );
    support::print_model_metrics("model-metrics-after-completion", handle.metric_snapshot());

    if let Some(image_path) = image_path {
        println!("\n--- image + text chat ---");
        if descriptor.capabilities.supports(CapabilityKind::Vision) {
            let image_bytes = std::fs::read(&image_path)
                .with_context(|| format!("failed to read image `{image_path}`"))?;
            let media_type = infer_media_type(&image_path)
                .with_context(|| format!("unsupported image type for `{image_path}`"))?;
            let image_started_at = Instant::now();
            let image_response = chat
                .generate(ChatRequest {
                    messages: vec![
                        ChatMessage::text(
                            ChatRole::System,
                            "Be concise. Describe visible details and answer the question.",
                        ),
                        ChatMessage::with_parts(
                            ChatRole::User,
                            vec![
                                ContentPart::image(image_bytes, media_type),
                                ContentPart::text(input.clone()),
                            ],
                        ),
                    ],
                    ..Default::default()
                })
                .await
                .context("image+text chat generation should succeed")?;
            let image_latency = image_started_at.elapsed();

            println!("image-path: {image_path}");
            println!("image-response: {}", image_response.content);
            println!(
                "image-latency-ms: {:.2}",
                image_latency.as_secs_f64() * 1000.0
            );
            support::print_process_snapshot(
                "process-after-image-chat",
                &support::current_process_snapshot(),
            );
            support::print_model_metrics(
                "model-metrics-after-image-chat",
                handle.metric_snapshot(),
            );
        } else {
            println!(
                "skipped: this curated Qwen3.6 GGUF bundle is text-only; the core ChatRequest type accepts images, but llama.cpp mmproj/image execution is not wired yet"
            );
        }
    } else {
        println!("\n--- image + text chat ---");
        println!("skipped: pass --image=/path/to/image to exercise this path if vision is enabled");
    }

    handle
        .shutdown()
        .await
        .context("bundle shutdown should succeed")?;
    support::print_process_snapshot(
        "process-after-shutdown",
        &support::current_process_snapshot(),
    );

    Ok(())
}

fn parse_precision(value: Option<&str>) -> Result<Option<QuantizationBits>> {
    match value {
        Some("q4") => Ok(Some(QuantizationBits::Four)),
        Some("q5") => Ok(Some(QuantizationBits::Five)),
        Some("q8") => Ok(Some(QuantizationBits::Eight)),
        None => Ok(None),
        Some("fp8") => bail!(
            "--precision=fp8 is reserved for CUDA builds once a curated FP8 GGUF artifact exists; current Qwen3.6 GGUF artifacts support q4, q5, and q8"
        ),
        Some(other) => bail!("unknown precision `{other}`; use q4, q5, q8, or fp8"),
    }
}

fn requested_precision_label(quantization: Option<QuantizationBits>) -> &'static str {
    match quantization {
        Some(QuantizationBits::Four) => "q4",
        Some(QuantizationBits::Five) => "q5",
        Some(QuantizationBits::Eight) => "q8",
        Some(QuantizationBits::FloatEight) => "fp8",
        None => "bundle recommended",
    }
}

fn gpu_offload_summary() -> String {
    if std::env::var("MOTLIE_MODEL_FORCE_CPU").ok().as_deref() == Some("1") {
        return "forced CPU (MOTLIE_MODEL_FORCE_CPU=1)".into();
    }
    if let Ok(layers) = std::env::var("MOTLIE_MODEL_GPU_LAYERS") {
        return format!("explicit GPU layers (MOTLIE_MODEL_GPU_LAYERS={layers})");
    }
    if cfg!(feature = "llama-cpp-cuda") {
        "CUDA build; llama.cpp default full GPU offload".into()
    } else {
        "default llama.cpp GPU-layer policy; no CUDA feature enabled".into()
    }
}

fn infer_media_type(path: &str) -> Result<&'static str> {
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => Ok("image/jpeg"),
        Some("png") => Ok("image/png"),
        Some("webp") => Ok("image/webp"),
        Some("gif") => Ok("image/gif"),
        Some("bmp") => Ok("image/bmp"),
        Some(other) => bail!("unsupported image extension `{other}`"),
        None => bail!("missing image extension"),
    }
}
