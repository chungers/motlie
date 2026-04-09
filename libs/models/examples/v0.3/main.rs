use anyhow::{Context, Result, bail, ensure};
use motlie_model::{
    ArtifactPolicy, ChatMessage, ChatRequest, ChatRole, ContentPart, QuantizationBits, StartOptions,
};
use motlie_models::{ModelSelector, chat::ChatModels, default_artifact_root};
use std::path::Path;
use std::time::Instant;

#[path = "../support.rs"]
mod support;

#[tokio::main]
async fn main() -> Result<()> {
    let mut chat_selector = None;
    let mut precision = None;
    let mut image_path = None;
    let mut download_artifacts = false;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if let Some(selector) = arg.strip_prefix("--chat=") {
            chat_selector = Some(selector.to_owned());
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
            "usage: cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example models_v0_3 -- \
             [--download-artifacts] [--chat=google/gemma4_e2b] [--precision=q4|q8|f32] [--image=/path/to/image] <prompt>"
        );
    }

    let quantization = match precision.as_deref() {
        Some("q4") | None => Some(QuantizationBits::Four),
        Some("q8") => Some(QuantizationBits::Eight),
        Some("f32") => None,
        Some(other) => bail!("unknown precision `{other}` — use q4, q8, or f32"),
    };

    let (selector_label, bundle_id, descriptor, bundle, path_kind) =
        if let Some(selector) = chat_selector {
            let model_selector: ModelSelector = format!("chat:{selector}")
                .parse()
                .with_context(|| format!("failed to parse model selector `chat:{selector}`"))?;
            (
                model_selector.to_string(),
                model_selector.bundle_id(),
                model_selector.descriptor(),
                model_selector.bundle(),
                "selector",
            )
        } else {
            let model = ChatModels::Gemma4E2B;
            (
                model.to_string(),
                model.bundle_id(),
                model.descriptor(),
                model.bundle(),
                "direct-enum",
            )
        };

    let artifact_root = default_artifact_root();
    let catalog = motlie_models::Catalog::with_defaults();

    println!("catalog-entry-count: {}", catalog.len());
    ensure!(
        catalog.len() == 1,
        "models_v0_3 must be built with exactly one curated bundle feature enabled"
    );

    println!("bundle-selector: {selector_label}");
    println!("resolution-path: {path_kind}");
    println!("bundle-id: {}", bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    support::print_process_snapshot(
        "process-before-start",
        &support::current_process_snapshot(),
    );
    println!(
        "quantization: {}",
        match quantization {
            Some(QuantizationBits::Four) => "ISQ Q4",
            Some(QuantizationBits::Eight) => "ISQ Q8",
            None => "F32 (none)",
        }
    );

    if download_artifacts {
        let summary =
            motlie_models::download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
                .with_context(|| {
                    format!("failed to download curated artifacts for `{bundle_id}`")
                })?;
        println!("downloaded-files: {}", summary.downloaded.len());
    } else {
        println!("downloaded-files: skipped (using existing local artifacts only)");
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

    println!("starting bundle (this includes ISQ quantization if enabled)...");
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
        .context("bundle startup should succeed from pre-downloaded local artifacts")?;
    let startup_elapsed = startup_at.elapsed();
    let startup_stats = startup_sampler.finish().await;
    println!(
        "startup-latency-ms: {:.0} ({:.1}s)",
        startup_elapsed.as_secs_f64() * 1000.0,
        startup_elapsed.as_secs_f64()
    );
    support::print_startup_stats(&startup_stats);
    support::print_process_snapshot("process-after-start", &support::current_process_snapshot());

    let chat = handle.chat().context("gemma4 bundle should expose chat")?;

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

    if let Some(image_path) = image_path {
        println!("\n--- image + text chat ---");
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
    } else {
        println!("\n--- image + text chat ---");
        println!("skipped: pass --image=/path/to/image to exercise the multimodal path");
    }

    handle
        .shutdown()
        .await
        .context("bundle shutdown should succeed")?;
    support::print_process_snapshot("process-after-shutdown", &support::current_process_snapshot());

    Ok(())
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
