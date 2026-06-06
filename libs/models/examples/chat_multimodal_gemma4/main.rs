#[cfg(not(any(
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e4b",
    feature = "model-gemma4-12b"
)))]
compile_error!(
    "chat_multimodal_gemma4 requires at least one of model-gemma4-e2b, model-gemma4-e4b, or model-gemma4-12b"
);

use anyhow::{bail, ensure, Context, Result};
use motlie_model::{
    ArtifactPolicy, BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole, ContentPart,
    QuantizationBits, StartOptions,
};
use motlie_model_mistral::MistralMultimodalSpec;
use motlie_models::{
    chat::ChatModels, default_artifact_root, quantization_label_isq, BundleDescriptor,
    CuratedBundle,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "../support.rs"]
mod support;
#[path = "../tool_demo_support.rs"]
mod tool_demo_support;

#[tokio::main]
async fn main() -> Result<()> {
    let mut precision = None;
    let mut model_name = None;
    let mut image_path = None;
    let mut artifact_root = None;
    let mut download_artifacts = false;
    let mut tool_demo = false;
    let mut tool_demo_only = false;
    let mut input_parts = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if arg == "--tool-demo" {
            tool_demo = true;
        } else if arg == "--tool-demo-only" {
            tool_demo = true;
            tool_demo_only = true;
        } else if arg == "--artifact-root" {
            artifact_root = Some(PathBuf::from(
                args.next()
                    .context("--artifact-root requires a path argument")?,
            ));
        } else if let Some(path) = arg.strip_prefix("--artifact-root=") {
            artifact_root = Some(PathBuf::from(path));
        } else if arg == "--model" {
            model_name = Some(args.next().context("--model requires a model name")?);
        } else if let Some(model) = arg.strip_prefix("--model=") {
            model_name = Some(model.to_owned());
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
            "usage: cargo run -p motlie-models --no-default-features --features model-gemma4-e2b[,model-gemma4-e4b,model-gemma4-12b] --example chat_multimodal_gemma4 -- \
             [--model=gemma4-e2b|gemma4-e4b|gemma4-12b] [--download-artifacts] [--artifact-root <path>] [--tool-demo|--tool-demo-only] [--precision=q4|q8|f32] [--image=/path/to/image] <prompt>"
        );
    }

    let selected_model = select_model(model_name.as_deref())?;
    let quantization = resolve_quantization(
        precision.as_deref(),
        selected_model.recommended_quantization,
    )?;
    let selector_label = selected_model.selector_label.clone();
    let bundle_id = selected_model.bundle_id.clone();
    let descriptor = selected_model.descriptor.clone();
    let bundle = selected_model.bundle;

    let artifact_root = artifact_root.unwrap_or_else(default_artifact_root);
    let catalog = motlie_models::Catalog::with_defaults();

    println!("catalog-entry-count: {}", catalog.len());
    ensure!(
        catalog.bundle(&bundle_id).is_some(),
        "selected model `{selector_label}` is not registered in the curated catalog; available model flags: {}",
        available_model_names().join(", ")
    );

    println!("bundle-selector: {selector_label}");
    println!("bundle-id: {}", bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    support::print_process_snapshot("process-before-start", &support::current_process_snapshot());
    println!("quantization: {}", quantization_label_isq(quantization));

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
    support::print_model_metrics("model-metrics-after-start", handle.metric_snapshot());

    let chat = handle.chat().context("gemma4 bundle should expose chat")?;
    let generation_defaults = selected_model.generation_defaults.clone();

    if tool_demo_only {
        tool_demo_support::run_tool_demo_with_options(
            chat,
            tool_demo_support::ToolDemoOptions {
                generation_defaults: &generation_defaults,
                system_prompt: None,
                thinking: None,
            },
        )
        .await?;
        support::print_process_snapshot(
            "process-after-tool-demo",
            &support::current_process_snapshot(),
        );
        support::print_model_metrics("model-metrics-after-tool-demo", handle.metric_snapshot());
        handle
            .shutdown()
            .await
            .context("bundle shutdown should succeed")?;
        support::print_process_snapshot(
            "process-after-shutdown",
            &support::current_process_snapshot(),
        );
        return Ok(());
    }

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
        support::print_model_metrics("model-metrics-after-image-chat", handle.metric_snapshot());
    } else {
        println!("\n--- image + text chat ---");
        println!("skipped: pass --image=/path/to/image to exercise the multimodal path");
    }

    if tool_demo {
        tool_demo_support::run_tool_demo_with_options(
            chat,
            tool_demo_support::ToolDemoOptions {
                generation_defaults: &generation_defaults,
                system_prompt: None,
                thinking: None,
            },
        )
        .await?;
        support::print_process_snapshot(
            "process-after-tool-demo",
            &support::current_process_snapshot(),
        );
        support::print_model_metrics("model-metrics-after-tool-demo", handle.metric_snapshot());
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

struct SelectedGemmaModel {
    selector_label: String,
    bundle_id: motlie_model::BundleId,
    descriptor: BundleDescriptor,
    bundle: CuratedBundle,
    generation_defaults: motlie_model::GenerationParams,
    recommended_quantization: Option<QuantizationBits>,
}

fn select_model(model_name: Option<&str>) -> Result<SelectedGemmaModel> {
    let default_name;
    let requested = if let Some(model_name) = model_name {
        model_name
    } else {
        default_name = default_model_name()
            .context("no Gemma 4 model feature is enabled for chat_multimodal_gemma4")?;
        default_name
    };
    let normalized = requested.to_ascii_lowercase().replace('_', "-");

    match normalized.as_str() {
        "e2b" | "gemma4-e2b" | "google/gemma4-e2b" => {
            #[cfg(feature = "model-gemma4-e2b")]
            {
                return Ok(selected_from_chat_model(
                    ChatModels::Gemma4E2B,
                    MistralMultimodalSpec::gemma4_e2b(),
                ));
            }
            #[cfg(not(feature = "model-gemma4-e2b"))]
            bail_model_feature_disabled(requested, "model-gemma4-e2b")
        }
        "e4b" | "gemma4-e4b" | "google/gemma4-e4b" => {
            #[cfg(feature = "model-gemma4-e4b")]
            {
                return Ok(selected_from_chat_model(
                    ChatModels::Gemma4E4B,
                    MistralMultimodalSpec::gemma4_e4b(),
                ));
            }
            #[cfg(not(feature = "model-gemma4-e4b"))]
            bail_model_feature_disabled(requested, "model-gemma4-e4b")
        }
        "12b" | "gemma4-12b" | "google/gemma4-12b" => {
            #[cfg(feature = "model-gemma4-12b")]
            {
                return Ok(selected_from_chat_model(
                    ChatModels::Gemma4_12B,
                    MistralMultimodalSpec::gemma4_12b(),
                ));
            }
            #[cfg(not(feature = "model-gemma4-12b"))]
            bail_model_feature_disabled(requested, "model-gemma4-12b")
        }
        other => bail!(
            "unknown Gemma 4 model `{other}`; use one of: {}",
            available_model_names().join(", ")
        ),
    }
}

fn selected_from_chat_model(model: ChatModels, spec: MistralMultimodalSpec) -> SelectedGemmaModel {
    SelectedGemmaModel {
        selector_label: model.to_string(),
        bundle_id: model.bundle_id(),
        descriptor: model.descriptor(),
        bundle: model.bundle(),
        generation_defaults: spec.recommended_generation_params,
        recommended_quantization: spec.quantization.recommended(),
    }
}

fn resolve_quantization(
    precision: Option<&str>,
    recommended: Option<QuantizationBits>,
) -> Result<Option<QuantizationBits>> {
    match precision {
        Some("q4") => Ok(Some(QuantizationBits::Four)),
        Some("q8") => Ok(Some(QuantizationBits::Eight)),
        Some("f32") | Some("full") | Some("none") => Ok(None),
        None => Ok(recommended),
        Some(other) => bail!("unknown precision `{other}` — use q4, q8, or f32"),
    }
}

fn default_model_name() -> Option<&'static str> {
    #[cfg(feature = "model-gemma4-e2b")]
    {
        return Some("gemma4-e2b");
    }
    #[cfg(all(not(feature = "model-gemma4-e2b"), feature = "model-gemma4-e4b"))]
    {
        return Some("gemma4-e4b");
    }
    #[cfg(all(
        not(feature = "model-gemma4-e2b"),
        not(feature = "model-gemma4-e4b"),
        feature = "model-gemma4-12b"
    ))]
    {
        return Some("gemma4-12b");
    }
    #[allow(unreachable_code)]
    None
}

fn available_model_names() -> Vec<&'static str> {
    let mut names = Vec::new();
    #[cfg(feature = "model-gemma4-e2b")]
    names.push("gemma4-e2b");
    #[cfg(feature = "model-gemma4-e4b")]
    names.push("gemma4-e4b");
    #[cfg(feature = "model-gemma4-12b")]
    names.push("gemma4-12b");
    names
}

#[allow(dead_code)]
fn bail_model_feature_disabled(requested: &str, feature: &str) -> Result<SelectedGemmaModel> {
    bail!(
        "requested model `{requested}` requires `{feature}`; enabled model flags: {}",
        available_model_names().join(", ")
    )
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
