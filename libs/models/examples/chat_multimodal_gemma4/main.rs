use anyhow::{Context, Result, bail, ensure};
use motlie_model::{
    ArtifactPolicy, BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole, ContentPart,
    GenerationParams, QuantizationBits, StartOptions, ToolChoice, ToolError, ToolRegistry,
};
use motlie_models::{chat::ChatModels, default_artifact_root, quantization_label_isq};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[path = "../support.rs"]
mod support;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
    units: TemperatureUnits,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum TemperatureUnits {
    Celsius,
    Fahrenheit,
}

#[derive(Debug, Serialize)]
struct WeatherOutput {
    city: String,
    temperature: f32,
    units: TemperatureUnits,
    summary: String,
}

async fn get_weather(args: WeatherArgs) -> std::result::Result<WeatherOutput, ToolError> {
    Ok(WeatherOutput {
        city: args.city,
        temperature: match args.units {
            TemperatureUnits::Celsius => 22.0,
            TemperatureUnits::Fahrenheit => 72.0,
        },
        units: args.units,
        summary: "clear".to_string(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut precision = None;
    let mut image_path = None;
    let mut download_artifacts = false;
    let mut tool_demo = false;
    let mut tool_demo_only = false;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if arg == "--tool-demo" {
            tool_demo = true;
        } else if arg == "--tool-demo-only" {
            tool_demo = true;
            tool_demo_only = true;
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
            "usage: cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- \
             [--download-artifacts] [--tool-demo|--tool-demo-only] [--precision=q4|q8|f32] [--image=/path/to/image] <prompt>"
        );
    }

    let quantization = match precision.as_deref() {
        Some("q4") | None => Some(QuantizationBits::Four),
        Some("q8") => Some(QuantizationBits::Eight),
        Some("f32") => None,
        Some(other) => bail!("unknown precision `{other}` — use q4, q8, or f32"),
    };

    let model = ChatModels::Gemma4E2B;
    let selector_label = model.to_string();
    let bundle_id = model.bundle_id();
    let descriptor = model.descriptor();
    let bundle = model.bundle();

    let artifact_root = default_artifact_root();
    let catalog = motlie_models::Catalog::with_defaults();

    println!("catalog-entry-count: {}", catalog.len());
    ensure!(
        catalog.len() == 1,
        "chat_multimodal_gemma4 must be built with exactly one curated bundle feature enabled"
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

    if tool_demo_only {
        run_tool_demo(chat).await?;
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
        run_tool_demo(chat).await?;
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

async fn run_tool_demo(chat: &impl ChatModel) -> Result<()> {
    println!("\n--- tool calling ---");

    let mut registry = ToolRegistry::new();
    registry
        .insert_fn(
            "get_weather",
            "Return a current weather summary for a city.",
            get_weather,
        )
        .context("register get_weather tool")?;

    let tools = registry.specs();
    let mut messages = vec![
        ChatMessage::new(
            ChatRole::System,
            "Use tools when they are relevant. After receiving tool results, answer in one concise sentence.",
        ),
        ChatMessage::new(
            ChatRole::User,
            "What is the current weather in Seattle in fahrenheit?",
        ),
    ];

    let tool_request_started_at = Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: messages.clone(),
            params: tool_demo_generation_params(),
            tools: tools.clone(),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        })
        .await
        .context("tool-call generation should succeed")?;
    let tool_request_latency = tool_request_started_at.elapsed();

    println!("tool-request-response: {}", response.content);
    println!("tool-call-count: {}", response.tool_calls.len());
    println!(
        "tool-request-latency-ms: {:.2}",
        tool_request_latency.as_secs_f64() * 1000.0
    );
    ensure!(
        !response.tool_calls.is_empty(),
        "model did not return any tool calls"
    );

    messages.push(ChatMessage::assistant_tool_calls(
        response.tool_calls.clone(),
    ));
    for call in response.tool_calls {
        println!("tool-call-id: {}", call.id);
        println!("tool-call-name: {}", call.name);
        println!("tool-call-args: {}", call.arguments.raw_json_str());

        let tool_message = registry
            .call_to_message(call)
            .await
            .context("execute model-requested tool")?;
        for part in &tool_message.content {
            if let ContentPart::Text(text) = part {
                println!("tool-result: {text}");
            }
        }
        messages.push(tool_message);
    }

    let final_started_at = Instant::now();
    let final_response = chat
        .generate(ChatRequest {
            messages,
            params: tool_demo_generation_params(),
            tools,
            tool_choice: Some(ToolChoice::None),
            ..Default::default()
        })
        .await
        .context("final answer after tool results should succeed")?;
    let final_latency = final_started_at.elapsed();

    println!("tool-final-response: {}", final_response.content);
    println!(
        "tool-final-latency-ms: {:.2}",
        final_latency.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn tool_demo_generation_params() -> GenerationParams {
    GenerationParams {
        max_tokens: Some(128),
        temperature: Some(0.2),
        ..Default::default()
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
