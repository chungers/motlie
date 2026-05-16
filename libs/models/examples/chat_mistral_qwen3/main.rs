use anyhow::{bail, ensure, Context, Result};
use motlie_model::{
    ArtifactPolicy, BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole, CompletionModel,
    ContentPart, GenerationParams, QuantizationBits, StartOptions, ToolChoice,
};
use motlie_models::{
    chat::ChatModels, default_artifact_root, quantization_label_isq, ModelSelector, ToolError,
    ToolRegistry,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
    let mut chat_selector = None;
    let mut precision = None;
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
        } else if let Some(selector) = arg.strip_prefix("--chat=") {
            chat_selector = Some(selector.to_owned());
        } else if let Some(p) = arg.strip_prefix("--precision=") {
            precision = Some(p.to_owned());
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    if input.trim().is_empty() {
        bail!(
            "usage: cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- \
             [--download-artifacts] [--tool-demo|--tool-demo-only] [--chat=qwen/qwen3_4b] [--precision=q4|q8|f32] <prompt>"
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
                model_selector.bundle()?,
                "selector",
            )
        } else {
            let model = ChatModels::Qwen3_4B;
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
        "chat_mistral_qwen3 must be built with exactly one curated bundle feature enabled"
    );

    println!("bundle-selector: {selector_label}");
    println!("resolution-path: {path_kind}");
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
        .context("bundle startup should succeed from local artifacts")?;
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

    let chat = handle.chat().context("qwen3 bundle should expose chat")?;

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

    // Single-turn request.
    println!("\n--- single-turn ---");
    let started_at = Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: vec![
                ChatMessage::new(ChatRole::System, "Be concise. Answer in one paragraph."),
                ChatMessage::new(ChatRole::User, &input),
            ],
            ..Default::default()
        })
        .await
        .context("chat generation should succeed")?;
    let latency = started_at.elapsed();

    println!("prompt: {input}");
    println!("response: {}", response.content);
    println!("latency-ms: {:.2}", latency.as_secs_f64() * 1000.0);
    support::print_process_snapshot(
        "process-after-single-turn",
        &support::current_process_snapshot(),
    );
    support::print_model_metrics("model-metrics-after-single-turn", handle.metric_snapshot());

    // Multi-turn follow-up.
    println!("\n--- multi-turn follow-up ---");
    let followup_started_at = Instant::now();
    let followup = chat
        .generate(ChatRequest {
            messages: vec![
                ChatMessage::new(ChatRole::System, "Be concise. Answer in one paragraph."),
                ChatMessage::new(ChatRole::User, &input),
                ChatMessage::new(ChatRole::Assistant, &response.content),
                ChatMessage::new(ChatRole::User, "Now explain that in simpler terms."),
            ],
            ..Default::default()
        })
        .await
        .context("multi-turn chat should succeed")?;
    let followup_latency = followup_started_at.elapsed();

    println!("follow-up-prompt: Now explain that in simpler terms.");
    println!("follow-up-response: {}", followup.content);
    println!(
        "follow-up-latency-ms: {:.2}",
        followup_latency.as_secs_f64() * 1000.0
    );
    support::print_process_snapshot(
        "process-after-follow-up",
        &support::current_process_snapshot(),
    );
    support::print_model_metrics("model-metrics-after-follow-up", handle.metric_snapshot());

    if tool_demo {
        run_tool_demo(chat).await?;
        support::print_process_snapshot(
            "process-after-tool-demo",
            &support::current_process_snapshot(),
        );
        support::print_model_metrics("model-metrics-after-tool-demo", handle.metric_snapshot());
    }

    // Completion path.
    println!("\n--- completion ---");
    let completion = handle
        .completion()
        .context("qwen3 bundle should expose completion")?;
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
