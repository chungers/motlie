#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
use anyhow::Context;
use anyhow::{bail, Result};
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, ChatMessage, ChatModel, ChatRequest, ChatRole,
    CompletionModel, GenerationParams, QuantizationBits, StartOptions, ThinkingMode,
};
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
use motlie_model_llama_cpp::LlamaCppTextSpec;
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
use motlie_models::{
    chat::ChatModels, default_artifact_root, quantization_label_gguf, BundleDescriptor,
    CuratedBundle, ModelSelector,
};
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
use std::time::Instant;

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
#[path = "../support.rs"]
mod support;
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
#[path = "../tool_demo_support.rs"]
mod tool_demo_support;

#[cfg(not(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
)))]
fn main() -> Result<()> {
    bail!(
        "enable at least one GGUF chat feature: model-qwen3-4b-gguf, model-qwen3-6-27b-gguf, model-gemma4-e2b-gguf, or model-gemma4-e4b-gguf"
    )
}

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
#[tokio::main]
async fn main() -> Result<()> {
    let mut chat_selector = None;
    let mut precision = None;
    let mut download_artifacts = false;
    let mut tool_demo = false;
    let mut tool_demo_only = false;
    let mut system_prompt_override = None;
    let mut no_system = false;
    let mut assistant_priming = None;
    let mut thinking = None;
    let mut input_parts = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--download-artifacts" {
            download_artifacts = true;
        } else if arg == "--tool-demo" {
            tool_demo = true;
        } else if arg == "--tool-demo-only" {
            tool_demo = true;
            tool_demo_only = true;
        } else if arg == "--no-system" {
            no_system = true;
        } else if let Some(selector) = arg.strip_prefix("--chat=") {
            chat_selector = Some(selector.to_owned());
        } else if let Some(p) = arg.strip_prefix("--precision=") {
            precision = Some(p.to_owned());
        } else if let Some(prompt) = arg.strip_prefix("--system=") {
            system_prompt_override = Some(prompt.to_owned());
            no_system = false;
        } else if let Some(prompt) = arg.strip_prefix("--assistant=") {
            assistant_priming = Some(prompt.to_owned());
        } else if let Some(mode) = arg.strip_prefix("--thinking=") {
            thinking = Some(parse_thinking(mode)?);
        } else {
            input_parts.push(arg);
        }
    }

    let input = input_parts.join(" ");
    if input.trim().is_empty() {
        bail!(
            "usage: cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf --example chat_gguf_gwen3_gemma4 -- \
             [--download-artifacts] [--tool-demo|--tool-demo-only] [--chat=qwen/qwen3_4b_gguf|google/gemma4_e2b_gguf|google/gemma4_e4b_gguf] \
             [--precision=q4|q5|q8|f16] [--thinking=off|disabled|auto] [--system=TEXT|--no-system] [--assistant=TEXT] <prompt>\n\n\
             This example demonstrates chat generation via the llama.cpp backend using GGUF-quantized weights."
        );
    }

    let selected = select_model(chat_selector)?;
    let quantization = match precision.as_deref() {
        Some("q4") => Some(QuantizationBits::Four),
        Some("q5") => Some(QuantizationBits::Five),
        Some("q8") => Some(QuantizationBits::Eight),
        Some("f16") => None,
        None => selected.spec.quantization.recommended(),
        Some(other) => bail!("unknown precision `{other}` — use q4, q5, q8, or f16"),
    };
    let chat_params =
        GenerationParams::default().with_defaults(&selected.spec.recommended_generation_params);
    let effective_system_prompt = if no_system {
        None
    } else if let Some(prompt) = system_prompt_override {
        Some(prompt)
    } else if let Some(prompt) = selected.spec.recommended_system_prompt {
        Some(prompt.to_owned())
    } else {
        Some("Be concise. Answer in one paragraph.".to_owned())
    };

    let artifact_root = default_artifact_root();

    println!("backend: llama.cpp (GGUF)");
    println!("bundle-selector: {}", selected.selector_label);
    println!("resolution-path: {}", selected.path_kind);
    println!("bundle-id: {}", selected.bundle_id.as_str());
    println!("artifact-root: {}", artifact_root.display());
    support::print_process_snapshot("process-before-start", &support::current_process_snapshot());
    println!("quantization: {}", quantization_label_gguf(quantization));
    println!(
        "recommended-generation-params: {:?}",
        selected.spec.recommended_generation_params
    );
    println!("effective-chat-params: {chat_params:?}");
    println!("thinking: {:?}", thinking.unwrap_or(selected.spec.thinking));
    println!(
        "system-prompt: {}",
        if no_system {
            "disabled"
        } else if effective_system_prompt.is_some() {
            "enabled"
        } else {
            "none"
        }
    );
    if assistant_priming.is_some() {
        println!("assistant-priming: enabled");
    }

    if download_artifacts {
        let catalog = motlie_models::Catalog::with_defaults();
        let summary =
            motlie_models::download_bundle_artifacts(&catalog, &selected.bundle_id, &artifact_root)
                .with_context(|| {
                    format!(
                        "failed to download curated GGUF artifacts for `{}`",
                        selected.bundle_id
                    )
                })?;
        println!("downloaded-files: {}", summary.downloaded.len());
    } else {
        println!("downloaded-files: skipped (using existing local GGUF artifacts only)");
    }

    println!("display-name: {}", selected.descriptor.display_name);
    println!("family: {:?}", selected.descriptor.family);
    println!("backend: {:?}", selected.descriptor.backend);
    println!("capabilities:");
    for capability in selected.descriptor.capability_descriptors() {
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
    let handle = selected
        .bundle
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

    let chat = handle
        .chat()
        .context("llama.cpp bundle should expose chat")?;

    if tool_demo_only {
        tool_demo_support::run_tool_demo_with_options(
            chat,
            tool_demo_support::ToolDemoOptions {
                generation_defaults: &selected.spec.recommended_generation_params,
                system_prompt: effective_system_prompt.as_deref(),
                thinking,
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

    println!("\n--- single-turn ---");
    let started_at = Instant::now();
    let response = chat
        .generate(ChatRequest {
            messages: chat_messages(
                effective_system_prompt.as_deref(),
                assistant_priming.as_deref(),
                &input,
            ),
            params: chat_params.clone(),
            thinking,
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

    println!("\n--- multi-turn follow-up ---");
    let followup_started_at = Instant::now();
    let mut followup_messages = chat_messages(
        effective_system_prompt.as_deref(),
        assistant_priming.as_deref(),
        &input,
    );
    followup_messages.push(ChatMessage::new(ChatRole::Assistant, &response.content));
    followup_messages.push(ChatMessage::new(
        ChatRole::User,
        "Now explain that in simpler terms.",
    ));
    let followup = chat
        .generate(ChatRequest {
            messages: followup_messages,
            params: chat_params.clone(),
            thinking,
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
        tool_demo_support::run_tool_demo_with_options(
            chat,
            tool_demo_support::ToolDemoOptions {
                generation_defaults: &selected.spec.recommended_generation_params,
                system_prompt: effective_system_prompt.as_deref(),
                thinking,
            },
        )
        .await?;
        support::print_process_snapshot(
            "process-after-tool-demo",
            &support::current_process_snapshot(),
        );
        support::print_model_metrics("model-metrics-after-tool-demo", handle.metric_snapshot());
    }

    println!("\n--- completion ---");
    let completion = handle
        .completion()
        .context("llama.cpp bundle should expose completion")?;
    let completion_started_at = Instant::now();
    let completion_response = completion
        .complete(motlie_model::CompletionRequest {
            prompt: format!("Complete this sentence: {input}"),
            params: chat_params,
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

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
struct SelectedGgufModel {
    selector_label: String,
    bundle_id: BundleId,
    descriptor: BundleDescriptor,
    bundle: CuratedBundle,
    spec: LlamaCppTextSpec,
    path_kind: &'static str,
}

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
fn select_model(chat_selector: Option<String>) -> Result<SelectedGgufModel> {
    let (model, path_kind) = if let Some(selector) = chat_selector {
        let model_selector: ModelSelector = format!("chat:{selector}")
            .parse()
            .with_context(|| format!("failed to parse model selector `chat:{selector}`"))?;
        let ModelSelector::Chat(model) = model_selector else {
            bail!("selector `chat:{selector}` did not resolve to a chat model");
        };
        (model, "selector")
    } else {
        (default_gguf_model()?, "direct-enum")
    };
    let spec = spec_for_chat_model(model)?;

    Ok(SelectedGgufModel {
        selector_label: model.to_string(),
        bundle_id: model.bundle_id(),
        descriptor: model.descriptor(),
        bundle: model.bundle(),
        spec,
        path_kind,
    })
}

#[cfg(feature = "model-qwen3-4b-gguf")]
fn default_gguf_model() -> Result<ChatModels> {
    Ok(ChatModels::Qwen3_4B_Gguf)
}

#[cfg(all(
    not(feature = "model-qwen3-4b-gguf"),
    feature = "model-gemma4-e4b-gguf"
))]
fn default_gguf_model() -> Result<ChatModels> {
    Ok(ChatModels::Gemma4E4B_Gguf)
}

#[cfg(all(
    not(feature = "model-qwen3-4b-gguf"),
    not(feature = "model-gemma4-e4b-gguf"),
    feature = "model-gemma4-e2b-gguf"
))]
fn default_gguf_model() -> Result<ChatModels> {
    Ok(ChatModels::Gemma4E2B_Gguf)
}

#[cfg(all(
    not(feature = "model-qwen3-4b-gguf"),
    not(feature = "model-gemma4-e4b-gguf"),
    not(feature = "model-gemma4-e2b-gguf"),
    feature = "model-qwen3-6-27b-gguf"
))]
fn default_gguf_model() -> Result<ChatModels> {
    Ok(ChatModels::Qwen3_6_27B_Gguf)
}

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
fn spec_for_chat_model(model: ChatModels) -> Result<LlamaCppTextSpec> {
    #[allow(unreachable_patterns)]
    match model {
        #[cfg(feature = "model-qwen3-4b-gguf")]
        ChatModels::Qwen3_4B_Gguf => Ok(LlamaCppTextSpec::qwen3_4b()),
        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        ChatModels::Qwen3_6_27B_Gguf => Ok(LlamaCppTextSpec::qwen3_6_27b()),
        #[cfg(feature = "model-gemma4-e2b-gguf")]
        ChatModels::Gemma4E2B_Gguf => Ok(LlamaCppTextSpec::gemma4_e2b()),
        #[cfg(feature = "model-gemma4-e4b-gguf")]
        ChatModels::Gemma4E4B_Gguf => Ok(LlamaCppTextSpec::gemma4_e4b()),
        other => bail!("`{}` is not a llama.cpp GGUF chat model", other.as_str()),
    }
}

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
fn parse_thinking(value: &str) -> Result<ThinkingMode> {
    match value {
        "off" | "disabled" => Ok(ThinkingMode::Disabled),
        "auto" => Ok(ThinkingMode::Auto),
        other => bail!("unknown thinking mode `{other}` — use off, disabled, or auto"),
    }
}

#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
))]
fn chat_messages(
    system_prompt: Option<&str>,
    assistant_priming: Option<&str>,
    user_prompt: &str,
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = system_prompt {
        messages.push(ChatMessage::new(ChatRole::System, system_prompt));
    }
    if let Some(assistant_priming) = assistant_priming {
        messages.push(ChatMessage::new(ChatRole::Assistant, assistant_priming));
    }
    messages.push(ChatMessage::new(ChatRole::User, user_prompt));
    messages
}
