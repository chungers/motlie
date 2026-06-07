use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::{
    BundleHandle, CapabilityKind, ChatMessage, ChatModel, ChatRequest, ChatResponse, ChatRole,
    CompletionModel, CompletionRequest, GenerationParams, ToolChoice, ToolInputSchema, ToolName,
    ToolSpec,
};

use crate::metrics::{CapabilityPerformanceMetrics, ChatPerformanceMetrics, PerformanceMetrics};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms,
    evaluate_performance_measured, evaluate_resource_status, prepare_bundle, start_options,
    SectionEvaluation,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, ChatAssertions};

pub struct ChatRunner;

#[async_trait]
impl ScenarioRunner for ChatRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Chat,
            "scenario `{}` is not a chat scenario",
            context.scenario.id
        );
        let chat_scenario = context
            .scenario
            .chat()
            .context("chat scenario should carry chat input/assertions")?
            .clone();

        let mut extra_capabilities = Vec::new();
        if chat_scenario.input.completion_prompt.is_some()
            || chat_scenario.assertions.min_completion_chars.is_some()
        {
            extra_capabilities.push(CapabilityKind::Completion);
        }
        if chat_scenario.input.tool_prompt.is_some()
            || chat_scenario.assertions.min_tool_calls.is_some()
        {
            extra_capabilities.push(CapabilityKind::ToolUse);
        }

        let prepared = prepare_bundle(
            &context,
            bundle_filter_capability_kind(context.scenario.bundle_filter.capability),
            &extra_capabilities,
        )?;

        context.metrics_sampler.sample();
        let startup_started_at = std::time::Instant::now();
        let handle = prepared
            .bundle
            .start(start_options(&context, &prepared))
            .await
            .with_context(|| format!("failed to start bundle `{}`", prepared.bundle_id))?;
        let startup_ms = elapsed_ms(startup_started_at.elapsed());
        context.metrics_sampler.sample();

        let chat = handle
            .chat()
            .context("selected bundle should expose chat")?;
        let params = generation_params(
            chat_scenario.input.max_tokens,
            chat_scenario.input.temperature,
        );

        let primary_started_at = std::time::Instant::now();
        let primary = chat
            .generate(ChatRequest {
                messages: primary_messages(
                    chat_scenario.input.system_prompt.as_deref(),
                    &chat_scenario.input.prompt,
                ),
                params: params.clone(),
                ..Default::default()
            })
            .await
            .context("primary chat generation failed")?;
        let primary_latency_ms = elapsed_ms(primary_started_at.elapsed());
        context.metrics_sampler.sample();

        let mut request_latencies_ms = vec![primary_latency_ms];
        let followup = if let Some(followup_prompt) = &chat_scenario.input.followup_prompt {
            let followup_started_at = std::time::Instant::now();
            let response = chat
                .generate(ChatRequest {
                    messages: followup_messages(
                        chat_scenario.input.system_prompt.as_deref(),
                        &chat_scenario.input.prompt,
                        &primary.content,
                        followup_prompt,
                    ),
                    params: params.clone(),
                    ..Default::default()
                })
                .await
                .context("follow-up chat generation failed")?;
            request_latencies_ms.push(elapsed_ms(followup_started_at.elapsed()));
            context.metrics_sampler.sample();
            Some(response)
        } else {
            None
        };

        let completion = if let Some(completion_prompt) = &chat_scenario.input.completion_prompt {
            let completion = handle
                .completion()
                .context("selected bundle should expose completion")?;
            let completion_started_at = std::time::Instant::now();
            let response = completion
                .complete(CompletionRequest {
                    prompt: completion_prompt.clone(),
                    params: params.clone(),
                })
                .await
                .context("completion generation failed")?;
            request_latencies_ms.push(elapsed_ms(completion_started_at.elapsed()));
            context.metrics_sampler.sample();
            Some(response)
        } else {
            None
        };

        let tool_response = if let Some(tool_prompt) = &chat_scenario.input.tool_prompt {
            let tool_started_at = std::time::Instant::now();
            let response = chat
                .generate(ChatRequest {
                    messages: primary_messages(
                        chat_scenario.input.system_prompt.as_deref(),
                        tool_prompt,
                    ),
                    params,
                    tools: vec![weather_tool_spec(chat_scenario.input.tool_name.as_deref())?],
                    tool_choice: Some(ToolChoice::Auto),
                    ..Default::default()
                })
                .await
                .context("tool-use chat generation failed")?;
            request_latencies_ms.push(elapsed_ms(tool_started_at.elapsed()));
            context.metrics_sampler.sample();
            Some(response)
        } else {
            None
        };

        let model_metrics = handle.metric_snapshot();
        handle
            .shutdown()
            .await
            .with_context(|| format!("failed to shut down bundle `{}`", prepared.bundle_id))?;

        let completion_tokens = primary
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens)
            .map(u64::from);
        let prompt_tokens = primary
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens)
            .map(u64::from);
        let latency_sum = request_latencies_ms.iter().sum::<u64>();
        let tokens_per_second = completion_tokens.and_then(|tokens| {
            (latency_sum > 0).then(|| tokens as f64 / (latency_sum as f64 / 1000.0))
        });
        let model_text_metrics = model_metrics.and_then(|snapshot| snapshot.text_generation);
        let chat_metrics = ChatPerformanceMetrics {
            prompt_tokens,
            completion_tokens,
            tokens_per_second: tokens_per_second.or_else(|| {
                model_text_metrics
                    .as_ref()
                    .and_then(|metrics| metrics.avg_generated_tokens_per_sec)
                    .map(|tokens| tokens.0 as f64)
            }),
            response_chars: Some(char_count(&primary.content)),
            followup_response_chars: followup
                .as_ref()
                .map(|response| char_count(&response.content)),
            completion_chars: completion
                .as_ref()
                .map(|response| char_count(&response.content)),
            tool_call_count: tool_response
                .as_ref()
                .map(|response| response.tool_calls.len() as u64),
        };
        let performance = PerformanceMetrics {
            startup_ms: Some(startup_ms),
            request_latencies_ms,
            capability_metrics: CapabilityPerformanceMetrics::Chat(chat_metrics),
        };
        let resources = context.metrics_sampler.finish();
        let assertions = evaluate_assertions(
            &primary,
            followup.as_ref(),
            completion.as_ref(),
            tool_response.as_ref(),
            &chat_scenario.assertions,
        );
        let performance_evaluation = evaluate_performance_measured(
            performance.startup_ms.is_some() && !performance.request_latencies_ms.is_empty(),
            "performance metrics missing startup or request latency",
        );
        let resource_evaluation = evaluate_resource_status(&resources, &context);
        let record = build_record(
            &context,
            &prepared,
            performance,
            resources,
            assertions,
            performance_evaluation,
            resource_evaluation,
        );

        context.output_sink.emit(&record)?;
        Ok(record)
    }
}

fn generation_params(max_tokens: Option<u32>, temperature: Option<f32>) -> GenerationParams {
    GenerationParams {
        max_tokens,
        temperature,
        ..Default::default()
    }
}

fn primary_messages(system_prompt: Option<&str>, prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = system_prompt {
        messages.push(ChatMessage::text(ChatRole::System, system_prompt));
    }
    messages.push(ChatMessage::text(ChatRole::User, prompt));
    messages
}

fn followup_messages(
    system_prompt: Option<&str>,
    prompt: &str,
    response: &str,
    followup_prompt: &str,
) -> Vec<ChatMessage> {
    let mut messages = primary_messages(system_prompt, prompt);
    messages.push(ChatMessage::text(ChatRole::Assistant, response));
    messages.push(ChatMessage::text(ChatRole::User, followup_prompt));
    messages
}

fn weather_tool_spec(tool_name: Option<&str>) -> Result<ToolSpec> {
    Ok(ToolSpec {
        name: ToolName::new(tool_name.unwrap_or("get_weather"))?,
        description: "Return weather for a city.".to_owned(),
        input_schema: ToolInputSchema::from_json_schema(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
        )?,
    })
}

fn evaluate_assertions(
    primary: &ChatResponse,
    followup: Option<&ChatResponse>,
    completion: Option<&motlie_model::CompletionResponse>,
    tool_response: Option<&ChatResponse>,
    assertions: &ChatAssertions,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_response_chars) = assertions.min_response_chars {
        outcomes.push(assertion(
            "min_response_chars",
            primary.content.chars().count() >= min_response_chars,
            Some(format!(
                "response_chars={} min={min_response_chars}",
                primary.content.chars().count()
            )),
        ));
    }
    if let Some(min_followup_response_chars) = assertions.min_followup_response_chars {
        let actual = followup
            .map(|response| response.content.chars().count())
            .unwrap_or_default();
        outcomes.push(assertion(
            "min_followup_response_chars",
            actual >= min_followup_response_chars,
            Some(format!(
                "followup_response_chars={actual} min={min_followup_response_chars}"
            )),
        ));
    }
    if let Some(min_completion_chars) = assertions.min_completion_chars {
        let actual = completion
            .map(|response| response.content.chars().count())
            .unwrap_or_default();
        outcomes.push(assertion(
            "min_completion_chars",
            actual >= min_completion_chars,
            Some(format!(
                "completion_chars={actual} min={min_completion_chars}"
            )),
        ));
    }
    if let Some(min_tool_calls) = assertions.min_tool_calls {
        let actual = tool_response
            .map(|response| response.tool_calls.len())
            .unwrap_or_default();
        outcomes.push(assertion(
            "min_tool_calls",
            actual >= min_tool_calls,
            Some(format!("tool_calls={actual} min={min_tool_calls}")),
        ));
    }
    for required in &assertions.required_substrings {
        outcomes.push(assertion(
            format!("required_substring:{required}"),
            contains_case_insensitive(&primary.content, required),
            Some(format!("response_contains={required}")),
        ));
    }

    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "chat_response_non_empty".to_owned(),
            status: if primary.content.trim().is_empty() {
                AcceptanceStatus::Fail
            } else {
                AcceptanceStatus::Pass
            },
            message: Some(format!(
                "response_chars={}",
                primary.content.chars().count()
            )),
        });
    }

    outcomes
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn char_count(value: &str) -> u64 {
    value.chars().count() as u64
}

#[allow(dead_code)]
fn _assert_section_evaluation_is_send_sync(_: SectionEvaluation) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_assertions_name_missing_completion_gate() {
        let outcomes = evaluate_assertions(
            &ChatResponse::text("hello"),
            None,
            None,
            None,
            &ChatAssertions {
                min_completion_chars: Some(1),
                ..Default::default()
            },
        );

        assert_eq!(outcomes[0].name, "min_completion_chars");
        assert_eq!(outcomes[0].status, AcceptanceStatus::Fail);
    }

    #[test]
    fn required_substrings_are_case_insensitive() {
        let outcomes = evaluate_assertions(
            &ChatResponse::text("Hello from Motlie"),
            None,
            None,
            None,
            &ChatAssertions {
                required_substrings: vec!["hello".to_owned()],
                ..Default::default()
            },
        );

        assert_eq!(outcomes[0].status, AcceptanceStatus::Pass);
    }
}
