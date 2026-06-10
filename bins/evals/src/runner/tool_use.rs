use std::collections::BTreeSet;

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use motlie_eval_tools::{
    evaluate_cel_assertion, EvalToolRegistry, ToolTranscript, CEL_TOOL_NAME, WEATHER_TOOL_NAME,
};
use motlie_model::{
    BundleHandle, CapabilityKind, ChatMessage, ChatModel, ChatRequest, ChatResponse, ChatRole,
    GenerationParams, ToolChoice,
};

use crate::metrics::{CapabilityPerformanceMetrics, PerformanceMetrics, ToolUsePerformanceMetrics};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, elapsed_ms, evaluate_performance_measured, evaluate_resource_status,
    observe_backend_accelerator, prepare_bundle, start_options,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, ToolUseAssertions, ToolUseInput};

pub struct ToolUseRunner;

#[async_trait]
impl ScenarioRunner for ToolUseRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::ToolUse,
            "scenario `{}` is not a tool_use scenario",
            context.scenario.id
        );
        let tool_scenario = context
            .scenario
            .tool_use()
            .context("tool_use scenario should carry tool input/assertions")?
            .clone();
        let registry = registry_for_input(&tool_scenario.input)?;
        let tool_specs = registry.specs().context("collect eval tool specs")?;
        let prepared = prepare_bundle(&context, CapabilityKind::ToolUse, &[CapabilityKind::Chat])?;

        context.metrics_sampler.sample();
        let startup_started_at = std::time::Instant::now();
        let handle = prepared
            .bundle
            .start(start_options(&context, &prepared))
            .await
            .with_context(|| format!("failed to start bundle `{}`", prepared.bundle_id))?;
        let startup_ms = elapsed_ms(startup_started_at.elapsed());
        observe_backend_accelerator(&mut context, &handle);
        context.metrics_sampler.sample();

        let chat = handle
            .chat()
            .context("selected tool_use bundle should expose chat")?;
        let params = generation_params(&tool_scenario.input);
        let mut messages = initial_messages(&tool_scenario.input);
        let mut transcript = ToolTranscript::default();
        let mut request_latencies_ms = Vec::new();
        let mut total_prompt_tokens = 0_u64;
        let mut total_completion_tokens = 0_u64;
        let round_trip_started_at = std::time::Instant::now();
        let mut final_response = None;
        let mut tool_execution_latency_ms = 0_u64;

        for round in 1..=tool_scenario.input.max_rounds.max(1) {
            let request_started_at = std::time::Instant::now();
            let response = chat
                .generate(ChatRequest {
                    messages: messages.clone(),
                    params: params.clone(),
                    tools: tool_specs.clone(),
                    tool_choice: Some(ToolChoice::Auto),
                    ..Default::default()
                })
                .await
                .with_context(|| format!("tool-use generation round {round} failed"))?;
            request_latencies_ms.push(elapsed_ms(request_started_at.elapsed()));
            context.metrics_sampler.sample();
            accumulate_usage(
                &response,
                &mut total_prompt_tokens,
                &mut total_completion_tokens,
            );
            transcript.rounds = round;

            if response.tool_calls.is_empty() {
                final_response = Some(response);
                break;
            }

            messages.push(ChatMessage::assistant_tool_calls(
                response.tool_calls.clone(),
            ));
            for call in &response.tool_calls {
                let tool_started_at = std::time::Instant::now();
                match registry.execute_message(call) {
                    Ok((execution, message)) => {
                        tool_execution_latency_ms += elapsed_ms(tool_started_at.elapsed());
                        transcript.invocations.push(execution);
                        messages.push(message);
                    }
                    Err(error) => {
                        tool_execution_latency_ms += elapsed_ms(tool_started_at.elapsed());
                        let message = error.to_string();
                        transcript.tool_call_errors.push(message.clone());
                        messages.push(ChatMessage::tool_result_parts(
                            call.id.clone(),
                            call.name.clone(),
                            serde_json::json!({ "error": message }).to_string(),
                        ));
                    }
                }
            }
        }

        let final_response = match final_response {
            Some(response) => response,
            None => {
                let request_started_at = std::time::Instant::now();
                let response = chat
                    .generate(ChatRequest {
                        messages,
                        params: params.clone(),
                        tools: tool_specs,
                        tool_choice: Some(ToolChoice::None),
                        ..Default::default()
                    })
                    .await
                    .context("tool-use final answer generation failed")?;
                request_latencies_ms.push(elapsed_ms(request_started_at.elapsed()));
                context.metrics_sampler.sample();
                accumulate_usage(
                    &response,
                    &mut total_prompt_tokens,
                    &mut total_completion_tokens,
                );
                response
            }
        };
        transcript.final_response = Some(final_response.content.clone());

        let model_metrics = handle.metric_snapshot();
        handle
            .shutdown()
            .await
            .with_context(|| format!("failed to shut down bundle `{}`", prepared.bundle_id))?;

        let expected_tools = expected_tools(&tool_scenario.input, &tool_scenario.assertions);
        let precision = tool_selection_precision(&transcript, &expected_tools);
        let recall = tool_selection_recall(&transcript, &expected_tools);
        let (argument_precision, argument_recall) =
            argument_scores(&tool_scenario.input, &transcript);
        let latency_sum = request_latencies_ms.iter().sum::<u64>();
        let model_text_metrics = model_metrics.and_then(|snapshot| snapshot.text_generation);
        let tokens_per_second = if latency_sum > 0 && total_completion_tokens > 0 {
            Some(total_completion_tokens as f64 / (latency_sum as f64 / 1000.0))
        } else {
            model_text_metrics
                .as_ref()
                .and_then(|metrics| metrics.avg_generated_tokens_per_sec)
                .map(|tokens| tokens.0 as f64)
        };
        let tool_use_metrics = ToolUsePerformanceMetrics {
            tool_call_count: Some(transcript.tool_call_count(None) as u64),
            expected_tool_count: Some(expected_tools.len() as u64),
            tool_selection_precision: precision,
            tool_selection_recall: recall,
            argument_precision,
            argument_recall,
            repair_turns: Some(u64::from(transcript.rounds.saturating_sub(1))),
            round_trip_latency_ms: Some(elapsed_ms(round_trip_started_at.elapsed())),
            tool_execution_latency_ms: Some(tool_execution_latency_ms),
            final_response_chars: Some(final_response.content.chars().count() as u64),
        };
        let performance = PerformanceMetrics {
            startup_ms: Some(startup_ms),
            request_latencies_ms,
            capability_metrics: CapabilityPerformanceMetrics::ToolUse(tool_use_metrics),
            ..Default::default()
        };
        let resources = context.metrics_sampler.finish();
        let mut assertions = evaluate_assertions(
            &transcript,
            &tool_scenario.input,
            &tool_scenario.assertions,
            &expected_tools,
            tokens_per_second,
        );
        if assertions.is_empty() {
            assertions.push(assertion(
                "tool_use_round_trip_completed",
                transcript.final_response.is_some() && !transcript.invocations.is_empty(),
                Some(format!(
                    "tool_calls={} final_response={}",
                    transcript.tool_call_count(None),
                    transcript.final_response.is_some()
                )),
            ));
        }
        let performance_evaluation = evaluate_performance_measured(
            performance.startup_ms.is_some() && !performance.request_latencies_ms.is_empty(),
            "tool_use performance metrics missing startup or request latency",
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

fn registry_for_input(input: &ToolUseInput) -> Result<EvalToolRegistry> {
    if input.tools.is_empty() {
        Ok(EvalToolRegistry::with_default_tools())
    } else {
        Ok(EvalToolRegistry::with_tools(
            input.tools.iter().map(String::as_str),
        )?)
    }
}

fn generation_params(input: &ToolUseInput) -> GenerationParams {
    GenerationParams {
        max_tokens: input.max_tokens.or(Some(192)),
        temperature: input.temperature.or(Some(0.2)),
        ..Default::default()
    }
}

fn initial_messages(input: &ToolUseInput) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = &input.system_prompt {
        messages.push(ChatMessage::text(ChatRole::System, system_prompt));
    }
    messages.push(ChatMessage::text(ChatRole::User, &input.prompt));
    messages
}

fn accumulate_usage(response: &ChatResponse, prompt_tokens: &mut u64, completion_tokens: &mut u64) {
    if let Some(usage) = &response.usage {
        *prompt_tokens += usage.prompt_tokens.map(u64::from).unwrap_or_default();
        *completion_tokens += usage.completion_tokens.map(u64::from).unwrap_or_default();
    }
}

fn expected_tools(input: &ToolUseInput, assertions: &ToolUseAssertions) -> BTreeSet<String> {
    let mut expected = BTreeSet::new();
    if let Some(tool) = &input.expected_tool {
        expected.insert(tool.clone());
    }
    for tool in &assertions.required_tools {
        expected.insert(tool.clone());
    }
    if expected.is_empty() {
        expected.insert(WEATHER_TOOL_NAME.to_owned());
        if input.tools.iter().any(|tool| tool == CEL_TOOL_NAME) {
            expected.insert(CEL_TOOL_NAME.to_owned());
        }
    }
    expected
}

fn tool_selection_precision(
    transcript: &ToolTranscript,
    expected_tools: &BTreeSet<String>,
) -> Option<f64> {
    let total = transcript.tool_call_count(None);
    if total == 0 {
        return Some(0.0);
    }
    let expected_calls = transcript
        .invocations
        .iter()
        .filter(|invocation| expected_tools.contains(&invocation.name))
        .count();
    Some(expected_calls as f64 / total as f64)
}

fn tool_selection_recall(
    transcript: &ToolTranscript,
    expected_tools: &BTreeSet<String>,
) -> Option<f64> {
    if expected_tools.is_empty() {
        return None;
    }
    let called = expected_tools
        .iter()
        .filter(|tool| transcript.called(tool))
        .count();
    Some(called as f64 / expected_tools.len() as f64)
}

fn argument_scores(
    input: &ToolUseInput,
    transcript: &ToolTranscript,
) -> (Option<f64>, Option<f64>) {
    let (Some(tool), Some(key), Some(expected)) = (
        input.expected_tool.as_ref(),
        input.expected_argument_key.as_ref(),
        input.expected_argument_value.as_ref(),
    ) else {
        return (None, None);
    };
    let matching_tool_calls = transcript
        .invocations
        .iter()
        .filter(|invocation| invocation.name == *tool)
        .collect::<Vec<_>>();
    if matching_tool_calls.is_empty() {
        return (Some(0.0), Some(0.0));
    }
    let matching_args = matching_tool_calls
        .iter()
        .filter(|invocation| {
            invocation
                .arguments
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|actual| actual.eq_ignore_ascii_case(expected))
        })
        .count();
    let score = matching_args as f64 / matching_tool_calls.len() as f64;
    (Some(score), Some((matching_args > 0) as u8 as f64))
}

fn evaluate_assertions(
    transcript: &ToolTranscript,
    input: &ToolUseInput,
    assertions: &ToolUseAssertions,
    expected_tools: &BTreeSet<String>,
    tokens_per_second: Option<f64>,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_tool_calls) = assertions.min_tool_calls {
        let actual = transcript.tool_call_count(None);
        outcomes.push(assertion(
            "min_tool_calls",
            actual >= min_tool_calls,
            Some(format!("tool_calls={actual} min={min_tool_calls}")),
        ));
    }
    for tool in expected_tools {
        outcomes.push(assertion(
            format!("required_tool:{tool}"),
            transcript.called(tool),
            Some(format!("tool_called={tool}")),
        ));
    }
    if let (Some(tool), Some(key), Some(expected)) = (
        input.expected_tool.as_ref(),
        input.expected_argument_key.as_ref(),
        input.expected_argument_value.as_ref(),
    ) {
        let matched = transcript.invocations.iter().any(|invocation| {
            invocation.name == *tool
                && invocation
                    .arguments
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|actual| actual.eq_ignore_ascii_case(expected))
        });
        outcomes.push(assertion(
            format!("expected_argument:{tool}.{key}"),
            matched,
            Some(format!("expected={expected}")),
        ));
    }
    for required in &assertions.required_final_substrings {
        outcomes.push(assertion(
            format!("required_final_substring:{required}"),
            transcript.final_contains(required),
            Some(format!("final_contains={required}")),
        ));
    }
    for cel in &assertions.cel {
        match evaluate_cel_assertion(&cel.expression, transcript) {
            Ok(result) => outcomes.push(assertion(
                format!("cel:{}", cel.name),
                result.passed,
                Some(result.message),
            )),
            Err(error) => outcomes.push(AssertionOutcome {
                name: format!("cel:{}", cel.name),
                status: AcceptanceStatus::Blocked,
                message: Some(error.to_string()),
            }),
        }
    }
    if !transcript.tool_call_errors.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "tool_execution_errors".to_owned(),
            status: AcceptanceStatus::Fail,
            message: Some(transcript.tool_call_errors.join("; ")),
        });
    }
    outcomes.push(assertion(
        "tool_round_trip_final_response",
        transcript
            .final_response
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty()),
        Some(format!(
            "final_response_chars={}",
            transcript
                .final_response
                .as_deref()
                .map(|value| value.chars().count())
                .unwrap_or_default()
        )),
    ));
    if let Some(tokens_per_second) = tokens_per_second {
        outcomes.push(assertion(
            "tool_use_tokens_per_second_measured",
            tokens_per_second.is_finite() && tokens_per_second >= 0.0,
            Some(format!("tokens_per_second={tokens_per_second:.3}")),
        ));
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_eval_tools::{ToolExecution, ToolTranscript};

    #[test]
    fn argument_scores_report_precision_and_recall() {
        let transcript = ToolTranscript {
            invocations: vec![ToolExecution {
                call_id: "call-1".to_owned(),
                name: WEATHER_TOOL_NAME.to_owned(),
                arguments: serde_json::json!({"city":"Seattle"}),
                output: serde_json::json!({"temperature":72.0}),
                output_json: "{}".to_owned(),
            }],
            final_response: Some("Seattle weather is clear".to_owned()),
            rounds: 1,
            tool_call_errors: Vec::new(),
        };
        let input = ToolUseInput {
            expected_tool: Some(WEATHER_TOOL_NAME.to_owned()),
            expected_argument_key: Some("city".to_owned()),
            expected_argument_value: Some("Seattle".to_owned()),
            ..Default::default()
        };

        let (precision, recall) = argument_scores(&input, &transcript);

        assert_eq!(precision, Some(1.0));
        assert_eq!(recall, Some(1.0));
    }
}
