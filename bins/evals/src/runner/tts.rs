use std::future::Future;

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use motlie_model::typed::{AudioBuf, Mono, SpeechStream, SpeechSynthesizer, SynthesisRequest};
use motlie_model::{BundleHandle, ModelError, SpeechParams, StartOptions};

use crate::metrics::{CapabilityPerformanceMetrics, PerformanceMetrics, TtsPerformanceMetrics};
use crate::result::{AcceptanceStatus, AssertionOutcome};
use crate::runner::support::{
    assertion, build_record, bundle_filter_capability_kind, elapsed_ms,
    evaluate_performance_measured, evaluate_resource_status, observe_backend_accelerator,
    prepare_bundle,
};
use crate::runner::{RunContext, ScenarioRunner};
use crate::scenario::{CapabilityName, TtsAssertions};

pub struct TtsRunner;

#[async_trait]
impl ScenarioRunner for TtsRunner {
    async fn run(&self, mut context: RunContext) -> Result<crate::result::ResultRecord> {
        ensure!(
            context.scenario.capability() == CapabilityName::Tts,
            "scenario `{}` is not a TTS scenario",
            context.scenario.id
        );
        let tts_scenario = context
            .scenario
            .tts()
            .context("TTS scenario should carry TTS input/assertions")?
            .clone();
        let prepared = prepare_bundle(
            &context,
            bundle_filter_capability_kind(context.scenario.bundle_filter.capability),
            &[],
        )?;
        let request = SynthesisRequest {
            text: tts_scenario.input.text.clone(),
            params: SpeechParams {
                speaking_rate: tts_scenario.input.speaking_rate,
                ..Default::default()
            },
        };

        let eval = run_selected_tts(&mut context, &prepared, request).await?;

        let tts_metrics = TtsPerformanceMetrics {
            text_chars: Some(tts_scenario.input.text.chars().count() as u64),
            synthesis_latency_ms: Some(eval.synthesis_latency_ms),
            ttfa_first_chunk_ms: eval.ttfa_first_chunk_ms,
            audio_duration_ms: Some(eval.audio_duration_ms),
            real_time_factor: (eval.audio_duration_ms > 0)
                .then(|| eval.synthesis_latency_ms as f64 / eval.audio_duration_ms as f64),
            sample_count: Some(eval.sample_count),
            sample_rate_hz: Some(eval.sample_rate_hz),
            chunk_count: Some(eval.chunk_count),
        };
        let performance = PerformanceMetrics {
            startup_ms: Some(eval.startup_ms),
            request_latencies_ms: vec![eval.synthesis_latency_ms],
            capability_metrics: CapabilityPerformanceMetrics::Tts(tts_metrics),
            ..Default::default()
        };
        let resources = context.metrics_sampler.finish();
        let assertions = evaluate_assertions(
            eval.audio_duration_ms,
            eval.sample_count,
            &tts_scenario.assertions,
        );
        let performance_evaluation = evaluate_performance_measured(
            performance.startup_ms.is_some() && !performance.request_latencies_ms.is_empty(),
            "performance metrics missing startup or synthesis latency",
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

struct TtsEvalOutput {
    startup_ms: u64,
    synthesis_latency_ms: u64,
    ttfa_first_chunk_ms: Option<u64>,
    audio_duration_ms: u64,
    sample_count: u64,
    sample_rate_hz: u64,
    chunk_count: u64,
}

#[allow(unused_variables)]
async fn run_selected_tts(
    context: &mut RunContext,
    prepared: &crate::runner::support::PreparedBundle,
    request: SynthesisRequest,
) -> Result<TtsEvalOutput> {
    let bundle_id = prepared.bundle_id.as_str();

    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    if bundle_id == "piper_en_us_ljspeech_medium" {
        return run_typed_tts::<_, _, _, i16, 22_050>(
            context,
            crate::runner::support::start_options(context, prepared),
            request,
            motlie_models::tts::piper_en_us_ljspeech_medium::start_typed,
        )
        .await;
    }

    #[cfg(feature = "model-qwen3-tts-cpp")]
    if bundle_id == "qwen3_tts_cpp_0_6b" {
        return run_typed_tts::<_, _, _, f32, 24_000>(
            context,
            crate::runner::support::start_options(context, prepared),
            request,
            motlie_models::tts::qwen3_tts_cpp::start_typed,
        )
        .await;
    }

    bail!("TTS bundle `{bundle_id}` is not enabled or not supported by the eval runner")
}

#[allow(dead_code)]
async fn run_typed_tts<Handle, Start, StartFuture, Sample, const RATE_HZ: u32>(
    context: &mut RunContext,
    options: StartOptions,
    request: SynthesisRequest,
    start: Start,
) -> Result<TtsEvalOutput>
where
    Handle: BundleHandle
        + SpeechSynthesizer<Request = SynthesisRequest, Output = AudioBuf<Sample, RATE_HZ, Mono>>,
    Start: FnOnce(StartOptions) -> StartFuture,
    StartFuture: Future<Output = std::result::Result<Handle, ModelError>> + Send,
    Sample: Clone + Send + Sync + 'static,
{
    context.metrics_sampler.sample();
    let startup_started_at = std::time::Instant::now();
    let handle = start(options).await.context("failed to start TTS bundle")?;
    let startup_ms = elapsed_ms(startup_started_at.elapsed());
    observe_backend_accelerator(context, &handle);
    context.metrics_sampler.sample();

    let synthesis_started_at = std::time::Instant::now();
    let mut stream = handle
        .synthesize(request)
        .await
        .context("failed to open typed speech stream")?;
    let mut sample_count = 0_u64;
    let mut chunk_count = 0_u64;
    let mut ttfa_first_chunk_ms = None;
    while let Some(chunk) = stream
        .next_chunk()
        .await
        .context("speech next_chunk failed")?
    {
        sample_count = sample_count.saturating_add(chunk.samples().len() as u64);
        chunk_count = chunk_count.saturating_add(1);
        if ttfa_first_chunk_ms.is_none() {
            ttfa_first_chunk_ms = Some(elapsed_ms(synthesis_started_at.elapsed()));
        }
        context.metrics_sampler.sample();
    }
    stream
        .finish()
        .await
        .context("speech stream finish failed")?;
    let synthesis_latency_ms = elapsed_ms(synthesis_started_at.elapsed());
    handle.shutdown().await.context("TTS shutdown failed")?;

    let audio_duration_ms = sample_count.saturating_mul(1000) / u64::from(RATE_HZ);
    Ok(TtsEvalOutput {
        startup_ms,
        synthesis_latency_ms,
        ttfa_first_chunk_ms,
        audio_duration_ms,
        sample_count,
        sample_rate_hz: u64::from(RATE_HZ),
        chunk_count,
    })
}

fn evaluate_assertions(
    audio_duration_ms: u64,
    sample_count: u64,
    assertions: &TtsAssertions,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::new();
    if let Some(min_audio_duration_ms) = assertions.min_audio_duration_ms {
        outcomes.push(assertion(
            "min_audio_duration_ms",
            audio_duration_ms >= min_audio_duration_ms,
            Some(format!(
                "audio_duration_ms={audio_duration_ms} min={min_audio_duration_ms}"
            )),
        ));
    }
    if let Some(min_sample_count) = assertions.min_sample_count {
        outcomes.push(assertion(
            "min_sample_count",
            sample_count >= min_sample_count,
            Some(format!(
                "sample_count={sample_count} min={min_sample_count}"
            )),
        ));
    }
    if outcomes.is_empty() {
        outcomes.push(AssertionOutcome {
            name: "audio_non_empty".to_owned(),
            status: if sample_count > 0 {
                AcceptanceStatus::Pass
            } else {
                AcceptanceStatus::Fail
            },
            message: Some(format!("sample_count={sample_count}")),
        });
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_duration_assertion_names_gate() {
        let outcomes = evaluate_assertions(
            100,
            1,
            &TtsAssertions {
                min_audio_duration_ms: Some(200),
                ..Default::default()
            },
        );

        assert_eq!(outcomes[0].name, "min_audio_duration_ms");
        assert_eq!(outcomes[0].status, AcceptanceStatus::Fail);
    }
}
