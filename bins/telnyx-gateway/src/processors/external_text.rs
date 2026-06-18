use std::time::{Duration, Instant};

use anyhow::Context;
use tokio::time;

use crate::media::{SharedMediaRegistry, SpeechClearReason};
use crate::operator::state::{
    CallStatus, QualitySpanEmission, SharedState, SpeechOutputConfig, TtsPlaybackStatus,
};
use crate::speech::{self, SpeechConflictPolicy, SpeechQueueRequest};
use crate::text_calls::turns::{AgentTextFrame, GatewayTextFrame, PlaybackFinishedStatus};
use crate::text_calls::websocket::{
    hangup_gateway_call, text_call_session_config, AgentAppendTurn, AgentProvisionalTurn,
    AgentTurnDisposition, TextCallSessionConfig, TextCallSessionHandle, TextCallStreamServices,
    TextCallTurnTiming,
};
use crate::tts::StreamingSpeechTextPacker;

pub(crate) async fn handle_agent_message(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    handle: &TextCallSessionHandle,
    text: &str,
) -> anyhow::Result<()> {
    let frame: AgentTextFrame = serde_json::from_str(text).context("decode app text-call frame")?;
    match frame {
        AgentTextFrame::AgentTurnPartial {
            turn_id,
            text,
            append,
        } => {
            if !append {
                send_error_frame(handle, "invalid_partial", "agent.turn.partial must append")
                    .await?;
                return Ok(());
            }
            let disposition = handle.turns.lock().await.append_turn_disposition(&turn_id);
            let timing = match disposition {
                AgentTurnDisposition::Accepted { timing } => timing,
                AgentTurnDisposition::Superseded => {
                    finish_superseded_turn(handle, turn_id).await?;
                    return Ok(());
                }
                AgentTurnDisposition::Invalid => {
                    send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                    return Ok(());
                }
            };
            process_agent_turn_fragment(
                services,
                gateway_call_id,
                handle,
                turn_id,
                text,
                false,
                timing,
            )
            .await?;
        }
        AgentTextFrame::AgentTurn { turn_id, text } => {
            if handle.append_turns.lock().await.contains_key(&turn_id) {
                let disposition = handle.turns.lock().await.append_turn_disposition(&turn_id);
                let timing = match disposition {
                    AgentTurnDisposition::Accepted { timing } => timing,
                    AgentTurnDisposition::Superseded => {
                        finish_superseded_turn(handle, turn_id).await?;
                        return Ok(());
                    }
                    AgentTurnDisposition::Invalid => {
                        send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                        return Ok(());
                    }
                };
                process_agent_turn_fragment(
                    services,
                    gateway_call_id,
                    handle,
                    turn_id,
                    text,
                    true,
                    timing,
                )
                .await?;
                return Ok(());
            }
            let agent_turn_received_at = Instant::now();
            let disposition = handle.turns.lock().await.accept_agent_turn(&turn_id);
            let timing = match disposition {
                AgentTurnDisposition::Accepted { timing } => timing,
                AgentTurnDisposition::Superseded => {
                    finish_superseded_turn(handle, turn_id).await?;
                    return Ok(());
                }
                AgentTurnDisposition::Invalid => {
                    send_error_frame(handle, "invalid_turn", "turn is not active").await?;
                    return Ok(());
                }
            };
            emit_agent_turn_round_trip_span(
                services,
                gateway_call_id,
                &turn_id,
                timing,
                agent_turn_received_at,
            )
            .await;

            let queued = queue_agent_speech_with_media_wait(
                services,
                handle.speech_output,
                gateway_call_id.to_string(),
                text,
                handle.config,
                Some(timing.finalized_at),
                Some(turn_id.clone()),
            )
            .await?;
            if let Some(replaced_playback_id) = queued.replaced_playback_id.as_deref() {
                send_replaced_playback_canceled(handle, replaced_playback_id).await?;
            }

            handle.turns.lock().await.start_playback(
                turn_id.clone(),
                queued.playback_id.clone(),
                timing,
            );
            handle
                .send(GatewayTextFrame::PlaybackStarted {
                    turn_id: turn_id.clone(),
                    sequence: handle.next_sequence(),
                })
                .await?;
            spawn_playback_terminal_waiter(
                services.clone(),
                gateway_call_id.to_string(),
                handle.clone(),
                queued.playback_id.clone(),
            );
        }
        AgentTextFrame::AgentTurnProvisionalPartial {
            provisional_turn_id,
            generation,
            text,
            append,
        } => {
            if !append {
                send_error_frame(
                    handle,
                    "invalid_provisional_partial",
                    "agent.turn.provisional.partial must append",
                )
                .await?;
                return Ok(());
            }
            process_agent_provisional_fragment(
                services,
                gateway_call_id,
                handle,
                provisional_turn_id,
                generation,
                text,
                false,
            )
            .await?;
        }
        AgentTextFrame::AgentTurnProvisional {
            provisional_turn_id,
            generation,
            text,
        } => {
            process_agent_provisional_fragment(
                services,
                gateway_call_id,
                handle,
                provisional_turn_id,
                generation,
                text,
                true,
            )
            .await?;
        }
        AgentTextFrame::AgentClose { reason } => {
            handle
                .send(GatewayTextFrame::SessionEnd {
                    reason: reason.unwrap_or_else(|| "agent.close".to_string()),
                    sequence: handle.next_sequence(),
                })
                .await?;
            hangup_gateway_call(services, gateway_call_id, "agent requested close").await?;
        }
    }
    Ok(())
}

async fn process_agent_turn_fragment(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    handle: &TextCallSessionHandle,
    turn_id: String,
    text: String,
    final_fragment: bool,
    timing: TextCallTurnTiming,
) -> anyhow::Result<()> {
    let existing = handle.append_turns.lock().await.remove(&turn_id);
    let mut append_turn = if let Some(append_turn) = existing {
        append_turn
    } else {
        emit_agent_turn_round_trip_span(
            services,
            gateway_call_id,
            &turn_id,
            timing,
            Instant::now(),
        )
        .await;
        AgentAppendTurn {
            packer: StreamingSpeechTextPacker::new(
                handle.speech_output.tts_chunking_enabled,
                handle.speech_output.tts_max_text_chunk_chars,
                handle.speech_output.tts_first_chunk_max_chars,
            ),
            speech: None,
            timing,
        }
    };

    let chunks = append_turn.packer.push_fragment(&text, final_fragment);
    if append_turn.speech.is_none() && !chunks.is_empty() {
        let (speech_handle, queued) = queue_append_agent_speech_with_media_wait(
            services,
            AgentAppendSpeechRequest {
                speech_output: handle.speech_output,
                gateway_call_id: gateway_call_id.to_string(),
                initial_chunks: chunks,
                source_label: "text-call agent.turn.partial",
                config: handle.config,
                turn_finalized_at: Some(append_turn.timing.finalized_at),
                turn_id: Some(turn_id.clone()),
            },
        )
        .await?;
        if let Some(replaced_playback_id) = queued.replaced_playback_id.as_deref() {
            send_replaced_playback_canceled(handle, replaced_playback_id).await?;
        }
        handle.turns.lock().await.start_playback(
            turn_id.clone(),
            queued.playback_id.clone(),
            timing,
        );
        handle
            .send(GatewayTextFrame::PlaybackStarted {
                turn_id: turn_id.clone(),
                sequence: handle.next_sequence(),
            })
            .await?;
        spawn_playback_terminal_waiter(
            services.clone(),
            gateway_call_id.to_string(),
            handle.clone(),
            queued.playback_id.clone(),
        );
        if final_fragment {
            speech_handle.finish().await?;
        } else {
            append_turn.speech = Some(speech_handle);
        }
    } else if let Some(speech) = append_turn.speech.as_ref() {
        if speech.append_chunks(chunks, final_fragment).await.is_err() {
            finish_turn(handle, turn_id.clone(), PlaybackFinishedStatus::Canceled).await?;
            return Ok(());
        }
    } else if final_fragment {
        finish_turn(handle, turn_id.clone(), PlaybackFinishedStatus::Completed).await?;
    }

    if !final_fragment {
        handle
            .append_turns
            .lock()
            .await
            .insert(turn_id, append_turn);
    }
    Ok(())
}

async fn process_agent_provisional_fragment(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    handle: &TextCallSessionHandle,
    provisional_turn_id: String,
    generation: u64,
    text: String,
    final_fragment: bool,
) -> anyhow::Result<()> {
    let existing = handle
        .provisional_turns
        .lock()
        .await
        .remove(&provisional_turn_id);
    let mut provisional_turn = match existing {
        Some(existing) if existing.generation > generation => {
            handle
                .provisional_turns
                .lock()
                .await
                .insert(provisional_turn_id.clone(), existing);
            send_error_frame(
                handle,
                "stale_provisional_generation",
                "provisional generation is older than the active generation",
            )
            .await?;
            return Ok(());
        }
        Some(existing) if existing.generation < generation => {
            cancel_agent_provisional_speech(
                &services.media,
                gateway_call_id,
                existing,
                SpeechClearReason::CancelAndReplace,
            )
            .await;
            new_agent_provisional_turn(handle.speech_output, generation)
        }
        Some(existing) => existing,
        None => new_agent_provisional_turn(handle.speech_output, generation),
    };

    let chunks = provisional_turn.packer.push_fragment(&text, final_fragment);
    if provisional_turn.speech.is_none() && !chunks.is_empty() {
        let (speech_handle, queued) = queue_append_agent_speech_with_media_wait(
            services,
            AgentAppendSpeechRequest {
                speech_output: handle.speech_output,
                gateway_call_id: gateway_call_id.to_string(),
                initial_chunks: chunks,
                source_label: "text-call agent.turn.provisional.partial",
                config: handle.config,
                turn_finalized_at: None,
                turn_id: Some(provisional_turn_id.clone()),
            },
        )
        .await?;
        if let Some(replaced_playback_id) = queued.replaced_playback_id.as_deref() {
            send_replaced_playback_canceled(handle, replaced_playback_id).await?;
        }
        handle
            .send(GatewayTextFrame::ProvisionalPlaybackStarted {
                provisional_turn_id: provisional_turn_id.clone(),
                generation,
                playback_id: queued.playback_id.clone(),
                sequence: handle.next_sequence(),
            })
            .await?;
        if final_fragment {
            speech_handle.finish().await?;
        }
        provisional_turn.speech = Some(speech_handle);
        handle
            .provisional_turns
            .lock()
            .await
            .insert(provisional_turn_id, provisional_turn);
        return Ok(());
    }

    if let Some(speech) = provisional_turn.speech.as_ref() {
        if speech.append_chunks(chunks, final_fragment).await.is_err() {
            cancel_agent_provisional_speech(
                &services.media,
                gateway_call_id,
                provisional_turn,
                SpeechClearReason::CancelAndReplace,
            )
            .await;
            return Ok(());
        }
    }

    if !final_fragment || provisional_turn.speech.is_some() {
        handle
            .provisional_turns
            .lock()
            .await
            .insert(provisional_turn_id, provisional_turn);
    }
    Ok(())
}

fn new_agent_provisional_turn(
    speech_output: SpeechOutputConfig,
    generation: u64,
) -> AgentProvisionalTurn {
    AgentProvisionalTurn {
        generation,
        packer: StreamingSpeechTextPacker::new(
            speech_output.tts_chunking_enabled,
            speech_output.tts_max_text_chunk_chars,
            speech_output.tts_first_chunk_max_chars,
        ),
        speech: None,
    }
}

async fn cancel_agent_provisional_speech(
    media: &SharedMediaRegistry,
    gateway_call_id: &str,
    mut turn: AgentProvisionalTurn,
    reason: SpeechClearReason,
) {
    if let Some(speech) = turn.speech.take() {
        speech.cancel_now();
        let _ = media
            .cancel_speech_playback_for_reason(gateway_call_id, &speech.playback_id, reason)
            .await;
    }
}

pub(crate) async fn cancel_agent_provisional_turn(
    media: &SharedMediaRegistry,
    handle: &TextCallSessionHandle,
    gateway_call_id: &str,
    provisional_turn_id: &str,
    generation: u64,
    reason: SpeechClearReason,
) -> bool {
    let existing = handle
        .provisional_turns
        .lock()
        .await
        .remove(provisional_turn_id);
    let Some(existing) = existing else {
        return false;
    };
    if existing.generation != generation {
        handle
            .provisional_turns
            .lock()
            .await
            .insert(provisional_turn_id.to_string(), existing);
        return false;
    }
    cancel_agent_provisional_speech(media, gateway_call_id, existing, reason).await;
    true
}

pub(crate) async fn finish_agent_provisional_turn(
    handle: &TextCallSessionHandle,
    provisional_turn_id: &str,
    generation: u64,
) -> bool {
    let existing = handle
        .provisional_turns
        .lock()
        .await
        .remove(provisional_turn_id);
    let Some(existing) = existing else {
        return false;
    };
    if existing.generation != generation {
        handle
            .provisional_turns
            .lock()
            .await
            .insert(provisional_turn_id.to_string(), existing);
        return false;
    }
    if let Some(speech) = existing.speech {
        let _ = speech.finish().await;
    }
    true
}

fn spawn_playback_terminal_waiter(
    services: TextCallStreamServices,
    gateway_call_id: String,
    handle: TextCallSessionHandle,
    playback_id: String,
) {
    tokio::spawn(async move {
        if let Some(status) = wait_for_playback_terminal(
            &services.state,
            &handle,
            &gateway_call_id,
            &playback_id,
            handle.config.playback_wait_timeout,
        )
        .await
        {
            let turn_id = handle.turns.lock().await.close_playback(&playback_id);
            if let Some(turn_id) = turn_id {
                cancel_append_turn(&handle, &turn_id).await;
                let _ = send_playback_finished(&handle, turn_id, status).await;
            }
        }
    });
}

async fn emit_agent_turn_round_trip_span(
    services: &TextCallStreamServices,
    gateway_call_id: &str,
    turn_id: &str,
    timing: TextCallTurnTiming,
    agent_turn_received_at: Instant,
) {
    let payload = match serde_json::json!({
        "turn_id": turn_id,
        "finalize_to_caller_turn_sent_ms": timing
            .caller_turn_sent_at
            .saturating_duration_since(timing.finalized_at)
            .as_millis() as u64,
    }) {
        serde_json::Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    let mut guard = services.state.write().await;
    let config_id = guard.quality.config_id.clone();
    let redaction_mode = guard.quality.config.logging.redaction_mode;
    guard.emit_quality_span_finished(
        gateway_call_id,
        QualitySpanEmission {
            config_id,
            redaction_mode,
            span_name: "app.agent_turn_wait",
            category: "model_generation",
            duration: agent_turn_received_at.saturating_duration_since(timing.caller_turn_sent_at),
            critical_path: true,
            concurrent: false,
            payload,
        },
    );
}

pub(crate) async fn send_error_frame(
    handle: &TextCallSessionHandle,
    code: impl Into<String>,
    message: impl Into<String>,
) -> anyhow::Result<()> {
    handle
        .send(GatewayTextFrame::Error {
            code: code.into(),
            message: message.into(),
            sequence: handle.next_sequence(),
        })
        .await
}

pub(crate) async fn send_playback_finished(
    handle: &TextCallSessionHandle,
    turn_id: String,
    status: PlaybackFinishedStatus,
) -> anyhow::Result<()> {
    handle
        .send(GatewayTextFrame::PlaybackFinished {
            turn_id,
            sequence: handle.next_sequence(),
            status,
        })
        .await
}

pub(crate) async fn cancel_append_turn(handle: &TextCallSessionHandle, turn_id: &str) {
    let append_turn = handle.append_turns.lock().await.remove(turn_id);
    if let Some(mut append_turn) = append_turn {
        if let Some(speech) = append_turn.speech.take() {
            speech.cancel().await;
        }
    }
}

async fn finish_turn(
    handle: &TextCallSessionHandle,
    turn_id: String,
    status: PlaybackFinishedStatus,
) -> anyhow::Result<()> {
    cancel_append_turn(handle, &turn_id).await;
    handle.turns.lock().await.remove_turn(&turn_id);
    send_playback_finished(handle, turn_id, status).await
}

async fn finish_superseded_turn(
    handle: &TextCallSessionHandle,
    turn_id: String,
) -> anyhow::Result<()> {
    cancel_append_turn(handle, &turn_id).await;
    handle.turns.lock().await.close_superseded(&turn_id);
    send_playback_finished(handle, turn_id, PlaybackFinishedStatus::Superseded).await
}

pub(crate) async fn send_replaced_playback_canceled(
    handle: &TextCallSessionHandle,
    replaced_playback_id: &str,
) -> anyhow::Result<Option<String>> {
    let replaced_turn_id = handle
        .turns
        .lock()
        .await
        .close_playback(replaced_playback_id);
    if let Some(replaced_turn_id) = replaced_turn_id.as_ref() {
        cancel_append_turn(handle, replaced_turn_id).await;
        send_playback_finished(
            handle,
            replaced_turn_id.clone(),
            PlaybackFinishedStatus::Canceled,
        )
        .await?;
    }
    Ok(replaced_turn_id)
}

struct AgentAppendSpeechRequest {
    speech_output: SpeechOutputConfig,
    gateway_call_id: String,
    initial_chunks: Vec<String>,
    source_label: &'static str,
    config: TextCallSessionConfig,
    turn_finalized_at: Option<Instant>,
    turn_id: Option<String>,
}

async fn queue_append_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    request: AgentAppendSpeechRequest,
) -> anyhow::Result<(speech::AppendSpeechHandle, speech::QueuedSpeech)> {
    let AgentAppendSpeechRequest {
        speech_output,
        gateway_call_id,
        initial_chunks,
        source_label,
        config,
        turn_finalized_at,
        turn_id,
    } = request;
    let media_ready_deadline = Instant::now() + config.media_ready_timeout;
    let playback_ready_deadline = Instant::now() + config.playback_wait_timeout;
    let conflict_policy = if config.latest_response_wins {
        SpeechConflictPolicy::CancelAndReplace
    } else {
        SpeechConflictPolicy::Reject
    };
    let text = initial_chunks.join(" ");
    let tts_backend = speech_output.tts_backend;
    loop {
        match speech::queue_append_speech_with_request(
            &services.state,
            &services.media,
            &services.tts,
            SpeechQueueRequest {
                tts_backend,
                gateway_call_id: gateway_call_id.clone(),
                text: text.clone(),
                source_label: source_label.to_string(),
                conflict_policy,
                turn_finalized_at,
                latest_turn_finalized_at: turn_finalized_at,
                turn_id: turn_id.clone(),
                coalesced_turn_ids: turn_id.iter().cloned().collect(),
                source_asr_session_ids: Vec::new(),
                source_utterance_ids: Vec::new(),
                prebuffer_chunks_override: None,
                speech_output: Some(speech_output),
            },
            initial_chunks.clone(),
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error) => {
                let detail = format!("{error:#}");
                if detail.contains("media stream is not ready")
                    && Instant::now() < media_ready_deadline
                {
                    time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                if !config.latest_response_wins
                    && detail.contains("active speech job")
                    && Instant::now() < playback_ready_deadline
                {
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

async fn queue_agent_speech_with_media_wait(
    services: &TextCallStreamServices,
    speech_output: SpeechOutputConfig,
    gateway_call_id: String,
    text: String,
    config: TextCallSessionConfig,
    turn_finalized_at: Option<Instant>,
    turn_id: Option<String>,
) -> anyhow::Result<speech::QueuedSpeech> {
    let media_ready_deadline = Instant::now() + config.media_ready_timeout;
    let playback_ready_deadline = Instant::now() + config.playback_wait_timeout;
    let conflict_policy = if config.latest_response_wins {
        SpeechConflictPolicy::CancelAndReplace
    } else {
        SpeechConflictPolicy::Reject
    };
    let tts_backend = speech_output.tts_backend;
    loop {
        match speech::queue_speech_with_request(
            &services.state,
            &services.media,
            &services.tts,
            SpeechQueueRequest {
                tts_backend,
                gateway_call_id: gateway_call_id.clone(),
                text: text.clone(),
                source_label: "text-call agent.turn".to_string(),
                conflict_policy,
                turn_finalized_at,
                latest_turn_finalized_at: turn_finalized_at,
                turn_id: turn_id.clone(),
                coalesced_turn_ids: turn_id.iter().cloned().collect(),
                source_asr_session_ids: Vec::new(),
                source_utterance_ids: Vec::new(),
                prebuffer_chunks_override: None,
                speech_output: Some(speech_output),
            },
        )
        .await
        {
            Ok(queued) => return Ok(queued),
            Err(error) => {
                let detail = format!("{error:#}");
                if detail.contains("media stream is not ready")
                    && Instant::now() < media_ready_deadline
                {
                    time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                if !config.latest_response_wins
                    && detail.contains("active speech job")
                    && Instant::now() < playback_ready_deadline
                {
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

pub async fn queue_fallback_and_wait(
    services: &TextCallStreamServices,
    gateway_call_id: String,
    text: String,
) -> anyhow::Result<()> {
    let (config, speech_output) = text_call_session_config(services, &gateway_call_id).await;
    let queued = queue_agent_speech_with_media_wait(
        services,
        speech_output,
        gateway_call_id.clone(),
        text,
        config,
        None,
        None,
    )
    .await?;
    wait_for_playback_terminal_without_turn(
        &services.state,
        &gateway_call_id,
        &queued.playback_id,
        config.playback_wait_timeout,
    )
    .await;
    Ok(())
}

async fn wait_for_playback_terminal_without_turn(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
    playback_wait_timeout: Duration,
) {
    let deadline = Instant::now() + playback_wait_timeout;
    loop {
        if playback_terminal_status(state, gateway_call_id, playback_id)
            .await
            .is_some()
            || Instant::now() >= deadline
        {
            return;
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_playback_terminal(
    state: &SharedState,
    handle: &TextCallSessionHandle,
    gateway_call_id: &str,
    playback_id: &str,
    playback_wait_timeout: Duration,
) -> Option<PlaybackFinishedStatus> {
    let deadline = Instant::now() + playback_wait_timeout;
    loop {
        if !handle.turns.lock().await.is_playback_active(playback_id) {
            return None;
        }
        if let Some(status) = playback_terminal_status(state, gateway_call_id, playback_id).await {
            return Some(status);
        }
        if Instant::now() >= deadline {
            return Some(PlaybackFinishedStatus::Failed);
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn playback_terminal_status(
    state: &SharedState,
    gateway_call_id: &str,
    playback_id: &str,
) -> Option<PlaybackFinishedStatus> {
    let guard = state.read().await;
    let Some(call) = guard.calls.get(gateway_call_id) else {
        return Some(PlaybackFinishedStatus::Failed);
    };
    if matches!(call.status, CallStatus::Ended | CallStatus::Failed) {
        return Some(PlaybackFinishedStatus::Failed);
    }
    call.tts.as_ref().and_then(|tts| {
        if tts.playback_id == playback_id {
            playback_finished_status(tts.status)
        } else {
            None
        }
    })
}

pub(crate) fn playback_finished_status(
    status: TtsPlaybackStatus,
) -> Option<PlaybackFinishedStatus> {
    match status {
        TtsPlaybackStatus::Completed => Some(PlaybackFinishedStatus::Completed),
        TtsPlaybackStatus::Canceled => Some(PlaybackFinishedStatus::Canceled),
        TtsPlaybackStatus::Failed => Some(PlaybackFinishedStatus::Failed),
        TtsPlaybackStatus::Queued
        | TtsPlaybackStatus::Playing
        | TtsPlaybackStatus::MarkSent
        | TtsPlaybackStatus::Canceling => None,
    }
}
