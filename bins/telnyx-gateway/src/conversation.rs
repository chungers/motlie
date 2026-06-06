use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, Context};
use async_trait::async_trait;
use motlie_voice::app::{
    CallContext, CallIds, ConversationCommand, ConversationHandler, TranscriptEvent, VoiceAppError,
};
use motlie_voice::telephony::CallAction;

use crate::call_control::TelnyxClient;
use crate::media::SharedMediaRegistry;
use crate::operator::state::{ConversationMode, LogLevel, SharedState};
use crate::speech;
use crate::tts::{LiveTtsBackend, SharedTtsRegistry};

pub type SharedConversationHandler = Arc<dyn ConversationHandler>;

#[derive(Clone)]
pub struct ConversationRuntime {
    telnyx: TelnyxClient,
    tts: SharedTtsRegistry,
    handler: SharedConversationHandler,
}

impl ConversationRuntime {
    pub fn new(
        telnyx: TelnyxClient,
        tts: SharedTtsRegistry,
        handler: SharedConversationHandler,
    ) -> Self {
        Self {
            telnyx,
            tts,
            handler,
        }
    }
}

pub fn default_conversation_handler() -> SharedConversationHandler {
    Arc::new(FirstDuplexConversationHandler)
}

#[derive(Clone, Debug, Default)]
pub struct FirstDuplexConversationHandler;

#[async_trait]
impl ConversationHandler for FirstDuplexConversationHandler {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        _context: &mut CallContext,
    ) -> Result<ConversationCommand, VoiceAppError> {
        if !event.is_final() {
            return Ok(ConversationCommand::Noop);
        }
        let text = event.text().trim();
        if text.is_empty() {
            return Ok(ConversationCommand::Noop);
        }
        Ok(ConversationCommand::Say {
            text: format!("I heard: {text}"),
        })
    }
}

pub async fn handle_final_transcript(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    event: TranscriptEvent,
) -> anyhow::Result<()> {
    if !event.is_final() {
        return Ok(());
    }
    let user_text = event.text().trim().to_string();
    if user_text.is_empty() {
        return Ok(());
    }

    let Some(snapshot) = conversation_snapshot(state, gateway_call_id).await else {
        return Ok(());
    };
    if !snapshot.attached {
        return Ok(());
    }

    state
        .write()
        .await
        .record_conversation_user_turn(gateway_call_id, user_text);

    if let Some(playback_id) = media_registry
        .active_speech_playback_id(gateway_call_id)
        .await
    {
        speech::cancel_speech(
            state,
            media_registry,
            gateway_call_id,
            "conversation barge-in",
        )
        .await?;
        state
            .write()
            .await
            .record_conversation_interrupted(gateway_call_id, &playback_id);
        tracing::info!(
            gateway_call_id,
            playback_id,
            "conversation.barge_in.cancel_requested"
        );
    }

    let mut context = snapshot.context;
    let command = runtime
        .handler
        .on_transcript(event, &mut context)
        .await
        .map_err(|error| anyhow::anyhow!(error))?;
    apply_conversation_command(
        state,
        media_registry,
        runtime,
        gateway_call_id,
        snapshot.mode,
        snapshot.call_control_id,
        command,
    )
    .await
}

struct ConversationSnapshot {
    attached: bool,
    mode: ConversationMode,
    call_control_id: String,
    context: CallContext,
}

async fn conversation_snapshot(
    state: &SharedState,
    gateway_call_id: &str,
) -> Option<ConversationSnapshot> {
    let guard = state.read().await;
    let call = guard.calls.get(gateway_call_id)?;
    let mut custom_state = BTreeMap::new();
    custom_state.insert("gateway_call_id".to_string(), call.gateway_call_id.clone());
    custom_state.insert(
        "conversation_mode".to_string(),
        call.conversation.mode.label().to_string(),
    );
    if let Some(text) = &call.conversation.last_assistant_text {
        custom_state.insert("last_assistant_text".to_string(), text.clone());
    }
    Some(ConversationSnapshot {
        attached: call.conversation.attached,
        mode: call.conversation.mode,
        call_control_id: call.ids.call_control_id.clone(),
        context: CallContext {
            ids: Some(CallIds {
                provider_call_id: call.ids.call_control_id.clone(),
                provider_session_id: call.ids.call_session_id.clone(),
                media_stream_id: call.ids.stream_id.clone(),
            }),
            custom_state,
        },
    })
}

async fn apply_conversation_command(
    state: &SharedState,
    media_registry: &SharedMediaRegistry,
    runtime: &ConversationRuntime,
    gateway_call_id: &str,
    mode: ConversationMode,
    call_control_id: String,
    command: ConversationCommand,
) -> anyhow::Result<()> {
    match command {
        ConversationCommand::Noop => {
            state
                .write()
                .await
                .record_conversation_idle(gateway_call_id);
            Ok(())
        }
        ConversationCommand::Say { text } => {
            let response_text = text.trim().to_string();
            if response_text.is_empty() {
                state
                    .write()
                    .await
                    .record_conversation_idle(gateway_call_id);
                return Ok(());
            }
            match mode {
                ConversationMode::Manual => {
                    state
                        .write()
                        .await
                        .record_conversation_proposal(gateway_call_id, response_text);
                    Ok(())
                }
                ConversationMode::Auto => {
                    // M3's media-triggered conversation path is intentionally
                    // Piper-locked for the first Sherpa+Piper full-duplex pairing.
                    let queued = speech::queue_speech(
                        state,
                        media_registry,
                        &runtime.tts,
                        LiveTtsBackend::Piper,
                        gateway_call_id.to_string(),
                        response_text.clone(),
                        "conversation say",
                    )
                    .await
                    .with_context(|| format!("queue conversation response for {gateway_call_id}"));
                    match queued {
                        Ok(queued) => {
                            state.write().await.record_conversation_speaking(
                                gateway_call_id,
                                response_text,
                                queued.playback_id.clone(),
                            );
                            tracing::info!(
                                gateway_call_id,
                                playback_id = queued.playback_id,
                                "conversation.say.queued"
                            );
                            Ok(())
                        }
                        Err(error) => {
                            let error = format!("{error:#}");
                            state
                                .write()
                                .await
                                .record_conversation_failed(gateway_call_id, error.clone());
                            bail!(error)
                        }
                    }
                }
            }
        }
        ConversationCommand::Call(action) => match action {
            CallAction::Hangup => {
                runtime
                    .telnyx
                    .hangup_call(&call_control_id)
                    .await
                    .with_context(|| format!("hang up call {gateway_call_id}"))?;
                let mut guard = state.write().await;
                guard.record_conversation_idle(gateway_call_id);
                guard.log(
                    LogLevel::Info,
                    format!("conversation hangup requested for {gateway_call_id}"),
                );
                Ok(())
            }
            _ => {
                let error = "unsupported conversation call action".to_string();
                state
                    .write()
                    .await
                    .record_conversation_failed(gateway_call_id, error.clone());
                bail!(error)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::{shared_state, CallStatus, ConversationStatus, TelnyxIds};

    fn test_runtime() -> ConversationRuntime {
        ConversationRuntime::new(
            TelnyxClient::new("https://api.example.test", None, true),
            crate::tts::unavailable_registry(),
            default_conversation_handler(),
        )
    }

    async fn seed_conversation_call(state: &SharedState, mode: ConversationMode) -> String {
        let mut guard = state.write().await;
        let call_id = guard.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: "call-control-1".to_string(),
                call_session_id: Some("session-1".to_string()),
                call_leg_id: Some("leg-1".to_string()),
                stream_id: Some("stream-1".to_string()),
            },
            None,
            None,
            CallStatus::MediaStarted,
        );
        guard.attach_conversation(&call_id, mode);
        call_id
    }

    #[tokio::test]
    async fn default_handler_turns_final_transcript_into_say() {
        let handler = FirstDuplexConversationHandler;
        let mut context = CallContext {
            ids: None,
            custom_state: BTreeMap::new(),
        };
        let command = handler
            .on_transcript(
                TranscriptEvent::Final {
                    text: "hello".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                &mut context,
            )
            .await
            .expect("default handler should accept final transcript");

        match command {
            ConversationCommand::Say { text } => assert_eq!(text, "I heard: hello"),
            _ => panic!("expected say command"),
        }
    }

    #[tokio::test]
    async fn default_handler_ignores_partial_transcript() {
        let handler = FirstDuplexConversationHandler;
        let mut context = CallContext {
            ids: None,
            custom_state: BTreeMap::new(),
        };
        let command = handler
            .on_transcript(
                TranscriptEvent::Partial {
                    text: "hello".to_string(),
                    update: motlie_model::TranscriptionUpdate::default(),
                },
                &mut context,
            )
            .await
            .expect("default handler should accept partial transcript");

        match command {
            ConversationCommand::Noop => {}
            _ => panic!("expected noop command"),
        }
    }

    #[tokio::test]
    async fn apply_manual_say_records_proposal_without_speaking() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        let runtime = test_runtime();

        apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationMode::Manual,
            "call-control-1".to_string(),
            ConversationCommand::Say {
                text: "  assistant response  ".to_string(),
            },
        )
        .await
        .expect("manual say should record a proposal");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Proposed);
        assert_eq!(
            call.conversation.last_assistant_text.as_deref(),
            Some("assistant response")
        );
        assert!(call.conversation.last_playback_id.is_none());
        assert!(call.tts.is_none());
    }

    #[tokio::test]
    async fn apply_noop_marks_conversation_idle() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Manual).await;
        state
            .write()
            .await
            .record_conversation_user_turn(&gateway_call_id, "user turn".to_string());
        let runtime = test_runtime();

        apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationMode::Manual,
            "call-control-1".to_string(),
            ConversationCommand::Noop,
        )
        .await
        .expect("noop should mark idle");

        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Idle);
    }

    #[tokio::test]
    async fn apply_unsupported_call_action_fails_closed() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = seed_conversation_call(&state, ConversationMode::Auto).await;
        let runtime = test_runtime();

        let error = apply_conversation_command(
            &state,
            &SharedMediaRegistry::default(),
            &runtime,
            &gateway_call_id,
            ConversationMode::Auto,
            "call-control-1".to_string(),
            ConversationCommand::Call(CallAction::Transfer {
                destination: "sip:agent@example.test".to_string(),
            }),
        )
        .await
        .expect_err("unsupported action should fail");

        assert!(format!("{error:#}").contains("unsupported conversation call action"));
        let guard = state.read().await;
        let call = guard.calls.get(&gateway_call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Failed);
        assert_eq!(
            call.conversation.last_error.as_deref(),
            Some("unsupported conversation call action")
        );
    }
}
