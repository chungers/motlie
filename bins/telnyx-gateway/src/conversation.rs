use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, Context};
use async_trait::async_trait;
use motlie_voice::app::{
    CallContext, CallIds, ConversationCommand, ConversationHandler, TranscriptEvent,
    TranscriptSink, VoiceAppError,
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

pub struct ConversationTranscriptSink {
    handler: SharedConversationHandler,
}

impl ConversationTranscriptSink {
    pub fn new(handler: SharedConversationHandler) -> Self {
        Self { handler }
    }
}

#[async_trait]
impl TranscriptSink for ConversationTranscriptSink {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        context: &mut CallContext,
    ) -> Result<Vec<CallAction>, VoiceAppError> {
        if !event.is_final() {
            return Ok(Vec::new());
        }
        match self.handler.on_transcript(event, context).await? {
            ConversationCommand::Call(action) => Ok(vec![action]),
            ConversationCommand::Say { .. } | ConversationCommand::Noop => Ok(Vec::new()),
        }
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
}
