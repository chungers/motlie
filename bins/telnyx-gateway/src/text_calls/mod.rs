pub mod offers;
pub mod subscriber_registry;
pub mod turns;
pub mod websocket;

use anyhow::Context;

use crate::call_control::AnswerRequest;
use crate::operator::state::{CallStatus, LogLevel};
use crate::webhook::InboundTextCallTrigger;

use offers::{callback_http_client, CallbackDecision};
use turns::{TextCallDirection, TextCallInfo};
pub use websocket::{SharedTextCallRegistry, TextCallStreamServices};

pub const INBOUND_EXHAUSTED_FALLBACK: &str = "Sorry no one is here to handle your call. bye!";

pub async fn run_inbound_text_call_flow(
    services: TextCallStreamServices,
    trigger: InboundTextCallTrigger,
) {
    if let Err(error) = run_inbound_text_call_flow_inner(services.clone(), trigger.clone()).await {
        let mut guard = services.state.write().await;
        guard.log(
            LogLevel::Warn,
            format!(
                "inbound text-call flow failed for {}: {error:#}",
                trigger.gateway_call_id
            ),
        );
        if let Some(call) = guard.calls.get_mut(&trigger.gateway_call_id) {
            call.status = CallStatus::Failed;
            call.last_error = Some(format!("{error:#}"));
            call.push_timeline("inbound text-call flow failed");
        }
    }
}

async fn run_inbound_text_call_flow_inner(
    services: TextCallStreamServices,
    trigger: InboundTextCallTrigger,
) -> anyhow::Result<()> {
    let client = callback_http_client()?;
    let subscribers =
        subscriber_registry::ordered_subscribers_for_phone(&services.state, trigger.to.as_deref())
            .await;
    if subscribers.is_empty() {
        return Ok(());
    }

    let call = TextCallInfo {
        id: trigger.gateway_call_id.clone(),
        direction: TextCallDirection::Inbound,
        from: trigger.from.clone(),
        to: trigger.to.clone(),
    };
    let callback_timeout = {
        let guard = services.state.read().await;
        guard.quality.config.text_call.callback_timeout()
    };

    for subscription in subscribers {
        let attempt =
            offers::send_inbound_offer(&client, &subscription, call.clone(), callback_timeout)
                .await;
        match attempt.decision {
            CallbackDecision::Accept {
                call_url,
                emit_partials,
                emit_early_turns,
                response_mode,
            } => {
                let setup = websocket::TextCallSetup {
                    gateway_call_id: trigger.gateway_call_id.clone(),
                    call_url,
                    direction: TextCallDirection::Inbound,
                    emit_partials,
                    emit_early_turns,
                    response_mode,
                };
                if let Err(error) =
                    websocket::connect_application_stream(services.clone(), setup).await
                {
                    log_flow_note(
                        &services,
                        &trigger.gateway_call_id,
                        format!(
                            "accepted subscriber {} text stream failed: {error:#}",
                            subscription.id
                        ),
                    )
                    .await;
                    continue;
                }
                match answer_telnyx_call(&services, &trigger).await {
                    Ok(()) => {
                        log_flow_note(
                            &services,
                            &trigger.gateway_call_id,
                            format!("inbound text-call accepted by {}", subscription.id),
                        )
                        .await;
                        return Ok(());
                    }
                    Err(error) => {
                        services
                            .registry
                            .send_session_end(&trigger.gateway_call_id, "answer_failed")
                            .await;
                        log_flow_note(
                            &services,
                            &trigger.gateway_call_id,
                            format!(
                                "accepted subscriber {} could not be answered: {error:#}",
                                subscription.id
                            ),
                        )
                        .await;
                    }
                }
            }
            CallbackDecision::Decline => {
                log_flow_note(
                    &services,
                    &trigger.gateway_call_id,
                    format!("inbound text-call declined by {}", subscription.id),
                )
                .await;
            }
            CallbackDecision::Failed { reason } => {
                log_flow_note(
                    &services,
                    &trigger.gateway_call_id,
                    format!(
                        "inbound text-call offer {} failed: {reason}",
                        subscription.id
                    ),
                )
                .await;
            }
        }
    }

    answer_telnyx_call(&services, &trigger).await?;
    websocket::queue_fallback_and_wait(
        &services,
        trigger.gateway_call_id.clone(),
        INBOUND_EXHAUSTED_FALLBACK.to_string(),
    )
    .await
    .context("speak inbound text-call fallback")?;
    websocket::hangup_gateway_call(
        &services,
        &trigger.gateway_call_id,
        "inbound text-call exhausted",
    )
    .await?;
    Ok(())
}

async fn answer_telnyx_call(
    services: &TextCallStreamServices,
    trigger: &InboundTextCallTrigger,
) -> anyhow::Result<()> {
    let (stream_url, media) = {
        let guard = services.state.read().await;
        (
            guard
                .config
                .public_media_url
                .clone()
                .context("missing public media WebSocket URL")?,
            guard.config.telnyx_media,
        )
    };
    services
        .telnyx
        .answer_call(&AnswerRequest {
            call_control_id: &trigger.call_control_id,
            stream_url: &stream_url,
            media,
        })
        .await?;
    let mut guard = services.state.write().await;
    if let Some(call) = guard.calls.get_mut(&trigger.gateway_call_id) {
        call.status = CallStatus::Answering;
        call.push_timeline("answering for text-call subscriber");
    }
    Ok(())
}

async fn log_flow_note(services: &TextCallStreamServices, gateway_call_id: &str, message: String) {
    let mut guard = services.state.write().await;
    guard.log(LogLevel::Info, message.clone());
    if let Some(call) = guard.calls.get_mut(gateway_call_id) {
        call.push_timeline(message);
    }
}
