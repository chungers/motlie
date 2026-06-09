use std::fs;
use std::path::Path;

use anyhow::Context;

use crate::operator::state::{GatewayState, InboundMode};

pub fn render_state_dump(state: &GatewayState) -> String {
    let mut lines = vec![
        "# motlie telnyx-gateway state v1".to_string(),
        format!("# generated_at {}", chrono::Utc::now().to_rfc3339()),
    ];

    if let Some(url) = &state.config.public_webhook_url {
        lines.push(format!("config set webhook-url {url}"));
    }
    if let Some(url) = &state.config.public_media_url {
        lines.push(format!("config set media-url {url}"));
    }
    if state.config.telnyx_media != Default::default() {
        lines.push(format!(
            "config set media-codec {}",
            state.config.telnyx_media.codec.as_str()
        ));
        lines.push(format!(
            "config set media-sample-rate {}",
            state.config.telnyx_media.sample_rate_hz
        ));
    }
    if let Some(path) = &state.config.capture_dir {
        lines.push(format!("config set capture-dir {}", path.display()));
    }
    if let Some(number) = &state.config.default_from_number {
        lines.push(format!("config set from-number {number}"));
    }
    if let Some(path) = &state.config.state_path {
        lines.push(format!("config set state-path {}", path.display()));
    }
    if let Some(connection_id) = &state.config.selected_connection_id {
        lines.push(format!("telnyx app use {connection_id}"));
    }
    if let (Some(number), Some(connection_id)) = (
        &state.config.selected_phone_number,
        &state.config.selected_connection_id,
    ) {
        lines.push(format!("telnyx number use {number}"));
        lines.push(format!("telnyx number bind {number} {connection_id}"));
    }
    lines.extend(render_quality_dump(state));
    match state.inbound_mode {
        InboundMode::Disabled => lines.push("inbound disable".to_string()),
        InboundMode::Manual => lines.push("inbound enable --manual".to_string()),
        InboundMode::AutoTranscribe => {
            lines.push("inbound enable --auto-transcribe".to_string());
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn render_quality_dump(state: &GatewayState) -> Vec<String> {
    let config = &state.quality.config;
    vec![
        format!("quality profile {}", config.profile.label()),
        format!(
            "quality speech rms-threshold {}",
            config.speech.rms_threshold
        ),
        format!(
            "quality speech peak-threshold {}",
            config.speech.peak_threshold
        ),
        format!(
            "quality speech onset-min-silence-ms {}",
            config.speech.onset_min_silence_ms
        ),
        format!(
            "quality endpoint trailing-silence-ms {}",
            config.endpoint.trailing_silence_ms
        ),
        format!(
            "quality endpoint min-turn-words {}",
            config.endpoint.min_turn_words
        ),
        format!(
            "quality endpoint min-turn-chars {}",
            config.endpoint.min_turn_chars
        ),
        format!(
            "quality endpoint merge-window-ms {}",
            config.endpoint.merge_window_ms
        ),
        format!(
            "quality endpoint max-turn-words {}",
            config.endpoint.max_turn_words
        ),
        format!(
            "quality endpoint max-turn-duration-ms {}",
            config.endpoint.max_turn_duration_ms
        ),
        format!(
            "quality text-call max-active-turns {}",
            config.text_call.max_active_turns
        ),
        format!(
            "quality text-call latest-response-wins {}",
            if config.text_call.latest_response_wins {
                "on"
            } else {
                "off"
            }
        ),
        format!(
            "quality logging include-transcript-text {}",
            if config.logging.include_transcript_text {
                "on"
            } else {
                "off"
            }
        ),
        format!(
            "quality logging redaction-mode {}",
            config.logging.redaction_mode.label()
        ),
        if config.quality_judge.enabled {
            format!(
                "quality judge on --sample-rate {} --model {}",
                config.quality_judge.sample_rate, config.quality_judge.model
            )
        } else {
            "quality judge off".to_string()
        },
    ]
}

pub fn write_state_dump(path: &Path, state: &GatewayState) -> anyhow::Result<()> {
    let dump = render_state_dump(state);
    fs::write(path, dump).with_context(|| format!("write state dump '{}'", path.display()))
}
