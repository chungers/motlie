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
    if state.config.asr_backend != Default::default() {
        lines.push(format!("asr use {}", state.config.asr_backend.label()));
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

pub fn write_state_dump(path: &Path, state: &GatewayState) -> anyhow::Result<()> {
    let dump = render_state_dump(state);
    fs::write(path, dump).with_context(|| format!("write state dump '{}'", path.display()))
}
