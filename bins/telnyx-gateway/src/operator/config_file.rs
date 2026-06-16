use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::call_control::{TelnyxMediaConfig, TelnyxStreamCodec};
use crate::conversation::ConversationProcessorKind;
use crate::operator::script::expand_user_path;
use crate::operator::state::{GatewayState, InboundMode};
use crate::quality::VoiceQualityConfig;
use crate::tts::LiveTtsBackend;

const DEFAULT_TELNYX_API_BASE: &str = "https://api.telnyx.com/v2";
const DEFAULT_TELNYX_API_KEY_REF: &str = "env:TELNYX_API_KEY";

#[derive(Clone, Debug)]
pub struct LoadedGatewayConfig {
    pub process: ProcessConfig,
    pub telnyx: TelnyxConfig,
    pub gateway: DurableGatewayConfig,
    pub inbound: InboundConfig,
    pub conversation: ConversationConfig,
    pub startup: StartupConfig,
    pub voice_quality: VoiceQualityConfig,
}

impl LoadedGatewayConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let path = expand_user_path(path);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read gateway config {}", path.display()))?;
        let document: GatewayConfigDocument = toml::from_str(&raw)
            .with_context(|| format!("parse gateway config {}", path.display()))?;
        let voice_quality = VoiceQualityConfig::from_toml_str(&raw)
            .with_context(|| format!("parse voice_quality config {}", path.display()))?;
        Self::from_document(document, voice_quality)
            .with_context(|| format!("validate gateway config {}", path.display()))
    }

    pub fn defaults() -> Self {
        Self {
            process: ProcessConfig::default(),
            telnyx: TelnyxConfig::default(),
            gateway: DurableGatewayConfig::default(),
            inbound: InboundConfig::default(),
            conversation: ConversationConfig::default(),
            startup: StartupConfig::default(),
            voice_quality: VoiceQualityConfig::default(),
        }
    }

    fn from_document(
        document: GatewayConfigDocument,
        voice_quality: VoiceQualityConfig,
    ) -> Result<Self> {
        let process = ProcessConfig::from_document(document.process);
        let telnyx = TelnyxConfig::from_document(document.telnyx);
        let gateway = DurableGatewayConfig::from_document(document.gateway)?;
        let inbound = InboundConfig::from_document(document.inbound);
        let conversation_barge_in_explicit = document.conversation.barge_in_enabled.is_some();
        let mut conversation = ConversationConfig::from_document(document.conversation)?;
        if !conversation_barge_in_explicit {
            conversation.barge_in_enabled = voice_quality.barge_in.enabled;
        }
        let startup = StartupConfig::from_document(document.startup);
        Ok(Self {
            process,
            telnyx,
            gateway,
            inbound,
            conversation,
            startup,
            voice_quality,
        })
    }

    pub fn apply_to_state(&self, state: &mut GatewayState) {
        state.config.public_webhook_url = self.gateway.webhook_url.clone();
        state.config.public_media_url = self.gateway.media_url.clone();
        state.config.telnyx_media = self.gateway.media;
        state.config.capture_dir = self.gateway.capture_dir.clone();
        state.config.selected_connection_id = self.telnyx.selected_connection_id.clone();
        state.config.selected_application_name = self.telnyx.selected_application_name.clone();
        state.config.selected_phone_number = self.telnyx.selected_phone_number.clone();
        state.config.default_from_number = self.gateway.from_number.clone();
        state.config.state_path = self.gateway.state_path.clone();
        state.config.telnyx_api_base = self.telnyx.api_base.clone();
        state.config.telnyx_api_key_ref = Some(self.telnyx.api_key_ref.clone());
        state.config.dry_run_telnyx = self.telnyx.dry_run;
        state.config.artifact_root = self.process.artifact_root.clone();
        state.config.log_file = self.process.log_file.clone();
        state.config.tui = self.process.tui;
        state.config.socket = self.process.socket.clone();
        state.config.startup_warm_models = self.startup.warm_models;
        state.inbound_mode = self.inbound.mode;
        state.conversation_tts_backend = self.conversation.tts_backend;
        state.set_quality_config(self.voice_quality.clone());
    }

    pub fn telnyx_api_key(&self) -> Result<Option<String>> {
        resolve_secret_ref(&self.telnyx.api_key_ref)
    }
}

#[derive(Clone, Debug)]
pub struct ProcessConfig {
    pub bind: SocketAddr,
    pub tui: bool,
    pub socket: Option<PathBuf>,
    pub artifact_root: Option<PathBuf>,
    pub log_file: Option<PathBuf>,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080"
                .parse()
                .expect("default bind address is valid"),
            tui: false,
            socket: None,
            artifact_root: None,
            log_file: None,
        }
    }
}

impl ProcessConfig {
    fn from_document(document: ProcessConfigDocument) -> Self {
        let defaults = Self::default();
        Self {
            bind: document.bind.unwrap_or(defaults.bind),
            tui: document.tui.unwrap_or(defaults.tui),
            socket: document.socket.map(|path| expand_user_path(&path)),
            artifact_root: document.artifact_root.map(|path| expand_user_path(&path)),
            log_file: document.log_file.map(|path| expand_user_path(&path)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TelnyxConfig {
    pub api_base: String,
    pub api_key_ref: String,
    pub dry_run: bool,
    pub selected_connection_id: Option<String>,
    pub selected_application_name: Option<String>,
    pub selected_phone_number: Option<String>,
}

impl Default for TelnyxConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_TELNYX_API_BASE.to_string(),
            api_key_ref: DEFAULT_TELNYX_API_KEY_REF.to_string(),
            dry_run: false,
            selected_connection_id: None,
            selected_application_name: None,
            selected_phone_number: None,
        }
    }
}

impl TelnyxConfig {
    fn from_document(document: TelnyxConfigDocument) -> Self {
        let defaults = Self::default();
        Self {
            api_base: document.api_base.unwrap_or(defaults.api_base),
            api_key_ref: document.api_key_ref.unwrap_or(defaults.api_key_ref),
            dry_run: document.dry_run.unwrap_or(defaults.dry_run),
            selected_connection_id: document.selected_connection_id,
            selected_application_name: document.selected_application_name,
            selected_phone_number: document.selected_phone_number,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DurableGatewayConfig {
    pub webhook_url: Option<String>,
    pub media_url: Option<String>,
    pub media: TelnyxMediaConfig,
    pub capture_dir: Option<PathBuf>,
    pub from_number: Option<String>,
    pub state_path: Option<PathBuf>,
}

impl DurableGatewayConfig {
    fn from_document(document: DurableGatewayConfigDocument) -> Result<Self> {
        let defaults = Self::default();
        let codec = document
            .media_codec
            .as_deref()
            .map(TelnyxStreamCodec::from_str)
            .transpose()?
            .unwrap_or(defaults.media.codec);
        let sample_rate_hz = document
            .media_sample_rate_hz
            .unwrap_or_else(|| codec.default_sample_rate_hz());
        Ok(Self {
            webhook_url: document.webhook_url,
            media_url: document.media_url,
            media: TelnyxMediaConfig::new(codec, sample_rate_hz)?,
            capture_dir: document.capture_dir.map(|path| expand_user_path(&path)),
            from_number: document.from_number,
            state_path: document.state_path.map(|path| expand_user_path(&path)),
        })
    }
}

#[derive(Clone, Debug)]
pub struct InboundConfig {
    pub mode: InboundMode,
}

impl Default for InboundConfig {
    fn default() -> Self {
        Self {
            mode: InboundMode::Disabled,
        }
    }
}

impl InboundConfig {
    fn from_document(document: InboundConfigDocument) -> Self {
        Self {
            mode: document.mode.unwrap_or(InboundModeName::Disabled).into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConversationConfig {
    pub enabled: bool,
    pub final_coalescing_enabled: bool,
    pub barge_in_enabled: bool,
    pub processor: ConversationProcessorKind,
    pub tts_backend: LiveTtsBackend,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            final_coalescing_enabled: false,
            barge_in_enabled: true,
            processor: ConversationProcessorKind::Identity,
            tts_backend: LiveTtsBackend::default(),
        }
    }
}

impl ConversationConfig {
    fn from_document(document: ConversationConfigDocument) -> Result<Self> {
        let defaults = Self::default();
        let processor = match document.processor.as_deref().unwrap_or("identity") {
            "identity" => ConversationProcessorKind::Identity,
            other => bail!("unsupported conversation processor {other}; expected identity"),
        };
        let tts_backend = document
            .tts_backend
            .as_deref()
            .map(LiveTtsBackend::from_str)
            .transpose()?
            .unwrap_or(defaults.tts_backend);
        Ok(Self {
            enabled: document.enabled.unwrap_or(defaults.enabled),
            final_coalescing_enabled: document
                .final_coalescing_enabled
                .unwrap_or(defaults.final_coalescing_enabled),
            barge_in_enabled: document
                .barge_in_enabled
                .unwrap_or(defaults.barge_in_enabled),
            processor,
            tts_backend,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct StartupConfig {
    pub warm_models: bool,
}

impl StartupConfig {
    fn from_document(document: StartupConfigDocument) -> Self {
        Self {
            warm_models: document.warm_models.unwrap_or(false),
        }
    }
}

pub fn render_state_toml(state: &GatewayState) -> String {
    toml::to_string_pretty(&SerializableGatewayState::from(state))
        .expect("gateway state TOML serializes")
}

fn resolve_secret_ref(secret_ref: &str) -> Result<Option<String>> {
    let Some(env_name) = secret_ref.strip_prefix("env:") else {
        bail!("unsupported telnyx.api_key_ref {secret_ref}; expected env:<NAME>");
    };
    if env_name.is_empty() {
        bail!("telnyx.api_key_ref env name must be non-empty");
    }
    Ok(std::env::var(env_name).ok())
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct GatewayConfigDocument {
    process: ProcessConfigDocument,
    telnyx: TelnyxConfigDocument,
    gateway: DurableGatewayConfigDocument,
    inbound: InboundConfigDocument,
    conversation: ConversationConfigDocument,
    startup: StartupConfigDocument,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct ProcessConfigDocument {
    bind: Option<SocketAddr>,
    tui: Option<bool>,
    socket: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    log_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct TelnyxConfigDocument {
    api_base: Option<String>,
    api_key_ref: Option<String>,
    dry_run: Option<bool>,
    selected_connection_id: Option<String>,
    selected_application_name: Option<String>,
    selected_phone_number: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct DurableGatewayConfigDocument {
    webhook_url: Option<String>,
    media_url: Option<String>,
    media_codec: Option<String>,
    media_sample_rate_hz: Option<u32>,
    capture_dir: Option<PathBuf>,
    from_number: Option<String>,
    state_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct InboundConfigDocument {
    mode: Option<InboundModeName>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct ConversationConfigDocument {
    enabled: Option<bool>,
    final_coalescing_enabled: Option<bool>,
    barge_in_enabled: Option<bool>,
    processor: Option<String>,
    tts_backend: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct StartupConfigDocument {
    warm_models: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum InboundModeName {
    #[default]
    Disabled,
    Manual,
    AutoTranscribe,
}

impl From<InboundModeName> for InboundMode {
    fn from(mode: InboundModeName) -> Self {
        match mode {
            InboundModeName::Disabled => Self::Disabled,
            InboundModeName::Manual => Self::Manual,
            InboundModeName::AutoTranscribe => Self::AutoTranscribe,
        }
    }
}

impl From<InboundMode> for InboundModeName {
    fn from(mode: InboundMode) -> Self {
        match mode {
            InboundMode::Disabled => Self::Disabled,
            InboundMode::Manual => Self::Manual,
            InboundMode::AutoTranscribe => Self::AutoTranscribe,
        }
    }
}

#[derive(Serialize)]
struct SerializableGatewayState<'a> {
    version: u32,
    generated_at: String,
    process: SerializableProcess<'a>,
    telnyx: SerializableTelnyx<'a>,
    gateway: SerializableGateway<'a>,
    inbound: SerializableInbound,
    conversation: SerializableConversation<'a>,
    startup: SerializableStartup,
    voice_quality: &'a VoiceQualityConfig,
}

impl<'a> From<&'a GatewayState> for SerializableGatewayState<'a> {
    fn from(state: &'a GatewayState) -> Self {
        Self {
            version: 1,
            generated_at: Utc::now().to_rfc3339(),
            process: SerializableProcess {
                bind: state.config.bind,
                tui: state.config.tui,
                socket: state.config.socket.as_ref(),
                artifact_root: state.config.artifact_root.as_ref(),
                log_file: state.config.log_file.as_ref(),
            },
            telnyx: SerializableTelnyx {
                api_base: &state.config.telnyx_api_base,
                api_key_ref: state.config.telnyx_api_key_ref.as_deref(),
                dry_run: state.config.dry_run_telnyx,
                selected_connection_id: state.config.selected_connection_id.as_deref(),
                selected_application_name: state.config.selected_application_name.as_deref(),
                selected_phone_number: state.config.selected_phone_number.as_deref(),
            },
            gateway: SerializableGateway {
                webhook_url: state.config.public_webhook_url.as_deref(),
                media_url: state.config.public_media_url.as_deref(),
                media_codec: state.config.telnyx_media.codec.as_str(),
                media_sample_rate_hz: state.config.telnyx_media.sample_rate_hz,
                capture_dir: state.config.capture_dir.as_ref(),
                from_number: state.config.default_from_number.as_deref(),
                state_path: state.config.state_path.as_ref(),
            },
            inbound: SerializableInbound {
                mode: state.inbound_mode.into(),
            },
            conversation: SerializableConversation {
                enabled: state.config.conversation_enabled,
                final_coalescing_enabled: state.config.conversation_final_coalescing_enabled,
                barge_in_enabled: state.config.conversation_barge_in_enabled,
                processor: state.config.conversation_processor.label(),
                tts_backend: state.conversation_tts_backend.label(),
            },
            startup: SerializableStartup {
                warm_models: state.config.startup_warm_models,
            },
            voice_quality: &state.quality.config,
        }
    }
}

#[derive(Serialize)]
struct SerializableProcess<'a> {
    bind: Option<SocketAddr>,
    tui: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    socket: Option<&'a PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_root: Option<&'a PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    log_file: Option<&'a PathBuf>,
}

#[derive(Serialize)]
struct SerializableTelnyx<'a> {
    api_base: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key_ref: Option<&'a str>,
    dry_run: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_connection_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_application_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_phone_number: Option<&'a str>,
}

#[derive(Serialize)]
struct SerializableGateway<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    webhook_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_url: Option<&'a str>,
    media_codec: &'static str,
    media_sample_rate_hz: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_dir: Option<&'a PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_number: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_path: Option<&'a PathBuf>,
}

#[derive(Serialize)]
struct SerializableInbound {
    mode: InboundModeName,
}

#[derive(Serialize)]
struct SerializableConversation<'a> {
    enabled: bool,
    final_coalescing_enabled: bool,
    barge_in_enabled: bool,
    processor: &'a str,
    tts_backend: &'a str,
}

#[derive(Serialize)]
struct SerializableStartup {
    warm_models: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::GatewayState;

    #[test]
    fn gateway_config_loads_partial_voice_quality() {
        let raw = r#"
[process]
bind = "127.0.0.1:9090"
tui = true

[telnyx]
api_base = "https://api.example.test/v2"
api_key_ref = "env:TELNYX_TEST_KEY"
dry_run = true
selected_connection_id = "conn-1"
selected_phone_number = "+15551234567"

[gateway]
webhook_url = "https://gateway.example/telnyx/webhooks"
media_url = "wss://gateway.example/telnyx/media"
media_codec = "L16"
media_sample_rate_hz = 16000
from_number = "+15557654321"

[inbound]
mode = "auto-transcribe"

[conversation]
enabled = true
processor = "identity"
tts_backend = "piper"

[startup]
warm_models = true

[voice_quality.endpoint]
trailing_silence_ms = 650
"#;
        let document: GatewayConfigDocument = toml::from_str(raw).expect("parse gateway config");
        let voice_quality = VoiceQualityConfig::from_toml_str(raw).expect("parse quality config");
        let config =
            LoadedGatewayConfig::from_document(document, voice_quality).expect("load config");

        assert_eq!(config.process.bind.to_string(), "127.0.0.1:9090");
        assert!(config.process.tui);
        assert_eq!(config.telnyx.api_base, "https://api.example.test/v2");
        assert!(config.telnyx.dry_run);
        assert_eq!(config.gateway.media.codec, TelnyxStreamCodec::L16);
        assert_eq!(config.inbound.mode, InboundMode::AutoTranscribe);
        assert!(config.conversation.enabled);
        assert_eq!(config.conversation.tts_backend, LiveTtsBackend::Piper);
        assert!(config.startup.warm_models);
        assert_eq!(config.voice_quality.endpoint.trailing_silence_ms, 650);
    }

    #[test]
    fn state_dump_renders_readable_toml_with_voice_quality() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        state.config.public_media_url = Some("wss://gateway.example/telnyx/media".to_string());
        state.config.telnyx_api_key_ref = Some("env:TELNYX_API_KEY".to_string());
        state.config.selected_phone_number = Some("+15551234567".to_string());
        state.quality.config.endpoint.trailing_silence_ms = 650;
        state.quality.config_id = state.quality.config.config_id();

        let dump = render_state_toml(&state);
        assert!(dump.contains("[voice_quality.endpoint]"));
        assert!(dump.contains("trailing_silence_ms = 650"));
        assert!(dump.contains("[telnyx]"));
        assert!(dump.contains("api_key_ref = \"env:TELNYX_API_KEY\""));
        assert!(!dump.contains("restore-config"));
    }
}
