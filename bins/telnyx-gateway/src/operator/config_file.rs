use std::borrow::Cow;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use gray_matter::{engine::TOML, Matter, ParsedEntity};
use serde::{Deserialize, Serialize};

use motlie_agent::voice::turn_batching::IdentityTurnBatcherConfig;

use crate::call_control::{TelnyxMediaConfig, TelnyxStreamCodec};
use crate::operator::script::expand_user_path;
use crate::operator::state::{GatewayState, InboundMode};
use crate::processors::ConversationProcessorKind;
use crate::quality::{QualityEventSink, VoiceQualityConfig};
use crate::tts::LiveTtsBackend;

const DEFAULT_TELNYX_API_BASE: &str = "https://api.telnyx.com/v2";
const DEFAULT_TELNYX_API_KEY_REF: &str = "env:TELNYX_API_KEY";
const TOML_FRONT_MATTER_DELIMITER: &str = "+++";

#[derive(Clone, Debug)]
pub struct LoadedGatewayConfig {
    pub process: ProcessConfig,
    pub telnyx: TelnyxConfig,
    pub gateway: DurableGatewayConfig,
    pub inbound: InboundConfig,
    pub conversation: ConversationConfig,
    pub startup: StartupConfig,
    pub voice_quality: VoiceQualityConfig,
    pub quality_logging: QualityLoggingRuntimeConfig,
}

impl LoadedGatewayConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let path = expand_user_path(path);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read gateway config {}", path.display()))?;
        let config_toml = extract_config_toml_front_matter(&raw)
            .with_context(|| format!("extract gateway config front matter {}", path.display()))?;
        let document: GatewayConfigDocument = toml::from_str(config_toml.as_ref())
            .with_context(|| format!("parse gateway config {}", path.display()))?;
        let voice_quality = VoiceQualityConfig::from_toml_str(config_toml.as_ref())
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
            quality_logging: QualityLoggingRuntimeConfig::default(),
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
        // @codex-m6-ds-rv 2026-06-16 PDT -- Transitional precedence note:
        // voice_quality.barge_in is the runtime owner; conversation.barge_in_enabled
        // only preserves current startup ergonomics when omitted from gateway TOML.
        if !conversation_barge_in_explicit {
            conversation.barge_in_enabled = voice_quality.barge_in.enabled;
        }
        let startup = StartupConfig::from_document(document.startup);
        let quality_logging = QualityLoggingRuntimeConfig::from_document(document.quality_logging);
        Ok(Self {
            process,
            telnyx,
            gateway,
            inbound,
            conversation,
            startup,
            voice_quality,
            quality_logging,
        })
    }

    pub fn apply_to_state(&self, state: &mut GatewayState) -> Result<()> {
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
        state.config.conversation_enabled = self.conversation.enabled;
        state.config.conversation_final_coalescing_enabled =
            self.conversation.final_coalescing_enabled;
        state.config.conversation_barge_in_enabled = self.conversation.barge_in_enabled;
        state.config.conversation_processor = self.conversation.processor.clone();
        state.inbound_mode = self.inbound.mode;
        state.conversation_tts_backend = self.conversation.tts_backend;
        state.set_quality_config(self.voice_quality.clone());
        if self.voice_quality.logging.enabled {
            let path = self.quality_logging.path.as_ref().with_context(|| {
                "voice_quality.logging.enabled=true requires [quality_logging].path"
            })?;
            let sink = QualityEventSink::start_jsonl_writer(
                path,
                self.voice_quality.logging.queue_capacity,
            )?;
            state.set_quality_event_sink(sink, Some(path.clone()));
        } else {
            state.set_quality_event_sink(QualityEventSink::disabled(), None);
        }
        Ok(())
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
        let identity_turn_batcher = document.identity_turn_batcher;
        let processor = match document.processor.as_deref().unwrap_or("identity") {
            "identity" => {
                if identity_turn_batcher.is_some() {
                    bail!(
                        "conversation.identity_turn_batcher requires processor = \"turn_batched_identity\""
                    );
                }
                ConversationProcessorKind::Identity
            }
            "turn_batched_identity" | "turn-batched-identity" => {
                ConversationProcessorKind::turn_batched_identity(
                    identity_turn_batcher.unwrap_or_default(),
                )
            }
            "external_text_stream" | "external-text-stream" => {
                if identity_turn_batcher.is_some() {
                    bail!(
                        "conversation.identity_turn_batcher requires processor = \"turn_batched_identity\""
                    );
                }
                ConversationProcessorKind::ExternalTextStream
            }
            other => bail!(
                "unsupported conversation processor {other}; expected identity, turn_batched_identity, or external_text_stream"
            ),
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

#[derive(Clone, Debug, Default)]
pub struct QualityLoggingRuntimeConfig {
    pub path: Option<PathBuf>,
}

impl QualityLoggingRuntimeConfig {
    fn from_document(document: QualityLoggingRuntimeConfigDocument) -> Self {
        Self {
            path: document.path.map(|path| expand_user_path(&path)),
        }
    }
}

pub fn render_state_toml(state: &GatewayState) -> String {
    let quality_log_path = state
        .quality
        .event_sink
        .is_enabled()
        .then_some(state.quality.log_path.as_ref())
        .flatten();
    let mut voice_quality = state.quality.config.clone();
    voice_quality.logging.enabled = quality_log_path.is_some();
    toml::to_string_pretty(&SerializableGatewayState::from_parts(
        state,
        &voice_quality,
        quality_log_path,
    ))
    .expect("gateway state TOML serializes")
}

fn extract_config_toml_front_matter(raw: &str) -> Result<Cow<'_, str>> {
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let mut matter = Matter::<TOML>::new();
    matter.delimiter = TOML_FRONT_MATTER_DELIMITER.to_owned();

    let parsed: ParsedEntity = matter.parse(raw).context("parse TOML front matter")?;
    if !parsed.matter.is_empty() {
        return Ok(Cow::Owned(parsed.matter));
    }
    if starts_with_toml_front_matter(raw) {
        bail!("TOML front matter opened with +++ but no parsed TOML block was found");
    }
    Ok(Cow::Borrowed(raw))
}

fn starts_with_toml_front_matter(raw: &str) -> bool {
    raw.lines()
        .next()
        .is_some_and(|line| line.trim_end_matches('\r') == TOML_FRONT_MATTER_DELIMITER)
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
#[serde(default, deny_unknown_fields)]
struct GatewayConfigDocument {
    #[serde(rename = "version")]
    _version: Option<u32>,
    #[serde(rename = "generated_at")]
    _generated_at: Option<String>,
    process: ProcessConfigDocument,
    telnyx: TelnyxConfigDocument,
    gateway: DurableGatewayConfigDocument,
    inbound: InboundConfigDocument,
    conversation: ConversationConfigDocument,
    startup: StartupConfigDocument,
    #[serde(rename = "voice_quality")]
    _voice_quality: Option<toml::Value>,
    quality_logging: QualityLoggingRuntimeConfigDocument,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ProcessConfigDocument {
    bind: Option<SocketAddr>,
    tui: Option<bool>,
    socket: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    log_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TelnyxConfigDocument {
    api_base: Option<String>,
    api_key_ref: Option<String>,
    dry_run: Option<bool>,
    selected_connection_id: Option<String>,
    selected_application_name: Option<String>,
    selected_phone_number: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
#[serde(default, deny_unknown_fields)]
struct InboundConfigDocument {
    mode: Option<InboundModeName>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConversationConfigDocument {
    enabled: Option<bool>,
    final_coalescing_enabled: Option<bool>,
    barge_in_enabled: Option<bool>,
    processor: Option<String>,
    identity_turn_batcher: Option<IdentityTurnBatcherConfig>,
    tts_backend: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StartupConfigDocument {
    warm_models: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct QualityLoggingRuntimeConfigDocument {
    path: Option<PathBuf>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    quality_logging: Option<SerializableQualityLogging<'a>>,
    voice_quality: &'a VoiceQualityConfig,
}

impl<'a> SerializableGatewayState<'a> {
    fn from_parts(
        state: &'a GatewayState,
        voice_quality: &'a VoiceQualityConfig,
        quality_log_path: Option<&'a PathBuf>,
    ) -> Self {
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
                identity_turn_batcher: state
                    .config
                    .conversation_processor
                    .identity_turn_batcher_config(),
                tts_backend: state.conversation_tts_backend.label(),
            },
            startup: SerializableStartup {
                warm_models: state.config.startup_warm_models,
            },
            quality_logging: quality_log_path.map(|path| SerializableQualityLogging { path }),
            voice_quality,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    identity_turn_batcher: Option<&'a IdentityTurnBatcherConfig>,
    tts_backend: &'a str,
}

#[derive(Serialize)]
struct SerializableStartup {
    warm_models: bool,
}

#[derive(Serialize)]
struct SerializableQualityLogging<'a> {
    path: &'a PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::GatewayState;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_CONFIG_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn write_temp_config(raw: &str) -> PathBuf {
        let sequence = TEMP_CONFIG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "motlie-gateway-config-test-{}-{sequence}.toml",
            std::process::id()
        ));
        std::fs::write(&path, raw).expect("write temp gateway config");
        path
    }

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

    #[test]
    fn gateway_config_rejects_unknown_keys() {
        let raw = r#"
[process]
warm_modelz = true
"#;
        let error = toml::from_str::<GatewayConfigDocument>(raw)
            .expect_err("unknown keys should not be ignored");
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn gateway_config_loads_toml_front_matter_with_appended_report() {
        let raw = r#"+++
[conversation]
enabled = true
barge_in_enabled = false
processor = "identity"
tts_backend = "kokoro-82m"

[voice_quality.tts]
generation_mode = "streaming"
prebuffer_chunks = 1

[voice_quality.early_response]
enabled = true
boundary = "none"
+++
# Run Results

This markdown is intentionally not valid TOML.
[not-toml
"#;
        let path = write_temp_config(raw);
        let config = LoadedGatewayConfig::load(&path).expect("load hybrid front matter config");

        assert!(config.conversation.enabled);
        assert!(!config.conversation.barge_in_enabled);
        assert_eq!(config.conversation.tts_backend, LiveTtsBackend::Kokoro82m);
        assert_eq!(
            config.voice_quality.tts.generation_mode,
            crate::quality::TtsGenerationMode::Streaming
        );
        assert_eq!(config.voice_quality.tts.prebuffer_chunks, 1);
        assert_eq!(config.voice_quality.tts.streaming_start_buffer_ms, 300);
        assert_eq!(config.voice_quality.tts.tail_pad_ms, 200);
        assert!(config.voice_quality.early_response.enabled);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gateway_config_front_matter_still_rejects_unknown_quality_keys() {
        let raw = r#"+++
[voice_quality.tts]
generation_mod = "streaming"
+++
# Run Results

Mentioning generation_mod here is fine because this is not config.
"#;
        let path = write_temp_config(raw);
        let error = LoadedGatewayConfig::load(&path)
            .expect_err("unknown front-matter keys should remain strict");

        assert!(
            format!("{error:?}").contains("unknown field"),
            "unexpected error: {error:?}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gateway_config_front_matter_requires_closing_delimiter() {
        let error = extract_config_toml_front_matter("+++\n[process]\ntui = true\n")
            .expect_err("unterminated front matter should fail");

        assert!(
            error.to_string().contains("no parsed TOML block"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn checked_in_gateway_config_parses_strictly() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("gateway.toml");
        let raw = std::fs::read_to_string(&path).expect("read checked-in gateway config");
        let config = LoadedGatewayConfig::load(&path).expect("load checked-in gateway config");

        assert!(raw.contains("<telnyx-connection-id>"));
        assert!(raw.contains("<telnyx-phone-number>"));
        assert!(raw.contains("<public-host>"));
        assert_eq!(config.telnyx.api_key_ref, "env:TELNYX_API_KEY");
        assert_eq!(
            config.telnyx.selected_connection_id.as_deref(),
            Some("<telnyx-connection-id>")
        );
        assert_eq!(
            config.telnyx.selected_phone_number.as_deref(),
            Some("<telnyx-phone-number>")
        );
        assert_eq!(
            config.gateway.webhook_url.as_deref(),
            Some("https://<public-host>/telnyx/webhooks")
        );
        assert_eq!(
            config.gateway.media_url.as_deref(),
            Some("wss://<public-host>/telnyx/media")
        );
        assert_eq!(
            config.gateway.from_number.as_deref(),
            Some("<telnyx-phone-number>")
        );
        assert_eq!(
            config.voice_quality.tts.generation_mode,
            crate::quality::TtsGenerationMode::Streaming
        );
        assert_eq!(config.voice_quality.tts.streaming_start_buffer_ms, 450);
        assert_eq!(config.voice_quality.tts.tail_pad_ms, 200);
        assert!(config.quality_logging.path.is_some());
        assert!(config.voice_quality.logging.enabled);
    }

    #[test]
    fn docs_live_run_example_configs_parse_strictly() {
        for relative in [
            "docs/LIVE_RUN_CONFIG.example.toml",
            "docs/LIVE_RUN_INBOUND_IDENTITY.example.toml",
            "docs/LIVE_RUN_OUTBOUND_IDENTITY.example.toml",
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
            let raw = std::fs::read_to_string(&path).expect("read docs live-run example config");
            let config =
                LoadedGatewayConfig::load(&path).expect("load docs live-run example config");

            assert!(raw.starts_with("+++\n"), "{relative}");
            assert!(raw.contains("<telnyx-connection-id>"), "{relative}");
            assert!(raw.contains("<telnyx-phone-number>"), "{relative}");
            assert!(raw.contains("<public-host>"), "{relative}");
            assert!(raw.contains("## Run Results"), "{relative}");
            assert_eq!(config.telnyx.api_base, DEFAULT_TELNYX_API_BASE);
            assert_eq!(config.telnyx.api_key_ref, "env:TELNYX_API_KEY");
            assert_eq!(
                config.telnyx.selected_connection_id.as_deref(),
                Some("<telnyx-connection-id>")
            );
            assert_eq!(
                config.telnyx.selected_phone_number.as_deref(),
                Some("<telnyx-phone-number>")
            );
            assert_eq!(
                config.gateway.webhook_url.as_deref(),
                Some("https://<public-host>/telnyx/webhooks")
            );
            assert_eq!(
                config.gateway.media_url.as_deref(),
                Some("wss://<public-host>/telnyx/media")
            );
            assert_eq!(
                config.gateway.from_number.as_deref(),
                Some("<telnyx-phone-number>")
            );
            assert!(config.process.log_file.is_some());
            assert!(config.gateway.capture_dir.is_some());
            assert!(config.gateway.state_path.is_some());
            assert!(config.quality_logging.path.is_some());
            assert_eq!(config.inbound.mode, InboundMode::Manual);
            if relative == "docs/LIVE_RUN_OUTBOUND_IDENTITY.example.toml" {
                assert!(!config.conversation.enabled, "{relative}");
            } else {
                assert!(config.conversation.enabled, "{relative}");
            }
            assert!(config.conversation.final_coalescing_enabled);
            assert!(!config.conversation.barge_in_enabled);
            assert_eq!(config.conversation.tts_backend, LiveTtsBackend::Kokoro82m);
            assert!(config.startup.warm_models);
            assert_eq!(
                config.voice_quality.tts.generation_mode,
                crate::quality::TtsGenerationMode::Streaming
            );
            assert!(config.voice_quality.tts.chunking_enabled);
            assert_eq!(config.voice_quality.tts.max_text_chunk_chars, 70);
            assert_eq!(config.voice_quality.tts.first_chunk_max_chars, 40);
            assert_eq!(config.voice_quality.tts.prebuffer_chunks, 1);
            assert_eq!(config.voice_quality.tts.streaming_start_buffer_ms, 450);
            assert_eq!(config.voice_quality.tts.tail_pad_ms, 200);
            assert!(config.voice_quality.early_response.enabled);
            assert_eq!(config.voice_quality.early_response.debounce_ms, 180);
            assert_eq!(
                config
                    .voice_quality
                    .early_response
                    .max_updates_per_utterance,
                1
            );
            assert!(!config.voice_quality.barge_in.enabled);
            assert!(config.voice_quality.echo_suppression.enabled);
            assert!(config.voice_quality.logging.enabled);
        }
    }

    #[test]
    fn docs_live_run_test_records_parse_strictly() {
        for (relative, expected_barge_in, expected_streaming_start_buffer_ms) in [
            (
                "docs/tests/20260626-163544-7dcbe571-identity-bargein-v1.example.toml",
                true,
                300,
            ),
            (
                "docs/tests/20260628-141804-b5ecbbed-tts-startbuf450-nobarge-v1.example.toml",
                false,
                450,
            ),
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
            let raw = std::fs::read_to_string(&path).expect("read docs live-run test record");
            let config = LoadedGatewayConfig::load(&path).expect("load docs live-run test record");

            assert!(raw.starts_with("+++\n"), "{relative}");
            assert!(raw.contains("<telnyx-connection-id>"), "{relative}");
            assert!(raw.contains("<telnyx-phone-number>"), "{relative}");
            assert!(raw.contains("<public-host>"), "{relative}");
            assert!(raw.contains("## Run Results"), "{relative}");
            assert_eq!(config.telnyx.api_key_ref, "env:TELNYX_API_KEY");
            assert_eq!(
                config.telnyx.selected_connection_id.as_deref(),
                Some("<telnyx-connection-id>")
            );
            assert_eq!(
                config.telnyx.selected_phone_number.as_deref(),
                Some("<telnyx-phone-number>")
            );
            assert_eq!(
                config.gateway.webhook_url.as_deref(),
                Some("https://<public-host>/telnyx/webhooks")
            );
            assert_eq!(
                config.gateway.media_url.as_deref(),
                Some("wss://<public-host>/telnyx/media")
            );
            assert_eq!(
                config.gateway.from_number.as_deref(),
                Some("<telnyx-phone-number>")
            );
            assert!(config.conversation.enabled, "{relative}");
            assert_eq!(
                config.conversation.barge_in_enabled, expected_barge_in,
                "{relative}"
            );
            assert_eq!(
                config.conversation.processor,
                ConversationProcessorKind::Identity
            );
            assert_eq!(config.conversation.tts_backend, LiveTtsBackend::Kokoro82m);
            assert_eq!(
                config.voice_quality.tts.generation_mode,
                crate::quality::TtsGenerationMode::Streaming
            );
            assert_eq!(
                config.voice_quality.tts.streaming_start_buffer_ms,
                expected_streaming_start_buffer_ms,
                "{relative}"
            );
            assert_eq!(config.voice_quality.tts.tail_pad_ms, 200);
            assert_eq!(
                config.voice_quality.barge_in.enabled, expected_barge_in,
                "{relative}"
            );
            assert!(config.voice_quality.logging.enabled, "{relative}");
        }
    }

    #[tokio::test]
    async fn state_dump_round_trips_durable_gateway_config() {
        let log_path = std::env::temp_dir().join(format!(
            "motlie-gateway-config-roundtrip-{}-quality.jsonl",
            std::process::id()
        ));
        let mut state = GatewayState::new("127.0.0.1:9091".parse().expect("valid addr"));
        state.config.public_webhook_url =
            Some("https://gateway.example/telnyx/webhooks".to_string());
        state.config.public_media_url = Some("wss://gateway.example/telnyx/media".to_string());
        state.config.selected_connection_id = Some("conn-1".to_string());
        state.config.selected_application_name = Some("voice-app".to_string());
        state.config.selected_phone_number = Some("+15551234567".to_string());
        state.config.default_from_number = Some("+15557654321".to_string());
        state.config.startup_warm_models = true;
        state.config.conversation_enabled = true;
        state.config.conversation_final_coalescing_enabled = true;
        state.config.conversation_barge_in_enabled = false;
        state.config.conversation_processor = ConversationProcessorKind::Identity;
        state.inbound_mode = InboundMode::AutoTranscribe;
        state.conversation_tts_backend = LiveTtsBackend::Piper;
        state.quality.config.endpoint.trailing_silence_ms = 650;
        state
            .quality
            .config
            .set_tts_generation_mode(crate::quality::TtsGenerationMode::Buffered);
        state.quality.config.logging.enabled = true;
        state.quality.config.logging.queue_capacity = 4;
        state.quality.config_id = state.quality.config.config_id();
        let sink = QualityEventSink::start_jsonl_writer(&log_path, 4)
            .expect("quality log writer should start");
        state.set_quality_event_sink(sink, Some(log_path.clone()));

        let dump = render_state_toml(&state);
        assert!(dump.contains("[quality_logging]"));
        assert!(dump.contains("path ="));

        let document: GatewayConfigDocument = toml::from_str(&dump).expect("parse dumped config");
        let voice_quality = VoiceQualityConfig::from_toml_str(&dump).expect("parse dumped quality");
        let loaded = LoadedGatewayConfig::from_document(document, voice_quality)
            .expect("load dumped config");
        assert!(loaded.conversation.enabled);
        assert!(loaded.conversation.final_coalescing_enabled);
        assert!(!loaded.conversation.barge_in_enabled);
        assert_eq!(
            loaded.conversation.processor,
            ConversationProcessorKind::Identity
        );
        assert_eq!(loaded.conversation.tts_backend, LiveTtsBackend::Piper);
        assert_eq!(loaded.quality_logging.path.as_ref(), Some(&log_path));
        assert!(loaded.voice_quality.logging.enabled);

        let mut restored = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        loaded
            .apply_to_state(&mut restored)
            .expect("apply dumped config");
        assert!(restored.config.conversation_enabled);
        assert!(restored.config.conversation_final_coalescing_enabled);
        assert!(!restored.config.conversation_barge_in_enabled);
        assert_eq!(
            restored.config.conversation_processor,
            ConversationProcessorKind::Identity
        );
        assert_eq!(restored.conversation_tts_backend, LiveTtsBackend::Piper);
        assert_eq!(restored.quality.config.endpoint.trailing_silence_ms, 650);
        assert!(restored.quality.event_sink.is_enabled());
        assert_eq!(restored.quality.log_path.as_ref(), Some(&log_path));

        let _ = std::fs::remove_file(log_path);
    }

    #[test]
    fn gateway_config_loads_turn_batched_identity_config() {
        let raw = r#"
[conversation]
enabled = true
processor = "turn_batched_identity"

[conversation.identity_turn_batcher]
fixed_batch_size = 3
max_batch_turns = 5
max_batch_wait_ms = 250
"#;
        let document: GatewayConfigDocument = toml::from_str(raw).expect("parse gateway config");
        let voice_quality = VoiceQualityConfig::from_toml_str(raw).expect("parse quality config");
        let config =
            LoadedGatewayConfig::from_document(document, voice_quality).expect("load config");
        let expected_config = IdentityTurnBatcherConfig::fixed_batch_size(3)
            .with_max_batch_turns(5)
            .with_max_batch_wait_ms(250);

        assert_eq!(
            config.conversation.processor,
            ConversationProcessorKind::turn_batched_identity(expected_config.clone())
        );

        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        state.config.conversation_processor =
            ConversationProcessorKind::turn_batched_identity(expected_config);
        let dump = render_state_toml(&state);
        assert!(dump.contains("processor = \"turn_batched_identity\""));
        assert!(dump.contains("[conversation.identity_turn_batcher]"));
        assert!(dump.contains("fixed_batch_size = 3"));
        assert!(dump.contains("max_batch_turns = 5"));
        assert!(dump.contains("max_batch_wait_ms = 250"));
    }

    #[test]
    fn gateway_config_rejects_turn_batcher_config_for_other_processors() {
        let raw = r#"
[conversation]
processor = "identity"

[conversation.identity_turn_batcher]
fixed_batch_size = 2
"#;
        let document: GatewayConfigDocument = toml::from_str(raw).expect("parse gateway config");
        let voice_quality = VoiceQualityConfig::from_toml_str(raw).expect("parse quality config");
        let error = LoadedGatewayConfig::from_document(document, voice_quality)
            .expect_err("identity batcher config should be rejected");

        assert!(
            error
                .to_string()
                .contains("conversation.identity_turn_batcher requires processor"),
            "unexpected error: {error}"
        );
    }
}
