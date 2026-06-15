use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::adapter::{AsrRegistry, EchoAsrFactory, LiveAsrBackend, SharedAsrRegistry};
use crate::call_control::{
    AnswerRequest, DialRequest, TelnyxClient, TelnyxMediaConfig, TelnyxStreamCodec,
};
use crate::conversation::{default_conversation_handler, ConversationRuntime};
use crate::media::SharedMediaRegistry;
#[cfg(test)]
use crate::media::{OutboundMediaCommand, SpeechCancelToken};
use crate::operator::persistence::write_state_dump;
use crate::operator::session::OperatorSession;
use crate::operator::state::{
    asr_warm_key, tts_warm_key, CallStatus, ConversationMode, ConversationStatus, GatewayState,
    InboundMode, LogLevel, SharedState,
};
use crate::quality::{
    OnsetDuringPlaybackPolicy, QualityEventSink, QualityProfile, RedactionMode, TtsGenerationMode,
    VoiceQualityConfig,
};
use crate::speech;
use crate::text_calls::{SharedTextCallRegistry, TextCallStreamServices};
use crate::tts::{unavailable_registry, LiveTtsBackend, SharedTtsRegistry};
use async_trait::async_trait;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use motlie_driver::{CommandOutput, CommandSet, DriverError, DriverResult};

#[derive(Clone)]
pub struct GatewayContext {
    pub state: SharedState,
    pub telnyx: TelnyxClient,
    pub asr: SharedAsrRegistry,
    pub media: SharedMediaRegistry,
    pub tts: SharedTtsRegistry,
    pub conversation: ConversationRuntime,
    pub text_calls: SharedTextCallRegistry,
    pub session: OperatorSession,
}

impl GatewayContext {
    pub fn new(state: SharedState, telnyx: TelnyxClient) -> Self {
        let media = SharedMediaRegistry::default();
        let echo = Arc::new(EchoAsrFactory);
        let asr = Arc::new(AsrRegistry::new(echo.clone(), echo));
        let tts = unavailable_registry();
        let conversation = ConversationRuntime::new(
            telnyx.clone(),
            tts.clone(),
            default_conversation_handler(),
            false,
        );
        Self::with_services(
            state,
            telnyx,
            asr,
            media,
            tts,
            conversation,
            LiveAsrBackend::default(),
        )
    }

    pub fn with_services(
        state: SharedState,
        telnyx: TelnyxClient,
        asr: SharedAsrRegistry,
        media: SharedMediaRegistry,
        tts: SharedTtsRegistry,
        conversation: ConversationRuntime,
        next_asr_backend: LiveAsrBackend,
    ) -> Self {
        Self {
            state,
            telnyx,
            asr,
            media,
            tts,
            conversation,
            text_calls: SharedTextCallRegistry::default(),
            session: OperatorSession::new(next_asr_backend),
        }
    }

    pub fn with_text_calls(mut self, text_calls: SharedTextCallRegistry) -> Self {
        self.text_calls = text_calls;
        self
    }

    pub fn text_call_services(&self) -> TextCallStreamServices {
        TextCallStreamServices {
            registry: self.text_calls.clone(),
            state: self.state.clone(),
            media: self.media.clone(),
            tts: self.tts.clone(),
            telnyx: self.telnyx.clone(),
        }
    }

    pub fn for_new_source(&self) -> Self {
        let mut context = Self::with_services(
            self.state.clone(),
            self.telnyx.clone(),
            self.asr.clone(),
            self.media.clone(),
            self.tts.clone(),
            self.conversation.clone(),
            self.session.next_asr_backend,
        )
        .with_text_calls(self.text_calls.clone());
        context.session.next_tts_backend = self.session.next_tts_backend;
        context
    }

    async fn answer_call(&mut self, target: Option<String>) -> DriverResult<CommandOutput> {
        let asr_backend = self.session.next_asr_backend;
        let (gateway_call_id, call_control_id, stream_url, media) = {
            let mut guard = self.state.write().await;
            let media_url = guard
                .config
                .public_media_url
                .clone()
                .ok_or_else(|| DriverError::message("config set media-url <wss-url> first"))?;
            let media = guard.config.telnyx_media;
            let call_id = resolve_answer_call_id(
                &guard,
                target.as_deref(),
                self.session.selected_call.as_deref(),
            )?;
            let (gateway_call_id, call_control_id) = {
                let call = guard
                    .calls
                    .get_mut(&call_id)
                    .ok_or_else(|| DriverError::NotFound {
                        kind: "call",
                        name: call_id.clone(),
                    })?;
                if call.status != CallStatus::PendingInbound {
                    return Err(DriverError::message(format!(
                        "call {} is {}, expected waiting",
                        call.gateway_call_id,
                        call.status.label()
                    )));
                }
                call.asr_backend = Some(asr_backend);
                call.status = CallStatus::Answering;
                call.push_timeline("operator requested answer");
                call.push_timeline(format!("asr backend -> {}", asr_backend.model_label()));
                (
                    call.gateway_call_id.clone(),
                    call.ids.call_control_id.clone(),
                )
            };
            guard.attach_conversation(&gateway_call_id, ConversationMode::Auto);
            (gateway_call_id, call_control_id, media_url, media)
        };
        self.session.selected_call = Some(gateway_call_id.clone());

        {
            let mut guard = self.state.write().await;
            guard.log(
                LogLevel::Info,
                format!("answering inbound call {gateway_call_id}"),
            );
        }

        self.telnyx
            .answer_call(&AnswerRequest {
                call_control_id: &call_control_id,
                stream_url: &stream_url,
                media,
            })
            .await
            .map_err(driver_anyhow)?;

        tracing::info!(
            gateway_call_id,
            call_control_id,
            stream_url,
            stream_codec = media.codec.as_str(),
            stream_sample_rate_hz = media.sample_rate_hz,
            asr_backend = asr_backend.label(),
            asr_model = asr_backend.model_label(),
            "call.answering"
        );
        Ok(CommandOutput::line(format!(
            "answer requested for {gateway_call_id}"
        )))
    }
}

#[derive(Debug, Parser)]
struct GatewayRoot {
    #[command(subcommand)]
    command: GatewayCommand,
}

#[derive(Debug, Subcommand)]
pub enum GatewayCommand {
    Status(CallTarget),
    Listener {
        #[command(subcommand)]
        command: ListenerCommand,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    State {
        #[command(subcommand)]
        command: StateCommand,
    },
    Shutdown(ShutdownArgs),
    Telnyx {
        #[command(subcommand)]
        command: TelnyxCommand,
    },
    Inbound {
        #[command(subcommand)]
        command: InboundCommand,
    },
    Asr {
        #[command(subcommand)]
        command: AsrCommand,
    },
    Tts {
        #[command(subcommand)]
        command: TtsCommand,
    },
    Conversation {
        #[command(subcommand)]
        command: ConversationCommand,
    },
    Calls,
    Call {
        #[command(subcommand)]
        command: CallCommand,
    },
    Dial(DialArgs),
    Speak(SpeakArgs),
    Answer(CallTarget),
    Reject(CallTarget),
    Hangup(CallTarget),
    Warm(WarmArgs),
    Test {
        #[command(subcommand)]
        command: TestCommand,
    },
    Transcript {
        #[command(subcommand)]
        command: TranscriptCommand,
    },
    Log {
        #[command(subcommand)]
        command: LogCommand,
    },
    Quality {
        #[command(subcommand)]
        command: QualityCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum ListenerCommand {
    Status,
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    Show,
    Set { key: String, value: String },
}

#[derive(Debug, Subcommand)]
pub enum StateCommand {
    Dump { path: PathBuf },
}

#[derive(Debug, Args)]
pub struct ShutdownArgs {
    pub dump_path: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum TelnyxCommand {
    App {
        #[command(subcommand)]
        command: TelnyxAppCommand,
    },
    Number {
        #[command(subcommand)]
        command: TelnyxNumberCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum TelnyxAppCommand {
    List,
    Create {
        name: String,
    },
    Use {
        connection_id: String,
    },
    Show,
    Webhook {
        #[command(subcommand)]
        command: TelnyxAppWebhookCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum TelnyxAppWebhookCommand {
    Set { url: String },
}

#[derive(Debug, Subcommand)]
pub enum TelnyxNumberCommand {
    List,
    Use { e164: String },
    Bind { e164: String, connection_id: String },
}

#[derive(Debug, Subcommand)]
pub enum InboundCommand {
    Status,
    Enable(InboundEnableArgs),
    Disable,
}

#[derive(Debug, Subcommand)]
pub enum AsrCommand {
    List,
    Status,
    Use {
        #[arg(value_enum)]
        backend: LiveAsrBackend,
    },
}

#[derive(Debug, Subcommand)]
pub enum TtsCommand {
    List,
    Status,
    Use {
        #[arg(value_enum)]
        backend: LiveTtsBackend,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum WarmTarget {
    All,
    Asr,
    Tts,
}

impl WarmTarget {
    fn includes_asr(self) -> bool {
        matches!(self, Self::All | Self::Asr)
    }

    fn includes_tts(self) -> bool {
        matches!(self, Self::All | Self::Tts)
    }
}

#[derive(Debug, Args)]
pub struct WarmArgs {
    #[arg(value_enum, default_value_t = WarmTarget::All)]
    pub target: WarmTarget,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConversationModeArg {
    Manual,
    Auto,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConversationSmokeTestArg {
    On,
    Off,
}

impl ConversationSmokeTestArg {
    fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConversationBargeInArg {
    On,
    Off,
    Status,
}

impl ConversationBargeInArg {
    fn enabled(self) -> Option<bool> {
        match self {
            Self::On => Some(true),
            Self::Off => Some(false),
            Self::Status => None,
        }
    }
}

impl From<ConversationModeArg> for ConversationMode {
    fn from(mode: ConversationModeArg) -> Self {
        match mode {
            ConversationModeArg::Manual => Self::Manual,
            ConversationModeArg::Auto => Self::Auto,
        }
    }
}

#[derive(Debug, Subcommand)]
#[command(disable_help_subcommand = true)]
pub enum ConversationCommand {
    Help,
    Status {
        call: Option<String>,
    },
    SmokeTest {
        #[arg(value_enum)]
        state: ConversationSmokeTestArg,
    },
    BargeIn {
        #[arg(value_enum)]
        state: Option<ConversationBargeInArg>,
    },
    Attach {
        call: Option<String>,
    },
    Detach {
        call: Option<String>,
    },
    Disapprove {
        call: Option<String>,
    },
    Approve {
        call: Option<String>,
    },
    Say {
        call: Option<String>,
    },
    Mode {
        #[arg(value_enum)]
        mode: ConversationModeArg,
        call: Option<String>,
    },
}

#[derive(Debug, Args)]
pub struct InboundEnableArgs {
    #[arg(long)]
    manual: bool,
    #[arg(long)]
    auto_transcribe: bool,
}

#[derive(Debug, Subcommand)]
pub enum CallCommand {
    Show { call: Option<String> },
    Use { call: String },
}

#[derive(Debug, Args)]
pub struct CallTarget {
    pub call: Option<String>,
}

#[derive(Debug, Args)]
pub struct DialArgs {
    pub to: String,
    #[arg(long)]
    pub from: Option<String>,
}

#[derive(Debug, Args)]
pub struct SpeakArgs {
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    pub parts: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum TestCommand {
    DialTranscribe(DialTranscribeArgs),
}

#[derive(Debug, Args)]
pub struct DialTranscribeArgs {
    pub to: String,
    #[arg(long)]
    pub from: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum TranscriptCommand {
    Follow { call: Option<String> },
    Clear { call: Option<String> },
}

#[derive(Debug, Subcommand)]
pub enum LogCommand {
    Clear,
}

#[derive(Debug, Subcommand)]
pub enum QualityCommand {
    Status,
    #[command(hide = true)]
    RestoreConfig {
        encoded: String,
    },
    Profile {
        profile: QualityProfileArg,
    },
    Endpoint {
        #[command(subcommand)]
        command: QualityEndpointCommand,
    },
    Speech {
        #[command(subcommand)]
        command: QualitySpeechCommand,
    },
    Asr {
        #[command(subcommand)]
        command: QualityAsrCommand,
    },
    TextCall {
        #[command(subcommand)]
        command: QualityTextCallCommand,
    },
    Tts {
        #[command(subcommand)]
        command: QualityTtsCommand,
    },
    Logging {
        #[command(subcommand)]
        command: QualityLoggingCommand,
    },
    Judge {
        #[command(subcommand)]
        command: QualityJudgeCommand,
    },
    BargeIn {
        #[command(subcommand)]
        command: QualityBargeInCommand,
    },
    EchoSuppression {
        #[command(subcommand)]
        command: QualityEchoSuppressionCommand,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QualityProfileArg {
    Fast,
    Balanced,
    Complete,
    Noisy,
}

impl From<QualityProfileArg> for QualityProfile {
    fn from(value: QualityProfileArg) -> Self {
        match value {
            QualityProfileArg::Fast => Self::Fast,
            QualityProfileArg::Balanced => Self::Balanced,
            QualityProfileArg::Complete => Self::Complete,
            QualityProfileArg::Noisy => Self::Noisy,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OnOffArg {
    On,
    Off,
}

impl OnOffArg {
    fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TtsGenerationModeArg {
    Buffered,
    Streaming,
}

impl From<TtsGenerationModeArg> for TtsGenerationMode {
    fn from(value: TtsGenerationModeArg) -> Self {
        match value {
            TtsGenerationModeArg::Buffered => Self::Buffered,
            TtsGenerationModeArg::Streaming => Self::Streaming,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OnsetDuringPlaybackArg {
    DeferToPartial,
    Trust,
}

impl From<OnsetDuringPlaybackArg> for OnsetDuringPlaybackPolicy {
    fn from(value: OnsetDuringPlaybackArg) -> Self {
        match value {
            OnsetDuringPlaybackArg::DeferToPartial => Self::DeferToPartial,
            OnsetDuringPlaybackArg::Trust => Self::Trust,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum RedactionModeArg {
    MetricsOnly,
    HashedText,
    RedactedText,
    SensitivePlaintext,
}

impl From<RedactionModeArg> for RedactionMode {
    fn from(value: RedactionModeArg) -> Self {
        match value {
            RedactionModeArg::MetricsOnly => Self::MetricsOnly,
            RedactionModeArg::HashedText => Self::HashedText,
            RedactionModeArg::RedactedText => Self::RedactedText,
            RedactionModeArg::SensitivePlaintext => Self::SensitivePlaintext,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum QualityEndpointCommand {
    Status,
    TrailingSilenceMs { ms: u64 },
    MinTurnWords { n: usize },
    MinTurnChars { n: usize },
    MergeWindowMs { ms: u64 },
    FinalSettleMs { ms: u64 },
    ConversationIncompleteTailHoldMs { ms: u64 },
    ConversationLowConfidenceThresholdPercent { percent: u64 },
    ConversationPlaybackHoldPollMs { ms: u64 },
    MaxTurnWords { n: usize },
    MaxTurnDurationMs { ms: u64 },
}

#[derive(Debug, Subcommand)]
pub enum QualitySpeechCommand {
    Status,
    RmsThreshold { value: f32 },
    PeakThreshold { value: i32 },
    OnsetMinSilenceMs { ms: u64 },
}

#[derive(Debug, Subcommand)]
pub enum QualityAsrCommand {
    Status,
    FinishPadMs { ms: u64 },
    RepeatedTokenRunThreshold { n: usize },
    RepeatedQRunThreshold { n: usize },
}

#[derive(Debug, Subcommand)]
pub enum QualityTextCallCommand {
    Status,
    MaxActiveTurns { n: usize },
    MediaReadyTimeoutMs { ms: u64 },
    PlaybackWaitTimeoutMs { ms: u64 },
    LatestResponseWins { state: OnOffArg },
    CallbackTimeoutMs { ms: u64 },
}

#[derive(Debug, Subcommand)]
pub enum QualityTtsCommand {
    Status,
    GenerationMode { mode: TtsGenerationModeArg },
    Chunking { state: OnOffArg },
    MaxTextChunkChars { n: usize },
    FirstChunkMaxChars { n: usize },
    PrebufferChunks { n: usize },
}

#[derive(Debug, Subcommand)]
pub enum QualityLoggingCommand {
    Status,
    On { path: PathBuf },
    Off,
    IncludeTranscriptText { state: OnOffArg },
    RedactionMode { mode: RedactionModeArg },
}

#[derive(Debug, Subcommand)]
pub enum QualityJudgeCommand {
    Status,
    On {
        #[arg(long)]
        sample_rate: Option<f32>,
        #[arg(long)]
        model: Option<String>,
    },
    Off,
}

#[derive(Debug, Subcommand)]
pub enum QualityBargeInCommand {
    Status,
    On,
    Off,
    SpeechOnset { state: OnOffArg },
    OnsetDuringPlayback { policy: OnsetDuringPlaybackArg },
    PartialAsr { state: OnOffArg },
    FinalAsr { state: OnOffArg },
    ClearTimeoutMs { ms: u64 },
}

#[derive(Debug, Subcommand)]
pub enum QualityEchoSuppressionCommand {
    Status,
    On,
    Off,
    MinTextChars { n: usize },
    TailWindowMs { ms: u64 },
    ShortTokenCoveragePercent { n: u64 },
    ShortLongestTokenRun { n: usize },
    LongMinTokens { n: usize },
    LongTokenCoveragePercent { n: u64 },
    LongLongestTokenRun { n: usize },
}

#[async_trait]
impl CommandSet<GatewayContext> for GatewayCommand {
    type CompletionContext = ();
    type Resolved = Self;

    fn root_command() -> clap::Command {
        GatewayRoot::command().name("telnyx-gateway")
    }

    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self> {
        Ok(GatewayRoot::from_arg_matches(matches)?.command)
    }

    fn completion_context(_context: &GatewayContext) -> Self::CompletionContext {}

    fn help(topic: &[String]) -> Option<String> {
        gateway_help(topic)
    }

    fn resolve_command(self, _context: &GatewayContext) -> DriverResult<Self::Resolved> {
        Ok(self)
    }

    async fn execute(
        resolved: Self::Resolved,
        context: &mut GatewayContext,
    ) -> DriverResult<CommandOutput> {
        match resolved {
            Self::Status(target) => status(context, target.call).await,
            Self::Listener {
                command: ListenerCommand::Status,
            } => listener_status(&context.state).await,
            Self::Config { command } => config_command(context, command).await,
            Self::State { command } => state_command(context, command).await,
            Self::Shutdown(args) => shutdown(context, args).await,
            Self::Telnyx { command } => telnyx_command(context, command).await,
            Self::Inbound { command } => inbound_command(context, command).await,
            Self::Asr { command } => asr_command(context, command).await,
            Self::Tts { command } => tts_command(context, command).await,
            Self::Conversation { command } => conversation_command(context, command).await,
            Self::Calls => calls(&context.state).await,
            Self::Call { command } => call_command(context, command).await,
            Self::Dial(args) => dial(context, args).await,
            Self::Speak(args) => speak(context, args).await,
            Self::Answer(target) => context.answer_call(target.call).await,
            Self::Reject(target) => call_control(context, target.call, CallControlOp::Reject).await,
            Self::Hangup(target) => call_control(context, target.call, CallControlOp::Hangup).await,
            Self::Warm(args) => warm_command(context, args).await,
            Self::Test { command } => test_command(context, command).await,
            Self::Transcript { command } => transcript_command(context, command).await,
            Self::Log { command } => log_command(context, command).await,
            Self::Quality { command } => quality_command(context, command).await,
        }
    }
}

async fn status(context: &GatewayContext, target: Option<String>) -> DriverResult<CommandOutput> {
    if target.is_some() {
        return call_show(context, target).await;
    }
    let guard = context.state.read().await;
    let mut lines = vec![
        format!(
            "listener: {}",
            guard
                .config
                .bind
                .map(|addr| addr.to_string())
                .unwrap_or_else(|| "<disabled>".to_string())
        ),
        format!("inbound: {}", guard.inbound_mode.label()),
        format!(
            "asr-next: {} ({})",
            context.session.next_asr_backend.label(),
            context.session.next_asr_backend.model_label()
        ),
        format!(
            "asr-default: {} ({})",
            LiveAsrBackend::default().label(),
            LiveAsrBackend::default().model_label()
        ),
        format!(
            "conversation-handler: {}",
            context.conversation.handler_label()
        ),
        format!(
            "conversation-tts: {} ({}) {}",
            guard.conversation_tts_backend.label(),
            guard.conversation_tts_backend.model_label(),
            tts_availability_label(&context.tts.factory(guard.conversation_tts_backend))
        ),
        format!("conversation-barge-in: {}", quality_barge_in_label(&guard)),
        format!(
            "webhook-url: {}",
            guard
                .config
                .public_webhook_url
                .as_deref()
                .unwrap_or("<unset>")
        ),
        format!(
            "media-url: {}",
            guard
                .config
                .public_media_url
                .as_deref()
                .unwrap_or("<unset>")
        ),
        format!(
            "telnyx-app: {}",
            guard
                .config
                .selected_connection_id
                .as_deref()
                .unwrap_or("<unset>")
        ),
        format!(
            "phone-number: {}",
            guard
                .config
                .selected_phone_number
                .as_deref()
                .unwrap_or("<unset>")
        ),
        format!(
            "media-codec: {} {}Hz",
            guard.config.telnyx_media.codec.as_str(),
            guard.config.telnyx_media.sample_rate_hz
        ),
        format!(
            "capture-dir: {}",
            guard
                .config
                .capture_dir
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string())
        ),
        format!(
            "tts-next: {} ({}) {}",
            context.session.next_tts_backend.label(),
            context.session.next_tts_backend.model_label(),
            tts_availability_label(&context.tts.factory(context.session.next_tts_backend))
        ),
        format!("calls: {}", guard.calls.len()),
    ];
    if guard.inbound_mode == InboundMode::Disabled {
        lines.push("inbound calls will not be answered until enabled".to_string());
    }
    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn listener_status(state: &SharedState) -> DriverResult<CommandOutput> {
    let guard = state.read().await;
    Ok(CommandOutput::line(format!(
        "listening on {}",
        guard
            .config
            .bind
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| "<unknown>".to_string())
    )))
}

async fn config_command(
    context: &mut GatewayContext,
    command: ConfigCommand,
) -> DriverResult<CommandOutput> {
    match command {
        ConfigCommand::Show => {
            let guard = context.state.read().await;
            Ok(CommandOutput::text(format!(
                "webhook-url={}\nmedia-url={}\nmedia-codec={}\nmedia-sample-rate={}\ncapture-dir={}\nfrom-number={}\nstate-path={}",
                guard
                    .config
                    .public_webhook_url
                    .as_deref()
                    .unwrap_or("<unset>"),
                guard
                    .config
                    .public_media_url
                    .as_deref()
                    .unwrap_or("<unset>"),
                guard.config.telnyx_media.codec.as_str(),
                guard.config.telnyx_media.sample_rate_hz,
                guard
                    .config
                    .capture_dir
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<unset>".to_string()),
                guard
                    .config
                    .default_from_number
                    .as_deref()
                    .unwrap_or("<unset>"),
                guard
                    .config
                    .state_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<unset>".to_string())
            )))
        }
        ConfigCommand::Set { key, value } => {
            let mut guard = context.state.write().await;
            match key.as_str() {
                "webhook-url" => guard.config.public_webhook_url = Some(value.clone()),
                "media-url" => guard.config.public_media_url = Some(value.clone()),
                "media-codec" => {
                    let codec = value.parse::<TelnyxStreamCodec>().map_err(driver_anyhow)?;
                    guard.config.telnyx_media =
                        TelnyxMediaConfig::new(codec, codec.default_sample_rate_hz())
                            .map_err(driver_anyhow)?;
                }
                "media-sample-rate" => {
                    let sample_rate_hz = value.parse::<u32>().map_err(|error| {
                        DriverError::invalid_argument(
                            "value",
                            format!("invalid media sample rate {value}: {error}"),
                        )
                    })?;
                    guard.config.telnyx_media =
                        TelnyxMediaConfig::new(guard.config.telnyx_media.codec, sample_rate_hz)
                            .map_err(driver_anyhow)?;
                }
                "capture-dir" => guard.config.capture_dir = Some(expand_user_path(&value)),
                "from-number" => guard.config.default_from_number = Some(value.clone()),
                "state-path" => guard.config.state_path = Some(expand_user_path(&value)),
                _ => {
                    return Err(DriverError::invalid_argument(
                        "key",
                        format!("unknown config key {key}"),
                    ));
                }
            }
            guard.log(LogLevel::Info, format!("config set {key} {value}"));
            Ok(CommandOutput::line(format!("set {key}")))
        }
    }
}

async fn state_command(
    context: &mut GatewayContext,
    command: StateCommand,
) -> DriverResult<CommandOutput> {
    match command {
        StateCommand::Dump { path } => {
            let guard = context.state.read().await;
            write_state_dump(&path, &guard).map_err(driver_anyhow)?;
            Ok(CommandOutput::line(format!(
                "wrote state dump {}",
                path.display()
            )))
        }
    }
}

async fn shutdown(context: &mut GatewayContext, args: ShutdownArgs) -> DriverResult<CommandOutput> {
    let mut guard = context.state.write().await;
    if let Some(path) = args.dump_path.or_else(|| guard.config.state_path.clone()) {
        write_state_dump(&path, &guard).map_err(driver_anyhow)?;
    }
    guard.shutdown_requested = true;
    Ok(CommandOutput::line("shutdown requested")
        .with_effect(motlie_driver::CommandEffect::ExitShell))
}

async fn telnyx_command(
    context: &mut GatewayContext,
    command: TelnyxCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TelnyxCommand::App { command } => telnyx_app_command(context, command).await,
        TelnyxCommand::Number { command } => telnyx_number_command(context, command).await,
    }
}

async fn telnyx_app_command(
    context: &mut GatewayContext,
    command: TelnyxAppCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TelnyxAppCommand::List => {
            let apps = context
                .telnyx
                .list_applications()
                .await
                .map_err(driver_anyhow)?;
            let lines = if apps.is_empty() {
                vec!["no call control applications returned".to_string()]
            } else {
                apps.into_iter()
                    .map(|app| {
                        format!(
                            "{} {} webhook={}",
                            app.id,
                            app.application_name
                                .unwrap_or_else(|| "<unnamed>".to_string()),
                            app.webhook_event_url
                                .unwrap_or_else(|| "<unset>".to_string())
                        )
                    })
                    .collect()
            };
            Ok(CommandOutput {
                lines,
                effects: Vec::new(),
            })
        }
        TelnyxAppCommand::Create { name } => {
            let webhook_url = context
                .state
                .read()
                .await
                .config
                .public_webhook_url
                .clone()
                .ok_or_else(|| DriverError::message("config set webhook-url <https-url> first"))?;
            let app = context
                .telnyx
                .create_application(&name, &webhook_url)
                .await
                .map_err(driver_anyhow)?;
            let mut guard = context.state.write().await;
            guard.config.selected_connection_id = Some(app.id.clone());
            guard.config.selected_application_name = app.application_name.clone();
            guard.log(LogLevel::Info, format!("created Telnyx app {}", app.id));
            Ok(CommandOutput::line(format!("created app {}", app.id)))
        }
        TelnyxAppCommand::Use { connection_id } => {
            let app = if context.telnyx.dry_run() {
                None
            } else {
                Some(
                    context
                        .telnyx
                        .retrieve_application(&connection_id)
                        .await
                        .map_err(driver_anyhow)?,
                )
            };
            let mut guard = context.state.write().await;
            guard.config.selected_connection_id = Some(connection_id.clone());
            guard.config.selected_application_name =
                app.and_then(|value| value.application_name.clone());
            guard.log(
                LogLevel::Info,
                format!("selected Telnyx app {connection_id}"),
            );
            Ok(CommandOutput::line(format!("selected app {connection_id}")))
        }
        TelnyxAppCommand::Show => {
            let guard = context.state.read().await;
            Ok(CommandOutput::text(format!(
                "connection_id={}\nname={}",
                guard
                    .config
                    .selected_connection_id
                    .as_deref()
                    .unwrap_or("<unset>"),
                guard
                    .config
                    .selected_application_name
                    .as_deref()
                    .unwrap_or("<unknown>")
            )))
        }
        TelnyxAppCommand::Webhook {
            command: TelnyxAppWebhookCommand::Set { url },
        } => {
            let connection_id = context
                .state
                .read()
                .await
                .config
                .selected_connection_id
                .clone()
                .ok_or_else(|| DriverError::message("telnyx app use <connection-id> first"))?;
            let app = context
                .telnyx
                .update_application_webhook(&connection_id, &url)
                .await
                .map_err(driver_anyhow)?;
            let mut guard = context.state.write().await;
            guard.config.public_webhook_url = Some(url.clone());
            guard.config.selected_application_name = app.application_name;
            guard.log(LogLevel::Info, format!("set Telnyx webhook URL {url}"));
            Ok(CommandOutput::line("webhook updated"))
        }
    }
}

async fn telnyx_number_command(
    context: &mut GatewayContext,
    command: TelnyxNumberCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TelnyxNumberCommand::List => {
            let numbers = context
                .telnyx
                .list_phone_numbers()
                .await
                .map_err(driver_anyhow)?;
            let lines = if numbers.is_empty() {
                vec!["no phone numbers returned".to_string()]
            } else {
                numbers
                    .into_iter()
                    .map(|number| {
                        format!(
                            "{} {} connection={}",
                            number.id,
                            number
                                .phone_number
                                .unwrap_or_else(|| "<unknown>".to_string()),
                            number
                                .connection_id
                                .unwrap_or_else(|| "<unset>".to_string())
                        )
                    })
                    .collect()
            };
            Ok(CommandOutput {
                lines,
                effects: Vec::new(),
            })
        }
        TelnyxNumberCommand::Use { e164 } => {
            let mut guard = context.state.write().await;
            guard.config.selected_phone_number = Some(e164.clone());
            guard.log(LogLevel::Info, format!("selected phone number {e164}"));
            Ok(CommandOutput::line(format!("selected number {e164}")))
        }
        TelnyxNumberCommand::Bind {
            e164,
            connection_id,
        } => {
            let number = context
                .telnyx
                .bind_phone_number(&e164, &connection_id)
                .await
                .map_err(driver_anyhow)?;
            let mut guard = context.state.write().await;
            guard.config.selected_phone_number = Some(e164.clone());
            guard.config.selected_connection_id = Some(connection_id.clone());
            guard.log(
                LogLevel::Info,
                format!("bound phone number {e164} to {connection_id}"),
            );
            Ok(CommandOutput::line(format!(
                "bound {} to {}",
                number.phone_number.unwrap_or(e164),
                connection_id
            )))
        }
    }
}

async fn inbound_command(
    context: &mut GatewayContext,
    command: InboundCommand,
) -> DriverResult<CommandOutput> {
    match command {
        InboundCommand::Status => {
            let guard = context.state.read().await;
            Ok(CommandOutput::line(format!(
                "inbound {}",
                guard.inbound_mode.label()
            )))
        }
        InboundCommand::Enable(args) => {
            let mode = if args.auto_transcribe {
                InboundMode::AutoTranscribe
            } else {
                InboundMode::Manual
            };
            let mut guard = context.state.write().await;
            guard.inbound_mode = mode;
            guard.log(LogLevel::Info, format!("inbound enabled {}", mode.label()));
            Ok(CommandOutput::line(format!(
                "inbound enabled {}",
                mode.label()
            )))
        }
        InboundCommand::Disable => {
            let mut guard = context.state.write().await;
            guard.inbound_mode = InboundMode::Disabled;
            guard.log(LogLevel::Info, "inbound disabled");
            Ok(CommandOutput::line("inbound disabled"))
        }
    }
}

async fn asr_command(
    context: &mut GatewayContext,
    command: AsrCommand,
) -> DriverResult<CommandOutput> {
    match command {
        AsrCommand::List => Ok(CommandOutput {
            lines: LiveAsrBackend::available()
                .into_iter()
                .map(|backend| format!("{} {}", backend.label(), backend.model_label()))
                .collect(),
            effects: Vec::new(),
        }),
        AsrCommand::Status => {
            let available = LiveAsrBackend::available()
                .into_iter()
                .map(|backend| backend.label())
                .collect::<Vec<_>>()
                .join(",");
            Ok(CommandOutput::text(format!(
                "next={}\nnext_model={}\ndefault={}\ndefault_model={}\navailable={available}",
                context.session.next_asr_backend.label(),
                context.session.next_asr_backend.model_label(),
                LiveAsrBackend::default().label(),
                LiveAsrBackend::default().model_label()
            )))
        }
        AsrCommand::Use { backend } => {
            context.session.next_asr_backend = backend;
            let mut guard = context.state.write().await;
            guard.log(
                LogLevel::Info,
                format!(
                    "source selected ASR backend {} ({}) for next calls",
                    backend.label(),
                    backend.model_label()
                ),
            );
            Ok(CommandOutput::line(format!(
                "asr backend for next calls: {} ({})",
                backend.label(),
                backend.model_label()
            )))
        }
    }
}

async fn tts_command(
    context: &mut GatewayContext,
    command: TtsCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TtsCommand::List => Ok(CommandOutput {
            lines: LiveTtsBackend::available()
                .into_iter()
                .map(|backend| {
                    let factory = context.tts.factory(backend);
                    if let Some(reason) = factory.unavailable_reason() {
                        format!(
                            "{} {} unavailable: {}",
                            backend.label(),
                            backend.model_label(),
                            reason
                        )
                    } else {
                        format!("{} {} available", backend.label(), backend.model_label())
                    }
                })
                .collect(),
            effects: Vec::new(),
        }),
        TtsCommand::Status => {
            let guard = context.state.read().await;
            let active = guard
                .calls
                .values()
                .filter_map(|call| {
                    call.tts.as_ref().map(|tts| {
                        format!(
                            "active-call {} {} backend={} playback={} frames={}/{} text={}",
                            call.gateway_call_id,
                            tts.status.label(),
                            tts.backend.label(),
                            tts.playback_id,
                            tts.frames_sent,
                            tts.frames_queued,
                            tts.text_preview
                        )
                    })
                })
                .collect::<Vec<_>>();
            let backend = context.session.next_tts_backend;
            let conversation_backend = guard.conversation_tts_backend;
            let factory = context.tts.factory(backend);
            let available = LiveTtsBackend::available()
                .into_iter()
                .filter(|backend| context.tts.factory(*backend).is_available())
                .map(|backend| backend.label())
                .collect::<Vec<_>>();
            let mut lines = vec![
                format!("next={}", backend.label()),
                format!("next_model={}", backend.model_label()),
                format!("default={}", LiveTtsBackend::default().label()),
                format!("default_model={}", LiveTtsBackend::default().model_label()),
                format!("conversation={}", conversation_backend.label()),
                format!("conversation_model={}", conversation_backend.model_label()),
                format!(
                    "available={}",
                    if available.is_empty() {
                        "<none>".to_string()
                    } else {
                        available.join(",")
                    }
                ),
                format!(
                    "status={}",
                    if factory.is_available() {
                        "available"
                    } else {
                        "unavailable"
                    }
                ),
            ];
            if let Some(reason) = factory.unavailable_reason() {
                lines.push(format!("reason={reason}"));
            }
            lines.push(format!("active={}", active.len()));
            lines.extend(active);
            Ok(CommandOutput {
                lines,
                effects: Vec::new(),
            })
        }
        TtsCommand::Use { backend } => {
            context.session.next_tts_backend = backend;
            let factory = context.tts.factory(backend);
            let mut guard = context.state.write().await;
            guard.conversation_tts_backend = backend;
            guard.log(
                LogLevel::Info,
                format!(
                    "source selected TTS backend {} ({}) for next speech and conversation replies",
                    backend.label(),
                    backend.model_label()
                ),
            );
            let mut lines = vec![format!(
                "tts backend for next speech and conversation replies: {} ({})",
                backend.label(),
                backend.model_label()
            )];
            if let Some(reason) = factory.unavailable_reason() {
                lines.push(format!("unavailable: {reason}"));
            }
            Ok(CommandOutput {
                lines,
                effects: Vec::new(),
            })
        }
    }
}

fn tts_availability_label(factory: &crate::tts::SharedTtsFactory) -> &'static str {
    if factory.is_available() {
        "available"
    } else {
        "unavailable"
    }
}

async fn warm_command(context: &mut GatewayContext, args: WarmArgs) -> DriverResult<CommandOutput> {
    let mut lines = Vec::new();
    if args.target.includes_asr() {
        lines.push(warm_asr_model(context).await);
    }
    if args.target.includes_tts() {
        lines.push(warm_tts_model(context).await);
    }
    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn warm_asr_model(context: &mut GatewayContext) -> String {
    let backend = context.session.next_asr_backend;
    let started = Instant::now();
    match context.asr.warm(backend).await {
        Ok(()) => {
            let elapsed_ms = elapsed_ms(started);
            let mut guard = context.state.write().await;
            guard.mark_model_warm(
                asr_warm_key(backend),
                backend.label(),
                backend.model_label(),
                elapsed_ms,
            );
            guard.log(
                LogLevel::Info,
                format!(
                    "warmed ASR backend {} ({}) in {elapsed_ms}ms",
                    backend.label(),
                    backend.model_label()
                ),
            );
            format!(
                "warm asr={} model={} status=ready elapsed_ms={elapsed_ms}",
                backend.label(),
                backend.model_label()
            )
        }
        Err(error) => {
            let message = format!("{error:#}");
            context.state.write().await.log(
                LogLevel::Warn,
                format!(
                    "failed to warm ASR backend {} ({}): {message}",
                    backend.label(),
                    backend.model_label()
                ),
            );
            format!(
                "warm asr={} model={} status=failed error={message}",
                backend.label(),
                backend.model_label()
            )
        }
    }
}

async fn warm_tts_model(context: &mut GatewayContext) -> String {
    let backend = context.session.next_tts_backend;
    let generation_mode = context
        .state
        .read()
        .await
        .quality
        .config
        .tts
        .generation_mode;
    let started = Instant::now();
    let warm_result = match generation_mode {
        TtsGenerationMode::Buffered => context.tts.warm(backend).await,
        TtsGenerationMode::Streaming => context.tts.warm_streaming(backend).await,
    };
    match warm_result {
        Ok(()) => {
            let elapsed_ms = elapsed_ms(started);
            let mut guard = context.state.write().await;
            guard.mark_model_warm(
                tts_warm_key(backend),
                backend.label(),
                backend.model_label(),
                elapsed_ms,
            );
            guard.log(
                LogLevel::Info,
                format!(
                    "warmed TTS backend {} ({}) mode={} in {elapsed_ms}ms",
                    backend.label(),
                    backend.model_label(),
                    generation_mode.label()
                ),
            );
            format!(
                "warm tts={} model={} mode={} status=ready elapsed_ms={elapsed_ms}",
                backend.label(),
                backend.model_label(),
                generation_mode.label()
            )
        }
        Err(error) => {
            let message = format!("{error:#}");
            context.state.write().await.log(
                LogLevel::Warn,
                format!(
                    "failed to warm TTS backend {} ({}) mode={}: {message}",
                    backend.label(),
                    backend.model_label(),
                    generation_mode.label()
                ),
            );
            format!(
                "warm tts={} model={} mode={} status=failed error={message}",
                backend.label(),
                backend.model_label(),
                generation_mode.label()
            )
        }
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

async fn dial(context: &mut GatewayContext, args: DialArgs) -> DriverResult<CommandOutput> {
    let asr_backend = context.session.next_asr_backend;
    let (connection_id, from, stream_url, webhook_url, media) =
        outbound_dial_config(context, args.from.clone()).await?;

    let dialed = context
        .telnyx
        .dial_call(&DialRequest {
            connection_id: &connection_id,
            to: &args.to,
            from: &from,
            stream_url: &stream_url,
            webhook_url: webhook_url.as_deref(),
            media,
        })
        .await
        .map_err(|error| driver_anyhow(outbound_prereq_context(error)))?;

    let gateway_call_id = register_outbound_dial(
        context,
        &dialed,
        Some(from.clone()),
        Some(args.to.clone()),
        asr_backend,
        "outbound dial requested",
    )
    .await;
    context.session.selected_call = Some(gateway_call_id.clone());

    tracing::info!(
        gateway_call_id,
        call_control_id = dialed.call_control_id,
        call_session_id = dialed.call_session_id.as_deref(),
        call_leg_id = dialed.call_leg_id.as_deref(),
        to = args.to,
        from,
        stream_url,
        stream_codec = media.codec.as_str(),
        stream_sample_rate_hz = media.sample_rate_hz,
        asr_backend = asr_backend.label(),
        asr_model = asr_backend.model_label(),
        "call.outbound.dial"
    );
    Ok(CommandOutput {
        lines: vec![
            format!("dial requested for {gateway_call_id}"),
            format!("selected call: {gateway_call_id}"),
            "conversation attached in auto mode; smoke-test echo replies require:".to_string(),
            "  conversation smoke-test on".to_string(),
            format!("manual TTS smoke test: speak {gateway_call_id} <text to say>"),
            "use `status` or `call show` to inspect TTS playback; use `speak cancel` to clear active speech".to_string(),
        ],
        effects: Vec::new(),
    })
}

async fn speak(context: &mut GatewayContext, args: SpeakArgs) -> DriverResult<CommandOutput> {
    let guard = context.state.read().await;
    let parsed = parse_speak_args(
        &guard,
        &args.parts,
        context.session.selected_call.as_deref(),
    )?;
    drop(guard);
    match parsed {
        SpeakRequest::Cancel { call_id } => cancel_speech(context, call_id).await,
        SpeakRequest::Text { call_id, text } => start_speech(context, call_id, text).await,
    }
}

async fn start_speech(
    context: &mut GatewayContext,
    gateway_call_id: String,
    text: String,
) -> DriverResult<CommandOutput> {
    if context.text_calls.contains(&gateway_call_id).await {
        return Err(DriverError::message(format!(
            "manual speak is disabled while a text-call stream is attached for {gateway_call_id}; send agent.turn over the stream or detach it first"
        )));
    }
    let queued = speech::queue_speech(
        &context.state,
        &context.media,
        &context.tts,
        context.session.next_tts_backend,
        gateway_call_id.clone(),
        text,
        "speak",
    )
    .await
    .map_err(driver_anyhow)?;
    Ok(CommandOutput::line(format!(
        "speak queued for {gateway_call_id} playback={}",
        queued.playback_id
    )))
}

async fn cancel_speech(
    context: &mut GatewayContext,
    gateway_call_id: String,
) -> DriverResult<CommandOutput> {
    let playback_id =
        speech::cancel_speech(&context.state, &context.media, &gateway_call_id, "speak")
            .await
            .map_err(driver_anyhow)?;
    Ok(CommandOutput::line(format!(
        "speak cancel requested for {gateway_call_id} playback={playback_id}"
    )))
}

enum SpeakRequest {
    Text { call_id: String, text: String },
    Cancel { call_id: String },
}

fn parse_speak_args(
    state: &GatewayState,
    parts: &[String],
    selected_call: Option<&str>,
) -> DriverResult<SpeakRequest> {
    if parts.first().is_some_and(|part| part == "cancel") {
        if parts.len() > 2 {
            return Err(DriverError::message("usage: speak cancel [call-id]"));
        }
        let call_id = resolve_call_id(state, parts.get(1).map(String::as_str), selected_call)?;
        return Ok(SpeakRequest::Cancel { call_id });
    }

    let (call_id, text_parts) = if parts
        .first()
        .is_some_and(|candidate| state.calls.contains_key(candidate))
    {
        let call_id = parts
            .first()
            .cloned()
            .ok_or_else(|| DriverError::message("usage: speak [call-id] <text...>"))?;
        (call_id, &parts[1..])
    } else {
        (resolve_call_id(state, None, selected_call)?, parts)
    };
    if text_parts.is_empty() {
        return Err(DriverError::message("usage: speak [call-id] <text...>"));
    }
    Ok(SpeakRequest::Text {
        call_id,
        text: text_parts.join(" "),
    })
}

async fn outbound_dial_config(
    context: &GatewayContext,
    requested_from: Option<String>,
) -> DriverResult<(String, String, String, Option<String>, TelnyxMediaConfig)> {
    let guard = context.state.read().await;
    let connection_id = guard
        .config
        .selected_connection_id
        .clone()
        .ok_or_else(|| DriverError::message("telnyx app use <connection-id> first"))?;
    let from = requested_from
        .or_else(|| guard.config.default_from_number.clone())
        .or_else(|| guard.config.selected_phone_number.clone())
        .ok_or_else(|| {
            DriverError::message(
                "pass --from <e164> or config set from-number <e164> first; the number must be outbound-enabled and the Telnyx app must have an Outbound Voice Profile",
            )
        })?;
    let stream_url = guard
        .config
        .public_media_url
        .clone()
        .ok_or_else(|| DriverError::message("config set media-url <wss-url> first"))?;
    Ok((
        connection_id,
        from,
        stream_url,
        guard.config.public_webhook_url.clone(),
        guard.config.telnyx_media,
    ))
}

async fn register_outbound_dial(
    context: &mut GatewayContext,
    dialed: &crate::call_control::DialedCall,
    from: Option<String>,
    to: Option<String>,
    asr_backend: LiveAsrBackend,
    timeline: &'static str,
) -> String {
    let mut guard = context.state.write().await;
    let gateway_call_id = guard.add_or_update_outbound_call(
        crate::operator::state::TelnyxIds {
            call_control_id: dialed.call_control_id.clone(),
            call_session_id: dialed.call_session_id.clone(),
            call_leg_id: dialed.call_leg_id.clone(),
            stream_id: None,
        },
        from,
        to.clone(),
        CallStatus::Dialing,
    );
    if let Some(call) = guard.calls.get_mut(&gateway_call_id) {
        call.asr_backend = Some(asr_backend);
        call.push_timeline(timeline);
        call.push_timeline(format!("asr backend -> {}", asr_backend.model_label()));
    }
    guard.attach_conversation(&gateway_call_id, ConversationMode::Auto);
    guard.log(
        LogLevel::Info,
        format!(
            "outbound dial requested for {gateway_call_id} to {}",
            to.as_deref().unwrap_or("<unknown>")
        ),
    );
    gateway_call_id
}

fn outbound_prereq_context(error: anyhow::Error) -> anyhow::Error {
    let message = format!("{error:#}");
    if message.contains("403") || message.contains("D38") {
        anyhow::anyhow!(
            "{message}; outbound dialing requires the Telnyx Call Control application to have an Outbound Voice Profile and the from-number to be outbound-enabled"
        )
    } else {
        anyhow::anyhow!(message)
    }
}

async fn test_command(
    context: &mut GatewayContext,
    command: TestCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TestCommand::DialTranscribe(args) => dial_transcribe(context, args).await,
    }
}

async fn dial_transcribe(
    context: &mut GatewayContext,
    args: DialTranscribeArgs,
) -> DriverResult<CommandOutput> {
    let asr_backend = context.session.next_asr_backend;
    let (connection_id, from, stream_url, webhook_url, media) =
        outbound_dial_config(context, args.from.clone()).await?;

    let dialed = context
        .telnyx
        .dial_call(&DialRequest {
            connection_id: &connection_id,
            to: &args.to,
            from: &from,
            stream_url: &stream_url,
            webhook_url: webhook_url.as_deref(),
            media,
        })
        .await
        .map_err(|error| driver_anyhow(outbound_prereq_context(error)))?;

    let gateway_call_id = register_outbound_dial(
        context,
        &dialed,
        Some(from.clone()),
        Some(args.to.clone()),
        asr_backend,
        "dial-transcribe requested",
    )
    .await;
    context.session.selected_call = Some(gateway_call_id.clone());

    tracing::info!(
        gateway_call_id,
        call_control_id = dialed.call_control_id,
        call_session_id = dialed.call_session_id.as_deref(),
        call_leg_id = dialed.call_leg_id.as_deref(),
        to = args.to,
        from,
        stream_url,
        stream_codec = media.codec.as_str(),
        stream_sample_rate_hz = media.sample_rate_hz,
        asr_backend = asr_backend.label(),
        asr_model = asr_backend.model_label(),
        "call.outbound.dial_transcribe"
    );
    Ok(CommandOutput::line(format!(
        "dial-transcribe requested for {gateway_call_id}"
    )))
}

async fn calls(state: &SharedState) -> DriverResult<CommandOutput> {
    let guard = state.read().await;
    let lines = if guard.calls.is_empty() {
        vec!["no calls".to_string()]
    } else {
        guard
            .calls
            .values()
            .map(|call| {
                format!(
                    "{} {} from={} to={} stream={} asr={} tts={} conversation={}",
                    call.gateway_call_id,
                    call.status.label(),
                    call.from.as_deref().unwrap_or("<unknown>"),
                    call.to.as_deref().unwrap_or("<unknown>"),
                    call.ids.stream_id.as_deref().unwrap_or("<none>"),
                    call.asr_backend
                        .map(|backend| backend.label())
                        .unwrap_or("<unbound>"),
                    call.tts_status_label(),
                    call.conversation_status_label()
                )
            })
            .collect()
    };
    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn call_command(
    context: &mut GatewayContext,
    command: CallCommand,
) -> DriverResult<CommandOutput> {
    match command {
        CallCommand::Show { call } => call_show(context, call).await,
        CallCommand::Use { call } => {
            let mut guard = context.state.write().await;
            if !context.session.select_call(&mut guard, &call) {
                return Err(DriverError::NotFound {
                    kind: "call",
                    name: call,
                });
            }
            Ok(CommandOutput::line(format!("selected call {call}")))
        }
    }
}

async fn call_show(
    context: &GatewayContext,
    target: Option<String>,
) -> DriverResult<CommandOutput> {
    let guard = context.state.read().await;
    let id = resolve_call_id(
        &guard,
        target.as_deref(),
        context.session.selected_call.as_deref(),
    )?;
    let call = guard.calls.get(&id).ok_or_else(|| DriverError::NotFound {
        kind: "call",
        name: id.clone(),
    })?;
    let mut lines = vec![
        format!("call: {}", call.gateway_call_id),
        format!("state: {}", call.status.label()),
        format!("from: {}", call.from.as_deref().unwrap_or("<unknown>")),
        format!("to: {}", call.to.as_deref().unwrap_or("<unknown>")),
        format!("call_control_id: {}", call.ids.call_control_id),
        format!(
            "call_session_id: {}",
            call.ids.call_session_id.as_deref().unwrap_or("<none>")
        ),
        format!(
            "call_leg_id: {}",
            call.ids.call_leg_id.as_deref().unwrap_or("<none>")
        ),
        format!(
            "stream_id: {}",
            call.ids.stream_id.as_deref().unwrap_or("<none>")
        ),
        format!(
            "media: {} {}Hz {}ch",
            call.media.encoding.as_deref().unwrap_or("<unknown>"),
            call.media
                .sample_rate_hz
                .map(|rate| rate.to_string())
                .unwrap_or_else(|| "?".to_string()),
            call.media
                .channels
                .map(|channels| channels.to_string())
                .unwrap_or_else(|| "?".to_string())
        ),
        format!(
            "asr: {}",
            call.asr_backend
                .map(|backend| format!("{} ({})", backend.label(), backend.model_label()))
                .unwrap_or_else(|| "<unbound>".to_string())
        ),
        format!("tts: {}", call_tts_status(call)),
        format!("conversation: {}", call_conversation_status(call)),
    ];
    if call.echo_suppressed_transcripts > 0 {
        let mut echo_line = format!("echo-suppressed: {}", call.echo_suppressed_transcripts);
        if let Some(preview) = &call.last_echo_suppressed_preview {
            echo_line.push_str(&format!(" last={}", preview));
        }
        lines.push(echo_line);
    }
    if let Some(reason) = &call.terminal_reason {
        lines.push(format!("ended: {reason}"));
    }
    lines.push("assembled transcript:".to_string());
    lines.push(call.assembled_transcript_text());
    lines.push("recent events:".to_string());
    for transcript in call.transcripts.iter().rev().take(12).rev() {
        let prefix = match transcript.kind {
            crate::operator::state::TranscriptKind::Partial => "partial",
            crate::operator::state::TranscriptKind::Final => "final",
        };
        lines.push(format!("{prefix}: {}", transcript.text));
    }
    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn conversation_command(
    context: &mut GatewayContext,
    command: ConversationCommand,
) -> DriverResult<CommandOutput> {
    match command {
        ConversationCommand::Help => Ok(CommandOutput::text(conversation_help())),
        ConversationCommand::Status { call } => {
            let guard = context.state.read().await;
            let id = resolve_call_id(
                &guard,
                call.as_deref(),
                context.session.selected_call.as_deref(),
            )?;
            let call = guard.calls.get(&id).ok_or_else(|| DriverError::NotFound {
                kind: "call",
                name: id.clone(),
            })?;
            Ok(CommandOutput {
                lines: conversation_status_lines(
                    call,
                    context.conversation.handler_label(),
                    quality_barge_in_label(&guard),
                ),
                effects: Vec::new(),
            })
        }
        ConversationCommand::SmokeTest { state } => {
            let enabled = state.enabled();
            context.conversation.set_smoke_test_enabled(enabled);
            let label = if enabled { "on" } else { "off" };
            if enabled {
                set_barge_in_enabled(context, false, "conversation smoke-test").await?;
            }
            context
                .state
                .write()
                .await
                .log(LogLevel::Info, format!("conversation smoke-test {label}"));
            if enabled {
                Ok(CommandOutput {
                    lines: vec![
                        format!("conversation smoke-test: {label}"),
                        "conversation barge-in: off".to_string(),
                    ],
                    effects: Vec::new(),
                })
            } else {
                Ok(CommandOutput::line(format!(
                    "conversation smoke-test: {label}"
                )))
            }
        }
        ConversationCommand::BargeIn { state } => {
            if let Some(enabled) = state.and_then(ConversationBargeInArg::enabled) {
                set_barge_in_enabled(context, enabled, "conversation").await?;
            }
            let guard = context.state.read().await;
            Ok(CommandOutput::line(format!(
                "conversation barge-in: {}",
                quality_barge_in_label(&guard)
            )))
        }
        ConversationCommand::Attach { call } => {
            let mut guard = context.state.write().await;
            let id = resolve_call_id(
                &guard,
                call.as_deref(),
                context.session.selected_call.as_deref(),
            )?;
            if !guard.calls.contains_key(&id) {
                return Err(DriverError::NotFound {
                    kind: "call",
                    name: id,
                });
            }
            guard.attach_conversation(&id, ConversationMode::Auto);
            context.session.selected_call = Some(id.clone());
            Ok(CommandOutput::line(format!(
                "conversation attached for {id} mode=auto"
            )))
        }
        ConversationCommand::Detach { call } => {
            let mut guard = context.state.write().await;
            let id = resolve_call_id(
                &guard,
                call.as_deref(),
                context.session.selected_call.as_deref(),
            )?;
            if !guard.calls.contains_key(&id) {
                return Err(DriverError::NotFound {
                    kind: "call",
                    name: id,
                });
            }
            guard.detach_conversation(&id);
            Ok(CommandOutput::line(format!(
                "conversation detached for {id}"
            )))
        }
        ConversationCommand::Disapprove { call } => disapprove_conversation(context, call).await,
        ConversationCommand::Approve { call } | ConversationCommand::Say { call } => {
            approve_conversation_proposal(context, call).await
        }
        ConversationCommand::Mode { mode, call } => {
            let mode = ConversationMode::from(mode);
            let mut guard = context.state.write().await;
            let id = resolve_call_id(
                &guard,
                call.as_deref(),
                context.session.selected_call.as_deref(),
            )?;
            let Some(call) = guard.calls.get(&id) else {
                return Err(DriverError::NotFound {
                    kind: "call",
                    name: id,
                });
            };
            let attached_before = call.conversation.attached;
            guard.set_conversation_mode(&id, mode);
            context.session.selected_call = Some(id.clone());
            let attached_note = if attached_before { "" } else { " (attached)" };
            Ok(CommandOutput::line(format!(
                "conversation mode for {id}: {}{attached_note}",
                mode.label()
            )))
        }
    }
}

async fn disapprove_conversation(
    context: &mut GatewayContext,
    call: Option<String>,
) -> DriverResult<CommandOutput> {
    let id = {
        let guard = context.state.read().await;
        let id = resolve_call_id(
            &guard,
            call.as_deref(),
            context.session.selected_call.as_deref(),
        )?;
        if !guard.calls.contains_key(&id) {
            return Err(DriverError::NotFound {
                kind: "call",
                name: id,
            });
        }
        id
    };

    let canceled_playback_id = if context.media.active_speech_playback_id(&id).await.is_some() {
        Some(
            speech::cancel_speech(
                &context.state,
                &context.media,
                &id,
                "conversation disapprove",
            )
            .await
            .map_err(driver_anyhow)?,
        )
    } else {
        None
    };

    {
        let mut guard = context.state.write().await;
        guard.detach_conversation(&id);
    }
    context.session.selected_call = Some(id.clone());

    let line = match canceled_playback_id {
        Some(playback_id) => {
            format!(
                "conversation disapproved for {id}; canceled playback={playback_id}; transcription-only"
            )
        }
        None => format!("conversation disapproved for {id}; transcription-only"),
    };
    Ok(CommandOutput::line(line))
}

async fn approve_conversation_proposal(
    context: &mut GatewayContext,
    call: Option<String>,
) -> DriverResult<CommandOutput> {
    let (id, text) = {
        let guard = context.state.read().await;
        let id = resolve_call_id(
            &guard,
            call.as_deref(),
            context.session.selected_call.as_deref(),
        )?;
        let call = guard.calls.get(&id).ok_or_else(|| DriverError::NotFound {
            kind: "call",
            name: id.clone(),
        })?;
        if !call.conversation.attached {
            return Err(DriverError::message(format!(
                "conversation is detached for {id}; run conversation attach {id} first"
            )));
        }
        if call.conversation.status != ConversationStatus::Proposed {
            return Err(DriverError::message(format!(
                "no pending conversation proposal for {id}"
            )));
        }
        let text = call
            .conversation
            .last_assistant_text
            .clone()
            .ok_or_else(|| {
                DriverError::message(format!("no pending conversation proposal for {id}"))
            })?;
        (id, text)
    };

    let queued = speech::queue_speech(
        &context.state,
        &context.media,
        &context.tts,
        context.session.next_tts_backend,
        id.clone(),
        text,
        "conversation approve",
    )
    .await;
    match queued {
        Ok(queued) => {
            context
                .state
                .write()
                .await
                .record_conversation_approved_speaking(&id, queued.playback_id.clone());
            Ok(CommandOutput::line(format!(
                "conversation approved for {id} playback={}",
                queued.playback_id
            )))
        }
        Err(error) => {
            let error = format!("{error:#}");
            context
                .state
                .write()
                .await
                .record_conversation_failed(&id, error.clone());
            Err(DriverError::message(error))
        }
    }
}

fn conversation_status_lines(
    call: &crate::operator::state::CallSession,
    handler_label: &str,
    barge_in_label: &str,
) -> Vec<String> {
    let conversation = &call.conversation;
    let mut lines = vec![
        format!("call: {}", call.gateway_call_id),
        format!("conversation: {}", conversation.status_label()),
        format!("attached: {}", conversation.attached),
        format!("mode: {}", conversation.mode.label()),
        format!("handler: {handler_label}"),
        format!("barge_in: {barge_in_label}"),
        format!("status: {}", conversation.status.label()),
        format!(
            "last_user: {}",
            conversation.last_user_text.as_deref().unwrap_or("<none>")
        ),
        format!(
            "last_assistant: {}",
            conversation
                .last_assistant_text
                .as_deref()
                .unwrap_or("<none>")
        ),
        format!(
            "last_playback: {}",
            conversation.last_playback_id.as_deref().unwrap_or("<none>")
        ),
    ];
    if let Some(error) = &conversation.last_error {
        lines.push(format!("error: {error}"));
    }
    lines
}

fn call_conversation_status(call: &crate::operator::state::CallSession) -> String {
    let conversation = &call.conversation;
    let mut status = format!(
        "{} mode={} attached={}",
        conversation.status_label(),
        conversation.mode.label(),
        conversation.attached
    );
    if let Some(playback) = &conversation.last_playback_id {
        status.push_str(&format!(" playback={playback}"));
    }
    if let Some(error) = &conversation.last_error {
        status.push_str(&format!(" error={error}"));
    }
    status
}

fn call_tts_status(call: &crate::operator::state::CallSession) -> String {
    let Some(tts) = &call.tts else {
        return "idle".to_string();
    };
    let buffer_frames = tts.frames_queued.saturating_sub(tts.frames_sent);
    let mut status = format!(
        "{} backend={} playback={} frames={}/{} buffer={} text={}",
        tts.status.label(),
        tts.backend.label(),
        tts.playback_id,
        tts.frames_sent,
        tts.frames_queued,
        buffer_frames,
        tts.text_preview
    );
    if let Some(first_audio_ms) = tts.first_audio_latency_ms {
        status.push_str(&format!(" first_audio_ms={first_audio_ms}"));
    }
    if tts.pre_audio_wait_ticks > 0 {
        status.push_str(&format!(
            " pre_audio_wait_ms~{}",
            tts.pre_audio_wait_ticks.saturating_mul(20)
        ));
    }
    if tts.underrun_ticks > 0 {
        status.push_str(&format!(
            " underrun_ms~{}",
            tts.underrun_ticks.saturating_mul(20)
        ));
    }
    if let Some(mark) = &tts.mark_name {
        status.push_str(&format!(" mark={mark}"));
    }
    if let Some(error) = &tts.error {
        status.push_str(&format!(" error={error}"));
    }
    status
}

enum CallControlOp {
    Reject,
    Hangup,
}

async fn call_control(
    context: &mut GatewayContext,
    target: Option<String>,
    op: CallControlOp,
) -> DriverResult<CommandOutput> {
    let (gateway_call_id, call_control_id) = {
        let mut guard = context.state.write().await;
        let call = resolve_call_mut(
            &mut guard,
            target.as_deref(),
            context.session.selected_call.as_deref(),
        )?;
        call.push_timeline(match op {
            CallControlOp::Reject => "operator requested reject",
            CallControlOp::Hangup => "operator requested hangup",
        });
        (
            call.gateway_call_id.clone(),
            call.ids.call_control_id.clone(),
        )
    };
    match op {
        CallControlOp::Reject => {
            context
                .telnyx
                .reject_call(&call_control_id)
                .await
                .map_err(driver_anyhow)?;
            Ok(CommandOutput::line(format!(
                "reject requested for {gateway_call_id}"
            )))
        }
        CallControlOp::Hangup => {
            context
                .telnyx
                .hangup_call(&call_control_id)
                .await
                .map_err(driver_anyhow)?;
            Ok(CommandOutput::line(format!(
                "hangup requested for {gateway_call_id}"
            )))
        }
    }
}

async fn transcript_command(
    context: &mut GatewayContext,
    command: TranscriptCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TranscriptCommand::Follow { call } => {
            if let Some(call) = call {
                let mut guard = context.state.write().await;
                if !context.session.select_call(&mut guard, &call) {
                    return Err(DriverError::NotFound {
                        kind: "call",
                        name: call,
                    });
                }
            }
            Ok(CommandOutput::line(
                "selected-call detail follows transcript",
            ))
        }
        TranscriptCommand::Clear { call } => {
            let mut guard = context.state.write().await;
            let session = resolve_call_mut(
                &mut guard,
                call.as_deref(),
                context.session.selected_call.as_deref(),
            )?;
            session.transcripts.clear();
            session.final_transcript.clear();
            session.current_partial = None;
            session.push_timeline("transcript cleared");
            Ok(CommandOutput::line("transcript cleared"))
        }
    }
}

async fn quality_command(
    context: &mut GatewayContext,
    command: QualityCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityCommand::Status => quality_status(context).await,
        QualityCommand::RestoreConfig { encoded } => restore_quality_config(context, encoded).await,
        QualityCommand::Profile { profile } => {
            mutate_quality_config(context, |config| Ok(config.set_profile(profile.into()))).await
        }
        QualityCommand::Endpoint { command } => quality_endpoint_command(context, command).await,
        QualityCommand::Speech { command } => quality_speech_command(context, command).await,
        QualityCommand::Asr { command } => quality_asr_command(context, command).await,
        QualityCommand::TextCall { command } => quality_text_call_command(context, command).await,
        QualityCommand::Tts { command } => quality_tts_command(context, command).await,
        QualityCommand::Logging { command } => quality_logging_command(context, command).await,
        QualityCommand::Judge { command } => quality_judge_command(context, command).await,
        QualityCommand::BargeIn { command } => quality_barge_in_command(context, command).await,
        QualityCommand::EchoSuppression { command } => {
            quality_echo_suppression_command(context, command).await
        }
    }
}

async fn quality_status(context: &GatewayContext) -> DriverResult<CommandOutput> {
    let guard = context.state.read().await;
    let quality = &guard.quality;
    let lines = vec![
        format!("profile={}", quality.config.profile.label()),
        format!("config_id={}", quality.config_id),
        format!("logging_enabled={}", quality.event_sink.is_enabled()),
        format!(
            "logging_path={}",
            quality
                .log_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string())
        ),
        format!("dropped_events={}", quality.event_sink.dropped_count()),
        format!(
            "include_transcript_text={}",
            quality.config.logging.include_transcript_text
        ),
        format!(
            "redaction_mode={}",
            quality.config.logging.redaction_mode.label()
        ),
        format!(
            "endpoint.trailing_silence_ms={}",
            quality.config.endpoint.trailing_silence_ms
        ),
        format!(
            "endpoint.merge_window_ms={}",
            quality.config.endpoint.merge_window_ms
        ),
        format!(
            "endpoint.final_settle_ms={}",
            quality.config.endpoint.final_settle_ms
        ),
        format!("asr.finish_pad_ms={}", quality.config.asr.finish_pad_ms),
        format!(
            "speech.rms_threshold={}",
            quality.config.speech.rms_threshold
        ),
        format!(
            "speech.peak_threshold={}",
            quality.config.speech.peak_threshold
        ),
        format!(
            "text_call.max_active_turns={}",
            quality.config.text_call.max_active_turns
        ),
        format!(
            "tts.generation_mode={}",
            quality.config.tts.generation_mode.label()
        ),
        format!(
            "tts.chunking_enabled={}",
            quality.config.tts.chunking_enabled
        ),
        format!(
            "tts.max_text_chunk_chars={}",
            quality.config.tts.max_text_chunk_chars
        ),
        format!(
            "tts.first_chunk_max_chars={}",
            quality.config.tts.first_chunk_max_chars
        ),
        format!(
            "tts.prebuffer_chunks={}",
            quality.config.tts.prebuffer_chunks
        ),
        format!(
            "barge_in.onset_during_playback={}",
            quality.config.barge_in.onset_during_playback.label()
        ),
        format!(
            "echo_suppression.enabled={}",
            quality.config.echo_suppression.enabled
        ),
        format!(
            "echo_suppression.tail_window_ms={}",
            quality.config.echo_suppression.tail_window_ms
        ),
    ];
    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn quality_endpoint_command(
    context: &mut GatewayContext,
    command: QualityEndpointCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityEndpointCommand::Status => {
            let guard = context.state.read().await;
            let endpoint = &guard.quality.config.endpoint;
            Ok(CommandOutput::text(format!(
                "trailing_silence_ms={}\nmin_turn_words={}\nmin_turn_chars={}\nmerge_window_ms={}\nfinal_settle_ms={}\nconversation_incomplete_tail_hold_ms={}\nconversation_low_confidence_threshold_percent={}\nconversation_playback_hold_poll_ms={}\nmax_turn_words={}\nmax_turn_duration_ms={}",
                endpoint.trailing_silence_ms,
                endpoint.min_turn_words,
                endpoint.min_turn_chars,
                endpoint.merge_window_ms,
                endpoint.final_settle_ms,
                endpoint.conversation_incomplete_tail_hold_ms,
                endpoint.conversation_low_confidence_threshold_percent,
                endpoint.conversation_playback_hold_poll_ms,
                endpoint.max_turn_words,
                endpoint.max_turn_duration_ms
            )))
        }
        QualityEndpointCommand::TrailingSilenceMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_endpoint_trailing_silence_ms(ms))
            })
            .await
        }
        QualityEndpointCommand::MinTurnWords { n } => {
            mutate_quality_config(context, |config| Ok(config.set_endpoint_min_turn_words(n))).await
        }
        QualityEndpointCommand::MinTurnChars { n } => {
            mutate_quality_config(context, |config| Ok(config.set_endpoint_min_turn_chars(n))).await
        }
        QualityEndpointCommand::MergeWindowMs { ms } => {
            mutate_quality_config(
                context,
                |config| Ok(config.set_endpoint_merge_window_ms(ms)),
            )
            .await
        }
        QualityEndpointCommand::FinalSettleMs { ms } => {
            mutate_quality_config(
                context,
                |config| Ok(config.set_endpoint_final_settle_ms(ms)),
            )
            .await
        }
        QualityEndpointCommand::ConversationIncompleteTailHoldMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_endpoint_conversation_incomplete_tail_hold_ms(ms))
            })
            .await
        }
        QualityEndpointCommand::ConversationLowConfidenceThresholdPercent { percent } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_endpoint_conversation_low_confidence_threshold_percent(percent))
            })
            .await
        }
        QualityEndpointCommand::ConversationPlaybackHoldPollMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_endpoint_conversation_playback_hold_poll_ms(ms))
            })
            .await
        }
        QualityEndpointCommand::MaxTurnWords { n } => {
            mutate_quality_config(context, |config| Ok(config.set_endpoint_max_turn_words(n))).await
        }
        QualityEndpointCommand::MaxTurnDurationMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_endpoint_max_turn_duration_ms(ms))
            })
            .await
        }
    }
}

async fn quality_speech_command(
    context: &mut GatewayContext,
    command: QualitySpeechCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualitySpeechCommand::Status => {
            let guard = context.state.read().await;
            let speech = &guard.quality.config.speech;
            Ok(CommandOutput::text(format!(
                "rms_threshold={}\npeak_threshold={}\nonset_min_silence_ms={}",
                speech.rms_threshold, speech.peak_threshold, speech.onset_min_silence_ms
            )))
        }
        QualitySpeechCommand::RmsThreshold { value } => {
            mutate_quality_config(context, |config| {
                config
                    .set_speech_rms_threshold(value)
                    .map_err(|error| DriverError::invalid_argument("value", error.to_string()))
            })
            .await
        }
        QualitySpeechCommand::PeakThreshold { value } => {
            mutate_quality_config(
                context,
                |config| Ok(config.set_speech_peak_threshold(value)),
            )
            .await
        }
        QualitySpeechCommand::OnsetMinSilenceMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_speech_onset_min_silence_ms(ms))
            })
            .await
        }
    }
}

async fn quality_asr_command(
    context: &mut GatewayContext,
    command: QualityAsrCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityAsrCommand::Status => {
            let guard = context.state.read().await;
            let asr = &guard.quality.config.asr;
            Ok(CommandOutput::text(format!(
                "finish_pad_ms={}\nrepeated_token_run_threshold={}\nrepeated_q_run_threshold={}",
                asr.finish_pad_ms, asr.repeated_token_run_threshold, asr.repeated_q_run_threshold
            )))
        }
        QualityAsrCommand::FinishPadMs { ms } => {
            mutate_quality_config(context, |config| Ok(config.set_asr_finish_pad_ms(ms))).await
        }
        QualityAsrCommand::RepeatedTokenRunThreshold { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_asr_repeated_token_run_threshold(n))
            })
            .await
        }
        QualityAsrCommand::RepeatedQRunThreshold { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_asr_repeated_q_run_threshold(n))
            })
            .await
        }
    }
}

async fn quality_text_call_command(
    context: &mut GatewayContext,
    command: QualityTextCallCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityTextCallCommand::Status => {
            let guard = context.state.read().await;
            let text_call = &guard.quality.config.text_call;
            Ok(CommandOutput::text(format!(
                "max_active_turns={}\nmedia_ready_timeout_ms={}\nplayback_wait_timeout_ms={}\nlatest_response_wins={}\ncallback_timeout_ms={}",
                text_call.max_active_turns,
                text_call.media_ready_timeout_ms,
                text_call.playback_wait_timeout_ms,
                text_call.latest_response_wins,
                text_call.callback_timeout_ms
            )))
        }
        QualityTextCallCommand::MaxActiveTurns { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_text_call_max_active_turns(n))
            })
            .await
        }
        QualityTextCallCommand::MediaReadyTimeoutMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_text_call_media_ready_timeout_ms(ms))
            })
            .await
        }
        QualityTextCallCommand::PlaybackWaitTimeoutMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_text_call_playback_wait_timeout_ms(ms))
            })
            .await
        }
        QualityTextCallCommand::LatestResponseWins { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_text_call_latest_response_wins(state.enabled()))
            })
            .await
        }
        QualityTextCallCommand::CallbackTimeoutMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_text_call_callback_timeout_ms(ms))
            })
            .await
        }
    }
}

async fn quality_tts_command(
    context: &mut GatewayContext,
    command: QualityTtsCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityTtsCommand::Status => {
            let guard = context.state.read().await;
            let tts = &guard.quality.config.tts;
            Ok(CommandOutput::text(format!(
                "generation_mode={}
chunking_enabled={}
max_text_chunk_chars={}
first_chunk_max_chars={}
prebuffer_chunks={}",
                tts.generation_mode.label(),
                tts.chunking_enabled,
                tts.max_text_chunk_chars,
                tts.first_chunk_max_chars,
                tts.prebuffer_chunks
            )))
        }
        QualityTtsCommand::GenerationMode { mode } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_tts_generation_mode(mode.into()))
            })
            .await
        }
        QualityTtsCommand::Chunking { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_tts_chunking_enabled(state.enabled()))
            })
            .await
        }
        QualityTtsCommand::MaxTextChunkChars { n } => {
            mutate_quality_config(context, |config| Ok(config.set_tts_max_text_chunk_chars(n)))
                .await
        }
        QualityTtsCommand::FirstChunkMaxChars { n } => {
            mutate_quality_config(
                context,
                |config| Ok(config.set_tts_first_chunk_max_chars(n)),
            )
            .await
        }
        QualityTtsCommand::PrebufferChunks { n } => {
            mutate_quality_config(context, |config| Ok(config.set_tts_prebuffer_chunks(n))).await
        }
    }
}

async fn quality_logging_command(
    context: &mut GatewayContext,
    command: QualityLoggingCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityLoggingCommand::Status => {
            let guard = context.state.read().await;
            let logging = &guard.quality.config.logging;
            Ok(CommandOutput::text(format!(
                "enabled={}\npath={}\nqueue_capacity={}\nper_frame_sample_rate={}\ninclude_transcript_text={}\nredaction_mode={}\ndropped_events={}",
                guard.quality.event_sink.is_enabled(),
                guard
                    .quality
                    .log_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<unset>".to_string()),
                logging.queue_capacity,
                logging.per_frame_sample_rate,
                logging.include_transcript_text,
                logging.redaction_mode.label(),
                guard.quality.event_sink.dropped_count()
            )))
        }
        QualityLoggingCommand::On { path } => {
            let capacity = context
                .state
                .read()
                .await
                .quality
                .config
                .logging
                .queue_capacity;
            let sink =
                QualityEventSink::start_jsonl_writer(&path, capacity).map_err(driver_anyhow)?;
            let mut guard = context.state.write().await;
            let outcome = guard.quality.config.set_logging_enabled(true);
            guard.quality.config_id = guard.quality.config.config_id();
            guard.set_quality_event_sink(sink, Some(path.clone()));
            emit_quality_snapshots_for_active_calls(
                &mut guard,
                "live_change",
                outcome.apply_boundary.label(),
                None,
            );
            guard.log(
                LogLevel::Info,
                format!("quality logging on {}", path.display()),
            );
            Ok(CommandOutput::line(format!(
                "quality logging on path={} config_id={} applies={}",
                path.display(),
                guard.quality.config_id,
                outcome.apply_boundary.label()
            )))
        }
        QualityLoggingCommand::Off => {
            let mut guard = context.state.write().await;
            let outcome = guard.quality.config.set_logging_enabled(false);
            guard.quality.config_id = guard.quality.config.config_id();
            guard.set_quality_event_sink(QualityEventSink::disabled(), None);
            guard.log(LogLevel::Info, "quality logging off");
            Ok(CommandOutput::line(format!(
                "quality logging off config_id={} applies={}",
                guard.quality.config_id,
                outcome.apply_boundary.label()
            )))
        }
        QualityLoggingCommand::IncludeTranscriptText { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_logging_include_transcript_text(state.enabled()))
            })
            .await
        }
        QualityLoggingCommand::RedactionMode { mode } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_logging_redaction_mode(mode.into()))
            })
            .await
        }
    }
}

async fn quality_judge_command(
    context: &mut GatewayContext,
    command: QualityJudgeCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityJudgeCommand::Status => {
            let guard = context.state.read().await;
            let judge = &guard.quality.config.quality_judge;
            Ok(CommandOutput::text(format!(
                "enabled={}\nmode={:?}\nsample_rate={}\nmodel={}\nbatch_size={}\ntimeout_ms={}",
                judge.enabled,
                judge.mode,
                judge.sample_rate,
                judge.model,
                judge.batch_size,
                judge.timeout_ms
            )))
        }
        QualityJudgeCommand::On { sample_rate, model } => {
            mutate_quality_config(context, |config| {
                let mut outcome = config.set_quality_judge_enabled(true);
                if let Some(sample_rate) = sample_rate {
                    outcome =
                        config
                            .set_quality_judge_sample_rate(sample_rate)
                            .map_err(|error| {
                                DriverError::invalid_argument("sample_rate", error.to_string())
                            })?;
                }
                if let Some(model) = model {
                    if model.trim().is_empty() {
                        return Err(DriverError::invalid_argument(
                            "model",
                            "model must be non-empty",
                        ));
                    }
                    config.quality_judge.model = model;
                    outcome = config.set_quality_judge_enabled(true);
                }
                Ok(outcome)
            })
            .await
        }
        QualityJudgeCommand::Off => {
            mutate_quality_config(
                context,
                |config| Ok(config.set_quality_judge_enabled(false)),
            )
            .await
        }
    }
}

async fn quality_barge_in_command(
    context: &mut GatewayContext,
    command: QualityBargeInCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityBargeInCommand::Status => {
            let guard = context.state.read().await;
            let barge_in = &guard.quality.config.barge_in;
            Ok(CommandOutput::text(format!(
                "enabled={}\nspeech_onset_cancel_enabled={}\nonset_during_playback={}\npartial_asr_cancel_enabled={}\nfinal_asr_cancel_enabled={}\nclear_timeout_ms={}",
                barge_in.enabled,
                barge_in.speech_onset_cancel_enabled,
                barge_in.onset_during_playback.label(),
                barge_in.partial_asr_cancel_enabled,
                barge_in.final_asr_cancel_enabled,
                barge_in.clear_timeout_ms
            )))
        }
        QualityBargeInCommand::On => set_barge_in_enabled(context, true, "quality").await,
        QualityBargeInCommand::Off => set_barge_in_enabled(context, false, "quality").await,
        QualityBargeInCommand::SpeechOnset { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_barge_in_speech_onset_cancel_enabled(state.enabled()))
            })
            .await
        }
        QualityBargeInCommand::OnsetDuringPlayback { policy } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_barge_in_onset_during_playback(policy.into()))
            })
            .await
        }
        QualityBargeInCommand::PartialAsr { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_barge_in_partial_asr_cancel_enabled(state.enabled()))
            })
            .await
        }
        QualityBargeInCommand::FinalAsr { state } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_barge_in_final_asr_cancel_enabled(state.enabled()))
            })
            .await
        }
        QualityBargeInCommand::ClearTimeoutMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_barge_in_clear_timeout_ms(ms))
            })
            .await
        }
    }
}

async fn quality_echo_suppression_command(
    context: &mut GatewayContext,
    command: QualityEchoSuppressionCommand,
) -> DriverResult<CommandOutput> {
    match command {
        QualityEchoSuppressionCommand::Status => {
            let guard = context.state.read().await;
            let echo = &guard.quality.config.echo_suppression;
            Ok(CommandOutput::text(format!(
                "enabled={}
min_text_chars={}
tail_window_ms={}
short_token_coverage_percent={}
short_longest_token_run={}
long_min_tokens={}
long_token_coverage_percent={}
long_longest_token_run={}",
                echo.enabled,
                echo.min_text_chars,
                echo.tail_window_ms,
                echo.short_token_coverage_percent,
                echo.short_longest_token_run,
                echo.long_min_tokens,
                echo.long_token_coverage_percent,
                echo.long_longest_token_run
            )))
        }
        QualityEchoSuppressionCommand::On => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_enabled(true))
            })
            .await
        }
        QualityEchoSuppressionCommand::Off => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_enabled(false))
            })
            .await
        }
        QualityEchoSuppressionCommand::MinTextChars { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_min_text_chars(n))
            })
            .await
        }
        QualityEchoSuppressionCommand::TailWindowMs { ms } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_tail_window_ms(ms))
            })
            .await
        }
        QualityEchoSuppressionCommand::ShortTokenCoveragePercent { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_short_token_coverage_percent(n))
            })
            .await
        }
        QualityEchoSuppressionCommand::ShortLongestTokenRun { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_short_longest_token_run(n))
            })
            .await
        }
        QualityEchoSuppressionCommand::LongMinTokens { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_long_min_tokens(n))
            })
            .await
        }
        QualityEchoSuppressionCommand::LongTokenCoveragePercent { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_long_token_coverage_percent(n))
            })
            .await
        }
        QualityEchoSuppressionCommand::LongLongestTokenRun { n } => {
            mutate_quality_config(context, |config| {
                Ok(config.set_echo_suppression_long_longest_token_run(n))
            })
            .await
        }
    }
}

async fn restore_quality_config(
    context: &mut GatewayContext,
    encoded: String,
) -> DriverResult<CommandOutput> {
    let config = VoiceQualityConfig::from_replay_hex(&encoded)
        .map_err(|error| DriverError::invalid_argument("encoded", format!("{error:#}")))?;
    context
        .conversation
        .set_barge_in_enabled(config.barge_in.enabled);
    let mut guard = context.state.write().await;
    guard.set_quality_config(config);
    if !guard.quality.config.logging.enabled {
        guard.set_quality_event_sink(QualityEventSink::disabled(), None);
    }
    emit_quality_snapshots_for_active_calls(&mut guard, "restore_config", "restored", None);
    guard.log(LogLevel::Info, "quality config restored");
    Ok(CommandOutput::line(format!(
        "quality restored config_id={}",
        guard.quality.config_id
    )))
}

async fn set_barge_in_enabled(
    context: &mut GatewayContext,
    enabled: bool,
    source: &'static str,
) -> DriverResult<CommandOutput> {
    let output =
        mutate_quality_config(context, |config| Ok(config.set_barge_in_enabled(enabled))).await?;
    context.conversation.set_barge_in_enabled(enabled);
    let label = if enabled { "on" } else { "off" };
    context
        .state
        .write()
        .await
        .log(LogLevel::Info, format!("{source} barge-in {label}"));
    Ok(output)
}

async fn mutate_quality_config(
    context: &mut GatewayContext,
    mutate: impl FnOnce(
        &mut VoiceQualityConfig,
    ) -> DriverResult<crate::quality::config::QualityMutationOutcome>,
) -> DriverResult<CommandOutput> {
    let mut guard = context.state.write().await;
    let outcome = mutate(&mut guard.quality.config)?;
    guard.quality.config_id = guard.quality.config.config_id();
    context
        .conversation
        .set_barge_in_enabled(guard.quality.config.barge_in.enabled);
    emit_quality_snapshots_for_active_calls(
        &mut guard,
        "live_change",
        outcome.apply_boundary.label(),
        None,
    );
    guard.log(
        LogLevel::Info,
        format!("quality {} {}", outcome.key, outcome.value),
    );
    Ok(CommandOutput::line(render_quality_mutation(&outcome)))
}

fn quality_barge_in_label(state: &GatewayState) -> &'static str {
    if state.quality.config.barge_in.enabled {
        "on"
    } else {
        "off"
    }
}

fn emit_quality_snapshots_for_active_calls(
    state: &mut GatewayState,
    snapshot_reason: &'static str,
    effective_scope: &'static str,
    effective_after_asr_session_id: Option<String>,
) {
    let call_ids = state
        .calls
        .values()
        .filter(|call| {
            !matches!(
                call.status,
                CallStatus::Ended | CallStatus::Failed | CallStatus::IgnoredInbound
            )
        })
        .map(|call| call.gateway_call_id.clone())
        .collect::<Vec<_>>();
    for call_id in call_ids {
        state.emit_quality_config_snapshot(
            &call_id,
            snapshot_reason,
            effective_scope,
            effective_after_asr_session_id.clone(),
        );
    }
}

fn render_quality_mutation(outcome: &crate::quality::config::QualityMutationOutcome) -> String {
    let mut line = format!(
        "quality changed key={} value={} config_id={} applies={}",
        outcome.key,
        outcome.value,
        outcome.config_id,
        outcome.apply_boundary.label()
    );
    if outcome.clamped {
        line.push_str(" clamped=true");
    }
    line
}

async fn log_command(
    context: &mut GatewayContext,
    command: LogCommand,
) -> DriverResult<CommandOutput> {
    match command {
        LogCommand::Clear => {
            context.state.write().await.logs.clear();
            Ok(CommandOutput::line("log cleared"))
        }
    }
}

fn resolve_call_mut<'a>(
    state: &'a mut GatewayState,
    target: Option<&str>,
    selected_call: Option<&str>,
) -> DriverResult<&'a mut crate::operator::state::CallSession> {
    let id = resolve_call_id(state, target, selected_call)?;
    state
        .calls
        .get_mut(&id)
        .ok_or_else(|| DriverError::NotFound {
            kind: "call",
            name: id,
        })
}

fn resolve_call_id(
    state: &GatewayState,
    target: Option<&str>,
    selected_call: Option<&str>,
) -> DriverResult<String> {
    let id = match target {
        Some(value) => value.to_string(),
        None => selected_call
            .filter(|call_id| state.calls.contains_key(*call_id))
            .map(str::to_string)
            .or_else(|| {
                if state.calls.len() == 1 {
                    state.calls.keys().next().cloned()
                } else {
                    None
                }
            })
            .ok_or_else(|| DriverError::message("no call selected; run call use <call>"))?,
    };
    Ok(id)
}

fn resolve_answer_call_id(
    state: &GatewayState,
    target: Option<&str>,
    selected_call: Option<&str>,
) -> DriverResult<String> {
    if let Some(value) = target {
        return Ok(value.to_string());
    }

    let mut waiting = state
        .calls
        .values()
        .filter(|call| call.status == CallStatus::PendingInbound)
        .map(|call| call.gateway_call_id.clone());
    let Some(first_waiting) = waiting.next() else {
        return resolve_call_id(state, None, selected_call);
    };
    if waiting.next().is_some() {
        return Err(DriverError::message(
            "multiple waiting calls; run answer <call>",
        ));
    }
    Ok(first_waiting)
}

fn driver_anyhow(error: anyhow::Error) -> DriverError {
    DriverError::message(format!("{error:#}"))
}

fn gateway_help(topic: &[String]) -> Option<String> {
    match topic {
        [] => Some(gateway_root_help()),
        [topic] if topic == "status" => Some(status_help()),
        [topic] if topic == "listener" => Some(listener_help()),
        [topic] if topic == "config" => Some(config_help()),
        [topic] if topic == "state" => Some(state_help()),
        [topic] if topic == "quit" || topic == "shutdown" => Some(quit_help()),
        [topic] if topic == "load" || topic == "source" => Some(load_help()),
        [topic] if topic == "telnyx" => Some(telnyx_help()),
        [root, topic] if root == "telnyx" && topic == "app" => Some(telnyx_app_help()),
        [root, topic] if root == "telnyx" && topic == "number" => Some(telnyx_number_help()),
        [root, topic, nested] if root == "telnyx" && topic == "app" && nested == "webhook" => {
            Some(telnyx_app_webhook_help())
        }
        [topic] if topic == "inbound" => Some(inbound_help()),
        [topic] if topic == "asr" => Some(asr_help()),
        [topic] if topic == "tts" => Some(tts_help()),
        [topic] if topic == "warm" => Some(warm_help()),
        [topic] if topic == "conversation" || topic == "chat" => Some(conversation_help()),
        [topic] if topic == "calls" || topic == "call" => Some(call_help()),
        [topic] if topic == "dial" || topic == "speak" || topic == "outbound" => {
            Some(outbound_help())
        }
        [topic] if topic == "answer" || topic == "reject" || topic == "hangup" => {
            Some(call_control_help())
        }
        [topic] if topic == "test" => Some(test_help()),
        [root, topic] if root == "test" && topic == "dial-transcribe" => {
            Some(dial_transcribe_help())
        }
        [topic] if topic == "transcript" => Some(transcript_help()),
        [topic] if topic == "log" => Some(log_help()),
        [topic] if topic == "quality" => Some(quality_help()),
        [topic] if topic == "socket" || topic == "agent" => Some(socket_help()),
        _ => None,
    }
}

fn gateway_root_help() -> String {
    [
        "telnyx-gateway operator commands",
        "",
        "The TUI shell and each agent socket connection execute the same command language.",
        "Selections are source-local: a TUI-selected call or ASR backend does not leak into another socket.",
        "Run `help conversation` for M3 call flows and `help socket` for agent protocol details.",
        "",
        "Core:",
        "  status                         Show listener, Telnyx, ASR, media, conversation handler, and calls",
        "  listener status                Show the local HTTP/WebSocket bind address",
        "  config show                    Show replayable gateway config values",
        "  config set <key> <value>       Set webhook-url, media-url, codec, capture dir, etc.",
        "  load <path>                    Replay a .repl command file; comments and blanks are ignored",
        "  source <path>                  Alias for load",
        "  quit [dump_path]               Optionally dump replay commands, then stop the gateway",
        "",
        "Telnyx setup:",
        "  telnyx app list",
        "  telnyx app create <name>",
        "  telnyx app use <connection-id>",
        "  telnyx app show",
        "  telnyx app webhook set <https-url>",
        "  telnyx number list",
        "  telnyx number use <+e164>",
        "  telnyx number bind <+e164> <connection-id>",
        "",
        "Inbound calls and ASR:",
        "  inbound status",
        "  inbound enable --manual        Surface calls as waiting until answer",
        "  inbound enable --auto-transcribe",
        "  inbound disable",
        "  asr list",
        "  asr status",
        "  asr use kroko-2025|sherpa-2023 Select backend for the next answered/dialed call",
        "  tts list",
        "  tts status",
        "  tts use kokoro-82m|piper       Select TTS backend for speak and conversation replies",
        "  warm [all|asr|tts]             Load selected ASR/TTS models before serving calls; TTS warm runs a tiny probe synthesis",
        "",
        "Calls:",
        "  calls                          List calls in operator roster order",
        "  call use <call-id>             Select a call for this TUI/socket source",
        "  call show [call-id]            Show selected call detail, diagnostics, and transcript",
        "  status [call-id]               Show gateway status or selected call status",
        "  answer [call-id]               Answer one waiting inbound call; auto-attach conversation",
        "  dial <+e164> [--from +e164]    Place an outbound call; auto-attach conversation",
        "  speak [call-id] <text...>      Queue debug TTS; disabled while a text-call stream is attached",
        "  speak cancel [call-id]         Clear active TTS; allowed as an emergency control",
        "  conversation status [call-id]  Show attachment, mode, handler, and latest turns",
        "  conversation smoke-test on|off Enable or disable test-only echo replies",
        "  conversation barge-in on|off|status Enable or disable transcript-triggered TTS clear",
        "  conversation disapprove [call-id] Stop TTS and detach conversation",
        "  reject [call-id]",
        "  hangup [call-id]",
        "  transcript follow [call-id]",
        "  transcript clear [call-id]",
        "  log clear",
        "  quality status                  Show M6 quality config/logging status",
        "  quality endpoint trailing-silence-ms <ms>",
        "",
        "Testing:",
        "  conversation smoke-test on    Enable test-only echo reply handler for M3 smoke tests",
        "  test dial-transcribe <+e164> [--from +e164]",
        "",
        "Helpful topics:",
        "  help config       help telnyx       help inbound      help asr",
        "  help tts          help call         help outbound     help socket",
        "  help transcript   help test",
    ]
    .join("\n")
}

fn quality_help() -> String {
    [
        "quality status",
        "quality profile fast|balanced|complete|noisy  default=balanced applies=next_asr_session",
        "quality endpoint status",
        "quality endpoint trailing-silence-ms <ms>      range=100..5000 default=900ms applies=next_asr_session",
        "quality endpoint min-turn-words <n>            range=0..50 default=2 report_only",
        "quality endpoint min-turn-chars <n>            range=0..200 default=6 report_only",
        "quality endpoint merge-window-ms <ms>          range=0..5000 default=350ms applies=new_turn",
        "quality endpoint final-settle-ms <ms>          range=0..5000 default=800ms applies=next_asr_session",
        "quality endpoint conversation-incomplete-tail-hold-ms <ms> range=0..10000 default=2500ms applies=new_turn",
        "quality endpoint conversation-low-confidence-threshold-percent <n> range=0..100 default=45 applies=new_turn",
        "quality endpoint conversation-playback-hold-poll-ms <ms> range=10..1000 default=100ms applies=new_turn",
        "quality endpoint max-turn-words <n>            range=1..500 default=80 report_only",
        "quality endpoint max-turn-duration-ms <ms>     range=1000..120000 default=12000ms report_only",
        "quality speech status",
        "quality speech rms-threshold <value>           range=0.0..20000.0 default=220.0 applies=next_asr_session",
        "quality speech peak-threshold <value>          range=0..32767 default=1100 applies=next_asr_session",
        "quality speech onset-min-silence-ms <ms>       range=0..2000 default=180ms applies=next_asr_session",
        "quality asr status",
        "quality asr finish-pad-ms <ms>                 range=0..2000 default=320ms applies=next_asr_session",
        "quality asr repeated-token-run-threshold <n>   range=2..128 default=16 applies=next_asr_session",
        "quality asr repeated-q-run-threshold <n>       range=2..64 default=8 applies=next_asr_session",
        "quality text-call status",
        "quality text-call max-active-turns <n>         range=1..1024 default=32 applies=new_text_call_session",
        "quality text-call media-ready-timeout-ms <ms>  range=1000..120000 default=20000ms applies=new_playback_request",
        "quality text-call playback-wait-timeout-ms <ms> range=1000..600000 default=180000ms applies=new_playback_request",
        "quality text-call latest-response-wins on|off  bool default=true applies=new_turn",
        "quality text-call callback-timeout-ms <ms>     range=100..60000 default=5000ms applies=new_turn",
        "quality tts status",
        "quality tts chunking on|off                    bool default=true applies=new_playback_request",
        "quality tts max-text-chunk-chars <n>           range=40..500 default=90 applies=new_playback_request",
        "quality tts first-chunk-max-chars <n>          range=0|40..500 default=40 applies=new_playback_request",
        "quality tts prebuffer-chunks <n>               range=1..64 default=1 applies=new_playback_request",
        "quality logging on <path>",
        "quality logging off",
        "quality logging include-transcript-text on|off bool default=false applies=immediate sensitive_opt_in",
        "quality logging redaction-mode metrics-only|hashed-text|redacted-text|sensitive-plaintext default=metrics-only applies=immediate",
        "quality judge status|on|off",
        "quality barge-in status|on|off                 bool default=true applies=next_asr_session",
        "quality barge-in speech-onset on|off           bool default=true applies=next_asr_session",
        "quality barge-in onset-during-playback defer-to-partial|trust default=defer_to_partial applies=next_asr_session",
        "quality barge-in partial-asr on|off            bool default=true applies=next_asr_session",
        "quality barge-in final-asr on|off              bool default=true applies=next_asr_session",
        "quality barge-in clear-timeout-ms <ms>         range=100..10000 default=1000ms applies=new_turn",
        "quality echo-suppression status|on|off         bool default=true applies=next_asr_session",
        "quality echo-suppression min-text-chars <n>    range=1..500 default=10 applies=next_asr_session",
        "quality echo-suppression tail-window-ms <ms>   range=0..10000 default=2000 applies=next_asr_session",
        "quality echo-suppression short-token-coverage-percent <n> range=0..100 default=66 applies=next_asr_session",
        "quality echo-suppression short-longest-token-run <n> range=1..64 default=2 applies=next_asr_session",
        "quality echo-suppression long-min-tokens <n>   range=2..64 default=4 applies=next_asr_session",
        "quality echo-suppression long-token-coverage-percent <n> range=0..100 default=60 applies=next_asr_session",
        "quality echo-suppression long-longest-token-run <n> range=1..64 default=3 applies=next_asr_session",
        "",
        "M6 quality commands use the existing line-oriented operator dispatcher.",
        "Transcript text remains disabled unless explicitly enabled.",
        "Operator TUI Up/Down recalls submitted commands through motlie-driver HistoryBuffer.",
    ]
    .join("\n")
}

fn status_help() -> String {
    [
        "status",
        "",
        "Show the gateway's current operational state.",
        "",
        "Includes:",
        "  listener bind address",
        "  inbound mode",
        "  source-local next ASR backend and code default ASR backend",
        "  public webhook/media URLs",
        "  selected Telnyx app and phone number",
        "  media codec/sample rate",
        "  capture directory",
        "  TTS backend",
        "  conversation handler mode",
        "  call count",
        "",
        "Example:",
        "  status",
    ]
    .join("\n")
}

fn listener_help() -> String {
    [
        "listener status",
        "",
        "Show the local HTTP/WebSocket bind address the gateway is serving.",
        "",
        "Example:",
        "  listener status",
    ]
    .join("\n")
}

fn config_help() -> String {
    [
        "config show",
        "config set <key> <value>",
        "",
        "Inspect or update gateway configuration used by Telnyx API calls and media streaming.",
        "",
        "Keys:",
        "  webhook-url          Public HTTPS URL for Telnyx call-control webhooks",
        "  media-url            Public WSS URL for Telnyx media streaming",
        "  media-codec          PCMU, PCMA, or L16",
        "  media-sample-rate    Sample rate requested from Telnyx for the selected codec",
        "  capture-dir          Directory for replay/capture artifacts",
        "  from-number          Default outbound caller ID for dial/test dial-transcribe",
        "  state-path           Default path used by quit/shutdown when no dump path is supplied",
        "",
        "Examples:",
        "  config set webhook-url https://host.example/telnyx/webhooks",
        "  config set media-url wss://host.example/telnyx/media",
        "  config set media-codec PCMU",
        "  config set capture-dir ~/telnyx-test/captures",
        "  config show",
    ]
    .join("\n")
}

fn warm_help() -> String {
    [
        "warm",
        "warm all",
        "warm asr",
        "warm tts",
        "",
        "Load the source-local next ASR and/or TTS model handles before serving a call.",
        "Warm status is runtime-only and appears in the TUI Runtime pane; TTS warm includes a tiny discarded probe synthesis.",
    ]
    .join("\n")
}

fn tts_help() -> String {
    [
        "tts list",
        "tts status",
        "tts use kokoro-82m",
        "tts use piper",
        "",
        "Inspect or select the outbound TTS backend used by `speak` and conversation replies.",
        "Kokoro-82M is the default live backend; Piper remains selectable",
        "and is used automatically as the fallback when Kokoro fails.",
        "",
        "If `tts status` reports unavailable, restart the gateway from a binary",
        "built with `--features \"sherpa piper kokoro\"`.",
        "",
        "After `dial`, wait until the call state is `media` or `transcribing`, then run:",
        "  speak <text...>",
        "or explicitly:",
        "  speak <call-id> <text...>",
        "",
        "Examples:",
        "  tts list",
        "  tts status",
        "  tts use kokoro-82m",
        "  tts use piper",
        "  warm tts",
    ]
    .join("\n")
}

fn state_help() -> String {
    [
        "state dump <path>",
        "",
        "Write replayable commands for the current gateway configuration.",
        "Use `load <path>` or start with `--load <path>` to rehydrate a later session.",
        "",
        "Example:",
        "  state dump ~/telnyx-test/config.repl",
    ]
    .join("\n")
}

fn quit_help() -> String {
    [
        "quit [dump_path]",
        "shutdown [dump_path]",
        "",
        "Stop the gateway. If a path is supplied, or state-path is configured, the gateway writes",
        "replayable commands before exiting.",
        "",
        "Operator-facing spelling is `quit`; `shutdown` remains available as the low-level command.",
        "",
        "Examples:",
        "  quit",
        "  quit ~/telnyx-test/config.repl",
    ]
    .join("\n")
}

fn load_help() -> String {
    [
        "load <path>",
        "source <path>",
        "",
        "Replay commands from a .repl file in the current TUI or socket source.",
        "Blank lines and lines beginning with # are ignored. `~` and `~/...` are expanded.",
        "",
        "Examples:",
        "  load ~/telnyx-test/config.repl",
        "  source /tmp/telnyx.repl",
    ]
    .join("\n")
}

fn telnyx_help() -> String {
    [
        "telnyx app ...",
        "telnyx number ...",
        "",
        "Configure Telnyx Call Control resources from the gateway.",
        "",
        "Application commands:",
        "  telnyx app list",
        "  telnyx app create <name>",
        "  telnyx app use <connection-id>",
        "  telnyx app show",
        "  telnyx app webhook set <https-url>",
        "",
        "Number commands:",
        "  telnyx number list",
        "  telnyx number use <+e164>",
        "  telnyx number bind <+e164> <connection-id>",
        "",
        "Common setup flow:",
        "  config set webhook-url https://your-host/telnyx/webhooks",
        "  config set media-url wss://your-host/telnyx/media",
        "  telnyx app create motlie-test",
        "  telnyx number bind <phone-number> <connection-id>",
    ]
    .join("\n")
}

fn telnyx_app_help() -> String {
    [
        "telnyx app list",
        "telnyx app create <name>",
        "telnyx app use <connection-id>",
        "telnyx app show",
        "telnyx app webhook set <https-url>",
        "",
        "Manage the selected Telnyx Call Control application.",
        "",
        "Notes:",
        "  create requires config webhook-url first",
        "  use selects an existing Telnyx connection/application id",
        "  webhook set updates the selected app and stores the URL in gateway config",
    ]
    .join("\n")
}

fn telnyx_app_webhook_help() -> String {
    [
        "telnyx app webhook set <https-url>",
        "",
        "Update the selected Telnyx application's webhook URL.",
        "",
        "Example:",
        "  telnyx app webhook set https://host.example/telnyx/webhooks",
    ]
    .join("\n")
}

fn telnyx_number_help() -> String {
    [
        "telnyx number list",
        "telnyx number use <+e164>",
        "telnyx number bind <+e164> <connection-id>",
        "",
        "Select and bind Telnyx phone numbers.",
        "",
        "Examples:",
        "  telnyx number list",
        "  telnyx number use <phone-number>",
        "  telnyx number bind <phone-number> <connection-id>",
    ]
    .join("\n")
}

fn inbound_help() -> String {
    [
        "inbound status",
        "inbound enable --manual",
        "inbound enable --auto-transcribe",
        "inbound disable",
        "",
        "Control whether inbound calls are handled.",
        "",
        "Modes:",
        "  disabled         Default at startup; inbound webhooks are ignored",
        "  manual           Calls enter the roster as waiting; operator/agent must answer",
        "  auto-transcribe  Gateway answers/transcribes automatically when configured",
        "",
        "Examples:",
        "  inbound enable --manual",
        "  calls",
        "  answer <call-id>",
        "  inbound disable",
    ]
    .join("\n")
}

fn asr_help() -> String {
    [
        "asr list",
        "asr status",
        "asr use kroko-2025",
        "asr use sherpa-2023",
        "",
        "Select the live ASR backend for the next call answered or dialed by this source.",
        "",
        "Important:",
        "  Switching is between-call only. It does not change an active call.",
        "  The selection is source-local. A TUI choice does not affect a socket connection,",
        "  and one socket's choice does not affect another socket.",
        "",
        "Examples:",
        "  asr list",
        "  asr status",
        "  asr use kroko-2025",
    ]
    .join("\n")
}

fn conversation_help() -> String {
    [
        "conversation commands",
        "",
        "Attach or detach the selected call from the gateway-local conversation handler.",
        "The TUI and command socket share this command path.",
        "",
        "Default live behavior:",
        "  answer auto-attaches inbound calls in auto mode.",
        "  dial auto-attaches outbound calls in auto mode.",
        "  The built-in handler is disabled by default; attached calls transcribe",
        "  and record user turns without speaking.",
        "  Use `conversation smoke-test on` or startup `--conversation-smoke-test`",
        "  only when intentionally testing the echo/repeat TTS loop.",
        "",
        "Usage:",
        "  conversation help",
        "  conversation status [call-id]",
        "  conversation smoke-test <on|off>",
        "  conversation barge-in [on|off|status]",
        "  conversation attach [call-id]",
        "  conversation detach [call-id]",
        "  conversation disapprove [call-id]",
        "  conversation approve [call-id]",
        "  conversation say [call-id]",
        "  conversation mode <manual|auto> [call-id]",
        "",
        "Normal inbound TUI/socket flow:",
        "  inbound enable --manual",
        "  answer",
        "  conversation status",
        "",
        "Normal outbound TUI/socket flow:",
        "  dial <callee-e164>",
        "  conversation status",
        "",
        "Smoke-test two-way loop:",
        "  conversation smoke-test on   # also turns barge-in off for deterministic echo",
        "  conversation barge-in on     # optional: explicitly test interruption behavior",
        "  answer    # or: dial <callee-e164>",
        "  conversation status",
        "  speak cancel",
        "  conversation smoke-test off",
        "",
        "Controls:",
        "  smoke-test enablement turns barge-in off; turn it back on only to test interruption.",
        "  barge-in off keeps active TTS from being cleared by partial/final transcripts.",
        "  disapprove cancels active conversation TTS and leaves transcription-only mode.",
        "  mode manual records assistant proposals; approve/say speaks the pending proposal",
        "  using this source's selected TTS backend.",
        "",
        "Socket agents send the same commands over the newline protocol; `status`,",
        "`calls`, `call show`, and TTS discovery commands also return structured JSON data.",
    ]
    .join("\n")
}

fn call_help() -> String {
    [
        "calls",
        "call use <call-id>",
        "call show [call-id]",
        "",
        "Inspect and select calls.",
        "",
        "TUI parity:",
        "  The Calls pane cursor is a visual shortcut for choosing a call.",
        "  Socket agents use `calls`, `call use <call-id>`, and `call show` for the same state.",
        "",
        "Examples:",
        "  calls",
        "  call use gwc_...",
        "  call show",
        "  call show gwc_...",
    ]
    .join("\n")
}

fn call_control_help() -> String {
    [
        "answer [call-id]",
        "reject [call-id]",
        "hangup [call-id]",
        "",
        "Control a call. If call-id is omitted, the command uses this source's selected call",
        "or the single waiting inbound call when that is unambiguous.",
        "Answering attaches conversation in auto mode; the handler stays disabled unless",
        "`conversation smoke-test on` or startup `--conversation-smoke-test` is used.",
        "",
        "Examples:",
        "  answer",
        "  answer gwc_...",
        "  hangup",
        "  reject gwc_...",
    ]
    .join("\n")
}

fn outbound_help() -> String {
    [
        "dial <+e164-or-sip-uri> [--from +e164]",
        "speak [call-id] <text...>",
        "speak cancel [call-id]",
        "status [call-id]",
        "",
        "Place an outbound call over the existing bidirectional Telnyx media WebSocket.",
        "`dial` selects the new call and auto-attaches conversation in auto mode. The",
        "built-in conversation handler remains disabled by default, so normal live calls",
        "transcribe without echo replies. `speak` is manual, non-blocking, and cancellable;",
        "`speak cancel` sends Telnyx clear and drops local queued outbound audio.",
        "",
        "Conversation smoke test:",
        "  conversation smoke-test on enables the test-only `I heard: ...` reply loop.",
        "  conversation smoke-test off returns attached calls to transcription-only behavior.",
        "",
        "Prerequisites:",
        "  telnyx app use <connection-id>",
        "  config set media-url <wss-url>",
        "  config set from-number <outbound-enabled +e164>",
        "  Telnyx account/application has an Outbound Voice Profile assigned",
        "",
        "Examples:",
        "  dial <callee-e164>",
        "  status",
        "  conversation status",
        "  conversation smoke-test on",
        "  speak Hello from Motlie.",
        "  speak gwc_... Hello from Motlie.",
        "  speak cancel",
    ]
    .join("\n")
}

fn test_help() -> String {
    [
        "test dial-transcribe <+e164> [--from +e164]",
        "",
        "Place an outbound test call, attach media streaming, and transcribe what the callee says.",
        "This is for ASR quality testing; outbound TTS is not part of this command.",
        "",
        "Example:",
        "  test dial-transcribe <to-phone-number> --from <from-phone-number>",
    ]
    .join("\n")
}

fn dial_transcribe_help() -> String {
    test_help()
}

fn transcript_help() -> String {
    [
        "transcript follow [call-id]",
        "transcript clear [call-id]",
        "",
        "Inspect or clear retained transcript lines for a call.",
        "",
        "Notes:",
        "  call show displays the assembled transcript.",
        "  transcript follow returns retained transcript events for polling-style agents.",
        "  transcript clear clears transcript state for the selected or specified call.",
        "",
        "Examples:",
        "  transcript follow",
        "  transcript follow gwc_...",
        "  transcript clear gwc_...",
    ]
    .join("\n")
}

fn log_help() -> String {
    [
        "log clear",
        "",
        "Clear the gateway's retained in-memory operator log buffer.",
        "",
        "Example:",
        "  log clear",
    ]
    .join("\n")
}

fn socket_help() -> String {
    [
        "Agent socket interface",
        "",
        "Start with:",
        "  telnyx-gateway --socket /tmp/telnyx-gateway.sock",
        "  telnyx-gateway --tui --socket /tmp/telnyx-gateway.sock",
        "  telnyx-gateway --conversation-smoke-test --socket /tmp/telnyx-gateway.sock",
        "",
        "Protocol:",
        "  Send one command line terminated by newline.",
        "  Receive one JSON object per command:",
        "    {\"ok\":true,\"lines\":[...],\"data\":{...},\"effects\":[...],\"error\":null}",
        "  `data` is present for status, calls, call show, tts list, and tts status polling.",
        "",
        "Debug text stream mode:",
        "  stream attach <call-id>",
        "  stream attach --partials <call-id>",
        "  {\"type\":\"debug.attach\",\"protocol\":\"motlie.telnyx.text.v1\",\"extension\":\"motlie.telnyx.text.debug.v1\",\"call_id\":\"<call-id>\"}",
        "  --partials enables advisory caller.partial frames with optional confidence/stability.",
        "  Stream mode sends motlie.telnyx.text.v1 JSONL frames and accepts agent frames.",
        "  While stream mode owns a call, command-mode speak <text> is rejected to avoid competing audio.",
        "  Send {\"type\":\"debug.detach\",\"reason\":\"done\"} to return to command mode.",
        "",
        "Discovery:",
        "  help",
        "  help socket",
        "  help inbound",
        "  help asr",
        "  help tts",
        "  help warm",
        "  help conversation",
        "  help transcript",
        "  help call",
        "  help outbound",
        "",
        "Agent workflows:",
        "  inbound: calls -> answer [call-id] -> conversation status [call-id]",
        "  outbound: dial <callee-e164> -> conversation status",
        "  smoke test: conversation smoke-test on -> answer or dial (barge-in defaults off)",
        "  stop assistant audio: conversation disapprove [call-id]",
        "  inspect: status, calls, call show [call-id], transcript follow [call-id]",
        "  call show includes TTS buffer/latency/underrun and echo-suppression counters",
        "",
        "Operational parity:",
        "  TUI and socket both use the same typed command language.",
        "  TUI-only keystrokes have command equivalents:",
        "    Tab focus change       no socket equivalent needed",
        "    Calls pane cursor     calls + call use <call-id>",
        "    Calls pane `a` attach call use <call-id>",
        "    Detail pane scroll    transcript follow / call show polling",
        "  Outbound and M3 conversation commands are also shared:",
        "    dial <+e164> [--from +e164]",
        "    speak [call-id] <text...>     blocked while stream attach/app-agent owns the call",
        "    speak cancel [call-id]        emergency clear remains available",
        "    conversation status [call-id]",
        "    conversation attach|detach [call-id]",
        "    conversation smoke-test on|off",
        "    conversation barge-in [on|off|status]",
        "    conversation approve [call-id]",
        "    conversation disapprove [call-id]",
        "    hangup [call-id]",
        "",
        "Source-local state:",
        "  Each socket connection has its own selected call and next ASR backend.",
        "  `tts use` changes the source's next manual speak backend and the gateway-wide",
        "  conversation reply backend, because media-triggered turns are not associated",
        "  with one TUI/socket command source.",
        "  The smoke-test handler mode and barge-in mode are also gateway-wide.",
    ]
    .join("\n")
}

fn expand_user_path(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use motlie_driver::CommandEngine;
    use tokio::sync::mpsc;
    use tokio::time::{timeout, Duration};

    use super::*;
    use crate::adapter::LiveAsrBackend;
    use crate::call_control::TelnyxClient;
    use crate::conversation::handle_transcript_event;
    use crate::operator::state::{
        shared_state, CallStatus, GatewayState, MediaMetadata, TelnyxIds, TtsPlaybackStatus,
    };
    use crate::tts::{StaticTtsFactory, TtsRegistry, KOKORO_SAMPLE_RATE_HZ};
    use motlie_model::TranscriptionUpdate;
    use motlie_voice::app::TranscriptEvent;

    #[tokio::test]
    async fn inbound_is_disabled_by_default() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("inbound status").await.expect("status");

        assert_eq!(output.lines, vec!["inbound disabled"]);
    }

    #[tokio::test]
    async fn status_formats_listener_without_option_debug() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("status").await.expect("status");

        assert!(output
            .lines
            .iter()
            .any(|line| line.starts_with("listener: 127.0.0.1:")));
        assert!(!output.lines.iter().any(|line| line.contains("Some(")));
    }

    #[tokio::test]
    async fn manual_enable_changes_mode() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line("inbound enable --manual")
            .await
            .expect("enable");

        assert_eq!(state.read().await.inbound_mode, InboundMode::Manual);
    }

    #[tokio::test]
    async fn asr_use_sets_tui_source_next_backend() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine
            .run_line("asr use sherpa-2023")
            .await
            .expect("asr use");

        assert_eq!(
            output.lines,
            vec!["asr backend for next calls: sherpa-2023 (sherpa-zipformer-en-2023-06-26)"]
        );
        assert_eq!(
            engine.context().session.next_asr_backend,
            LiveAsrBackend::Sherpa2023
        );
        assert_eq!(LiveAsrBackend::default(), LiveAsrBackend::Kroko2025);
    }

    #[tokio::test]
    async fn asr_list_reports_available_live_backends() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("asr list").await.expect("asr list");

        assert_eq!(
            output.lines,
            vec![
                "kroko-2025 sherpa-zipformer-en-kroko-2025-08-06",
                "sherpa-2023 sherpa-zipformer-en-2023-06-26"
            ]
        );
    }

    #[tokio::test]
    async fn gateway_help_documents_shared_tui_and_socket_commands() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("help").await.expect("help output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("TUI shell and each agent socket connection"));
        assert!(rendered.contains("load <path>"));
        assert!(rendered.contains("quit [dump_path]"));
        assert!(rendered.contains("tts list"));
        assert!(rendered.contains("tts use kokoro-82m|piper"));
        assert!(rendered.contains("conversation status [call-id]"));
        assert!(rendered.contains("test-only echo reply handler"));
        assert!(rendered.contains("help socket"));
    }

    #[tokio::test]
    async fn gateway_help_topics_cover_agent_relevant_commands() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let asr = engine.run_line("help asr").await.expect("asr help");
        let tts = engine.run_line("help tts").await.expect("tts help");
        let call = engine.run_line("help call").await.expect("call help");
        let outbound = engine
            .run_line("help outbound")
            .await
            .expect("outbound help");
        let socket = engine.run_line("help socket").await.expect("socket help");
        let conversation = engine
            .run_line("help conversation")
            .await
            .expect("conversation help");
        let conversation_alias = engine
            .run_line("conversation help")
            .await
            .expect("conversation help alias");

        assert!(asr.lines.join("\n").contains("source-local"));
        assert!(tts.lines.join("\n").contains("tts list"));
        assert!(tts.lines.join("\n").contains("speak <text...>"));
        assert!(call.lines.join("\n").contains("call use <call-id>"));
        let outbound = outbound.lines.join("\n");
        let socket = socket.lines.join("\n");
        let conversation = conversation.lines.join("\n");
        let conversation_alias = conversation_alias.lines.join("\n");
        assert!(outbound.contains("auto-attaches conversation in auto mode"));
        assert!(outbound.contains("conversation smoke-test on"));
        assert!(socket.contains("Receive one JSON object"));
        assert!(socket.contains("Agent workflows"));
        assert!(socket.contains("smoke-test handler mode and barge-in mode are also gateway-wide"));
        assert!(conversation.contains("Default live behavior"));
        assert!(conversation_alias.contains("Default live behavior"));
        assert!(conversation.contains("Normal inbound TUI/socket flow"));
        assert!(conversation.contains("Normal outbound TUI/socket flow"));
        assert!(conversation.contains("conversation disapprove [call-id]"));
    }

    #[tokio::test]
    async fn tts_list_reports_backend_availability() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state, telnyx, SharedMediaRegistry::default());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("tts list").await.expect("tts list");

        assert_eq!(
            output.lines,
            vec![
                "kokoro-82m kokoro/kokoro_82m available",
                "piper piper/en_us_ljspeech_medium available",
            ]
        );
    }

    #[tokio::test]
    async fn tts_status_reports_unavailable_backend_clearly() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("tts status").await.expect("tts status");

        assert!(output.lines.iter().any(|line| line == "next=kokoro-82m"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "next_model=kokoro/kokoro_82m"));
        assert!(output.lines.iter().any(|line| line == "status=unavailable"));
        assert!(output.lines.iter().any(|line| {
            line == "reason=Kokoro-82M TTS is unavailable; rebuild with --features kokoro"
        }));
    }

    #[tokio::test]
    async fn tts_use_matches_asr_command_shape() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state, telnyx, SharedMediaRegistry::default());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        assert_eq!(
            engine.context().session.next_tts_backend,
            LiveTtsBackend::Kokoro82m
        );

        let output = engine
            .run_line("tts use piper")
            .await
            .expect("tts use piper");

        assert_eq!(
            output.lines,
            vec![
                "tts backend for next speech and conversation replies: piper (piper/en_us_ljspeech_medium)"
            ]
        );
        assert_eq!(
            engine.context().session.next_tts_backend,
            LiveTtsBackend::Piper
        );
        assert_eq!(
            engine.context().state.read().await.conversation_tts_backend,
            LiveTtsBackend::Piper
        );

        let output = engine
            .run_line("tts use kokoro-82m")
            .await
            .expect("tts use kokoro");

        assert_eq!(
            output.lines,
            vec![
                "tts backend for next speech and conversation replies: kokoro-82m (kokoro/kokoro_82m)"
            ]
        );
        assert_eq!(
            engine.context().session.next_tts_backend,
            LiveTtsBackend::Kokoro82m
        );
        assert_eq!(
            engine.context().state.read().await.conversation_tts_backend,
            LiveTtsBackend::Kokoro82m
        );
    }

    #[tokio::test]
    async fn conversation_smoke_reply_uses_selected_tts_backend() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let media = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(16);
        let call_id = {
            let mut guard = state.write().await;
            add_streaming_call(&mut guard, "call-1", "stream-1")
        };
        media.register_call(call_id.clone(), tx).await;
        let context = context_with_static_tts(state.clone(), telnyx, media.clone());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line("tts use piper")
            .await
            .expect("select piper for conversation replies");
        engine
            .run_line("quality endpoint merge-window-ms 0")
            .await
            .expect("disable coalescing for deterministic smoke test");
        engine
            .run_line("conversation smoke-test on")
            .await
            .expect("enable smoke handler");
        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        engine
            .run_line("conversation attach")
            .await
            .expect("attach conversation");

        handle_transcript_event(
            &state,
            &media,
            &engine.context().conversation,
            &call_id,
            TranscriptEvent::Final {
                text: "backend check".to_string(),
                update: TranscriptionUpdate::default(),
            },
            None,
        )
        .await
        .expect("final transcript should queue smoke reply");

        let playback_id = receive_frame_playback(&mut rx).await;
        let guard = state.read().await;
        let call = guard.calls.get(&call_id).expect("call exists");
        let tts = call
            .tts
            .as_ref()
            .expect("conversation TTS should be queued");
        assert_eq!(tts.playback_id, playback_id);
        assert_eq!(tts.backend, LiveTtsBackend::Piper);
    }

    #[tokio::test]
    async fn warm_command_records_selected_model_status() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context =
            context_with_static_tts(state.clone(), telnyx, SharedMediaRegistry::default());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("warm").await.expect("warm models");

        assert_eq!(output.lines.len(), 2);
        assert!(output.lines[0]
            .starts_with("warm asr=kroko-2025 model=sherpa-zipformer-en-kroko-2025-08-06 status=ready elapsed_ms="));
        assert!(output.lines[1].starts_with(
            "warm tts=kokoro-82m model=kokoro/kokoro_82m mode=buffered status=ready elapsed_ms="
        ));
        let guard = state.read().await;
        assert!(guard
            .model_warmups
            .contains_key(&asr_warm_key(LiveAsrBackend::Kroko2025)));
        assert!(guard
            .model_warmups
            .contains_key(&tts_warm_key(LiveTtsBackend::Kokoro82m)));
    }

    #[tokio::test]
    async fn warm_tts_only_skips_asr_status() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context =
            context_with_static_tts(state.clone(), telnyx, SharedMediaRegistry::default());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("warm tts").await.expect("warm tts");

        assert_eq!(output.lines.len(), 1);
        assert!(output.lines[0].starts_with("warm tts=kokoro-82m"));
        let guard = state.read().await;
        assert!(!guard
            .model_warmups
            .contains_key(&asr_warm_key(LiveAsrBackend::Kroko2025)));
        assert!(guard
            .model_warmups
            .contains_key(&tts_warm_key(LiveTtsBackend::Kokoro82m)));
    }

    #[tokio::test]
    async fn source_local_asr_choice_binds_on_answer() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let (call_one, call_two) = {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let call_one = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::PendingInbound,
            );
            let call_two = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-2".to_string(),
                    call_session_id: Some("sess-2".to_string()),
                    call_leg_id: Some("leg-2".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::PendingInbound,
            );
            (call_one, call_two)
        };
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let base = GatewayContext::new(state.clone(), telnyx);
        let mut tui_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());
        let mut socket_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());

        tui_engine
            .run_line("asr use sherpa-2023")
            .await
            .expect("tui asr use");
        tui_engine
            .run_line(&format!("answer {call_one}"))
            .await
            .expect("tui answer");
        socket_engine
            .run_line(&format!("answer {call_two}"))
            .await
            .expect("socket answer");

        let guard = state.read().await;
        let call_one_state = guard.calls.get(&call_one).expect("call one should exist");
        let call_two_state = guard.calls.get(&call_two).expect("call two should exist");
        assert_eq!(call_one_state.asr_backend, Some(LiveAsrBackend::Sherpa2023));
        assert!(call_one_state.conversation.attached);
        assert_eq!(call_one_state.conversation.mode, ConversationMode::Auto);
        assert_eq!(call_two_state.asr_backend, Some(LiveAsrBackend::Kroko2025));
        assert!(call_two_state.conversation.attached);
        assert_eq!(call_two_state.conversation.mode, ConversationMode::Auto);
    }

    #[tokio::test]
    async fn answer_requires_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let _gateway_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::PendingInbound,
            );
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("answer").await.expect("answer");

        assert!(output.lines[0].starts_with("answer requested for gwc_"));
        let guard = state.read().await;
        let call = guard.calls.values().next().expect("call exists");
        assert_eq!(call.status, CallStatus::Answering);
        assert_eq!(call.asr_backend, Some(LiveAsrBackend::Kroko2025));
        assert!(call.conversation.attached);
        assert_eq!(call.conversation.mode, ConversationMode::Auto);
    }

    #[tokio::test]
    async fn answer_rejects_non_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let _gateway_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::Answered,
            );
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let error = engine
            .run_line("answer")
            .await
            .expect_err("non-pending answer should be rejected");

        assert!(error.to_string().contains("expected waiting"));
        let guard = state.read().await;
        let call = guard.calls.values().next().expect("call exists");
        assert_eq!(call.status, CallStatus::Answered);
    }

    #[tokio::test]
    async fn answer_prefers_single_waiting_call_over_selected_ended_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let (waiting_call_id, ended_call_id) = {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let ended_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-ended".to_string(),
                    call_session_id: Some("sess-ended".to_string()),
                    call_leg_id: Some("leg-ended".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::Ended,
            );
            let waiting_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-waiting".to_string(),
                    call_session_id: Some("sess-waiting".to_string()),
                    call_leg_id: Some("leg-waiting".to_string()),
                    stream_id: None,
                },
                None,
                None,
                CallStatus::PendingInbound,
            );
            (waiting_call_id, ended_call_id)
        };
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let mut context = GatewayContext::new(state.clone(), telnyx);
        context.session.selected_call = Some(ended_call_id);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("answer").await.expect("answer");

        assert_eq!(
            output.lines,
            vec![format!("answer requested for {waiting_call_id}")]
        );
        let guard = state.read().await;
        assert_eq!(
            engine.context().session.selected_call.as_deref(),
            Some(waiting_call_id.as_str())
        );
        let waiting_call = guard
            .calls
            .get(&waiting_call_id)
            .expect("waiting call exists");
        assert_eq!(waiting_call.status, CallStatus::Answering);
        assert!(waiting_call.conversation.attached);
        assert_eq!(waiting_call.conversation.mode, ConversationMode::Auto);
    }

    #[tokio::test]
    async fn dial_transcribe_creates_outbound_call_session() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.selected_connection_id = Some("conn-1".to_string());
            guard.config.selected_phone_number = Some("<from-phone-number>".to_string());
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            guard.config.public_webhook_url =
                Some("https://example.test/telnyx/webhooks".to_string());
            guard.config.telnyx_media = TelnyxMediaConfig::new(TelnyxStreamCodec::L16, 16_000)
                .expect("L16 config should be valid");
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line("asr use kroko-2025")
            .await
            .expect("asr use");
        let output = engine
            .run_line("test dial-transcribe <to-phone-number>")
            .await
            .expect("dial-transcribe");

        assert!(output.lines[0].starts_with("dial-transcribe requested for gwc_"));
        let guard = state.read().await;
        let call_id = engine
            .context()
            .session
            .selected_call
            .as_deref()
            .expect("outbound call should be selected");
        let call = guard.calls.get(call_id).expect("call should exist");
        assert_eq!(
            call.direction,
            crate::operator::state::CallDirection::Outbound
        );
        assert_eq!(call.status, CallStatus::Dialing);
        assert_eq!(call.from.as_deref(), Some("<from-phone-number>"));
        assert_eq!(call.to.as_deref(), Some("<to-phone-number>"));
        assert!(call.ids.call_control_id.starts_with("dry-run-dial-"));
        assert_eq!(call.asr_backend, Some(LiveAsrBackend::Kroko2025));
        assert!(call.conversation.attached);
        assert_eq!(call.conversation.mode, ConversationMode::Auto);
    }

    #[tokio::test]
    async fn dial_creates_outbound_call_session_for_tts() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.selected_connection_id = Some("conn-1".to_string());
            guard.config.default_from_number = Some("<from-phone-number>".to_string());
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            guard.config.public_webhook_url =
                Some("https://example.test/telnyx/webhooks".to_string());
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine
            .run_line("dial <to-phone-number>")
            .await
            .expect("dial should create dry-run outbound call");

        assert!(output.lines[0].starts_with("dial requested for gwc_"));
        assert!(output
            .lines
            .iter()
            .any(|line| line
                == "conversation attached in auto mode; smoke-test echo replies require:"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "  conversation smoke-test on"));
        assert!(output
            .lines
            .iter()
            .any(|line| line.contains("speak ") && line.contains("<text to say>")));
        let guard = state.read().await;
        let call_id = engine
            .context()
            .session
            .selected_call
            .as_deref()
            .expect("outbound dial should select call");
        let call = guard.calls.get(call_id).expect("call should exist");
        assert_eq!(
            call.direction,
            crate::operator::state::CallDirection::Outbound
        );
        assert_eq!(call.status, CallStatus::Dialing);
        assert_eq!(call.from.as_deref(), Some("<from-phone-number>"));
        assert_eq!(call.to.as_deref(), Some("<to-phone-number>"));
        assert!(call.ids.call_control_id.starts_with("dry-run-dial-"));
        assert_eq!(call.asr_backend, Some(LiveAsrBackend::Kroko2025));
    }

    #[tokio::test]
    async fn speak_queues_tts_frames_over_media_registry() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            add_streaming_call(&mut guard, "call-1", "stream-1")
        };
        let media = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(16);
        media.register_call(call_id.clone(), tx).await;
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state.clone(), telnyx, media);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let output = engine
            .run_line("speak Hello from Motlie.")
            .await
            .expect("speak should queue TTS");

        assert!(output.lines[0].contains("speak queued"));
        let mut frames = 0usize;
        let mut mark_playback = None;
        for _ in 0..8 {
            match receive_outbound(&mut rx).await {
                OutboundMediaCommand::Frame(frame) => {
                    frames = frames.saturating_add(1);
                    assert!(!frame.payload.is_empty());
                }
                OutboundMediaCommand::Mark { playback_id } => {
                    mark_playback = Some(playback_id);
                    break;
                }
                OutboundMediaCommand::Clear { .. } => panic!("speak should not clear media"),
                OutboundMediaCommand::AppendState { .. } => {}
            }
        }

        assert_eq!(frames, 5);
        assert!(mark_playback.is_some());
        let guard = state.read().await;
        let tts = guard
            .calls
            .get(&call_id)
            .and_then(|call| call.tts.as_ref())
            .expect("tts state should exist");
        assert_eq!(tts.status, TtsPlaybackStatus::Playing);
        assert_eq!(tts.frames_queued, 5);
        assert_eq!(tts.frames_sent, 0);
    }

    #[tokio::test]
    async fn speak_rejects_manual_audio_when_text_call_stream_is_attached() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            add_streaming_call(&mut guard, "call-1", "stream-1")
        };
        let media = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(16);
        media.register_call(call_id.clone(), tx).await;
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state, telnyx, media);
        let _text_call_rx = context
            .text_calls
            .insert_test_session(call_id.clone())
            .await;
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let error = engine
            .run_line("speak this should not interrupt the agent stream")
            .await
            .expect_err("manual speak should be rejected during text-call stream");

        assert!(error.to_string().contains("manual speak is disabled"));
        assert!(error.to_string().contains("text-call stream"));
    }

    #[tokio::test]
    async fn speak_cancel_sends_clear_and_cancels_active_job() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            add_streaming_call(&mut guard, "call-1", "stream-1")
        };
        let media = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media.register_call(call_id.clone(), tx).await;
        let cancel = SpeechCancelToken::default();
        media
            .start_speech(&call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        state.write().await.start_tts_job(
            &call_id,
            "tts_test".to_string(),
            LiveTtsBackend::default(),
            "hello",
        );
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state.clone(), telnyx, media);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let output = engine
            .run_line("speak cancel")
            .await
            .expect("cancel should send clear");

        assert_eq!(
            output.lines,
            vec![format!(
                "speak cancel requested for {call_id} playback=tts_test"
            )]
        );
        assert!(cancel.is_canceled());
        let guard = state.read().await;
        let status = guard
            .calls
            .get(&call_id)
            .and_then(|call| call.tts.as_ref())
            .map(|tts| tts.status)
            .expect("tts status should exist");
        assert_eq!(status, TtsPlaybackStatus::Canceling);
    }

    #[tokio::test]
    async fn speak_rejects_overlapping_active_tts_job() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            add_streaming_call(&mut guard, "call-1", "stream-1")
        };
        let media = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media.register_call(call_id.clone(), tx).await;
        media
            .start_speech(
                &call_id,
                "tts_existing".to_string(),
                SpeechCancelToken::default(),
            )
            .await
            .expect("register active speech");
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state, telnyx, media);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let error = engine
            .run_line("speak overlapping speech")
            .await
            .expect_err("overlapping speech should be rejected");

        assert!(error.to_string().contains("active speech job"));
    }

    #[tokio::test]
    async fn command_sources_keep_speech_selection_isolated() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let (call_one, call_two) = {
            let mut guard = state.write().await;
            (
                add_streaming_call(&mut guard, "call-1", "stream-1"),
                add_streaming_call(&mut guard, "call-2", "stream-2"),
            )
        };
        let media = SharedMediaRegistry::default();
        let (tx_one, mut rx_one) = mpsc::channel(16);
        let (tx_two, mut rx_two) = mpsc::channel(16);
        media.register_call(call_one.clone(), tx_one).await;
        media.register_call(call_two.clone(), tx_two).await;
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let base = context_with_static_tts(state, telnyx, media);
        let mut tui_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());
        let mut socket_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());

        tui_engine
            .run_line(&format!("call use {call_one}"))
            .await
            .expect("TUI call use");
        socket_engine
            .run_line(&format!("call use {call_two}"))
            .await
            .expect("socket call use");
        tui_engine
            .run_line("speak TUI source says hello.")
            .await
            .expect("TUI speak");
        socket_engine
            .run_line("speak Socket source says hello.")
            .await
            .expect("socket speak");

        let first_playback = receive_frame_playback(&mut rx_one).await;
        let second_playback = receive_frame_playback(&mut rx_two).await;
        assert_ne!(first_playback, second_playback);
        assert_eq!(tui_engine.context().session.selected_call, Some(call_one));
        assert_eq!(
            socket_engine.context().session.selected_call,
            Some(call_two)
        );
    }

    #[tokio::test]
    async fn conversation_smoke_test_command_toggles_echo_handler() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let status = engine.run_line("status").await.expect("status");
        assert!(status
            .lines
            .iter()
            .any(|line| line == "conversation-handler: disabled"));

        let enabled = engine
            .run_line("conversation smoke-test on")
            .await
            .expect("enable smoke test");
        assert_eq!(
            enabled.lines,
            vec!["conversation smoke-test: on", "conversation barge-in: off"]
        );
        assert!(engine.context().conversation.smoke_test_enabled());
        assert!(!engine.context().conversation.barge_in_enabled());
        assert!(!state.read().await.quality.config.barge_in.enabled);

        let disabled = engine
            .run_line("conversation smoke-test off")
            .await
            .expect("disable smoke test");
        assert_eq!(disabled.lines, vec!["conversation smoke-test: off"]);
        assert!(!engine.context().conversation.smoke_test_enabled());
    }

    #[tokio::test]
    async fn conversation_barge_in_command_updates_quality_config_and_runtime_bridge() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let status = engine
            .run_line("conversation barge-in status")
            .await
            .expect("barge-in status");
        assert_eq!(status.lines, vec!["conversation barge-in: on"]);
        assert!(engine.context().conversation.barge_in_enabled());
        assert!(state.read().await.quality.config.barge_in.enabled);

        let disabled = engine
            .run_line("conversation barge-in off")
            .await
            .expect("disable barge-in");
        assert_eq!(disabled.lines, vec!["conversation barge-in: off"]);
        assert!(!engine.context().conversation.barge_in_enabled());
        assert!(!state.read().await.quality.config.barge_in.enabled);

        let enabled = engine
            .run_line("conversation barge-in on")
            .await
            .expect("enable barge-in");
        assert_eq!(enabled.lines, vec!["conversation barge-in: on"]);
        assert!(engine.context().conversation.barge_in_enabled());
        assert!(state.read().await.quality.config.barge_in.enabled);
    }

    #[tokio::test]
    async fn conversation_commands_attach_mode_detach_and_keep_source_selection_isolated() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let (call_one, call_two) = {
            let mut guard = state.write().await;
            (
                add_streaming_call(&mut guard, "call-1", "stream-1"),
                add_streaming_call(&mut guard, "call-2", "stream-2"),
            )
        };
        let media = SharedMediaRegistry::default();
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let base = context_with_static_tts(state.clone(), telnyx, media);
        let mut tui_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());
        let mut socket_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(base.for_new_source());

        tui_engine
            .run_line(&format!("call use {call_one}"))
            .await
            .expect("TUI call use");
        socket_engine
            .run_line(&format!("call use {call_two}"))
            .await
            .expect("socket call use");
        let attach = tui_engine
            .run_line("conversation attach")
            .await
            .expect("TUI conversation attach");
        let mode = socket_engine
            .run_line("conversation mode auto")
            .await
            .expect("socket conversation mode");

        assert_eq!(
            attach.lines,
            vec![format!("conversation attached for {call_one} mode=auto")]
        );
        assert_eq!(
            mode.lines,
            vec![format!("conversation mode for {call_two}: auto (attached)")]
        );
        let socket_status = socket_engine
            .run_line("conversation status")
            .await
            .expect("socket conversation status");
        assert!(socket_status.lines.iter().any(|line| line == "mode: auto"));

        tui_engine
            .run_line("conversation detach")
            .await
            .expect("TUI conversation detach");
        let guard = state.read().await;
        let tui_call = guard.calls.get(&call_one).expect("TUI call exists");
        let socket_call = guard.calls.get(&call_two).expect("socket call exists");
        assert!(!tui_call.conversation.attached);
        assert_eq!(tui_call.conversation.mode, ConversationMode::Auto);
        assert!(socket_call.conversation.attached);
        assert_eq!(socket_call.conversation.mode, ConversationMode::Auto);
    }

    #[tokio::test]
    async fn conversation_approve_speaks_pending_manual_proposal() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            let call_id = add_streaming_call(&mut guard, "call-1", "stream-1");
            guard.attach_conversation(&call_id, ConversationMode::Manual);
            guard.record_conversation_proposal(&call_id, "Approved response".to_string());
            call_id
        };
        let media = SharedMediaRegistry::default();
        let (tx, mut rx) = mpsc::channel(16);
        media.register_call(call_id.clone(), tx).await;
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state.clone(), telnyx, media);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let output = engine
            .run_line("conversation approve")
            .await
            .expect("approve should queue proposed response");

        assert!(
            output.lines[0].starts_with(&format!("conversation approved for {call_id} playback="))
        );
        let playback_id = receive_frame_playback(&mut rx).await;
        let guard = state.read().await;
        let call = guard.calls.get(&call_id).expect("call exists");
        assert_eq!(call.conversation.status, ConversationStatus::Speaking);
        assert_eq!(
            call.conversation.last_playback_id.as_deref(),
            Some(playback_id.as_str())
        );
        assert_eq!(call.conversation.lines.len(), 1);
        assert_eq!(call.conversation.lines[0].text, "Approved response");
    }

    #[tokio::test]
    async fn conversation_disapprove_detaches_and_cancels_active_tts() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let call_id = {
            let mut guard = state.write().await;
            let call_id = add_streaming_call(&mut guard, "call-1", "stream-1");
            guard.attach_conversation(&call_id, ConversationMode::Auto);
            call_id
        };
        let media = SharedMediaRegistry::default();
        let (tx, _rx) = mpsc::channel(4);
        media.register_call(call_id.clone(), tx).await;
        let cancel = SpeechCancelToken::default();
        media
            .start_speech(&call_id, "tts_test".to_string(), cancel.clone())
            .await
            .expect("register active speech");
        state.write().await.start_tts_job(
            &call_id,
            "tts_test".to_string(),
            LiveTtsBackend::default(),
            "hello",
        );
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state.clone(), telnyx, media);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line(&format!("call use {call_id}"))
            .await
            .expect("select call");
        let output = engine
            .run_line("conversation disapprove")
            .await
            .expect("disapprove should cancel active speech");

        assert_eq!(
            output.lines,
            vec![format!(
                "conversation disapproved for {call_id}; canceled playback=tts_test; transcription-only"
            )]
        );
        assert!(cancel.is_canceled());
        let guard = state.read().await;
        let call = guard.calls.get(&call_id).expect("call should exist");
        assert!(!call.conversation.attached);
        assert_eq!(call.conversation.status, ConversationStatus::Idle);
        let status = call
            .tts
            .as_ref()
            .map(|tts| tts.status)
            .expect("tts status should exist");
        assert_eq!(status, TtsPlaybackStatus::Canceling);
    }

    #[tokio::test]
    async fn call_show_returns_assembled_transcript() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let gateway_call_id = {
            let mut guard = state.write().await;
            let gateway_call_id = guard.add_or_update_inbound_call(
                TelnyxIds {
                    call_control_id: "call-1".to_string(),
                    call_session_id: Some("sess-1".to_string()),
                    call_leg_id: Some("leg-1".to_string()),
                    stream_id: Some("stream-1".to_string()),
                },
                Some("<from-phone-number>".to_string()),
                Some("<to-phone-number>".to_string()),
                CallStatus::Answered,
            );
            guard.add_transcript(
                &gateway_call_id,
                crate::operator::state::TranscriptKind::Final,
                "HELLO".to_string(),
            );
            guard.add_transcript(
                &gateway_call_id,
                crate::operator::state::TranscriptKind::Partial,
                "WORLD".to_string(),
            );
            gateway_call_id
        };
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let mut context = GatewayContext::new(state, telnyx);
        context.session.selected_call = Some(gateway_call_id);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("call show").await.expect("call show");

        assert!(output
            .lines
            .iter()
            .any(|line| line == "assembled transcript:"));
        assert!(output.lines.iter().any(|line| line == "HELLO WORLD"));
        assert!(output.lines.iter().any(|line| line == "final: HELLO"));
        assert!(output.lines.iter().any(|line| line == "partial: WORLD"));
    }

    #[tokio::test]
    async fn command_sources_keep_selected_calls_isolated() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let (call_one, call_two) = {
            let mut guard = state.write().await;
            let _ = add_test_call(&mut guard, "call-1");
            let _ = add_test_call(&mut guard, "call-2");
            let ordered = crate::operator::session::ordered_call_ids(&guard);
            (ordered[0].clone(), ordered[1].clone())
        };
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let mut tui_engine = CommandEngine::<GatewayContext, GatewayCommand>::new(
            GatewayContext::new(state.clone(), telnyx.clone()),
        );
        let mut socket_engine = CommandEngine::<GatewayContext, GatewayCommand>::new(
            GatewayContext::new(state.clone(), telnyx),
        );

        tui_engine
            .run_line(&format!("call use {call_one}"))
            .await
            .expect("TUI call use");
        socket_engine
            .run_line(&format!("call use {call_one}"))
            .await
            .expect("socket call use");
        {
            let mut guard = state.write().await;
            let moved = tui_engine
                .context_mut()
                .session
                .move_selection(&mut guard, 1)
                .expect("TUI cursor move should select another call");
            assert_eq!(moved, call_two);
        }

        let tui_show = tui_engine.run_line("call show").await.expect("TUI show");
        let socket_show = socket_engine
            .run_line("call show")
            .await
            .expect("socket show");

        assert!(tui_show
            .lines
            .iter()
            .any(|line| line == &format!("call: {call_two}")));
        assert!(socket_show
            .lines
            .iter()
            .any(|line| line == &format!("call: {call_one}")));
    }

    fn add_test_call(state: &mut GatewayState, call_control_id: &str) -> String {
        state.add_or_update_inbound_call(
            TelnyxIds {
                call_control_id: call_control_id.to_string(),
                call_session_id: None,
                call_leg_id: None,
                stream_id: None,
            },
            None,
            None,
            CallStatus::PendingInbound,
        )
    }

    fn add_streaming_call(
        state: &mut GatewayState,
        call_control_id: &str,
        stream_id: &str,
    ) -> String {
        let call_id = state.add_or_update_outbound_call(
            TelnyxIds {
                call_control_id: call_control_id.to_string(),
                call_session_id: Some(format!("session-{call_control_id}")),
                call_leg_id: Some(format!("leg-{call_control_id}")),
                stream_id: Some(stream_id.to_string()),
            },
            Some("<from-phone-number>".to_string()),
            Some("<to-phone-number>".to_string()),
            CallStatus::MediaStarted,
        );
        let call = state
            .calls
            .get_mut(&call_id)
            .expect("new streaming call should exist");
        call.media = MediaMetadata {
            stream_id: Some(stream_id.to_string()),
            encoding: Some("PCMU".to_string()),
            sample_rate_hz: Some(8_000),
            channels: Some(1),
            track: Some("inbound".to_string()),
        };
        call.asr_backend = Some(LiveAsrBackend::Kroko2025);
        call_id
    }

    #[tokio::test]
    async fn quality_status_defaults_to_transcript_text_disabled() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine
            .run_line("quality status")
            .await
            .expect("quality status");

        assert!(output
            .lines
            .iter()
            .any(|line| line == "include_transcript_text=false"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "redaction_mode=metrics-only"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "barge_in.onset_during_playback=defer_to_partial"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "echo_suppression.enabled=true"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "echo_suppression.tail_window_ms=2000"));
    }

    #[tokio::test]
    async fn quality_endpoint_command_clamps_and_marks_report_only_knobs() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let live_output = engine
            .run_line("quality endpoint trailing-silence-ms 10")
            .await
            .expect("set trailing silence");
        assert!(live_output.lines[0].contains("key=endpoint.trailing_silence_ms"));
        assert!(live_output.lines[0].contains("applies=next_asr_session"));
        assert!(live_output.lines[0].contains("clamped=true"));
        assert_eq!(
            state
                .read()
                .await
                .quality
                .config
                .endpoint
                .trailing_silence_ms,
            100
        );

        let report_output = engine
            .run_line("quality endpoint merge-window-ms 9999")
            .await
            .expect("set merge window");
        assert!(report_output.lines[0].contains("key=endpoint.merge_window_ms"));
        assert!(report_output.lines[0].contains("applies=new_turn"));
        assert_eq!(
            state.read().await.quality.config.endpoint.merge_window_ms,
            5_000
        );

        let final_settle_output = engine
            .run_line("quality endpoint final-settle-ms 9999")
            .await
            .expect("set final settle");
        assert!(final_settle_output.lines[0].contains("key=endpoint.final_settle_ms"));
        assert!(final_settle_output.lines[0].contains("applies=next_asr_session"));
        assert!(final_settle_output.lines[0].contains("clamped=true"));
        assert_eq!(
            state.read().await.quality.config.endpoint.final_settle_ms,
            5_000
        );

        let conversation_hold_output = engine
            .run_line("quality endpoint conversation-incomplete-tail-hold-ms 99999")
            .await
            .expect("set conversation incomplete tail hold");
        assert!(conversation_hold_output.lines[0]
            .contains("key=endpoint.conversation_incomplete_tail_hold_ms"));
        assert!(conversation_hold_output.lines[0].contains("applies=new_turn"));
        assert!(conversation_hold_output.lines[0].contains("clamped=true"));
        assert_eq!(
            state
                .read()
                .await
                .quality
                .config
                .endpoint
                .conversation_incomplete_tail_hold_ms,
            10_000
        );
    }

    #[tokio::test]
    async fn quality_text_call_commands_update_live_config_knobs() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        engine
            .run_line("quality text-call media-ready-timeout-ms 2345")
            .await
            .expect("set media-ready timeout");
        engine
            .run_line("quality text-call playback-wait-timeout-ms 3456")
            .await
            .expect("set playback-wait timeout");
        engine
            .run_line("quality text-call latest-response-wins off")
            .await
            .expect("set latest policy");
        engine
            .run_line("quality text-call callback-timeout-ms 4567")
            .await
            .expect("set callback timeout");

        let status = engine
            .run_line("quality text-call status")
            .await
            .expect("text-call status");
        assert!(status
            .lines
            .iter()
            .any(|line| line == "media_ready_timeout_ms=2345"));
        assert!(status
            .lines
            .iter()
            .any(|line| line == "playback_wait_timeout_ms=3456"));
        assert!(status
            .lines
            .iter()
            .any(|line| line == "latest_response_wins=false"));
        assert!(status
            .lines
            .iter()
            .any(|line| line == "callback_timeout_ms=4567"));

        let guard = state.read().await;
        assert_eq!(guard.quality.config.text_call.media_ready_timeout_ms, 2_345);
        assert_eq!(
            guard.quality.config.text_call.playback_wait_timeout_ms,
            3_456
        );
        assert!(!guard.quality.config.text_call.latest_response_wins);
        assert_eq!(guard.quality.config.text_call.callback_timeout_ms, 4_567);
    }

    #[tokio::test]
    async fn quality_tts_commands_update_live_config_knobs() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let mode_output = engine
            .run_line("quality tts generation-mode streaming")
            .await
            .expect("set TTS generation mode");
        assert!(mode_output.lines[0].contains("key=tts.generation_mode"));
        assert!(mode_output.lines[0].contains("value=streaming"));
        assert!(mode_output.lines[0].contains("applies=new_playback_request"));

        let chunk_output = engine
            .run_line("quality tts max-text-chunk-chars 10")
            .await
            .expect("set max text chunk chars");
        assert!(chunk_output.lines[0].contains("key=tts.max_text_chunk_chars"));
        assert!(chunk_output.lines[0].contains("clamped=true"));
        let first_chunk_output = engine
            .run_line("quality tts first-chunk-max-chars 10")
            .await
            .expect("set first chunk max chars");
        assert!(first_chunk_output.lines[0].contains("key=tts.first_chunk_max_chars"));
        assert!(first_chunk_output.lines[0].contains("clamped=true"));
        let disabled_first_chunk_output = engine
            .run_line("quality tts first-chunk-max-chars 0")
            .await
            .expect("disable first chunk max chars");
        assert!(disabled_first_chunk_output.lines[0].contains("key=tts.first_chunk_max_chars"));
        assert!(disabled_first_chunk_output.lines[0].contains("value=0"));
        let first_chunk_output = engine
            .run_line("quality tts first-chunk-max-chars 45")
            .await
            .expect("set first chunk max chars");
        assert!(first_chunk_output.lines[0].contains("value=45"));
        let prebuffer_output = engine
            .run_line("quality tts prebuffer-chunks 3")
            .await
            .expect("set prebuffer chunks");
        assert!(prebuffer_output.lines[0].contains("key=tts.prebuffer_chunks"));
        assert!(prebuffer_output.lines[0].contains("applies=new_playback_request"));

        let status = engine
            .run_line("quality tts status")
            .await
            .expect("tts status");
        assert!(status
            .lines
            .iter()
            .any(|line| line == "generation_mode=streaming"));
        assert!(status
            .lines
            .iter()
            .any(|line| line == "max_text_chunk_chars=40"));
        assert!(status
            .lines
            .iter()
            .any(|line| line == "first_chunk_max_chars=45"));
        assert!(status.lines.iter().any(|line| line == "prebuffer_chunks=3"));

        let guard = state.read().await;
        assert_eq!(
            guard.quality.config.tts.generation_mode,
            TtsGenerationMode::Streaming
        );
        assert_eq!(guard.quality.config.tts.max_text_chunk_chars, 40);
        assert_eq!(guard.quality.config.tts.first_chunk_max_chars, 45);
        assert_eq!(guard.quality.config.tts.prebuffer_chunks, 3);
    }

    #[tokio::test]
    async fn quality_barge_in_and_echo_commands_update_live_config_knobs() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let onset_output = engine
            .run_line("quality barge-in onset-during-playback trust")
            .await
            .expect("set onset during playback policy");
        assert!(onset_output.lines[0].contains("key=barge_in.onset_during_playback"));
        assert!(onset_output.lines[0].contains("value=trust"));
        assert!(onset_output.lines[0].contains("applies=next_asr_session"));

        let disabled = engine
            .run_line("quality echo-suppression off")
            .await
            .expect("disable echo suppression");
        assert!(disabled.lines[0].contains("key=echo_suppression.enabled"));
        assert!(disabled.lines[0].contains("value=false"));
        let tail = engine
            .run_line("quality echo-suppression tail-window-ms 50")
            .await
            .expect("set echo tail window");
        assert!(tail.lines[0].contains("key=echo_suppression.tail_window_ms"));
        let short = engine
            .run_line("quality echo-suppression short-token-coverage-percent 101")
            .await
            .expect("set echo short coverage");
        assert!(short.lines[0].contains("value=100"));
        assert!(short.lines[0].contains("clamped=true"));

        let barge_status = engine
            .run_line("quality barge-in status")
            .await
            .expect("barge status");
        assert!(barge_status
            .lines
            .iter()
            .any(|line| line == "onset_during_playback=trust"));
        let echo_status = engine
            .run_line("quality echo-suppression status")
            .await
            .expect("echo status");
        assert!(echo_status.lines.iter().any(|line| line == "enabled=false"));
        assert!(echo_status
            .lines
            .iter()
            .any(|line| line == "tail_window_ms=50"));
        assert!(echo_status
            .lines
            .iter()
            .any(|line| line == "short_token_coverage_percent=100"));

        let guard = state.read().await;
        assert_eq!(
            guard.quality.config.barge_in.onset_during_playback,
            OnsetDuringPlaybackPolicy::Trust
        );
        assert!(!guard.quality.config.echo_suppression.enabled);
        assert_eq!(guard.quality.config.echo_suppression.tail_window_ms, 50);
        assert_eq!(
            guard
                .quality
                .config
                .echo_suppression
                .short_token_coverage_percent,
            100
        );
    }

    #[tokio::test]
    async fn quality_state_dump_replays_exact_resolved_config() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            let config = &mut guard.quality.config;
            config.set_profile(QualityProfile::Noisy);
            config.set_speech_rms_threshold(321.0).expect("finite RMS");
            config.set_speech_peak_threshold(1_234);
            config.set_speech_onset_min_silence_ms(333);
            config.set_endpoint_trailing_silence_ms(777);
            config.set_endpoint_min_turn_words(4);
            config.set_endpoint_min_turn_chars(12);
            config.set_endpoint_merge_window_ms(444);
            config.set_endpoint_final_settle_ms(333);
            config.set_endpoint_conversation_incomplete_tail_hold_ms(2_345);
            config.set_endpoint_conversation_low_confidence_threshold_percent(39);
            config.set_endpoint_conversation_playback_hold_poll_ms(77);
            config.set_endpoint_max_turn_words(123);
            config.set_endpoint_max_turn_duration_ms(7_654);
            config.set_asr_repeated_token_run_threshold(42);
            config.set_asr_repeated_q_run_threshold(12);
            config.set_asr_finish_pad_ms(222);
            config.set_text_call_max_active_turns(9);
            config.set_text_call_media_ready_timeout_ms(12_345);
            config.set_text_call_playback_wait_timeout_ms(54_321);
            config.set_text_call_latest_response_wins(false);
            config.set_text_call_callback_timeout_ms(1_234);
            config.set_tts_generation_mode(TtsGenerationMode::Streaming);
            config.set_tts_max_text_chunk_chars(88);
            config.set_tts_first_chunk_max_chars(44);
            config.set_tts_prebuffer_chunks(4);
            config.set_barge_in_enabled(false);
            config.barge_in.speech_onset_cancel_enabled = false;
            config.barge_in.partial_asr_cancel_enabled = false;
            config.barge_in.final_asr_cancel_enabled = true;
            config.set_barge_in_clear_timeout_ms(2_222);
            config.set_logging_enabled(true);
            config.set_logging_queue_capacity(8_192);
            config
                .set_logging_per_frame_sample_rate(0.25)
                .expect("valid sample rate");
            config.set_logging_include_transcript_text(false);
            config.set_logging_redaction_mode(RedactionMode::HashedText);
            config.set_quality_judge_enabled(true);
            config
                .set_quality_judge_sample_rate(0.5)
                .expect("valid judge sample rate");
            config.quality_judge.model = "judge-model-test".to_string();
            config.set_quality_judge_batch_size(7);
            config.set_quality_judge_timeout_ms(22_000);
            config.targets.p50_endpoint_trailing_silence_ms = 801;
            config.targets.p95_endpoint_trailing_silence_ms = 1_401;
            config.targets.p50_turn_to_playback_started_ms = 1_101;
            config.targets.p95_turn_to_playback_started_ms = 2_901;
            config.targets.max_incomplete_turn_rate = 0.11;
            config.targets.max_overmerged_turn_rate = 0.12;
            config.targets.max_garbled_turn_rate = 0.13;
            config.targets.max_inappropriate_cancel_rate = 0.14;
            guard.quality.config_id = config.config_id();
        }
        let expected_config = state.read().await.quality.config.clone();
        let expected_config_id = expected_config.config_id();
        let dump = {
            let guard = state.read().await;
            crate::operator::persistence::render_state_dump(&guard)
        };
        assert!(dump.contains("quality restore-config "));

        let path = std::env::temp_dir().join(format!(
            "motlie-quality-state-{}.repl",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, dump).expect("write quality state dump");

        let replay_state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let replay_telnyx = TelnyxClient::new("https://api.example.test", None, true);
        let replay_context = GatewayContext::new(replay_state.clone(), replay_telnyx);
        let mut replay_engine =
            CommandEngine::<GatewayContext, GatewayCommand>::new(replay_context);
        crate::operator::script::run_repl_file(&mut replay_engine, &path)
            .await
            .expect("replay quality state dump");

        let replay_guard = replay_state.read().await;
        assert_eq!(replay_guard.quality.config, expected_config);
        assert_eq!(replay_guard.quality.config_id, expected_config_id);
        let _ = std::fs::remove_file(path);
    }

    fn test_asr_registry() -> SharedAsrRegistry {
        let echo = Arc::new(EchoAsrFactory);
        Arc::new(AsrRegistry::new(echo.clone(), echo))
    }

    fn context_with_static_tts(
        state: SharedState,
        telnyx: TelnyxClient,
        media: SharedMediaRegistry,
    ) -> GatewayContext {
        let tts = Arc::new(TtsRegistry::new(
            Arc::new(StaticTtsFactory::with_sample_rate(
                vec![1_000; 2_400],
                KOKORO_SAMPLE_RATE_HZ,
            )),
            Arc::new(StaticTtsFactory::new(vec![1_000; 2_205])),
        ));
        let conversation = ConversationRuntime::new(
            telnyx.clone(),
            tts.clone(),
            default_conversation_handler(),
            false,
        );
        GatewayContext::with_services(
            state,
            telnyx,
            test_asr_registry(),
            media,
            tts,
            conversation,
            LiveAsrBackend::Kroko2025,
        )
    }

    async fn receive_outbound(
        rx: &mut mpsc::Receiver<OutboundMediaCommand>,
    ) -> OutboundMediaCommand {
        timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("outbound media command should arrive")
            .expect("outbound media channel should remain open")
    }

    async fn receive_frame_playback(rx: &mut mpsc::Receiver<OutboundMediaCommand>) -> String {
        match receive_outbound(rx).await {
            OutboundMediaCommand::Frame(frame) => frame.playback_id,
            other => panic!("expected outbound frame, got {other:?}"),
        }
    }
}
