use std::path::PathBuf;

use crate::adapter::LiveAsrBackend;
use crate::call_control::{
    AnswerRequest, DialRequest, TelnyxClient, TelnyxMediaConfig, TelnyxStreamCodec,
};
use crate::media::SharedMediaRegistry;
#[cfg(test)]
use crate::media::{OutboundMediaCommand, SpeechCancelToken};
use crate::operator::persistence::write_state_dump;
use crate::operator::session::OperatorSession;
use crate::operator::state::{
    CallStatus, ConversationMode, GatewayState, InboundMode, LogLevel, SharedState,
};
use crate::speech;
use crate::tts::{unavailable_registry, LiveTtsBackend, SharedTtsRegistry};
use async_trait::async_trait;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use motlie_driver::{CommandOutput, CommandSet, DriverError, DriverResult};

#[derive(Clone)]
pub struct GatewayContext {
    pub state: SharedState,
    pub telnyx: TelnyxClient,
    pub media: SharedMediaRegistry,
    pub tts: SharedTtsRegistry,
    pub session: OperatorSession,
}

impl GatewayContext {
    pub fn new(state: SharedState, telnyx: TelnyxClient) -> Self {
        Self::with_services(
            state,
            telnyx,
            SharedMediaRegistry::default(),
            unavailable_registry(),
            LiveAsrBackend::default(),
        )
    }

    pub fn with_services(
        state: SharedState,
        telnyx: TelnyxClient,
        media: SharedMediaRegistry,
        tts: SharedTtsRegistry,
        next_asr_backend: LiveAsrBackend,
    ) -> Self {
        Self {
            state,
            telnyx,
            media,
            tts,
            session: OperatorSession::new(next_asr_backend),
        }
    }

    pub fn for_new_source(&self) -> Self {
        let mut context = Self::with_services(
            self.state.clone(),
            self.telnyx.clone(),
            self.media.clone(),
            self.tts.clone(),
            self.session.next_asr_backend,
        );
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

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConversationModeArg {
    Manual,
    Auto,
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
pub enum ConversationCommand {
    Status {
        call: Option<String>,
    },
    Attach {
        call: Option<String>,
    },
    Detach {
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
            Self::Test { command } => test_command(context, command).await,
            Self::Transcript { command } => transcript_command(context, command).await,
            Self::Log { command } => log_command(context, command).await,
        }
    }
}

async fn status(context: &GatewayContext, target: Option<String>) -> DriverResult<CommandOutput> {
    if target.is_some() {
        return call_show(context, target).await;
    }
    let guard = context.state.read().await;
    let mut lines = vec![
        format!("listener: {:?}", guard.config.bind),
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
                            "active-call {} {} playback={} frames={}/{} text={}",
                            call.gateway_call_id,
                            tts.status.label(),
                            tts.playback_id,
                            tts.frames_sent,
                            tts.frames_queued,
                            tts.text_preview
                        )
                    })
                })
                .collect::<Vec<_>>();
            let backend = context.session.next_tts_backend;
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
            guard.log(
                LogLevel::Info,
                format!(
                    "source selected TTS backend {} ({}) for next speech",
                    backend.label(),
                    backend.model_label()
                ),
            );
            let mut lines = vec![format!(
                "tts backend for next speech: {} ({})",
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
            "wait for call state media/transcribing, then run:".to_string(),
            format!("  speak {gateway_call_id} <text to say>"),
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
                lines: conversation_status_lines(call),
                effects: Vec::new(),
            })
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
            guard.attach_conversation(&id, ConversationMode::Manual);
            context.session.selected_call = Some(id.clone());
            Ok(CommandOutput::line(format!(
                "conversation attached for {id} mode=manual"
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
        ConversationCommand::Mode { mode, call } => {
            let mode = ConversationMode::from(mode);
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
            guard.set_conversation_mode(&id, mode);
            context.session.selected_call = Some(id.clone());
            Ok(CommandOutput::line(format!(
                "conversation mode for {id}: {}",
                mode.label()
            )))
        }
    }
}

fn conversation_status_lines(call: &crate::operator::state::CallSession) -> Vec<String> {
    let conversation = &call.conversation;
    let mut lines = vec![
        format!("call: {}", call.gateway_call_id),
        format!("conversation: {}", conversation.status_label()),
        format!("attached: {}", conversation.attached),
        format!("mode: {}", conversation.mode.label()),
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
    let mut status = format!(
        "{} playback={} frames={}/{} text={}",
        tts.status.label(),
        tts.playback_id,
        tts.frames_sent,
        tts.frames_queued,
        tts.text_preview
    );
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
        "",
        "Core:",
        "  status                         Show listener, Telnyx, ASR, media, capture, and call state",
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
        "  tts use piper                 Select TTS backend for the next speak command",
        "",
        "Calls:",
        "  calls                          List calls in operator roster order",
        "  call use <call-id>             Select a call for this TUI/socket source",
        "  call show [call-id]            Show selected call detail and assembled transcript",
        "  status [call-id]               Show gateway status or selected call status",
        "  answer [call-id]               Answer one waiting inbound call",
        "  dial <+e164> [--from +e164]    Place an outbound call",
        "  speak [call-id] <text...>      Queue cancellable Piper TTS over the media socket",
        "  speak cancel [call-id]         Clear active TTS on the selected call",
        "  reject [call-id]",
        "  hangup [call-id]",
        "  transcript follow [call-id]",
        "  transcript clear [call-id]",
        "  log clear",
        "",
        "Testing:",
        "  test dial-transcribe <+e164> [--from +e164]",
        "",
        "Helpful topics:",
        "  help config       help telnyx       help inbound      help asr",
        "  help tts          help call         help outbound     help socket",
        "  help transcript   help test",
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

fn tts_help() -> String {
    [
        "tts list",
        "tts status",
        "tts use piper",
        "",
        "Inspect or select the outbound TTS backend used by `speak`.",
        "",
        "Milestone 2 supports Piper. If `tts status` reports unavailable, restart",
        "the gateway from a binary built with `--features \"sherpa piper\"`.",
        "",
        "After `dial`, wait until the call state is `media` or `transcribing`, then run:",
        "  speak <text...>",
        "or explicitly:",
        "  speak <call-id> <text...>",
        "",
        "Examples:",
        "  tts list",
        "  tts status",
        "  tts use piper",
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
        "  telnyx number bind +15551234567 <connection-id>",
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
        "  telnyx number use +15551234567",
        "  telnyx number bind +15551234567 1234567890",
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
        "Usage:",
        "  conversation status [call-id]",
        "  conversation attach [call-id]",
        "  conversation detach [call-id]",
        "  conversation mode <manual|auto> [call-id]",
        "",
        "Manual mode records the assistant response for operator review. Auto mode routes",
        "the generated response through the same Piper `speak` media path used by M2.",
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
        "Place an outbound call and send Motlie-generated TTS over the existing bidirectional",
        "Telnyx media WebSocket. `speak` is non-blocking and cancellable; `speak cancel` sends",
        "Telnyx clear and drops local queued outbound audio.",
        "",
        "Prerequisites:",
        "  telnyx app use <connection-id>",
        "  config set media-url <wss-url>",
        "  config set from-number <outbound-enabled +e164>",
        "  Telnyx account/application has an Outbound Voice Profile assigned",
        "",
        "Examples:",
        "  dial +15551234567",
        "  status",
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
        "  test dial-transcribe +15551234567 --from +15557654321",
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
        "",
        "Protocol:",
        "  Send one command line terminated by newline.",
        "  Receive one JSON object per command:",
        "    {\"ok\":true,\"lines\":[...],\"data\":{...},\"effects\":[...],\"error\":null}",
        "  `data` is present for status, calls, call show, tts list, and tts status polling.",
        "",
        "Discovery:",
        "  help",
        "  help socket",
        "  help inbound",
        "  help asr",
        "  help tts",
        "  help call",
        "  help dial",
        "",
        "Operational parity:",
        "  TUI and socket both use the same typed command language.",
        "  TUI-only keystrokes have command equivalents:",
        "    Tab focus change       no socket equivalent needed",
        "    Calls pane cursor     calls + call use <call-id>",
        "    Calls pane `a` attach call use <call-id>",
        "    Detail pane scroll    transcript follow / call show polling",
        "  Outbound M2 commands are also shared:",
        "    dial <+e164> [--from +e164]",
        "    speak [call-id] <text...>",
        "    speak cancel [call-id]",
        "    hangup [call-id]",
        "",
        "Source-local state:",
        "  Each socket connection has its own selected call, next ASR backend, and next TTS backend.",
        "  A later socket starts from the code defaults, not another source's choices.",
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
    use crate::operator::state::{
        shared_state, CallStatus, GatewayState, MediaMetadata, TelnyxIds, TtsPlaybackStatus,
    };
    use crate::tts::{StaticTtsFactory, TtsRegistry};

    #[tokio::test]
    async fn inbound_is_disabled_by_default() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("inbound status").await.expect("status");

        assert_eq!(output.lines, vec!["inbound disabled"]);
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
        assert!(rendered.contains("tts use piper"));
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
        let socket = engine.run_line("help socket").await.expect("socket help");

        assert!(asr.lines.join("\n").contains("source-local"));
        assert!(tts.lines.join("\n").contains("tts list"));
        assert!(tts.lines.join("\n").contains("speak <text...>"));
        assert!(call.lines.join("\n").contains("call use <call-id>"));
        assert!(socket.lines.join("\n").contains("Receive one JSON object"));
        assert!(socket.lines.join("\n").contains("Operational parity"));
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
            vec!["piper piper/en_us_ljspeech_medium available"]
        );
    }

    #[tokio::test]
    async fn tts_status_reports_unavailable_backend_clearly() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("tts status").await.expect("tts status");

        assert!(output.lines.iter().any(|line| line == "next=piper"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "next_model=piper/en_us_ljspeech_medium"));
        assert!(output.lines.iter().any(|line| line == "status=unavailable"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "reason=Piper TTS is unavailable; rebuild with --features piper"));
    }

    #[tokio::test]
    async fn tts_use_matches_asr_command_shape() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = context_with_static_tts(state, telnyx, SharedMediaRegistry::default());
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("tts use piper").await.expect("tts use");

        assert_eq!(
            output.lines,
            vec!["tts backend for next speech: piper (piper/en_us_ljspeech_medium)"]
        );
        assert_eq!(
            engine.context().session.next_tts_backend,
            LiveTtsBackend::Piper
        );
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
        assert_eq!(
            guard
                .calls
                .get(&call_one)
                .expect("call one should exist")
                .asr_backend,
            Some(LiveAsrBackend::Sherpa2023)
        );
        assert_eq!(
            guard
                .calls
                .get(&call_two)
                .expect("call two should exist")
                .asr_backend,
            Some(LiveAsrBackend::Kroko2025)
        );
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
        assert_eq!(
            guard
                .calls
                .get(&waiting_call_id)
                .expect("waiting call exists")
                .status,
            CallStatus::Answering
        );
    }

    #[tokio::test]
    async fn dial_transcribe_creates_outbound_call_session() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.selected_connection_id = Some("conn-1".to_string());
            guard.config.selected_phone_number = Some("+15550000001".to_string());
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
            .run_line("test dial-transcribe +15550000002")
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
        assert_eq!(call.from.as_deref(), Some("+15550000001"));
        assert_eq!(call.to.as_deref(), Some("+15550000002"));
        assert!(call.ids.call_control_id.starts_with("dry-run-dial-"));
        assert_eq!(call.asr_backend, Some(LiveAsrBackend::Kroko2025));
    }

    #[tokio::test]
    async fn dial_creates_outbound_call_session_for_tts() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.selected_connection_id = Some("conn-1".to_string());
            guard.config.default_from_number = Some("+15550000001".to_string());
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            guard.config.public_webhook_url =
                Some("https://example.test/telnyx/webhooks".to_string());
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine
            .run_line("dial +15550000002")
            .await
            .expect("dial should create dry-run outbound call");

        assert!(output.lines[0].starts_with("dial requested for gwc_"));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "wait for call state media/transcribing, then run:"));
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
        assert_eq!(call.from.as_deref(), Some("+15550000001"));
        assert_eq!(call.to.as_deref(), Some("+15550000002"));
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
        state
            .write()
            .await
            .start_tts_job(&call_id, "tts_test".to_string(), "hello");
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
            vec![format!("conversation attached for {call_one} mode=manual")]
        );
        assert_eq!(
            mode.lines,
            vec![format!("conversation mode for {call_two}: auto")]
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
        assert_eq!(tui_call.conversation.mode, ConversationMode::Manual);
        assert!(socket_call.conversation.attached);
        assert_eq!(socket_call.conversation.mode, ConversationMode::Auto);
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
                Some("+15550000001".to_string()),
                Some("+15550000002".to_string()),
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
            Some("+15550000001".to_string()),
            Some("+15550000002".to_string()),
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

    fn context_with_static_tts(
        state: SharedState,
        telnyx: TelnyxClient,
        media: SharedMediaRegistry,
    ) -> GatewayContext {
        let tts = Arc::new(TtsRegistry::new(Arc::new(StaticTtsFactory::new(vec![
            1_000;
            2_205
        ]))));
        GatewayContext::with_services(state, telnyx, media, tts, LiveAsrBackend::Kroko2025)
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
