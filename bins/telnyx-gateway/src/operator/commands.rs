use std::path::PathBuf;

use async_trait::async_trait;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use motlie_driver::{CommandOutput, CommandSet, DriverError, DriverResult};

use crate::call_control::{
    AnswerRequest, DialRequest, TelnyxClient, TelnyxMediaConfig, TelnyxStreamCodec,
};
use crate::operator::persistence::write_state_dump;
use crate::operator::state::{CallStatus, GatewayState, InboundMode, LogLevel, SharedState};

#[derive(Clone)]
pub struct GatewayContext {
    pub state: SharedState,
    pub telnyx: TelnyxClient,
}

impl GatewayContext {
    pub fn new(state: SharedState, telnyx: TelnyxClient) -> Self {
        Self { state, telnyx }
    }

    async fn answer_call(&self, target: Option<String>) -> DriverResult<CommandOutput> {
        let (gateway_call_id, call_control_id, stream_url, media) = {
            let mut guard = self.state.write().await;
            let media_url = guard
                .config
                .public_media_url
                .clone()
                .ok_or_else(|| DriverError::message("config set media-url <wss-url> first"))?;
            let media = guard.config.telnyx_media;
            let call_id = resolve_answer_call_id(&guard, target.as_deref())?;
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
                call.status = CallStatus::Answering;
                call.push_timeline("operator requested answer");
                (
                    call.gateway_call_id.clone(),
                    call.ids.call_control_id.clone(),
                )
            };
            guard.selected_call = Some(gateway_call_id.clone());
            (gateway_call_id, call_control_id, media_url, media)
        };

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
    Status,
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
    Calls,
    Call {
        #[command(subcommand)]
        command: CallCommand,
    },
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

    fn resolve_command(self, _context: &GatewayContext) -> DriverResult<Self::Resolved> {
        Ok(self)
    }

    async fn execute(
        resolved: Self::Resolved,
        context: &mut GatewayContext,
    ) -> DriverResult<CommandOutput> {
        match resolved {
            Self::Status => status(&context.state).await,
            Self::Listener {
                command: ListenerCommand::Status,
            } => listener_status(&context.state).await,
            Self::Config { command } => config_command(context, command).await,
            Self::State { command } => state_command(context, command).await,
            Self::Shutdown(args) => shutdown(context, args).await,
            Self::Telnyx { command } => telnyx_command(context, command).await,
            Self::Inbound { command } => inbound_command(context, command).await,
            Self::Calls => calls(&context.state).await,
            Self::Call { command } => call_command(context, command).await,
            Self::Answer(target) => context.answer_call(target.call).await,
            Self::Reject(target) => call_control(context, target.call, CallControlOp::Reject).await,
            Self::Hangup(target) => call_control(context, target.call, CallControlOp::Hangup).await,
            Self::Test { command } => test_command(context, command).await,
            Self::Transcript { command } => transcript_command(context, command).await,
            Self::Log { command } => log_command(context, command).await,
        }
    }
}

async fn status(state: &SharedState) -> DriverResult<CommandOutput> {
    let guard = state.read().await;
    let mut lines = vec![
        format!("listener: {:?}", guard.config.bind),
        format!("inbound: {}", guard.inbound_mode.label()),
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
    let (connection_id, from, stream_url, webhook_url, media) = {
        let guard = context.state.read().await;
        let connection_id = guard
            .config
            .selected_connection_id
            .clone()
            .ok_or_else(|| DriverError::message("telnyx app use <connection-id> first"))?;
        let from = args
            .from
            .clone()
            .or_else(|| guard.config.default_from_number.clone())
            .or_else(|| guard.config.selected_phone_number.clone())
            .ok_or_else(|| {
                DriverError::message("pass --from <e164> or config set from-number <e164> first")
            })?;
        let stream_url = guard
            .config
            .public_media_url
            .clone()
            .ok_or_else(|| DriverError::message("config set media-url <wss-url> first"))?;
        (
            connection_id,
            from,
            stream_url,
            guard.config.public_webhook_url.clone(),
            guard.config.telnyx_media,
        )
    };

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
        .map_err(driver_anyhow)?;

    let gateway_call_id = {
        let mut guard = context.state.write().await;
        let gateway_call_id = guard.add_or_update_outbound_call(
            crate::operator::state::TelnyxIds {
                call_control_id: dialed.call_control_id.clone(),
                call_session_id: dialed.call_session_id.clone(),
                call_leg_id: dialed.call_leg_id.clone(),
                stream_id: None,
            },
            Some(from.clone()),
            Some(args.to.clone()),
            CallStatus::Dialing,
        );
        guard.log(
            LogLevel::Info,
            format!(
                "dial-transcribe requested for {gateway_call_id} to {}",
                args.to
            ),
        );
        gateway_call_id
    };

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
                    "{} {} from={} to={} stream={}",
                    call.gateway_call_id,
                    call.status.label(),
                    call.from.as_deref().unwrap_or("<unknown>"),
                    call.to.as_deref().unwrap_or("<unknown>"),
                    call.ids.stream_id.as_deref().unwrap_or("<none>")
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
        CallCommand::Show { call } => call_show(&context.state, call).await,
        CallCommand::Use { call } => {
            let mut guard = context.state.write().await;
            if !guard.calls.contains_key(&call) {
                return Err(DriverError::NotFound {
                    kind: "call",
                    name: call,
                });
            }
            guard.selected_call = Some(call.clone());
            if let Some(session) = guard.calls.get_mut(&call) {
                session.unread_events = 0;
            }
            Ok(CommandOutput::line(format!("selected call {call}")))
        }
    }
}

async fn call_show(state: &SharedState, target: Option<String>) -> DriverResult<CommandOutput> {
    let guard = state.read().await;
    let id = resolve_call_id(&guard, target.as_deref())?;
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
    ];
    if let Some(reason) = &call.terminal_reason {
        lines.push(format!("ended: {reason}"));
    }
    lines.push("assembled transcript:".to_string());
    lines.push(assembled_transcript_text(call));
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

fn assembled_transcript_text(call: &crate::operator::state::CallSession) -> String {
    match (
        call.final_transcript.trim(),
        call.current_partial.as_deref().map(str::trim),
    ) {
        ("", Some(partial)) if !partial.is_empty() => partial.to_string(),
        (final_text, Some(partial)) if !final_text.is_empty() && !partial.is_empty() => {
            format!("{final_text} {partial}")
        }
        (final_text, _) if !final_text.is_empty() => final_text.to_string(),
        _ => "<none>".to_string(),
    }
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
        let call = resolve_call_mut(&mut guard, target.as_deref())?;
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
                if !guard.calls.contains_key(&call) {
                    return Err(DriverError::NotFound {
                        kind: "call",
                        name: call,
                    });
                }
                guard.selected_call = Some(call.clone());
            }
            Ok(CommandOutput::line(
                "selected-call detail follows transcript",
            ))
        }
        TranscriptCommand::Clear { call } => {
            let mut guard = context.state.write().await;
            let session = resolve_call_mut(&mut guard, call.as_deref())?;
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
) -> DriverResult<&'a mut crate::operator::state::CallSession> {
    let id = resolve_call_id(state, target)?;
    state
        .calls
        .get_mut(&id)
        .ok_or_else(|| DriverError::NotFound {
            kind: "call",
            name: id,
        })
}

fn resolve_call_id(state: &GatewayState, target: Option<&str>) -> DriverResult<String> {
    let id = match target {
        Some(value) => value.to_string(),
        None => state
            .selected_call
            .clone()
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

fn resolve_answer_call_id(state: &GatewayState, target: Option<&str>) -> DriverResult<String> {
    if let Some(value) = target {
        return Ok(value.to_string());
    }

    let mut waiting = state
        .calls
        .values()
        .filter(|call| call.status == CallStatus::PendingInbound)
        .map(|call| call.gateway_call_id.clone());
    let Some(first_waiting) = waiting.next() else {
        return resolve_call_id(state, None);
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
    use motlie_driver::CommandEngine;

    use super::*;
    use crate::call_control::TelnyxClient;
    use crate::operator::state::{CallStatus, TelnyxIds, shared_state};

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
    async fn answer_requires_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let gateway_call_id = guard.add_or_update_inbound_call(
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
            guard.selected_call = Some(gateway_call_id);
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("answer").await.expect("answer");

        assert!(output.lines[0].starts_with("answer requested for gwc_"));
        let guard = state.read().await;
        let call = guard.calls.values().next().expect("call exists");
        assert_eq!(call.status, CallStatus::Answering);
    }

    #[tokio::test]
    async fn answer_rejects_non_pending_call() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
            let mut guard = state.write().await;
            guard.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
            let gateway_call_id = guard.add_or_update_inbound_call(
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
            guard.selected_call = Some(gateway_call_id);
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
        let waiting_call_id = {
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
            guard.selected_call = Some(ended_call_id);
            waiting_call_id
        };
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("answer").await.expect("answer");

        assert_eq!(
            output.lines,
            vec![format!("answer requested for {waiting_call_id}")]
        );
        let guard = state.read().await;
        assert_eq!(
            guard.selected_call.as_deref(),
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

        let output = engine
            .run_line("test dial-transcribe +15550000002")
            .await
            .expect("dial-transcribe");

        assert!(output.lines[0].starts_with("dial-transcribe requested for gwc_"));
        let guard = state.read().await;
        let call_id = guard
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
    }

    #[tokio::test]
    async fn call_show_returns_assembled_transcript() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        {
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
            guard.selected_call = Some(gateway_call_id.clone());
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
        }
        let telnyx = TelnyxClient::new("https://api.telnyx.com/v2", None, true);
        let context = GatewayContext::new(state, telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = engine.run_line("call show").await.expect("call show");

        assert!(
            output
                .lines
                .iter()
                .any(|line| line == "assembled transcript:")
        );
        assert!(output.lines.iter().any(|line| line == "HELLO WORLD"));
        assert!(output.lines.iter().any(|line| line == "final: HELLO"));
        assert!(output.lines.iter().any(|line| line == "partial: WORLD"));
    }
}
