use std::collections::HashSet;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Result, bail};
use async_trait::async_trait;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};

use motlie_tmux::{
    CreateSessionOptions, CreateWindowOptions, HostHandle, KeySequence, ScrollbackQuery,
    SessionWatchHandle, SessionWatchOptions, SplitDirection, SplitPaneOptions, SplitSize,
    SshConfig, Target, TransferOptions,
};

use crate::completion::{CompletionCandidate, CompletionRequest};
use crate::engine::{CommandOutput, CommandSet};

pub struct TmuxState {
    pub host_uri: String,
    pub host: HostHandle,
    owned_sessions: HashSet<String>,
    known_sessions: Vec<String>,
    known_targets: Vec<String>,
    active_watch: Option<SessionWatchHandle>,
    active_stream: Option<ManagedStream>,
}

impl TmuxState {
    pub async fn connect(uri: &str) -> Result<Self> {
        let host = SshConfig::parse(uri)?.connect().await?;
        let mut state = Self::new(uri.to_string(), host);
        state.refresh_discovery().await?;
        Ok(state)
    }

    pub fn new(host_uri: impl Into<String>, host: HostHandle) -> Self {
        Self {
            host_uri: host_uri.into(),
            host,
            owned_sessions: HashSet::new(),
            known_sessions: Vec::new(),
            known_targets: Vec::new(),
            active_watch: None,
            active_stream: None,
        }
    }

    pub fn session_names(&self) -> impl Iterator<Item = &str> {
        self.known_sessions.iter().map(String::as_str)
    }

    pub fn target_names(&self) -> impl Iterator<Item = &str> {
        self.known_targets.iter().map(String::as_str)
    }

    pub async fn refresh_discovery(&mut self) -> Result<()> {
        let trees = self.host.snapshot_targets().await?;
        let mut sessions = Vec::new();
        let mut targets = Vec::new();

        for session in trees {
            sessions.push(session.info.name.clone());
            targets.push(session.info.name.clone());
            for window in session.windows {
                targets.push(window.target);
                for pane in window.panes {
                    targets.push(pane.target);
                }
            }
        }

        sessions.sort();
        sessions.dedup();
        targets.sort();
        targets.dedup();

        self.known_sessions = sessions;
        self.known_targets = targets;
        Ok(())
    }

    async fn clear_watch(&mut self) -> Result<()> {
        if let Some(watch) = self.active_watch.take() {
            let _ = watch.shutdown().await?;
        }
        Ok(())
    }

    fn clear_stream(&mut self) {
        self.active_stream = None;
    }

    fn mark_owned_session(&mut self, session: &str) {
        self.owned_sessions.insert(session.to_string());
    }

    fn unmark_owned_session(&mut self, session: &str) {
        self.owned_sessions.remove(session);
    }
}

pub struct ManagedStream {
    pub target: String,
    pub spec: StreamSpec,
}

#[derive(Debug, Default, Clone)]
pub struct TmuxCompletionContext {
    sessions: Vec<String>,
    targets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StreamSpec {
    pub mode: StreamModeArg,
    pub lines: usize,
    pub interval_ms: u64,
    pub pattern: Option<String>,
}

#[derive(Parser)]
struct TmuxRoot {
    #[command(subcommand)]
    command: TmuxCommand,
}

#[derive(Subcommand)]
pub enum TmuxCommand {
    Create(CreateCommand),
    NewWindow(NewWindowCommand),
    SplitPane(SplitPaneCommand),
    Kill(KillCommand),
    Targets(TargetsCommand),
    Send(SendCommand),
    Keys(KeysCommand),
    Capture(CaptureCommand),
    Monitor(MonitorCommand),
    History(HistoryCommand),
    Stream(StreamCommand),
    Upload(UploadCommand),
    Download(DownloadCommand),
}

#[derive(Args)]
pub struct CreateCommand {
    pub name: String,
    #[arg(long)]
    pub size: Option<SizeSpec>,
    #[arg(long)]
    pub history: Option<u32>,
}

#[derive(Args)]
pub struct NewWindowCommand {
    pub session: String,
    pub name: String,
    #[arg(long)]
    pub size: Option<SizeSpec>,
}

#[derive(Args)]
pub struct SplitPaneCommand {
    pub target: String,
    #[arg(long, conflicts_with = "vertical")]
    pub horizontal: bool,
    #[arg(long, conflicts_with = "horizontal")]
    pub vertical: bool,
    #[arg(long, conflicts_with = "cells")]
    pub percent: Option<u8>,
    #[arg(long, conflicts_with = "percent")]
    pub cells: Option<u16>,
}

#[derive(Args)]
pub struct KillCommand {
    pub target: String,
}

#[derive(Args)]
pub struct TargetsCommand;

#[derive(Args)]
pub struct SendCommand {
    pub target: String,
    #[arg(required = true, trailing_var_arg = true)]
    pub text: Vec<String>,
}

#[derive(Args)]
pub struct KeysCommand {
    pub target: String,
    #[arg(required = true, trailing_var_arg = true)]
    pub keys: Vec<String>,
}

#[derive(Args)]
pub struct CaptureCommand {
    pub target: String,
    pub lines: usize,
}

#[derive(Args)]
pub struct MonitorCommand {
    pub session: String,
    pub seconds: Option<u64>,
}

#[derive(Args)]
pub struct HistoryCommand {
    #[arg(required = true, num_args = 1..)]
    pub sessions: Vec<String>,
}

#[derive(Args)]
pub struct StreamCommand {
    pub target: String,
    #[arg(long, value_enum, default_value_t = StreamModeArg::Tail)]
    pub mode: StreamModeArg,
    #[arg(long, default_value_t = 50)]
    pub lines: usize,
    #[arg(long = "interval", default_value_t = 200)]
    pub interval_ms: u64,
    #[arg(long)]
    pub pattern: Option<String>,
}

#[derive(Args)]
pub struct UploadCommand {
    pub local_path: String,
    pub remote_path: String,
    #[arg(short = 'r', long = "recursive")]
    pub recursive: bool,
}

#[derive(Args)]
pub struct DownloadCommand {
    pub remote_path: String,
    pub local_path: String,
    #[arg(short = 'r', long = "recursive")]
    pub recursive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StreamModeArg {
    Visible,
    Tail,
    Until,
    Fidelity,
    Monitor,
    Render,
}

impl StreamModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::Tail => "tail",
            Self::Until => "until",
            Self::Fidelity => "fidelity",
            Self::Monitor => "monitor",
            Self::Render => "render",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeSpec {
    pub width: u16,
    pub height: u16,
}

impl FromStr for SizeSpec {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (width, height) = value
            .split_once('x')
            .ok_or_else(|| "size must be WxH".to_string())?;

        let width = width
            .parse::<u16>()
            .map_err(|_| "size width must be a positive integer".to_string())?;
        let height = height
            .parse::<u16>()
            .map_err(|_| "size height must be a positive integer".to_string())?;

        if width == 0 || height == 0 {
            return Err("size values must be > 0".to_string());
        }

        Ok(Self { width, height })
    }
}

#[async_trait]
impl CommandSet<TmuxState> for TmuxCommand {
    type CompletionContext = TmuxCompletionContext;

    fn root_command() -> clap::Command {
        TmuxRoot::command().name("tmux")
    }

    fn from_matches(matches: &clap::ArgMatches) -> Result<Self> {
        Ok(TmuxRoot::from_arg_matches(matches)?.command)
    }

    fn completion_context(context: &TmuxState) -> Self::CompletionContext {
        TmuxCompletionContext {
            sessions: context.known_sessions.clone(),
            targets: context.known_targets.clone(),
        }
    }

    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        match (request.command_path, request.arg_id) {
            (["new-window"], Some("session"))
            | (["monitor"], Some("session"))
            | (["history"], Some("sessions")) => context
                .sessions
                .iter()
                .map(String::as_str)
                .filter(|name| name.starts_with(request.prefix))
                .map(CompletionCandidate::new)
                .collect(),

            (["split-pane"], Some("target"))
            | (["kill"], Some("target"))
            | (["send"], Some("target"))
            | (["keys"], Some("target"))
            | (["capture"], Some("target"))
            | (["stream"], Some("target")) => context
                .targets
                .iter()
                .map(String::as_str)
                .filter(|name| name.starts_with(request.prefix))
                .map(CompletionCandidate::new)
                .collect(),

            (["stream"], Some("mode")) => StreamModeArg::value_variants()
                .iter()
                .filter_map(|mode| mode.to_possible_value())
                .map(|value| value.get_name().to_string())
                .filter(|name| name.starts_with(request.prefix))
                .map(CompletionCandidate::new)
                .collect(),

            _ => Vec::new(),
        }
    }

    async fn execute(self, context: &mut TmuxState) -> Result<CommandOutput> {
        match self {
            Self::Create(cmd) => execute_create(context, cmd).await,
            Self::NewWindow(cmd) => execute_new_window(context, cmd).await,
            Self::SplitPane(cmd) => execute_split_pane(context, cmd).await,
            Self::Kill(cmd) => execute_kill(context, cmd).await,
            Self::Targets(_) => execute_targets(context).await,
            Self::Send(cmd) => execute_send(context, cmd).await,
            Self::Keys(cmd) => execute_keys(context, cmd).await,
            Self::Capture(cmd) => execute_capture(context, cmd).await,
            Self::Monitor(cmd) => execute_monitor(context, cmd).await,
            Self::History(cmd) => execute_history(context, cmd).await,
            Self::Stream(cmd) => execute_stream(context, cmd).await,
            Self::Upload(cmd) => execute_upload(context, cmd).await,
            Self::Download(cmd) => execute_download(context, cmd).await,
        }
    }
}

async fn execute_create(context: &mut TmuxState, cmd: CreateCommand) -> Result<CommandOutput> {
    let mut opts = CreateSessionOptions::default();
    if let Some(size) = cmd.size {
        opts.width = Some(size.width);
        opts.height = Some(size.height);
    }
    opts.history_limit = cmd.history;

    context.host.create_session(&cmd.name, &opts).await?;
    context.mark_owned_session(&cmd.name);
    context.refresh_discovery().await?;

    let mut detail = String::new();
    if let Some(size) = cmd.size {
        detail.push_str(&format!(" ({}x{})", size.width, size.height));
    }
    if let Some(history) = cmd.history {
        detail.push_str(&format!(" history={history}"));
    }

    Ok(CommandOutput::line(format!(
        "Created: {}{}",
        cmd.name, detail
    )))
}

async fn execute_new_window(
    context: &mut TmuxState,
    cmd: NewWindowCommand,
) -> Result<CommandOutput> {
    let session = resolve_target(&context.host, &cmd.session).await?;
    let mut opts = CreateWindowOptions {
        name: Some(cmd.name),
        ..Default::default()
    };
    if let Some(size) = cmd.size {
        opts.width = Some(size.width);
        opts.height = Some(size.height);
    }

    let window = session.new_window(&opts).await?;
    context.refresh_discovery().await?;
    Ok(CommandOutput::line(format!(
        "Created window: {}",
        window.target_string()
    )))
}

async fn execute_split_pane(
    context: &mut TmuxState,
    cmd: SplitPaneCommand,
) -> Result<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let mut opts = SplitPaneOptions::default();

    if cmd.horizontal {
        opts.direction = SplitDirection::Horizontal;
    }
    if cmd.vertical {
        opts.direction = SplitDirection::Vertical;
    }
    if let Some(percent) = cmd.percent {
        opts.size = Some(SplitSize::percent(percent)?);
    }
    if let Some(cells) = cmd.cells {
        if cells == 0 {
            bail!("--cells must be a positive integer");
        }
        opts.size = Some(SplitSize::Cells(cells));
    }

    let pane = target.split_pane(&opts).await?;
    context.refresh_discovery().await?;
    Ok(CommandOutput::line(format!(
        "Created pane: {}",
        pane.target_string()
    )))
}

async fn execute_kill(context: &mut TmuxState, cmd: KillCommand) -> Result<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let session_name = target.session_name().to_string();
    target.kill().await?;
    context.unmark_owned_session(&session_name);
    context.refresh_discovery().await?;
    Ok(CommandOutput::line(format!("Killed: {}", cmd.target)))
}

async fn execute_targets(context: &mut TmuxState) -> Result<CommandOutput> {
    let trees = context.host.snapshot_targets().await?;
    context.refresh_discovery().await?;

    if trees.is_empty() {
        return Ok(CommandOutput::line("  (no sessions)"));
    }

    let mut lines = Vec::new();
    for session in trees {
        lines.push(format!(
            "  {:<20} (Session, {} window{})",
            session.info.name,
            session.windows.len(),
            if session.windows.len() == 1 { "" } else { "s" }
        ));
        for window in session.windows {
            lines.push(format!(
                "    {:<18} (Window, '{}', {} pane{})",
                window.target,
                window.info.name,
                window.panes.len(),
                if window.panes.len() == 1 { "" } else { "s" }
            ));
            for pane in window.panes {
                lines.push(format!(
                    "      {:<16} (Pane, {})",
                    pane.target, pane.address.pane_id
                ));
            }
        }
    }

    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn execute_send(context: &mut TmuxState, cmd: SendCommand) -> Result<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let text = cmd.text.join(" ");
    let enter = KeySequence::parse("{Enter}")?;
    target.send_text(&text).await?;
    target.send_keys(&enter).await?;
    Ok(CommandOutput::line(format!("Sent to {}", cmd.target)))
}

async fn execute_keys(context: &mut TmuxState, cmd: KeysCommand) -> Result<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let keys = KeySequence::parse(&cmd.keys.join(" "))?;
    target.send_keys(&keys).await?;
    Ok(CommandOutput::line(format!("Sent keys to {}", cmd.target)))
}

async fn execute_capture(context: &mut TmuxState, cmd: CaptureCommand) -> Result<CommandOutput> {
    if cmd.lines == 0 {
        bail!("<lines> must be a positive integer");
    }

    context.clear_stream();
    context.clear_watch().await?;

    let target = resolve_target(&context.host, &cmd.target).await?;
    let text = target
        .sample_text(&ScrollbackQuery::LastLines(cmd.lines))
        .await?;

    if text.is_empty() {
        Ok(CommandOutput::line("(empty)"))
    } else {
        Ok(CommandOutput::text(text))
    }
}

async fn execute_monitor(context: &mut TmuxState, cmd: MonitorCommand) -> Result<CommandOutput> {
    context.clear_stream();
    context.clear_watch().await?;

    let watch = context
        .host
        .watch_session(&cmd.session, &SessionWatchOptions::default())
        .await?;

    if let Some(seconds) = cmd.seconds {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
        let rendered = watch.render_text().await;
        let _ = watch.shutdown().await?;
        if rendered.trim().is_empty() {
            return Ok(CommandOutput::line("(empty)"));
        }
        return Ok(CommandOutput::text(rendered));
    }

    context.active_watch = Some(watch);
    Ok(CommandOutput::line(format!(
        "Watching session: {}",
        cmd.session
    )))
}

async fn execute_history(context: &mut TmuxState, cmd: HistoryCommand) -> Result<CommandOutput> {
    context.clear_stream();
    context.clear_watch().await?;

    let mut output = String::new();
    let mut captured = Vec::new();

    for name in &cmd.sessions {
        match context.host.session(name).await? {
            Some(target) => {
                let panes = target.capture_all().await?;
                let mut pane_list = panes.iter().collect::<Vec<_>>();
                pane_list.sort_by_key(|(address, _)| (address.window, address.pane));
                for (address, content) in pane_list {
                    if !motlie_tmux::has_visible_text(content) {
                        continue;
                    }
                    output.push_str(&format!("--- {}({}) ---\n", name, address.pane_id));
                    output.push_str(content);
                    if !content.ends_with('\n') {
                        output.push('\n');
                    }
                }
                captured.push(name.clone());
            }
            None => bail!("session '{}' not found", name),
        }
    }

    if output.is_empty() {
        return Ok(CommandOutput::line("(no visible content)"));
    }

    Ok(CommandOutput::text(output))
}

async fn execute_stream(context: &mut TmuxState, cmd: StreamCommand) -> Result<CommandOutput> {
    context.clear_watch().await?;
    context.clear_stream();

    let target = resolve_target(&context.host, &cmd.target).await?;

    if matches!(cmd.mode, StreamModeArg::Until) && cmd.pattern.is_none() {
        bail!("--pattern is required when --mode until");
    }

    context.active_stream = Some(ManagedStream {
        target: target.target_string(),
        spec: StreamSpec {
            mode: cmd.mode,
            lines: cmd.lines,
            interval_ms: cmd.interval_ms,
            pattern: cmd.pattern.clone(),
        },
    });

    Ok(CommandOutput::line(format!(
        "Streaming {} [mode={}, interval={}ms]",
        cmd.target,
        cmd.mode.as_str(),
        cmd.interval_ms
    )))
}

async fn execute_upload(context: &mut TmuxState, cmd: UploadCommand) -> Result<CommandOutput> {
    let opts = TransferOptions {
        overwrite: true,
        recursive: cmd.recursive,
    };
    context
        .host
        .upload(cmd.local_path.as_ref(), cmd.remote_path.as_ref(), &opts)
        .await?;
    Ok(CommandOutput::line(format!(
        "Uploaded {} -> {}",
        cmd.local_path, cmd.remote_path
    )))
}

async fn execute_download(context: &mut TmuxState, cmd: DownloadCommand) -> Result<CommandOutput> {
    let opts = TransferOptions {
        overwrite: true,
        recursive: cmd.recursive,
    };
    context
        .host
        .download(cmd.remote_path.as_ref(), cmd.local_path.as_ref(), &opts)
        .await?;
    Ok(CommandOutput::line(format!(
        "Downloaded {} -> {}",
        cmd.remote_path, cmd.local_path
    )))
}

async fn resolve_target(host: &HostHandle, target_str: &str) -> Result<Target> {
    host.resolve_target_str(target_str)
        .await?
        .ok_or_else(|| anyhow::anyhow!("target '{}' not found", target_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CommandEngine;

    #[test]
    fn tmux_root_exposes_expected_subcommands() {
        let root = TmuxCommand::root_command();
        let names = root
            .get_subcommands()
            .map(|subcommand| subcommand.get_name().to_string())
            .collect::<Vec<_>>();

        assert!(names.contains(&"create".to_string()));
        assert!(names.contains(&"stream".to_string()));
        assert!(names.contains(&"download".to_string()));
    }

    #[test]
    fn tmux_completion_suggests_targets() {
        let state = TmuxState {
            host_uri: "ssh://localhost".to_string(),
            host: HostHandle::local(),
            owned_sessions: HashSet::new(),
            known_sessions: vec!["demo".to_string()],
            known_targets: vec![
                "demo".to_string(),
                "demo:0".to_string(),
                "demo:0.0".to_string(),
            ],
            active_watch: None,
            active_stream: None,
        };

        let engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let completions = engine.complete("kill demo:", "kill demo:".len());
        let values = completions
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();

        assert!(values.contains(&"demo:0".to_string()));
        assert!(values.contains(&"demo:0.0".to_string()));
    }

    #[test]
    fn tmux_completion_suggests_subcommands() {
        let state = TmuxState {
            host_uri: "ssh://localhost".to_string(),
            host: HostHandle::local(),
            owned_sessions: HashSet::new(),
            known_sessions: Vec::new(),
            known_targets: Vec::new(),
            active_watch: None,
            active_stream: None,
        };

        let engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let completions = engine.complete("cr", 2);
        let values = completions
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();

        assert!(values.contains(&"create".to_string()));
    }

    #[tokio::test]
    async fn tmux_help_uses_composed_command_tree() {
        let state = TmuxState {
            host_uri: "ssh://localhost".to_string(),
            host: HostHandle::local(),
            owned_sessions: HashSet::new(),
            known_sessions: Vec::new(),
            known_targets: Vec::new(),
            active_watch: None,
            active_stream: None,
        };

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine.run_line("help create").await.expect("help output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("create"));
        assert!(rendered.contains("--size"));
    }
}
