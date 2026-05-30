use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use regex::Regex;

use motlie_tmux::{
    has_visible_text, overlap_deduplicate, CaptureNormalizeMode, CaptureOptions,
    CreateSessionOptions, CreateWindowOptions, FidelityIssue, HostHandle, KeySequence,
    MonitorHealth, PaneAddress, ScrollbackQuery, SessionWatchHandle, SessionWatchOptions,
    SplitDirection, SplitPaneOptions, SplitSize, SshConfig, Target, TransferOptions,
};

use crate::completion::{CompletionCandidate, CompletionRequest};
use crate::engine::{CommandEffect, CommandOutput, CommandSet};
use crate::error::{DriverError, DriverResult};
use crate::history::{HistoryBuffer, HistoryPage};

const TMUX_HISTORY_CAPACITY: usize = 256;

pub struct TmuxState {
    pub host_uri: String,
    pub host: HostHandle,
    owned_sessions: HashSet<String>,
    pub(crate) known_sessions: Vec<String>,
    pub(crate) known_targets: Vec<String>,
    active_watch: Option<SessionWatchHandle>,
    active_stream: Option<ManagedStream>,
    mirror_text: String,
    mirror_label: String,
    mirror_ansi: bool,
    mirror_auto_refresh: bool,
    mirror_history: HistoryBuffer<TmuxHistoryEntry>,
}

impl TmuxState {
    pub async fn connect(uri: &str) -> DriverResult<Self> {
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
            mirror_text: String::new(),
            mirror_label: String::new(),
            mirror_ansi: false,
            mirror_auto_refresh: false,
            mirror_history: HistoryBuffer::new(tmux_history_capacity()),
        }
    }

    pub fn session_names(&self) -> impl Iterator<Item = &str> {
        self.known_sessions.iter().map(String::as_str)
    }

    pub fn target_names(&self) -> impl Iterator<Item = &str> {
        self.known_targets.iter().map(String::as_str)
    }

    pub async fn refresh_discovery(&mut self) -> DriverResult<()> {
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

    async fn clear_watch(&mut self) -> DriverResult<()> {
        if let Some(watch) = self.active_watch.take() {
            watch.shutdown().await?;
        }
        Ok(())
    }

    fn clear_stream(&mut self) {
        self.active_stream = None;
    }

    pub async fn refresh_mirror(&mut self) -> DriverResult<()> {
        if self.active_stream.is_some() {
            self.tick_stream().await?;
        } else if self.mirror_auto_refresh {
            if let Some(ref watch) = self.active_watch {
                self.mirror_text = watch.render_text().await.replace('\r', "");
                self.record_current_mirror_state();
            }
        }
        Ok(())
    }

    pub fn mirror_snapshot(&self) -> TmuxMirrorSnapshot {
        TmuxMirrorSnapshot {
            text: self.mirror_text.clone(),
            label: self.mirror_label.clone(),
            ansi: self.mirror_ansi,
            watch_health: self.active_watch.as_ref().map(SessionWatchHandle::health),
        }
    }

    pub fn mirror_history_page(
        &self,
        after: Option<u64>,
        limit: usize,
    ) -> HistoryPage<TmuxHistoryEntry> {
        self.mirror_history.page_after(after, limit)
    }

    pub fn has_live_follow(&self) -> bool {
        self.active_watch.is_some() || self.active_stream.is_some()
    }

    pub async fn shutdown_managed_state(&mut self) -> DriverResult<()> {
        self.clear_stream();
        self.clear_watch().await
    }

    fn mark_owned_session(&mut self, session: &str) {
        self.owned_sessions.insert(session.to_string());
    }

    fn unmark_owned_session(&mut self, session: &str) {
        self.owned_sessions.remove(session);
    }

    async fn tick_stream(&mut self) -> DriverResult<()> {
        let mut stream = match self.active_stream.take() {
            Some(stream) => stream,
            None => return Ok(()),
        };

        if stream.last_tick.elapsed() < stream.interval {
            self.active_stream = Some(stream);
            return Ok(());
        }
        stream.last_tick = tokio::time::Instant::now();

        let target_name = stream.target.clone();
        let Some(target) = self.host.resolve_target_str(&target_name).await? else {
            self.note_stream_error(
                &mut stream,
                DriverError::NotFound {
                    kind: "tmux target",
                    name: target_name,
                },
            );
            self.active_stream = Some(stream);
            return Ok(());
        };

        match stream.spec.mode {
            StreamModeArg::Visible => {
                let opts = CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable);
                match target.capture_with_options(&opts).await {
                    Ok(result) => {
                        self.clear_stream_error(&mut stream);
                        if result.text != stream.previous {
                            self.mirror_text = result.text.clone();
                            stream.previous = result.text;
                            self.record_current_mirror_state();
                        }
                    }
                    Err(error) => self.note_stream_error(&mut stream, error.into()),
                }
            }
            StreamModeArg::Tail => {
                let query = ScrollbackQuery::LastLines(stream.spec.lines);
                match target.sample_text(&query).await {
                    Ok(current) => {
                        self.clear_stream_error(&mut stream);
                        if !current.is_empty() && current != stream.previous {
                            let (merged, _) = overlap_deduplicate(&stream.previous, &current, 5);
                            if merged != stream.previous {
                                if merged.starts_with(&stream.previous) {
                                    self.mirror_text.push_str(&merged[stream.previous.len()..]);
                                } else {
                                    self.mirror_text.push_str(&current);
                                    if !current.ends_with('\n') {
                                        self.mirror_text.push('\n');
                                    }
                                }
                            }
                            stream.previous = merged;
                        }
                        trim_mirror_text(&mut self.mirror_text);
                        self.record_current_mirror_state();
                    }
                    Err(error) => self.note_stream_error(&mut stream, error.into()),
                }
            }
            StreamModeArg::Until => {
                if let Some(pattern) = stream.pattern.as_ref() {
                    let query = ScrollbackQuery::Until {
                        pattern: pattern.clone(),
                        max_lines: stream.spec.lines,
                    };
                    match target.sample_text(&query).await {
                        Ok(content) => {
                            self.clear_stream_error(&mut stream);
                            if content != stream.previous {
                                self.mirror_text = content.clone();
                                stream.previous = content;
                                self.record_current_mirror_state();
                            }
                        }
                        Err(error) => self.note_stream_error(&mut stream, error.into()),
                    }
                }
            }
            StreamModeArg::Fidelity => {
                let opts = CaptureOptions {
                    detect_reflow: true,
                    normalize: CaptureNormalizeMode::PlainText,
                    ..Default::default()
                };
                match target.capture_with_options(&opts).await {
                    Ok(result) => {
                        self.clear_stream_error(&mut stream);
                        if result.text != stream.previous {
                            let mut text = result.text.clone();
                            if result.fidelity.degraded {
                                if let Some(ref issues) = result.fidelity.issues {
                                    let names = issues
                                        .iter()
                                        .map(|issue| match issue {
                                            FidelityIssue::ClientResize => "ClientResize",
                                            FidelityIssue::PaneResize => "PaneResize",
                                            FidelityIssue::HistoryTruncated => "HistoryTruncated",
                                            FidelityIssue::OverlapResync => "OverlapResync",
                                        })
                                        .collect::<Vec<_>>();
                                    text.push_str(&format!("\n[DEGRADED: {}]", names.join(", ")));
                                }
                            } else {
                                text.push_str("\n[FIDELITY: CLEAN]");
                            }
                            self.mirror_text = text;
                            stream.previous = result.text;
                            self.record_current_mirror_state();
                        }
                    }
                    Err(error) => self.note_stream_error(&mut stream, error.into()),
                }
            }
            StreamModeArg::Render => match target.capture_all().await {
                Ok(panes) => {
                    self.clear_stream_error(&mut stream);
                    let rendered = render_pane_snapshot(target.session_name(), &panes);
                    if rendered != stream.previous {
                        self.mirror_text = rendered.clone();
                        stream.previous = rendered;
                        self.record_current_mirror_state();
                    }
                }
                Err(error) => self.note_stream_error(&mut stream, error.into()),
            },
            StreamModeArg::Monitor => {}
        }

        self.active_stream = Some(stream);
        Ok(())
    }

    fn current_mirror_entry(&self) -> TmuxHistoryEntry {
        TmuxHistoryEntry {
            text: self.mirror_text.clone(),
            label: self.mirror_label.clone(),
            ansi: self.mirror_ansi,
            watch_health: self.active_watch.as_ref().map(SessionWatchHandle::health),
        }
    }

    fn record_current_mirror_state(&mut self) {
        let entry = self.current_mirror_entry();
        let should_push = self
            .mirror_history
            .latest()
            .map(|record| record.item != entry)
            .unwrap_or(true);

        if should_push {
            let _ = self.mirror_history.push(entry);
        }
    }

    fn clear_stream_error(&mut self, stream: &mut ManagedStream) {
        if stream.consecutive_failures == 0 {
            return;
        }

        stream.consecutive_failures = 0;
        stream.last_error = None;
        self.mirror_label = stream.base_label();
        self.record_current_mirror_state();
    }

    fn note_stream_error(&mut self, stream: &mut ManagedStream, error: DriverError) {
        stream.consecutive_failures += 1;
        let reason = error.to_string();
        let changed = stream.last_error.as_deref() != Some(reason.as_str());
        stream.last_error = Some(reason.clone());
        self.mirror_label = format!(
            "{} [error x{}: {}]",
            stream.base_label(),
            stream.consecutive_failures,
            reason
        );
        if changed || stream.consecutive_failures == 1 {
            self.record_current_mirror_state();
        }
    }
}

pub struct ManagedStream {
    pub target: String,
    pub spec: StreamSpec,
    pattern: Option<Regex>,
    previous: String,
    interval: Duration,
    last_tick: tokio::time::Instant,
    consecutive_failures: usize,
    last_error: Option<String>,
}

impl ManagedStream {
    fn base_label(&self) -> String {
        format!("stream: {} [{}]", self.target, self.spec.mode.as_str())
    }
}

#[derive(Debug, Default, Clone)]
pub struct TmuxCompletionContext {
    pub sessions: Vec<String>,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TmuxMirrorSnapshot {
    pub text: String,
    pub label: String,
    pub ansi: bool,
    pub watch_health: Option<MonitorHealth>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmuxHistoryEntry {
    pub text: String,
    pub label: String,
    pub ansi: bool,
    pub watch_health: Option<MonitorHealth>,
}

#[async_trait]
pub trait TmuxFrontendState: Send {
    fn frontend_host_uri(&self) -> String;
    fn mirror_snapshot(&self) -> TmuxMirrorSnapshot;
    fn mirror_history_page(
        &self,
        after: Option<u64>,
        limit: usize,
    ) -> HistoryPage<TmuxHistoryEntry>;
    fn has_live_follow(&self) -> bool;
    async fn refresh_mirror(&mut self) -> DriverResult<()>;
    async fn shutdown_managed_state(&mut self) -> DriverResult<()>;
}

#[async_trait]
impl TmuxFrontendState for TmuxState {
    fn frontend_host_uri(&self) -> String {
        self.host_uri.clone()
    }

    fn mirror_snapshot(&self) -> TmuxMirrorSnapshot {
        TmuxState::mirror_snapshot(self)
    }

    fn mirror_history_page(
        &self,
        after: Option<u64>,
        limit: usize,
    ) -> HistoryPage<TmuxHistoryEntry> {
        TmuxState::mirror_history_page(self, after, limit)
    }

    fn has_live_follow(&self) -> bool {
        TmuxState::has_live_follow(self)
    }

    async fn refresh_mirror(&mut self) -> DriverResult<()> {
        TmuxState::refresh_mirror(self).await
    }

    async fn shutdown_managed_state(&mut self) -> DriverResult<()> {
        TmuxState::shutdown_managed_state(self).await
    }
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
    Mirror(MirrorCommand),
    Tui(TuiCommand),
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
pub struct MirrorCommand {
    #[command(subcommand)]
    pub action: MirrorActionCommand,
}

#[derive(Subcommand)]
pub enum MirrorActionCommand {
    History(MirrorHistoryCommand),
    Clear,
}

#[derive(Args)]
pub struct MirrorHistoryCommand {
    #[arg(long)]
    pub after: Option<u64>,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
}

#[derive(Args)]
pub struct TuiCommand {
    #[command(subcommand)]
    pub action: TuiActionCommand,
}

#[derive(Subcommand)]
pub enum TuiActionCommand {
    On,
    Off,
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
    #[command(subcommand)]
    pub action: MonitorActionCommand,
}

#[derive(Subcommand)]
pub enum MonitorActionCommand {
    Start(MonitorStartCommand),
    Stop,
}

#[derive(Args)]
pub struct MonitorStartCommand {
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
    type Resolved = Self;

    fn root_command() -> clap::Command {
        TmuxRoot::command().name("tmux")
    }

    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self> {
        Ok(TmuxRoot::from_arg_matches(matches)?.command)
    }

    fn completion_context(context: &TmuxState) -> Self::CompletionContext {
        TmuxCompletionContext {
            sessions: context.known_sessions.clone(),
            targets: context.known_targets.clone(),
        }
    }

    fn help(topic: &[String]) -> Option<String> {
        tmux_help(topic)
    }

    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        tmux_complete(request, context)
    }

    fn resolve_command(self, _context: &TmuxState) -> DriverResult<Self::Resolved> {
        Ok(self)
    }

    async fn execute(
        resolved: Self::Resolved,
        context: &mut TmuxState,
    ) -> DriverResult<CommandOutput> {
        execute_tmux_command(context, resolved).await
    }
}

pub(crate) fn tmux_help(topic: &[String]) -> Option<String> {
    Some(match topic {
        [] => tmux_root_help(),
        [topic] if topic == "stream" => tmux_stream_help(),
        [topic] if topic == "monitor" => tmux_monitor_help(),
        [topic] if topic == "mirror" => tmux_mirror_help(),
        [topic] if topic == "capture" => tmux_capture_help(),
        [topic] if topic == "history" => tmux_history_help(),
        [topic] if topic == "tui" => tmux_tui_help(),
        _ => return None,
    })
}

pub(crate) fn tmux_complete(
    request: CompletionRequest<'_>,
    context: &TmuxCompletionContext,
) -> Vec<CompletionCandidate> {
    match (request.command_path, request.arg_id) {
        (["new-window"], Some("session"))
        | (["monitor", "start"], Some("session"))
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

pub(crate) async fn execute_tmux_command(
    context: &mut TmuxState,
    command: TmuxCommand,
) -> DriverResult<CommandOutput> {
    match command {
        TmuxCommand::Create(cmd) => execute_create(context, cmd).await,
        TmuxCommand::NewWindow(cmd) => execute_new_window(context, cmd).await,
        TmuxCommand::SplitPane(cmd) => execute_split_pane(context, cmd).await,
        TmuxCommand::Kill(cmd) => execute_kill(context, cmd).await,
        TmuxCommand::Mirror(cmd) => execute_mirror(context, cmd),
        TmuxCommand::Tui(cmd) => execute_tui(cmd),
        TmuxCommand::Targets(_) => execute_targets(context).await,
        TmuxCommand::Send(cmd) => execute_send(context, cmd).await,
        TmuxCommand::Keys(cmd) => execute_keys(context, cmd).await,
        TmuxCommand::Capture(cmd) => execute_capture(context, cmd).await,
        TmuxCommand::Monitor(cmd) => execute_monitor(context, cmd).await,
        TmuxCommand::History(cmd) => execute_history(context, cmd).await,
        TmuxCommand::Stream(cmd) => execute_stream(context, cmd).await,
        TmuxCommand::Upload(cmd) => execute_upload(context, cmd).await,
        TmuxCommand::Download(cmd) => execute_download(context, cmd).await,
    }
}

async fn execute_create(
    context: &mut TmuxState,
    cmd: CreateCommand,
) -> DriverResult<CommandOutput> {
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
) -> DriverResult<CommandOutput> {
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
) -> DriverResult<CommandOutput> {
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
            return Err(DriverError::invalid_argument(
                "cells",
                "must be a positive integer",
            ));
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

async fn execute_kill(context: &mut TmuxState, cmd: KillCommand) -> DriverResult<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let session_name = target.session_name().to_string();
    target.kill().await?;
    context.unmark_owned_session(&session_name);
    context.refresh_discovery().await?;
    Ok(CommandOutput::line(format!("Killed: {}", cmd.target)))
}

fn execute_tui(cmd: TuiCommand) -> DriverResult<CommandOutput> {
    let output = match cmd.action {
        TuiActionCommand::On => {
            CommandOutput::line("Entering TUI mode").with_effect(CommandEffect::EnterTui)
        }
        TuiActionCommand::Off => {
            CommandOutput::line("Leaving TUI mode").with_effect(CommandEffect::ExitTui)
        }
    };
    Ok(output)
}

fn execute_mirror(context: &mut TmuxState, cmd: MirrorCommand) -> DriverResult<CommandOutput> {
    match cmd.action {
        MirrorActionCommand::History(cmd) => execute_mirror_history(context, cmd),
        MirrorActionCommand::Clear => {
            context.mirror_history.clear();
            Ok(CommandOutput::line("Cleared local mirror history"))
        }
    }
}

fn execute_mirror_history(
    context: &mut TmuxState,
    cmd: MirrorHistoryCommand,
) -> DriverResult<CommandOutput> {
    if cmd.limit == 0 {
        return Err(DriverError::invalid_argument(
            "limit",
            "must be a positive integer",
        ));
    }

    let page = context.mirror_history_page(cmd.after, cmd.limit);
    if page.items.is_empty() {
        return Ok(CommandOutput::line("No retained mirror history"));
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "mirror history: {} item(s) oldest={} newest={} next-after={}",
        page.items.len(),
        page.oldest_available
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string()),
        page.newest_available
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string()),
        page.next_after
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    ));

    for record in page.items {
        lines.push(format!(
            "[seq={} label={} ansi={} watch={}]",
            record.seq,
            if record.item.label.is_empty() {
                "-".to_string()
            } else {
                record.item.label.clone()
            },
            record.item.ansi,
            match record.item.watch_health {
                Some(health) => format!("{health:?}"),
                None => "idle".to_string(),
            }
        ));

        if record.item.text.is_empty() {
            lines.push("  (empty)".to_string());
            continue;
        }

        for line in record.item.text.lines() {
            lines.push(format!("  {line}"));
        }
    }

    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn execute_targets(context: &mut TmuxState) -> DriverResult<CommandOutput> {
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

async fn execute_send(context: &mut TmuxState, cmd: SendCommand) -> DriverResult<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let text = cmd.text.join(" ");
    let enter = KeySequence::parse("{Enter}")?;
    target.send_text(&text).await?;
    target.send_keys(&enter).await?;
    Ok(CommandOutput::line(format!("Sent to {}", cmd.target)))
}

async fn execute_keys(context: &mut TmuxState, cmd: KeysCommand) -> DriverResult<CommandOutput> {
    let target = resolve_target(&context.host, &cmd.target).await?;
    let keys = KeySequence::parse(&cmd.keys.join(" "))?;
    target.send_keys(&keys).await?;
    Ok(CommandOutput::line(format!("Sent keys to {}", cmd.target)))
}

async fn execute_capture(
    context: &mut TmuxState,
    cmd: CaptureCommand,
) -> DriverResult<CommandOutput> {
    if cmd.lines == 0 {
        return Err(DriverError::invalid_argument(
            "lines",
            "must be a positive integer",
        ));
    }

    context.clear_stream();
    context.clear_watch().await?;

    let target = resolve_target(&context.host, &cmd.target).await?;
    let text = target
        .sample_text(&ScrollbackQuery::LastLines(cmd.lines))
        .await?;

    context.mirror_text = if text.is_empty() {
        "(empty)".to_string()
    } else {
        text.clone()
    };
    context.mirror_label = format!("capture: {} (last {})", cmd.target, cmd.lines);
    context.mirror_ansi = false;
    context.mirror_auto_refresh = false;
    context.record_current_mirror_state();

    if text.is_empty() {
        Ok(CommandOutput::line("(empty)"))
    } else {
        Ok(CommandOutput::text(text))
    }
}

async fn execute_monitor(
    context: &mut TmuxState,
    cmd: MonitorCommand,
) -> DriverResult<CommandOutput> {
    match cmd.action {
        MonitorActionCommand::Start(cmd) => execute_monitor_start(context, cmd).await,
        MonitorActionCommand::Stop => execute_monitor_stop(context).await,
    }
}

async fn execute_monitor_start(
    context: &mut TmuxState,
    cmd: MonitorStartCommand,
) -> DriverResult<CommandOutput> {
    context.clear_stream();
    context.clear_watch().await?;

    let watch = context
        .host
        .watch_session(&cmd.session, &SessionWatchOptions::default())
        .await?;

    if let Some(seconds) = cmd.seconds {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
        let rendered = watch.render_text().await;
        watch.shutdown().await?;
        context.mirror_text = if rendered.trim().is_empty() {
            "(empty)".to_string()
        } else {
            rendered.clone()
        };
        context.mirror_label = format!("watching: {}", cmd.session);
        context.mirror_ansi = false;
        context.mirror_auto_refresh = false;
        context.record_current_mirror_state();
        if rendered.trim().is_empty() {
            return Ok(CommandOutput::line("(empty)"));
        }
        return Ok(CommandOutput::text(rendered));
    }

    context.active_watch = Some(watch);
    context.mirror_text.clear();
    context.mirror_label = format!("watching: {}", cmd.session);
    context.mirror_ansi = false;
    context.mirror_auto_refresh = true;
    context.record_current_mirror_state();
    Ok(CommandOutput::line(format!(
        "Watching session: {} (attached; Ctrl-C to stop live follow)",
        cmd.session
    )))
}

async fn execute_monitor_stop(context: &mut TmuxState) -> DriverResult<CommandOutput> {
    let had_live_follow = context.has_live_follow();
    context.shutdown_managed_state().await?;
    if had_live_follow {
        context.mirror_label = "monitor: stopped".to_string();
        context.mirror_auto_refresh = false;
        context.record_current_mirror_state();
        return Ok(CommandOutput::line("Stopped active monitor/stream"));
    }
    Ok(CommandOutput::line("No active monitor/stream"))
}

async fn execute_history(
    context: &mut TmuxState,
    cmd: HistoryCommand,
) -> DriverResult<CommandOutput> {
    context.clear_stream();
    context.clear_watch().await?;

    let mut output = String::new();
    let mut captured = Vec::new();

    for name in &cmd.sessions {
        match context.host.session(name).await? {
            Some(target) => {
                let panes = target.capture_all().await?;
                output.push_str(&render_pane_snapshot(name, &panes));
                captured.push(name.clone());
            }
            None => {
                return Err(DriverError::NotFound {
                    kind: "tmux session",
                    name: name.clone(),
                });
            }
        }
    }

    if output.is_empty() {
        context.mirror_text = "(no visible content)".to_string();
        context.mirror_label = format!("history: {}", cmd.sessions.join(", "));
        context.mirror_ansi = false;
        context.mirror_auto_refresh = false;
        context.record_current_mirror_state();
        return Ok(CommandOutput::line("(no visible content)"));
    }

    context.mirror_text = output.clone();
    context.mirror_label = format!("history: {}", cmd.sessions.join(", "));
    context.mirror_ansi = false;
    context.mirror_auto_refresh = false;
    context.record_current_mirror_state();
    Ok(CommandOutput::text(output))
}

async fn execute_stream(
    context: &mut TmuxState,
    cmd: StreamCommand,
) -> DriverResult<CommandOutput> {
    context.clear_watch().await?;
    context.clear_stream();

    let target = resolve_target(&context.host, &cmd.target).await?;

    if matches!(cmd.mode, StreamModeArg::Until) && cmd.pattern.is_none() {
        return Err(DriverError::invalid_argument(
            "pattern",
            "is required when --mode until",
        ));
    }

    if matches!(cmd.mode, StreamModeArg::Monitor) {
        let session_name = target.session_name().to_string();
        let watch = context
            .host
            .watch_session(&session_name, &SessionWatchOptions::default())
            .await?;
        context.active_watch = Some(watch);
        context.mirror_text.clear();
        context.mirror_label = format!("stream: {} [{}]", cmd.target, cmd.mode.as_str());
        context.mirror_ansi = false;
        context.mirror_auto_refresh = true;
        context.record_current_mirror_state();
        return Ok(CommandOutput::line(format!(
            "Streaming {} [mode={}, event-driven]",
            cmd.target,
            cmd.mode.as_str()
        )));
    }

    let initial = if matches!(cmd.mode, StreamModeArg::Tail) {
        target
            .sample_text(&ScrollbackQuery::LastLines(cmd.lines))
            .await?
    } else {
        String::new()
    };
    let pattern = match (&cmd.mode, cmd.pattern.as_deref()) {
        (StreamModeArg::Until, Some(pattern)) => Some(Regex::new(pattern)?),
        _ => None,
    };

    context.mirror_text = initial.clone();
    context.mirror_label = format!("stream: {} [{}]", cmd.target, cmd.mode.as_str());
    context.mirror_ansi = matches!(cmd.mode, StreamModeArg::Visible);
    context.mirror_auto_refresh = false;
    context.record_current_mirror_state();
    context.active_stream = Some(ManagedStream {
        target: target.target_string(),
        spec: StreamSpec {
            mode: cmd.mode,
            lines: cmd.lines,
            interval_ms: cmd.interval_ms,
            pattern: cmd.pattern.clone(),
        },
        pattern,
        previous: initial,
        interval: Duration::from_millis(cmd.interval_ms),
        last_tick: tokio::time::Instant::now() - Duration::from_millis(cmd.interval_ms),
        consecutive_failures: 0,
        last_error: None,
    });

    Ok(CommandOutput::line(format!(
        "Streaming {} [mode={}, interval={}ms]",
        cmd.target,
        cmd.mode.as_str(),
        cmd.interval_ms
    )))
}

async fn execute_upload(
    context: &mut TmuxState,
    cmd: UploadCommand,
) -> DriverResult<CommandOutput> {
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

async fn execute_download(
    context: &mut TmuxState,
    cmd: DownloadCommand,
) -> DriverResult<CommandOutput> {
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

async fn resolve_target(host: &HostHandle, target_str: &str) -> DriverResult<Target> {
    host.resolve_target_str(target_str)
        .await?
        .ok_or_else(|| DriverError::NotFound {
            kind: "tmux target",
            name: target_str.to_string(),
        })
}

fn tmux_history_capacity() -> NonZeroUsize {
    NonZeroUsize::new(TMUX_HISTORY_CAPACITY).unwrap_or(NonZeroUsize::MIN)
}

fn render_pane_snapshot(
    session_name: &str,
    panes: &std::collections::HashMap<PaneAddress, String>,
) -> String {
    let mut pane_list = panes.iter().collect::<Vec<_>>();
    pane_list.sort_by_key(|(address, _)| (address.window, address.pane));

    let mut rendered = String::new();
    for (address, content) in pane_list {
        if !has_visible_text(content) {
            continue;
        }
        rendered.push_str(&format!("--- {}({}) ---\n", session_name, address.pane_id));
        rendered.push_str(content);
        if !content.ends_with('\n') {
            rendered.push('\n');
        }
    }

    rendered
}

fn trim_mirror_text(text: &mut String) {
    const MAX_MIRROR: usize = 100_000;
    if text.len() <= MAX_MIRROR {
        return;
    }

    let mut trim = text.len() - MAX_MIRROR * 3 / 4;
    while trim < text.len() && !text.is_char_boundary(trim) {
        trim += 1;
    }
    let boundary = text[trim..]
        .find('\n')
        .map(|offset| trim + offset + 1)
        .unwrap_or(trim);
    *text = text[boundary..].to_string();
}

fn tmux_root_help() -> String {
    [
        "tmux driver commands",
        "",
        "General:",
        "  help [command]                        Show help or detailed topic help",
        "  tui on                               Enter split-screen TUI mode",
        "  tui off                              Leave split-screen TUI mode",
        "  quit                                 Exit the shell",
        "",
        "Session / target management:",
        "  create <name> [--size WxH] [--history N]",
        "  new-window <session> <name> [--size WxH]",
        "  split-pane <target> [--horizontal|--vertical] [--percent N|--cells N]",
        "  kill <target>",
        "  targets",
        "",
        "Input / output:",
        "  send <target> <text...>",
        "  keys <target> <keys...>",
        "  capture <target> <n>",
        "  monitor start <session> [seconds]",
        "  monitor stop",
        "  mirror history [--after N] [--limit N]",
        "  mirror clear",
        "  history <session> [session...]",
        "  stream <target> [--mode MODE] [--lines N] [--interval MS] [--pattern REGEX]",
        "",
        "File transfer:",
        "  upload <local_path> <remote_path> [--recursive]",
        "  download <remote_path> <local_path> [--recursive]",
        "",
        "Targets:",
        "  session              first pane of first window",
        "  session:window       first pane of a window",
        "  session:window.pane  specific pane",
        "",
        "Helpful topics:",
        "  help stream",
        "  help monitor",
        "  help mirror",
        "  help capture",
        "  help history",
        "  help tui",
    ]
    .join("\n")
}

fn tmux_stream_help() -> String {
    [
        "stream <target> [OPTIONS]",
        "",
        "Continuously refresh the mirror pane from a target.",
        "",
        "Options:",
        "  --mode MODE       visible | tail | until | fidelity | monitor | render",
        "  --lines N         scrollback line count for tail/until [default: 50]",
        "  --interval MS     polling interval for polling modes [default: 200]",
        "  --pattern REGEX   required for --mode until",
        "",
        "Modes:",
        "  tail      Append new scrollback like tail -f. Best for shells and logs.",
        "  visible   Re-capture the visible pane. Best for a single TUI pane.",
        "  until     Re-capture from the last line matching REGEX to the end.",
        "  fidelity  Plain-text capture with reflow/fidelity annotations.",
        "  render    Render all panes in the target session with pane headers.",
        "  monitor   Event-driven session stream using tmux control mode.",
        "",
        "Mode guidance:",
        "  tail / monitor    Best for shell commands and log output.",
        "  visible / render  Best for TUIs like vim, top, htop.",
        "  fidelity          Best when you need to see resize/reflow degradation.",
        "",
        "Examples:",
        "  stream demo",
        "  stream demo --mode visible",
        "  stream demo --mode render",
        "  stream demo --mode tail --lines 100 --interval 500",
        "  stream demo --mode until --pattern '^\\$ '",
        "  stream demo:0.1 --mode fidelity",
        "  stream demo --mode monitor",
    ]
    .join("\n")
}

fn tmux_monitor_help() -> String {
    [
        "monitor start <session> [seconds]",
        "monitor stop",
        "",
        "Start event-driven monitoring of a tmux session.",
        "Output is mirrored in real time using tmux control mode.",
        "",
        "Behavior:",
        "  Only one watch/stream is active at a time.",
        "  Starting monitor replaces any active stream or watch.",
        "  Without [seconds], monitoring stays attached until stopped or replaced.",
        "  With [seconds], monitoring runs for that duration and returns a one-shot transcript.",
        "  `monitor stop` stops the active watch/stream and returns to idle state.",
        "",
        "Examples:",
        "  monitor start demo",
        "  monitor start demo 5",
        "  monitor stop",
    ]
    .join("\n")
}

fn tmux_mirror_help() -> String {
    [
        "mirror history [--after N] [--limit N]",
        "mirror clear",
        "",
        "Inspect or clear the driver-local retained mirror history.",
        "",
        "This is local session history owned by the driver, not remote tmux scrollback.",
        "It is populated by monitor, stream, capture, and history commands.",
        "",
        "Options:",
        "  --after N   return items with sequence ids greater than N",
        "  --limit N   maximum number of retained entries to return [default: 20]",
        "",
        "Examples:",
        "  mirror history",
        "  mirror history --after 12 --limit 5",
        "  mirror clear",
    ]
    .join("\n")
}

fn tmux_capture_help() -> String {
    [
        "capture <target> <n>",
        "",
        "Capture the last N lines of scrollback from a target.",
        "This is a one-shot snapshot. Any active watch/stream is stopped first.",
        "",
        "Examples:",
        "  capture demo 50",
        "  capture demo:0.1 200",
    ]
    .join("\n")
}

fn tmux_history_help() -> String {
    [
        "history <session> [session...]",
        "",
        "Capture all visible panes from one or more sessions.",
        "Each pane is rendered with a header like --- session(%pane) ---.",
        "Empty panes are omitted. Any active watch/stream is stopped first.",
        "",
        "Examples:",
        "  history demo",
        "  history demo agent-a agent-b",
    ]
    .join("\n")
}

fn tmux_tui_help() -> String {
    [
        "tui on",
        "tui off",
        "",
        "Frontend control commands for the tmux driver examples.",
        "",
        "tui on:",
        "  Enter the split-screen TUI from the line REPL.",
        "",
        "tui off:",
        "  Leave the split-screen TUI and return to the line REPL.",
        "  Active monitor/stream state is torn down when leaving the TUI.",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CommandEngine;

    fn test_state() -> TmuxState {
        TmuxState {
            host_uri: "ssh://localhost".to_string(),
            host: HostHandle::local(),
            owned_sessions: HashSet::new(),
            known_sessions: Vec::new(),
            known_targets: Vec::new(),
            active_watch: None,
            active_stream: None,
            mirror_text: String::new(),
            mirror_label: String::new(),
            mirror_ansi: false,
            mirror_auto_refresh: false,
            mirror_history: HistoryBuffer::new(tmux_history_capacity()),
        }
    }

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
        let mut state = test_state();
        state.known_sessions = vec!["demo".to_string()];
        state.known_targets = vec![
            "demo".to_string(),
            "demo:0".to_string(),
            "demo:0.0".to_string(),
        ];

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
        let state = test_state();

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
        let state = test_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine.run_line("help create").await.expect("help output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("create"));
        assert!(rendered.contains("--size"));
    }

    #[tokio::test]
    async fn tmux_help_uses_rich_stream_topic() {
        let state = test_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine.run_line("help stream").await.expect("help output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("Mode guidance:"));
        assert!(rendered.contains("tail / monitor"));
        assert!(rendered.contains("stream demo --mode render"));
    }

    #[tokio::test]
    async fn tmux_tui_on_returns_frontend_effect() {
        let state = test_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine.run_line("tui on").await.expect("tui output");

        assert!(output.effects.contains(&CommandEffect::EnterTui));
    }

    #[tokio::test]
    async fn tmux_monitor_stop_is_safe_when_idle() {
        let state = test_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine
            .run_line("monitor stop")
            .await
            .expect("monitor stop output");

        assert_eq!(output.lines, vec!["No active monitor/stream"]);
    }

    #[tokio::test]
    async fn tmux_monitor_requires_explicit_subcommand() {
        let state = test_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let error = engine
            .run_line("monitor demo")
            .await
            .expect_err("monitor start syntax required");

        assert!(error.to_string().contains("unrecognized subcommand"));
    }

    #[tokio::test]
    async fn tmux_mirror_history_reads_retained_entries() {
        let mut state = test_state();
        state.mirror_text = "hello\nworld".to_string();
        state.mirror_label = "watching: demo".to_string();
        state.record_current_mirror_state();

        let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
        let output = engine
            .run_line("mirror history --limit 5")
            .await
            .expect("mirror history output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("mirror history: 1 item(s)"));
        assert!(rendered.contains("[seq=1 label=watching: demo"));
        assert!(rendered.contains("  hello"));
        assert!(rendered.contains("  world"));
    }
}
