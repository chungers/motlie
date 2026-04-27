use std::cmp::{max, min};
use std::io;
use std::net::{IpAddr, ToSocketAddrs};
use std::process::Command;
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, size as terminal_size, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use motlie_tmux::{
    has_visible_text, strip_ansi, CaptureNormalizeMode, CaptureOptions, CaptureResult,
    CreateSessionOptions, HostEvent, HostHandle, PaneAddress, ScrollbackQuery, SessionInfo,
    SshConfig,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span as TuiSpan, Text};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;

const DEFAULT_DETAIL_LINES: usize = 80;
const MIN_LEFT_PERCENT: u16 = 25;
const MAX_LEFT_PERCENT: u16 = 75;
const MIN_TOP_PERCENT: u16 = 25;
const MAX_TOP_PERCENT: u16 = 75;
const STATUS_BAR_BG: Color = Color::Blue;

const MOTLIE_PLACEHOLDER: &str = r#"                 _   _ _
 _ __ ___   ___ ┃ ┃_┃ (_) ___   ╲╲ ║ ╱╱
┃ '▄ ` ▄ ╲ ╱ ▄ ╲┃ ▄▄┃ ┃ ┃╱ ▄ ╲  ══ ╬ ══
┃ ┃ ┃ ┃ ┃ ┃ (_) ┃ ┃_┃ ┃ ┃  __╱  ╱╱ ║ ╲╲
┃▄┃ ┃▄┃ ┃▄┃╲▄▄▄╱ ╲▄▄┃▄┃▄┃╲▄▄▄┃"#;

const COMPACT_MOTLIE_PLACEHOLDER: &str = MOTLIE_PLACEHOLDER;
const BUILD_GIT_SHA: &str = env!("MMUX_GIT_SHA");
const NORMAL_STATUS_KEYS: &str =
    "↑/↓ select | ←/→ pane | m monitor | n new | k kill | h help | enter/a attach | mod-←/→ resize | q quit";
const PORTRAIT_STATUS_KEYS: &str =
    "↑/↓ select | ←/→ pane | m monitor | n new | k kill | h help | enter/a attach | mod-↑/↓ resize | q quit";
const HELP_KEY_FUNCTIONS: &str = r#"Keys:
↑/↓ select session or scroll detail
←/→ switch list/detail panes
PgUp/PgDn page current pane
Home/End jump current pane
m monitor highlighted session
n create session
k kill highlighted session
h help
Enter/a attach highlighted session
mod-←/→ resize L/R in landscape
mod-↑/↓ resize T/B in portrait
q/Ctrl-C quit"#;

#[derive(Debug, Clone, Parser)]
#[command(name = "mmux")]
#[command(about = "Select, preview, monitor, and attach tmux sessions")]
struct Cli {
    /// Force portrait layout instead of auto-detecting from the current PTY.
    #[arg(short = 'p', long, conflicts_with = "landscape")]
    portrait: bool,
    /// Force landscape layout instead of auto-detecting from the current PTY.
    #[arg(short = 'l', long, conflicts_with = "portrait")]
    landscape: bool,
    /// Print the selected session name for shell-script integration instead of attaching.
    #[arg(long)]
    script: bool,
    /// Optional SSH URI target. Omitted means local host.
    ssh_uri: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    Normal,
    Portrait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    List,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Button {
    Cancel,
    Ok,
}

#[derive(Debug, Clone)]
enum ModalState {
    NewSession {
        input: String,
        button: Button,
    },
    KillSession {
        id: String,
        name: String,
        button: Button,
    },
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetailMode {
    Sample,
    Monitor,
}

#[derive(Debug, Clone)]
struct SelectedSession {
    id: String,
    name: String,
}

trait SessionDetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()>;

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String>;

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String>;

    async fn deactivate(&mut self) -> Result<()>;

    fn mode(&self) -> DetailMode;
}

#[derive(Debug, Default)]
struct SampleDetailSource;

impl SessionDetailSource for SampleDetailSource {
    async fn activate(&mut self, _host: &HostHandle, _session: &SelectedSession) -> Result<()> {
        Ok(())
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        let Some(target) = host.session_by_id(&session.id).await? else {
            return Ok(format!("session {} disappeared", session.name));
        };
        target
            .sample_text_with_options(
                &ScrollbackQuery::LastLines(DEFAULT_DETAIL_LINES),
                &CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable),
                None,
            )
            .await
            .map(|capture| capture.text)
            .context("sample selected session")
    }

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String> {
        let Some(target) = host.session_by_id(&session.id).await? else {
            return Ok(String::new());
        };
        target
            .sample_text_with_options(
                &ScrollbackQuery::LinesRange {
                    older_than_lines,
                    count,
                },
                &CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable),
                None,
            )
            .await
            .map(|capture| capture.text)
            .context("fetch older sample lines")
    }

    async fn deactivate(&mut self) -> Result<()> {
        Ok(())
    }

    fn mode(&self) -> DetailMode {
        DetailMode::Sample
    }
}

struct MonitorDetailSource {
    session_id: Option<String>,
}

impl MonitorDetailSource {
    fn new() -> Self {
        Self { session_id: None }
    }
}

impl SessionDetailSource for MonitorDetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()> {
        self.deactivate().await?;
        if host.session_by_id(&session.id).await?.is_none() {
            return Err(anyhow!("session {} disappeared", session.name));
        }
        self.session_id = Some(session.id.clone());
        Ok(())
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        let Some(target) = host.session_by_id(&session.id).await? else {
            return Ok(format!("session {} disappeared", session.name));
        };
        let panes = target
            .capture_all_with_options(&CaptureOptions::with_mode(
                CaptureNormalizeMode::ScreenStable,
            ))
            .await
            .context("capture monitored session screen")?;
        Ok(render_screen_capture(&session.name, panes))
    }

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String> {
        let mut sample = SampleDetailSource;
        sample
            .fetch_older(host, session, older_than_lines, count)
            .await
    }

    async fn deactivate(&mut self) -> Result<()> {
        self.session_id = None;
        Ok(())
    }

    fn mode(&self) -> DetailMode {
        DetailMode::Monitor
    }
}

fn render_screen_capture(
    session_name: &str,
    panes: std::collections::HashMap<PaneAddress, CaptureResult>,
) -> String {
    if panes.is_empty() {
        return "(no visible content)\n".to_string();
    }

    let mut pane_list: Vec<_> = panes.into_iter().collect();
    pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

    if pane_list.len() == 1 {
        let text = pane_list.remove(0).1.text;
        if has_visible_text(&text) {
            return text;
        }
        return "(no visible content)\n".to_string();
    }

    let mut rendered = String::new();
    for (addr, result) in pane_list {
        if !has_visible_text(&result.text) {
            continue;
        }
        rendered.push_str(&format!("--- {}({}) ---\n", session_name, addr.pane_id));
        rendered.push_str(&result.text);
        if !result.text.ends_with('\n') {
            rendered.push('\n');
        }
    }

    if rendered.is_empty() {
        "(no visible content)\n".to_string()
    } else {
        rendered
    }
}

enum DetailSource {
    Sample(SampleDetailSource),
    Monitor(Box<MonitorDetailSource>),
}

impl DetailSource {
    fn sample() -> Self {
        Self::Sample(SampleDetailSource)
    }

    fn monitor() -> Self {
        Self::Monitor(Box::new(MonitorDetailSource::new()))
    }

    fn monitored_session_id(&self) -> Option<&str> {
        match self {
            DetailSource::Sample(_) => None,
            DetailSource::Monitor(source) => source.session_id.as_deref(),
        }
    }
}

impl SessionDetailSource for DetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()> {
        match self {
            DetailSource::Sample(source) => source.activate(host, session).await,
            DetailSource::Monitor(source) => source.activate(host, session).await,
        }
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        match self {
            DetailSource::Sample(source) => source.render(host, session).await,
            DetailSource::Monitor(source) => source.render(host, session).await,
        }
    }

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String> {
        match self {
            DetailSource::Sample(source) => {
                source
                    .fetch_older(host, session, older_than_lines, count)
                    .await
            }
            DetailSource::Monitor(source) => {
                source
                    .fetch_older(host, session, older_than_lines, count)
                    .await
            }
        }
    }

    async fn deactivate(&mut self) -> Result<()> {
        match self {
            DetailSource::Sample(source) => source.deactivate().await,
            DetailSource::Monitor(source) => source.deactivate().await,
        }
    }

    fn mode(&self) -> DetailMode {
        match self {
            DetailSource::Sample(source) => source.mode(),
            DetailSource::Monitor(source) => source.mode(),
        }
    }
}

struct AppState {
    host_label: String,
    host_ip_address: String,
    layout_mode: LayoutMode,
    focus: Focus,
    motd: String,
    motd_is_placeholder: bool,
    sessions: Vec<SessionInfo>,
    selected: usize,
    list_scroll: usize,
    detail_lines: Vec<String>,
    detail_scroll: usize,
    detail_view_height: usize,
    detail_source: DetailSource,
    status: String,
    left_percent: u16,
    top_percent: u16,
    modal: Option<ModalState>,
    auto_tail: bool,
}

impl AppState {
    #[cfg(test)]
    fn new(host_label: String, layout_mode: LayoutMode, motd: String, placeholder: bool) -> Self {
        Self::new_with_host_ip(
            host_label,
            "unknown".to_string(),
            layout_mode,
            motd,
            placeholder,
        )
    }

    fn new_with_host_ip(
        host_label: String,
        host_ip_address: String,
        layout_mode: LayoutMode,
        motd: String,
        placeholder: bool,
    ) -> Self {
        Self {
            host_label,
            host_ip_address,
            layout_mode,
            focus: Focus::List,
            motd,
            motd_is_placeholder: placeholder,
            sessions: Vec::new(),
            selected: 0,
            list_scroll: 0,
            detail_lines: Vec::new(),
            detail_scroll: 0,
            detail_view_height: 1,
            detail_source: DetailSource::sample(),
            status: "loading sessions".to_string(),
            left_percent: 42,
            top_percent: 30,
            modal: None,
            auto_tail: true,
        }
    }

    fn selected_session(&self) -> Option<SelectedSession> {
        self.sessions
            .get(self.selected)
            .map(|session| SelectedSession {
                id: session.id.clone(),
                name: session.name.clone(),
            })
    }

    fn preserve_selection(&mut self, previous_id: Option<String>) {
        if self.sessions.is_empty() {
            self.selected = 0;
            self.list_scroll = 0;
            return;
        }

        if let Some(id) = previous_id {
            if let Some(pos) = self.sessions.iter().position(|s| s.id == id) {
                self.selected = pos;
            } else {
                self.selected = min(self.selected, self.sessions.len().saturating_sub(1));
            }
        } else {
            self.selected = min(self.selected, self.sessions.len().saturating_sub(1));
        }
    }

    fn move_selection(&mut self, delta: isize) -> bool {
        if self.sessions.is_empty() {
            return false;
        }
        let old = self.selected;
        let max_index = self.sessions.len().saturating_sub(1) as isize;
        let next = (self.selected as isize + delta).clamp(0, max_index) as usize;
        self.selected = next;
        old != next
    }

    fn set_detail_text(&mut self, text: String) {
        self.detail_lines = text.lines().map(|line| line.to_string()).collect();
        self.detail_scroll = 0;
        self.auto_tail = true;
    }

    fn max_detail_scroll(&self) -> usize {
        self.detail_lines
            .len()
            .saturating_sub(self.detail_view_height)
    }

    fn scroll_detail(&mut self, delta: isize) {
        let old = self.detail_scroll;
        if delta < 0 {
            self.detail_scroll = self.detail_scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.detail_scroll = self
                .detail_scroll
                .saturating_add(delta as usize)
                .min(self.max_detail_scroll());
        }
        self.auto_tail = self.detail_scroll == 0;
        if old != self.detail_scroll {
            self.status = format!("detail offset {}", self.detail_scroll);
        }
    }

    fn detail_home(&mut self) {
        self.detail_scroll = self.max_detail_scroll();
        self.auto_tail = false;
    }

    fn detail_end(&mut self) {
        self.detail_scroll = 0;
        self.auto_tail = true;
    }
}

enum SelectorOutcome {
    Selected(SelectedSession),
    Cancelled,
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stderr>>,
    active: bool,
    keyboard_enhanced: bool,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stderr = io::stderr();
        execute!(
            stderr,
            PushKeyboardEnhancementFlags(keyboard_enhancement_flags()),
            EnterAlternateScreen,
            Hide
        )
        .context("enter alternate screen")?;
        let backend = CrosstermBackend::new(stderr);
        let terminal = Terminal::new(backend).context("create terminal backend")?;
        Ok(Self {
            terminal,
            active: true,
            keyboard_enhanced: true,
        })
    }

    fn draw(&mut self, app: &mut AppState) -> Result<()> {
        self.terminal.draw(|frame| draw(frame, app))?;
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        if self.active {
            disable_raw_mode().context("disable terminal raw mode")?;
            if self.keyboard_enhanced {
                execute!(
                    self.terminal.backend_mut(),
                    PopKeyboardEnhancementFlags,
                    Show,
                    LeaveAlternateScreen
                )
                .context("leave alternate screen")?;
                self.keyboard_enhanced = false;
            } else {
                execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen)
                    .context("leave alternate screen")?;
            }
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_raw_mode();
            if self.keyboard_enhanced {
                let _ = execute!(
                    self.terminal.backend_mut(),
                    PopKeyboardEnhancementFlags,
                    Show,
                    LeaveAlternateScreen
                );
            } else {
                let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
            }
        }
    }
}

fn keyboard_enhancement_flags() -> KeyboardEnhancementFlags {
    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
}

#[tokio::main]
async fn main() {
    let code = match run().await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("mmux: {err:#}");
            1
        }
    };
    std::process::exit(code);
}

async fn run() -> Result<i32> {
    if let Some(code) = maybe_run_forcecommand_bypass()? {
        return Ok(code);
    }

    let cli = Cli::parse();
    let (host, identity) = connect_host(&cli).await?;
    let layout = select_layout(cli.forced_layout());

    loop {
        let outcome = run_selector_once(&host, &identity, layout).await?;
        let selected = match outcome {
            SelectorOutcome::Selected(selected) => selected,
            SelectorOutcome::Cancelled => return Ok(if cli.script { 1 } else { 0 }),
        };

        if cli.script {
            println!("{}", selected.name);
            return Ok(0);
        }

        let target = match host.session_by_id(&selected.id).await? {
            Some(target) => target,
            None => {
                eprintln!("mmux: selected session disappeared; returning to selector");
                continue;
            }
        };
        let exit = target.attach_current_pty().await?;
        let code = exit.shell_status();
        if should_reenter_after_attach(&host, &selected, &exit).await? {
            continue;
        }
        return Ok(code);
    }
}

async fn should_reenter_after_attach(
    host: &HostHandle,
    selected: &SelectedSession,
    exit: &motlie_tmux::AttachExit,
) -> Result<bool> {
    if exit.success() {
        return Ok(true);
    }
    Ok(host.session_by_id(&selected.id).await?.is_some())
}

fn maybe_run_forcecommand_bypass() -> Result<Option<i32>> {
    let original = match std::env::var("SSH_ORIGINAL_COMMAND") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return Ok(None),
    };

    if std::env::var("MOTLIE_MMUX_BYPASS").ok().as_deref() == Some("1") {
        let status = Command::new("sh")
            .arg("-lc")
            .arg(original)
            .status()
            .context("run SSH_ORIGINAL_COMMAND bypass")?;
        return Ok(Some(shell_status(&status)));
    }

    eprintln!(
        "mmux: SSH_ORIGINAL_COMMAND is disabled for this account; set MOTLIE_MMUX_BYPASS=1 to delegate explicitly"
    );
    Ok(Some(126))
}

fn shell_status(status: &std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(code) = status.code() {
            code
        } else if let Some(signal) = status.signal() {
            128 + signal
        } else {
            1
        }
    }
    #[cfg(not(unix))]
    {
        status.code().unwrap_or(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HostIdentity {
    hostname: String,
    ip_address: String,
}

impl HostIdentity {
    fn new(hostname: String, port: u16) -> Self {
        let ip_address = resolve_ip_address(&hostname, port);
        Self {
            hostname,
            ip_address,
        }
    }

    fn local() -> Self {
        let hostname = local_hostname();
        let ip_address =
            local_ip_address(&hostname).unwrap_or_else(|| resolve_ip_address(&hostname, 22));
        Self {
            hostname,
            ip_address,
        }
    }
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| command_first_token("hostname", &[]))
        .unwrap_or_else(|| "localhost".to_string())
}

fn local_ip_address(hostname: &str) -> Option<String> {
    command_first_token("hostname", &["-I"]).or_else(|| {
        let resolved = resolve_ip_address(hostname, 22);
        if resolved == "unknown" {
            None
        } else {
            Some(resolved)
        }
    })
}

fn command_first_token(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .filter(|value| !value.is_empty())
}

fn resolve_ip_address(hostname: &str, port: u16) -> String {
    if let Ok(ip) = hostname.parse::<IpAddr>() {
        return ip.to_string();
    }

    let Ok(addrs) = (hostname, port).to_socket_addrs() else {
        return "unknown".to_string();
    };
    let ips = addrs.map(|addr| addr.ip()).collect::<Vec<_>>();
    ips.iter()
        .find(|ip| matches!(ip, IpAddr::V4(_)))
        .or_else(|| ips.first())
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

async fn connect_host(cli: &Cli) -> Result<(HostHandle, HostIdentity)> {
    match &cli.ssh_uri {
        Some(uri) => {
            let config = SshConfig::parse(uri).context("parse ssh target")?;
            let identity = HostIdentity::new(config.host().to_string(), config.port());
            let host = config.connect().await.context("connect ssh target")?;
            Ok((host, identity))
        }
        None => Ok((HostHandle::local(), HostIdentity::local())),
    }
}

impl Cli {
    fn forced_layout(&self) -> Option<LayoutMode> {
        if self.portrait {
            Some(LayoutMode::Portrait)
        } else if self.landscape {
            Some(LayoutMode::Normal)
        } else {
            None
        }
    }
}

fn select_layout(force: Option<LayoutMode>) -> LayoutMode {
    if let Some(layout) = force {
        return layout;
    }
    match terminal_size() {
        Ok((columns, rows)) if is_portrait_pty(columns, rows) => LayoutMode::Portrait,
        _ => LayoutMode::Normal,
    }
}

fn is_portrait_pty(columns: u16, rows: u16) -> bool {
    rows > 0 && (columns as u32) <= (rows as u32).saturating_mul(4)
}

async fn run_selector_once(
    host: &HostHandle,
    identity: &HostIdentity,
    layout: LayoutMode,
) -> Result<SelectorOutcome> {
    let (motd, placeholder) = load_motd(host).await;
    let mut app = AppState::new_with_host_ip(
        identity.hostname.clone(),
        identity.ip_address.clone(),
        layout,
        motd,
        placeholder,
    );
    refresh_sessions(host, &mut app, true).await?;

    let mut event_rx = host
        .watch_host_events()
        .await
        .ok()
        .map(HostEventStreamExt::into_receiver);
    let mut terminal = TerminalSession::enter()?;
    let mut last_detail_refresh = Instant::now();

    loop {
        drain_host_events(host, &mut app, &mut event_rx).await?;
        if last_detail_refresh.elapsed() >= Duration::from_millis(750) {
            refresh_detail(host, &mut app, false).await?;
            last_detail_refresh = Instant::now();
        }
        terminal.draw(&mut app)?;

        if event::poll(Duration::from_millis(100)).context("poll terminal event")? {
            let Event::Key(key) = event::read().context("read terminal event")? else {
                continue;
            };
            if key.kind == KeyEventKind::Release {
                continue;
            }
            match handle_key(host, &mut app, key).await? {
                KeyOutcome::Continue => {}
                KeyOutcome::Select(selected) => {
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Selected(selected));
                }
                KeyOutcome::Cancel => {
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Cancelled);
                }
            }
        }
    }
}

trait HostEventStreamExt {
    fn into_receiver(self) -> mpsc::Receiver<HostEvent>;
}

impl HostEventStreamExt for motlie_tmux::HostEventStream {
    fn into_receiver(self) -> mpsc::Receiver<HostEvent> {
        motlie_tmux::HostEventStream::into_receiver(self)
    }
}

async fn load_motd(host: &HostHandle) -> (String, bool) {
    match host.exec_shell("cat /etc/motd 2>/dev/null").await {
        Ok(text) if !text.trim().is_empty() => (text.trim_end().to_string(), false),
        _ => (MOTLIE_PLACEHOLDER.to_string(), true),
    }
}

async fn refresh_sessions(host: &HostHandle, app: &mut AppState, force_detail: bool) -> Result<()> {
    let previous = app.selected_session().map(|s| s.id);
    app.sessions = host.list_sessions().await.context("list tmux sessions")?;
    app.preserve_selection(previous);
    app.status = if app.sessions.is_empty() {
        format!("no sessions on {}", app.host_label)
    } else {
        format!("{} session(s)", app.sessions.len())
    };
    refresh_detail(host, app, force_detail).await?;
    Ok(())
}

async fn refresh_detail(host: &HostHandle, app: &mut AppState, force: bool) -> Result<()> {
    if app.sessions.is_empty() {
        app.detail_lines.clear();
        return Ok(());
    }

    match app.detail_source.mode() {
        DetailMode::Sample => {
            if force || app.detail_lines.is_empty() {
                let Some(selected) = app.selected_session() else {
                    return Ok(());
                };
                let text = app
                    .detail_source
                    .render(host, &selected)
                    .await
                    .unwrap_or_else(|err| format!("sample error: {err:#}"));
                app.set_detail_text(text);
            }
        }
        DetailMode::Monitor => {
            let Some(selected) = app.selected_session() else {
                return Ok(());
            };
            let text = app
                .detail_source
                .render(host, &selected)
                .await
                .unwrap_or_else(|err| format!("monitor error: {err:#}"));
            app.detail_lines = text.lines().map(|line| line.to_string()).collect();
            if app.auto_tail {
                app.detail_scroll = 0;
            }
        }
    }
    Ok(())
}

async fn drain_host_events(
    host: &HostHandle,
    app: &mut AppState,
    event_rx: &mut Option<mpsc::Receiver<HostEvent>>,
) -> Result<()> {
    let Some(rx) = event_rx.as_mut() else {
        return Ok(());
    };
    let mut should_refresh = false;
    let mut closed_monitored_session = None;
    loop {
        match rx.try_recv() {
            Ok(HostEvent::Disconnect { reason }) => {
                app.status = format!("event stream degraded: {reason}");
            }
            Ok(HostEvent::SessionClosed { id, name }) => {
                if closed_monitored_session.is_none() {
                    closed_monitored_session = stop_monitor_if_closed(app, id.as_str(), name).await;
                }
                should_refresh = true;
            }
            Ok(_) => should_refresh = true,
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                app.status = "event stream disconnected; using manual refreshes".to_string();
                break;
            }
        }
    }
    if should_refresh {
        refresh_sessions(host, app, true).await?;
        if let Some(name) = closed_monitored_session {
            app.status = format!("monitored session {name} closed");
        }
    }
    Ok(())
}

async fn stop_monitor_if_closed(app: &mut AppState, id: &str, name: String) -> Option<String> {
    if app.detail_source.monitored_session_id() == Some(id) {
        stop_detail_source(app).await;
        app.detail_source = DetailSource::sample();
        app.detail_lines.clear();
        Some(name)
    } else {
        None
    }
}

enum KeyOutcome {
    Continue,
    Select(SelectedSession),
    Cancel,
}

async fn handle_key(host: &HostHandle, app: &mut AppState, key: KeyEvent) -> Result<KeyOutcome> {
    if app.modal.is_some() {
        return handle_modal_key(host, app, key).await;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            return Ok(KeyOutcome::Cancel);
        }
        (KeyCode::Char('q'), _) => return Ok(KeyOutcome::Cancel),
        (KeyCode::Esc, _) => app.focus = Focus::List,
        (KeyCode::Char('h'), _) => app.modal = Some(ModalState::Help),
        (KeyCode::Char('m'), _) => start_monitor(host, app).await?,
        (KeyCode::Char('n'), _) => {
            app.modal = Some(ModalState::NewSession {
                input: String::new(),
                button: Button::Ok,
            });
        }
        (KeyCode::Char('k'), _) => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::KillSession {
                    id: selected.id,
                    name: selected.name,
                    button: Button::Cancel,
                });
            } else {
                app.status = "no session selected".to_string();
            }
        }
        (KeyCode::Enter, _) | (KeyCode::Char('a'), _) => {
            if let Some(selected) = app.selected_session() {
                return Ok(KeyOutcome::Select(selected));
            }
            app.status = "no session selected".to_string();
        }
        (KeyCode::Up, modifiers)
            if app.layout_mode == LayoutMode::Portrait && is_resize_modifier(modifiers) =>
        {
            app.top_percent = app.top_percent.saturating_sub(5).max(MIN_TOP_PERCENT);
        }
        (KeyCode::Down, modifiers)
            if app.layout_mode == LayoutMode::Portrait && is_resize_modifier(modifiers) =>
        {
            app.top_percent = app.top_percent.saturating_add(5).min(MAX_TOP_PERCENT);
        }
        (KeyCode::Left, modifiers)
            if app.layout_mode == LayoutMode::Normal && is_resize_modifier(modifiers) =>
        {
            app.left_percent = app.left_percent.saturating_sub(5).max(MIN_LEFT_PERCENT);
        }
        (KeyCode::Right, modifiers)
            if app.layout_mode == LayoutMode::Normal && is_resize_modifier(modifiers) =>
        {
            app.left_percent = app.left_percent.saturating_add(5).min(MAX_LEFT_PERCENT);
        }
        (KeyCode::Char('b'), modifiers)
            if app.layout_mode == LayoutMode::Normal && is_word_left_resize(modifiers) =>
        {
            app.left_percent = app.left_percent.saturating_sub(5).max(MIN_LEFT_PERCENT);
        }
        (KeyCode::Char('f'), modifiers)
            if app.layout_mode == LayoutMode::Normal && is_word_right_resize(modifiers) =>
        {
            app.left_percent = app.left_percent.saturating_add(5).min(MAX_LEFT_PERCENT);
        }
        (KeyCode::Right, _) if app.focus == Focus::List => app.focus = Focus::Detail,
        (KeyCode::Left, _) if app.focus == Focus::Detail => app.focus = Focus::List,
        (KeyCode::Up, _) => {
            if app.focus == Focus::List {
                if app.move_selection(-1) {
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.scroll_detail(1);
            }
        }
        (KeyCode::Down, _) => {
            if app.focus == Focus::List {
                if app.move_selection(1) {
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.scroll_detail(-1);
            }
        }
        (KeyCode::PageUp, _) => {
            if app.focus == Focus::List {
                if app.move_selection(-10) {
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                fetch_older_detail(host, app).await?;
                app.scroll_detail(10);
            }
        }
        (KeyCode::PageDown, _) => {
            if app.focus == Focus::List {
                if app.move_selection(10) {
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.scroll_detail(-10);
            }
        }
        (KeyCode::Home, _) => {
            if app.focus == Focus::List {
                if !app.sessions.is_empty() {
                    app.selected = 0;
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.detail_home();
            }
        }
        (KeyCode::End, _) => {
            if app.focus == Focus::List {
                if !app.sessions.is_empty() {
                    app.selected = app.sessions.len().saturating_sub(1);
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.detail_end();
            }
        }
        _ => {}
    }

    Ok(KeyOutcome::Continue)
}

fn is_resize_modifier(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT)
}

fn is_word_left_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
}

fn is_word_right_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
}

async fn handle_modal_key(
    host: &HostHandle,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    let mut close = false;
    let mut apply = false;
    match app.modal.as_mut() {
        Some(ModalState::NewSession { input, button }) => match key.code {
            KeyCode::Esc => close = true,
            KeyCode::Left => *button = Button::Cancel,
            KeyCode::Right => *button = Button::Ok,
            KeyCode::Enter => {
                apply = *button == Button::Ok;
                close = true;
            }
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                input.push(c);
            }
            _ => {}
        },
        Some(ModalState::KillSession { button, .. }) => match key.code {
            KeyCode::Esc => close = true,
            KeyCode::Left => *button = Button::Cancel,
            KeyCode::Right => *button = Button::Ok,
            KeyCode::Enter => {
                apply = *button == Button::Ok;
                close = true;
            }
            _ => {}
        },
        Some(ModalState::Help) => match key.code {
            KeyCode::Esc | KeyCode::Enter => close = true,
            _ => {}
        },
        None => {}
    }

    if close {
        let modal = app.modal.take();
        if apply {
            match modal {
                Some(ModalState::NewSession { input, .. }) => {
                    let name = input.trim();
                    if name.is_empty() {
                        app.status = "new session name is empty".to_string();
                    } else {
                        match host
                            .create_session(name, &CreateSessionOptions::default())
                            .await
                        {
                            Ok(_) => {
                                app.status = format!("created session {name}");
                                refresh_sessions(host, app, true).await?;
                            }
                            Err(err) => app.status = format!("create failed: {err}"),
                        }
                    }
                }
                Some(ModalState::KillSession { id, name, .. }) => {
                    match host.session_by_id(&id).await? {
                        Some(target) => match target.kill().await {
                            Ok(()) => {
                                app.status = format!("killed session {name}");
                                refresh_sessions(host, app, true).await?;
                            }
                            Err(err) => app.status = format!("kill failed: {err}"),
                        },
                        None => {
                            app.status = format!("session {name} disappeared before kill");
                            refresh_sessions(host, app, true).await?;
                        }
                    }
                }
                Some(ModalState::Help) => {}
                None => {}
            }
        }
    }
    Ok(KeyOutcome::Continue)
}

async fn start_monitor(host: &HostHandle, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
        app.status = "no session selected".to_string();
        return Ok(());
    };

    stop_detail_source(app).await;
    let mut source = DetailSource::monitor();
    match source.activate(host, &selected).await {
        Ok(()) => {
            app.detail_source = source;
            app.status = format!("monitoring {}", selected.name);
            app.detail_lines.clear();
        }
        Err(err) => {
            app.detail_source = DetailSource::sample();
            app.status = format!("monitor failed: {err}");
            refresh_detail(host, app, true).await?;
        }
    }
    Ok(())
}

async fn stop_detail_source(app: &mut AppState) {
    let _ = app.detail_source.deactivate().await;
}

async fn fetch_older_detail(host: &HostHandle, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
        return Ok(());
    };
    let older_than_lines = app.detail_lines.len();
    let text = app
        .detail_source
        .fetch_older(host, &selected, older_than_lines, DEFAULT_DETAIL_LINES)
        .await
        .unwrap_or_default();
    if !text.trim().is_empty() {
        let mut older = text
            .lines()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        older.append(&mut app.detail_lines);
        app.detail_lines = older;
        app.status = "fetched older scrollback".to_string();
    }
    Ok(())
}

fn draw(frame: &mut Frame<'_>, app: &mut AppState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    match app.layout_mode {
        LayoutMode::Normal => draw_normal(frame, chunks[0], app),
        LayoutMode::Portrait => draw_portrait(frame, chunks[0], app),
    }
    draw_status(frame, chunks[1], app);
    if let Some(modal) = &app.modal {
        draw_modal(frame, area, modal);
    }
}

fn draw_normal(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(app.left_percent),
            Constraint::Percentage(100 - app.left_percent),
        ])
        .split(area);

    let compact_placeholder = use_compact_placeholder(app, columns[0].width, columns[0].height);
    let motd_lines = motd_render_line_count(app, compact_placeholder);
    let max_motd = max(3, area.height.saturating_mul(30) / 100);
    let desired = motd_lines.saturating_add(2);
    let upper = if app.motd_is_placeholder {
        min(desired, columns[0].height.saturating_sub(3))
    } else {
        min(desired, max_motd)
    };
    let upper = max(3, min(upper, columns[0].height.saturating_sub(3)));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(upper), Constraint::Min(3)])
        .split(columns[0]);

    draw_motd(frame, left[0], app);
    draw_sessions(frame, left[1], app);
    draw_detail(frame, columns[1], app, " Detail ");
}

fn draw_portrait(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(app.top_percent),
            Constraint::Percentage(100 - app.top_percent),
        ])
        .split(area);
    draw_sessions(frame, rows[0], app);
    draw_detail(frame, rows[1], app, " Detail ");
}

fn focused_style(app: &AppState, focus: Focus) -> Style {
    if app.focus == focus {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn use_compact_placeholder(app: &AppState, width: u16, height: u16) -> bool {
    if !app.motd_is_placeholder {
        return false;
    }
    let full_height = MOTLIE_PLACEHOLDER
        .lines()
        .count()
        .saturating_add(1)
        .saturating_add(2) as u16;
    width < 63 || height < full_height.saturating_add(3)
}

fn motd_render_line_count(app: &AppState, compact_placeholder: bool) -> u16 {
    if app.motd_is_placeholder && compact_placeholder {
        COMPACT_MOTLIE_PLACEHOLDER.lines().count().saturating_add(1) as u16
    } else if app.motd_is_placeholder {
        app.motd.lines().count().saturating_add(1) as u16
    } else {
        app.motd.lines().count() as u16
    }
}

fn motd_render_text(app: &AppState, area: Rect) -> String {
    if !app.motd_is_placeholder {
        return app.motd.clone();
    }
    if use_compact_placeholder(app, area.width, area.height) {
        format!("{COMPACT_MOTLIE_PLACEHOLDER}\n(no /etc/motd)")
    } else {
        format!("{}\n(no /etc/motd)", app.motd)
    }
}

fn draw_motd(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let text_style = if app.motd_is_placeholder {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    let text = motd_render_text(app, area);
    let paragraph = Paragraph::new(text)
        .style(text_style)
        .block(
            Block::default()
                .title(" MOTD ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn sessions_title(app: &AppState) -> String {
    format!(
        " Sessions [{}] @ {}, {} ",
        app.sessions.len(),
        app.host_label,
        app.host_ip_address
    )
}

fn draw_sessions(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let height = area.height.saturating_sub(2) as usize;
    let mut lines = Vec::new();
    if app.sessions.is_empty() {
        lines.push(Line::from(TuiSpan::styled(
            format!("(no sessions on {} - press n to create)", app.host_label),
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let start = app.selected.saturating_sub(height.saturating_sub(1));
        for (idx, session) in app.sessions.iter().enumerate().skip(start).take(height) {
            let selected = idx == app.selected;
            let marker = if selected { ">" } else { " " };
            let attached = if session.attached { "*" } else { " " };
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(TuiSpan::styled(
                format!("{marker}{attached} {:<28} {}", session.name, session.id),
                style,
            )));
        }
    }

    let block = Block::default()
        .title(sessions_title(app))
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::List));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_detail(frame: &mut Frame<'_>, area: Rect, app: &mut AppState, title: &str) {
    let height = area.height.saturating_sub(2) as usize;
    app.detail_view_height = max(1, height);
    app.detail_scroll = app.detail_scroll.min(app.max_detail_scroll());
    let total = app.detail_lines.len();
    let end = total.saturating_sub(app.detail_scroll);
    let start = end.saturating_sub(height);
    let visible = if start < end {
        app.detail_lines[start..end].join("\n")
    } else if app.sessions.is_empty() {
        "press n to create a session".to_string()
    } else {
        String::new()
    };
    let position = if total == 0 {
        "0/0".to_string()
    } else {
        format!("{}-{}/{}", start + 1, end, total)
    };
    let title = match app.detail_source.mode() {
        DetailMode::Sample => format!("{title} {position} "),
        DetailMode::Monitor => format!(" Detail - monitor {position} "),
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::Detail));
    frame.render_widget(
        Paragraph::new(detail_text_for_render(&visible))
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    if total > height {
        let mut scrollbar_state = ScrollbarState::new(total)
            .position(start)
            .viewport_content_length(height);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .thumb_style(Style::default().fg(Color::Green))
                .track_style(Style::default().fg(Color::DarkGray)),
            area,
            &mut scrollbar_state,
        );
    }
}

fn detail_text_for_render(text: &str) -> Text<'_> {
    text.into_text()
        .unwrap_or_else(|_| Text::raw(strip_ansi(text)))
}

fn draw_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let text = status_line_text(app, &chrono::Local::now().format("%H:%M:%S").to_string());
    let paragraph = Paragraph::new(Line::from(vec![
        TuiSpan::styled(text, Style::default().fg(Color::White).bg(STATUS_BAR_BG)),
        TuiSpan::styled(
            format!(" | {}", app.status),
            Style::default().fg(Color::Yellow).bg(STATUS_BAR_BG),
        ),
    ]))
    .style(Style::default().bg(STATUS_BAR_BG));
    frame.render_widget(paragraph, area);
}

fn status_line_text(app: &AppState, time: &str) -> String {
    let keys = if app.layout_mode == LayoutMode::Portrait {
        PORTRAIT_STATUS_KEYS
    } else {
        NORMAL_STATUS_KEYS
    };
    format!(" {} | {} ", time, keys)
}

fn draw_modal(frame: &mut Frame<'_>, area: Rect, modal: &ModalState) {
    let (title, body, button) = modal_content(modal);
    let body_width = body
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0) as u16;
    let width = min(
        max(60, body_width.saturating_add(4)),
        area.width.saturating_sub(4),
    );
    let body_height = body.lines().count() as u16;
    let height = min(
        max(7, body_height.saturating_add(2)),
        area.height.saturating_sub(2),
    );
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    let rect = Rect::new(x, y, width, height);
    frame.render_widget(Clear, rect);

    let border = if button == Button::Ok {
        Color::Green
    } else {
        Color::Yellow
    };
    let paragraph = Paragraph::new(body)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, rect);
}

fn modal_content(modal: &ModalState) -> (&'static str, String, Button) {
    let (title, body, button) = match modal {
        ModalState::NewSession { input, button } => (
            " New Session ",
            format!(
                "Name: {input}\n\n{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            *button,
        ),
        ModalState::KillSession { name, button, .. } => (
            " Kill Session ",
            format!(
                "Kill session {name}?\n\n{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            *button,
        ),
        ModalState::Help => (
            " Help ",
            format!(
                "{MOTLIE_PLACEHOLDER}\n\n{HELP_KEY_FUNCTIONS}\n\nGit SHA: {BUILD_GIT_SHA}\n\n[Ok]"
            ),
            Button::Ok,
        ),
    };
    (title, body, button)
}

fn button_text(active: Button, button: Button) -> String {
    let label = match button {
        Button::Cancel => "Cancel",
        Button::Ok => "Ok",
    };
    if active == button {
        format!("[{label}]")
    } else {
        format!(" {label} ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use motlie_tmux::{transport::MockTransport, TransportKind};

    #[test]
    fn cli_accepts_script_and_rejects_removed_mode_flags() {
        let script = Cli::try_parse_from(["mmux", "--script"]).unwrap();
        assert!(script.script);

        let print_session = Cli::try_parse_from(["mmux", "--print-session"]);
        assert!(print_session.is_err());

        let dashboard = Cli::try_parse_from(["mmux", "--dashboard"]);
        assert!(dashboard.is_err());

        Cli::command().debug_assert();
    }

    #[test]
    fn cli_accepts_layout_force_flags_and_rejects_old_short_flag() {
        let portrait = Cli::try_parse_from(["mmux", "--portrait"]).unwrap();
        assert!(portrait.portrait);
        assert_eq!(portrait.forced_layout(), Some(LayoutMode::Portrait));

        let portrait_short = Cli::try_parse_from(["mmux", "-p"]).unwrap();
        assert!(portrait_short.portrait);

        let landscape = Cli::try_parse_from(["mmux", "--landscape"]).unwrap();
        assert!(landscape.landscape);
        assert_eq!(landscape.forced_layout(), Some(LayoutMode::Normal));

        let landscape_short = Cli::try_parse_from(["mmux", "-l"]).unwrap();
        assert!(landscape_short.landscape);

        let old_short = Cli::try_parse_from(["mmux", "-s"]);
        assert!(old_short.is_err());

        let conflicting = Cli::try_parse_from(["mmux", "--portrait", "--landscape"]);
        assert!(conflicting.is_err());
    }

    #[test]
    fn layout_auto_detection_uses_pty_aspect_ratio() {
        assert_eq!(
            select_layout(Some(LayoutMode::Portrait)),
            LayoutMode::Portrait
        );
        assert_eq!(select_layout(Some(LayoutMode::Normal)), LayoutMode::Normal);
        assert!(is_portrait_pty(64, 32));
        assert!(is_portrait_pty(60, 30));
        assert!(is_portrait_pty(66, 30));
        assert!(is_portrait_pty(80, 24));
        assert!(is_portrait_pty(100, 30));
        assert!(is_portrait_pty(160, 40));
        assert!(is_portrait_pty(40, 40));
        assert!(!is_portrait_pty(161, 40));
        assert!(!is_portrait_pty(200, 40));
    }

    #[test]
    fn portrait_default_split_is_30_70() {
        let app = AppState::new(
            "host".to_string(),
            LayoutMode::Portrait,
            "motd".to_string(),
            false,
        );

        assert_eq!(app.top_percent, 30);
    }

    #[test]
    fn status_line_omits_layout_mode() {
        let normal = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        let normal_status = status_line_text(&normal, "12:34:56");
        assert!(normal_status.contains(" 12:34:56 | ↑/↓ select"));
        assert!(!normal_status.contains("keys"));
        assert!(!normal_status.contains("host"));
        assert!(normal_status.contains("↑/↓ select"));
        assert!(normal_status.contains("←/→ pane"));
        assert!(normal_status.contains("h help"));
        assert!(normal_status.contains("mod-←/→ resize"));
        assert!(!normal_status.contains("up/down"));
        assert!(!normal_status.contains("right/left"));
        assert!(!normal_status.contains("mod-left/right"));
        assert!(!normal_status.contains("list"));
        assert!(!normal_status.contains("detail"));
        assert!(!normal_status.contains("focus"));
        assert!(!normal_status.contains("normal"));
        assert!(!normal_status.contains("landscape"));
        assert!(!normal_status.contains("portrait"));

        let portrait = AppState::new(
            "host".to_string(),
            LayoutMode::Portrait,
            "motd".to_string(),
            false,
        );
        let portrait_status = status_line_text(&portrait, "12:34:56");
        assert!(portrait_status.contains(" 12:34:56 | ↑/↓ select"));
        assert!(!portrait_status.contains("keys"));
        assert!(!portrait_status.contains("host"));
        assert!(portrait_status.contains("↑/↓ select"));
        assert!(portrait_status.contains("←/→ pane"));
        assert!(portrait_status.contains("h help"));
        assert!(portrait_status.contains("mod-↑/↓ resize"));
        assert!(!portrait_status.contains("up/down"));
        assert!(!portrait_status.contains("right/left"));
        assert!(!portrait_status.contains("mod-up/down"));
        assert!(!portrait_status.contains("list"));
        assert!(!portrait_status.contains("detail"));
        assert!(!portrait_status.contains("focus"));
        assert!(!portrait_status.contains("normal"));
        assert!(!portrait_status.contains("landscape"));
        assert!(!portrait_status.contains("portrait"));
    }

    #[test]
    fn sessions_title_includes_host_label() {
        let mut app = AppState::new_with_host_ip(
            "target-host".to_string(),
            "192.0.2.10".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        app.sessions = vec![
            SessionInfo {
                name: "dev".to_string(),
                id: "$1".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            },
            SessionInfo {
                name: "build".to_string(),
                id: "$2".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            },
        ];

        assert_eq!(
            sessions_title(&app),
            " Sessions [2] @ target-host, 192.0.2.10 "
        );
    }

    #[test]
    fn resolve_ip_address_preserves_literal_ip() {
        assert_eq!(resolve_ip_address("192.0.2.10", 22), "192.0.2.10");
    }

    #[test]
    fn selection_preserves_stable_id_after_rename() {
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        app.sessions = vec![SessionInfo {
            name: "old".to_string(),
            id: "$1".to_string(),
            created: 0,
            attached: false,
            window_count: 1,
            group: None,
        }];
        app.selected = 0;
        app.sessions = vec![SessionInfo {
            name: "new".to_string(),
            id: "$1".to_string(),
            created: 0,
            attached: false,
            window_count: 1,
            group: None,
        }];
        app.preserve_selection(Some("$1".to_string()));
        assert_eq!(app.selected, 0);
        assert_eq!(
            app.selected_session().map(|s| s.name),
            Some("new".to_string())
        );
    }

    #[test]
    fn motd_placeholder_uses_compact_graphic_when_narrow() {
        let app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            MOTLIE_PLACEHOLDER.to_string(),
            true,
        );
        let text = motd_render_text(&app, Rect::new(0, 0, 40, 5));
        assert!(text.contains('▄'));
        assert!(text.contains("(no /etc/motd)"));
    }

    #[tokio::test]
    async fn closed_monitored_session_resets_detail_source() {
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        app.detail_source = DetailSource::Monitor(Box::new(MonitorDetailSource {
            session_id: Some("$1".to_string()),
        }));
        app.detail_lines = vec!["live".to_string()];

        let closed = stop_monitor_if_closed(&mut app, "$1", "dev".to_string()).await;

        assert_eq!(closed, Some("dev".to_string()));
        assert_eq!(app.detail_source.mode(), DetailMode::Sample);
        assert!(app.detail_lines.is_empty());
    }

    #[test]
    fn detail_scroll_up_moves_toward_older_content() {
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        app.detail_lines = (0..100).map(|n| format!("line {n}")).collect();
        app.detail_view_height = 10;

        app.scroll_detail(1);
        assert_eq!(app.detail_scroll, 1);
        assert!(!app.auto_tail);

        app.scroll_detail(-1);
        assert_eq!(app.detail_scroll, 0);
        assert!(app.auto_tail);
    }

    #[tokio::test]
    async fn q_exits_like_ctrl_c() {
        let host = HostHandle::local();
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Cancel));
    }

    fn app_with_session() -> AppState {
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        app.sessions = vec![SessionInfo {
            name: "dev".to_string(),
            id: "$1".to_string(),
            created: 0,
            attached: false,
            window_count: 1,
            group: None,
        }];
        app
    }

    #[tokio::test]
    async fn a_attaches_like_enter() {
        let host = HostHandle::local();
        let mut app = app_with_session();

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(
            outcome,
            KeyOutcome::Select(SelectedSession { name, .. }) if name == "dev"
        ));
    }

    #[tokio::test]
    async fn g_no_longer_attaches() {
        let host = HostHandle::local();
        let mut app = app_with_session();

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
    }

    #[tokio::test]
    async fn h_opens_help_modal_and_enter_or_escape_closes_it() {
        let host = HostHandle::local();
        let mut app = app_with_session();

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        let Some(ModalState::Help) = app.modal.as_ref() else {
            panic!("expected help modal");
        };
        let (title, body, button) = modal_content(app.modal.as_ref().unwrap());
        assert_eq!(title, " Help ");
        assert_eq!(button, Button::Ok);
        assert!(body.contains(MOTLIE_PLACEHOLDER));
        assert!(body.contains(HELP_KEY_FUNCTIONS));
        assert!(body.contains("↑/↓ select session or scroll detail"));
        assert!(body.contains("mod-←/→ resize L/R in landscape"));
        assert!(body.contains("mod-↑/↓ resize T/B in portrait"));
        assert!(body.contains("Git SHA: "));
        assert!(body.contains(BUILD_GIT_SHA));
        assert!(body.contains("[Ok]"));
        let logo_pos = body.find(MOTLIE_PLACEHOLDER).unwrap();
        let keys_pos = body.find(HELP_KEY_FUNCTIONS).unwrap();
        let ok_pos = body.find("[Ok]").unwrap();
        assert!(logo_pos < keys_pos);
        assert!(keys_pos < ok_pos);

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        )
        .await
        .unwrap();
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(app.modal.is_none());

        handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
        )
        .await
        .unwrap();
        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        )
        .await
        .unwrap();
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(app.modal.is_none());
    }

    #[tokio::test]
    async fn right_arrow_focuses_detail_from_session_list() {
        let host = HostHandle::local();
        let mut app = app_with_session();

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        assert_eq!(app.focus, Focus::Detail);
    }

    #[tokio::test]
    async fn right_arrow_focuses_monitor_detail_from_session_list() {
        let host = HostHandle::local();
        let mut app = app_with_session();
        app.detail_source = DetailSource::Monitor(Box::new(MonitorDetailSource {
            session_id: Some("$1".to_string()),
        }));

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        assert_eq!(app.focus, Focus::Detail);
        assert_eq!(app.detail_source.mode(), DetailMode::Monitor);
    }

    #[tokio::test]
    async fn left_arrow_focuses_list_from_detail_or_monitor() {
        let host = HostHandle::local();
        let mut app = app_with_session();
        app.focus = Focus::Detail;
        app.detail_source = DetailSource::Monitor(Box::new(MonitorDetailSource {
            session_id: Some("$1".to_string()),
        }));

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        assert_eq!(app.focus, Focus::List);
        assert_eq!(app.detail_source.mode(), DetailMode::Monitor);
    }

    #[tokio::test]
    async fn ctrl_left_with_extra_modifiers_resizes_normal_layout() {
        let host = HostHandle::local();
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        let before = app.left_percent;

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL | KeyModifiers::SHIFT),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(app.left_percent < before);
    }

    #[tokio::test]
    async fn modified_left_and_right_resize_normal_layout() {
        let host = HostHandle::local();
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        let initial = app.left_percent;

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::SHIFT),
        )
        .await
        .unwrap();
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(app.left_percent < initial);

        let shrunk = app.left_percent;
        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT),
        )
        .await
        .unwrap();
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(app.left_percent > shrunk);
    }

    #[tokio::test]
    async fn word_arrow_fallback_sequences_resize_normal_layout() {
        let host = HostHandle::local();
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        let initial = app.left_percent;

        handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT),
        )
        .await
        .unwrap();
        assert!(app.left_percent < initial);

        let shrunk = app.left_percent;
        handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT),
        )
        .await
        .unwrap();
        assert!(app.left_percent > shrunk);
    }

    #[tokio::test]
    async fn plain_left_does_not_resize_normal_layout() {
        let host = HostHandle::local();
        let mut app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            "motd".to_string(),
            false,
        );
        let initial = app.left_percent;

        let outcome = handle_key(
            &host,
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::NONE),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, KeyOutcome::Continue));
        assert_eq!(app.left_percent, initial);
    }

    #[test]
    fn detail_uses_ansi_vte_parser_for_screen_content() {
        let text = detail_text_for_render("\x1b[31mred\x1b[0m");
        assert_eq!(text.lines[0].spans[0].content.as_ref(), "red");
        assert!(!text.lines[0].spans[0].content.contains('\x1b'));
    }

    #[tokio::test]
    async fn sample_detail_preserves_ansi_color_for_detail_pane() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "dev $7 0 0 1 \n")
            .with_response("capture-pane -ep", "\x1b[34mBLUE\x1b[0m\n");
        let host = HostHandle::new(TransportKind::Mock(mock), None);
        let selected = SelectedSession {
            id: "$7".to_string(),
            name: "dev".to_string(),
        };
        let mut source = SampleDetailSource;

        let rendered = source.render(&host, &selected).await.unwrap();

        assert!(rendered.contains("\x1b[34mBLUE\x1b[0m"));
    }

    #[tokio::test]
    async fn monitor_detail_captures_rendered_screen_with_ansi() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "dash $7 0 0 1 \n")
            .with_response("list-panes", "%0 dash 0 0 main bash 100 80 24 1\n")
            .with_response("capture-pane -ep", "\x1b[32mREADY\x1b[0m\n");
        let host = HostHandle::new(TransportKind::Mock(mock), None);
        let selected = SelectedSession {
            id: "$7".to_string(),
            name: "dash".to_string(),
        };
        let mut source = MonitorDetailSource::new();

        source.activate(&host, &selected).await.unwrap();
        let rendered = source.render(&host, &selected).await.unwrap();

        assert!(rendered.contains("\x1b[32mREADY\x1b[0m"));
        assert!(!rendered.contains("%output"));
    }
}
