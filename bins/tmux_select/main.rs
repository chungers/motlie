use std::cmp::{max, min};
use std::io;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use motlie_tmux::{
    CreateSessionOptions, HistoryOptions, HostEvent, HostHandle, LabelFormat, RenderMode,
    ScrollbackQuery, SessionInfo, SessionWatchHandle, SessionWatchOptions, SshConfig,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span as TuiSpan};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;

const DEFAULT_DETAIL_LINES: usize = 80;
const MONITOR_HISTORY_LINES: usize = 10_000;
const MIN_LEFT_PERCENT: u16 = 25;
const MAX_LEFT_PERCENT: u16 = 75;
const MIN_TOP_PERCENT: u16 = 25;
const MAX_TOP_PERCENT: u16 = 75;

const MOTLIE_PLACEHOLDER: &str = r#"███╗   ███╗  ██████╗  ████████╗ ██╗      ██╗ ███████╗
 ████╗ ████║ ██╔═══██╗ ╚══██╔══╝ ██║      ██║ ██╔════╝
 ██╔████╔██║ ██║   ██║    ██║    ██║      ██║ █████╗    ╲╲ ║ ╱╱
 ██║╚██╔╝██║ ██║   ██║    ██║    ██║      ██║ ██╔══╝    ══ ╬ ══
 ██║ ╚═╝ ██║ ╚██████╔╝    ██║    ███████╗ ██║ ███████╗  ╱╱ ║ ╲╲
 ╚═╝     ╚═╝  ╚═════╝     ╚═╝    ╚══════╝ ╚═╝ ╚══════╝"#;

#[derive(Debug, Clone, Parser)]
#[command(name = "tmux_select")]
#[command(about = "Select, preview, monitor, and attach tmux sessions")]
struct Cli {
    /// Use compact short mode.
    #[arg(short = 's')]
    short: bool,
    /// Print the selected session name and exit instead of attaching.
    #[arg(long, conflicts_with = "dashboard")]
    print_session: bool,
    /// Re-enter the selector after a clean attach detach.
    #[arg(long)]
    dashboard: bool,
    /// Optional SSH URI target. Omitted means local host.
    ssh_uri: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    Normal,
    Short,
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
            .sample_text(&ScrollbackQuery::LastLines(DEFAULT_DETAIL_LINES))
            .await
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
            .sample_text(&ScrollbackQuery::LinesRange {
                older_than_lines,
                count,
            })
            .await
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
    handle: Option<SessionWatchHandle>,
    session_id: Option<String>,
}

impl MonitorDetailSource {
    fn new() -> Self {
        Self {
            handle: None,
            session_id: None,
        }
    }
}

impl SessionDetailSource for MonitorDetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()> {
        self.deactivate().await?;
        let opts = SessionWatchOptions {
            queue_capacity: 256,
            history: HistoryOptions {
                max_entries: MONITOR_HISTORY_LINES,
                max_render_chars: 0,
                label_format: LabelFormat::Bracketed,
                include_omission_marker: true,
                render_mode: RenderMode::Interleaved,
                global_max_render_chars: 0,
            },
        };
        self.handle = Some(
            host.watch_session(&session.name, &opts)
                .await
                .context("start session monitor")?,
        );
        self.session_id = Some(session.id.clone());
        Ok(())
    }

    async fn render(&mut self, _host: &HostHandle, _session: &SelectedSession) -> Result<String> {
        match self.handle.as_ref() {
            Some(handle) => Ok(handle.render_text().await),
            None => Ok(String::new()),
        }
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
        if let Some(handle) = self.handle.take() {
            handle.shutdown().await?;
        }
        self.session_id = None;
        Ok(())
    }

    fn mode(&self) -> DetailMode {
        DetailMode::Monitor
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
    layout_mode: LayoutMode,
    focus: Focus,
    motd: String,
    motd_is_placeholder: bool,
    sessions: Vec<SessionInfo>,
    selected: usize,
    list_scroll: usize,
    detail_lines: Vec<String>,
    detail_scroll: usize,
    detail_source: DetailSource,
    status: String,
    left_percent: u16,
    top_percent: u16,
    modal: Option<ModalState>,
    auto_tail: bool,
}

impl AppState {
    fn new(host_label: String, layout_mode: LayoutMode, motd: String, placeholder: bool) -> Self {
        Self {
            host_label,
            layout_mode,
            focus: Focus::List,
            motd,
            motd_is_placeholder: placeholder,
            sessions: Vec::new(),
            selected: 0,
            list_scroll: 0,
            detail_lines: Vec::new(),
            detail_scroll: 0,
            detail_source: DetailSource::sample(),
            status: "loading sessions".to_string(),
            left_percent: 42,
            top_percent: 40,
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

    fn scroll_detail(&mut self, delta: isize) {
        let old = self.detail_scroll;
        if delta < 0 {
            self.detail_scroll = self.detail_scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.detail_scroll = self.detail_scroll.saturating_add(delta as usize);
        }
        self.auto_tail = self.detail_scroll == 0;
        if old != self.detail_scroll {
            self.status = format!("detail offset {}", self.detail_scroll);
        }
    }

    fn detail_home(&mut self) {
        self.detail_scroll = self.detail_lines.len();
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
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stderr = io::stderr();
        execute!(stderr, EnterAlternateScreen, Hide).context("enter alternate screen")?;
        let backend = CrosstermBackend::new(stderr);
        let terminal = Terminal::new(backend).context("create terminal backend")?;
        Ok(Self {
            terminal,
            active: true,
        })
    }

    fn draw(&mut self, app: &mut AppState) -> Result<()> {
        self.terminal.draw(|frame| draw(frame, app))?;
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        if self.active {
            disable_raw_mode().context("disable terminal raw mode")?;
            execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen)
                .context("leave alternate screen")?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_raw_mode();
            let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
        }
    }
}

#[tokio::main]
async fn main() {
    let code = match run().await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("tmux_select: {err:#}");
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
    let (host, label) = connect_host(&cli).await?;
    let layout = if cli.short {
        LayoutMode::Short
    } else {
        LayoutMode::Normal
    };

    loop {
        let outcome = run_selector_once(&host, &label, layout).await?;
        let selected = match outcome {
            SelectorOutcome::Selected(selected) => selected,
            SelectorOutcome::Cancelled => return Ok(if cli.print_session { 1 } else { 0 }),
        };

        if cli.print_session {
            println!("{}", selected.name);
            return Ok(0);
        }

        let target = match host.session_by_id(&selected.id).await? {
            Some(target) => target,
            None => {
                if cli.dashboard {
                    eprintln!("tmux_select: selected session disappeared; returning to dashboard");
                    continue;
                }
                return Err(anyhow!("selected session disappeared before attach"));
            }
        };
        let exit = target.attach_current_pty().await?;
        let code = exit.shell_status();
        if cli.dashboard && exit.success() {
            continue;
        }
        return Ok(code);
    }
}

fn maybe_run_forcecommand_bypass() -> Result<Option<i32>> {
    let original = match std::env::var("SSH_ORIGINAL_COMMAND") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return Ok(None),
    };

    if std::env::var("MOTLIE_TMUX_SELECT_BYPASS").ok().as_deref() == Some("1") {
        let status = Command::new("sh")
            .arg("-lc")
            .arg(original)
            .status()
            .context("run SSH_ORIGINAL_COMMAND bypass")?;
        return Ok(Some(shell_status(&status)));
    }

    eprintln!(
        "tmux_select: SSH_ORIGINAL_COMMAND is disabled for this account; set MOTLIE_TMUX_SELECT_BYPASS=1 to delegate explicitly"
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

async fn connect_host(cli: &Cli) -> Result<(HostHandle, String)> {
    match &cli.ssh_uri {
        Some(uri) => {
            let config = SshConfig::parse(uri).context("parse ssh target")?;
            let label = config.host().to_string();
            let host = config.connect().await.context("connect ssh target")?;
            Ok((host, label))
        }
        None => Ok((HostHandle::local(), "localhost".to_string())),
    }
}

async fn run_selector_once(
    host: &HostHandle,
    label: &str,
    layout: LayoutMode,
) -> Result<SelectorOutcome> {
    let (motd, placeholder) = load_motd(host).await;
    let mut app = AppState::new(label.to_string(), layout, motd, placeholder);
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
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(KeyOutcome::Cancel),
        (KeyCode::Char('v'), _) => app.focus = Focus::Detail,
        (KeyCode::Char('l'), _) | (KeyCode::Esc, _) => app.focus = Focus::List,
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
        (KeyCode::Enter, _) | (KeyCode::Char('g'), _) => {
            if let Some(selected) = app.selected_session() {
                return Ok(KeyOutcome::Select(selected));
            }
            app.status = "no session selected".to_string();
        }
        (KeyCode::Up, KeyModifiers::CONTROL) if app.layout_mode == LayoutMode::Short => {
            app.top_percent = app.top_percent.saturating_sub(5).max(MIN_TOP_PERCENT);
        }
        (KeyCode::Down, KeyModifiers::CONTROL) if app.layout_mode == LayoutMode::Short => {
            app.top_percent = app.top_percent.saturating_add(5).min(MAX_TOP_PERCENT);
        }
        (KeyCode::Left, KeyModifiers::CONTROL) if app.layout_mode == LayoutMode::Normal => {
            app.left_percent = app.left_percent.saturating_sub(5).max(MIN_LEFT_PERCENT);
        }
        (KeyCode::Right, KeyModifiers::CONTROL) if app.layout_mode == LayoutMode::Normal => {
            app.left_percent = app.left_percent.saturating_add(5).min(MAX_LEFT_PERCENT);
        }
        (KeyCode::Up, _) => {
            if app.focus == Focus::List {
                if app.move_selection(-1) {
                    stop_detail_source(app).await;
                    app.detail_source = DetailSource::sample();
                    refresh_detail(host, app, true).await?;
                }
            } else {
                app.scroll_detail(-1);
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
                app.scroll_detail(1);
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
        LayoutMode::Short => draw_short(frame, chunks[0], app),
    }
    draw_status(frame, chunks[1], app);
    if let Some(modal) = &app.modal {
        draw_modal(frame, area, modal);
    }
}

fn draw_normal(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
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
    draw_sessions(frame, left[1], app, " Sessions ");
    draw_detail(frame, columns[1], app, " Detail ");
}

fn draw_short(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(app.top_percent),
            Constraint::Percentage(100 - app.top_percent),
        ])
        .split(area);
    draw_sessions(frame, rows[0], app, " Sessions ");
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
        1
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
        "motlie - no /etc/motd".to_string()
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

fn draw_sessions(frame: &mut Frame<'_>, area: Rect, app: &AppState, title: &str) {
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
        .title(title)
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::List));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_detail(frame: &mut Frame<'_>, area: Rect, app: &AppState, title: &str) {
    let height = area.height.saturating_sub(2) as usize;
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
    let title = match app.detail_source.mode() {
        DetailMode::Sample => title.to_string(),
        DetailMode::Monitor => " Detail - monitor ".to_string(),
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::Detail));
    frame.render_widget(
        Paragraph::new(visible)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let focus = match app.focus {
        Focus::List => "list",
        Focus::Detail => "detail",
    };
    let mode = match app.layout_mode {
        LayoutMode::Normal => "normal",
        LayoutMode::Short => "short",
    };
    let keys = if app.layout_mode == LayoutMode::Short {
        "keys: up/down select | v/l focus | m monitor | n new | k kill | enter/g attach | ctrl-up/down resize | ctrl-c quit"
    } else {
        "keys: up/down select | v/l focus | m monitor | n new | k kill | enter/g attach | ctrl-left/right resize | ctrl-c quit"
    };
    let text = format!(
        " {} | {} | {} | {} | {} ",
        app.host_label,
        chrono::Local::now().format("%H:%M:%S"),
        mode,
        focus,
        keys
    );
    let paragraph = Paragraph::new(Line::from(vec![
        TuiSpan::styled(text, Style::default().fg(Color::Black).bg(Color::White)),
        TuiSpan::styled(
            format!(" | {}", app.status),
            Style::default().fg(Color::Yellow).bg(Color::Black),
        ),
    ]));
    frame.render_widget(paragraph, area);
}

fn draw_modal(frame: &mut Frame<'_>, area: Rect, modal: &ModalState) {
    let width = min(60, area.width.saturating_sub(4));
    let height = 7;
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    let rect = Rect::new(x, y, width, height);
    frame.render_widget(Clear, rect);

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
    };
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

    #[test]
    fn cli_rejects_print_session_with_dashboard() {
        let result = Cli::try_parse_from(["tmux_select", "--print-session", "--dashboard"]);
        assert!(result.is_err());
        Cli::command().debug_assert();
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
    fn motd_placeholder_uses_compact_text_when_narrow() {
        let app = AppState::new(
            "host".to_string(),
            LayoutMode::Normal,
            MOTLIE_PLACEHOLDER.to_string(),
            true,
        );
        let text = motd_render_text(&app, Rect::new(0, 0, 40, 5));
        assert_eq!(text, "motlie - no /etc/motd");
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
            handle: None,
            session_id: Some("$1".to_string()),
        }));
        app.detail_lines = vec!["live".to_string()];

        let closed = stop_monitor_if_closed(&mut app, "$1", "dev".to_string()).await;

        assert_eq!(closed, Some("dev".to_string()));
        assert_eq!(app.detail_source.mode(), DetailMode::Sample);
        assert!(app.detail_lines.is_empty());
    }
}
