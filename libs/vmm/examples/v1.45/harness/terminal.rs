use std::cmp::min;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use motlie_vmm::ssh::{
    GuestPtySession, PtyRead, PtyRequest, PtyTranscriptEvent, PtyTranscriptEventKind, SshProxyError,
};
use serde::{Deserialize, Serialize};
use shadow_terminal::wezterm_term;
use thiserror::Error;
use tokio::time::Instant;

const DEFAULT_SCROLLBACK: usize = 2_000;
const DEFAULT_CHUNK_TIMEOUT: Duration = Duration::from_millis(500);
const CURSOR_POSITION_REQUEST: &[u8] = b"\x1b[6n";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalBackendKind {
    Vt100,
    #[default]
    Shadow,
}

impl std::fmt::Display for TerminalBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vt100 => write!(f, "vt100"),
            Self::Shadow => write!(f, "shadow"),
        }
    }
}

impl std::str::FromStr for TerminalBackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "vt100" => Ok(Self::Vt100),
            "shadow" => Ok(Self::Shadow),
            other => Err(format!(
                "unsupported terminal backend '{other}' (expected 'vt100' or 'shadow')"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalScreenMode {
    Primary,
    Alternate,
}

#[derive(Debug, Clone, Serialize)]
pub struct VteScreenSnapshot {
    pub backend: TerminalBackendKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen_mode: Option<TerminalScreenMode>,
    pub rows: u16,
    pub cols: u16,
    pub cursor_row: u16,
    pub cursor_col: u16,
    pub visible_text: String,
    pub visible_lines: Vec<String>,
}

#[derive(Debug, Error)]
pub enum TerminalSessionError {
    #[error(transparent)]
    ControlPlane(#[from] SshProxyError),
    #[error("terminal state poisoned")]
    StatePoisoned,
    #[error("failed to persist terminal artifact {path}: {reason}")]
    Persist { path: PathBuf, reason: String },
    #[error("PTY step '{step}' expected {expected}, got: {observed_excerpt}")]
    Assertion {
        step: &'static str,
        expected: String,
        observed_excerpt: String,
    },
}

struct TerminalFeedResult {
    auto_responses: Vec<Vec<u8>>,
}

enum TerminalEmulator {
    Vt100(Box<Vt100Engine>),
    Shadow(Box<ShadowEngine>),
}

impl TerminalEmulator {
    fn new(kind: TerminalBackendKind, request: &PtyRequest) -> Self {
        let rows = request.row_height.try_into().unwrap_or(u16::MAX);
        let cols = request.col_width.try_into().unwrap_or(u16::MAX);
        match kind {
            TerminalBackendKind::Vt100 => Self::Vt100(Box::new(Vt100Engine::new(rows, cols))),
            TerminalBackendKind::Shadow => Self::Shadow(Box::new(ShadowEngine::new(rows, cols))),
        }
    }

    fn resize(&mut self, rows: u16, cols: u16) {
        match self {
            Self::Vt100(engine) => engine.resize(rows, cols),
            Self::Shadow(engine) => engine.resize(rows, cols),
        }
    }

    fn apply_bytes(&mut self, bytes: &[u8]) -> TerminalFeedResult {
        match self {
            Self::Vt100(engine) => engine.apply_bytes(bytes),
            Self::Shadow(engine) => engine.apply_bytes(bytes),
        }
    }

    fn snapshot(&self) -> VteScreenSnapshot {
        match self {
            Self::Vt100(engine) => engine.snapshot(),
            Self::Shadow(engine) => engine.snapshot(),
        }
    }
}

struct Vt100Engine {
    parser: vt100::Parser,
}

impl Vt100Engine {
    fn new(rows: u16, cols: u16) -> Self {
        Self {
            parser: vt100::Parser::new(rows, cols, DEFAULT_SCROLLBACK),
        }
    }

    fn resize(&mut self, rows: u16, cols: u16) {
        self.parser.screen_mut().set_size(rows, cols);
    }

    fn apply_bytes(&mut self, bytes: &[u8]) -> TerminalFeedResult {
        self.parser.process(bytes);
        TerminalFeedResult {
            auto_responses: Vec::new(),
        }
    }

    fn snapshot(&self) -> VteScreenSnapshot {
        let screen = self.parser.screen();
        let contents = screen.contents();
        let visible_lines = contents.lines().map(str::to_string).collect::<Vec<_>>();
        let (cursor_row, cursor_col) = screen.cursor_position();
        let (rows, cols) = screen.size();
        VteScreenSnapshot {
            backend: TerminalBackendKind::Vt100,
            screen_mode: None,
            rows,
            cols,
            cursor_row,
            cursor_col,
            visible_text: contents,
            visible_lines,
        }
    }
}

#[derive(Debug)]
struct HarnessWeztermConfig {
    scrollback: usize,
}

impl wezterm_term::TerminalConfiguration for HarnessWeztermConfig {
    fn scrollback_size(&self) -> usize {
        self.scrollback
    }

    fn color_palette(&self) -> wezterm_term::color::ColorPalette {
        wezterm_term::color::ColorPalette::default()
    }
}

struct ShadowEngine {
    terminal: wezterm_term::Terminal,
}

impl ShadowEngine {
    fn new(rows: u16, cols: u16) -> Self {
        let terminal = wezterm_term::Terminal::new(
            wezterm_size(cols.into(), rows.into()),
            Arc::new(HarnessWeztermConfig {
                scrollback: DEFAULT_SCROLLBACK,
            }),
            "motlie-vmm",
            "v1.4",
            Box::<Vec<u8>>::default(),
        );
        Self { terminal }
    }

    fn resize(&mut self, rows: u16, cols: u16) {
        self.terminal.resize(wezterm_size(cols.into(), rows.into()));
    }

    fn apply_bytes(&mut self, bytes: &[u8]) -> TerminalFeedResult {
        let mut auto_responses = Vec::new();
        let mut cursor = 0usize;

        while let Some(relative_match) = find_subslice(&bytes[cursor..], CURSOR_POSITION_REQUEST) {
            let request_start = cursor + relative_match;
            if request_start > cursor {
                self.terminal.advance_bytes(&bytes[cursor..request_start]);
            }
            auto_responses.push(self.cursor_position_response());
            self.terminal.advance_bytes(CURSOR_POSITION_REQUEST);
            cursor = request_start + CURSOR_POSITION_REQUEST.len();
        }

        if cursor < bytes.len() {
            self.terminal.advance_bytes(&bytes[cursor..]);
        }

        TerminalFeedResult { auto_responses }
    }

    fn snapshot(&self) -> VteScreenSnapshot {
        let size = self.terminal.get_size();
        let mut screen = self.terminal.screen().clone();
        let mut visible_lines = Vec::with_capacity(size.rows);
        for row in 0..size.rows {
            let mut line = String::new();
            for col in 0..size.cols {
                if let Some(cell) = screen.get_cell(col, row.try_into().unwrap_or(i64::MAX)) {
                    line.push_str(cell.str());
                }
            }
            visible_lines.push(line);
        }
        let cursor = self.terminal.cursor_pos();
        VteScreenSnapshot {
            backend: TerminalBackendKind::Shadow,
            screen_mode: Some(if self.terminal.is_alt_screen_active() {
                TerminalScreenMode::Alternate
            } else {
                TerminalScreenMode::Primary
            }),
            rows: size.rows.try_into().unwrap_or(u16::MAX),
            cols: size.cols.try_into().unwrap_or(u16::MAX),
            cursor_row: cursor.y.try_into().unwrap_or(u16::MAX),
            cursor_col: cursor.x.try_into().unwrap_or(u16::MAX),
            visible_text: visible_lines.join("\n"),
            visible_lines,
        }
    }

    fn cursor_position_response(&self) -> Vec<u8> {
        let cursor = self.terminal.cursor_pos();
        format!("\x1b[{};{}R", cursor.y + 1, cursor.x + 1).into_bytes()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

const fn wezterm_size(cols: usize, rows: usize) -> wezterm_term::TerminalSize {
    wezterm_term::TerminalSize {
        cols,
        rows,
        pixel_width: 0,
        pixel_height: 0,
        dpi: 0,
    }
}

pub struct HarnessTerminalSession {
    name: String,
    inner: GuestPtySession,
    emulator: Mutex<TerminalEmulator>,
    backend: TerminalBackendKind,
    term: String,
    command: Option<String>,
    initial_cols: u16,
    initial_rows: u16,
    recorded_at_unix: u64,
    transcript_path: PathBuf,
    screen_path: PathBuf,
    screen_svg_path: PathBuf,
    asciicast_path: PathBuf,
}

impl std::fmt::Debug for HarnessTerminalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessTerminalSession")
            .field("name", &self.name)
            .field("backend", &self.backend)
            .field("transcript_path", &self.transcript_path)
            .field("screen_path", &self.screen_path)
            .field("screen_svg_path", &self.screen_svg_path)
            .field("asciicast_path", &self.asciicast_path)
            .finish_non_exhaustive()
    }
}

impl HarnessTerminalSession {
    #[allow(clippy::too_many_arguments)] // The harness wires one PTY session to four artifact sinks.
    pub fn new(
        name: impl Into<String>,
        inner: GuestPtySession,
        request: &PtyRequest,
        backend: TerminalBackendKind,
        transcript_path: PathBuf,
        screen_path: PathBuf,
        screen_svg_path: PathBuf,
        asciicast_path: PathBuf,
    ) -> Self {
        Self {
            name: name.into(),
            inner,
            emulator: Mutex::new(TerminalEmulator::new(backend, request)),
            backend,
            term: request.term.clone(),
            command: request.command.clone(),
            initial_cols: request.col_width.try_into().unwrap_or(u16::MAX),
            initial_rows: request.row_height.try_into().unwrap_or(u16::MAX),
            recorded_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            transcript_path,
            screen_path,
            screen_svg_path,
            asciicast_path,
        }
    }

    pub fn backend(&self) -> TerminalBackendKind {
        self.backend
    }

    pub async fn send(&self, data: &[u8]) -> Result<(), TerminalSessionError> {
        self.inner.send(data).await?;
        Ok(())
    }

    pub async fn send_line(&self, line: &str) -> Result<(), TerminalSessionError> {
        self.inner.send_line(line).await?;
        Ok(())
    }

    pub async fn resize(
        &self,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    ) -> Result<(), TerminalSessionError> {
        self.inner
            .resize(col_width, row_height, pix_width, pix_height)
            .await?;
        let mut emulator = self
            .emulator
            .lock()
            .map_err(|_| TerminalSessionError::StatePoisoned)?;
        emulator.resize(
            row_height.try_into().unwrap_or(u16::MAX),
            col_width.try_into().unwrap_or(u16::MAX),
        );
        Ok(())
    }

    pub async fn read_for(&self, timeout: Duration) -> Result<PtyRead, TerminalSessionError> {
        let read = self.inner.read_for(timeout).await?;
        self.apply_bytes(&read.bytes).await?;
        Ok(read)
    }

    pub async fn read_until_contains(
        &self,
        step: &'static str,
        needle: &str,
        timeout: Duration,
    ) -> Result<PtyRead, TerminalSessionError> {
        let deadline = Instant::now() + timeout;
        let mut combined = PtyRead::default();

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(TerminalSessionError::Assertion {
                    step,
                    expected: format!("output containing '{needle}'"),
                    observed_excerpt: excerpt(&combined.output),
                });
            }

            let chunk = self.read_for(min(remaining, DEFAULT_CHUNK_TIMEOUT)).await?;
            combined.bytes.extend_from_slice(&chunk.bytes);
            combined.output.push_str(&chunk.output);
            combined.exit_status = chunk.exit_status.or(combined.exit_status);
            combined.eof |= chunk.eof;
            combined.closed |= chunk.closed;

            if combined.output.contains(needle) {
                return Ok(combined);
            }
            if combined.eof || combined.closed {
                return Err(TerminalSessionError::Assertion {
                    step,
                    expected: format!("output containing '{needle}'"),
                    observed_excerpt: excerpt(&combined.output),
                });
            }
        }
    }

    pub async fn read_until_screen_contains(
        &self,
        step: &'static str,
        needle: &str,
        timeout: Duration,
    ) -> Result<PtyRead, TerminalSessionError> {
        let deadline = Instant::now() + timeout;
        let mut combined = PtyRead::default();

        let initial_screen = self.snapshot()?;
        if initial_screen.visible_text.contains(needle) {
            return Ok(combined);
        }

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let screen = self.snapshot()?;
                return Err(TerminalSessionError::Assertion {
                    step,
                    expected: format!("screen containing '{needle}'"),
                    observed_excerpt: excerpt(&screen.visible_text),
                });
            }

            let chunk = self.read_for(min(remaining, DEFAULT_CHUNK_TIMEOUT)).await?;
            combined.bytes.extend_from_slice(&chunk.bytes);
            combined.output.push_str(&chunk.output);
            combined.exit_status = chunk.exit_status.or(combined.exit_status);
            combined.eof |= chunk.eof;
            combined.closed |= chunk.closed;

            let screen = self.snapshot()?;
            if screen.visible_text.contains(needle) {
                return Ok(combined);
            }
            if combined.eof || combined.closed {
                return Err(TerminalSessionError::Assertion {
                    step,
                    expected: format!("screen containing '{needle}'"),
                    observed_excerpt: excerpt(&screen.visible_text),
                });
            }
        }
    }

    pub async fn read_until_terminal(
        &self,
        timeout: Duration,
    ) -> Result<PtyRead, TerminalSessionError> {
        let deadline = Instant::now() + timeout;
        let mut combined = PtyRead::default();

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(combined);
            }

            let chunk = self.read_for(min(remaining, DEFAULT_CHUNK_TIMEOUT)).await?;
            combined.bytes.extend_from_slice(&chunk.bytes);
            combined.output.push_str(&chunk.output);
            combined.exit_status = chunk.exit_status.or(combined.exit_status);
            combined.eof |= chunk.eof;
            combined.closed |= chunk.closed;
            if combined.eof || combined.closed || combined.exit_status.is_some() {
                return Ok(combined);
            }
        }
    }

    pub fn snapshot(&self) -> Result<VteScreenSnapshot, TerminalSessionError> {
        let emulator = self
            .emulator
            .lock()
            .map_err(|_| TerminalSessionError::StatePoisoned)?;
        Ok(emulator.snapshot())
    }

    pub fn persist_artifacts(&self) -> Result<(), TerminalSessionError> {
        let transcript = self.inner.transcript()?;
        let snapshot = self.snapshot()?;
        persist_transcript_ndjson(&self.transcript_path, &transcript)?;
        persist_screen_json(&self.screen_path, &snapshot)?;
        persist_screen_svg(&self.screen_svg_path, &snapshot)?;
        persist_asciicast(
            &self.asciicast_path,
            &self.name,
            &self.term,
            self.command.as_deref(),
            self.initial_cols,
            self.initial_rows,
            self.recorded_at_unix,
            &transcript,
        )?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<PtyTranscriptEvent>, TerminalSessionError> {
        self.inner.transcript().map_err(Into::into)
    }

    pub fn transcript_path(&self) -> &Path {
        &self.transcript_path
    }

    pub fn screen_path(&self) -> &Path {
        &self.screen_path
    }

    pub fn asciicast_path(&self) -> &Path {
        &self.asciicast_path
    }

    pub fn screen_svg_path(&self) -> &Path {
        &self.screen_svg_path
    }

    async fn apply_bytes(&self, bytes: &[u8]) -> Result<(), TerminalSessionError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let auto_responses = {
            let mut emulator = self
                .emulator
                .lock()
                .map_err(|_| TerminalSessionError::StatePoisoned)?;
            emulator.apply_bytes(bytes).auto_responses
        };
        for response in auto_responses {
            self.inner.send(&response).await?;
        }
        Ok(())
    }
}

pub fn persist_transcript_ndjson(
    path: &Path,
    transcript: &[PtyTranscriptEvent],
) -> Result<(), TerminalSessionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| TerminalSessionError::Persist {
            path: parent.to_path_buf(),
            reason: source.to_string(),
        })?;
    }

    let file = std::fs::File::create(path).map_err(|source| TerminalSessionError::Persist {
        path: path.to_path_buf(),
        reason: source.to_string(),
    })?;
    let mut writer = BufWriter::new(file);
    for event in transcript {
        serde_json::to_writer(&mut writer, event).map_err(|source| {
            TerminalSessionError::Persist {
                path: path.to_path_buf(),
                reason: source.to_string(),
            }
        })?;
        writer
            .write_all(b"\n")
            .map_err(|source| TerminalSessionError::Persist {
                path: path.to_path_buf(),
                reason: source.to_string(),
            })?;
    }
    writer
        .flush()
        .map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct AsciicastHeader<'a> {
    version: u8,
    term: AsciicastTerm<'a>,
    timestamp: u64,
    title: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct AsciicastTerm<'a> {
    cols: u16,
    rows: u16,
    #[serde(rename = "type")]
    term_type: &'a str,
}

#[allow(clippy::too_many_arguments)] // Asciicast v2 metadata is flat by file-format design.
pub fn persist_asciicast(
    path: &Path,
    title: &str,
    term: &str,
    command: Option<&str>,
    initial_cols: u16,
    initial_rows: u16,
    recorded_at_unix: u64,
    transcript: &[PtyTranscriptEvent],
) -> Result<(), TerminalSessionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| TerminalSessionError::Persist {
            path: parent.to_path_buf(),
            reason: source.to_string(),
        })?;
    }

    let file = std::fs::File::create(path).map_err(|source| TerminalSessionError::Persist {
        path: path.to_path_buf(),
        reason: source.to_string(),
    })?;
    let mut writer = BufWriter::new(file);

    let header = AsciicastHeader {
        version: 3,
        term: AsciicastTerm {
            cols: initial_cols,
            rows: initial_rows,
            term_type: term,
        },
        timestamp: recorded_at_unix,
        title,
        command,
    };
    serde_json::to_writer(&mut writer, &header).map_err(|source| {
        TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        }
    })?;
    writer
        .write_all(b"\n")
        .map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;

    let mut previous_offset_ms = 0u64;
    for event in transcript {
        let delta_ms = event.offset_ms.saturating_sub(previous_offset_ms);
        previous_offset_ms = event.offset_ms;
        let time = delta_ms as f64 / 1000.0;

        match &event.event {
            PtyTranscriptEventKind::Sent { data } => write_asciicast_event(
                &mut writer,
                path,
                time,
                "i",
                String::from_utf8_lossy(data).as_ref(),
            )?,
            PtyTranscriptEventKind::Received { data } => write_asciicast_event(
                &mut writer,
                path,
                time,
                "o",
                String::from_utf8_lossy(data).as_ref(),
            )?,
            PtyTranscriptEventKind::Resized {
                col_width,
                row_height,
                ..
            } => write_asciicast_event(
                &mut writer,
                path,
                time,
                "r",
                &format!("{col_width}x{row_height}"),
            )?,
            PtyTranscriptEventKind::ExitStatus { exit_status } => {
                write_asciicast_event(&mut writer, path, time, "x", &exit_status.to_string())?
            }
            PtyTranscriptEventKind::Eof | PtyTranscriptEventKind::Close => {}
        }
    }

    writer
        .flush()
        .map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    Ok(())
}

fn write_asciicast_event(
    writer: &mut BufWriter<std::fs::File>,
    path: &Path,
    time: f64,
    code: &str,
    data: &str,
) -> Result<(), TerminalSessionError> {
    serde_json::to_writer(&mut *writer, &(time, code, data)).map_err(|source| {
        TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        }
    })?;
    writer
        .write_all(b"\n")
        .map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    Ok(())
}

pub fn persist_screen_json(
    path: &Path,
    snapshot: &VteScreenSnapshot,
) -> Result<(), TerminalSessionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| TerminalSessionError::Persist {
            path: parent.to_path_buf(),
            reason: source.to_string(),
        })?;
    }
    let bytes =
        serde_json::to_vec_pretty(snapshot).map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    std::fs::write(path, bytes).map_err(|source| TerminalSessionError::Persist {
        path: path.to_path_buf(),
        reason: source.to_string(),
    })?;
    Ok(())
}

pub fn persist_screen_svg(
    path: &Path,
    snapshot: &VteScreenSnapshot,
) -> Result<(), TerminalSessionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| TerminalSessionError::Persist {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    }
    let svg = render_screen_svg(snapshot);
    std::fs::write(path, svg).map_err(|source| TerminalSessionError::Persist {
        path: path.to_path_buf(),
        reason: source.to_string(),
    })?;
    Ok(())
}

fn render_screen_svg(snapshot: &VteScreenSnapshot) -> String {
    const FONT_SIZE: usize = 16;
    const LINE_HEIGHT: usize = 22;
    const CELL_WIDTH: usize = 10;
    const PADDING_X: usize = 18;
    const PADDING_Y: usize = 18;

    let width = snapshot.cols as usize * CELL_WIDTH + PADDING_X * 2;
    let height = snapshot.rows as usize * LINE_HEIGHT + PADDING_Y * 2 + 24;
    let title = match snapshot.screen_mode {
        Some(TerminalScreenMode::Alternate) => {
            format!("{} terminal snapshot (alternate screen)", snapshot.backend)
        }
        _ => format!("{} terminal snapshot", snapshot.backend),
    };
    let escaped_title = escape_xml(&title);

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{}">"#,
        escaped_title.as_str()
    ));
    svg.push_str(
        r#"<defs><style><![CDATA[
text {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 16px;
  fill: #e8f0ea;
}
.meta {
  font-size: 13px;
  fill: #9ab0a1;
}
]]></style></defs>"#,
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#0f1412"/>"##);
    svg.push_str(&format!(
        r##"<rect x="10" y="10" width="{}" height="{}" rx="14" fill="#161d19" stroke="#2c3a33"/>"##,
        width.saturating_sub(20),
        height.saturating_sub(20)
    ));
    svg.push_str(&format!(
        r#"<text class="meta" x="{PADDING_X}" y="30">{}</text>"#,
        escaped_title.as_str()
    ));

    let cursor_x = PADDING_X + snapshot.cursor_col as usize * CELL_WIDTH;
    let cursor_y =
        PADDING_Y + 22 + snapshot.cursor_row as usize * LINE_HEIGHT - (LINE_HEIGHT - FONT_SIZE);
    if (snapshot.cursor_row as usize) < snapshot.visible_lines.len() {
        svg.push_str(&format!(
            r##"<rect x="{cursor_x}" y="{}" width="{CELL_WIDTH}" height="{LINE_HEIGHT}" fill="#83d97b" fill-opacity="0.28"/>"##,
            cursor_y.saturating_sub(FONT_SIZE)
        ));
    }

    svg.push_str(r#"<text xml:space="preserve">"#);
    for (index, line) in snapshot.visible_lines.iter().enumerate() {
        let y = PADDING_Y + 22 + index * LINE_HEIGHT;
        let escaped_line = if line.is_empty() {
            " ".to_string()
        } else {
            escape_xml(line)
        };
        svg.push_str(&format!(
            r#"<tspan x="{PADDING_X}" y="{y}">{}</tspan>"#,
            escaped_line
        ));
    }
    svg.push_str("</text></svg>");
    svg
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn excerpt(output: &str) -> String {
    const LIMIT: usize = 160;
    let normalized = output.replace('\n', "\\n");
    if normalized.len() <= LIMIT {
        normalized
    } else {
        format!("{}...", &normalized[..LIMIT])
    }
}
