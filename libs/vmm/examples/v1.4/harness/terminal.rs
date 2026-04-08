use std::cmp::min;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use motlie_vmm::ssh::{
    GuestPtySession, PtyRead, PtyRequest, PtyTranscriptEvent, PtyTranscriptEventKind, SshProxyError,
};
use serde::Serialize;
use thiserror::Error;
use tokio::time::Instant;

const DEFAULT_SCROLLBACK: usize = 2_000;
const DEFAULT_CHUNK_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Serialize)]
pub struct VteScreenSnapshot {
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

pub struct HarnessTerminalSession {
    name: String,
    inner: GuestPtySession,
    parser: Mutex<vt100::Parser>,
    term: String,
    command: Option<String>,
    initial_cols: u16,
    initial_rows: u16,
    recorded_at_unix: u64,
    transcript_path: PathBuf,
    screen_path: PathBuf,
    asciicast_path: PathBuf,
}

impl std::fmt::Debug for HarnessTerminalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessTerminalSession")
            .field("name", &self.name)
            .field("transcript_path", &self.transcript_path)
            .field("screen_path", &self.screen_path)
            .field("asciicast_path", &self.asciicast_path)
            .finish_non_exhaustive()
    }
}

impl HarnessTerminalSession {
    pub fn new(
        name: impl Into<String>,
        inner: GuestPtySession,
        request: &PtyRequest,
        transcript_path: PathBuf,
        screen_path: PathBuf,
        asciicast_path: PathBuf,
    ) -> Self {
        Self {
            name: name.into(),
            inner,
            parser: Mutex::new(vt100::Parser::new(
                request.row_height.try_into().unwrap_or(u16::MAX),
                request.col_width.try_into().unwrap_or(u16::MAX),
                DEFAULT_SCROLLBACK,
            )),
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
            asciicast_path,
        }
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
        let mut parser = self
            .parser
            .lock()
            .map_err(|_| TerminalSessionError::StatePoisoned)?;
        parser.screen_mut().set_size(
            row_height.try_into().unwrap_or(u16::MAX),
            col_width.try_into().unwrap_or(u16::MAX),
        );
        Ok(())
    }

    pub async fn read_for(&self, timeout: Duration) -> Result<PtyRead, TerminalSessionError> {
        let read = self.inner.read_for(timeout).await?;
        self.apply_bytes(&read.bytes)?;
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
        let parser = self
            .parser
            .lock()
            .map_err(|_| TerminalSessionError::StatePoisoned)?;
        let screen = parser.screen();
        let contents = screen.contents();
        let visible_lines = contents.lines().map(str::to_string).collect::<Vec<_>>();
        let (cursor_row, cursor_col) = screen.cursor_position();
        let (rows, cols) = screen.size();
        Ok(VteScreenSnapshot {
            rows,
            cols,
            cursor_row,
            cursor_col,
            visible_text: contents,
            visible_lines,
        })
    }

    pub fn persist_artifacts(&self) -> Result<(), TerminalSessionError> {
        let transcript = self.inner.transcript()?;
        let snapshot = self.snapshot()?;
        persist_transcript_ndjson(&self.transcript_path, &transcript)?;
        persist_screen_json(&self.screen_path, &snapshot)?;
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

    fn apply_bytes(&self, bytes: &[u8]) -> Result<(), TerminalSessionError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut parser = self
            .parser
            .lock()
            .map_err(|_| TerminalSessionError::StatePoisoned)?;
        parser.process(bytes);
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

fn excerpt(output: &str) -> String {
    const LIMIT: usize = 160;
    let normalized = output.replace('\n', "\\n");
    if normalized.len() <= LIMIT {
        normalized
    } else {
        format!("{}...", &normalized[..LIMIT])
    }
}
