use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct AsciicastMetadata {
    pub title: String,
    pub command: Option<String>,
    pub term_type: String,
    pub cols: u16,
    pub rows: u16,
}

#[derive(Debug)]
pub struct AsciicastRecorder {
    path: PathBuf,
    writer: BufWriter<File>,
    started_at: Instant,
    last_offset_s: f64,
}

#[derive(Debug, Error)]
pub enum AsciicastError {
    #[error("failed to persist asciicast artifact {path}: {reason}")]
    Persist { path: PathBuf, reason: String },
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

impl AsciicastRecorder {
    pub fn create(path: impl AsRef<Path>, meta: &AsciicastMetadata) -> Result<Self, AsciicastError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| AsciicastError::Persist {
                path: parent.to_path_buf(),
                reason: source.to_string(),
            })?;
        }

        let file = File::create(&path).map_err(|source| AsciicastError::Persist {
            path: path.clone(),
            reason: source.to_string(),
        })?;
        let mut writer = BufWriter::new(file);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let header = AsciicastHeader {
            version: 3,
            term: AsciicastTerm {
                cols: meta.cols,
                rows: meta.rows,
                term_type: &meta.term_type,
            },
            timestamp,
            title: &meta.title,
            command: meta.command.as_deref(),
        };
        serde_json::to_writer(&mut writer, &header).map_err(|source| AsciicastError::Persist {
            path: path.clone(),
            reason: source.to_string(),
        })?;
        writer
            .write_all(b"\n")
            .and_then(|_| writer.flush())
            .map_err(|source| AsciicastError::Persist {
                path: path.clone(),
                reason: source.to_string(),
            })?;

        Ok(Self {
            path,
            writer,
            started_at: Instant::now(),
            last_offset_s: 0.0,
        })
    }

    pub fn record_input(&mut self, text: &str) -> Result<(), AsciicastError> {
        self.record_event("i", text)
    }

    pub fn record_output(&mut self, text: &str) -> Result<(), AsciicastError> {
        self.record_event("o", text)
    }

    pub fn record_resize(&mut self, cols: u16, rows: u16) -> Result<(), AsciicastError> {
        self.record_event("r", &format!("{cols}x{rows}"))
    }

    pub fn flush(&mut self) -> Result<(), AsciicastError> {
        self.writer.flush().map_err(|source| AsciicastError::Persist {
            path: self.path.clone(),
            reason: source.to_string(),
        })
    }

    fn record_event(&mut self, code: &str, data: &str) -> Result<(), AsciicastError> {
        if data.is_empty() {
            return Ok(());
        }

        let offset_s = self.started_at.elapsed().as_secs_f64();
        let delta_s = (offset_s - self.last_offset_s).max(0.0);
        self.last_offset_s = offset_s;

        serde_json::to_writer(&mut self.writer, &(delta_s, code, data)).map_err(|source| {
            AsciicastError::Persist {
                path: self.path.clone(),
                reason: source.to_string(),
            }
        })?;
        self.writer
            .write_all(b"\n")
            .and_then(|_| self.writer.flush())
            .map_err(|source| AsciicastError::Persist {
                path: self.path.clone(),
                reason: source.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{AsciicastMetadata, AsciicastRecorder};

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nonce}.cast"))
    }

    #[test]
    fn recorder_writes_header_and_events() {
        let path = temp_path("motlie-driver-asciicast");
        let meta = AsciicastMetadata {
            title: "tmux-driver".to_string(),
            command: Some("motlie-tmux-driver".to_string()),
            term_type: "xterm-256color".to_string(),
            cols: 80,
            rows: 24,
        };

        let mut recorder = AsciicastRecorder::create(&path, &meta).expect("create recorder");
        recorder.record_output("Connected\n").expect("output");
        recorder.record_input("targets\n").expect("input");
        recorder.record_resize(100, 30).expect("resize");
        recorder.flush().expect("flush");

        let contents = std::fs::read_to_string(&path).expect("read cast");
        let lines = contents.lines().collect::<Vec<_>>();
        assert!(lines.len() >= 4);
        assert!(lines[0].contains("\"version\":3"));
        assert!(lines[0].contains("\"title\":\"tmux-driver\""));
        assert!(lines[1].contains("\"o\""));
        assert!(lines[2].contains("\"i\""));
        assert!(lines[3].contains("\"r\""));

        let _ = std::fs::remove_file(path);
    }
}
