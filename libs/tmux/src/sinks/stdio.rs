//! StdioSink — default reference implementation (DC12).
//!
//! Writes TargetOutput to stdout with configurable formatting.
//! Immediate flush, no batching.

use anyhow::Result;
use std::io::Write;

use crate::sink::SinkEvent;

/// Output format for StdioSink.
#[derive(Debug, Clone, Copy)]
pub enum StdioFormat {
    /// Raw content only, no metadata prefix.
    Raw,
    /// "[host] session:window.pane | content"
    Prefixed,
    /// JSON lines: {"host":"...","target":"...","content":"...","seq":N}
    Json,
}

/// Default reference sink: writes to stdout.
pub struct StdioSink {
    format: StdioFormat,
}

impl StdioSink {
    pub fn new(format: StdioFormat) -> Self {
        StdioSink { format }
    }

    pub fn name(&self) -> &str {
        "stdio"
    }

    pub async fn write(&mut self, event: SinkEvent) -> Result<()> {
        match event {
            SinkEvent::Data(output) => {
                let line = match self.format {
                    StdioFormat::Raw => output.content.clone(),
                    StdioFormat::Prefixed => {
                        format!(
                            "[{}] {} | {}",
                            output.host,
                            output.target_string(),
                            output.content
                        )
                    }
                    StdioFormat::Json => {
                        format!(
                            r#"{{"host":"{}","target":"{}","content":"{}","seq":{}}}"#,
                            output.host,
                            output.target_string(),
                            output.content.replace('\\', "\\\\").replace('"', "\\\""),
                            output.sequence
                        )
                    }
                };
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                writeln!(handle, "{}", line)?;
                handle.flush()?;
                Ok(())
            }
            SinkEvent::Gap { dropped, .. } => {
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                writeln!(handle, "[gap: {} events dropped]", dropped)?;
                handle.flush()?;
                Ok(())
            }
        }
    }

    pub async fn flush(&mut self) -> Result<()> {
        std::io::stdout().flush()?;
        Ok(())
    }
}
