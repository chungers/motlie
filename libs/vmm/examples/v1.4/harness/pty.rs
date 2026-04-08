use std::cmp::min;
use std::path::PathBuf;
use std::time::Duration;

use motlie_vmm::orchestrator::{OrchestratorError, VmHandle};
use motlie_vmm::ssh::{GuestPtySession, PtyRead, PtyRequest, PtyTranscriptEvent};
use serde::Serialize;
use thiserror::Error;
use tokio::time::Instant;

#[derive(Debug, Serialize)]
pub struct PtyCheck {
    pub name: String,
    pub expected: String,
    pub observed_excerpt: String,
}

#[derive(Debug, Serialize)]
pub struct PtyTranscriptSummary {
    pub event_count: usize,
    pub sent_events: usize,
    pub received_events: usize,
    pub resized_events: usize,
    pub eof_seen: bool,
    pub close_seen: bool,
    pub exit_status: Option<u32>,
    pub bytes_sent: usize,
    pub bytes_received: usize,
}

#[derive(Debug, Serialize)]
pub struct PtyScenarioResult {
    pub transcript_path: PathBuf,
    pub transcript_summary: PtyTranscriptSummary,
    pub checks: Vec<PtyCheck>,
}

pub struct PtyScenarioRun {
    pub result: PtyScenarioResult,
    pub transcript: Vec<PtyTranscriptEvent>,
}

#[derive(Debug, Error)]
pub enum PtyScenarioError {
    #[error(transparent)]
    Open(#[from] OrchestratorError),
    #[error(transparent)]
    ControlPlane(#[from] motlie_vmm::ssh::SshProxyError),
    #[error("PTY step '{step}' expected {expected}, got: {observed_excerpt}")]
    Assertion {
        step: &'static str,
        expected: String,
        observed_excerpt: String,
    },
    #[error("PTY transcript was empty after the scenario completed")]
    EmptyTranscript,
    #[error("PTY transcript did not include a terminal close/eof/exit-status event")]
    IncompleteTranscript,
}

pub async fn run_pty_smoke(
    handle: &VmHandle,
    transcript_path: PathBuf,
) -> Result<PtyScenarioRun, PtyScenarioError> {
    let pty = handle
        .open_pty(PtyRequest::default(), Duration::from_secs(10))
        .await?;
    let mut checks = Vec::new();

    let login = read_until_contains(
        &pty,
        "login_banner",
        "Start tmux session?",
        Duration::from_secs(20),
    )
    .await?;
    checks.push(check_contains(
        "motd",
        "v1.4 extraction / agent-state demo",
        &login.output,
    )?);

    pty.send_line("n").await?;
    let shell = read_until_contains(
        &pty,
        "shell_prompt",
        "alice@motlie-alice",
        Duration::from_secs(10),
    )
    .await?;
    checks.push(check_contains(
        "shell_prompt",
        "alice@motlie-alice",
        &shell.output,
    )?);

    pty.send_line("pwd").await?;
    let pwd = read_until_contains(&pty, "pwd", "/home/alice", Duration::from_secs(10)).await?;
    checks.push(check_contains("pwd", "/home/alice", &pwd.output)?);

    pty.resize(120, 40, 0, 0).await?;
    pty.send_line("stty size").await?;
    let size = read_until_contains(&pty, "resize_ack", "40 120", Duration::from_secs(10)).await?;
    checks.push(check_contains("resize_ack", "40 120", &size.output)?);

    pty.send_line("exit").await?;
    let close = read_until_terminal(&pty, Duration::from_secs(5)).await?;
    let transcript = pty.transcript()?;
    if transcript.is_empty() {
        return Err(PtyScenarioError::EmptyTranscript);
    }
    let transcript_summary = summarize_transcript(&transcript);
    if !transcript_summary.close_seen
        && !transcript_summary.eof_seen
        && transcript_summary.exit_status.is_none()
    {
        return Err(PtyScenarioError::IncompleteTranscript);
    }
    checks.push(PtyCheck {
        name: "terminal_close".to_string(),
        expected: "exit status, eof, or close event after exit".to_string(),
        observed_excerpt: excerpt(&close.output),
    });

    Ok(PtyScenarioRun {
        result: PtyScenarioResult {
            transcript_path,
            transcript_summary,
            checks,
        },
        transcript,
    })
}

fn check_contains(
    name: &'static str,
    expected: &str,
    output: &str,
) -> Result<PtyCheck, PtyScenarioError> {
    if output.contains(expected) {
        Ok(PtyCheck {
            name: name.to_string(),
            expected: format!("output containing '{expected}'"),
            observed_excerpt: excerpt(output),
        })
    } else {
        Err(PtyScenarioError::Assertion {
            step: name,
            expected: format!("output containing '{expected}'"),
            observed_excerpt: excerpt(output),
        })
    }
}

async fn read_until_contains(
    pty: &GuestPtySession,
    step: &'static str,
    needle: &str,
    timeout: Duration,
) -> Result<PtyRead, PtyScenarioError> {
    let deadline = Instant::now() + timeout;
    let mut combined = PtyRead::default();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(PtyScenarioError::Assertion {
                step,
                expected: format!("output containing '{needle}'"),
                observed_excerpt: excerpt(&combined.output),
            });
        }

        let chunk = pty
            .read_for(min(remaining, Duration::from_millis(500)))
            .await?;
        combined.output.push_str(&chunk.output);
        combined.exit_status = chunk.exit_status.or(combined.exit_status);
        combined.eof |= chunk.eof;
        combined.closed |= chunk.closed;

        if combined.output.contains(needle) {
            return Ok(combined);
        }
        if combined.eof || combined.closed {
            return Err(PtyScenarioError::Assertion {
                step,
                expected: format!("output containing '{needle}'"),
                observed_excerpt: excerpt(&combined.output),
            });
        }
    }
}

async fn read_until_terminal(
    pty: &GuestPtySession,
    timeout: Duration,
) -> Result<PtyRead, PtyScenarioError> {
    let deadline = Instant::now() + timeout;
    let mut combined = PtyRead::default();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(combined);
        }

        let chunk = pty
            .read_for(min(remaining, Duration::from_millis(500)))
            .await?;
        combined.output.push_str(&chunk.output);
        combined.exit_status = chunk.exit_status.or(combined.exit_status);
        combined.eof |= chunk.eof;
        combined.closed |= chunk.closed;
        if combined.eof || combined.closed || combined.exit_status.is_some() {
            return Ok(combined);
        }
    }
}

fn summarize_transcript(events: &[PtyTranscriptEvent]) -> PtyTranscriptSummary {
    let mut summary = PtyTranscriptSummary {
        event_count: events.len(),
        sent_events: 0,
        received_events: 0,
        resized_events: 0,
        eof_seen: false,
        close_seen: false,
        exit_status: None,
        bytes_sent: 0,
        bytes_received: 0,
    };

    for event in events {
        match event {
            PtyTranscriptEvent::Sent(bytes) => {
                summary.sent_events += 1;
                summary.bytes_sent += bytes.len();
            }
            PtyTranscriptEvent::Received(bytes) => {
                summary.received_events += 1;
                summary.bytes_received += bytes.len();
            }
            PtyTranscriptEvent::Resized { .. } => {
                summary.resized_events += 1;
            }
            PtyTranscriptEvent::ExitStatus(status) => {
                summary.exit_status = Some(*status);
            }
            PtyTranscriptEvent::Eof => {
                summary.eof_seen = true;
            }
            PtyTranscriptEvent::Close => {
                summary.close_seen = true;
            }
        }
    }

    summary
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
