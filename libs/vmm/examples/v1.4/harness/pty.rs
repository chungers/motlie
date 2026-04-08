use std::path::PathBuf;
use std::time::Duration;

use motlie_vmm::orchestrator::{OrchestratorError, VmHandle};
use motlie_vmm::ssh::{PtyRequest, PtyTranscriptEvent, PtyTranscriptEventKind};
use serde::Serialize;
use thiserror::Error;

use crate::terminal::{HarnessTerminalSession, TerminalSessionError, VteScreenSnapshot};

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
    pub screen_path: PathBuf,
    pub asciicast_path: PathBuf,
    pub final_screen: VteScreenSnapshot,
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
    Terminal(#[from] TerminalSessionError),
    #[error("PTY transcript was empty after the scenario completed")]
    EmptyTranscript,
    #[error("PTY transcript did not include a terminal close/eof/exit-status event")]
    IncompleteTranscript,
}

pub async fn run_pty_smoke(
    handle: &VmHandle,
    transcript_path: PathBuf,
    screen_path: PathBuf,
    asciicast_path: PathBuf,
) -> Result<PtyScenarioRun, PtyScenarioError> {
    let request = PtyRequest::default();
    let terminal = HarnessTerminalSession::new(
        "alice-pty-smoke",
        handle
            .open_pty(request.clone(), Duration::from_secs(10))
            .await?,
        &request,
        transcript_path.clone(),
        screen_path.clone(),
        asciicast_path.clone(),
    );
    let scenario = async {
        let mut checks = Vec::new();

        let login = terminal
            .read_until_contains(
                "login_banner",
                "Start tmux session?",
                Duration::from_secs(20),
            )
            .await?;
        checks.push(check_contains(
            "login_prompt",
            "Start tmux session?",
            &login.output,
        )?);
        checks.push(check_screen_contains(
            "login_screen",
            "Start tmux session?",
            &terminal.snapshot()?,
        )?);

        terminal.send_line("n").await?;
        let shell = terminal
            .read_until_contains(
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

        terminal.send_line("pwd").await?;
        let pwd = terminal
            .read_until_contains("pwd", "/home/alice", Duration::from_secs(10))
            .await?;
        checks.push(check_contains("pwd", "/home/alice", &pwd.output)?);

        terminal.resize(120, 40, 0, 0).await?;
        terminal.send_line("stty size").await?;
        let size = terminal
            .read_until_contains("resize_ack", "40 120", Duration::from_secs(10))
            .await?;
        checks.push(check_contains("resize_ack", "40 120", &size.output)?);

        terminal.send_line("exit").await?;
        let close = terminal.read_until_terminal(Duration::from_secs(5)).await?;
        let transcript = terminal.transcript()?;
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

        Ok::<_, PtyScenarioError>((checks, transcript_summary, transcript))
    }
    .await;

    let persist_result = terminal.persist_artifacts();
    let final_screen = terminal.snapshot()?;

    match scenario {
        Ok((checks, transcript_summary, transcript)) => {
            persist_result?;
            Ok(PtyScenarioRun {
                result: PtyScenarioResult {
                    transcript_path,
                    screen_path,
                    asciicast_path,
                    final_screen,
                    transcript_summary,
                    checks,
                },
                transcript,
            })
        }
        Err(error) => {
            let _ = persist_result;
            Err(error)
        }
    }
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
        Err(TerminalSessionError::Assertion {
            step: name,
            expected: format!("output containing '{expected}'"),
            observed_excerpt: excerpt(output),
        }
        .into())
    }
}

fn check_screen_contains(
    name: &'static str,
    expected: &str,
    screen: &VteScreenSnapshot,
) -> Result<PtyCheck, PtyScenarioError> {
    if screen.visible_text.contains(expected) {
        Ok(PtyCheck {
            name: name.to_string(),
            expected: format!("screen containing '{expected}'"),
            observed_excerpt: excerpt(&screen.visible_text),
        })
    } else {
        Err(TerminalSessionError::Assertion {
            step: name,
            expected: format!("screen containing '{expected}'"),
            observed_excerpt: excerpt(&screen.visible_text),
        }
        .into())
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
        match &event.event {
            PtyTranscriptEventKind::Sent { data } => {
                summary.sent_events += 1;
                summary.bytes_sent += data.len();
            }
            PtyTranscriptEventKind::Received { data } => {
                summary.received_events += 1;
                summary.bytes_received += data.len();
            }
            PtyTranscriptEventKind::Resized { .. } => {
                summary.resized_events += 1;
            }
            PtyTranscriptEventKind::ExitStatus { exit_status } => {
                summary.exit_status = Some(*exit_status);
            }
            PtyTranscriptEventKind::Eof => {
                summary.eof_seen = true;
            }
            PtyTranscriptEventKind::Close => {
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
