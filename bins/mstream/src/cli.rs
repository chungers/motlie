use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use anyhow::bail;
use clap::{Args, Parser, Subcommand};

use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LeaveRequest, NewRequest,
    OpenRequest, PasteMode, RecruitRequest, SendRequest, SessionMarkRequest, SnapshotRequest,
    SummaryInputRequest, TimerStartRequest, WorkstreamSettings, DEFAULT_STATUS_ACTIVE_WINDOW_SECS,
    DEFAULT_STATUS_IDLE_AFTER_SECS, DEFAULT_TIMER_INPUT_QUIET_FOR_SECS,
    DEFAULT_TIMER_SUBMIT_RETRIES, DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS,
    DEFAULT_WORKSTREAM_EVENT_LIMIT,
};

#[derive(Debug, Parser)]
#[command(name = "mstream")]
#[command(about = "Agent-facing tmux workstream orchestrator")]
pub struct Cli {
    #[arg(long, global = true)]
    pub socket: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn socket_path(&self) -> PathBuf {
        self.socket
            .clone()
            .or_else(|| env::var("MSTREAM_SOCKET").ok().map(PathBuf::from))
            .unwrap_or_else(default_socket_path)
    }
}

pub fn default_socket_path() -> PathBuf {
    let user = env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    PathBuf::from(format!("/tmp/mstream-{user}.sock"))
}

#[derive(Debug, Subcommand)]
pub enum Command {
    #[command(subcommand)]
    Daemon(DaemonCommand),
    Connect(ConnectArgs),
    Hosts,
    Scan {
        alias: String,
    },
    Disconnect {
        alias: String,
    },
    Open(OpenArgs),
    List,
    Show {
        workstream: String,
    },
    Close(CloseArgs),
    Join(JoinArgs),
    New(NewArgs),
    Leave(LeaveArgs),
    Kill {
        target: String,
    },
    Send(SendArgs),
    Interrupt(InterruptArgs),
    Broadcast(BroadcastArgs),
    #[command(subcommand)]
    Session(SessionCommand),
    #[command(subcommand)]
    Handoff(HandoffCommand),
    #[command(subcommand)]
    Timer(TimerCommand),
    Status(StatusArgs),
    Events(EventsArgs),
    Snapshot(SnapshotArgs),
    SummaryInput(SummaryInputArgs),
    Recruit(RecruitArgs),
}

impl Command {
    pub fn into_request(self) -> anyhow::Result<ClientRequest> {
        match self {
            Command::Daemon(DaemonCommand::Status) => Ok(ClientRequest::DaemonStatus),
            Command::Daemon(DaemonCommand::Stop) => Ok(ClientRequest::DaemonStop),
            Command::Daemon(DaemonCommand::Start(_)) => {
                bail!("daemon start is handled before client request conversion")
            }
            Command::Connect(args) => Ok(ClientRequest::Connect(args.into_request()?)),
            Command::Hosts => Ok(ClientRequest::Hosts),
            Command::Scan { alias } => Ok(ClientRequest::Scan { alias }),
            Command::Disconnect { alias } => Ok(ClientRequest::Disconnect { alias }),
            Command::Open(args) => Ok(ClientRequest::Open(OpenRequest {
                workstream: args.workstream,
                title: args.title,
                goal: args.goal,
                domain: args.domain,
                settings: WorkstreamSettings {
                    event_limit: args.event_limit,
                },
            })),
            Command::List => Ok(ClientRequest::List),
            Command::Show { workstream } => Ok(ClientRequest::Show { workstream }),
            Command::Close(args) => Ok(ClientRequest::Close(CloseRequest {
                workstream: args.workstream,
                summary: args.summary,
                domain: args.domain,
                specialties: args.specialty,
            })),
            Command::Join(args) => Ok(ClientRequest::Join(JoinRequest {
                workstream: args.workstream,
                target: args.target,
                role: args.role,
                task: args.task,
            })),
            Command::New(args) => Ok(ClientRequest::New(NewRequest {
                workstream: args.workstream,
                target: args.target,
                role: args.role,
                cwd: args.cwd,
                agent: args.agent,
                task: args.task,
            })),
            Command::Leave(args) => Ok(ClientRequest::Leave(LeaveRequest {
                workstream: args.workstream,
                target: args.target,
                available: args.available,
            })),
            Command::Kill { target } => Ok(ClientRequest::Kill { target }),
            Command::Send(args) => Ok(ClientRequest::Send(args.into_request()?)),
            Command::Interrupt(args) => Ok(ClientRequest::Interrupt(InterruptRequest {
                target: args.target,
                key: args.key,
            })),
            Command::Broadcast(args) => Ok(ClientRequest::Broadcast(args.into_request()?)),
            Command::Session(SessionCommand::List) => Ok(ClientRequest::SessionList),
            Command::Session(SessionCommand::Mark(args)) => {
                Ok(ClientRequest::SessionMark(args.into_request()?))
            }
            Command::Handoff(HandoffCommand::Arm(args)) => {
                Ok(ClientRequest::HandoffArm(HandoffArmRequest {
                    workstream: args.workstream,
                    from: args.from,
                    to: args.to,
                    on: args.on,
                    task: args.task,
                    only_on_transition: args.only_on_transition,
                }))
            }
            Command::Handoff(HandoffCommand::List { workstream }) => {
                Ok(ClientRequest::HandoffList { workstream })
            }
            Command::Handoff(HandoffCommand::Cancel {
                workstream,
                handoff_id,
            }) => Ok(ClientRequest::HandoffCancel {
                workstream,
                handoff_id,
            }),
            Command::Timer(TimerCommand::Start(args)) => {
                Ok(ClientRequest::TimerStart(args.into_request()?))
            }
            Command::Timer(TimerCommand::List) => Ok(ClientRequest::TimerList),
            Command::Timer(TimerCommand::Stop { name }) => Ok(ClientRequest::TimerStop { name }),
            Command::Timer(TimerCommand::Fire { name }) => Ok(ClientRequest::TimerFire { name }),
            Command::Status(args) => Ok(ClientRequest::Status {
                workstream: args.workstream,
                active_window_secs: args.active_window_secs,
                idle_after_secs: args.idle_after_secs,
            }),
            Command::Events(args) => Ok(ClientRequest::Events(EventsRequest {
                workstream: args.workstream,
                after: args.after,
                limit: args.limit,
            })),
            Command::Snapshot(args) => Ok(ClientRequest::Snapshot(SnapshotRequest {
                workstream: args.workstream,
                after: args.after,
                max_chars: args.max_chars,
            })),
            Command::SummaryInput(args) => Ok(ClientRequest::SummaryInput(SummaryInputRequest {
                workstream: args.workstream,
                since: args.since,
                max_chars: args.max_chars,
            })),
            Command::Recruit(args) => Ok(ClientRequest::Recruit(args.into_request()?)),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum DaemonCommand {
    Start(DaemonStartArgs),
    Status,
    Stop,
}

#[derive(Debug, Args)]
pub struct DaemonStartArgs {
    #[arg(long)]
    pub foreground: bool,
}

#[derive(Debug, Args)]
pub struct ConnectArgs {
    pub alias: String,
    pub ssh_uri: String,
    #[arg(long = "label")]
    pub labels: Vec<String>,
    #[arg(long = "capacity")]
    pub capacity: Vec<String>,
    #[arg(long)]
    pub work_root: Option<PathBuf>,
}

impl ConnectArgs {
    fn into_request(self) -> anyhow::Result<ConnectRequest> {
        Ok(ConnectRequest {
            alias: self.alias,
            ssh_uri: self.ssh_uri,
            labels: parse_pairs("label", self.labels)?,
            capacity: parse_pairs("capacity", self.capacity)?,
            work_root: self.work_root,
        })
    }
}

#[derive(Debug, Args)]
pub struct OpenArgs {
    pub workstream: String,
    #[arg(long)]
    pub title: String,
    #[arg(long)]
    pub goal: Option<String>,
    #[arg(long)]
    pub domain: Option<String>,
    #[arg(long, default_value_t = DEFAULT_WORKSTREAM_EVENT_LIMIT)]
    pub event_limit: usize,
}

#[derive(Debug, Args)]
pub struct CloseArgs {
    pub workstream: String,
    #[arg(long)]
    pub summary: Option<String>,
    #[arg(long)]
    pub domain: Option<String>,
    #[arg(long = "specialty")]
    pub specialty: Vec<String>,
}

#[derive(Debug, Args)]
pub struct JoinArgs {
    pub workstream: String,
    pub target: String,
    #[arg(long)]
    pub role: String,
    #[arg(long)]
    pub task: Option<String>,
}

#[derive(Debug, Args)]
pub struct NewArgs {
    pub workstream: String,
    pub target: String,
    #[arg(long)]
    pub role: String,
    #[arg(long)]
    pub cwd: PathBuf,
    #[arg(long)]
    pub agent: String,
    #[arg(long)]
    pub task: Option<String>,
}

#[derive(Debug, Args)]
pub struct LeaveArgs {
    pub workstream: String,
    pub target: String,
    #[arg(long)]
    pub available: bool,
}

#[derive(Debug, Args)]
pub struct SendArgs {
    pub workstream: String,
    pub target: String,
    #[arg(long)]
    pub text: String,
    #[arg(long, value_enum, default_value_t = PasteMode::Bracketed)]
    pub paste_mode: PasteMode,
    #[arg(long)]
    pub enter: bool,
    #[arg(long)]
    pub no_enter: bool,
    #[arg(long)]
    pub interrupt_first: bool,
    #[arg(long, default_value_t = 500)]
    pub settle_ms: u64,
    #[arg(long, value_enum)]
    pub require_state: Option<AgentState>,
    #[arg(long, value_enum)]
    pub set_state: Option<AgentState>,
}

impl SendArgs {
    fn into_request(self) -> anyhow::Result<SendRequest> {
        let enter = resolve_enter(self.enter, self.no_enter)?;
        Ok(SendRequest {
            workstream: self.workstream,
            target: self.target,
            text: self.text,
            paste_mode: self.paste_mode,
            enter,
            interrupt_first: self.interrupt_first,
            settle_ms: self.settle_ms,
            require_state: self.require_state,
            set_state: self.set_state,
        })
    }
}

#[derive(Debug, Args)]
pub struct InterruptArgs {
    pub target: String,
    #[arg(long, value_enum, default_value_t = InterruptKey::Esc)]
    pub key: InterruptKey,
}

#[derive(Debug, Args)]
pub struct BroadcastArgs {
    pub workstream: String,
    #[arg(long)]
    pub text: String,
    #[arg(long, value_enum, default_value_t = PasteMode::Bracketed)]
    pub paste_mode: PasteMode,
    #[arg(long)]
    pub enter: bool,
    #[arg(long)]
    pub no_enter: bool,
    #[arg(long)]
    pub role: Option<String>,
    #[arg(long, value_enum)]
    pub state: Option<AgentState>,
}

impl BroadcastArgs {
    fn into_request(self) -> anyhow::Result<BroadcastRequest> {
        let enter = resolve_enter(self.enter, self.no_enter)?;
        Ok(BroadcastRequest {
            workstream: self.workstream,
            text: self.text,
            paste_mode: self.paste_mode,
            enter,
            role: self.role,
            state: self.state,
        })
    }
}

#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    List,
    Mark(SessionMarkArgs),
}

#[derive(Debug, Args)]
pub struct SessionMarkArgs {
    pub target: String,
    #[arg(long, value_enum)]
    pub state: AgentState,
    #[arg(long)]
    pub summary: String,
}

impl SessionMarkArgs {
    fn into_request(self) -> anyhow::Result<SessionMarkRequest> {
        Ok(SessionMarkRequest {
            target: self.target,
            state: self.state,
            summary: self.summary,
        })
    }
}

#[derive(Debug, Subcommand)]
pub enum HandoffCommand {
    Arm(HandoffArmArgs),
    List {
        workstream: String,
    },
    Cancel {
        workstream: String,
        handoff_id: String,
    },
}

#[derive(Debug, Args)]
pub struct HandoffArmArgs {
    pub workstream: String,
    #[arg(long)]
    pub from: String,
    #[arg(long)]
    pub to: String,
    #[arg(long, value_enum)]
    pub on: AgentState,
    #[arg(long)]
    pub task: String,
    #[arg(long)]
    pub only_on_transition: bool,
}

#[derive(Debug, Subcommand)]
pub enum TimerCommand {
    Start(TimerStartArgs),
    List,
    Stop { name: String },
    Fire { name: String },
}

#[derive(Debug, Args)]
pub struct TimerStartArgs {
    pub name: String,
    #[arg(long = "every", value_parser = parse_duration_secs)]
    pub every_secs: u64,
    #[arg(long)]
    pub target: String,
    #[arg(long)]
    pub prompt: String,
    #[arg(long)]
    pub enter: bool,
    #[arg(long)]
    pub no_enter: bool,
    #[arg(long, default_value_t = DEFAULT_TIMER_SUBMIT_RETRIES)]
    pub submit_retries: u8,
    #[arg(long, default_value_t = DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS)]
    pub submit_retry_delay_ms: u64,
    #[arg(
        long = "input-quiet-for",
        value_parser = parse_duration_secs,
        default_value_t = DEFAULT_TIMER_INPUT_QUIET_FOR_SECS
    )]
    pub input_quiet_for_secs: u64,
    #[arg(long)]
    pub no_input_guard: bool,
}

impl TimerStartArgs {
    fn into_request(self) -> anyhow::Result<TimerStartRequest> {
        if self.name.trim().is_empty() {
            bail!("timer name cannot be empty");
        }
        if self.prompt.is_empty() {
            bail!("--prompt cannot be empty");
        }
        let enter = resolve_enter(self.enter, self.no_enter)?;
        Ok(TimerStartRequest {
            name: self.name,
            every_secs: self.every_secs,
            target: self.target,
            prompt: self.prompt,
            enter,
            submit_retries: if enter { self.submit_retries } else { 0 },
            submit_retry_delay_ms: self.submit_retry_delay_ms,
            input_quiet_for_secs: if self.no_input_guard {
                None
            } else {
                Some(self.input_quiet_for_secs)
            },
        })
    }
}

#[derive(Debug, Args)]
pub struct StatusArgs {
    pub workstream: String,
    #[arg(long, default_value_t = DEFAULT_STATUS_ACTIVE_WINDOW_SECS)]
    pub active_window_secs: u64,
    #[arg(long, default_value_t = DEFAULT_STATUS_IDLE_AFTER_SECS)]
    pub idle_after_secs: u64,
}

#[derive(Debug, Args)]
pub struct EventsArgs {
    pub workstream: String,
    #[arg(long)]
    pub after: Option<String>,
    #[arg(long, default_value_t = 200)]
    pub limit: usize,
}

#[derive(Debug, Args)]
pub struct SnapshotArgs {
    pub workstream: String,
    #[arg(long)]
    pub after: Option<String>,
    #[arg(long, default_value_t = 12_000)]
    pub max_chars: usize,
}

#[derive(Debug, Args)]
pub struct SummaryInputArgs {
    pub workstream: String,
    #[arg(long)]
    pub since: Option<String>,
    #[arg(long, default_value_t = 12_000)]
    pub max_chars: usize,
}

#[derive(Debug, Args)]
pub struct RecruitArgs {
    pub workstream: String,
    #[arg(long)]
    pub role: String,
    #[arg(long)]
    pub agent: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub count: usize,
    #[arg(long)]
    pub goal: Option<String>,
    #[arg(long = "selector")]
    pub selectors: Vec<String>,
    #[arg(long)]
    pub task: Option<String>,
}

impl RecruitArgs {
    fn into_request(self) -> anyhow::Result<RecruitRequest> {
        Ok(RecruitRequest {
            workstream: self.workstream,
            role: self.role,
            agent: self.agent,
            count: self.count,
            goal: self.goal,
            selectors: parse_pairs("selector", self.selectors)?,
            task: self.task,
        })
    }
}

fn resolve_enter(enter: bool, no_enter: bool) -> anyhow::Result<bool> {
    if enter && no_enter {
        bail!("--enter and --no-enter are mutually exclusive");
    }
    Ok(!no_enter)
}

fn parse_pairs(name: &str, values: Vec<String>) -> anyhow::Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    for value in values {
        let Some((key, val)) = value.split_once('=') else {
            bail!("--{name} must use key=value syntax: {value}");
        };
        if key.is_empty() {
            bail!("--{name} key cannot be empty: {value}");
        }
        parsed.insert(key.to_string(), val.to_string());
    }
    Ok(parsed)
}

fn parse_duration_secs(value: &str) -> Result<u64, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("duration cannot be empty".to_string());
    }

    let (digits, multiplier) = if let Some(digits) = trimmed.strip_suffix("seconds") {
        (digits, 1)
    } else if let Some(digits) = trimmed.strip_suffix("second") {
        (digits, 1)
    } else if let Some(digits) = trimmed.strip_suffix("secs") {
        (digits, 1)
    } else if let Some(digits) = trimmed.strip_suffix("sec") {
        (digits, 1)
    } else if let Some(digits) = trimmed.strip_suffix("minutes") {
        (digits, 60)
    } else if let Some(digits) = trimmed.strip_suffix("minute") {
        (digits, 60)
    } else if let Some(digits) = trimmed.strip_suffix("mins") {
        (digits, 60)
    } else if let Some(digits) = trimmed.strip_suffix("min") {
        (digits, 60)
    } else if let Some(digits) = trimmed.strip_suffix('s') {
        (digits, 1)
    } else if let Some(digits) = trimmed.strip_suffix('m') {
        (digits, 60)
    } else {
        (trimmed, 1)
    };

    let count = digits.trim().parse::<u64>().map_err(|_| {
        format!("invalid duration '{value}', expected seconds like 30s or minutes like 5m")
    })?;
    if count == 0 {
        return Err("duration must be greater than zero".to_string());
    }
    count
        .checked_mul(multiplier)
        .ok_or_else(|| format!("duration '{value}' is too large"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_accepts_seconds_and_minutes() {
        assert_eq!(parse_duration_secs("30").expect("seconds"), 30);
        assert_eq!(parse_duration_secs("30s").expect("seconds suffix"), 30);
        assert_eq!(parse_duration_secs("2m").expect("minutes suffix"), 120);
        assert_eq!(parse_duration_secs("3 minutes").expect("minutes word"), 180);
    }

    #[test]
    fn parse_duration_rejects_empty_zero_and_invalid_values() {
        assert!(parse_duration_secs("").is_err());
        assert!(parse_duration_secs("0s").is_err());
        assert!(parse_duration_secs("soon").is_err());
    }

    #[test]
    fn timer_start_command_builds_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "start",
            "issue-337-poll",
            "--every",
            "5m",
            "--target",
            "local::orchestrator",
            "--prompt",
            "Wake up and poll.",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert_eq!(request.name, "issue-337-poll");
        assert_eq!(request.every_secs, 300);
        assert_eq!(request.target, "local::orchestrator");
        assert_eq!(request.prompt, "Wake up and poll.");
        assert!(request.enter);
        assert_eq!(request.submit_retries, 1);
        assert_eq!(request.submit_retry_delay_ms, 750);
        assert_eq!(
            request.input_quiet_for_secs,
            Some(DEFAULT_TIMER_INPUT_QUIET_FOR_SECS)
        );
    }

    #[test]
    fn timer_start_command_allows_submit_retry_override() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "start",
            "issue-337-poll",
            "--every",
            "5m",
            "--target",
            "local::orchestrator",
            "--prompt",
            "Wake up and poll.",
            "--submit-retries",
            "2",
            "--submit-retry-delay-ms",
            "1000",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert_eq!(request.submit_retries, 2);
        assert_eq!(request.submit_retry_delay_ms, 1000);
    }

    #[test]
    fn timer_start_command_allows_input_guard_override() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "start",
            "issue-337-poll",
            "--every",
            "5m",
            "--target",
            "local::orchestrator",
            "--prompt",
            "Wake up and poll.",
            "--input-quiet-for",
            "30s",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert_eq!(request.input_quiet_for_secs, Some(30));
    }

    #[test]
    fn timer_start_command_allows_disabling_input_guard() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "start",
            "issue-337-poll",
            "--every",
            "5m",
            "--target",
            "local::orchestrator",
            "--prompt",
            "Wake up and poll.",
            "--no-input-guard",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert_eq!(request.input_quiet_for_secs, None);
    }

    #[test]
    fn timer_start_no_enter_disables_submit_retries() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "start",
            "issue-337-poll",
            "--every",
            "5m",
            "--target",
            "local::orchestrator",
            "--prompt",
            "Wake up and poll.",
            "--no-enter",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert!(!request.enter);
        assert_eq!(request.submit_retries, 0);
    }
}
