use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use anyhow::bail;
use clap::{Args, Parser, Subcommand};

use crate::build_info;
use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LabelRequest, LeaveRequest,
    NewRequest, OpenRequest, PasteMode, RecruitRequest, RetireRequest, SendRequest,
    SessionMarkRequest, SessionRetagRequest, SnapshotRequest, SummaryInputRequest,
    TimerStartRequest, WorkstreamSettings, DEFAULT_STATUS_ACTIVE_WINDOW_SECS,
    DEFAULT_STATUS_IDLE_AFTER_SECS, DEFAULT_SUBMIT_RETRIES, DEFAULT_SUBMIT_RETRY_DELAY_MS,
    DEFAULT_SUBMIT_SETTLE_MS, DEFAULT_TIMER_INPUT_QUIET_FOR_SECS, DEFAULT_TIMER_SUBMIT_RETRIES,
    DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS, DEFAULT_WORKSTREAM_EVENT_LIMIT,
};

#[derive(Debug, Parser)]
#[command(name = "mstream")]
#[command(about = "Agent-facing tmux workstream orchestrator")]
#[command(version = build_info::VERSION)]
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
    Label(LabelArgs),
    List,
    Show {
        workstream: String,
    },
    Close(CloseArgs),
    Join(JoinArgs),
    New(NewArgs),
    Leave(LeaveArgs),
    Retire(RetireArgs),
    Reclaim {
        target: String,
    },
    Send(SendArgs),
    Interrupt(InterruptArgs),
    Broadcast(BroadcastArgs),
    Rename(RenameArgs),
    #[command(subcommand)]
    Session(SessionCommand),
    #[command(subcommand)]
    Handoff(HandoffCommand),
    #[command(subcommand)]
    Timer(TimerCommand),
    Status(StatusArgs),
    #[command(about = "Read durable workstream audit events and call-log entries")]
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
                mmux_label: args.mmux_label,
                settings: WorkstreamSettings {
                    event_limit: args.event_limit,
                },
            })),
            Command::Label(args) => Ok(ClientRequest::Label(LabelRequest {
                workstream: args.workstream,
                mmux_label: args.mmux_label,
            })),
            Command::List => Ok(ClientRequest::List),
            Command::Show { workstream } => Ok(ClientRequest::Show { workstream }),
            Command::Close(args) => Ok(ClientRequest::Close(CloseRequest {
                workstream: args.workstream,
                summary: args.summary,
                domain: args.domain,
                specialties: args.specialty,
                stop_timers: args.stop_timers,
                standby_agents: args.standby_agents,
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
                agent_args: args.agent_args,
                task: args.task,
            })),
            Command::Leave(args) => Ok(ClientRequest::Leave(LeaveRequest {
                workstream: args.workstream,
                target: args.target,
                available: args.available,
            })),
            Command::Retire(args) => Ok(ClientRequest::Retire(RetireRequest {
                workstream: args.workstream,
                target: args.target,
            })),
            Command::Reclaim { target } => Ok(ClientRequest::Reclaim { target }),
            Command::Send(args) => Ok(ClientRequest::Send(args.into_request()?)),
            Command::Interrupt(args) => Ok(ClientRequest::Interrupt(InterruptRequest {
                target: args.target,
                key: args.key,
            })),
            Command::Broadcast(args) => Ok(ClientRequest::Broadcast(args.into_request()?)),
            Command::Rename(args) => Ok(ClientRequest::SessionRetag(args.into_request()?)),
            Command::Session(SessionCommand::List) => Ok(ClientRequest::SessionList),
            Command::Session(SessionCommand::Mark(args)) => {
                Ok(ClientRequest::SessionMark(args.into_request()?))
            }
            Command::Session(SessionCommand::Retag(args)) => {
                Ok(ClientRequest::SessionRetag(args.into_request()?))
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
            Command::Timer(TimerCommand::List(args)) => Ok(ClientRequest::TimerList {
                workstream: args.workstream,
            }),
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
                readable: args.readable,
            })),
            Command::Snapshot(args) => Ok(ClientRequest::Snapshot(SnapshotRequest {
                workstream: args.workstream,
                target: args.target,
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
    #[arg(long)]
    pub mmux_label: Option<String>,
    #[arg(long, default_value_t = DEFAULT_WORKSTREAM_EVENT_LIMIT)]
    pub event_limit: usize,
}

#[derive(Debug, Args)]
pub struct LabelArgs {
    pub workstream: String,
    #[arg(long)]
    pub mmux_label: String,
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
    #[arg(long)]
    pub stop_timers: bool,
    #[arg(long)]
    pub standby_agents: bool,
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
    #[arg(
        long,
        help = "Agent executable to start. Remote lookup uses the host non-login SSH PATH; pass an absolute path if needed."
    )]
    pub agent: String,
    #[arg(
        long = "agent-arg",
        allow_hyphen_values = true,
        num_args = 1,
        help = "Argument to pass to the agent executable. Repeat for multiple argv entries."
    )]
    pub agent_args: Vec<String>,
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
pub struct RetireArgs {
    pub workstream: String,
    pub target: String,
}

#[derive(Debug, Args)]
pub struct SendArgs {
    pub workstream: String,
    pub target: String,
    #[arg(long)]
    pub text: String,
    #[arg(long, value_enum, default_value_t = PasteMode::Bracketed)]
    pub paste_mode: PasteMode,
    #[arg(long, hide = true)]
    pub enter: bool,
    #[arg(
        long = "no-prompt-submit",
        alias = "no-enter",
        help = "Write the prompt without performing the settle-delayed verified prompt submit."
    )]
    pub no_prompt_submit: bool,
    #[arg(long)]
    pub interrupt_first: bool,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_SETTLE_MS)]
    pub settle_ms: u64,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_RETRIES)]
    pub submit_retries: u8,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_RETRY_DELAY_MS)]
    pub submit_retry_delay_ms: u64,
    #[arg(
        long,
        help = "Verify the prompt is visible after delivery; bracketed sends retry once with literal paste before failing."
    )]
    pub verify_delivery: bool,
    #[arg(long, value_enum)]
    pub require_state: Option<AgentState>,
    #[arg(long, value_enum)]
    pub set_state: Option<AgentState>,
}

impl SendArgs {
    fn into_request(self) -> anyhow::Result<SendRequest> {
        let enter = resolve_prompt_submit(self.enter, self.no_prompt_submit)?;
        Ok(SendRequest {
            workstream: self.workstream,
            target: self.target,
            text: self.text,
            paste_mode: self.paste_mode,
            enter,
            interrupt_first: self.interrupt_first,
            settle_ms: self.settle_ms,
            submit_retries: if enter { self.submit_retries } else { 0 },
            submit_retry_delay_ms: self.submit_retry_delay_ms,
            verify_delivery: self.verify_delivery,
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
    #[arg(long, hide = true)]
    pub enter: bool,
    #[arg(
        long = "no-prompt-submit",
        alias = "no-enter",
        help = "Write the prompt without performing the settle-delayed verified prompt submit."
    )]
    pub no_prompt_submit: bool,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_SETTLE_MS)]
    pub settle_ms: u64,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_RETRIES)]
    pub submit_retries: u8,
    #[arg(long, default_value_t = DEFAULT_SUBMIT_RETRY_DELAY_MS)]
    pub submit_retry_delay_ms: u64,
    #[arg(long)]
    pub role: Option<String>,
    #[arg(long, value_enum)]
    pub state: Option<AgentState>,
}

impl BroadcastArgs {
    fn into_request(self) -> anyhow::Result<BroadcastRequest> {
        let enter = resolve_prompt_submit(self.enter, self.no_prompt_submit)?;
        Ok(BroadcastRequest {
            workstream: self.workstream,
            text: self.text,
            paste_mode: self.paste_mode,
            enter,
            settle_ms: self.settle_ms,
            submit_retries: if enter { self.submit_retries } else { 0 },
            submit_retry_delay_ms: self.submit_retry_delay_ms,
            role: self.role,
            state: self.state,
        })
    }
}

#[derive(Debug, Args)]
pub struct RenameArgs {
    pub target: String,
    pub new_name: String,
    #[arg(long)]
    pub role: Option<String>,
    #[arg(long)]
    pub workstream: Option<String>,
    #[arg(long)]
    pub mmux_label: Option<String>,
}

impl RenameArgs {
    fn into_request(self) -> anyhow::Result<SessionRetagRequest> {
        Ok(SessionRetagRequest {
            target: self.target,
            new_name: Some(self.new_name),
            role: self.role,
            workstream: self.workstream,
            mmux_label: self.mmux_label,
        })
    }
}

#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    List,
    Mark(SessionMarkArgs),
    Retag(SessionRetagArgs),
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

#[derive(Debug, Args)]
pub struct SessionRetagArgs {
    pub target: String,
    #[arg(long)]
    pub role: Option<String>,
    #[arg(long)]
    pub workstream: Option<String>,
    #[arg(long)]
    pub mmux_label: Option<String>,
}

impl SessionRetagArgs {
    fn into_request(self) -> anyhow::Result<SessionRetagRequest> {
        Ok(SessionRetagRequest {
            target: self.target,
            new_name: None,
            role: self.role,
            workstream: self.workstream,
            mmux_label: self.mmux_label,
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
    List(TimerListArgs),
    Stop { name: String },
    Fire { name: String },
}

#[derive(Debug, Args)]
pub struct TimerStartArgs {
    pub name: String,
    #[arg(long)]
    pub workstream: Option<String>,
    #[arg(long = "every", value_parser = parse_duration_secs)]
    pub every_secs: u64,
    #[arg(long)]
    pub target: Option<String>,
    #[arg(long = "self")]
    pub self_target: bool,
    #[arg(long, default_value = "local")]
    pub self_host: String,
    #[arg(long)]
    pub prompt: String,
    #[arg(long, value_enum, default_value_t = PasteMode::Bracketed)]
    pub paste_mode: PasteMode,
    #[arg(long, hide = true)]
    pub enter: bool,
    #[arg(
        long = "no-prompt-submit",
        alias = "no-enter",
        help = "Write the prompt without performing the settle-delayed verified prompt submit."
    )]
    pub no_prompt_submit: bool,
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
        let target = resolve_timer_target(
            self.target,
            self.self_target,
            self.self_host,
            current_tmux_session,
        )?;
        let enter = resolve_prompt_submit(self.enter, self.no_prompt_submit)?;
        Ok(TimerStartRequest {
            name: self.name,
            workstream: self.workstream,
            every_secs: self.every_secs,
            target,
            prompt: self.prompt,
            paste_mode: self.paste_mode,
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
pub struct TimerListArgs {
    #[arg(long)]
    pub workstream: Option<String>,
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
    #[arg(long, help = "Opaque event cursor returned by a previous events call.")]
    pub after: Option<String>,
    #[arg(
        long,
        default_value_t = 200,
        help = "Maximum retained audit events to return; 0 means no page limit."
    )]
    pub limit: usize,
    #[arg(
        long,
        help = "Render the durable audit call log as readable transcript text."
    )]
    pub readable: bool,
}

#[derive(Debug, Args)]
pub struct SnapshotArgs {
    pub workstream: String,
    #[arg(long)]
    pub target: Option<String>,
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
    #[arg(
        long = "agent-arg",
        allow_hyphen_values = true,
        num_args = 1,
        help = "Argument expected in the recruited agent argv. Repeat for multiple argv entries."
    )]
    pub agent_args: Vec<String>,
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
            agent_args: self.agent_args,
            count: self.count,
            goal: self.goal,
            selectors: parse_pairs("selector", self.selectors)?,
            task: self.task,
        })
    }
}

fn resolve_prompt_submit(enter: bool, no_prompt_submit: bool) -> anyhow::Result<bool> {
    if enter && no_prompt_submit {
        bail!("--enter and --no-prompt-submit are mutually exclusive");
    }
    Ok(!no_prompt_submit)
}

fn resolve_timer_target(
    target: Option<String>,
    self_target: bool,
    self_host: String,
    current_session: impl FnOnce() -> anyhow::Result<String>,
) -> anyhow::Result<String> {
    match (target, self_target) {
        (Some(_), true) => bail!("--target and --self are mutually exclusive"),
        (Some(target), false) => Ok(target),
        (None, true) => {
            let session = current_session()?;
            Ok(format!("{self_host}::{session}"))
        }
        (None, false) => bail!("timer start requires --target <target> or --self"),
    }
}

fn current_tmux_session() -> anyhow::Result<String> {
    let output = std::process::Command::new("tmux")
        .args(["display-message", "-p", "#S"])
        .output()
        .map_err(|err| anyhow::anyhow!("failed to run tmux to resolve --self target: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!(
            "failed to resolve --self target with tmux display-message: {}",
            if stderr.is_empty() {
                output.status.to_string()
            } else {
                stderr
            }
        );
    }
    let session = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if session.is_empty() {
        bail!("failed to resolve --self target: tmux session name was empty");
    }
    Ok(session)
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
    use clap::CommandFactory;

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
    fn new_command_help_documents_remote_agent_path_lookup() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("new")
            .expect("new subcommand")
            .render_long_help()
            .to_string();

        assert!(help.contains("Remote lookup uses the host non-login SSH PATH"));
        assert!(help.contains("absolute path"));
        assert!(help.contains("--agent-arg <AGENT_ARGS>"));
    }

    #[test]
    fn prompt_submit_help_uses_new_flag_and_hides_legacy_aliases() {
        let mut send_command = Cli::command();
        let send_help = send_command
            .find_subcommand_mut("send")
            .expect("send subcommand")
            .render_long_help()
            .to_string();
        assert!(send_help.contains("--no-prompt-submit"));
        assert!(send_help.contains("verified prompt submit"));
        assert!(!send_help.contains("--no-enter"));
        assert!(!send_help.contains("--enter"));

        let mut broadcast_command = Cli::command();
        let broadcast_help = broadcast_command
            .find_subcommand_mut("broadcast")
            .expect("broadcast subcommand")
            .render_long_help()
            .to_string();
        assert!(broadcast_help.contains("--no-prompt-submit"));
        assert!(broadcast_help.contains("verified prompt submit"));
        assert!(!broadcast_help.contains("--no-enter"));
        assert!(!broadcast_help.contains("--enter"));

        let mut timer_command = Cli::command();
        let timer_help = timer_command
            .find_subcommand_mut("timer")
            .expect("timer subcommand")
            .find_subcommand_mut("start")
            .expect("timer start subcommand")
            .render_long_help()
            .to_string();
        assert!(timer_help.contains("--no-prompt-submit"));
        assert!(timer_help.contains("verified prompt submit"));
        assert!(!timer_help.contains("--no-enter"));
        assert!(!timer_help.contains("--enter"));
    }

    #[test]
    fn send_no_prompt_submit_disables_submit_retries() {
        let cli = Cli::try_parse_from([
            "mstream",
            "send",
            "issue-421-agent-inbox",
            "local::agent",
            "--text",
            "Hold this draft.",
            "--no-prompt-submit",
        ])
        .expect("send command parses");

        let request = cli.command.into_request().expect("send request");
        let ClientRequest::Send(request) = request else {
            panic!("expected send request");
        };
        assert!(!request.enter);
        assert_eq!(request.submit_retries, 0);
    }

    #[test]
    fn send_verify_delivery_builds_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "send",
            "issue-453",
            "local::agent",
            "--text",
            "/permissions",
            "--verify-delivery",
        ])
        .expect("send command parses");

        let request = cli.command.into_request().expect("send request");
        let ClientRequest::Send(request) = request else {
            panic!("expected send request");
        };
        assert!(request.verify_delivery);
    }

    #[test]
    fn broadcast_no_prompt_submit_disables_submit_retries() {
        let cli = Cli::try_parse_from([
            "mstream",
            "broadcast",
            "issue-421-agent-inbox",
            "--text",
            "Hold this draft.",
            "--no-prompt-submit",
        ])
        .expect("broadcast command parses");

        let request = cli.command.into_request().expect("broadcast request");
        let ClientRequest::Broadcast(request) = request else {
            panic!("expected broadcast request");
        };
        assert!(!request.enter);
        assert_eq!(request.submit_retries, 0);
    }

    #[test]
    fn new_command_builds_request_with_agent_args() {
        let cli = Cli::try_parse_from([
            "mstream",
            "new",
            "issue-410",
            "local::claude-reviewer",
            "--role",
            "reviewer",
            "--cwd",
            "/tmp/issue-410",
            "--agent",
            "claude",
            "--agent-arg",
            "--permission-mode",
            "--agent-arg",
            "auto",
        ])
        .expect("new command parses");

        let request = cli.command.into_request().expect("new request");
        let ClientRequest::New(request) = request else {
            panic!("expected new request");
        };
        assert_eq!(request.agent, "claude");
        assert_eq!(request.agent_args, ["--permission-mode", "auto"]);
    }

    #[test]
    fn recruit_command_builds_request_with_agent_args() {
        let cli = Cli::try_parse_from([
            "mstream",
            "recruit",
            "issue-410",
            "--role",
            "reviewer",
            "--agent",
            "claude",
            "--agent-arg",
            "--permission-mode",
            "--agent-arg",
            "auto",
        ])
        .expect("recruit command parses");

        let request = cli.command.into_request().expect("recruit request");
        let ClientRequest::Recruit(request) = request else {
            panic!("expected recruit request");
        };
        assert_eq!(request.agent.as_deref(), Some("claude"));
        assert_eq!(request.agent_args, ["--permission-mode", "auto"]);
    }

    #[test]
    fn open_command_allows_mmux_label() {
        let cli = Cli::try_parse_from([
            "mstream",
            "open",
            "issue-349-mmux-workstream-labels",
            "--title",
            "Issue 349: mmux labels",
            "--mmux-label",
            "349 labels",
        ])
        .expect("open command parses");

        let request = cli.command.into_request().expect("open request");
        let ClientRequest::Open(request) = request else {
            panic!("expected open request");
        };
        assert_eq!(request.workstream, "issue-349-mmux-workstream-labels");
        assert_eq!(request.mmux_label.as_deref(), Some("349 labels"));
    }

    #[test]
    fn label_command_builds_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "label",
            "issue-349-mmux-workstream-labels",
            "--mmux-label",
            "349 labels",
        ])
        .expect("label command parses");

        let request = cli.command.into_request().expect("label request");
        let ClientRequest::Label(request) = request else {
            panic!("expected label request");
        };
        assert_eq!(request.workstream, "issue-349-mmux-workstream-labels");
        assert_eq!(request.mmux_label, "349 labels");
    }

    #[test]
    fn rename_command_builds_session_retag_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "rename",
            "local::old-agent",
            "new-agent",
            "--role",
            "reviewer",
            "--workstream",
            "issue-360",
            "--mmux-label",
            "360 review",
        ])
        .expect("rename command parses");

        let request = cli.command.into_request().expect("rename request");
        let ClientRequest::SessionRetag(request) = request else {
            panic!("expected session retag request");
        };
        assert_eq!(request.target, "local::old-agent");
        assert_eq!(request.new_name.as_deref(), Some("new-agent"));
        assert_eq!(request.role.as_deref(), Some("reviewer"));
        assert_eq!(request.workstream.as_deref(), Some("issue-360"));
        assert_eq!(request.mmux_label.as_deref(), Some("360 review"));
    }

    #[test]
    fn session_retag_command_builds_metadata_only_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "session",
            "retag",
            "local::$1",
            "--role",
            "implementer",
            "--workstream",
            "issue-360",
            "--mmux-label",
            "360 impl",
        ])
        .expect("session retag command parses");

        let request = cli.command.into_request().expect("session retag request");
        let ClientRequest::SessionRetag(request) = request else {
            panic!("expected session retag request");
        };
        assert_eq!(request.target, "local::$1");
        assert_eq!(request.new_name, None);
        assert_eq!(request.role.as_deref(), Some("implementer"));
        assert_eq!(request.workstream.as_deref(), Some("issue-360"));
        assert_eq!(request.mmux_label.as_deref(), Some("360 impl"));
    }

    #[test]
    fn retire_command_builds_request() {
        let cli = Cli::try_parse_from(["mstream", "retire", "issue-401", "local::$1"])
            .expect("retire command parses");

        let request = cli.command.into_request().expect("retire request");
        let ClientRequest::Retire(request) = request else {
            panic!("expected retire request");
        };
        assert_eq!(request.workstream, "issue-401");
        assert_eq!(request.target, "local::$1");
    }

    #[test]
    fn reclaim_command_builds_request() {
        let cli = Cli::try_parse_from(["mstream", "reclaim", "local::$1"])
            .expect("reclaim command parses");

        let request = cli.command.into_request().expect("reclaim request");
        let ClientRequest::Reclaim { target } = request else {
            panic!("expected reclaim request");
        };
        assert_eq!(target, "local::$1");
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
            "--workstream",
            "issue-337-tmux-fleet-api",
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
        assert_eq!(
            request.workstream.as_deref(),
            Some("issue-337-tmux-fleet-api")
        );
        assert_eq!(request.every_secs, 300);
        assert_eq!(request.target, "local::orchestrator");
        assert_eq!(request.prompt, "Wake up and poll.");
        assert_eq!(request.paste_mode, PasteMode::Bracketed);
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
    fn timer_start_command_allows_literal_paste_mode() {
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
            "--paste-mode",
            "literal",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert_eq!(request.paste_mode, PasteMode::Literal);
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
    fn timer_start_no_prompt_submit_disables_submit_retries() {
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
            "--no-prompt-submit",
        ])
        .expect("timer command parses");

        let request = cli.command.into_request().expect("timer request");
        let ClientRequest::TimerStart(request) = request else {
            panic!("expected timer start request");
        };
        assert!(!request.enter);
        assert_eq!(request.submit_retries, 0);
    }

    #[test]
    fn timer_start_no_enter_alias_disables_submit_retries() {
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

    #[test]
    fn timer_list_allows_workstream_filter() {
        let cli = Cli::try_parse_from([
            "mstream",
            "timer",
            "list",
            "--workstream",
            "issue-342-mmux-endpoint-identity",
        ])
        .expect("timer list parses");

        let request = cli.command.into_request().expect("timer list request");
        let ClientRequest::TimerList { workstream } = request else {
            panic!("expected timer list request");
        };
        assert_eq!(
            workstream.as_deref(),
            Some("issue-342-mmux-endpoint-identity")
        );
    }

    #[test]
    fn snapshot_target_builds_request() {
        let cli = Cli::try_parse_from([
            "mstream",
            "snapshot",
            "issue-453",
            "--target",
            "local::$1",
            "--max-chars",
            "4000",
        ])
        .expect("snapshot command parses");

        let request = cli.command.into_request().expect("snapshot request");
        let ClientRequest::Snapshot(request) = request else {
            panic!("expected snapshot request");
        };
        assert_eq!(request.target.as_deref(), Some("local::$1"));
        assert_eq!(request.max_chars, 4000);
    }

    #[test]
    fn resolve_timer_target_supports_self_target() {
        let target = resolve_timer_target(None, true, "local".to_string(), || {
            Ok("gpt55-324-330-og".to_string())
        })
        .expect("self target");

        assert_eq!(target, "local::gpt55-324-330-og");
    }

    #[test]
    fn resolve_timer_target_rejects_ambiguous_or_missing_target() {
        assert!(resolve_timer_target(
            Some("local::orchestrator".to_string()),
            true,
            "local".to_string(),
            || Ok("self".to_string()),
        )
        .is_err());
        assert!(resolve_timer_target(None, false, "local".to_string(), || {
            Ok("self".to_string())
        })
        .is_err());
    }
}
