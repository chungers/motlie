use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const DEFAULT_WORKSTREAM_EVENT_LIMIT: usize = 1_000;
pub const DEFAULT_STATUS_ACTIVE_WINDOW_SECS: u64 = 30;
pub const DEFAULT_STATUS_IDLE_AFTER_SECS: u64 = 300;
pub const DEFAULT_SUBMIT_SETTLE_MS: u64 = 500;
pub const DEFAULT_SUBMIT_RETRIES: u8 = 1;
pub const DEFAULT_SUBMIT_RETRY_DELAY_MS: u64 = 750;
pub const DEFAULT_TIMER_SUBMIT_RETRIES: u8 = DEFAULT_SUBMIT_RETRIES;
pub const DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS: u64 = DEFAULT_SUBMIT_RETRY_DELAY_MS;
pub const DEFAULT_TIMER_INPUT_QUIET_FOR_SECS: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum AgentState {
    Available,
    Reserved,
    Busy,
    Idle,
    Done,
    Blocked,
    NeedsInput,
    Quarantined,
}

impl AgentState {
    pub fn as_str(self) -> &'static str {
        match self {
            AgentState::Available => "available",
            AgentState::Reserved => "reserved",
            AgentState::Busy => "busy",
            AgentState::Idle => "idle",
            AgentState::Done => "done",
            AgentState::Blocked => "blocked",
            AgentState::NeedsInput => "needs-input",
            AgentState::Quarantined => "quarantined",
        }
    }

    pub fn event_kind(self) -> Option<&'static str> {
        match self {
            AgentState::Done => Some("completed"),
            AgentState::Blocked => Some("blocked"),
            AgentState::NeedsInput => Some("needs_input"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum PasteMode {
    Bracketed,
    Literal,
}

impl PasteMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PasteMode::Bracketed => "bracketed",
            PasteMode::Literal => "literal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum InterruptKey {
    Esc,
    CtrlC,
}

impl InterruptKey {
    pub fn as_str(self) -> &'static str {
        match self {
            InterruptKey::Esc => "esc",
            InterruptKey::CtrlC => "ctrl-c",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ClientRequest {
    DaemonStatus,
    DaemonStop,
    ResolveAttach(AttachResolveRequest),
    Connect(ConnectRequest),
    Hosts,
    Scan {
        alias: String,
    },
    Disconnect {
        alias: String,
    },
    Open(OpenRequest),
    Label(LabelRequest),
    List,
    Show {
        workstream: String,
    },
    Close(CloseRequest),
    Join(JoinRequest),
    New(NewRequest),
    Leave(LeaveRequest),
    Retire(RetireRequest),
    Reclaim {
        target: String,
    },
    Send(SendRequest),
    Interrupt(InterruptRequest),
    Broadcast(BroadcastRequest),
    SessionRetag(SessionRetagRequest),
    SessionList(SessionListRequest),
    Doctor(DoctorRequest),
    SessionMark(SessionMarkRequest),
    HandoffArm(HandoffArmRequest),
    HandoffList {
        workstream: String,
    },
    HandoffCancel {
        workstream: String,
        handoff_id: String,
    },
    TimerStart(TimerStartRequest),
    TimerList {
        workstream: Option<String>,
    },
    TimerStop {
        name: String,
    },
    TimerFire {
        name: String,
    },
    Status {
        workstream: String,
        #[serde(default = "default_status_active_window_secs")]
        active_window_secs: u64,
        #[serde(default = "default_status_idle_after_secs")]
        idle_after_secs: u64,
    },
    Events(EventsRequest),
    Snapshot(SnapshotRequest),
    SummaryInput(SummaryInputRequest),
    Recruit(RecruitRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachResolveRequest {
    pub target: String,
    #[serde(default)]
    pub mode: AttachResolveMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachResolveMode {
    #[default]
    Pty,
    WindowInjection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachCommandRecord {
    pub program: String,
    pub args: Vec<String>,
    pub shell: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachResolveRecord {
    #[serde(rename = "type")]
    pub record_type: String,
    pub op: String,
    pub target: String,
    pub command: AttachCommandRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectRequest {
    pub alias: String,
    pub ssh_uri: String,
    pub labels: BTreeMap<String, String>,
    pub capacity: BTreeMap<String, String>,
    pub work_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRequest {
    pub workstream: String,
    pub title: String,
    pub goal: Option<String>,
    pub domain: Option<String>,
    #[serde(default)]
    pub mmux_label: Option<String>,
    #[serde(default)]
    pub settings: WorkstreamSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelRequest {
    pub workstream: String,
    pub mmux_label: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorkstreamSettings {
    pub event_limit: usize,
}

impl Default for WorkstreamSettings {
    fn default() -> Self {
        Self {
            event_limit: DEFAULT_WORKSTREAM_EVENT_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseRequest {
    pub workstream: String,
    pub summary: Option<String>,
    pub domain: Option<String>,
    pub specialties: Vec<String>,
    #[serde(default)]
    pub stop_timers: bool,
    #[serde(default)]
    pub standby_agents: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinRequest {
    pub workstream: String,
    pub target: String,
    pub role: String,
    pub task: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequest {
    pub workstream: String,
    pub target: String,
    pub role: String,
    pub cwd: PathBuf,
    pub agent: String,
    #[serde(default)]
    pub agent_args: Vec<String>,
    pub task: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveRequest {
    pub workstream: String,
    pub target: String,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetireRequest {
    pub workstream: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendRequest {
    pub workstream: String,
    pub target: String,
    pub text: String,
    pub paste_mode: PasteMode,
    pub enter: bool,
    pub interrupt_first: bool,
    #[serde(default = "default_submit_settle_ms")]
    pub settle_ms: u64,
    #[serde(default = "default_submit_retries")]
    pub submit_retries: u8,
    #[serde(default = "default_submit_retry_delay_ms")]
    pub submit_retry_delay_ms: u64,
    #[serde(default)]
    pub verify_delivery: bool,
    #[serde(default = "default_input_quiet_for")]
    pub input_quiet_for_secs: Option<u64>,
    pub require_state: Option<AgentState>,
    pub set_state: Option<AgentState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptRequest {
    pub target: String,
    pub key: InterruptKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastRequest {
    pub workstream: String,
    pub text: String,
    pub paste_mode: PasteMode,
    pub enter: bool,
    #[serde(default = "default_submit_settle_ms")]
    pub settle_ms: u64,
    #[serde(default = "default_submit_retries")]
    pub submit_retries: u8,
    #[serde(default = "default_submit_retry_delay_ms")]
    pub submit_retry_delay_ms: u64,
    #[serde(default)]
    pub verify_delivery: bool,
    #[serde(default = "default_input_quiet_for")]
    pub input_quiet_for_secs: Option<u64>,
    pub role: Option<String>,
    pub state: Option<AgentState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionListRequest {
    #[serde(default)]
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DoctorRequest {
    #[serde(default)]
    pub cached: bool,
    #[serde(default)]
    pub quarantine_dead: bool,
    #[serde(default)]
    pub prune_quarantined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMarkRequest {
    pub target: String,
    pub state: AgentState,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRetagRequest {
    pub target: String,
    #[serde(default)]
    pub new_name: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub workstream: Option<String>,
    #[serde(default)]
    pub mmux_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffArmRequest {
    pub workstream: String,
    pub from: String,
    pub to: String,
    pub on: AgentState,
    pub task: String,
    pub only_on_transition: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerStartRequest {
    pub name: String,
    #[serde(default)]
    pub workstream: Option<String>,
    pub every_secs: u64,
    pub target: String,
    pub prompt: String,
    #[serde(default = "default_paste_mode")]
    pub paste_mode: PasteMode,
    pub enter: bool,
    #[serde(default = "default_submit_settle_ms")]
    pub settle_ms: u64,
    #[serde(default = "default_timer_submit_retries")]
    pub submit_retries: u8,
    #[serde(default = "default_timer_submit_retry_delay_ms")]
    pub submit_retry_delay_ms: u64,
    #[serde(default)]
    pub verify_delivery: bool,
    #[serde(default = "default_timer_input_quiet_for")]
    pub input_quiet_for_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsRequest {
    pub workstream: String,
    pub after: Option<String>,
    pub limit: usize,
    #[serde(default)]
    pub readable: bool,
}

fn default_status_active_window_secs() -> u64 {
    DEFAULT_STATUS_ACTIVE_WINDOW_SECS
}

fn default_status_idle_after_secs() -> u64 {
    DEFAULT_STATUS_IDLE_AFTER_SECS
}

fn default_submit_settle_ms() -> u64 {
    DEFAULT_SUBMIT_SETTLE_MS
}

fn default_submit_retries() -> u8 {
    DEFAULT_SUBMIT_RETRIES
}

fn default_submit_retry_delay_ms() -> u64 {
    DEFAULT_SUBMIT_RETRY_DELAY_MS
}

fn default_timer_submit_retries() -> u8 {
    DEFAULT_TIMER_SUBMIT_RETRIES
}

fn default_timer_submit_retry_delay_ms() -> u64 {
    DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS
}

fn default_timer_input_quiet_for() -> Option<u64> {
    Some(DEFAULT_TIMER_INPUT_QUIET_FOR_SECS)
}

fn default_input_quiet_for() -> Option<u64> {
    Some(DEFAULT_TIMER_INPUT_QUIET_FOR_SECS)
}

fn default_paste_mode() -> PasteMode {
    PasteMode::Bracketed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub workstream: String,
    #[serde(default)]
    pub target: Option<String>,
    pub after: Option<String>,
    pub max_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryInputRequest {
    pub workstream: String,
    pub since: Option<String>,
    pub max_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecruitRequest {
    pub workstream: String,
    pub role: String,
    pub agent: Option<String>,
    #[serde(default)]
    pub agent_args: Vec<String>,
    pub count: usize,
    pub goal: Option<String>,
    pub selectors: BTreeMap<String, String>,
    pub task: Option<String>,
}
