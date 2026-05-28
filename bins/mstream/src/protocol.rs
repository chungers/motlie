use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const DEFAULT_WORKSTREAM_EVENT_LIMIT: usize = 1_000;
pub const DEFAULT_STATUS_ACTIVE_WINDOW_SECS: u64 = 30;
pub const DEFAULT_STATUS_IDLE_AFTER_SECS: u64 = 300;
pub const DEFAULT_TIMER_SUBMIT_RETRIES: u8 = 1;
pub const DEFAULT_TIMER_SUBMIT_RETRY_DELAY_MS: u64 = 750;
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
    Connect(ConnectRequest),
    Hosts,
    Scan {
        alias: String,
    },
    Disconnect {
        alias: String,
    },
    Open(OpenRequest),
    List,
    Show {
        workstream: String,
    },
    Close(CloseRequest),
    Join(JoinRequest),
    New(NewRequest),
    Leave(LeaveRequest),
    Kill {
        target: String,
    },
    Send(SendRequest),
    Interrupt(InterruptRequest),
    Broadcast(BroadcastRequest),
    SessionList,
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
    TimerList,
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
    pub settings: WorkstreamSettings,
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
    pub task: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveRequest {
    pub workstream: String,
    pub target: String,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendRequest {
    pub workstream: String,
    pub target: String,
    pub text: String,
    pub paste_mode: PasteMode,
    pub enter: bool,
    pub interrupt_first: bool,
    pub settle_ms: u64,
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
    pub role: Option<String>,
    pub state: Option<AgentState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMarkRequest {
    pub target: String,
    pub state: AgentState,
    pub summary: String,
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
    pub every_secs: u64,
    pub target: String,
    pub prompt: String,
    pub enter: bool,
    #[serde(default = "default_timer_submit_retries")]
    pub submit_retries: u8,
    #[serde(default = "default_timer_submit_retry_delay_ms")]
    pub submit_retry_delay_ms: u64,
    #[serde(default = "default_timer_input_quiet_for")]
    pub input_quiet_for_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsRequest {
    pub workstream: String,
    pub after: Option<String>,
    pub limit: usize,
}

fn default_status_active_window_secs() -> u64 {
    DEFAULT_STATUS_ACTIVE_WINDOW_SECS
}

fn default_status_idle_after_secs() -> u64 {
    DEFAULT_STATUS_IDLE_AFTER_SECS
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub workstream: String,
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
    pub count: usize,
    pub goal: Option<String>,
    pub selectors: BTreeMap<String, String>,
    pub task: Option<String>,
}
