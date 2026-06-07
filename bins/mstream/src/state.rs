use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};
use chrono::{DateTime, Utc};
use motlie_tmux::{
    CreateSessionOptions, Error as TmuxError, Fleet, FleetTargetSpec, HostHandle, KeySequence,
    ResolvedFleetTarget, SessionEnvVar, SessionInfo, SessionTag, SshConfig, Target,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use unicode_width::UnicodeWidthStr;

use crate::jsonl;
use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LabelRequest, LeaveRequest,
    NewRequest, OpenRequest, PasteMode, RecruitRequest, RetireRequest, SendRequest,
    SessionMarkRequest, SessionRetagRequest, SnapshotRequest, SummaryInputRequest,
    TimerStartRequest, WorkstreamSettings,
};
use crate::tags;
use crate::timeline::PublicCursor;

type SessionTarget = FleetTargetSpec;
type ResolvedTarget = ResolvedFleetTarget;

const MMUX_TAG_PREFIX: &str = "mmux";
const MMUX_SELECTED_KEY: &str = "__selected-key";
const MMUX_WORKSTREAM_KEY: &str = "mstream";
const MSTREAM_MMUX_LABEL_KEY: &str = "mmux-label";
const MSTREAM_MMUX_SELECTED_KEY: &str = "mmux-selected-key";
const MSTREAM_MMUX_PREVIOUS_SELECTED_KEY: &str = "mmux-previous-selected-key";
const MMUX_LABEL_MAX_CHARS: usize = 24;

pub struct DaemonState {
    fleet: Fleet,
    hosts: BTreeMap<String, HostRecord>,
    sessions: BTreeMap<SessionTarget, SessionRecord>,
    workstreams: BTreeMap<String, WorkstreamRecord>,
    timers: BTreeMap<String, TimerRecord>,
    next_generation: u64,
    next_handoff_id: u64,
    next_timer_generation: u64,
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            fleet: Fleet::new(),
            hosts: BTreeMap::new(),
            sessions: BTreeMap::new(),
            workstreams: BTreeMap::new(),
            timers: BTreeMap::new(),
            next_generation: 1,
            next_handoff_id: 1,
            next_timer_generation: 1,
        }
    }
}

struct HostRecord {
    uri: String,
    labels: BTreeMap<String, String>,
    capacity: BTreeMap<String, String>,
    work_root: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct SessionRecord {
    role: Option<String>,
    agent: Option<String>,
    identity: String,
    state: AgentState,
    cwd: Option<PathBuf>,
    workstream: Option<String>,
    last_report_kind: Option<String>,
    last_report_summary: Option<String>,
    updated_at: DateTime<Utc>,
    context_domains: BTreeSet<String>,
    context_specialties: BTreeSet<String>,
    context_summary: Option<String>,
    last_workstream: Option<String>,
    last_workstream_title: Option<String>,
    mmux_label: Option<String>,
    mmux_previous_selected_key: Option<String>,
    tmux_session_created: Option<u64>,
    last_tmux_activity: Option<u64>,
    activity_observed_at: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct WorkstreamRecord {
    title: String,
    goal: Option<String>,
    domain: Option<String>,
    mmux_label: Option<String>,
    mmux_label_conflicts: BTreeSet<String>,
    settings: WorkstreamSettings,
    state: WorkstreamState,
    sessions: BTreeSet<SessionTarget>,
    generation: u64,
    next_sequence: u64,
    events: VecDeque<EventRecord>,
    handoffs: BTreeMap<String, HandoffRecord>,
}

struct TimerRecord {
    name: String,
    workstream: Option<String>,
    target: SessionTarget,
    tmux_session_created: Option<u64>,
    every_secs: u64,
    prompt: String,
    enter: bool,
    submit_retries: u8,
    submit_retry_delay_ms: u64,
    input_quiet_for_secs: Option<u64>,
    generation: u64,
    started_at: DateTime<Utc>,
    next_fire_at: Option<DateTime<Utc>>,
    last_fired_at: Option<DateTime<Utc>>,
    fire_count: u64,
    defer_count: u64,
    last_deferred_at: Option<DateTime<Utc>>,
    last_defer_reason: Option<String>,
    last_input_activity: Option<u64>,
    last_error: Option<String>,
    task: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkstreamState {
    Open,
    Closed,
}

#[derive(Debug, Clone, Serialize)]
struct EventRecord {
    cursor: String,
    sequence: u64,
    generation: u64,
    workstream: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    handoff_id: Option<String>,
    timestamp: String,
}

#[derive(Debug, Clone)]
struct EventDraft {
    kind: String,
    target: Option<String>,
    text: Option<String>,
    state: Option<AgentState>,
    summary: Option<String>,
    handoff_id: Option<String>,
}

impl EventDraft {
    fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            target: None,
            text: None,
            state: None,
            summary: None,
            handoff_id: None,
        }
    }

    fn target(mut self, target: impl ToString) -> Self {
        self.target = Some(target.to_string());
        self
    }

    fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    fn state(mut self, state: AgentState) -> Self {
        self.state = Some(state);
        self
    }

    fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    fn handoff_id(mut self, handoff_id: impl Into<String>) -> Self {
        self.handoff_id = Some(handoff_id.into());
        self
    }
}

#[derive(Debug, Clone, Serialize)]
struct HandoffRecord {
    id: String,
    from: SessionTarget,
    to: SessionTarget,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_tmux_session_created: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_tmux_session_created: Option<u64>,
    on: AgentState,
    task: String,
    only_on_transition: bool,
    fired: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    canceled_reason: Option<String>,
    created_at: String,
}

#[derive(Debug, Clone)]
struct StateChange {
    target: SessionTarget,
    state: AgentState,
    previous_state: Option<AgentState>,
}

struct AssignmentTags<'a> {
    workstream: &'a str,
    role: &'a str,
    identity: &'a str,
    agent: Option<&'a str>,
    cwd: Option<&'a Path>,
    state: AgentState,
}

#[derive(Debug, Clone)]
struct WorkstreamMeta {
    title: String,
    goal: Option<String>,
    domain: Option<String>,
    mmux_label: Option<String>,
}

struct SessionRetagPlan {
    old_name: String,
    new_name: String,
    old_workstream: Option<String>,
    new_workstream: Option<String>,
    role: Option<String>,
    agent: Option<String>,
    cwd: Option<PathBuf>,
    state: AgentState,
    workstream_meta: Option<WorkstreamMeta>,
    mmux_label: Option<String>,
    clear_mmux_workstream: Option<String>,
    requested_mmux_label: Option<String>,
    metadata_requested: bool,
}

#[derive(Debug, Default)]
struct MmuxLabelCleanup {
    cleared: bool,
    restored_previous: bool,
    selected_key_unchanged: bool,
    skipped_workstream_mismatch: bool,
}

#[derive(Debug, Default)]
struct MmuxLabelApplyResult {
    applied: usize,
    failed: usize,
}

struct BroadcastTarget {
    target: ResolvedTarget,
    state: AgentState,
}

struct BroadcastTargetSnapshot {
    target: SessionTarget,
    handle: HostHandle,
    state: AgentState,
}

struct RecruitPlan {
    target: ResolvedTarget,
    agent: Option<String>,
    cwd: Option<PathBuf>,
    state: AgentState,
}

struct RecruitPlanSnapshot {
    target: SessionTarget,
    handle: HostHandle,
    agent: Option<String>,
    cwd: Option<PathBuf>,
    state: AgentState,
}

struct TimerFireSnapshot {
    name: String,
    target: SessionTarget,
    tmux_session_created: Option<u64>,
    handle: HostHandle,
    prompt: String,
    enter: bool,
    submit_retries: u8,
    submit_retry_delay_ms: u64,
    input_quiet_for_secs: Option<u64>,
    generation: u64,
}

struct TimerDeferSnapshot {
    name: String,
    target: SessionTarget,
    generation: u64,
    reason: &'static str,
    latest_client_activity: Option<u64>,
    latest_client_activity_age_secs: Option<u64>,
    quiet_for_secs: u64,
    next_fire_at: Option<DateTime<Utc>>,
}

enum TimerFireOutcome {
    Sent(TimerFireSnapshot),
    Deferred(TimerDeferSnapshot),
}

struct InputGuardDecision {
    latest_client_activity: Option<u64>,
    latest_client_activity_age_secs: Option<u64>,
    retry_after_secs: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveTimer {
    generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct StatusActivityOptions {
    active_window_secs: u64,
    idle_after_secs: u64,
}

#[derive(Debug, Clone)]
enum LiveActivity {
    Present(SessionInfo),
    Missing,
    Error(String),
}

impl DaemonState {
    pub async fn handle_shared(
        shared: Arc<Mutex<Self>>,
        request: ClientRequest,
    ) -> anyhow::Result<Vec<Value>> {
        match request {
            ClientRequest::Connect(request) => Self::connect_shared(shared, request).await,
            ClientRequest::Scan { alias } => Self::scan_shared(shared, alias).await,
            ClientRequest::Close(request) => Self::close_shared(shared, request).await,
            ClientRequest::Join(request) => Self::join_shared(shared, request).await,
            ClientRequest::New(request) => Self::new_session_shared(shared, request).await,
            ClientRequest::Leave(request) => Self::leave_shared(shared, request).await,
            ClientRequest::Retire(request) => Self::retire_shared(shared, request).await,
            ClientRequest::Reclaim { target } => Self::reclaim_shared(shared, target).await,
            ClientRequest::Send(request) => Self::send_shared(shared, request).await,
            ClientRequest::Interrupt(request) => Self::interrupt_shared(shared, request).await,
            ClientRequest::Broadcast(request) => Self::broadcast_shared(shared, request).await,
            ClientRequest::SessionRetag(request) => {
                Self::session_retag_shared(shared, request).await
            }
            ClientRequest::SessionMark(request) => Self::session_mark_shared(shared, request).await,
            ClientRequest::HandoffArm(request) => Self::handoff_arm_shared(shared, request).await,
            ClientRequest::TimerStart(request) => Self::timer_start_shared(shared, request).await,
            ClientRequest::TimerFire { name } => Self::timer_fire_shared(shared, name).await,
            ClientRequest::Snapshot(request) => Self::snapshot_shared(shared, request).await,
            ClientRequest::SummaryInput(request) => {
                Self::summary_input_shared(shared, request).await
            }
            ClientRequest::Recruit(request) => Self::recruit_shared(shared, request).await,
            ClientRequest::DaemonStatus => {
                let state = shared.lock().await;
                Ok(vec![json!({
                    "type": "status",
                    "daemon": "running",
                    "hosts": state.hosts.len(),
                    "workstreams": state.workstreams.len(),
                    "sessions": state.sessions.len(),
                })])
            }
            ClientRequest::DaemonStop => Ok(vec![jsonl::ok("daemon_stop")]),
            ClientRequest::Hosts => Ok(vec![shared.lock().await.hosts_json()]),
            ClientRequest::Disconnect { alias } => shared.lock().await.disconnect(&alias),
            ClientRequest::Open(request) => shared.lock().await.open(request),
            ClientRequest::Label(request) => Self::label_shared(shared, request).await,
            ClientRequest::List => Ok(vec![shared.lock().await.workstream_list_json()]),
            ClientRequest::Show { workstream } => {
                Ok(vec![shared.lock().await.show(&workstream)?])
            }
            ClientRequest::SessionList => Ok(vec![shared.lock().await.session_list_json()]),
            ClientRequest::HandoffList { workstream } => {
                Ok(vec![shared.lock().await.handoff_list(&workstream)?])
            }
            ClientRequest::HandoffCancel {
                workstream,
                handoff_id,
            } => shared.lock().await.handoff_cancel(&workstream, &handoff_id),
            ClientRequest::TimerList { workstream } => Ok(vec![shared
                .lock()
                .await
                .timer_list_json(workstream.as_deref())]),
            ClientRequest::TimerStop { name } => shared.lock().await.timer_stop(&name),
            ClientRequest::Status {
                workstream,
                active_window_secs,
                idle_after_secs,
            } => Self::status_shared(shared, workstream, active_window_secs, idle_after_secs).await,
            ClientRequest::Events(request) => Ok(vec![shared.lock().await.events(request)?]),
        }
    }

    pub fn should_stop(request: &ClientRequest) -> bool {
        matches!(request, ClientRequest::DaemonStop)
    }

    async fn connect_shared(
        shared: Arc<Mutex<Self>>,
        request: ConnectRequest,
    ) -> anyhow::Result<Vec<Value>> {
        {
            let state = shared.lock().await;
            if state.hosts.contains_key(&request.alias) {
                bail!("host alias '{}' is already connected", request.alias);
            }
        }
        let config = SshConfig::parse(&request.ssh_uri)
            .with_context(|| format!("failed to parse ssh uri for '{}'", request.alias))?;
        let handle = config
            .connect_with_alias(&request.alias)
            .await
            .with_context(|| format!("failed to connect host '{}'", request.alias))?;
        let mut state = shared.lock().await;
        if state.hosts.contains_key(&request.alias) {
            bail!("host alias '{}' is already connected", request.alias);
        }
        state.fleet.register(&request.alias, handle)?;
        state.hosts.insert(
            request.alias.clone(),
            HostRecord {
                uri: request.ssh_uri.clone(),
                labels: request.labels,
                capacity: request.capacity,
                work_root: request.work_root,
            },
        );
        Ok(vec![json!({
            "type": "ok",
            "op": "connect",
            "host": request.alias,
            "uri": request.ssh_uri,
        })])
    }

    async fn scan_shared(shared: Arc<Mutex<Self>>, alias: String) -> anyhow::Result<Vec<Value>> {
        let handle = {
            let state = shared.lock().await;
            state.host_handle(&alias)?
        };
        let sessions = handle.list_sessions().await?;
        let tags_by_session = handle
            .list_tags_for_session_infos(tags::PREFIX, &sessions)
            .await?;

        let mut monitor_targets = Vec::new();
        let (hydrated, abort_tasks) = {
            let mut state = shared.lock().await;
            let mut hydrated = 0usize;
            let mut abort_tasks = Vec::new();
            let mut observed_targets = BTreeSet::new();
            for session in sessions {
                let target = SessionTarget::session_id(&alias, session.id.as_str())?;
                observed_targets.insert(target.clone());
                if let Some(reason) = state.stale_session_reuse_reason(&target, &session, None) {
                    abort_tasks
                        .extend(state.quarantine_stale_session_target(&target, &reason, None));
                } else if let Some(record) = state.sessions.get_mut(&target) {
                    record.observe_tmux_session(&session);
                }
                let tags = tags_by_session
                    .get(&session.id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let parsed = ParsedTags::from_tags(tags);
                if parsed.managed || parsed.workstream.is_some() || parsed.state.is_some() {
                    hydrated += 1;
                    let state_value = parsed.state.unwrap_or(AgentState::Idle);
                    let record = state.sessions.entry(target.clone()).or_insert_with(|| {
                        SessionRecord::from_target(&target, state_value, parsed.updated_at)
                    });
                    record.observe_tmux_session(&session);
                    record.state = state_value;
                    record.role = parsed.role.clone();
                    record.agent = parsed.agent.clone();
                    record.workstream = parsed.workstream.clone();
                    record.last_report_kind = parsed.last_report_kind.clone();
                    record.last_report_summary = parsed.last_report_summary.clone();
                    record.cwd = parsed.cwd.clone();
                    record.context_domains = parsed.context_domains.clone();
                    record.context_specialties = parsed.context_specialties.clone();
                    record.context_summary = parsed.context_summary.clone();
                    record.last_workstream = parsed.last_workstream.clone();
                    record.last_workstream_title = parsed.last_workstream_title.clone();
                    record.mmux_label = parsed.mmux_label.clone();
                    record.mmux_previous_selected_key = parsed.mmux_previous_selected_key.clone();

                    if let Some(workstream_name) = parsed.workstream {
                        let title = parsed
                            .workstream_title
                            .unwrap_or_else(|| workstream_name.clone());
                        if !state.workstreams.contains_key(&workstream_name) {
                            let generation = state.next_generation();
                            let workstream = WorkstreamRecord::new(
                                title.clone(),
                                None,
                                None,
                                parsed.mmux_label.clone(),
                                WorkstreamSettings::default(),
                                generation,
                            );
                            state
                                .workstreams
                                .insert(workstream_name.clone(), workstream);
                        }
                        let workstream = state.workstream_mut(&workstream_name)?;
                        if let Some(label) = parsed.mmux_label.clone() {
                            workstream.merge_hydrated_mmux_label(label);
                        }
                        workstream.sessions.insert(target.clone());
                        monitor_targets.push(target.clone());
                    }
                }
            }
            let stale_targets: Vec<SessionTarget> = state
                .sessions
                .keys()
                .filter_map(|target| {
                    if target.host_alias() == alias.as_str() && !observed_targets.contains(target) {
                        Some(target.clone())
                    } else {
                        None
                    }
                })
                .collect();
            for target in stale_targets {
                let reason = format!("tmux session {target} is no longer present");
                abort_tasks.extend(state.quarantine_stale_session_target(&target, &reason, None));
            }
            (hydrated, abort_tasks)
        };
        for task in abort_tasks {
            task.abort();
        }

        for target in monitor_targets {
            let _ = Self::ensure_monitoring_target_shared(Arc::clone(&shared), &target).await;
        }

        Ok(vec![json!({
            "type": "ok",
            "op": "scan",
            "host": alias,
            "hydrated_sessions": hydrated,
        })])
    }

    async fn join_shared(
        shared: Arc<Mutex<Self>>,
        request: JoinRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let (handle, meta) = {
            let state = shared.lock().await;
            state.ensure_workstream_open(&request.workstream)?;
            (
                state.host_handle(target.host_alias())?,
                state.workstream_meta(&request.workstream)?,
            )
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::prepare_resolved_target_for_adoption_shared(Arc::clone(&shared), &resolved).await?;
        Self::write_assignment_to_target(
            &resolved.target,
            &meta,
            AssignmentTags {
                workstream: &request.workstream,
                role: &request.role,
                identity: resolved.target.session_name(),
                agent: None,
                cwd: None,
                state: AgentState::Busy,
            },
        )
        .await?;
        if let Some(label) = &meta.mmux_label {
            Self::apply_mmux_workstream_label(&resolved.target, label).await?;
        }
        Self::ensure_monitoring_target_shared(Arc::clone(&shared), &resolved.spec).await?;
        if let Some(task) = &request.task {
            Self::send_text_to_resolved(
                &resolved,
                &managed_prompt(
                    &request.workstream,
                    &resolved.spec,
                    Some(&request.role),
                    None,
                    Some(task),
                ),
                PasteMode::Bracketed,
                true,
            )
            .await?;
        }
        let cursor = {
            let stable_target = resolved.spec.clone();
            let mut state = shared.lock().await;
            state.add_session_to_workstream(
                &request.workstream,
                stable_target.clone(),
                request.role,
                None,
                None,
                AgentState::Busy,
            )?;
            if let (Some(record), Some(session)) = (
                state.sessions.get_mut(&stable_target),
                resolved.target.session_info(),
            ) {
                record.observe_tmux_session(session);
            }
            let mut cursor = state.record_event(
                &request.workstream,
                EventDraft::new("joined")
                    .target(&stable_target)
                    .state(AgentState::Busy),
            )?;
            if let Some(label) = &meta.mmux_label {
                cursor = state.record_event(
                    &request.workstream,
                    EventDraft::new("mmux_label_applied")
                        .target(&stable_target)
                        .summary(format!("mmux label applied: {label}")),
                )?;
            }
            cursor
        };
        Ok(vec![json!({
            "type": "ok",
            "op": "join",
            "workstream": request.workstream,
            "target": resolved.spec.to_string(),
            "cursor": cursor,
        })])
    }

    async fn new_session_shared(
        shared: Arc<Mutex<Self>>,
        request: NewRequest,
    ) -> anyhow::Result<Vec<Value>> {
        if !request.cwd.is_absolute() {
            bail!("--cwd must be an absolute path");
        }
        let target: SessionTarget = request.target.parse()?;
        let (handle, meta) = {
            let state = shared.lock().await;
            state.ensure_workstream_open(&request.workstream)?;
            (
                state.host_handle(target.host_alias())?,
                state.workstream_meta(&request.workstream)?,
            )
        };
        validate_agent_executable(&handle, &request.agent).await?;
        let command = bootstrap_command(&request.cwd, &request.agent);
        let env = session_environment(&request.workstream, &request.role)?;
        let opts = CreateSessionOptions {
            command: Some(command),
            initial_environment: env,
            ..Default::default()
        };
        let tmux_target = match handle.create_session(target.session_name(), &opts).await {
            Ok(tmux_target) => tmux_target,
            Err(error) if is_created_session_not_found(&error, target.session_name()) => {
                bail!(
                    "agent session {} on {} exited immediately during startup while running agent executable {}; tmux did not report it as live after creation",
                    target.session_name(),
                    target.host_alias(),
                    request.agent
                );
            }
            Err(error) => return Err(error.into()),
        };
        let stable_target = SessionTarget::session_id(
            target.host_alias(),
            tmux_target
                .session_id()
                .with_context(|| "created session target missing session id")?,
        )?;
        let resolved = ResolvedTarget {
            spec: stable_target.clone(),
            host: handle,
            target: tmux_target,
        };
        Self::prepare_resolved_target_for_adoption_shared(Arc::clone(&shared), &resolved).await?;
        Self::write_assignment_to_target(
            &resolved.target,
            &meta,
            AssignmentTags {
                workstream: &request.workstream,
                role: &request.role,
                identity: resolved.target.session_name(),
                agent: Some(&request.agent),
                cwd: Some(&request.cwd),
                state: AgentState::Busy,
            },
        )
        .await?;
        if let Some(label) = &meta.mmux_label {
            Self::apply_mmux_workstream_label(&resolved.target, label).await?;
        }
        Self::ensure_monitoring_target_shared(Arc::clone(&shared), &resolved.spec).await?;
        if let Some(task) = &request.task {
            Self::send_text_to_resolved(
                &resolved,
                &managed_prompt(
                    &request.workstream,
                    &stable_target,
                    Some(&request.role),
                    Some(&request.cwd),
                    Some(task),
                ),
                PasteMode::Bracketed,
                true,
            )
            .await?;
        }
        let cursor = {
            let mut state = shared.lock().await;
            state.add_session_to_workstream(
                &request.workstream,
                stable_target.clone(),
                request.role,
                Some(request.agent),
                Some(request.cwd.clone()),
                AgentState::Busy,
            )?;
            if let (Some(record), Some(session)) = (
                state.sessions.get_mut(&stable_target),
                resolved.target.session_info(),
            ) {
                record.observe_tmux_session(session);
            }
            let mut cursor = state.record_event(
                &request.workstream,
                EventDraft::new("created")
                    .target(&stable_target)
                    .state(AgentState::Busy),
            )?;
            if let Some(label) = &meta.mmux_label {
                cursor = state.record_event(
                    &request.workstream,
                    EventDraft::new("mmux_label_applied")
                        .target(&stable_target)
                        .summary(format!("mmux label applied: {label}")),
                )?;
            }
            cursor
        };
        Ok(vec![json!({
            "type": "ok",
            "op": "new",
            "workstream": request.workstream,
            "target": stable_target.to_string(),
            "cursor": cursor,
        })])
    }

    async fn close_shared(
        shared: Arc<Mutex<Self>>,
        request: CloseRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let (sessions, title) = {
            let state = shared.lock().await;
            (
                state.workstream(&request.workstream)?.sessions.clone(),
                state.workstream(&request.workstream)?.title.clone(),
            )
        };
        let mut changes = Vec::new();
        let mut standby_notified = 0usize;
        let mut standby_failed = 0usize;
        let mut mmux_labels_cleared = 0usize;
        let mut mmux_label_restore_failed = 0usize;
        for target in &sessions {
            let handle = {
                let state = shared.lock().await;
                state.host_handle(target.host_alias()).ok()
            };
            if let Some(handle) = handle {
                if let Ok(resolved) = Self::resolve_target(handle, target.clone()).await {
                    if let Err(err) = Self::ensure_resolved_target_fresh_shared(
                        Arc::clone(&shared),
                        &resolved,
                        None,
                        None,
                    )
                    .await
                    {
                        if request.standby_agents {
                            standby_failed += 1;
                        }
                        shared.lock().await.record_event(
                            &request.workstream,
                            EventDraft::new("session_stale")
                                .target(target)
                                .summary(format!("session skipped during closeout: {err}")),
                        )?;
                        continue;
                    }
                    if request.standby_agents {
                        let message = format!(
                            "Orchestrator closeout: workstream '{}' is closed. Please stand by and do not start new work unless assigned.",
                            request.workstream
                        );
                        match Self::send_text_to_resolved(
                            &resolved,
                            &message,
                            PasteMode::Bracketed,
                            true,
                        )
                        .await
                        {
                            Ok(()) => {
                                standby_notified += 1;
                                shared.lock().await.record_event(
                                    &request.workstream,
                                    EventDraft::new("standby_sent")
                                        .target(target)
                                        .text(message.clone())
                                        .summary("standby instruction sent"),
                                )?;
                            }
                            Err(err) => {
                                standby_failed += 1;
                                shared.lock().await.record_event(
                                    &request.workstream,
                                    EventDraft::new("standby_failed")
                                        .target(target)
                                        .summary(format!("standby send failed: {err}")),
                                )?;
                            }
                        }
                    }
                    match Self::clear_mmux_workstream_label(&resolved.target, &request.workstream)
                        .await
                    {
                        Ok(cleanup) => {
                            if cleanup.cleared {
                                mmux_labels_cleared += 1;
                                shared.lock().await.record_event(
                                    &request.workstream,
                                    EventDraft::new("mmux_label_cleared")
                                        .target(target)
                                        .summary("mmux workstream label cleared"),
                                )?;
                            }
                            if cleanup.skipped_workstream_mismatch {
                                shared.lock().await.record_event(
                                    &request.workstream,
                                    EventDraft::new("mmux_label_cleanup_skipped")
                                        .target(target)
                                        .summary(
                                            "mmux label cleanup skipped: session belongs to a different workstream",
                                        ),
                                )?;
                            }
                        }
                        Err(err) => {
                            mmux_label_restore_failed += 1;
                            shared.lock().await.record_event(
                                &request.workstream,
                                EventDraft::new("mmux_label_restore_failed")
                                    .target(target)
                                    .summary(format!("mmux label cleanup failed: {err}")),
                            )?;
                        }
                    }
                    let mut pairs = vec![
                        ("last-workstream", request.workstream.clone()),
                        ("last-workstream-title", title.clone()),
                    ];
                    if let Some(summary) = &request.summary {
                        pairs.push(("context-summary", summary.clone()));
                    }
                    if let Some(domain) = &request.domain {
                        pairs.push(("context-domains", domain.clone()));
                    }
                    if !request.specialties.is_empty() {
                        pairs.push(("context-specialties", request.specialties.join(",")));
                    }
                    resolved
                        .target
                        .tags(tags::PREFIX)
                        .await?
                        .set_many(pairs)
                        .await?;
                    resolved
                        .target
                        .tags(tags::PREFIX)
                        .await?
                        .unset_many([
                            "workstream",
                            "workstream-title",
                            "workstream-state",
                            "workstream-goal",
                            "workstream-domain",
                            "role",
                        ])
                        .await?;
                    changes.push(
                        Self::apply_session_state_shared(
                            Arc::clone(&shared),
                            target,
                            AgentState::Available,
                            None,
                        )
                        .await?,
                    );
                }
            }
            let mut state = shared.lock().await;
            if let Some(record) = state.sessions.get_mut(target) {
                record.workstream = None;
                record.mmux_label = None;
                record.mmux_previous_selected_key = None;
                record.last_workstream = Some(request.workstream.clone());
                record.last_workstream_title = Some(title.clone());
                if let Some(summary) = &request.summary {
                    record.context_summary = Some(summary.clone());
                }
                if let Some(domain) = &request.domain {
                    record.context_domains.insert(domain.clone());
                }
                for specialty in &request.specialties {
                    record.context_specialties.insert(specialty.clone());
                }
            }
        }
        let cursor = {
            let mut state = shared.lock().await;
            let cursor = if let Some(summary) = request.summary.clone() {
                state.record_event(
                    &request.workstream,
                    EventDraft::new("closed")
                        .text(summary.clone())
                        .summary(summary),
                )?
            } else {
                state.record_event(&request.workstream, EventDraft::new("closed"))?
            };
            let workstream = state.workstream_mut(&request.workstream)?;
            workstream.state = WorkstreamState::Closed;
            workstream.sessions.clear();
            cursor
        };
        let stopped_timers = if request.stop_timers {
            shared
                .lock()
                .await
                .stop_timers_for_workstream(&request.workstream)
        } else {
            Vec::new()
        };
        let mut records = vec![json!({
            "type": "ok",
            "op": "close",
            "workstream": request.workstream,
            "cursor": cursor,
            "standby_agents": request.standby_agents,
            "standby_notified": standby_notified,
            "standby_failed": standby_failed,
            "mmux_labels_cleared": mmux_labels_cleared,
            "mmux_label_restore_failed": mmux_label_restore_failed,
            "stopped_timers": stopped_timers,
        })];
        records.extend(Self::fire_handoffs_for_changes_shared(shared, changes).await?);
        Ok(records)
    }

    async fn leave_shared(
        shared: Arc<Mutex<Self>>,
        request: LeaveRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let stable_target = resolved.spec.clone();
        let mmux_cleanup =
            match Self::clear_mmux_workstream_label(&resolved.target, &request.workstream).await {
                Ok(cleanup) => cleanup,
                Err(err) => {
                    shared.lock().await.record_event(
                        &request.workstream,
                        EventDraft::new("mmux_label_restore_failed")
                            .target(&stable_target)
                            .summary(format!("mmux label cleanup failed: {err}")),
                    )?;
                    MmuxLabelCleanup::default()
                }
            };
        resolved
            .target
            .tags(tags::PREFIX)
            .await?
            .unset_many([
                "workstream",
                "workstream-title",
                "workstream-state",
                "workstream-goal",
                "workstream-domain",
                "role",
            ])
            .await?;
        let change = if request.available {
            Some(
                Self::apply_resolved_session_state_shared(
                    Arc::clone(&shared),
                    &resolved,
                    AgentState::Available,
                    None,
                )
                .await?,
            )
        } else {
            None
        };
        let cursor = {
            let mut state = shared.lock().await;
            if let Some(record) = state.sessions.get_mut(&stable_target) {
                record.workstream = None;
                record.mmux_label = None;
                record.mmux_previous_selected_key = None;
            }
            let mut event = EventDraft::new("left").target(&stable_target);
            if request.available {
                event = event.state(AgentState::Available);
            }
            let cursor = state.record_event(&request.workstream, event)?;
            if mmux_cleanup.cleared {
                state.record_event(
                    &request.workstream,
                    EventDraft::new("mmux_label_cleared")
                        .target(&stable_target)
                        .summary("mmux workstream label cleared"),
                )?;
            }
            if mmux_cleanup.skipped_workstream_mismatch {
                state.record_event(
                    &request.workstream,
                    EventDraft::new("mmux_label_cleanup_skipped")
                        .target(&stable_target)
                        .summary(
                            "mmux label cleanup skipped: session belongs to a different workstream",
                        ),
                )?;
            }
            let workstream = state.workstream_mut(&request.workstream)?;
            workstream.sessions.remove(&stable_target);
            cursor
        };
        let mut records = vec![json!({
            "type": "ok",
            "op": "leave",
            "workstream": request.workstream,
            "target": stable_target.to_string(),
            "cursor": cursor,
            "mmux_label_cleared": mmux_cleanup.cleared,
            "mmux_previous_selected_key_restored": mmux_cleanup.restored_previous,
            "mmux_selected_key_unchanged": mmux_cleanup.selected_key_unchanged,
            "mmux_label_cleanup_skipped": mmux_cleanup.skipped_workstream_mismatch,
        })];
        if let Some(change) = change {
            records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        }
        Ok(records)
    }

    async fn retire_shared(
        shared: Arc<Mutex<Self>>,
        request: RetireRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.ensure_workstream_open(&request.workstream)?;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let stable_target = resolved.spec.clone();
        {
            let state = shared.lock().await;
            state.ensure_target_in_workstream(&request.workstream, &stable_target)?;
        }
        let change = Self::apply_resolved_session_state_shared(
            Arc::clone(&shared),
            &resolved,
            AgentState::Quarantined,
            None,
        )
        .await?;
        let cursor = {
            let mut state = shared.lock().await;
            state.record_event(
                &request.workstream,
                EventDraft::new("retired")
                    .target(&stable_target)
                    .state(AgentState::Quarantined),
            )?
        };
        let previous_state = change.previous_state.map(AgentState::as_str);
        let mut records = vec![json!({
            "type": "ok",
            "op": "retire",
            "workstream": request.workstream,
            "target": stable_target.to_string(),
            "state": AgentState::Quarantined.as_str(),
            "previous_state": previous_state,
            "cursor": cursor,
        })];
        records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        Ok(records)
    }

    async fn reclaim_shared(
        shared: Arc<Mutex<Self>>,
        target: String,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let stable_target = resolved.spec.clone();
        let parsed = Self::parsed_mstream_tags(&resolved.target).await?;
        if !parsed.managed {
            bail!(
                "target {} is not managed by mstream; refusing to reclaim unmanaged tmux session",
                stable_target
            );
        }
        let tagged_state = parsed
            .state
            .with_context(|| format!("target {stable_target} has no mstream state tag"))?;
        if tagged_state != AgentState::Quarantined {
            bail!(
                "target {} is {}, not quarantined; retire it before reclaim",
                stable_target,
                tagged_state.as_str()
            );
        }
        let workstream = parsed.workstream.with_context(|| {
            format!("target {stable_target} has no workstream tag for reclaim audit log")
        })?;
        {
            let state = shared.lock().await;
            state.workstream(&workstream)?;
        }

        resolved.target.kill().await?;

        let (cursor, abort_tasks) = {
            let mut state = shared.lock().await;
            let cursor = state.record_event(
                &workstream,
                EventDraft::new("reclaimed")
                    .target(&stable_target)
                    .state(AgentState::Quarantined),
            )?;
            let reason = format!("target {stable_target} reclaimed");
            let abort_tasks = state.deregister_session_target(&stable_target, &reason, None);
            (cursor, abort_tasks)
        };
        for task in abort_tasks {
            task.abort();
        }

        Ok(vec![json!({
            "type": "ok",
            "op": "reclaim",
            "workstream": workstream,
            "target": stable_target.to_string(),
            "cursor": cursor,
        })])
    }

    async fn send_shared(
        shared: Arc<Mutex<Self>>,
        request: SendRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let stable_target = resolved.spec.clone();
        let current_state = {
            let state = shared.lock().await;
            state.ensure_target_in_workstream(&request.workstream, &stable_target)?;
            let current_state = state
                .sessions
                .get(&stable_target)
                .map(|record| record.state)
                .unwrap_or(AgentState::Idle);
            if let Some(required) = request.require_state {
                if current_state != required {
                    bail!(
                        "target {} is {}, not required state {}",
                        stable_target,
                        current_state.as_str(),
                        required.as_str()
                    );
                }
            }
            current_state
        };
        if request.interrupt_first {
            Self::send_interrupt_to_resolved(&resolved, InterruptKey::Esc).await?;
            sleep(Duration::from_millis(request.settle_ms)).await;
        }
        Self::send_text_to_resolved(&resolved, &request.text, request.paste_mode, request.enter)
            .await?;
        let change = if let Some(state) = request.set_state {
            Some(
                Self::apply_resolved_session_state_shared(
                    Arc::clone(&shared),
                    &resolved,
                    state,
                    None,
                )
                .await?,
            )
        } else {
            None
        };
        let (state_after, cursor) = {
            let mut state = shared.lock().await;
            let state_after = state
                .sessions
                .get(&stable_target)
                .map(|record| record.state)
                .unwrap_or(current_state);
            let cursor = state.record_event(
                &request.workstream,
                EventDraft::new("message_sent")
                    .target(&stable_target)
                    .text(request.text)
                    .state(state_after),
            )?;
            (state_after, cursor)
        };
        let mut records = vec![json!({
            "type": "ok",
            "op": "send",
            "workstream": request.workstream,
            "target": stable_target.to_string(),
            "target_state": state_after.as_str(),
            "mid_generation_risk": current_state == AgentState::Busy && !request.interrupt_first,
            "paste_mode": request.paste_mode.as_str(),
            "enter": request.enter,
            "cursor": cursor,
        })];
        if let Some(change) = change {
            records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        }
        Ok(records)
    }

    async fn interrupt_shared(
        shared: Arc<Mutex<Self>>,
        request: InterruptRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let stable_target = resolved.spec.clone();
        let event_context = {
            let state = shared.lock().await;
            state.sessions.get(&stable_target).and_then(|record| {
                record
                    .workstream
                    .clone()
                    .map(|workstream| (workstream, record.state))
            })
        };
        Self::send_interrupt_to_resolved(&resolved, request.key).await?;
        let mut record = json!({
            "type": "ok",
            "op": "interrupt",
            "target": stable_target.to_string(),
            "key": request.key.as_str(),
        });
        if let Some((workstream, state_value)) = event_context {
            let cursor = shared.lock().await.record_event(
                &workstream,
                EventDraft::new("interrupted")
                    .target(&stable_target)
                    .text(request.key.as_str())
                    .state(state_value),
            )?;
            if let Some(object) = record.as_object_mut() {
                object.insert("workstream".to_string(), json!(workstream));
                object.insert("cursor".to_string(), json!(cursor));
            }
        }
        Ok(vec![record])
    }

    async fn broadcast_shared(
        shared: Arc<Mutex<Self>>,
        request: BroadcastRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let targets = Self::broadcast_targets_shared(Arc::clone(&shared), &request).await?;
        let mut records = Vec::new();
        let mut sent = 0usize;
        for target in targets {
            Self::send_text_to_resolved(
                &target.target,
                &request.text,
                request.paste_mode,
                request.enter,
            )
            .await?;
            Self::touch_resolved_session_shared(Arc::clone(&shared), &target.target).await?;
            sent += 1;
            let cursor = shared.lock().await.record_event(
                &request.workstream,
                EventDraft::new("broadcast_sent")
                    .target(&target.target.spec)
                    .text(request.text.clone())
                    .state(target.state),
            )?;
            records.push(json!({
                "type": "ok",
                "op": "broadcast_sent",
                "workstream": request.workstream,
                "target": target.target.spec.to_string(),
                "cursor": cursor,
            }));
        }
        records.push(json!({
            "type": "ok",
            "op": "broadcast",
            "workstream": request.workstream,
            "sent": sent,
        }));
        Ok(records)
    }

    async fn session_retag_shared(
        shared: Arc<Mutex<Self>>,
        request: SessionRetagRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let op = if request.new_name.is_some() {
            "rename"
        } else {
            "session_retag"
        };
        let requested_new_name = request
            .new_name
            .as_deref()
            .map(validate_session_display_name)
            .transpose()?;
        let requested_role = request
            .role
            .as_deref()
            .map(|role| validate_retag_text("--role", role))
            .transpose()?;
        let requested_workstream = request
            .workstream
            .as_deref()
            .map(|workstream| validate_retag_text("--workstream", workstream))
            .transpose()?;
        let requested_mmux_label = request
            .mmux_label
            .as_deref()
            .map(validate_mmux_label)
            .transpose()?;

        let target: SessionTarget = request.target.parse()?;
        ensure_session_target(&target)?;
        let handle = {
            let state = shared.lock().await;
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target).await?;
        ensure_session_target(&resolved.spec)?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;

        let stable_target = resolved.spec.clone();
        let plan = {
            let state = shared.lock().await;
            state.session_retag_plan(
                &stable_target,
                &resolved.target,
                requested_new_name.as_deref(),
                requested_role.clone(),
                requested_workstream.clone(),
                requested_mmux_label.clone(),
            )?
        };
        let renamed = requested_new_name
            .as_deref()
            .is_some_and(|name| name != plan.old_name.as_str());
        let tmux_target = if renamed {
            resolved.target.rename(&plan.new_name).await?
        } else {
            resolved.target.clone()
        };
        if renamed {
            Self::observe_resolved_session_shared(
                Arc::clone(&shared),
                &stable_target,
                &tmux_target,
            )
            .await;
        }

        if let Some(workstream) = &plan.clear_mmux_workstream {
            Self::clear_mmux_workstream_label(&tmux_target, workstream).await?;
        }
        if let (Some(workstream), Some(meta), Some(role)) = (
            plan.new_workstream.as_deref(),
            plan.workstream_meta.as_ref(),
            plan.role.as_deref(),
        ) {
            Self::write_assignment_to_target(
                &tmux_target,
                meta,
                AssignmentTags {
                    workstream,
                    role,
                    identity: &plan.new_name,
                    agent: plan.agent.as_deref(),
                    cwd: plan.cwd.as_deref(),
                    state: plan.state,
                },
            )
            .await?;
        } else {
            Self::write_session_identity_to_target(
                &tmux_target,
                &plan.new_name,
                plan.role.as_deref(),
            )
            .await?;
        }
        if plan.requested_mmux_label.is_none() {
            if let Some(label) = &plan.mmux_label {
                Self::apply_mmux_workstream_label(&tmux_target, label).await?;
            }
        }

        let requested_mmux_apply = if let Some(label) = &plan.requested_mmux_label {
            let workstream = plan
                .new_workstream
                .as_ref()
                .or(plan.old_workstream.as_ref())
                .with_context(|| "--mmux-label requires an effective workstream")?;
            Some((workstream.clone(), label.clone()))
        } else {
            None
        };

        let (cursor, stale_handoffs_canceled) = {
            let mut state = shared.lock().await;
            state.apply_session_retag_plan(&stable_target, &tmux_target, &plan, renamed)?
        };

        let mmux_apply_result = if let Some((workstream, label)) = requested_mmux_apply {
            Some(
                Self::apply_mmux_label_to_workstream_shared(
                    Arc::clone(&shared),
                    &workstream,
                    &label,
                )
                .await?,
            )
        } else {
            None
        };

        Ok(vec![json!({
            "type": "ok",
            "op": op,
            "target": stable_target.to_string(),
            "renamed": renamed,
            "old_name": plan.old_name,
            "new_name": plan.new_name,
            "role": plan.role,
            "workstream": plan.new_workstream,
            "mmux_label": plan.mmux_label,
            "mmux_label_cleared": plan.clear_mmux_workstream.is_some(),
            "mmux_labels_applied": mmux_apply_result.as_ref().map(|result| result.applied),
            "mmux_label_apply_failed": mmux_apply_result.as_ref().map(|result| result.failed),
            "stale_handoffs_canceled": stale_handoffs_canceled,
            "cursor": cursor,
        })])
    }

    async fn apply_mmux_label_to_workstream_shared(
        shared: Arc<Mutex<Self>>,
        workstream: &str,
        label: &str,
    ) -> anyhow::Result<MmuxLabelApplyResult> {
        let targets = {
            let state = shared.lock().await;
            state.workstream_session_targets(workstream)?
        };

        let mut result = MmuxLabelApplyResult::default();
        for target in targets {
            let handle = {
                let state = shared.lock().await;
                state.host_handle(target.host_alias()).ok()
            };
            let apply_result = match handle {
                Some(handle) => match Self::resolve_target(handle, target.clone()).await {
                    Ok(resolved) => match Self::ensure_resolved_target_fresh_shared(
                        Arc::clone(&shared),
                        &resolved,
                        None,
                        None,
                    )
                    .await
                    {
                        Ok(()) => Self::apply_mmux_workstream_label(&resolved.target, label).await,
                        Err(err) => Err(err),
                    },
                    Err(err) => Err(err),
                },
                None => Err(anyhow::anyhow!(
                    "host alias '{}' is not connected",
                    target.host_alias()
                )),
            };
            match apply_result {
                Ok(()) => {
                    result.applied += 1;
                    let mut state = shared.lock().await;
                    let assigned_to_workstream = state
                        .sessions
                        .get(&target)
                        .is_some_and(|record| record.workstream.as_deref() == Some(workstream));
                    if let Some(record) = state.sessions.get_mut(&target) {
                        record.mmux_label = Some(label.to_string());
                    }
                    if assigned_to_workstream {
                        state
                            .workstream_mut(workstream)?
                            .sessions
                            .insert(target.clone());
                    }
                    state.record_event(
                        workstream,
                        EventDraft::new("mmux_label_applied")
                            .target(&target)
                            .summary(format!("mmux label applied: {label}")),
                    )?;
                }
                Err(err) => {
                    result.failed += 1;
                    shared.lock().await.record_event(
                        workstream,
                        EventDraft::new("mmux_label_apply_failed")
                            .target(&target)
                            .summary(format!("mmux label apply failed: {err}")),
                    )?;
                }
            }
        }

        Ok(result)
    }

    async fn session_mark_shared(
        shared: Arc<Mutex<Self>>,
        request: SessionMarkRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let change = Self::apply_session_state_shared(
            Arc::clone(&shared),
            &target,
            request.state,
            Some(&request.summary),
        )
        .await?;
        let mut records = Vec::new();
        let workstream = {
            let state = shared.lock().await;
            state
                .sessions
                .get(&change.target)
                .and_then(|record| record.workstream.clone())
        };
        if let Some(workstream) = workstream {
            let kind = request.state.event_kind().unwrap_or("session_marked");
            let cursor = shared.lock().await.record_event(
                &workstream,
                EventDraft::new(kind)
                    .target(&change.target)
                    .state(request.state)
                    .summary(request.summary.clone()),
            )?;
            records.push(json!({
                "type": "event",
                "kind": kind,
                "workstream": workstream,
                "target": change.target.to_string(),
                "state": request.state.as_str(),
                "summary": request.summary,
                "cursor": cursor,
            }));
        } else {
            records.push(json!({
                "type": "ok",
                "op": "session_mark",
                "target": change.target.to_string(),
                "state": request.state.as_str(),
            }));
        }
        records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        Ok(records)
    }

    async fn label_shared(
        shared: Arc<Mutex<Self>>,
        request: LabelRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let label = validate_mmux_label(&request.mmux_label)?;
        {
            let mut state = shared.lock().await;
            state.ensure_workstream_open(&request.workstream)?;
            let workstream = state.workstream_mut(&request.workstream)?;
            workstream.mmux_label = Some(label.clone());
            workstream.mmux_label_conflicts.clear();
        }

        let apply_result = Self::apply_mmux_label_to_workstream_shared(
            Arc::clone(&shared),
            &request.workstream,
            &label,
        )
        .await?;

        let cursor = shared.lock().await.record_event(
            &request.workstream,
            EventDraft::new("mmux_label_set").summary(format!("mmux label set: {label}")),
        )?;
        Ok(vec![json!({
            "type": "ok",
            "op": "label",
            "workstream": request.workstream,
            "mmux_label": label,
            "applied": apply_result.applied,
            "failed": apply_result.failed,
            "cursor": cursor,
        })])
    }

    async fn handoff_arm_shared(
        shared: Arc<Mutex<Self>>,
        request: HandoffArmRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let from_input: SessionTarget = request.from.parse()?;
        let to_input: SessionTarget = request.to.parse()?;
        let (from_handle, to_handle) = {
            let state = shared.lock().await;
            (
                state.host_handle(from_input.host_alias())?,
                state.host_handle(to_input.host_alias())?,
            )
        };
        let from_resolved = Self::resolve_target(from_handle, from_input).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &from_resolved, None, None)
            .await?;
        let to_resolved = Self::resolve_target(to_handle, to_input).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &to_resolved, None, None)
            .await?;
        let from = from_resolved.spec.clone();
        let to = to_resolved.spec.clone();
        let from_tmux_session_created = from_resolved
            .target
            .session_info()
            .map(|session| session.created);
        let to_tmux_session_created = to_resolved
            .target
            .session_info()
            .map(|session| session.created);
        let immediate = {
            let mut state = shared.lock().await;
            state.ensure_target_in_workstream(&request.workstream, &from)?;
            state.ensure_target_in_workstream(&request.workstream, &to)?;
            let id = format!("h{}", state.next_handoff_id);
            state.next_handoff_id += 1;
            let handoff = HandoffRecord {
                id: id.clone(),
                from: from.clone(),
                to,
                from_tmux_session_created,
                to_tmux_session_created,
                on: request.on,
                task: request.task,
                only_on_transition: request.only_on_transition,
                fired: false,
                canceled_reason: None,
                created_at: tags::now_tag(),
            };
            state
                .workstream_mut(&request.workstream)?
                .handoffs
                .insert(id.clone(), handoff);
            let already_met = state
                .sessions
                .get(&from)
                .is_some_and(|record| record.state == request.on);
            if already_met && !request.only_on_transition {
                state.claim_handoff(&request.workstream, &id)?
            } else {
                return Ok(vec![json!({
                    "type": "ok",
                    "op": "handoff_arm",
                    "workstream": request.workstream,
                    "handoff_id": id,
                    "from": from.to_string(),
                    "on": request.on.as_str(),
                    "only_on_transition": request.only_on_transition,
                })]);
            }
        };
        let (record, change) =
            Self::fire_claimed_handoff_shared(Arc::clone(&shared), &request.workstream, immediate)
                .await?;
        let mut records = vec![record];
        records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        Ok(records)
    }

    async fn timer_start_shared(
        shared: Arc<Mutex<Self>>,
        request: TimerStartRequest,
    ) -> anyhow::Result<Vec<Value>> {
        if request.name.trim().is_empty() {
            bail!("timer name cannot be empty");
        }
        if request.every_secs == 0 {
            bail!("timer interval must be greater than zero");
        }
        if request.prompt.is_empty() {
            bail!("timer prompt cannot be empty");
        }
        let submit_retries = if request.enter {
            request.submit_retries
        } else {
            0
        };

        let target: SessionTarget = request.target.parse()?;
        let handle = {
            let state = shared.lock().await;
            if state.timers.contains_key(&request.name) {
                bail!("timer '{}' already exists", request.name);
            }
            state.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
            .await?;
        let tmux_session_created = resolved
            .target
            .session_info()
            .map(|session| session.created);
        let stable_target = resolved.spec;

        let (generation, next_fire_at) = {
            let mut state = shared.lock().await;
            if state.timers.contains_key(&request.name) {
                bail!("timer '{}' already exists", request.name);
            }
            let generation = state.next_timer_generation;
            state.next_timer_generation += 1;
            let started_at = Utc::now();
            let next_fire_at = add_secs(started_at, request.every_secs);
            state.timers.insert(
                request.name.clone(),
                TimerRecord {
                    name: request.name.clone(),
                    workstream: request.workstream.clone(),
                    target: stable_target.clone(),
                    tmux_session_created,
                    every_secs: request.every_secs,
                    prompt: request.prompt,
                    enter: request.enter,
                    submit_retries,
                    submit_retry_delay_ms: request.submit_retry_delay_ms,
                    input_quiet_for_secs: request.input_quiet_for_secs,
                    generation,
                    started_at,
                    next_fire_at,
                    last_fired_at: None,
                    fire_count: 0,
                    defer_count: 0,
                    last_deferred_at: None,
                    last_defer_reason: None,
                    last_input_activity: None,
                    last_error: None,
                    task: None,
                },
            );
            (generation, next_fire_at)
        };
        let name = request.name.clone();
        let task = tokio::spawn(Self::timer_loop(Arc::clone(&shared), name, generation));
        {
            let mut state = shared.lock().await;
            if let Some(timer) = state.timers.get_mut(&request.name) {
                if timer.generation == generation {
                    timer.task = Some(task);
                } else {
                    task.abort();
                }
            } else {
                task.abort();
            }
        }

        Ok(vec![json!({
            "type": "ok",
            "op": "timer_start",
            "name": request.name,
            "workstream": request.workstream,
            "target": stable_target.to_string(),
            "every_secs": request.every_secs,
            "enter": request.enter,
            "submit_retries": submit_retries,
            "submit_retry_delay_ms": request.submit_retry_delay_ms,
            "input_quiet_for_secs": option_u64_json(request.input_quiet_for_secs),
            "generation": generation,
            "next_fire_at": datetime_option_json(next_fire_at),
        })])
    }

    async fn timer_fire_shared(
        shared: Arc<Mutex<Self>>,
        name: String,
    ) -> anyhow::Result<Vec<Value>> {
        let outcome = Self::timer_fire_once_shared(Arc::clone(&shared), &name, None, false).await?;
        Ok(vec![timer_fire_outcome_json(outcome)])
    }

    async fn timer_loop(shared: Arc<Mutex<Self>>, name: String, generation: u64) {
        loop {
            let sleep_for = {
                let state = shared.lock().await;
                let Some(timer) = state.timers.get(&name) else {
                    break;
                };
                if timer.generation != generation {
                    break;
                }
                let Some(next_fire_at) = timer.next_fire_at else {
                    break;
                };
                let now = Utc::now();
                next_fire_at
                    .signed_duration_since(now)
                    .to_std()
                    .unwrap_or(Duration::ZERO)
            };
            sleep(sleep_for).await;
            if let Err(err) =
                Self::timer_fire_once_shared(Arc::clone(&shared), &name, Some(generation), true)
                    .await
            {
                eprintln!("mstream timer '{name}' failed: {err}");
            }
        }
    }

    async fn timer_fire_once_shared(
        shared: Arc<Mutex<Self>>,
        name: &str,
        required_generation: Option<u64>,
        scheduled: bool,
    ) -> anyhow::Result<TimerFireOutcome> {
        let snapshot = {
            let state = shared.lock().await;
            let timer = state
                .timers
                .get(name)
                .with_context(|| format!("timer '{name}' not found"))?;
            if required_generation.is_some_and(|generation| timer.generation != generation) {
                bail!("timer '{name}' is no longer active");
            }
            TimerFireSnapshot {
                name: timer.name.clone(),
                target: timer.target.clone(),
                tmux_session_created: timer.tmux_session_created,
                handle: state.host_handle(timer.target.host_alias())?,
                prompt: timer.prompt.clone(),
                enter: timer.enter,
                submit_retries: timer.submit_retries,
                submit_retry_delay_ms: timer.submit_retry_delay_ms,
                input_quiet_for_secs: timer.input_quiet_for_secs,
                generation: timer.generation,
            }
        };

        let resolved =
            match Self::resolve_target(snapshot.handle.clone(), snapshot.target.clone()).await {
                Ok(resolved) => resolved,
                Err(err) => {
                    Self::record_timer_error_shared(
                        Arc::clone(&shared),
                        name,
                        snapshot.generation,
                        scheduled,
                        err.to_string(),
                    )
                    .await;
                    return Err(err);
                }
            };
        Self::ensure_resolved_target_fresh_shared(
            Arc::clone(&shared),
            &resolved,
            snapshot.tmux_session_created,
            Some((
                name,
                ActiveTimer {
                    generation: snapshot.generation,
                },
            )),
        )
        .await?;
        let now = Utc::now();
        if let Some(quiet_for_secs) = snapshot.input_quiet_for_secs {
            let decision = match Self::evaluate_input_guard(&resolved, quiet_for_secs, now).await {
                Ok(decision) => decision,
                Err(err) => {
                    Self::record_timer_error_shared(
                        Arc::clone(&shared),
                        name,
                        snapshot.generation,
                        scheduled,
                        err.to_string(),
                    )
                    .await;
                    return Err(err);
                }
            };
            if let Some(retry_after_secs) = decision.retry_after_secs {
                let next_fire_at = if scheduled {
                    add_secs(now, retry_after_secs)
                } else {
                    None
                };
                let defer_snapshot = TimerDeferSnapshot {
                    name: snapshot.name.clone(),
                    target: snapshot.target.clone(),
                    generation: snapshot.generation,
                    reason: "recent_client_input",
                    latest_client_activity: decision.latest_client_activity,
                    latest_client_activity_age_secs: decision.latest_client_activity_age_secs,
                    quiet_for_secs,
                    next_fire_at,
                };
                let mut state = shared.lock().await;
                if let Some(timer) = state.timers.get_mut(name) {
                    if timer.generation == snapshot.generation {
                        if scheduled {
                            timer.next_fire_at = next_fire_at;
                        }
                        timer.defer_count += 1;
                        timer.last_deferred_at = Some(now);
                        timer.last_defer_reason = Some(defer_snapshot.reason.to_string());
                        timer.last_input_activity = decision.latest_client_activity;
                        timer.last_error = None;
                    }
                }
                return Ok(TimerFireOutcome::Deferred(defer_snapshot));
            }
        }

        let result = async {
            Self::send_text_to_resolved(
                &resolved,
                &snapshot.prompt,
                PasteMode::Bracketed,
                snapshot.enter,
            )
            .await?;
            Self::send_submit_retries_to_resolved(
                &resolved,
                snapshot.submit_retries,
                snapshot.submit_retry_delay_ms,
            )
            .await
        }
        .await;

        let now = Utc::now();
        let mut state = shared.lock().await;
        if let Some(timer) = state.timers.get_mut(name) {
            if timer.generation == snapshot.generation {
                if scheduled {
                    timer.next_fire_at = add_secs(now, timer.every_secs);
                }
                match &result {
                    Ok(()) => {
                        timer.last_fired_at = Some(now);
                        timer.fire_count += 1;
                        timer.last_error = None;
                    }
                    Err(err) => {
                        timer.last_error = Some(err.to_string());
                    }
                }
            }
        }
        result?;
        Ok(TimerFireOutcome::Sent(snapshot))
    }

    async fn record_timer_error_shared(
        shared: Arc<Mutex<Self>>,
        name: &str,
        generation: u64,
        scheduled: bool,
        message: String,
    ) {
        let now = Utc::now();
        let mut state = shared.lock().await;
        if let Some(timer) = state.timers.get_mut(name) {
            if timer.generation == generation {
                if scheduled {
                    timer.next_fire_at = add_secs(now, timer.every_secs);
                }
                timer.last_error = Some(message);
            }
        }
    }

    async fn snapshot_shared(
        shared: Arc<Mutex<Self>>,
        request: SnapshotRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let text = Self::capture_workstream_shared(
            Arc::clone(&shared),
            &request.workstream,
            request.max_chars,
        )
        .await?;
        Ok(vec![json!({
            "type": "snapshot",
            "workstream": request.workstream,
            "after": request.after,
            "text": text,
            "ordering": "arrival",
        })])
    }

    async fn summary_input_shared(
        shared: Arc<Mutex<Self>>,
        request: SummaryInputRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let text = Self::capture_workstream_shared(
            Arc::clone(&shared),
            &request.workstream,
            request.max_chars,
        )
        .await?;
        Ok(vec![json!({
            "type": "summary_input",
            "workstream": request.workstream,
            "since": request.since,
            "text": compact_text(&text, request.max_chars),
            "ordering": "arrival",
        })])
    }

    async fn status_shared(
        shared: Arc<Mutex<Self>>,
        workstream: String,
        active_window_secs: u64,
        idle_after_secs: u64,
    ) -> anyhow::Result<Vec<Value>> {
        let options = StatusActivityOptions {
            active_window_secs,
            idle_after_secs,
        };
        let (targets, hosts) = {
            let state = shared.lock().await;
            let workstream_record = state.workstream(&workstream)?;
            let targets = workstream_record
                .sessions
                .iter()
                .map(|target| {
                    (
                        target.clone(),
                        state
                            .sessions
                            .get(target)
                            .and_then(|record| record.tmux_session_created),
                    )
                })
                .collect::<Vec<_>>();
            let mut hosts = BTreeMap::new();
            for (target, _) in &targets {
                if !hosts.contains_key(target.host_alias()) {
                    hosts.insert(
                        target.host_alias().to_string(),
                        state.host_handle(target.host_alias())?,
                    );
                }
            }
            (targets, hosts)
        };

        let mut live = BTreeMap::new();
        let mut stale_targets = Vec::new();
        for (alias, handle) in hosts {
            let host_targets = targets
                .iter()
                .filter(|(target, _)| target.host_alias() == alias)
                .cloned()
                .collect::<Vec<_>>();
            match handle.list_sessions().await {
                Ok(sessions) => {
                    let mut by_id = sessions
                        .into_iter()
                        .map(|session| (session.id.as_str().to_string(), session))
                        .collect::<BTreeMap<_, _>>();
                    for (target, expected_created) in host_targets {
                        let activity = target.session_id_selector().and_then(|id| {
                            by_id.remove(id.as_str()).map(|session| {
                                if let Some(reason) = expected_created.and_then(|created| {
                                    session_reuse_reason(&target, created, session.created)
                                }) {
                                    stale_targets.push((target.clone(), reason));
                                    LiveActivity::Missing
                                } else {
                                    LiveActivity::Present(session)
                                }
                            })
                        });
                        live.insert(target, activity.unwrap_or(LiveActivity::Missing));
                    }
                }
                Err(err) => {
                    let message = err.to_string();
                    for (target, _) in host_targets {
                        live.insert(target, LiveActivity::Error(message.clone()));
                    }
                }
            }
        }

        let now = Utc::now();
        let (status, abort_tasks) = {
            let mut state = shared.lock().await;
            let mut abort_tasks = Vec::new();
            for (target, reason) in stale_targets {
                abort_tasks.extend(state.quarantine_stale_session_target(&target, &reason, None));
            }
            let status = state.status(&workstream, &live, options, now)?;
            (status, abort_tasks)
        };
        for task in abort_tasks {
            task.abort();
        }
        Ok(vec![status])
    }

    async fn recruit_shared(
        shared: Arc<Mutex<Self>>,
        request: RecruitRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let plans = Self::recruit_plans_shared(Arc::clone(&shared), &request).await?;
        let mut records = Vec::new();
        let mut changes = Vec::new();
        for plan in plans {
            let meta = {
                let state = shared.lock().await;
                state.workstream_meta(&request.workstream)?
            };
            Self::write_assignment_to_target(
                &plan.target.target,
                &meta,
                AssignmentTags {
                    workstream: &request.workstream,
                    role: &request.role,
                    identity: plan.target.target.session_name(),
                    agent: plan.agent.as_deref(),
                    cwd: plan.cwd.as_deref(),
                    state: plan.state,
                },
            )
            .await?;
            if let Some(label) = &meta.mmux_label {
                Self::apply_mmux_workstream_label(&plan.target.target, label).await?;
            }
            Self::ensure_monitoring_target_shared(Arc::clone(&shared), &plan.target.spec).await?;
            if let Some(task) = &request.task {
                Self::send_text_to_resolved(
                    &plan.target,
                    &managed_prompt(
                        &request.workstream,
                        &plan.target.spec,
                        Some(&request.role),
                        plan.cwd.as_deref(),
                        Some(task),
                    ),
                    PasteMode::Bracketed,
                    true,
                )
                .await?;
            }
            {
                let mut state = shared.lock().await;
                state.add_session_to_workstream(
                    &request.workstream,
                    plan.target.spec.clone(),
                    request.role.clone(),
                    plan.agent.clone(),
                    plan.cwd.clone(),
                    plan.state,
                )?;
            }
            changes.push(
                Self::apply_resolved_session_state_shared(
                    Arc::clone(&shared),
                    &plan.target,
                    plan.state,
                    None,
                )
                .await?,
            );
            let cursor = {
                let mut state = shared.lock().await;
                let mut event = EventDraft::new("recruited")
                    .target(&plan.target.spec)
                    .state(plan.state);
                if let Some(goal) = request.goal.clone() {
                    event = event.text(goal);
                }
                let mut cursor = state.record_event(&request.workstream, event)?;
                if let Some(label) = &meta.mmux_label {
                    cursor = state.record_event(
                        &request.workstream,
                        EventDraft::new("mmux_label_applied")
                            .target(&plan.target.spec)
                            .summary(format!("mmux label applied: {label}")),
                    )?;
                }
                cursor
            };
            records.push(json!({
                "type": "ok",
                "op": "recruited",
                "workstream": request.workstream,
                "target": plan.target.spec.to_string(),
                "cursor": cursor,
            }));
        }
        records.extend(Self::fire_handoffs_for_changes_shared(shared, changes).await?);
        Ok(records)
    }

    fn disconnect(&mut self, alias: &str) -> anyhow::Result<Vec<Value>> {
        if self.hosts.remove(alias).is_none() {
            bail!("host alias '{alias}' is not connected");
        }
        self.fleet.unregister(alias)?;
        let removed_timers: Vec<String> = self
            .timers
            .iter()
            .filter(|(_, timer)| timer.target.host_alias() == alias)
            .map(|(name, _)| name.clone())
            .collect();
        for timer_name in &removed_timers {
            if let Some(timer) = self.timers.remove(timer_name) {
                if let Some(task) = timer.task {
                    task.abort();
                }
            }
        }
        let removed: Vec<SessionTarget> = self
            .sessions
            .keys()
            .filter(|target| target.host_alias() == alias)
            .cloned()
            .collect();
        for target in removed {
            self.sessions.remove(&target);
            for workstream in self.workstreams.values_mut() {
                workstream.sessions.remove(&target);
            }
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "disconnect",
            "host": alias,
            "stopped_timers": removed_timers,
        })])
    }

    fn open(&mut self, request: OpenRequest) -> anyhow::Result<Vec<Value>> {
        if request.settings.event_limit == 0 {
            bail!("--event-limit must be greater than zero");
        }
        let mmux_label = request
            .mmux_label
            .as_deref()
            .map(validate_mmux_label)
            .transpose()?;
        let cursor;
        if self.workstreams.contains_key(&request.workstream) {
            let mut needs_generation = false;
            {
                let workstream = self.workstream_mut(&request.workstream)?;
                if workstream.state == WorkstreamState::Closed {
                    needs_generation = true;
                }
            }
            let new_generation = if needs_generation {
                Some(self.next_generation())
            } else {
                None
            };
            let workstream = self.workstream_mut(&request.workstream)?;
            workstream.title = request.title;
            workstream.goal = request.goal;
            workstream.domain = request.domain;
            if let Some(label) = mmux_label {
                workstream.mmux_label = Some(label);
                workstream.mmux_label_conflicts.clear();
            }
            workstream.settings = request.settings;
            if let Some(generation) = new_generation {
                workstream.reopen(generation);
            } else {
                workstream.state = WorkstreamState::Open;
                workstream.prune_events();
            }
            cursor = workstream.cursor(&request.workstream)?;
        } else {
            let generation = self.next_generation();
            let workstream = WorkstreamRecord::new(
                request.title,
                request.goal,
                request.domain,
                mmux_label,
                request.settings,
                generation,
            );
            cursor = workstream.cursor(&request.workstream)?;
            self.workstreams
                .insert(request.workstream.clone(), workstream);
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "open",
            "workstream": request.workstream,
            "cursor": cursor,
        })])
    }

    fn handoff_cancel(&mut self, workstream: &str, handoff_id: &str) -> anyhow::Result<Vec<Value>> {
        let removed = self
            .workstream_mut(workstream)?
            .handoffs
            .remove(handoff_id)
            .is_some();
        if !removed {
            bail!("handoff '{handoff_id}' not found in workstream '{workstream}'");
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "handoff_cancel",
            "workstream": workstream,
            "handoff_id": handoff_id,
        })])
    }

    fn hosts_json(&self) -> Value {
        let hosts: Vec<Value> = self
            .hosts
            .iter()
            .map(|(alias, host)| {
                json!({
                    "alias": alias,
                    "uri": host.uri,
                    "labels": host.labels,
                    "capacity": host.capacity,
                    "work_root": host.work_root,
                })
            })
            .collect();
        json!({ "type": "hosts", "hosts": hosts })
    }

    fn workstream_list_json(&self) -> Value {
        let workstreams: Vec<Value> = self
            .workstreams
            .iter()
            .map(|(name, workstream)| {
                json!({
                    "name": name,
                    "title": workstream.title,
                    "state": workstream.state.as_str(),
                    "goal": workstream.goal,
                    "domain": workstream.domain,
                    "mmux_label": workstream.mmux_label,
                    "mmux_label_conflicts": workstream.mmux_label_conflicts,
                    "settings": workstream.settings,
                    "sessions": workstream.sessions.len(),
                    "generation": workstream.generation,
                })
            })
            .collect();
        json!({ "type": "workstreams", "workstreams": workstreams })
    }

    fn show(&self, workstream: &str) -> anyhow::Result<Value> {
        let record = self.workstream(workstream)?;
        Ok(json!({
            "type": "workstream",
            "name": workstream,
            "title": record.title,
            "state": record.state.as_str(),
            "goal": record.goal,
            "domain": record.domain,
            "mmux_label": record.mmux_label,
            "mmux_label_conflicts": record.mmux_label_conflicts,
            "settings": record.settings,
            "generation": record.generation,
            "cursor": record.cursor(workstream)?,
            "sessions": record.sessions.iter().map(ToString::to_string).collect::<Vec<_>>(),
        }))
    }

    fn session_list_json(&self) -> Value {
        let sessions: Vec<Value> = self
            .sessions
            .iter()
            .map(|(target, session)| session.to_json(target))
            .collect();
        json!({ "type": "sessions", "sessions": sessions })
    }

    fn handoff_list(&self, workstream: &str) -> anyhow::Result<Value> {
        let handoffs: Vec<&HandoffRecord> =
            self.workstream(workstream)?.handoffs.values().collect();
        Ok(json!({
            "type": "handoffs",
            "workstream": workstream,
            "handoffs": handoffs,
        }))
    }

    fn timer_list_json(&self, workstream: Option<&str>) -> Value {
        let timers: Vec<Value> = self
            .timers
            .values()
            .filter(|timer| match workstream {
                Some(workstream) => timer.workstream.as_deref() == Some(workstream),
                None => true,
            })
            .map(TimerRecord::to_json)
            .collect();
        json!({
            "type": "timers",
            "workstream": workstream,
            "timers": timers,
        })
    }

    fn timer_stop(&mut self, name: &str) -> anyhow::Result<Vec<Value>> {
        let Some(timer) = self.timers.remove(name) else {
            bail!("timer '{name}' not found");
        };
        if let Some(task) = timer.task {
            task.abort();
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "timer_stop",
            "name": name,
            "workstream": timer.workstream,
            "target": timer.target.to_string(),
            "generation": timer.generation,
            "fire_count": timer.fire_count,
            "defer_count": timer.defer_count,
        })])
    }

    fn stop_timers_for_workstream(&mut self, workstream: &str) -> Vec<String> {
        let timer_names = self
            .timers
            .iter()
            .filter(|(_, timer)| timer.workstream.as_deref() == Some(workstream))
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        for name in &timer_names {
            if let Some(timer) = self.timers.remove(name) {
                if let Some(task) = timer.task {
                    task.abort();
                }
            }
        }
        timer_names
    }

    fn status(
        &mut self,
        workstream: &str,
        live: &BTreeMap<SessionTarget, LiveActivity>,
        activity_options: StatusActivityOptions,
        now: DateTime<Utc>,
    ) -> anyhow::Result<Value> {
        let (
            workstream_state,
            generation,
            settings,
            mmux_label,
            mmux_label_conflicts,
            cursor,
            targets,
        ) = {
            let workstream_record = self.workstream(workstream)?;
            (
                workstream_record.state.as_str(),
                workstream_record.generation,
                workstream_record.settings,
                workstream_record.mmux_label.clone(),
                workstream_record.mmux_label_conflicts.clone(),
                workstream_record.cursor(workstream)?,
                workstream_record
                    .sessions
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        };
        let mut agents = Vec::new();
        for target in targets {
            if let Some(session) = self.sessions.get_mut(&target) {
                agents.push(session.status_json(&target, live.get(&target), activity_options, now));
            }
        }
        Ok(json!({
            "type": "status",
            "workstream": workstream,
            "state": workstream_state,
            "generation": generation,
            "settings": settings,
            "mmux_label": mmux_label,
            "mmux_label_conflicts": mmux_label_conflicts,
            "cursor": cursor,
            "activity": {
                "source": "tmux-list-sessions",
                "semantics": "max(session_activity,window_activity)",
                "active_window_secs": activity_options.active_window_secs,
                "idle_after_secs": activity_options.idle_after_secs,
            },
            "agents": agents,
        }))
    }

    fn events(&self, request: EventsRequest) -> anyhow::Result<Value> {
        let workstream = self.workstream(&request.workstream)?;
        let from = match request.after.as_deref() {
            Some(cursor) => match PublicCursor::decode(cursor) {
                Ok(cursor) => {
                    if cursor.workstream != request.workstream
                        || cursor.generation != workstream.generation
                    {
                        return Ok(jsonl::error_with_field(
                            "cursor_stale",
                            "cursor generation does not match current workstream timeline",
                            "workstream",
                            json!(request.workstream),
                        ));
                    }
                    cursor.next_sequence
                }
                Err(err) => {
                    return Ok(jsonl::error("invalid_cursor", err.to_string()));
                }
            },
            None => 1,
        };
        let limit = if request.limit == 0 {
            usize::MAX
        } else {
            request.limit
        };
        let events: Vec<&EventRecord> = workstream
            .events
            .iter()
            .filter(|event| event.sequence >= from)
            .take(limit)
            .collect();
        let cursor = if let Some(event) = events.last() {
            event.cursor.clone()
        } else {
            workstream.cursor(&request.workstream)?
        };
        if request.readable {
            let text = events
                .iter()
                .map(|event| render_readable_event(event))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(json!({
                "type": "events_readable",
                "workstream": request.workstream,
                "text": text,
                "cursor": cursor,
                "ordering": "arrival",
            }));
        }
        Ok(json!({
            "type": "events",
            "workstream": request.workstream,
            "events": events,
            "cursor": cursor,
            "ordering": "arrival",
        }))
    }

    async fn capture_workstream_shared(
        shared: Arc<Mutex<Self>>,
        workstream: &str,
        max_chars: usize,
    ) -> anyhow::Result<String> {
        let targets = {
            let state = shared.lock().await;
            let workstream = state.workstream(workstream)?;
            workstream
                .sessions
                .iter()
                .filter_map(|target| {
                    state
                        .host_handle(target.host_alias())
                        .ok()
                        .map(|handle| (target.clone(), handle))
                })
                .collect::<Vec<_>>()
        };
        let mut text = String::new();
        for (logical, handle) in targets {
            let capture = match Self::resolve_target(handle, logical.clone()).await {
                Ok(target_handle) => match Self::ensure_resolved_target_fresh_shared(
                    Arc::clone(&shared),
                    &target_handle,
                    None,
                    None,
                )
                .await
                {
                    Ok(()) => target_handle.target.capture().await.unwrap_or_default(),
                    Err(_) => String::new(),
                },
                Err(_) => String::new(),
            };
            text.push_str(&format!("=== {} ===\n", logical));
            text.push_str(&capture);
            if !capture.ends_with('\n') {
                text.push('\n');
            }
            if char_count(&text) >= max_chars {
                return Ok(truncate_chars(&text, max_chars));
            }
        }
        Ok(truncate_chars(&text, max_chars))
    }

    async fn fire_handoffs_for_changes_shared(
        shared: Arc<Mutex<Self>>,
        changes: Vec<StateChange>,
    ) -> anyhow::Result<Vec<Value>> {
        let mut records = Vec::new();
        let mut pending = changes;
        while let Some(change) = pending.pop() {
            let handoffs = {
                let mut state = shared.lock().await;
                state.claim_matching_handoffs(&change)
            };
            for (workstream, handoff) in handoffs {
                let (record, next_change) =
                    Self::fire_claimed_handoff_shared(Arc::clone(&shared), &workstream, handoff)
                        .await?;
                records.push(record);
                pending.push(next_change);
            }
        }
        Ok(records)
    }

    async fn fire_claimed_handoff_shared(
        shared: Arc<Mutex<Self>>,
        workstream: &str,
        handoff: HandoffRecord,
    ) -> anyhow::Result<(Value, StateChange)> {
        let handle = {
            let state = shared.lock().await;
            state.host_handle(handoff.to.host_alias())?
        };
        let resolved = Self::resolve_target(handle, handoff.to.clone()).await?;
        Self::ensure_resolved_target_fresh_shared(
            Arc::clone(&shared),
            &resolved,
            handoff.to_tmux_session_created,
            None,
        )
        .await?;
        let change = Self::apply_resolved_session_state_shared(
            Arc::clone(&shared),
            &resolved,
            AgentState::Busy,
            None,
        )
        .await?;
        Self::send_text_to_resolved(&resolved, &handoff.task, PasteMode::Bracketed, true).await?;
        let cursor = shared.lock().await.record_event(
            workstream,
            EventDraft::new("handoff_fired")
                .target(&handoff.to)
                .text(handoff.task)
                .state(AgentState::Busy)
                .handoff_id(handoff.id.clone()),
        )?;
        Ok((
            json!({
                "type": "event",
                "kind": "handoff_fired",
                "workstream": workstream,
                "handoff_id": handoff.id,
                "from": handoff.from.to_string(),
                "to": handoff.to.to_string(),
                "state": AgentState::Busy.as_str(),
                "cursor": cursor,
            }),
            change,
        ))
    }

    fn claim_matching_handoffs(&mut self, change: &StateChange) -> Vec<(String, HandoffRecord)> {
        let mut handoffs = Vec::new();
        let is_transition = change.previous_state != Some(change.state);
        let session_created = self
            .sessions
            .iter()
            .filter_map(|(target, record)| {
                record
                    .tmux_session_created
                    .map(|created| (target.clone(), created))
            })
            .collect::<BTreeMap<_, _>>();
        for (workstream_name, workstream) in &mut self.workstreams {
            for handoff in workstream.handoffs.values_mut() {
                let stale_endpoint = handoff
                    .from_tmux_session_created
                    .and_then(|created| {
                        (session_created.get(&handoff.from) != Some(&created)).then(|| {
                            format!(
                                "handoff source {} no longer matches stored session_created={created}",
                                handoff.from
                            )
                        })
                    })
                    .or_else(|| {
                        handoff.to_tmux_session_created.and_then(|created| {
                            (session_created.get(&handoff.to) != Some(&created)).then(|| {
                                format!(
                                    "handoff destination {} no longer matches stored session_created={created}",
                                    handoff.to
                                )
                            })
                        })
                    });
                if let Some(reason) = stale_endpoint {
                    handoff.canceled_reason = Some(reason);
                    continue;
                }
                if !handoff.fired
                    && handoff.canceled_reason.is_none()
                    && handoff.from == change.target
                    && handoff.on == change.state
                    && (!handoff.only_on_transition || is_transition)
                {
                    handoff.fired = true;
                    handoffs.push((workstream_name.clone(), handoff.clone()));
                }
            }
        }
        handoffs
    }

    fn claim_handoff(
        &mut self,
        workstream: &str,
        handoff_id: &str,
    ) -> anyhow::Result<HandoffRecord> {
        let workstream_record = self.workstream_mut(workstream)?;
        let Some(handoff) = workstream_record.handoffs.get_mut(handoff_id) else {
            bail!("handoff '{handoff_id}' not found in workstream '{workstream}'");
        };
        if let Some(reason) = &handoff.canceled_reason {
            bail!("handoff '{handoff_id}' is canceled: {reason}");
        }
        handoff.fired = true;
        Ok(handoff.clone())
    }

    async fn apply_session_state_shared(
        shared: Arc<Mutex<Self>>,
        target: &SessionTarget,
        state: AgentState,
        summary: Option<&str>,
    ) -> anyhow::Result<StateChange> {
        let handle = {
            let state_guard = shared.lock().await;
            state_guard.host_handle(target.host_alias())?
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::apply_resolved_session_state_shared(shared, &resolved, state, summary).await
    }

    async fn apply_resolved_session_state_shared(
        shared: Arc<Mutex<Self>>,
        resolved: &ResolvedTarget,
        state: AgentState,
        summary: Option<&str>,
    ) -> anyhow::Result<StateChange> {
        Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), resolved, None, None)
            .await?;
        let now = tags::now_tag();
        let mut pairs = vec![
            ("state", tags::state_value(state)),
            ("updated-at", now.clone()),
        ];
        if let Some(summary) = summary {
            pairs.push(("last-report-summary", summary.to_string()));
            pairs.push(("last-report-kind", state.as_str().to_string()));
        }
        resolved
            .target
            .tags(tags::PREFIX)
            .await?
            .set_many(pairs)
            .await?;
        let mut state_guard = shared.lock().await;
        let previous_state = state_guard
            .sessions
            .get(&resolved.spec)
            .map(|record| record.state);
        let record = state_guard
            .sessions
            .entry(resolved.spec.clone())
            .or_insert_with(|| SessionRecord::from_target(&resolved.spec, state, Some(Utc::now())));
        if let Some(session) = resolved.target.session_info() {
            record.observe_tmux_session(session);
        }
        record.state = state;
        record.updated_at = Utc::now();
        if let Some(summary) = summary {
            record.last_report_kind = Some(state.as_str().to_string());
            record.last_report_summary = Some(summary.to_string());
        }
        Ok(StateChange {
            target: resolved.spec.clone(),
            state,
            previous_state,
        })
    }

    async fn send_interrupt_to_resolved(
        target: &ResolvedTarget,
        key: InterruptKey,
    ) -> anyhow::Result<()> {
        let sequence = match key {
            InterruptKey::Esc => KeySequence::parse("{Escape}")?,
            InterruptKey::CtrlC => KeySequence::parse("{C-c}")?,
        };
        target.target.send_keys(&sequence).await?;
        Ok(())
    }

    async fn send_text_to_resolved(
        target: &ResolvedTarget,
        text: &str,
        paste_mode: PasteMode,
        enter: bool,
    ) -> anyhow::Result<()> {
        let payload = match paste_mode {
            PasteMode::Bracketed if text.contains('\n') => {
                format!("\x1b[200~{}\x1b[201~", text)
            }
            _ => text.to_string(),
        };
        let sequence = if enter {
            KeySequence::literal(&payload).then_enter()
        } else {
            KeySequence::literal(&payload)
        };
        target.target.send_keys(&sequence).await?;
        Ok(())
    }

    async fn send_submit_retries_to_resolved(
        target: &ResolvedTarget,
        submit_retries: u8,
        submit_retry_delay_ms: u64,
    ) -> anyhow::Result<()> {
        if submit_retries == 0 {
            return Ok(());
        }
        let enter = KeySequence::parse("{Enter}")?;
        for _ in 0..submit_retries {
            sleep(Duration::from_millis(submit_retry_delay_ms)).await;
            target.target.send_keys(&enter).await?;
        }
        Ok(())
    }

    async fn evaluate_input_guard(
        target: &ResolvedTarget,
        quiet_for_secs: u64,
        now: DateTime<Utc>,
    ) -> anyhow::Result<InputGuardDecision> {
        let activity = target
            .host
            .session_client_activity(target.target.session_name())
            .await?;
        let Some(latest_client_activity) = activity.latest_client_activity else {
            return Ok(InputGuardDecision {
                latest_client_activity: None,
                latest_client_activity_age_secs: None,
                retry_after_secs: None,
            });
        };

        let latest_client_activity_age_secs =
            seconds_since_epoch(now, latest_client_activity).unwrap_or(0);
        let retry_after_secs = if latest_client_activity_age_secs >= quiet_for_secs {
            None
        } else {
            Some((quiet_for_secs - latest_client_activity_age_secs).max(1))
        };

        Ok(InputGuardDecision {
            latest_client_activity: Some(latest_client_activity),
            latest_client_activity_age_secs: Some(latest_client_activity_age_secs),
            retry_after_secs,
        })
    }

    async fn resolve_target(
        handle: HostHandle,
        logical: SessionTarget,
    ) -> anyhow::Result<ResolvedTarget> {
        let target = handle
            .target(logical.target_spec())
            .await?
            .with_context(|| format!("session '{}' not found", logical))?;
        let stable = match target.session_id() {
            Some(session_id) => SessionTarget::session_id(logical.host_alias(), session_id)?,
            None => logical,
        };
        Ok(ResolvedTarget {
            spec: stable,
            host: handle,
            target,
        })
    }

    async fn ensure_resolved_target_fresh_shared(
        shared: Arc<Mutex<Self>>,
        resolved: &ResolvedTarget,
        expected_created: Option<u64>,
        active_timer: Option<(&str, ActiveTimer)>,
    ) -> anyhow::Result<()> {
        let Some(session) = resolved.target.session_info() else {
            return Ok(());
        };
        let (reason, tasks) = {
            let mut state = shared.lock().await;
            if let Some(reason) =
                state.stale_session_reuse_reason(&resolved.spec, session, expected_created)
            {
                let tasks =
                    state.quarantine_stale_session_target(&resolved.spec, &reason, active_timer);
                (Some(reason), tasks)
            } else {
                if let Some(record) = state.sessions.get_mut(&resolved.spec) {
                    record.observe_tmux_session(session);
                }
                (None, Vec::new())
            }
        };
        for task in tasks {
            task.abort();
        }
        if let Some(reason) = reason {
            bail!(reason);
        }
        Ok(())
    }

    async fn prepare_resolved_target_for_adoption_shared(
        shared: Arc<Mutex<Self>>,
        resolved: &ResolvedTarget,
    ) -> anyhow::Result<()> {
        let Some(session) = resolved.target.session_info() else {
            return Ok(());
        };
        let tasks = {
            let mut state = shared.lock().await;
            if let Some(reason) = state.stale_session_reuse_reason(&resolved.spec, session, None) {
                state.quarantine_stale_session_target(&resolved.spec, &reason, None)
            } else {
                if let Some(record) = state.sessions.get_mut(&resolved.spec) {
                    record.observe_tmux_session(session);
                }
                Vec::new()
            }
        };
        for task in tasks {
            task.abort();
        }
        Ok(())
    }

    async fn ensure_monitoring_target_shared(
        shared: Arc<Mutex<Self>>,
        target: &SessionTarget,
    ) -> anyhow::Result<()> {
        let mut state = shared.lock().await;
        state.fleet.ensure_monitoring_session(target).await?;
        Ok(())
    }

    async fn touch_resolved_session_shared(
        shared: Arc<Mutex<Self>>,
        target: &ResolvedTarget,
    ) -> anyhow::Result<()> {
        let now = tags::now_tag();
        target
            .target
            .tags(tags::PREFIX)
            .await?
            .set_many([("updated-at", now)])
            .await?;
        if let Some(record) = shared.lock().await.sessions.get_mut(&target.spec) {
            record.updated_at = Utc::now();
        }
        Ok(())
    }

    async fn observe_resolved_session_shared(
        shared: Arc<Mutex<Self>>,
        target: &SessionTarget,
        observed: &Target,
    ) {
        let Some(session) = observed.session_info() else {
            return;
        };
        let mut state = shared.lock().await;
        if let Some(record) = state.sessions.get_mut(target) {
            record.observe_tmux_session(session);
            record.updated_at = Utc::now();
        }
    }

    async fn parsed_mstream_tags(target: &Target) -> anyhow::Result<ParsedTags> {
        let tags = target.tags(tags::PREFIX).await?.list().await?;
        Ok(ParsedTags::from_tags(&tags))
    }

    async fn write_assignment_to_target(
        target: &Target,
        meta: &WorkstreamMeta,
        assignment: AssignmentTags<'_>,
    ) -> anyhow::Result<()> {
        let mut pairs = vec![
            ("version", "1".to_string()),
            ("managed", "true".to_string()),
            ("workstream", assignment.workstream.to_string()),
            ("workstream-title", meta.title.clone()),
            ("workstream-state", "open".to_string()),
            ("role", assignment.role.to_string()),
            ("identity", assignment.identity.to_string()),
            ("state", tags::state_value(assignment.state)),
            ("updated-at", tags::now_tag()),
        ];
        if let Some(goal) = &meta.goal {
            pairs.push(("workstream-goal", goal.clone()));
        }
        if let Some(domain) = &meta.domain {
            pairs.push(("workstream-domain", domain.clone()));
        }
        if let Some(agent) = assignment.agent {
            pairs.push(("agent", agent.to_string()));
        }
        if let Some(cwd) = assignment.cwd {
            pairs.push(("cwd", cwd.display().to_string()));
        }
        let tags = target.tags(tags::PREFIX).await?;
        tags.set_many(pairs).await?;
        let mut unset = Vec::new();
        if meta.goal.is_none() {
            unset.push("workstream-goal");
        }
        if meta.domain.is_none() {
            unset.push("workstream-domain");
        }
        if unset.is_empty() {
            return Ok(());
        }
        tags.unset_many(unset).await?;
        Ok(())
    }

    async fn write_session_identity_to_target(
        target: &Target,
        identity: &str,
        role: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut pairs = vec![
            ("identity", identity.to_string()),
            ("updated-at", tags::now_tag()),
        ];
        if let Some(role) = role {
            pairs.push(("role", role.to_string()));
        }
        target.tags(tags::PREFIX).await?.set_many(pairs).await?;
        Ok(())
    }

    async fn apply_mmux_workstream_label(target: &Target, label: &str) -> anyhow::Result<()> {
        let mstream_tags = target.tags(tags::PREFIX).await?;
        let mmux_tags = target.tags(MMUX_TAG_PREFIX).await?;
        let mut failures = Vec::new();

        match mmux_tags.read(MMUX_SELECTED_KEY).await {
            Ok(Some(selected_key)) if selected_key == MMUX_WORKSTREAM_KEY => {}
            Ok(Some(previous)) => {
                if let Err(err) = mstream_tags
                    .set(MSTREAM_MMUX_PREVIOUS_SELECTED_KEY, &previous)
                    .await
                {
                    failures.push(format!(
                        "set @mstream/{MSTREAM_MMUX_PREVIOUS_SELECTED_KEY}: {err}"
                    ));
                }
            }
            Ok(None) => {
                if let Err(err) = mstream_tags.unset(MSTREAM_MMUX_PREVIOUS_SELECTED_KEY).await {
                    failures.push(format!(
                        "unset @mstream/{MSTREAM_MMUX_PREVIOUS_SELECTED_KEY}: {err}"
                    ));
                }
            }
            Err(err) => failures.push(format!("read @mmux/{MMUX_SELECTED_KEY}: {err}")),
        }

        if let Err(err) = mstream_tags
            .set_many([
                (MSTREAM_MMUX_LABEL_KEY, label),
                (MSTREAM_MMUX_SELECTED_KEY, MMUX_WORKSTREAM_KEY),
            ])
            .await
        {
            failures.push(format!(
                "set @mstream/{MSTREAM_MMUX_LABEL_KEY},@mstream/{MSTREAM_MMUX_SELECTED_KEY}: {err}"
            ));
        }
        if let Err(err) = mmux_tags
            .set_many([
                (MMUX_WORKSTREAM_KEY, label),
                (MMUX_SELECTED_KEY, MMUX_WORKSTREAM_KEY),
            ])
            .await
        {
            failures.push(format!(
                "set @mmux/{MMUX_WORKSTREAM_KEY},@mmux/{MMUX_SELECTED_KEY}: {err}"
            ));
        }

        mmux_tag_result(failures)
    }

    async fn clear_mmux_workstream_label(
        target: &Target,
        workstream: &str,
    ) -> anyhow::Result<MmuxLabelCleanup> {
        let mstream_tags = target.tags(tags::PREFIX).await?;
        let mut read_failures = Vec::new();
        let owned_label = match mstream_tags.read(MSTREAM_MMUX_LABEL_KEY).await {
            Ok(value) => value,
            Err(err) => {
                read_failures.push(format!("read @mstream/{MSTREAM_MMUX_LABEL_KEY}: {err}"));
                None
            }
        };
        let owned_selected_key = match mstream_tags.read(MSTREAM_MMUX_SELECTED_KEY).await {
            Ok(value) => value,
            Err(err) => {
                read_failures.push(format!("read @mstream/{MSTREAM_MMUX_SELECTED_KEY}: {err}"));
                None
            }
        };
        let active_workstream = match mstream_tags.read("workstream").await {
            Ok(value) => value,
            Err(err) => {
                read_failures.push(format!("read @mstream/workstream: {err}"));
                None
            }
        };
        mmux_tag_result(read_failures)?;

        if owned_label.is_none() && owned_selected_key.is_none() {
            return Ok(MmuxLabelCleanup::default());
        }
        if active_workstream.as_deref() != Some(workstream) {
            return Ok(MmuxLabelCleanup {
                skipped_workstream_mismatch: true,
                ..Default::default()
            });
        }

        let mmux_tags = target.tags(MMUX_TAG_PREFIX).await?;
        let mut failures = Vec::new();
        let previous_selected_key =
            match mstream_tags.read(MSTREAM_MMUX_PREVIOUS_SELECTED_KEY).await {
                Ok(value) => value,
                Err(err) => {
                    failures.push(format!(
                        "read @mstream/{MSTREAM_MMUX_PREVIOUS_SELECTED_KEY}: {err}"
                    ));
                    None
                }
            };
        let selected_key = match mmux_tags.read(MMUX_SELECTED_KEY).await {
            Ok(value) => value,
            Err(err) => {
                failures.push(format!("read @mmux/{MMUX_SELECTED_KEY}: {err}"));
                None
            }
        };
        let mut cleanup = MmuxLabelCleanup {
            cleared: true,
            ..Default::default()
        };

        if let Err(err) = mmux_tags.unset(MMUX_WORKSTREAM_KEY).await {
            failures.push(format!("unset @mmux/{MMUX_WORKSTREAM_KEY}: {err}"));
        }
        if selected_key.as_deref() == Some(MMUX_WORKSTREAM_KEY) {
            if let Some(previous_selected_key) = previous_selected_key {
                if let Err(err) = mmux_tags
                    .set(MMUX_SELECTED_KEY, &previous_selected_key)
                    .await
                {
                    failures.push(format!(
                        "restore @mmux/{MMUX_SELECTED_KEY}={previous_selected_key}: {err}"
                    ));
                } else {
                    cleanup.restored_previous = true;
                }
            } else if let Err(err) = mmux_tags.unset(MMUX_SELECTED_KEY).await {
                failures.push(format!("unset @mmux/{MMUX_SELECTED_KEY}: {err}"));
            }
        } else {
            cleanup.selected_key_unchanged = true;
        }

        if let Err(err) = mstream_tags
            .unset_many([
                MSTREAM_MMUX_LABEL_KEY,
                MSTREAM_MMUX_SELECTED_KEY,
                MSTREAM_MMUX_PREVIOUS_SELECTED_KEY,
            ])
            .await
        {
            failures.push(format!(
                "unset @mstream/{MSTREAM_MMUX_LABEL_KEY},@mstream/{MSTREAM_MMUX_SELECTED_KEY},@mstream/{MSTREAM_MMUX_PREVIOUS_SELECTED_KEY}: {err}"
            ));
        }
        mmux_tag_result(failures).map(|()| cleanup)
    }

    fn add_session_to_workstream(
        &mut self,
        workstream: &str,
        target: SessionTarget,
        role: String,
        agent: Option<String>,
        cwd: Option<PathBuf>,
        state: AgentState,
    ) -> anyhow::Result<()> {
        let mmux_label = self.workstream(workstream)?.mmux_label.clone();
        self.workstream_mut(workstream)?
            .sessions
            .insert(target.clone());
        let record = self
            .sessions
            .entry(target.clone())
            .or_insert_with(|| SessionRecord::from_target(&target, state, Some(Utc::now())));
        record.role = Some(role);
        if let Some(agent) = agent {
            record.agent = Some(agent);
        }
        if let Some(cwd) = cwd {
            record.cwd = Some(cwd);
        }
        record.state = state;
        record.workstream = Some(workstream.to_string());
        record.mmux_label = mmux_label;
        record.updated_at = Utc::now();
        Ok(())
    }

    fn session_retag_plan(
        &self,
        target: &SessionTarget,
        resolved_target: &Target,
        requested_new_name: Option<&str>,
        requested_role: Option<String>,
        requested_workstream: Option<String>,
        requested_mmux_label: Option<String>,
    ) -> anyhow::Result<SessionRetagPlan> {
        let session = resolved_target
            .session_info()
            .with_context(|| format!("target '{target}' is not a session"))?;
        let metadata_requested = requested_role.is_some()
            || requested_workstream.is_some()
            || requested_mmux_label.is_some();
        let record = self.sessions.get(target);
        let old_workstream = record.and_then(|record| record.workstream.clone());
        let new_workstream = requested_workstream
            .clone()
            .or_else(|| old_workstream.clone());

        if let Some(workstream) = &requested_workstream {
            self.ensure_workstream_open(workstream)?;
        }
        if requested_mmux_label.is_some() && new_workstream.is_none() {
            bail!("--mmux-label requires --workstream or an existing workstream assignment");
        }
        if requested_mmux_label.is_some() {
            if let Some(workstream) = &new_workstream {
                self.ensure_workstream_open(workstream)?;
            }
        }

        let workstream_meta = new_workstream
            .as_deref()
            .map(|workstream| self.workstream_meta(workstream))
            .transpose()?;
        let role = requested_role.or_else(|| record.and_then(|record| record.role.clone()));
        if new_workstream.is_some() && role.is_none() {
            bail!("--role is required when assigning a session to a workstream");
        }
        let mmux_label = requested_mmux_label.clone().or_else(|| {
            workstream_meta
                .as_ref()
                .and_then(|meta| meta.mmux_label.clone())
        });
        let clear_mmux_workstream = if old_workstream.is_some() && old_workstream != new_workstream
        {
            old_workstream.clone()
        } else {
            None
        };

        Ok(SessionRetagPlan {
            old_name: session.name.clone(),
            new_name: requested_new_name
                .map(ToString::to_string)
                .unwrap_or_else(|| session.name.clone()),
            old_workstream,
            new_workstream,
            role,
            agent: record.and_then(|record| record.agent.clone()),
            cwd: record.and_then(|record| record.cwd.clone()),
            state: record
                .map(|record| record.state)
                .unwrap_or(AgentState::Idle),
            workstream_meta,
            mmux_label,
            clear_mmux_workstream,
            requested_mmux_label,
            metadata_requested,
        })
    }

    fn apply_session_retag_plan(
        &mut self,
        target: &SessionTarget,
        observed: &Target,
        plan: &SessionRetagPlan,
        renamed: bool,
    ) -> anyhow::Result<(Option<String>, usize)> {
        if let (Some(workstream), Some(label)) = (
            plan.new_workstream.as_deref(),
            plan.requested_mmux_label.as_ref(),
        ) {
            let workstream = self.workstream_mut(workstream)?;
            workstream.mmux_label = Some(label.clone());
            workstream.mmux_label_conflicts.clear();
        }

        let mut stale_handoffs_canceled = 0usize;
        if plan.old_workstream != plan.new_workstream {
            if let Some(old_workstream) = &plan.old_workstream {
                if let Some(workstream) = self.workstreams.get_mut(old_workstream) {
                    workstream.sessions.remove(target);
                }
                stale_handoffs_canceled += self.cancel_handoffs_for_target_in_workstream(
                    old_workstream,
                    target,
                    "session retagged to a different workstream",
                );
            }
            if let Some(new_workstream) = &plan.new_workstream {
                self.workstream_mut(new_workstream)?
                    .sessions
                    .insert(target.clone());
            }
        } else if let Some(new_workstream) = &plan.new_workstream {
            self.workstream_mut(new_workstream)?
                .sessions
                .insert(target.clone());
        }

        let record = self
            .sessions
            .entry(target.clone())
            .or_insert_with(|| SessionRecord::from_target(target, plan.state, Some(Utc::now())));
        if let Some(session) = observed.session_info() {
            record.observe_tmux_session(session);
        }
        record.role = plan.role.clone();
        record.agent = plan.agent.clone();
        record.cwd = plan.cwd.clone();
        record.state = plan.state;
        record.workstream = plan.new_workstream.clone();
        record.mmux_label = plan.mmux_label.clone();
        record.updated_at = Utc::now();

        let mut cursor = None;
        if let Some(workstream) = &plan.new_workstream {
            if renamed {
                cursor = Some(self.record_event(
                    workstream,
                    EventDraft::new("renamed").target(target).summary(format!(
                        "renamed session: {} -> {}",
                        plan.old_name, plan.new_name
                    )),
                )?);
            }
            if plan.old_workstream != plan.new_workstream
                || plan.requested_mmux_label.is_some()
                || plan.metadata_requested
            {
                cursor = Some(
                    self.record_event(
                        workstream,
                        EventDraft::new("retagged")
                            .target(target)
                            .state(plan.state)
                            .summary("session retagged"),
                    )?,
                );
            }
        }

        Ok((cursor, stale_handoffs_canceled))
    }

    fn cancel_handoffs_for_target_in_workstream(
        &mut self,
        workstream: &str,
        target: &SessionTarget,
        reason: &str,
    ) -> usize {
        let Some(workstream) = self.workstreams.get_mut(workstream) else {
            return 0;
        };
        let mut canceled = 0usize;
        for handoff in workstream.handoffs.values_mut() {
            if handoff.canceled_reason.is_none()
                && (handoff.from == *target || handoff.to == *target)
            {
                handoff.canceled_reason = Some(reason.to_string());
                canceled += 1;
            }
        }
        canceled
    }

    fn record_event(&mut self, workstream: &str, draft: EventDraft) -> anyhow::Result<String> {
        let workstream_record = self.workstream_mut(workstream)?;
        let sequence = workstream_record.next_sequence;
        workstream_record.next_sequence += 1;
        let cursor = PublicCursor::new(
            workstream,
            workstream_record.generation,
            workstream_record.next_sequence,
        )
        .encode()?;
        let event = EventRecord {
            cursor: cursor.clone(),
            sequence,
            generation: workstream_record.generation,
            workstream: workstream.to_string(),
            kind: draft.kind,
            target: draft.target,
            text: draft.text,
            state: draft.state.map(AgentState::as_str),
            summary: draft.summary,
            handoff_id: draft.handoff_id,
            timestamp: tags::now_tag(),
        };
        workstream_record.events.push_back(event);
        workstream_record.prune_events();
        Ok(cursor)
    }

    fn host_handle(&self, alias: &str) -> anyhow::Result<HostHandle> {
        self.fleet
            .host(alias)
            .cloned()
            .with_context(|| format!("host alias '{alias}' is not connected"))
    }

    fn stale_session_reuse_reason(
        &self,
        target: &SessionTarget,
        observed: &SessionInfo,
        expected_created: Option<u64>,
    ) -> Option<String> {
        let expected_created = expected_created.or_else(|| {
            self.sessions
                .get(target)
                .and_then(|record| record.tmux_session_created)
        })?;
        session_reuse_reason(target, expected_created, observed.created)
    }

    fn quarantine_stale_session_target(
        &mut self,
        target: &SessionTarget,
        reason: &str,
        active_timer: Option<(&str, ActiveTimer)>,
    ) -> Vec<JoinHandle<()>> {
        self.deregister_session_target(target, reason, active_timer)
    }

    fn deregister_session_target(
        &mut self,
        target: &SessionTarget,
        reason: &str,
        active_timer: Option<(&str, ActiveTimer)>,
    ) -> Vec<JoinHandle<()>> {
        self.sessions.remove(target);
        for workstream in self.workstreams.values_mut() {
            workstream.sessions.remove(target);
            for handoff in workstream.handoffs.values_mut() {
                if handoff.canceled_reason.is_none()
                    && (handoff.from == *target || handoff.to == *target)
                {
                    handoff.canceled_reason = Some(reason.to_string());
                }
            }
        }

        let mut tasks = Vec::new();
        for timer in self.timers.values_mut() {
            if timer.target != *target {
                continue;
            }
            timer.next_fire_at = None;
            timer.last_error = Some(reason.to_string());
            let is_active_timer = active_timer.is_some_and(|(name, active)| {
                timer.name == name && timer.generation == active.generation
            });
            if !is_active_timer {
                if let Some(task) = timer.task.take() {
                    tasks.push(task);
                }
            }
        }
        tasks
    }

    fn workstream(&self, name: &str) -> anyhow::Result<&WorkstreamRecord> {
        self.workstreams
            .get(name)
            .with_context(|| format!("workstream '{name}' is not open"))
    }

    fn workstream_mut(&mut self, name: &str) -> anyhow::Result<&mut WorkstreamRecord> {
        self.workstreams
            .get_mut(name)
            .with_context(|| format!("workstream '{name}' is not open"))
    }

    fn workstream_session_targets(&self, name: &str) -> anyhow::Result<Vec<SessionTarget>> {
        let mut targets = self.workstream(name)?.sessions.clone();
        targets.extend(
            self.sessions
                .iter()
                .filter(|(_, record)| record.workstream.as_deref() == Some(name))
                .map(|(target, _)| target.clone()),
        );
        Ok(targets.into_iter().collect())
    }

    fn workstream_meta(&self, name: &str) -> anyhow::Result<WorkstreamMeta> {
        let workstream = self.workstream(name)?;
        Ok(WorkstreamMeta {
            title: workstream.title.clone(),
            goal: workstream.goal.clone(),
            domain: workstream.domain.clone(),
            mmux_label: workstream.mmux_label.clone(),
        })
    }

    fn ensure_workstream_open(&self, name: &str) -> anyhow::Result<()> {
        let workstream = self.workstream(name)?;
        if workstream.state != WorkstreamState::Open {
            bail!("workstream '{name}' is not open");
        }
        Ok(())
    }

    fn ensure_target_in_workstream(
        &self,
        workstream: &str,
        target: &SessionTarget,
    ) -> anyhow::Result<()> {
        if !self.workstream(workstream)?.sessions.contains(target) {
            bail!(
                "target '{}' is not joined to workstream '{}'",
                target,
                workstream
            );
        }
        Ok(())
    }

    fn host_matches_selectors(&self, alias: &str, selectors: &BTreeMap<String, String>) -> bool {
        let Some(host) = self.hosts.get(alias) else {
            return false;
        };
        selectors
            .iter()
            .all(|(key, value)| host.labels.get(key) == Some(value))
    }

    async fn broadcast_targets_shared(
        shared: Arc<Mutex<Self>>,
        request: &BroadcastRequest,
    ) -> anyhow::Result<Vec<BroadcastTarget>> {
        let snapshots = {
            let state = shared.lock().await;
            state.broadcast_target_snapshots(request)?
        };
        let mut selected = Vec::new();
        for snapshot in snapshots {
            let resolved = Self::resolve_target(snapshot.handle, snapshot.target).await?;
            Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &resolved, None, None)
                .await?;
            selected.push(BroadcastTarget {
                target: resolved,
                state: snapshot.state,
            });
        }
        Ok(selected)
    }

    fn broadcast_target_snapshots(
        &self,
        request: &BroadcastRequest,
    ) -> anyhow::Result<Vec<BroadcastTargetSnapshot>> {
        let targets = self.workstream(&request.workstream)?.sessions.clone();
        let mut selected = Vec::new();
        for target in targets {
            let Some(session) = self.sessions.get(&target) else {
                continue;
            };
            if request
                .role
                .as_ref()
                .is_some_and(|role| session.role.as_ref() != Some(role))
            {
                continue;
            }
            if request.state.is_some_and(|state| session.state != state) {
                continue;
            }
            selected.push(BroadcastTargetSnapshot {
                handle: self.host_handle(target.host_alias())?,
                target,
                state: session.state,
            });
        }
        Ok(selected)
    }

    async fn recruit_plans_shared(
        shared: Arc<Mutex<Self>>,
        request: &RecruitRequest,
    ) -> anyhow::Result<Vec<RecruitPlan>> {
        let snapshots = {
            let state = shared.lock().await;
            state.recruit_plan_snapshots(request)?
        };
        let mut plans = Vec::new();
        for snapshot in snapshots {
            let target = Self::resolve_target(snapshot.handle, snapshot.target).await?;
            Self::ensure_resolved_target_fresh_shared(Arc::clone(&shared), &target, None, None)
                .await?;
            plans.push(RecruitPlan {
                target,
                agent: snapshot.agent,
                cwd: snapshot.cwd,
                state: snapshot.state,
            });
        }
        Ok(plans)
    }

    fn recruit_plan_snapshots(
        &self,
        request: &RecruitRequest,
    ) -> anyhow::Result<Vec<RecruitPlanSnapshot>> {
        self.ensure_workstream_open(&request.workstream)?;
        let mut candidates: Vec<SessionTarget> = self
            .sessions
            .iter()
            .filter_map(|(target, record)| {
                if record.state != AgentState::Available {
                    return None;
                }
                if request
                    .agent
                    .as_ref()
                    .is_some_and(|agent| record.agent.as_ref() != Some(agent))
                {
                    return None;
                }
                if !self.host_matches_selectors(target.host_alias(), &request.selectors) {
                    return None;
                }
                Some(target.clone())
            })
            .collect();
        candidates.sort();
        if candidates.len() < request.count {
            bail!(
                "only {} available session(s) match recruit request; need {}",
                candidates.len(),
                request.count
            );
        }
        let state = if request.task.is_some() {
            AgentState::Busy
        } else {
            AgentState::Reserved
        };
        candidates
            .into_iter()
            .take(request.count)
            .map(|target| {
                let record = self.sessions.get(&target);
                Ok(RecruitPlanSnapshot {
                    handle: self.host_handle(target.host_alias())?,
                    agent: request
                        .agent
                        .clone()
                        .or_else(|| record.and_then(|record| record.agent.clone())),
                    cwd: record.and_then(|record| record.cwd.clone()),
                    target,
                    state,
                })
            })
            .collect()
    }

    fn next_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation += 1;
        generation
    }
}

impl SessionRecord {
    fn from_target(
        target: &SessionTarget,
        state: AgentState,
        updated_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            role: None,
            agent: None,
            identity: session_identity_seed(target),
            state,
            cwd: None,
            workstream: None,
            last_report_kind: None,
            last_report_summary: None,
            updated_at: updated_at.unwrap_or_else(Utc::now),
            context_domains: BTreeSet::new(),
            context_specialties: BTreeSet::new(),
            context_summary: None,
            last_workstream: None,
            last_workstream_title: None,
            mmux_label: None,
            mmux_previous_selected_key: None,
            tmux_session_created: None,
            last_tmux_activity: None,
            activity_observed_at: None,
        }
    }

    fn observe_tmux_session(&mut self, session: &SessionInfo) {
        self.identity = session.name.clone();
        self.tmux_session_created = Some(session.created);
    }

    fn observe_activity(&mut self, tmux_activity: u64, now: DateTime<Utc>) -> u64 {
        if self.last_tmux_activity != Some(tmux_activity) {
            self.last_tmux_activity = Some(tmux_activity);
            self.activity_observed_at = Some(now);
            return 0;
        }
        self.activity_observed_at
            .and_then(|observed_at| seconds_between(observed_at, now))
            .unwrap_or(0)
    }

    fn status_json(
        &mut self,
        target: &SessionTarget,
        live: Option<&LiveActivity>,
        activity_options: StatusActivityOptions,
        now: DateTime<Utc>,
    ) -> Value {
        let activity = self.activity_json(live, activity_options, now);
        json!({
            "target": target.to_string(),
            "role": self.role,
            "agent": self.agent,
            "state": self.state.as_str(),
            "last_report_kind": self.last_report_kind,
            "last_report_summary": self.last_report_summary,
            "mmux_label": self.mmux_label,
            "mmux_previous_selected_key": self.mmux_previous_selected_key,
            "updated_at": self.updated_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "tmux_present": activity.tmux_present,
            "tmux_session_id": activity.tmux_session_id,
            "tmux_activity": activity.tmux_activity,
            "tmux_activity_at": activity.tmux_activity_at,
            "last_output_secs": activity.last_output_secs,
            "observed_activity_idle_secs": activity.observed_activity_idle_secs,
            "observed_activity_at": activity.observed_activity_at,
            "activity_hint": activity.activity_hint,
            "activity_error": activity.activity_error,
            "stuck_hint": Value::Null,
        })
    }

    fn activity_json(
        &mut self,
        live: Option<&LiveActivity>,
        activity_options: StatusActivityOptions,
        now: DateTime<Utc>,
    ) -> SessionActivityJson {
        match live {
            Some(LiveActivity::Present(session)) => {
                let observed_idle_secs = Some(self.observe_activity(session.activity, now));
                let last_output_secs = seconds_since_epoch(now, session.activity);
                SessionActivityJson {
                    tmux_present: json!(true),
                    tmux_session_id: json!(session.id.as_str()),
                    tmux_activity: json!(session.activity),
                    tmux_activity_at: epoch_seconds_json(session.activity),
                    last_output_secs: option_u64_json(last_output_secs),
                    observed_activity_idle_secs: option_u64_json(observed_idle_secs),
                    observed_activity_at: datetime_option_json(self.activity_observed_at),
                    activity_hint: json!(activity_hint(last_output_secs, activity_options)),
                    activity_error: Value::Null,
                }
            }
            Some(LiveActivity::Missing) | None => SessionActivityJson {
                tmux_present: json!(false),
                tmux_session_id: Value::Null,
                tmux_activity: option_u64_json(self.last_tmux_activity),
                tmux_activity_at: self
                    .last_tmux_activity
                    .map(epoch_seconds_json)
                    .unwrap_or(Value::Null),
                last_output_secs: option_u64_json(
                    self.last_tmux_activity
                        .and_then(|activity| seconds_since_epoch(now, activity)),
                ),
                observed_activity_idle_secs: option_u64_json(
                    self.activity_observed_at
                        .and_then(|observed_at| seconds_between(observed_at, now)),
                ),
                observed_activity_at: datetime_option_json(self.activity_observed_at),
                activity_hint: json!("missing"),
                activity_error: Value::Null,
            },
            Some(LiveActivity::Error(err)) => SessionActivityJson {
                tmux_present: Value::Null,
                tmux_session_id: Value::Null,
                tmux_activity: option_u64_json(self.last_tmux_activity),
                tmux_activity_at: self
                    .last_tmux_activity
                    .map(epoch_seconds_json)
                    .unwrap_or(Value::Null),
                last_output_secs: option_u64_json(
                    self.last_tmux_activity
                        .and_then(|activity| seconds_since_epoch(now, activity)),
                ),
                observed_activity_idle_secs: option_u64_json(
                    self.activity_observed_at
                        .and_then(|observed_at| seconds_between(observed_at, now)),
                ),
                observed_activity_at: datetime_option_json(self.activity_observed_at),
                activity_hint: json!("unknown"),
                activity_error: json!(err),
            },
        }
    }

    fn to_json(&self, target: &SessionTarget) -> Value {
        json!({
            "target": target.to_string(),
            "role": self.role,
            "agent": self.agent,
            "identity": self.identity,
            "state": self.state.as_str(),
            "cwd": self.cwd,
            "workstream": self.workstream,
            "last_report_kind": self.last_report_kind,
            "last_report_summary": self.last_report_summary,
            "updated_at": self.updated_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "context_domains": self.context_domains,
            "context_specialties": self.context_specialties,
            "context_summary": self.context_summary,
            "last_workstream": self.last_workstream,
            "last_workstream_title": self.last_workstream_title,
            "mmux_label": self.mmux_label,
            "mmux_previous_selected_key": self.mmux_previous_selected_key,
            "tmux_session_created": self.tmux_session_created,
            "tmux_activity": self.last_tmux_activity,
            "activity_observed_at": datetime_option_json(self.activity_observed_at),
        })
    }
}

fn session_identity_seed(target: &SessionTarget) -> String {
    if target.session_id_selector().is_some() {
        String::new()
    } else {
        target.session_name().to_string()
    }
}

fn session_reuse_reason(
    target: &SessionTarget,
    expected_created: u64,
    observed_created: u64,
) -> Option<String> {
    (expected_created != observed_created).then(|| {
        format!(
            "stale tmux session id for {target}: stored session_created={expected_created}, observed session_created={observed_created}"
        )
    })
}

impl TimerRecord {
    fn to_json(&self) -> Value {
        json!({
            "name": &self.name,
            "workstream": &self.workstream,
            "target": self.target.to_string(),
            "tmux_session_created": self.tmux_session_created,
            "every_secs": self.every_secs,
            "enter": self.enter,
            "submit_retries": self.submit_retries,
            "submit_retry_delay_ms": self.submit_retry_delay_ms,
            "input_quiet_for_secs": option_u64_json(self.input_quiet_for_secs),
            "generation": self.generation,
            "started_at": self.started_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "next_fire_at": datetime_option_json(self.next_fire_at),
            "last_fired_at": datetime_option_json(self.last_fired_at),
            "fire_count": self.fire_count,
            "defer_count": self.defer_count,
            "last_deferred_at": datetime_option_json(self.last_deferred_at),
            "last_defer_reason": &self.last_defer_reason,
            "last_input_activity": option_u64_json(self.last_input_activity),
            "last_input_activity_at": self
                .last_input_activity
                .map(epoch_seconds_json)
                .unwrap_or(Value::Null),
            "last_error": &self.last_error,
            "prompt_chars": char_count(&self.prompt),
        })
    }
}

struct SessionActivityJson {
    tmux_present: Value,
    tmux_session_id: Value,
    tmux_activity: Value,
    tmux_activity_at: Value,
    last_output_secs: Value,
    observed_activity_idle_secs: Value,
    observed_activity_at: Value,
    activity_hint: Value,
    activity_error: Value,
}

impl WorkstreamRecord {
    fn new(
        title: String,
        goal: Option<String>,
        domain: Option<String>,
        mmux_label: Option<String>,
        settings: WorkstreamSettings,
        generation: u64,
    ) -> Self {
        Self {
            title,
            goal,
            domain,
            mmux_label,
            mmux_label_conflicts: BTreeSet::new(),
            settings,
            state: WorkstreamState::Open,
            sessions: BTreeSet::new(),
            generation,
            next_sequence: 1,
            events: VecDeque::new(),
            handoffs: BTreeMap::new(),
        }
    }

    fn reopen(&mut self, generation: u64) {
        self.state = WorkstreamState::Open;
        self.generation = generation;
        self.next_sequence = 1;
        self.mmux_label_conflicts.clear();
        self.events.clear();
        self.handoffs.clear();
    }

    fn merge_hydrated_mmux_label(&mut self, label: String) {
        if self.mmux_label_conflicts.is_empty() {
            match self.mmux_label.as_deref() {
                None => self.mmux_label = Some(label),
                Some(existing) if existing == label => {}
                Some(existing) => {
                    self.mmux_label_conflicts.insert(existing.to_string());
                    self.mmux_label_conflicts.insert(label);
                    self.mmux_label = None;
                }
            }
        } else {
            self.mmux_label_conflicts.insert(label);
            self.mmux_label = None;
        }
    }

    fn cursor(&self, workstream: &str) -> anyhow::Result<String> {
        PublicCursor::new(workstream, self.generation, self.next_sequence).encode()
    }

    fn prune_events(&mut self) {
        while self.events.len() > self.settings.event_limit {
            self.events.pop_front();
        }
    }
}

impl WorkstreamState {
    fn as_str(self) -> &'static str {
        match self {
            WorkstreamState::Open => "open",
            WorkstreamState::Closed => "closed",
        }
    }
}

#[derive(Default)]
struct ParsedTags {
    managed: bool,
    workstream: Option<String>,
    workstream_title: Option<String>,
    role: Option<String>,
    agent: Option<String>,
    cwd: Option<PathBuf>,
    state: Option<AgentState>,
    last_report_kind: Option<String>,
    last_report_summary: Option<String>,
    context_domains: BTreeSet<String>,
    context_specialties: BTreeSet<String>,
    context_summary: Option<String>,
    last_workstream: Option<String>,
    last_workstream_title: Option<String>,
    mmux_label: Option<String>,
    mmux_previous_selected_key: Option<String>,
    updated_at: Option<DateTime<Utc>>,
}

impl ParsedTags {
    fn from_tags(tags: &[SessionTag]) -> Self {
        let mut parsed = ParsedTags::default();
        for tag in tags {
            match tag.key() {
                "managed" => parsed.managed = tag.value() == "true",
                "workstream" => parsed.workstream = Some(tag.value().to_string()),
                "workstream-title" => parsed.workstream_title = Some(tag.value().to_string()),
                "role" => parsed.role = Some(tag.value().to_string()),
                "agent" => parsed.agent = Some(tag.value().to_string()),
                "cwd" => parsed.cwd = Some(PathBuf::from(tag.value())),
                "state" => parsed.state = parse_state(tag.value()),
                "last-report-kind" => parsed.last_report_kind = Some(tag.value().to_string()),
                "last-report-summary" => parsed.last_report_summary = Some(tag.value().to_string()),
                "context-domains" => parsed.context_domains = split_tag_list(tag.value()),
                "context-specialties" => parsed.context_specialties = split_tag_list(tag.value()),
                "context-summary" => parsed.context_summary = Some(tag.value().to_string()),
                "last-workstream" => parsed.last_workstream = Some(tag.value().to_string()),
                "last-workstream-title" => {
                    parsed.last_workstream_title = Some(tag.value().to_string())
                }
                MSTREAM_MMUX_LABEL_KEY => parsed.mmux_label = Some(tag.value().to_string()),
                MSTREAM_MMUX_PREVIOUS_SELECTED_KEY => {
                    parsed.mmux_previous_selected_key = Some(tag.value().to_string())
                }
                "updated-at" => parsed.updated_at = tags::parse_updated_at(tag.value()),
                _ => {}
            }
        }
        parsed
    }
}

fn parse_state(value: &str) -> Option<AgentState> {
    match value {
        "available" => Some(AgentState::Available),
        "reserved" => Some(AgentState::Reserved),
        "busy" => Some(AgentState::Busy),
        "idle" => Some(AgentState::Idle),
        "done" => Some(AgentState::Done),
        "blocked" => Some(AgentState::Blocked),
        "needs-input" => Some(AgentState::NeedsInput),
        "quarantined" => Some(AgentState::Quarantined),
        _ => None,
    }
}

fn split_tag_list(value: &str) -> BTreeSet<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn mmux_tag_result(failures: Vec<String>) -> anyhow::Result<()> {
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("{}", failures.join("; "))
    }
}

fn validate_mmux_label(value: &str) -> anyhow::Result<String> {
    if value.chars().any(char::is_control) {
        bail!("--mmux-label cannot contain control characters");
    }
    if value.chars().any(is_unicode_format_char) {
        bail!("--mmux-label cannot contain Unicode format characters");
    }
    let words = value.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        bail!("--mmux-label cannot be empty");
    }
    if words.len() > 2 {
        bail!("--mmux-label must be one or two words");
    }
    let label = words.join(" ");
    if UnicodeWidthStr::width(label.as_str()) > MMUX_LABEL_MAX_CHARS {
        bail!("--mmux-label must be {MMUX_LABEL_MAX_CHARS} display columns or fewer");
    }
    Ok(label)
}

fn validate_session_display_name(value: &str) -> anyhow::Result<String> {
    if value.chars().any(char::is_control) {
        bail!("<new-name> cannot contain control characters");
    }
    if value.chars().any(is_unicode_format_char) {
        bail!("<new-name> cannot contain Unicode format characters");
    }
    let name = value.trim();
    if name.is_empty() {
        bail!("<new-name> cannot be empty");
    }
    if name.contains("::") {
        bail!("<new-name> cannot contain '::'");
    }
    if name.contains(':') {
        bail!("<new-name> cannot contain ':'");
    }
    if looks_like_tmux_session_id(name) {
        bail!("<new-name> cannot look like a tmux session id");
    }
    Ok(name.to_string())
}

fn validate_retag_text(flag: &str, value: &str) -> anyhow::Result<String> {
    if value.chars().any(char::is_control) {
        bail!("{flag} cannot contain control characters");
    }
    if value.chars().any(is_unicode_format_char) {
        bail!("{flag} cannot contain Unicode format characters");
    }
    let value = value.trim();
    if value.is_empty() {
        bail!("{flag} cannot be empty");
    }
    Ok(value.to_string())
}

fn looks_like_tmux_session_id(value: &str) -> bool {
    let Some(rest) = value.strip_prefix('$') else {
        return false;
    };
    !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
}

fn ensure_session_target(target: &SessionTarget) -> anyhow::Result<()> {
    if target.target_spec().window_selector().is_some()
        || target.target_spec().pane_index().is_some()
    {
        bail!("target '{target}' must identify a tmux session");
    }
    Ok(())
}

fn is_unicode_format_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x00AD
            | 0x0600..=0x0605
            | 0x061C
            | 0x06DD
            | 0x070F
            | 0x0890..=0x0891
            | 0x08E2
            | 0x180E
            | 0x200B..=0x200F
            | 0x202A..=0x202E
            | 0x2060..=0x206F
            | 0xFEFF
            | 0xFFF9..=0xFFFB
            | 0x110BD
            | 0x110CD
            | 0x13430..=0x1343F
            | 0x1BCA0..=0x1BCA3
            | 0x1D173..=0x1D17A
            | 0xE0000..=0xE0FFF
    )
}

async fn validate_agent_executable(handle: &HostHandle, agent: &str) -> anyhow::Result<()> {
    if agent.trim().is_empty() {
        bail!("--agent must not be empty");
    }
    let exists = handle
        .executable_exists_non_login(agent)
        .await
        .with_context(|| {
            format!(
                "failed to validate agent executable {} on {} non-login PATH",
                agent,
                handle.host_alias()
            )
        })?;
    if !exists {
        bail!(
            "agent executable {} not found on {} non-login PATH - pass an absolute path",
            agent,
            handle.host_alias()
        );
    }
    Ok(())
}

fn is_created_session_not_found(error: &TmuxError, session_name: &str) -> bool {
    if let TmuxError::State(message) = error {
        message == &format!("session '{session_name}' created but not found in list")
    } else {
        false
    }
}

fn bootstrap_command(cwd: &Path, agent: &str) -> String {
    format!(
        "mkdir -p {} && cd {} && exec {}",
        shell_quote(&cwd.display().to_string()),
        shell_quote(&cwd.display().to_string()),
        shell_quote(agent)
    )
}

fn session_environment(workstream: &str, role: &str) -> anyhow::Result<Vec<SessionEnvVar>> {
    Ok(vec![
        SessionEnvVar::new("MSTREAM_WORKSTREAM", workstream)?,
        SessionEnvVar::new("MSTREAM_ROLE", role)?,
    ])
}

fn timer_fire_outcome_json(outcome: TimerFireOutcome) -> Value {
    match outcome {
        TimerFireOutcome::Sent(snapshot) => json!({
            "type": "ok",
            "op": "timer_fire",
            "name": snapshot.name,
            "target": snapshot.target.to_string(),
            "generation": snapshot.generation,
            "input_quiet_for_secs": option_u64_json(snapshot.input_quiet_for_secs),
        }),
        TimerFireOutcome::Deferred(snapshot) => json!({
            "type": "ok",
            "op": "timer_deferred",
            "name": snapshot.name,
            "target": snapshot.target.to_string(),
            "generation": snapshot.generation,
            "reason": snapshot.reason,
            "latest_client_activity": option_u64_json(snapshot.latest_client_activity),
            "latest_client_activity_at": snapshot
                .latest_client_activity
                .map(epoch_seconds_json)
                .unwrap_or(Value::Null),
            "latest_client_activity_age_secs": option_u64_json(
                snapshot.latest_client_activity_age_secs
            ),
            "quiet_for_secs": snapshot.quiet_for_secs,
            "next_fire_at": datetime_option_json(snapshot.next_fire_at),
        }),
    }
}

fn render_readable_event(event: &EventRecord) -> String {
    let mut first_line = format!("{}  {}", event.timestamp, event.kind);
    if let Some(target) = &event.target {
        first_line.push_str(&format!("  {target}"));
    }
    if let Some(state) = event.state {
        first_line.push_str(&format!("  state={state}"));
    }
    let mut rendered = first_line;
    if let Some(summary) = &event.summary {
        rendered.push_str(&format!("\n  summary: {}", compact_inline(summary, 240)));
    }
    if let Some(text) = &event.text {
        if event.summary.as_deref() != Some(text.as_str()) {
            rendered.push_str(&format!("\n  {}", compact_inline(text, 320)));
        }
    }
    rendered
}

fn compact_inline(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&compact, max_chars)
}

fn managed_prompt(
    workstream: &str,
    target: &SessionTarget,
    role: Option<&str>,
    cwd: Option<&Path>,
    task: Option<&str>,
) -> String {
    let mut prompt = format!(
        "You are \"{}\". Workstream: {}. Target: {}.",
        target.session_name(),
        workstream,
        target
    );
    if let Some(role) = role {
        prompt.push_str(&format!(" Role: {}.", role));
    }
    if let Some(cwd) = cwd {
        prompt.push_str(&format!(" CWD: {}.", cwd.display()));
    }
    prompt.push_str(
        " Report progress, blockers, questions, PR links, pushed commits, and review comments \
         plainly in your normal output. The orchestrator owns workstream state and will decide \
         whether you should continue, stand by, or wait for feedback.",
    );
    if let Some(task) = task {
        prompt.push_str("\n\nTask:\n");
        prompt.push_str(task);
    }
    prompt
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    let escaped = value.replace('\'', "'\\''");
    format!("'{escaped}'")
}

fn compact_text(text: &str, max_chars: usize) -> String {
    let mut previous_blank = false;
    let mut compacted = String::new();
    for line in text.lines() {
        let blank = line.trim().is_empty();
        if blank && previous_blank {
            continue;
        }
        previous_blank = blank;
        compacted.push_str(line);
        compacted.push('\n');
        if char_count(&compacted) >= max_chars {
            break;
        }
    }
    truncate_chars(&compacted, max_chars)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    text.chars().take(max_chars).collect()
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

fn activity_hint(last_output_secs: Option<u64>, options: StatusActivityOptions) -> &'static str {
    match last_output_secs {
        Some(age) if age <= options.active_window_secs => "active",
        Some(age) if age >= options.idle_after_secs => "idle",
        Some(_) => "quiet",
        None => "unknown",
    }
}

fn seconds_since_epoch(now: DateTime<Utc>, epoch_seconds: u64) -> Option<u64> {
    let now_seconds = now.timestamp();
    if now_seconds < 0 {
        return None;
    }
    let now_seconds = now_seconds as u64;
    if epoch_seconds <= now_seconds {
        Some(now_seconds - epoch_seconds)
    } else {
        None
    }
}

fn seconds_between(start: DateTime<Utc>, end: DateTime<Utc>) -> Option<u64> {
    end.signed_duration_since(start)
        .to_std()
        .ok()
        .map(|duration| duration.as_secs())
}

fn epoch_seconds_json(epoch_seconds: u64) -> Value {
    if epoch_seconds > i64::MAX as u64 {
        return Value::Null;
    }
    DateTime::<Utc>::from_timestamp(epoch_seconds as i64, 0)
        .map(|timestamp| json!(timestamp.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)))
        .unwrap_or(Value::Null)
}

fn datetime_option_json(timestamp: Option<DateTime<Utc>>) -> Value {
    timestamp
        .map(|timestamp| json!(timestamp.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)))
        .unwrap_or(Value::Null)
}

fn option_u64_json(value: Option<u64>) -> Value {
    value.map(Value::from).unwrap_or(Value::Null)
}

fn add_secs(timestamp: DateTime<Utc>, seconds: u64) -> Option<DateTime<Utc>> {
    chrono::Duration::from_std(Duration::from_secs(seconds))
        .ok()
        .and_then(|duration| timestamp.checked_add_signed(duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_host(alias: &str, mock: motlie_tmux::transport::MockTransport) -> HostHandle {
        HostHandle::with_alias(motlie_tmux::TransportKind::Mock(mock), None, alias)
    }

    fn register_mock_host(
        state: &mut DaemonState,
        alias: &str,
        mock: motlie_tmux::transport::MockTransport,
    ) {
        state.fleet.register(alias, mock_host(alias, mock)).unwrap();
    }

    fn session_info(name: &str, id: &str, created: u64, activity: u64) -> SessionInfo {
        SessionInfo {
            name: name.to_string(),
            id: id.try_into().expect("session id"),
            created,
            attached_count: 0,
            window_count: 1,
            group: None,
            activity,
        }
    }

    fn open_test_workstream(state: &mut DaemonState, workstream: &str) {
        state
            .open(OpenRequest {
                workstream: workstream.to_string(),
                title: workstream.to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");
    }

    fn timer_record(name: &str, target: SessionTarget, created: Option<u64>) -> TimerRecord {
        let started_at = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        TimerRecord {
            name: name.to_string(),
            workstream: Some("issue-355".to_string()),
            target,
            tmux_session_created: created,
            every_secs: 60,
            prompt: "Wake up.".to_string(),
            enter: true,
            submit_retries: 0,
            submit_retry_delay_ms: 0,
            input_quiet_for_secs: None,
            generation: 1,
            started_at,
            next_fire_at: Some(started_at),
            last_fired_at: None,
            fire_count: 0,
            defer_count: 0,
            last_deferred_at: None,
            last_defer_reason: None,
            last_input_activity: None,
            last_error: None,
            task: None,
        }
    }

    fn handoff_record(id: &str, target: SessionTarget, created: Option<u64>) -> HandoffRecord {
        HandoffRecord {
            id: id.to_string(),
            from: target.clone(),
            to: target,
            from_tmux_session_created: created,
            to_tmux_session_created: created,
            on: AgentState::Done,
            task: "Take over.".to_string(),
            only_on_transition: false,
            fired: false,
            canceled_reason: None,
            created_at: tags::now_tag(),
        }
    }

    fn seed_stale_reused_target(state: &mut DaemonState, workstream: &str, target: &SessionTarget) {
        open_test_workstream(state, workstream);
        state
            .add_session_to_workstream(
                workstream,
                target.clone(),
                "reviewer".to_string(),
                Some("old-agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add stale session");
        state
            .sessions
            .get_mut(target)
            .expect("stale session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut(workstream)
            .expect("old workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );
    }

    fn assert_stale_target_quarantined(
        state: &DaemonState,
        workstream: &str,
        target: &SessionTarget,
    ) {
        assert!(!state
            .workstream(workstream)
            .expect("old workstream")
            .sessions
            .contains(target));
        let timer = state.timers.get("poll").expect("timer");
        assert_eq!(timer.next_fire_at, None);
        assert!(timer
            .last_error
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
        assert!(state
            .workstream(workstream)
            .expect("old workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
    }

    #[test]
    fn shell_quote_handles_single_quote() {
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn compact_text_collapses_repeated_blank_lines() {
        let text = "a\n\n\nb\n";
        assert_eq!(compact_text(text, 100), "a\n\nb\n");
    }

    #[test]
    fn validate_mmux_label_trims_and_bounds_label() {
        assert_eq!(
            validate_mmux_label("  Issue   349  ").expect("valid label"),
            "Issue 349"
        );
        assert!(validate_mmux_label("").is_err());
        assert!(validate_mmux_label("one two three").is_err());
        assert!(validate_mmux_label("abcdefghijklmnopqrstuvwxyz").is_err());
        assert!(validate_mmux_label("界界界界界界界界界界界界界").is_err());
        assert!(validate_mmux_label("bad\nlabel").is_err());
        assert!(validate_mmux_label("bad\u{200d}label").is_err());
        assert!(validate_mmux_label("bad\u{202e}label").is_err());
    }

    #[test]
    fn open_records_mmux_label_metadata() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "issue-349".to_string(),
                title: "Issue 349".to_string(),
                goal: None,
                domain: None,
                mmux_label: Some("349 labels".to_string()),
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");

        let show = state.show("issue-349").expect("show workstream");
        assert_eq!(show["mmux_label"], "349 labels");

        state
            .open(OpenRequest {
                workstream: "issue-349".to_string(),
                title: "Issue 349 updated".to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("reopen workstream");

        let show = state.show("issue-349").expect("show workstream");
        assert_eq!(show["mmux_label"], "349 labels");
    }

    #[test]
    fn scan_hydration_conflicts_do_not_choose_order_dependent_label() {
        let mut workstream = WorkstreamRecord::new(
            "Issue 349".to_string(),
            None,
            None,
            None,
            WorkstreamSettings { event_limit: 10 },
            1,
        );

        workstream.merge_hydrated_mmux_label("349 labels".to_string());
        assert_eq!(workstream.mmux_label.as_deref(), Some("349 labels"));
        assert!(workstream.mmux_label_conflicts.is_empty());

        workstream.merge_hydrated_mmux_label("349 review".to_string());
        assert_eq!(workstream.mmux_label, None);
        assert_eq!(
            workstream.mmux_label_conflicts,
            BTreeSet::from(["349 labels".to_string(), "349 review".to_string()])
        );

        workstream.merge_hydrated_mmux_label("349 labels".to_string());
        assert_eq!(workstream.mmux_label, None);
        assert_eq!(workstream.mmux_label_conflicts.len(), 2);
    }

    #[test]
    fn stable_session_targets_are_host_scoped() {
        let first = SessionTarget::session_id("host-a", "$0").expect("first target");
        let second = SessionTarget::session_id("host-b", "$0").expect("second target");

        assert_ne!(first, second);
    }

    #[test]
    fn session_record_does_not_seed_display_identity_from_session_id() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let record = SessionRecord::from_target(&target, AgentState::Busy, None);

        assert_eq!(record.identity, "");
    }

    #[test]
    fn session_record_display_name_updates_without_rekeying() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "issue-355".to_string(),
                title: "Issue 355".to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");

        let old_session = SessionInfo {
            name: "old-name".to_string(),
            id: "$1".try_into().expect("old session id"),
            created: 100,
            attached_count: 0,
            window_count: 1,
            group: None,
            activity: 200,
        };
        let renamed_session = SessionInfo {
            name: "new-name".to_string(),
            id: "$1".try_into().expect("new session id"),
            created: 100,
            attached_count: 0,
            window_count: 1,
            group: None,
            activity: 300,
        };

        state
            .sessions
            .get_mut(&target)
            .expect("session record")
            .observe_tmux_session(&old_session);
        state
            .sessions
            .get_mut(&target)
            .expect("session record")
            .observe_tmux_session(&renamed_session);

        assert_eq!(state.sessions.len(), 1);
        let record = state.sessions.get(&target).expect("stable keyed record");
        assert_eq!(record.identity, "new-name");
        assert_eq!(record.role.as_deref(), Some("reviewer"));
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
    }

    #[test]
    fn validate_session_display_name_rejects_ambiguous_names() {
        assert_eq!(
            validate_session_display_name("  renamed  ").expect("valid name"),
            "renamed"
        );
        assert!(validate_session_display_name("").is_err());
        assert!(validate_session_display_name("work:0").is_err());
        assert!(validate_session_display_name("local::work").is_err());
        assert!(validate_session_display_name("$7").is_err());
        assert!(validate_session_display_name("bad\nname").is_err());
        assert!(validate_session_display_name("bad\u{200d}name").is_err());
    }

    #[tokio::test]
    async fn session_rename_round_trip_keeps_stable_id_key() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ old-name $1 100 0 1  150\n")
            .with_response("rename-session -t '$1' 'new-name'", "")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .workstream_mut("issue-355")
            .expect("workstream")
            .mmux_label = Some("355 work".to_string());
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old-name", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-355")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::session_retag_shared(
            Arc::clone(&shared),
            SessionRetagRequest {
                target: "local::old-name".to_string(),
                new_name: Some("new-name".to_string()),
                role: None,
                workstream: None,
                mmux_label: None,
            },
        )
        .await
        .expect("rename");

        assert_eq!(records[0]["op"], "rename");
        assert_eq!(records[0]["target"], "local::$1");
        assert_eq!(records[0]["renamed"], true);
        assert_eq!(records[0]["old_name"], "old-name");
        assert_eq!(records[0]["new_name"], "new-name");
        assert_eq!(records[0]["mmux_label"], "355 work");

        let state = shared.lock().await;
        assert_eq!(state.sessions.len(), 1);
        let record = state.sessions.get(&target).expect("stable keyed record");
        assert_eq!(record.identity, "new-name");
        assert_eq!(record.role.as_deref(), Some("reviewer"));
        assert_eq!(record.mmux_label.as_deref(), Some("355 work"));
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
        assert_eq!(
            state.timers.get("poll").expect("timer").target,
            target,
            "timer stays keyed by stable target"
        );
        let handoff = state
            .workstream("issue-355")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff");
        assert_eq!(handoff.from, target);
        assert!(handoff.canceled_reason.is_none());
        let events = &state.workstream("issue-355").expect("workstream").events;
        assert!(events
            .iter()
            .any(|event| event.kind == "renamed" && event.target.as_deref() == Some("local::$1")));
    }

    #[tokio::test]
    async fn session_retag_moves_workstream_role_and_mmux_label() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ worker $1 100 0 1  150\n")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        open_test_workstream(&mut state, "issue-360");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                Some(PathBuf::from("/tmp/agent")),
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("worker", "$1", 100, 150));
        state
            .workstream_mut("issue-355")
            .expect("old workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::session_retag_shared(
            Arc::clone(&shared),
            SessionRetagRequest {
                target: "local::$1".to_string(),
                new_name: None,
                role: Some("implementer".to_string()),
                workstream: Some("issue-360".to_string()),
                mmux_label: Some("360 impl".to_string()),
            },
        )
        .await
        .expect("retag");

        assert_eq!(records[0]["op"], "session_retag");
        assert_eq!(records[0]["target"], "local::$1");
        assert_eq!(records[0]["role"], "implementer");
        assert_eq!(records[0]["workstream"], "issue-360");
        assert_eq!(records[0]["mmux_label"], "360 impl");
        assert_eq!(records[0]["mmux_label_cleared"], true);
        assert_eq!(records[0]["stale_handoffs_canceled"], 1);

        let state = shared.lock().await;
        assert!(!state
            .workstream("issue-355")
            .expect("old workstream")
            .sessions
            .contains(&target));
        assert!(state
            .workstream("issue-360")
            .expect("new workstream")
            .sessions
            .contains(&target));
        assert_eq!(
            state
                .workstream("issue-360")
                .expect("new workstream")
                .mmux_label
                .as_deref(),
            Some("360 impl")
        );
        let record = state.sessions.get(&target).expect("session");
        assert_eq!(record.identity, "worker");
        assert_eq!(record.role.as_deref(), Some("implementer"));
        assert_eq!(record.workstream.as_deref(), Some("issue-360"));
        assert_eq!(record.mmux_label.as_deref(), Some("360 impl"));
        assert!(state
            .workstream("issue-355")
            .expect("old workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("session retagged"));
    }

    #[tokio::test]
    async fn session_retag_mmux_label_refreshes_same_workstream_sibling() {
        let target_a = SessionTarget::session_id("local", "$1").expect("target a");
        let target_b = SessionTarget::session_id("local", "$2").expect("target b");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response(
                "list-sessions",
                "__MOTLIE_S__ worker-a $1 100 0 1  150\n\
                 __MOTLIE_S__ worker-b $2 101 0 1  151\n",
            )
            .with_default("");
        let command_log = mock.command_log();
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-360");
        state
            .workstream_mut("issue-360")
            .expect("workstream")
            .mmux_label = Some("old ws".to_string());
        state
            .add_session_to_workstream(
                "issue-360",
                target_a.clone(),
                "implementer".to_string(),
                Some("agent-a".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session a");
        state
            .add_session_to_workstream(
                "issue-360",
                target_b.clone(),
                "reviewer".to_string(),
                Some("agent-b".to_string()),
                None,
                AgentState::Idle,
            )
            .expect("add session b");
        state
            .workstream_mut("issue-360")
            .expect("workstream")
            .sessions
            .remove(&target_b);
        state
            .sessions
            .get_mut(&target_a)
            .expect("session a")
            .observe_tmux_session(&session_info("worker-a", "$1", 100, 150));
        state
            .sessions
            .get_mut(&target_b)
            .expect("session b")
            .observe_tmux_session(&session_info("worker-b", "$2", 101, 151));

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::session_retag_shared(
            Arc::clone(&shared),
            SessionRetagRequest {
                target: "local::$1".to_string(),
                new_name: None,
                role: None,
                workstream: None,
                mmux_label: Some("new ws".to_string()),
            },
        )
        .await
        .expect("retag mmux label");

        assert_eq!(records[0]["op"], "session_retag");
        assert_eq!(records[0]["target"], "local::$1");
        assert_eq!(records[0]["workstream"], "issue-360");
        assert_eq!(records[0]["mmux_label"], "new ws");
        assert_eq!(records[0]["mmux_labels_applied"], 2);
        assert_eq!(records[0]["mmux_label_apply_failed"], 0);

        let state = shared.lock().await;
        assert_eq!(
            state
                .workstream("issue-360")
                .expect("workstream")
                .mmux_label
                .as_deref(),
            Some("new ws")
        );
        assert_eq!(
            state
                .sessions
                .get(&target_a)
                .expect("session a")
                .mmux_label
                .as_deref(),
            Some("new ws")
        );
        assert_eq!(
            state
                .sessions
                .get(&target_b)
                .expect("session b")
                .mmux_label
                .as_deref(),
            Some("new ws")
        );
        assert!(state
            .workstream("issue-360")
            .expect("workstream")
            .sessions
            .contains(&target_b));
        let applied_targets = state
            .workstream("issue-360")
            .expect("workstream")
            .events
            .iter()
            .filter(|event| event.kind == "mmux_label_applied")
            .filter_map(|event| event.target.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            applied_targets,
            BTreeSet::from(["local::$1".to_string(), "local::$2".to_string()])
        );
        let commands = command_log.lock().expect("command log").clone();
        assert!(
            commands
                .iter()
                .any(|command| command.contains("set-option -t '$2' @mmux/mstream 'new ws'")),
            "sibling mmux tag was not refreshed; commands: {commands:#?}"
        );
    }

    #[tokio::test]
    async fn session_rename_quarantines_reused_session_id_before_mutating() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ reused $1 200 0 1  250\n")
            .with_error("rename-session", "rename should not be attempted")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));

        let shared = Arc::new(Mutex::new(state));
        let err = DaemonState::session_retag_shared(
            Arc::clone(&shared),
            SessionRetagRequest {
                target: "local::$1".to_string(),
                new_name: Some("new".to_string()),
                role: None,
                workstream: None,
                mmux_label: None,
            },
        )
        .await
        .expect_err("reused id should fail");

        assert!(err.to_string().contains("stale tmux session id"));
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        assert!(!state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
    }

    #[tokio::test]
    async fn session_rename_error_leaves_state_unchanged() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ old $1 100 0 1  150\n")
            .with_error("rename-session", "duplicate session")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));

        let shared = Arc::new(Mutex::new(state));
        let err = DaemonState::session_retag_shared(
            Arc::clone(&shared),
            SessionRetagRequest {
                target: "local::old".to_string(),
                new_name: Some("new".to_string()),
                role: None,
                workstream: None,
                mmux_label: None,
            },
        )
        .await
        .expect_err("rename collision should fail");

        assert!(err.to_string().contains("duplicate session"));
        let state = shared.lock().await;
        let record = state.sessions.get(&target).expect("session");
        assert_eq!(record.identity, "old");
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
    }

    #[tokio::test]
    async fn status_quarantines_reused_session_id_instead_of_reporting_present() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ new $1 200 0 1  250\n");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-355")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records =
            DaemonState::status_shared(Arc::clone(&shared), "issue-355".to_string(), 30, 300)
                .await
                .expect("status");

        assert_eq!(records[0]["agents"].as_array().expect("agents").len(), 0);
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        assert!(!state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
        let timer = state.timers.get("poll").expect("timer");
        assert_eq!(timer.next_fire_at, None);
        assert!(timer
            .last_error
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
    }

    #[tokio::test]
    async fn retire_quarantines_session_but_keeps_workstream_membership() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ worker $1 100 0 1  150\n")
            .with_default("");
        let command_log = mock.command_log();
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-401");
        state
            .add_session_to_workstream(
                "issue-401",
                target.clone(),
                "implementer".to_string(),
                Some("codex".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("worker", "$1", 100, 150));

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::retire_shared(
            Arc::clone(&shared),
            RetireRequest {
                workstream: "issue-401".to_string(),
                target: "local::$1".to_string(),
            },
        )
        .await
        .expect("retire");

        assert_eq!(records[0]["op"], "retire");
        assert_eq!(records[0]["state"], "quarantined");
        assert_eq!(records[0]["previous_state"], "busy");
        let state = shared.lock().await;
        let record = state.sessions.get(&target).expect("session retained");
        assert_eq!(record.state, AgentState::Quarantined);
        assert!(state
            .workstream("issue-401")
            .expect("workstream")
            .sessions
            .contains(&target));
        assert!(state
            .workstream("issue-401")
            .expect("workstream")
            .events
            .iter()
            .any(|event| event.kind == "retired"
                && event.target.as_deref() == Some("local::$1")
                && event.state == Some("quarantined")));
        drop(state);
        let commands = command_log.lock().expect("command log").clone();
        assert!(
            commands
                .iter()
                .any(|command| command.contains("set-option -t '$1' @mstream/state 'quarantined'")),
            "retire did not write quarantined state tag; commands: {commands:#?}"
        );
    }

    #[tokio::test]
    async fn reclaim_requires_quarantined_managed_session() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ worker $1 100 0 1  150\n")
            .with_response(
                "show-options -t '$1'",
                "@mstream/managed \"true\"\n@mstream/state \"busy\"\n@mstream/workstream \"issue-401\"\n",
            )
            .with_error("kill-session", "should not kill non-quarantined session");
        let command_log = mock.command_log();
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-401");
        state
            .add_session_to_workstream(
                "issue-401",
                target.clone(),
                "implementer".to_string(),
                Some("codex".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("worker", "$1", 100, 150));
        let shared = Arc::new(Mutex::new(state));

        let err = DaemonState::reclaim_shared(Arc::clone(&shared), "local::$1".to_string())
            .await
            .expect_err("busy session should not reclaim");

        assert!(err.to_string().contains("not quarantined"));
        let state = shared.lock().await;
        assert!(state.sessions.contains_key(&target));
        drop(state);
        let commands = command_log.lock().expect("command log").clone();
        assert!(
            !commands
                .iter()
                .any(|command| command.contains("kill-session")),
            "reclaim attempted kill before gate passed; commands: {commands:#?}"
        );
    }

    #[tokio::test]
    async fn reclaim_kills_and_deregisters_quarantined_managed_session() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ worker $1 100 0 1  150\n")
            .with_response(
                "show-options -t '$1'",
                "@mstream/managed \"true\"\n@mstream/state \"quarantined\"\n@mstream/workstream \"issue-401\"\n",
            )
            .with_response("kill-session -t '$1'", "")
            .with_default("");
        let command_log = mock.command_log();
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-401");
        state
            .add_session_to_workstream(
                "issue-401",
                target.clone(),
                "implementer".to_string(),
                Some("codex".to_string()),
                None,
                AgentState::Quarantined,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("worker", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-401")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::reclaim_shared(Arc::clone(&shared), "local::$1".to_string())
            .await
            .expect("reclaim");

        assert_eq!(records[0]["op"], "reclaim");
        assert_eq!(records[0]["workstream"], "issue-401");
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        let workstream = state.workstream("issue-401").expect("workstream");
        assert!(!workstream.sessions.contains(&target));
        assert!(
            workstream
                .events
                .iter()
                .any(|event| event.kind == "reclaimed"
                    && event.target.as_deref() == Some("local::$1"))
        );
        assert!(workstream
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("reclaimed"));
        assert!(state
            .timers
            .get("poll")
            .expect("timer")
            .last_error
            .as_deref()
            .unwrap()
            .contains("reclaimed"));
        drop(state);
        let commands = command_log.lock().expect("command log").clone();
        assert!(
            commands
                .iter()
                .any(|command| command.contains("kill-session -t '$1'")),
            "reclaim did not kill stable tmux id; commands: {commands:#?}"
        );
    }

    #[tokio::test]
    async fn scan_deregisters_missing_session_records() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-401");
        state
            .add_session_to_workstream(
                "issue-401",
                target.clone(),
                "implementer".to_string(),
                Some("codex".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("worker", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-401")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::scan_shared(Arc::clone(&shared), "local".to_string())
            .await
            .expect("scan");

        assert_eq!(records[0]["hydrated_sessions"], 0);
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        assert!(!state
            .workstream("issue-401")
            .expect("workstream")
            .sessions
            .contains(&target));
        assert!(state
            .timers
            .get("poll")
            .expect("timer")
            .last_error
            .as_deref()
            .unwrap()
            .contains("no longer present"));
        assert!(state
            .workstream("issue-401")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("no longer present"));
    }

    #[tokio::test]
    async fn scan_keeps_live_untagged_session_records() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ live $1 100 0 1  160\n")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-401");
        state
            .add_session_to_workstream(
                "issue-401",
                target.clone(),
                "implementer".to_string(),
                Some("codex".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("live", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-401")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::scan_shared(Arc::clone(&shared), "local".to_string())
            .await
            .expect("scan");

        assert_eq!(records[0]["hydrated_sessions"], 0);
        let state = shared.lock().await;
        assert!(state.sessions.contains_key(&target));
        assert!(state
            .workstream("issue-401")
            .expect("workstream")
            .sessions
            .contains(&target));
        assert!(state
            .timers
            .get("poll")
            .expect("timer")
            .last_error
            .is_none());
        assert!(state
            .workstream("issue-401")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .is_none());
    }

    #[tokio::test]
    async fn scan_quarantines_untagged_reused_session_id() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ new $1 200 0 1  250\n")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Busy,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        state
            .workstream_mut("issue-355")
            .expect("workstream")
            .handoffs
            .insert(
                "h1".to_string(),
                handoff_record("h1", target.clone(), Some(100)),
            );

        let shared = Arc::new(Mutex::new(state));
        let records = DaemonState::scan_shared(Arc::clone(&shared), "local".to_string())
            .await
            .expect("scan");

        assert_eq!(records[0]["hydrated_sessions"], 0);
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        assert!(!state
            .workstream("issue-355")
            .expect("workstream")
            .sessions
            .contains(&target));
        let timer = state.timers.get("poll").expect("timer");
        assert_eq!(timer.next_fire_at, None);
        assert!(timer
            .last_error
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
    }

    #[tokio::test]
    async fn timer_fire_aborts_before_sending_to_reused_session_id() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ new $1 200 0 1  250\n")
            .with_error("send-keys", "should not send to reused session id");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        state.sessions.insert(
            target.clone(),
            SessionRecord::from_target(&target, AgentState::Busy, None),
        );
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));
        state.timers.insert(
            "poll".to_string(),
            timer_record("poll", target.clone(), Some(100)),
        );
        let shared = Arc::new(Mutex::new(state));

        let err =
            match DaemonState::timer_fire_once_shared(Arc::clone(&shared), "poll", Some(1), true)
                .await
            {
                Ok(_) => panic!("stale timer should fail"),
                Err(err) => err,
            };

        assert!(err.to_string().contains("stale tmux session id"));
        let state = shared.lock().await;
        let timer = state.timers.get("poll").expect("timer");
        assert_eq!(timer.next_fire_at, None);
        assert!(timer
            .last_error
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
    }

    #[tokio::test]
    async fn handoff_fire_cancels_before_sending_to_reused_session_id() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ new $1 200 0 1  250\n")
            .with_error("send-keys", "should not send handoff to reused session id");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        open_test_workstream(&mut state, "issue-355");
        state
            .add_session_to_workstream(
                "issue-355",
                target.clone(),
                "reviewer".to_string(),
                Some("agent".to_string()),
                None,
                AgentState::Done,
            )
            .expect("add session");
        state
            .sessions
            .get_mut(&target)
            .expect("session")
            .observe_tmux_session(&session_info("old", "$1", 100, 150));
        let handoff = handoff_record("h1", target.clone(), Some(100));
        state
            .workstream_mut("issue-355")
            .expect("workstream")
            .handoffs
            .insert("h1".to_string(), handoff.clone());
        let shared = Arc::new(Mutex::new(state));

        let err =
            DaemonState::fire_claimed_handoff_shared(Arc::clone(&shared), "issue-355", handoff)
                .await
                .expect_err("stale handoff should fail");

        assert!(err.to_string().contains("stale tmux session id"));
        let state = shared.lock().await;
        assert!(!state.sessions.contains_key(&target));
        assert!(state
            .workstream("issue-355")
            .expect("workstream")
            .handoffs
            .get("h1")
            .expect("handoff")
            .canceled_reason
            .as_deref()
            .unwrap()
            .contains("stale tmux session id"));
    }

    #[tokio::test]
    async fn join_quarantines_reused_session_id_before_adopting() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ joined $1 200 0 1  250\n")
            .with_shell_data(vec![b"%output %5 ready\n".to_vec()])
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        seed_stale_reused_target(&mut state, "issue-355-old", &target);
        open_test_workstream(&mut state, "issue-355-new");
        let shared = Arc::new(Mutex::new(state));

        let records = DaemonState::join_shared(
            Arc::clone(&shared),
            JoinRequest {
                workstream: "issue-355-new".to_string(),
                target: target.to_string(),
                role: "implementer".to_string(),
                task: None,
            },
        )
        .await
        .expect("join reused id as fresh session");

        assert_eq!(records[0]["op"], "join");
        assert_eq!(records[0]["target"], target.to_string());
        let state = shared.lock().await;
        assert_stale_target_quarantined(&state, "issue-355-old", &target);
        assert!(state
            .workstream("issue-355-new")
            .expect("new workstream")
            .sessions
            .contains(&target));
        let record = state.sessions.get(&target).expect("fresh record");
        assert_eq!(record.role.as_deref(), Some("implementer"));
        assert_eq!(record.agent, None);
        assert_eq!(record.workstream.as_deref(), Some("issue-355-new"));
        assert_eq!(record.identity, "joined");
        assert_eq!(record.tmux_session_created, Some(200));
    }

    #[tokio::test]
    async fn new_session_fails_fast_when_agent_missing_on_non_login_path() {
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("command -v 'missing-agent'", "missing")
            .with_default("");
        let commands = mock.command_log();
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "remote", mock);
        open_test_workstream(&mut state, "issue-386");
        let shared = Arc::new(Mutex::new(state));

        let error = DaemonState::new_session_shared(
            shared,
            NewRequest {
                workstream: "issue-386".to_string(),
                target: "remote::worker".to_string(),
                role: "implementer".to_string(),
                cwd: PathBuf::from("/tmp/issue-386"),
                agent: "missing-agent".to_string(),
                task: None,
            },
        )
        .await
        .expect_err("missing agent should fail before creating a session");

        let message = error.to_string();
        assert!(
            message.contains(
                "agent executable missing-agent not found on remote non-login PATH - pass an absolute path"
            ),
            "unexpected error: {message}"
        );
        let commands = commands.lock().unwrap();
        assert!(commands
            .iter()
            .any(|command| command.contains("command -v 'missing-agent'")));
        assert!(!commands
            .iter()
            .any(|command| command.contains("new-session")));
    }

    #[tokio::test]
    async fn new_session_reports_immediate_agent_exit_after_creation() {
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("command -v 'agent-new'", "found")
            .with_response("list-sessions", "")
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "remote", mock);
        open_test_workstream(&mut state, "issue-386");
        let shared = Arc::new(Mutex::new(state));

        let error = DaemonState::new_session_shared(
            shared,
            NewRequest {
                workstream: "issue-386".to_string(),
                target: "remote::worker".to_string(),
                role: "implementer".to_string(),
                cwd: PathBuf::from("/tmp/issue-386"),
                agent: "agent-new".to_string(),
                task: None,
            },
        )
        .await
        .expect_err("immediate session death should be reported clearly");

        let message = error.to_string();
        assert!(message.contains("exited immediately during startup"));
        assert!(message.contains("agent-new"));
        assert!(
            !message.contains("not found in list"),
            "misleading lower-level error leaked: {message}"
        );
    }

    #[tokio::test]
    async fn new_session_quarantines_reused_session_id_before_adopting() {
        let target = SessionTarget::session_id("local", "$1").expect("target");
        let mock = motlie_tmux::transport::MockTransport::new()
            .with_response("command -v 'agent-new'", "found")
            .with_response("list-sessions", "__MOTLIE_S__ fresh $1 200 0 1  250\n")
            .with_shell_data(vec![b"%output %5 ready\n".to_vec()])
            .with_default("");
        let mut state = DaemonState::default();
        register_mock_host(&mut state, "local", mock);
        seed_stale_reused_target(&mut state, "issue-355-old", &target);
        open_test_workstream(&mut state, "issue-355-new");
        let cwd = PathBuf::from("/tmp/issue-355");
        let shared = Arc::new(Mutex::new(state));

        let records = DaemonState::new_session_shared(
            Arc::clone(&shared),
            NewRequest {
                workstream: "issue-355-new".to_string(),
                target: "local::fresh".to_string(),
                role: "implementer".to_string(),
                cwd: cwd.clone(),
                agent: "agent-new".to_string(),
                task: None,
            },
        )
        .await
        .expect("adopt newly created reused id as fresh session");

        assert_eq!(records[0]["op"], "new");
        assert_eq!(records[0]["target"], target.to_string());
        let state = shared.lock().await;
        assert_stale_target_quarantined(&state, "issue-355-old", &target);
        assert!(state
            .workstream("issue-355-new")
            .expect("new workstream")
            .sessions
            .contains(&target));
        let record = state.sessions.get(&target).expect("fresh record");
        assert_eq!(record.role.as_deref(), Some("implementer"));
        assert_eq!(record.agent.as_deref(), Some("agent-new"));
        assert_eq!(record.cwd.as_ref(), Some(&cwd));
        assert_eq!(record.workstream.as_deref(), Some("issue-355-new"));
        assert_eq!(record.identity, "fresh");
        assert_eq!(record.tmux_session_created, Some(200));
    }

    #[test]
    fn mmux_label_requests_round_trip_over_json() {
        let open = OpenRequest {
            workstream: "issue-349".to_string(),
            title: "Issue 349".to_string(),
            goal: None,
            domain: None,
            mmux_label: Some("349 labels".to_string()),
            settings: WorkstreamSettings { event_limit: 10 },
        };
        let decoded: OpenRequest =
            serde_json::from_value(serde_json::to_value(&open).expect("serialize open"))
                .expect("deserialize open");
        assert_eq!(decoded.mmux_label.as_deref(), Some("349 labels"));

        let label = LabelRequest {
            workstream: "issue-349".to_string(),
            mmux_label: "349 review".to_string(),
        };
        let decoded: LabelRequest =
            serde_json::from_value(serde_json::to_value(&label).expect("serialize label"))
                .expect("deserialize label");
        assert_eq!(decoded.workstream, "issue-349");
        assert_eq!(decoded.mmux_label, "349 review");
    }

    #[test]
    fn session_retag_request_round_trips_over_json() {
        let request = SessionRetagRequest {
            target: "local::$1".to_string(),
            new_name: Some("renamed".to_string()),
            role: Some("reviewer".to_string()),
            workstream: Some("issue-360".to_string()),
            mmux_label: Some("360 review".to_string()),
        };
        let decoded: SessionRetagRequest =
            serde_json::from_value(serde_json::to_value(&request).expect("serialize retag"))
                .expect("deserialize retag");

        assert_eq!(decoded.target, "local::$1");
        assert_eq!(decoded.new_name.as_deref(), Some("renamed"));
        assert_eq!(decoded.role.as_deref(), Some("reviewer"));
        assert_eq!(decoded.workstream.as_deref(), Some("issue-360"));
        assert_eq!(decoded.mmux_label.as_deref(), Some("360 review"));
    }

    #[test]
    fn events_cursor_advances_to_last_returned_event() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "pr-324".to_string(),
                title: "PR 324".to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");
        state
            .record_event("pr-324", EventDraft::new("first"))
            .expect("record first");
        state
            .record_event("pr-324", EventDraft::new("second"))
            .expect("record second");

        let first_page = state
            .events(EventsRequest {
                workstream: "pr-324".to_string(),
                after: None,
                limit: 1,
                readable: false,
            })
            .expect("first page");
        let cursor = first_page["cursor"].as_str().expect("cursor");
        let decoded = PublicCursor::decode(cursor).expect("decode cursor");
        assert_eq!(decoded.next_sequence, 2);

        let second_page = state
            .events(EventsRequest {
                workstream: "pr-324".to_string(),
                after: Some(cursor.to_string()),
                limit: 1,
                readable: false,
            })
            .expect("second page");
        assert_eq!(second_page["events"][0]["kind"], "second");
    }

    #[test]
    fn workstream_event_limit_prunes_old_events() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "pr-324".to_string(),
                title: "PR 324".to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 2 },
            })
            .expect("open workstream");

        state
            .record_event("pr-324", EventDraft::new("first"))
            .expect("record first");
        state
            .record_event("pr-324", EventDraft::new("second"))
            .expect("record second");
        state
            .record_event("pr-324", EventDraft::new("third"))
            .expect("record third");

        let events = state
            .events(EventsRequest {
                workstream: "pr-324".to_string(),
                after: None,
                limit: 0,
                readable: false,
            })
            .expect("read events");

        assert_eq!(events["events"].as_array().expect("events").len(), 2);
        assert_eq!(events["events"][0]["kind"], "second");
        assert_eq!(events["events"][1]["kind"], "third");
    }

    #[test]
    fn events_readable_renders_compact_timeline() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "issue-342".to_string(),
                title: "Issue 342".to_string(),
                goal: None,
                domain: None,
                mmux_label: None,
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");
        state
            .record_event(
                "issue-342",
                EventDraft::new("message_sent")
                    .target("amd2::gpt55-342-rv")
                    .text("Please merge PR #346 now."),
            )
            .expect("record event");

        let events = state
            .events(EventsRequest {
                workstream: "issue-342".to_string(),
                after: None,
                limit: 10,
                readable: true,
            })
            .expect("read events");

        assert_eq!(events["type"], "events_readable");
        let text = events["text"].as_str().expect("readable text");
        assert!(text.contains("message_sent"));
        assert!(text.contains("amd2::gpt55-342-rv"));
        assert!(text.contains("Please merge PR #346 now."));
    }

    #[test]
    fn events_readable_renders_mmux_label_events() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "issue-349".to_string(),
                title: "Issue 349".to_string(),
                goal: None,
                domain: None,
                mmux_label: Some("349 labels".to_string()),
                settings: WorkstreamSettings { event_limit: 10 },
            })
            .expect("open workstream");
        state
            .record_event(
                "issue-349",
                EventDraft::new("mmux_label_applied")
                    .target("amd2::opus47-349-rv")
                    .summary("mmux label applied: 349 labels"),
            )
            .expect("record apply");
        state
            .record_event(
                "issue-349",
                EventDraft::new("mmux_label_cleared")
                    .target("amd2::opus47-349-rv")
                    .summary("mmux workstream label cleared: 349 labels"),
            )
            .expect("record clear");

        let events = state
            .events(EventsRequest {
                workstream: "issue-349".to_string(),
                after: None,
                limit: 10,
                readable: true,
            })
            .expect("read events");
        let text = events["text"].as_str().expect("readable text");
        assert!(text.contains("mmux_label_applied"));
        assert!(text.contains("mmux_label_cleared"));
        assert!(text.contains("amd2::opus47-349-rv"));
        assert!(text.contains("349 labels"));
    }

    #[test]
    fn observe_activity_tracks_local_idle_since_last_change() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let mut record = SessionRecord::from_target(&target, AgentState::Busy, None);
        let first = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        let later = first + chrono::Duration::seconds(90);
        let changed = later + chrono::Duration::seconds(5);

        assert_eq!(record.observe_activity(10, first), 0);
        assert_eq!(record.observe_activity(10, later), 90);
        assert_eq!(record.observe_activity(11, changed), 0);
    }

    #[test]
    fn activity_hint_uses_active_and_idle_thresholds() {
        let options = StatusActivityOptions {
            active_window_secs: 30,
            idle_after_secs: 300,
        };

        assert_eq!(activity_hint(Some(10), options), "active");
        assert_eq!(activity_hint(Some(120), options), "quiet");
        assert_eq!(activity_hint(Some(900), options), "idle");
        assert_eq!(activity_hint(None, options), "unknown");
    }

    #[test]
    fn status_json_reports_live_tmux_activity() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let mut record = SessionRecord::from_target(&target, AgentState::Busy, None);
        let live = LiveActivity::Present(SessionInfo {
            name: "worker".to_string(),
            id: "$1".try_into().expect("session id"),
            created: 900,
            attached_count: 0,
            window_count: 1,
            group: None,
            activity: 970,
        });
        let now = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        let status = record.status_json(
            &target,
            Some(&live),
            StatusActivityOptions {
                active_window_secs: 30,
                idle_after_secs: 300,
            },
            now,
        );

        assert_eq!(status["tmux_present"], true);
        assert_eq!(status["tmux_session_id"], "$1");
        assert_eq!(status["last_output_secs"], 30);
        assert_eq!(status["observed_activity_idle_secs"], 0);
        assert_eq!(status["activity_hint"], "active");
    }

    #[test]
    fn session_environment_is_context_only() {
        let vars = session_environment("pr-324", "reviewer").expect("environment");
        let names = vars.iter().map(SessionEnvVar::name).collect::<Vec<_>>();

        assert_eq!(names, vec!["MSTREAM_WORKSTREAM", "MSTREAM_ROLE"]);
        assert!(!names.contains(&"MSTREAM_TARGET"));
        assert!(!names.contains(&"MSTREAM_SOCKET"));
    }

    #[test]
    fn timer_record_json_reports_input_guard_deferrals() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let started_at = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        let record = TimerRecord {
            name: "poll".to_string(),
            workstream: Some("pr-324".to_string()),
            target,
            tmux_session_created: None,
            every_secs: 60,
            prompt: "Wake up.".to_string(),
            enter: true,
            submit_retries: 1,
            submit_retry_delay_ms: 750,
            input_quiet_for_secs: Some(10),
            generation: 3,
            started_at,
            next_fire_at: add_secs(started_at, 7),
            last_fired_at: None,
            fire_count: 2,
            defer_count: 1,
            last_deferred_at: Some(started_at),
            last_defer_reason: Some("recent_client_input".to_string()),
            last_input_activity: Some(995),
            last_error: None,
            task: None,
        };

        let value = record.to_json();

        assert_eq!(value["input_quiet_for_secs"], 10);
        assert_eq!(value["workstream"], "pr-324");
        assert_eq!(value["fire_count"], 2);
        assert_eq!(value["defer_count"], 1);
        assert_eq!(value["last_defer_reason"], "recent_client_input");
        assert_eq!(value["last_input_activity"], 995);
        assert_eq!(value["last_input_activity_at"], "1970-01-01T00:16:35Z");
    }

    #[test]
    fn timer_list_filters_by_workstream() {
        let mut state = DaemonState::default();
        let started_at = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        for (name, workstream) in [
            ("issue-342-poll", Some("issue-342")),
            ("issue-337-poll", Some("issue-337")),
            ("global", None),
        ] {
            state.timers.insert(
                name.to_string(),
                TimerRecord {
                    name: name.to_string(),
                    workstream: workstream.map(ToString::to_string),
                    target: "local::worker".parse().expect("target"),
                    tmux_session_created: None,
                    every_secs: 60,
                    prompt: "Wake up.".to_string(),
                    enter: true,
                    submit_retries: 1,
                    submit_retry_delay_ms: 750,
                    input_quiet_for_secs: Some(10),
                    generation: 1,
                    started_at,
                    next_fire_at: Some(started_at),
                    last_fired_at: None,
                    fire_count: 0,
                    defer_count: 0,
                    last_deferred_at: None,
                    last_defer_reason: None,
                    last_input_activity: None,
                    last_error: None,
                    task: None,
                },
            );
        }

        let value = state.timer_list_json(Some("issue-342"));

        let timers = value["timers"].as_array().expect("timers");
        assert_eq!(timers.len(), 1);
        assert_eq!(timers[0]["name"], "issue-342-poll");
    }

    #[test]
    fn stop_timers_for_workstream_removes_only_scoped_timers() {
        let mut state = DaemonState::default();
        let started_at = DateTime::<Utc>::from_timestamp(1_000, 0).expect("timestamp");
        for (name, workstream) in [
            ("x-timer", Some("issue-x")),
            ("y-timer", Some("issue-y")),
            ("global-timer", None),
        ] {
            state.timers.insert(
                name.to_string(),
                TimerRecord {
                    name: name.to_string(),
                    workstream: workstream.map(ToString::to_string),
                    target: "local::worker".parse().expect("target"),
                    tmux_session_created: None,
                    every_secs: 60,
                    prompt: "Wake up.".to_string(),
                    enter: true,
                    submit_retries: 1,
                    submit_retry_delay_ms: 750,
                    input_quiet_for_secs: Some(10),
                    generation: 1,
                    started_at,
                    next_fire_at: Some(started_at),
                    last_fired_at: None,
                    fire_count: 0,
                    defer_count: 0,
                    last_deferred_at: None,
                    last_defer_reason: None,
                    last_input_activity: None,
                    last_error: None,
                    task: None,
                },
            );
        }

        let stopped = state.stop_timers_for_workstream("issue-x");

        assert_eq!(stopped, vec!["x-timer"]);
        assert!(!state.timers.contains_key("x-timer"));
        assert!(state.timers.contains_key("y-timer"));
        assert!(state.timers.contains_key("global-timer"));
    }

    #[test]
    fn timer_deferred_json_reports_guard_decision() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let next_fire_at = DateTime::<Utc>::from_timestamp(1_007, 0).expect("timestamp");
        let value = timer_fire_outcome_json(TimerFireOutcome::Deferred(TimerDeferSnapshot {
            name: "poll".to_string(),
            target,
            generation: 3,
            reason: "recent_client_input",
            latest_client_activity: Some(995),
            latest_client_activity_age_secs: Some(5),
            quiet_for_secs: 10,
            next_fire_at: Some(next_fire_at),
        }));

        assert_eq!(value["op"], "timer_deferred");
        assert_eq!(value["reason"], "recent_client_input");
        assert_eq!(value["latest_client_activity"], 995);
        assert_eq!(value["latest_client_activity_age_secs"], 5);
        assert_eq!(value["quiet_for_secs"], 10);
        assert_eq!(value["next_fire_at"], "1970-01-01T00:16:47Z");
    }

    #[test]
    fn managed_prompt_uses_normal_output_reporting() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let prompt = managed_prompt(
            "pr-324",
            &target,
            Some("reviewer"),
            None,
            Some("Review the PR."),
        );

        assert!(prompt.contains("Report progress, blockers, questions"));
        assert!(prompt.contains("The orchestrator owns workstream state"));
        assert!(!prompt.contains("mstream session mark"));
        assert!(!prompt.contains("MSTREAM_SOCKET"));
    }
}
