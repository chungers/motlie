use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};
use chrono::{DateTime, Utc};
use motlie_tmux::{
    CreateSessionOptions, HostHandle, KeySequence, SessionEnvVar, SessionInfo, SessionTag,
    SshConfig, Target,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::jsonl;
use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LeaveRequest, NewRequest,
    OpenRequest, PasteMode, RecruitRequest, SendRequest, SessionMarkRequest, SnapshotRequest,
    SummaryInputRequest, TimerStartRequest, WorkstreamSettings,
};
use crate::tags;
use crate::target::SessionTarget;
use crate::timeline::PublicCursor;

pub struct DaemonState {
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
    handle: HostHandle,
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
    last_tmux_activity: Option<u64>,
    activity_observed_at: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct WorkstreamRecord {
    title: String,
    goal: Option<String>,
    domain: Option<String>,
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
    target: SessionTarget,
    every_secs: u64,
    prompt: String,
    enter: bool,
    submit_retries: u8,
    submit_retry_delay_ms: u64,
    generation: u64,
    started_at: DateTime<Utc>,
    next_fire_at: Option<DateTime<Utc>>,
    last_fired_at: Option<DateTime<Utc>>,
    fire_count: u64,
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
    on: AgentState,
    task: String,
    only_on_transition: bool,
    fired: bool,
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
    session_target: &'a SessionTarget,
    role: &'a str,
    agent: Option<&'a str>,
    cwd: Option<&'a Path>,
    state: AgentState,
}

#[derive(Debug, Clone)]
struct WorkstreamMeta {
    title: String,
    goal: Option<String>,
    domain: Option<String>,
}

struct ResolvedTarget {
    logical: SessionTarget,
    handle: HostHandle,
    target: Target,
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
    handle: HostHandle,
    prompt: String,
    enter: bool,
    submit_retries: u8,
    submit_retry_delay_ms: u64,
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
            ClientRequest::Kill { target } => Self::kill_shared(shared, target).await,
            ClientRequest::Send(request) => Self::send_shared(shared, request).await,
            ClientRequest::Interrupt(request) => Self::interrupt_shared(shared, request).await,
            ClientRequest::Broadcast(request) => Self::broadcast_shared(shared, request).await,
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
            ClientRequest::TimerList => Ok(vec![shared.lock().await.timer_list_json()]),
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
            .connect()
            .await
            .with_context(|| format!("failed to connect host '{}'", request.alias))?;
        let mut state = shared.lock().await;
        if state.hosts.contains_key(&request.alias) {
            bail!("host alias '{}' is already connected", request.alias);
        }
        state.hosts.insert(
            request.alias.clone(),
            HostRecord {
                uri: request.ssh_uri.clone(),
                labels: request.labels,
                capacity: request.capacity,
                work_root: request.work_root,
                handle,
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
            state.host(&alias)?.handle.clone()
        };
        let sessions = handle.list_sessions().await?;
        let tags_by_session = handle
            .list_tags_for_session_infos(tags::PREFIX, &sessions)
            .await?;

        let mut monitor_sessions = Vec::new();
        let hydrated = {
            let mut state = shared.lock().await;
            let mut hydrated = 0usize;
            for session in sessions {
                let target = SessionTarget::new(&alias, session.name.clone())?;
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
                                WorkstreamSettings::default(),
                                generation,
                            );
                            state
                                .workstreams
                                .insert(workstream_name.clone(), workstream);
                        }
                        let workstream = state.workstream_mut(&workstream_name)?;
                        workstream.sessions.insert(target.clone());
                        monitor_sessions.push(target.session_name().to_string());
                    }
                }
            }
            hydrated
        };

        for session_name in monitor_sessions {
            let _ = handle.start_monitoring_session(&session_name).await;
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
                state.host(target.host_alias())?.handle.clone(),
                state.workstream_meta(&request.workstream)?,
            )
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::write_assignment_to_target(
            &resolved.target,
            &meta,
            AssignmentTags {
                workstream: &request.workstream,
                session_target: &target,
                role: &request.role,
                agent: None,
                cwd: None,
                state: AgentState::Busy,
            },
        )
        .await?;
        resolved
            .handle
            .start_monitoring_session(resolved.logical.session_name())
            .await?;
        if let Some(task) = &request.task {
            Self::send_text_to_resolved(
                &resolved,
                &managed_prompt(
                    &request.workstream,
                    &target,
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
            let mut state = shared.lock().await;
            state.add_session_to_workstream(
                &request.workstream,
                target.clone(),
                request.role,
                None,
                None,
                AgentState::Busy,
            )?;
            state.record_event(
                &request.workstream,
                EventDraft::new("joined")
                    .target(&target)
                    .state(AgentState::Busy),
            )?
        };
        Ok(vec![json!({
            "type": "ok",
            "op": "join",
            "workstream": request.workstream,
            "target": target.to_string(),
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
                state.host(target.host_alias())?.handle.clone(),
                state.workstream_meta(&request.workstream)?,
            )
        };
        let command = bootstrap_command(&request.cwd, &request.agent);
        let env = session_environment(&request.workstream, &target, &request.role)?;
        let opts = CreateSessionOptions {
            command: Some(command),
            initial_environment: env,
            ..Default::default()
        };
        let tmux_target = handle.create_session(target.session_name(), &opts).await?;
        let resolved = ResolvedTarget {
            logical: target.clone(),
            handle,
            target: tmux_target,
        };
        Self::write_assignment_to_target(
            &resolved.target,
            &meta,
            AssignmentTags {
                workstream: &request.workstream,
                session_target: &target,
                role: &request.role,
                agent: Some(&request.agent),
                cwd: Some(&request.cwd),
                state: AgentState::Busy,
            },
        )
        .await?;
        resolved
            .handle
            .start_monitoring_session(resolved.logical.session_name())
            .await?;
        if let Some(task) = &request.task {
            Self::send_text_to_resolved(
                &resolved,
                &managed_prompt(
                    &request.workstream,
                    &target,
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
                target.clone(),
                request.role,
                Some(request.agent),
                Some(request.cwd.clone()),
                AgentState::Busy,
            )?;
            state.record_event(
                &request.workstream,
                EventDraft::new("created")
                    .target(&target)
                    .state(AgentState::Busy),
            )?
        };
        Ok(vec![json!({
            "type": "ok",
            "op": "new",
            "workstream": request.workstream,
            "target": target.to_string(),
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
        for target in &sessions {
            let handle = {
                let state = shared.lock().await;
                state
                    .host(target.host_alias())
                    .map(|host| host.handle.clone())
                    .ok()
            };
            if let Some(handle) = handle {
                if let Ok(resolved) = Self::resolve_target(handle, target.clone()).await {
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
                    tags::set_many(&resolved.target, &pairs).await?;
                    tags::unset_many(
                        &resolved.target,
                        &[
                            "workstream",
                            "workstream-title",
                            "workstream-state",
                            "workstream-goal",
                            "workstream-domain",
                            "role",
                        ],
                    )
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
        let mut records = vec![json!({
            "type": "ok",
            "op": "close",
            "workstream": request.workstream,
            "cursor": cursor,
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
            state.host(target.host_alias())?.handle.clone()
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        tags::unset_many(
            &resolved.target,
            &[
                "workstream",
                "workstream-title",
                "workstream-state",
                "workstream-goal",
                "workstream-domain",
                "role",
            ],
        )
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
            if let Some(record) = state.sessions.get_mut(&target) {
                record.workstream = None;
            }
            let mut event = EventDraft::new("left").target(&target);
            if request.available {
                event = event.state(AgentState::Available);
            }
            let cursor = state.record_event(&request.workstream, event)?;
            let workstream = state.workstream_mut(&request.workstream)?;
            workstream.sessions.remove(&target);
            cursor
        };
        let mut records = vec![json!({
            "type": "ok",
            "op": "leave",
            "workstream": request.workstream,
            "target": target.to_string(),
            "cursor": cursor,
        })];
        if let Some(change) = change {
            records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        }
        Ok(records)
    }

    async fn kill_shared(shared: Arc<Mutex<Self>>, target: String) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = target.parse()?;
        let handle = {
            let state = shared.lock().await;
            state.host(target.host_alias())?.handle.clone()
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        resolved.target.kill().await?;
        let mut state = shared.lock().await;
        state.sessions.remove(&target);
        for workstream in state.workstreams.values_mut() {
            workstream.sessions.remove(&target);
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "kill",
            "target": target.to_string(),
        })])
    }

    async fn send_shared(
        shared: Arc<Mutex<Self>>,
        request: SendRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let (handle, current_state) = {
            let state = shared.lock().await;
            state.ensure_target_in_workstream(&request.workstream, &target)?;
            let current_state = state
                .sessions
                .get(&target)
                .map(|record| record.state)
                .unwrap_or(AgentState::Idle);
            if let Some(required) = request.require_state {
                if current_state != required {
                    bail!(
                        "target {} is {}, not required state {}",
                        target,
                        current_state.as_str(),
                        required.as_str()
                    );
                }
            }
            (
                state.host(target.host_alias())?.handle.clone(),
                current_state,
            )
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
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
                .get(&target)
                .map(|record| record.state)
                .unwrap_or(current_state);
            let cursor = state.record_event(
                &request.workstream,
                EventDraft::new("message_sent")
                    .target(&target)
                    .text(request.text)
                    .state(state_after),
            )?;
            (state_after, cursor)
        };
        let mut records = vec![json!({
            "type": "ok",
            "op": "send",
            "workstream": request.workstream,
            "target": target.to_string(),
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
        let (handle, event_context) = {
            let state = shared.lock().await;
            let handle = state.host(target.host_alias())?.handle.clone();
            let event_context = state.sessions.get(&target).and_then(|record| {
                record
                    .workstream
                    .clone()
                    .map(|workstream| (workstream, record.state))
            });
            (handle, event_context)
        };
        let resolved = Self::resolve_target(handle, target.clone()).await?;
        Self::send_interrupt_to_resolved(&resolved, request.key).await?;
        let mut record = json!({
            "type": "ok",
            "op": "interrupt",
            "target": target.to_string(),
            "key": request.key.as_str(),
        });
        if let Some((workstream, state_value)) = event_context {
            let cursor = shared.lock().await.record_event(
                &workstream,
                EventDraft::new("interrupted")
                    .target(&target)
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
                    .target(&target.target.logical)
                    .text(request.text.clone())
                    .state(target.state),
            )?;
            records.push(json!({
                "type": "ok",
                "op": "broadcast_sent",
                "workstream": request.workstream,
                "target": target.target.logical.to_string(),
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
                .get(&target)
                .and_then(|record| record.workstream.clone())
        };
        if let Some(workstream) = workstream {
            let kind = request.state.event_kind().unwrap_or("session_marked");
            let cursor = shared.lock().await.record_event(
                &workstream,
                EventDraft::new(kind)
                    .target(&target)
                    .state(request.state)
                    .summary(request.summary.clone()),
            )?;
            records.push(json!({
                "type": "event",
                "kind": kind,
                "workstream": workstream,
                "target": target.to_string(),
                "state": request.state.as_str(),
                "summary": request.summary,
                "cursor": cursor,
            }));
        } else {
            records.push(json!({
                "type": "ok",
                "op": "session_mark",
                "target": target.to_string(),
                "state": request.state.as_str(),
            }));
        }
        records.extend(Self::fire_handoffs_for_changes_shared(shared, vec![change]).await?);
        Ok(records)
    }

    async fn handoff_arm_shared(
        shared: Arc<Mutex<Self>>,
        request: HandoffArmRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let from: SessionTarget = request.from.parse()?;
        let to: SessionTarget = request.to.parse()?;
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
                on: request.on,
                task: request.task,
                only_on_transition: request.only_on_transition,
                fired: false,
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
            state.host(target.host_alias())?.handle.clone()
        };
        Self::resolve_target(handle, target.clone()).await?;

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
                    target: target.clone(),
                    every_secs: request.every_secs,
                    prompt: request.prompt,
                    enter: request.enter,
                    submit_retries,
                    submit_retry_delay_ms: request.submit_retry_delay_ms,
                    generation,
                    started_at,
                    next_fire_at,
                    last_fired_at: None,
                    fire_count: 0,
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
            "target": target.to_string(),
            "every_secs": request.every_secs,
            "enter": request.enter,
            "submit_retries": submit_retries,
            "submit_retry_delay_ms": request.submit_retry_delay_ms,
            "generation": generation,
            "next_fire_at": datetime_option_json(next_fire_at),
        })])
    }

    async fn timer_fire_shared(
        shared: Arc<Mutex<Self>>,
        name: String,
    ) -> anyhow::Result<Vec<Value>> {
        let snapshot =
            Self::timer_fire_once_shared(Arc::clone(&shared), &name, None, false).await?;
        Ok(vec![json!({
            "type": "ok",
            "op": "timer_fire",
            "name": snapshot.name,
            "target": snapshot.target.to_string(),
            "generation": snapshot.generation,
        })])
    }

    async fn timer_loop(shared: Arc<Mutex<Self>>, name: String, generation: u64) {
        loop {
            let every_secs = {
                let state = shared.lock().await;
                let Some(timer) = state.timers.get(&name) else {
                    break;
                };
                if timer.generation != generation {
                    break;
                }
                timer.every_secs
            };
            sleep(Duration::from_secs(every_secs)).await;
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
    ) -> anyhow::Result<TimerFireSnapshot> {
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
                handle: state.host(timer.target.host_alias())?.handle.clone(),
                prompt: timer.prompt.clone(),
                enter: timer.enter,
                submit_retries: timer.submit_retries,
                submit_retry_delay_ms: timer.submit_retry_delay_ms,
                generation: timer.generation,
            }
        };

        let result = async {
            let resolved =
                Self::resolve_target(snapshot.handle.clone(), snapshot.target.clone()).await?;
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
        Ok(snapshot)
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
                .cloned()
                .collect::<Vec<_>>();
            let mut hosts = BTreeMap::new();
            for target in &targets {
                if !hosts.contains_key(target.host_alias()) {
                    hosts.insert(
                        target.host_alias().to_string(),
                        state.host(target.host_alias())?.handle.clone(),
                    );
                }
            }
            (targets, hosts)
        };

        let mut live = BTreeMap::new();
        for (alias, handle) in hosts {
            let host_targets = targets
                .iter()
                .filter(|target| target.host_alias() == alias)
                .cloned()
                .collect::<Vec<_>>();
            match handle.list_sessions().await {
                Ok(sessions) => {
                    let mut by_name = sessions
                        .into_iter()
                        .map(|session| (session.name.clone(), session))
                        .collect::<BTreeMap<_, _>>();
                    for target in host_targets {
                        let activity = by_name
                            .remove(target.session_name())
                            .map(LiveActivity::Present)
                            .unwrap_or(LiveActivity::Missing);
                        live.insert(target, activity);
                    }
                }
                Err(err) => {
                    let message = err.to_string();
                    for target in host_targets {
                        live.insert(target, LiveActivity::Error(message.clone()));
                    }
                }
            }
        }

        let now = Utc::now();
        Ok(vec![shared.lock().await.status(
            &workstream,
            &live,
            options,
            now,
        )?])
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
                    session_target: &plan.target.logical,
                    role: &request.role,
                    agent: plan.agent.as_deref(),
                    cwd: plan.cwd.as_deref(),
                    state: plan.state,
                },
            )
            .await?;
            plan.target
                .handle
                .start_monitoring_session(plan.target.logical.session_name())
                .await?;
            if let Some(task) = &request.task {
                Self::send_text_to_resolved(
                    &plan.target,
                    &managed_prompt(
                        &request.workstream,
                        &plan.target.logical,
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
                    plan.target.logical.clone(),
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
                    .target(&plan.target.logical)
                    .state(plan.state);
                if let Some(goal) = request.goal.clone() {
                    event = event.text(goal);
                }
                state.record_event(&request.workstream, event)?
            };
            records.push(json!({
                "type": "ok",
                "op": "recruited",
                "workstream": request.workstream,
                "target": plan.target.logical.to_string(),
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

    fn timer_list_json(&self) -> Value {
        let timers: Vec<Value> = self.timers.values().map(TimerRecord::to_json).collect();
        json!({ "type": "timers", "timers": timers })
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
            "target": timer.target.to_string(),
            "generation": timer.generation,
            "fire_count": timer.fire_count,
        })])
    }

    fn status(
        &mut self,
        workstream: &str,
        live: &BTreeMap<SessionTarget, LiveActivity>,
        activity_options: StatusActivityOptions,
        now: DateTime<Utc>,
    ) -> anyhow::Result<Value> {
        let (workstream_state, generation, settings, cursor, targets) = {
            let workstream_record = self.workstream(workstream)?;
            (
                workstream_record.state.as_str(),
                workstream_record.generation,
                workstream_record.settings,
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
                        .hosts
                        .get(target.host_alias())
                        .map(|host| (target.clone(), host.handle.clone()))
                })
                .collect::<Vec<_>>()
        };
        let mut text = String::new();
        for (logical, handle) in targets {
            let capture = match Self::resolve_target(handle, logical.clone()).await {
                Ok(target_handle) => target_handle.target.capture().await.unwrap_or_default(),
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
            state.host(handoff.to.host_alias())?.handle.clone()
        };
        let resolved = Self::resolve_target(handle, handoff.to.clone()).await?;
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
        for (workstream_name, workstream) in &mut self.workstreams {
            for handoff in workstream.handoffs.values_mut() {
                if !handoff.fired
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
            state_guard.host(target.host_alias())?.handle.clone()
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
        let now = tags::now_tag();
        let mut pairs = vec![
            ("state", tags::state_value(state)),
            ("updated-at", now.clone()),
        ];
        if let Some(summary) = summary {
            pairs.push(("last-report-summary", summary.to_string()));
            pairs.push(("last-report-kind", state.as_str().to_string()));
        }
        tags::set_many(&resolved.target, &pairs).await?;
        let mut state_guard = shared.lock().await;
        let previous_state = state_guard
            .sessions
            .get(&resolved.logical)
            .map(|record| record.state);
        let record = state_guard
            .sessions
            .entry(resolved.logical.clone())
            .or_insert_with(|| {
                SessionRecord::from_target(&resolved.logical, state, Some(Utc::now()))
            });
        record.state = state;
        record.updated_at = Utc::now();
        if let Some(summary) = summary {
            record.last_report_kind = Some(state.as_str().to_string());
            record.last_report_summary = Some(summary.to_string());
        }
        Ok(StateChange {
            target: resolved.logical.clone(),
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

    async fn resolve_target(
        handle: HostHandle,
        logical: SessionTarget,
    ) -> anyhow::Result<ResolvedTarget> {
        let target = handle
            .target(&logical.target_spec())
            .await?
            .with_context(|| format!("session '{}' not found", logical))?;
        Ok(ResolvedTarget {
            logical,
            handle,
            target,
        })
    }

    async fn touch_resolved_session_shared(
        shared: Arc<Mutex<Self>>,
        target: &ResolvedTarget,
    ) -> anyhow::Result<()> {
        let now = tags::now_tag();
        tags::set_many(&target.target, &[("updated-at", now)]).await?;
        if let Some(record) = shared.lock().await.sessions.get_mut(&target.logical) {
            record.updated_at = Utc::now();
        }
        Ok(())
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
            (
                "identity",
                assignment.session_target.session_name().to_string(),
            ),
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
        tags::set_many(target, &pairs).await
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
        record.updated_at = Utc::now();
        Ok(())
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

    fn host(&self, alias: &str) -> anyhow::Result<&HostRecord> {
        self.hosts
            .get(alias)
            .with_context(|| format!("host alias '{alias}' is not connected"))
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

    fn workstream_meta(&self, name: &str) -> anyhow::Result<WorkstreamMeta> {
        let workstream = self.workstream(name)?;
        Ok(WorkstreamMeta {
            title: workstream.title.clone(),
            goal: workstream.goal.clone(),
            domain: workstream.domain.clone(),
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
            selected.push(BroadcastTarget {
                target: Self::resolve_target(snapshot.handle, snapshot.target).await?,
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
                handle: self.host(target.host_alias())?.handle.clone(),
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
            plans.push(RecruitPlan {
                target: Self::resolve_target(snapshot.handle, snapshot.target).await?,
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
                    handle: self.host(target.host_alias())?.handle.clone(),
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
            identity: target.session_name().to_string(),
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
            last_tmux_activity: None,
            activity_observed_at: None,
        }
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
            "tmux_activity": self.last_tmux_activity,
            "activity_observed_at": datetime_option_json(self.activity_observed_at),
        })
    }
}

impl TimerRecord {
    fn to_json(&self) -> Value {
        json!({
            "name": &self.name,
            "target": self.target.to_string(),
            "every_secs": self.every_secs,
            "enter": self.enter,
            "submit_retries": self.submit_retries,
            "submit_retry_delay_ms": self.submit_retry_delay_ms,
            "generation": self.generation,
            "started_at": self.started_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "next_fire_at": datetime_option_json(self.next_fire_at),
            "last_fired_at": datetime_option_json(self.last_fired_at),
            "fire_count": self.fire_count,
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
        settings: WorkstreamSettings,
        generation: u64,
    ) -> Self {
        Self {
            title,
            goal,
            domain,
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
        self.events.clear();
        self.handoffs.clear();
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

fn bootstrap_command(cwd: &Path, agent: &str) -> String {
    format!(
        "mkdir -p {} && cd {} && exec {}",
        shell_quote(&cwd.display().to_string()),
        shell_quote(&cwd.display().to_string()),
        shell_quote(agent)
    )
}

fn session_environment(
    workstream: &str,
    target: &SessionTarget,
    role: &str,
) -> anyhow::Result<Vec<SessionEnvVar>> {
    Ok(vec![
        SessionEnvVar::new("MSTREAM_WORKSTREAM", workstream)?,
        SessionEnvVar::new("MSTREAM_TARGET", target.to_string())?,
        SessionEnvVar::new("MSTREAM_ROLE", role)?,
    ])
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
    fn events_cursor_advances_to_last_returned_event() {
        let mut state = DaemonState::default();
        state
            .open(OpenRequest {
                workstream: "pr-324".to_string(),
                title: "PR 324".to_string(),
                goal: None,
                domain: None,
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
            })
            .expect("read events");

        assert_eq!(events["events"].as_array().expect("events").len(), 2);
        assert_eq!(events["events"][0]["kind"], "second");
        assert_eq!(events["events"][1]["kind"], "third");
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
    fn session_environment_does_not_include_orchestrator_socket() {
        let target: SessionTarget = "local::worker".parse().expect("target");
        let vars = session_environment("pr-324", &target, "reviewer").expect("environment");
        let names = vars.iter().map(SessionEnvVar::name).collect::<Vec<_>>();

        assert_eq!(
            names,
            vec!["MSTREAM_WORKSTREAM", "MSTREAM_TARGET", "MSTREAM_ROLE"]
        );
        assert!(!names.contains(&"MSTREAM_SOCKET"));
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
