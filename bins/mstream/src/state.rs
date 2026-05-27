use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};
use chrono::{DateTime, Utc};
use motlie_tmux::{
    CreateSessionOptions, HostHandle, KeySequence, SessionEnvVar, SessionTag, SshConfig, Target,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::jsonl;
use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LeaveRequest, NewRequest,
    OpenRequest, PasteMode, RecruitRequest, SendRequest, SessionMarkRequest, SnapshotRequest,
    SummaryInputRequest, WorkstreamSettings,
};
use crate::tags;
use crate::target::SessionTarget;
use crate::timeline::PublicCursor;

pub struct DaemonState {
    hosts: BTreeMap<String, HostRecord>,
    sessions: BTreeMap<SessionTarget, SessionRecord>,
    workstreams: BTreeMap<String, WorkstreamRecord>,
    next_generation: u64,
    next_handoff_id: u64,
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            hosts: BTreeMap::new(),
            sessions: BTreeMap::new(),
            workstreams: BTreeMap::new(),
            next_generation: 1,
            next_handoff_id: 1,
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

#[derive(Clone)]
struct TargetHandle {
    target: SessionTarget,
    handle: HostHandle,
}

#[derive(Clone)]
struct BroadcastTarget {
    target: SessionTarget,
    handle: HostHandle,
    state: AgentState,
}

#[derive(Clone)]
struct RecruitPlan {
    target: SessionTarget,
    handle: HostHandle,
    agent: Option<String>,
    cwd: Option<PathBuf>,
    state: AgentState,
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
            ClientRequest::Status { workstream } => {
                Ok(vec![shared.lock().await.status(&workstream)?])
            }
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
        let tmux_target = Self::tmux_target_from_handle(&handle, &target).await?;
        Self::write_assignment_to_target(
            &tmux_target,
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
        handle
            .start_monitoring_session(target.session_name())
            .await?;
        if let Some(task) = &request.task {
            Self::send_text_with_handle(
                &handle,
                &target,
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
        let socket_hint = std::env::var("MSTREAM_SOCKET").ok();
        let command = bootstrap_command(&request.cwd, &request.agent);
        let env = session_environment(
            socket_hint.as_deref(),
            &request.workstream,
            &target,
            &request.role,
        )?;
        let opts = CreateSessionOptions {
            command: Some(command),
            initial_environment: env,
            ..Default::default()
        };
        let tmux_target = handle.create_session(target.session_name(), &opts).await?;
        Self::write_assignment_to_target(
            &tmux_target,
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
        handle
            .start_monitoring_session(target.session_name())
            .await?;
        if let Some(task) = &request.task {
            Self::send_text_with_handle(
                &handle,
                &target,
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
                if let Ok(tmux_target) = Self::tmux_target_from_handle(&handle, target).await {
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
                    tags::set_many(&tmux_target, &pairs).await?;
                    tags::unset_many(
                        &tmux_target,
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
        let tmux_target = Self::tmux_target_from_handle(&handle, &target).await?;
        tags::unset_many(
            &tmux_target,
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
                Self::apply_session_state_shared(
                    Arc::clone(&shared),
                    &target,
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
        let tmux_target = Self::tmux_target_from_handle(&handle, &target).await?;
        tmux_target.kill().await?;
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
        if request.interrupt_first {
            Self::send_interrupt_with_handle(&handle, &target, InterruptKey::Esc).await?;
            sleep(Duration::from_millis(request.settle_ms)).await;
        }
        Self::send_text_with_handle(
            &handle,
            &target,
            &request.text,
            request.paste_mode,
            request.enter,
        )
        .await?;
        let change = if let Some(state) = request.set_state {
            Some(Self::apply_session_state_shared(Arc::clone(&shared), &target, state, None).await?)
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
        Self::send_interrupt_with_handle(&handle, &target, request.key).await?;
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
        let targets = {
            let state = shared.lock().await;
            state.broadcast_targets(&request)?
        };
        let mut records = Vec::new();
        let mut sent = 0usize;
        for target in targets {
            Self::send_text_with_handle(
                &target.handle,
                &target.target,
                &request.text,
                request.paste_mode,
                request.enter,
            )
            .await?;
            Self::touch_session_shared(Arc::clone(&shared), &target.target).await?;
            sent += 1;
            let cursor = shared.lock().await.record_event(
                &request.workstream,
                EventDraft::new("broadcast_sent")
                    .target(&target.target)
                    .text(request.text.clone())
                    .state(target.state),
            )?;
            records.push(json!({
                "type": "ok",
                "op": "broadcast_sent",
                "workstream": request.workstream,
                "target": target.target.to_string(),
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

    async fn recruit_shared(
        shared: Arc<Mutex<Self>>,
        request: RecruitRequest,
    ) -> anyhow::Result<Vec<Value>> {
        let plans = {
            let state = shared.lock().await;
            state.recruit_plans(&request)?
        };
        let mut records = Vec::new();
        let mut changes = Vec::new();
        for plan in plans {
            let meta = {
                let state = shared.lock().await;
                state.workstream_meta(&request.workstream)?
            };
            let tmux_target = Self::tmux_target_from_handle(&plan.handle, &plan.target).await?;
            Self::write_assignment_to_target(
                &tmux_target,
                &meta,
                AssignmentTags {
                    workstream: &request.workstream,
                    session_target: &plan.target,
                    role: &request.role,
                    agent: plan.agent.as_deref(),
                    cwd: plan.cwd.as_deref(),
                    state: plan.state,
                },
            )
            .await?;
            plan.handle
                .start_monitoring_session(plan.target.session_name())
                .await?;
            if let Some(task) = &request.task {
                Self::send_text_with_handle(
                    &plan.handle,
                    &plan.target,
                    &managed_prompt(
                        &request.workstream,
                        &plan.target,
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
                    plan.target.clone(),
                    request.role.clone(),
                    plan.agent.clone(),
                    plan.cwd.clone(),
                    plan.state,
                )?;
            }
            changes.push(
                Self::apply_session_state_shared(
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
                    .target(&plan.target)
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
                "target": plan.target.to_string(),
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

    fn status(&self, workstream: &str) -> anyhow::Result<Value> {
        let workstream_record = self.workstream(workstream)?;
        let agents: Vec<Value> = workstream_record
            .sessions
            .iter()
            .filter_map(|target| {
                self.sessions.get(target).map(|session| {
                    json!({
                        "target": target.to_string(),
                        "role": session.role,
                        "agent": session.agent,
                        "state": session.state.as_str(),
                        "last_report_kind": session.last_report_kind,
                        "last_report_summary": session.last_report_summary,
                        "updated_at": session.updated_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                        "stuck_hint": Value::Null,
                    })
                })
            })
            .collect();
        Ok(json!({
            "type": "status",
            "workstream": workstream,
            "state": workstream_record.state.as_str(),
            "generation": workstream_record.generation,
            "settings": workstream_record.settings,
            "cursor": workstream_record.cursor(workstream)?,
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
                        .map(|host| TargetHandle {
                            target: target.clone(),
                            handle: host.handle.clone(),
                        })
                })
                .collect::<Vec<_>>()
        };
        let mut text = String::new();
        for target in targets {
            let capture = match Self::tmux_target_from_handle(&target.handle, &target.target).await
            {
                Ok(target_handle) => target_handle.capture().await.unwrap_or_default(),
                Err(_) => String::new(),
            };
            text.push_str(&format!("=== {} ===\n", target.target));
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
        let change = Self::apply_session_state_shared(
            Arc::clone(&shared),
            &handoff.to,
            AgentState::Busy,
            None,
        )
        .await?;
        let handle = {
            let state = shared.lock().await;
            state.host(handoff.to.host_alias())?.handle.clone()
        };
        Self::send_text_with_handle(
            &handle,
            &handoff.to,
            &handoff.task,
            PasteMode::Bracketed,
            true,
        )
        .await?;
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
        let tmux_target = Self::tmux_target_from_handle(&handle, target).await?;
        let now = tags::now_tag();
        let mut pairs = vec![
            ("state", tags::state_value(state)),
            ("updated-at", now.clone()),
        ];
        if let Some(summary) = summary {
            pairs.push(("last-report-summary", summary.to_string()));
            pairs.push(("last-report-kind", state.as_str().to_string()));
        }
        tags::set_many(&tmux_target, &pairs).await?;
        let mut state_guard = shared.lock().await;
        let previous_state = state_guard.sessions.get(target).map(|record| record.state);
        let record = state_guard
            .sessions
            .entry(target.clone())
            .or_insert_with(|| SessionRecord::from_target(target, state, Some(Utc::now())));
        record.state = state;
        record.updated_at = Utc::now();
        if let Some(summary) = summary {
            record.last_report_kind = Some(state.as_str().to_string());
            record.last_report_summary = Some(summary.to_string());
        }
        Ok(StateChange {
            target: target.clone(),
            state,
            previous_state,
        })
    }

    async fn send_interrupt_with_handle(
        handle: &HostHandle,
        target: &SessionTarget,
        key: InterruptKey,
    ) -> anyhow::Result<()> {
        let target = Self::tmux_target_from_handle(handle, target).await?;
        let sequence = match key {
            InterruptKey::Esc => KeySequence::parse("{Escape}")?,
            InterruptKey::CtrlC => KeySequence::parse("{C-c}")?,
        };
        target.send_keys(&sequence).await?;
        Ok(())
    }

    async fn send_text_with_handle(
        handle: &HostHandle,
        target: &SessionTarget,
        text: &str,
        paste_mode: PasteMode,
        enter: bool,
    ) -> anyhow::Result<()> {
        let target = Self::tmux_target_from_handle(handle, target).await?;
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
        target.send_keys(&sequence).await?;
        Ok(())
    }

    async fn tmux_target_from_handle(
        handle: &HostHandle,
        target: &SessionTarget,
    ) -> anyhow::Result<Target> {
        handle
            .session(target.session_name())
            .await?
            .with_context(|| format!("session '{}' not found", target))
    }

    async fn touch_session_shared(
        shared: Arc<Mutex<Self>>,
        target: &SessionTarget,
    ) -> anyhow::Result<()> {
        let handle = {
            let state = shared.lock().await;
            state.host(target.host_alias())?.handle.clone()
        };
        let tmux_target = Self::tmux_target_from_handle(&handle, target).await?;
        let now = tags::now_tag();
        tags::set_many(&tmux_target, &[("updated-at", now)]).await?;
        if let Some(record) = shared.lock().await.sessions.get_mut(target) {
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

    fn broadcast_targets(
        &self,
        request: &BroadcastRequest,
    ) -> anyhow::Result<Vec<BroadcastTarget>> {
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
            selected.push(BroadcastTarget {
                handle: self.host(target.host_alias())?.handle.clone(),
                target,
                state: session.state,
            });
        }
        Ok(selected)
    }

    fn recruit_plans(&self, request: &RecruitRequest) -> anyhow::Result<Vec<RecruitPlan>> {
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
                Ok(RecruitPlan {
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
        })
    }
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
    socket: Option<&str>,
    workstream: &str,
    target: &SessionTarget,
    role: &str,
) -> anyhow::Result<Vec<SessionEnvVar>> {
    let mut vars = vec![
        SessionEnvVar::new("MSTREAM_WORKSTREAM", workstream)?,
        SessionEnvVar::new("MSTREAM_TARGET", target.to_string())?,
        SessionEnvVar::new("MSTREAM_ROLE", role)?,
    ];
    if let Some(socket) = socket {
        vars.push(SessionEnvVar::new("MSTREAM_SOCKET", socket)?);
    }
    Ok(vars)
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
        " When finished, run: mstream session mark self --state done --summary \"<summary>\". \
         If blocked, use --state blocked; if you need input, use --state needs-input.",
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
}
