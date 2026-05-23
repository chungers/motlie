use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context};
use chrono::{DateTime, Utc};
use motlie_tmux::{
    CreateSessionOptions, HostHandle, KeySequence, SessionEnvVar, SessionTag, SshConfig, Target,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::time::sleep;

use crate::jsonl;
use crate::protocol::{
    AgentState, BroadcastRequest, ClientRequest, CloseRequest, ConnectRequest, EventsRequest,
    HandoffArmRequest, InterruptKey, InterruptRequest, JoinRequest, LeaveRequest, NewRequest,
    OpenRequest, PasteMode, RecruitRequest, SendRequest, SessionMarkRequest, SnapshotRequest,
    SummaryInputRequest,
};
use crate::tags;
use crate::target::SessionTarget;
use crate::timeline::PublicCursor;

const EVENT_LIMIT: usize = 1_000;

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

impl DaemonState {
    pub async fn handle(&mut self, request: ClientRequest) -> anyhow::Result<Vec<Value>> {
        match request {
            ClientRequest::DaemonStatus => Ok(vec![json!({
                "type": "status",
                "daemon": "running",
                "hosts": self.hosts.len(),
                "workstreams": self.workstreams.len(),
                "sessions": self.sessions.len(),
            })]),
            ClientRequest::DaemonStop => Ok(vec![jsonl::ok("daemon_stop")]),
            ClientRequest::Connect(request) => self.connect(request).await,
            ClientRequest::Hosts => Ok(vec![self.hosts_json()]),
            ClientRequest::Scan { alias } => self.scan(&alias).await,
            ClientRequest::Disconnect { alias } => self.disconnect(&alias),
            ClientRequest::Open(request) => self.open(request),
            ClientRequest::List => Ok(vec![self.workstream_list_json()]),
            ClientRequest::Show { workstream } => Ok(vec![self.show(&workstream)?]),
            ClientRequest::Close(request) => self.close(request).await,
            ClientRequest::Join(request) => self.join(request).await,
            ClientRequest::New(request) => self.new_session(request).await,
            ClientRequest::Leave(request) => self.leave(request).await,
            ClientRequest::Kill { target } => self.kill(&target).await,
            ClientRequest::Send(request) => self.send(request).await,
            ClientRequest::Interrupt(request) => self.interrupt(request).await,
            ClientRequest::Broadcast(request) => self.broadcast(request).await,
            ClientRequest::SessionList => Ok(vec![self.session_list_json()]),
            ClientRequest::SessionMark(request) => self.session_mark(request).await,
            ClientRequest::HandoffArm(request) => self.handoff_arm(request).await,
            ClientRequest::HandoffList { workstream } => Ok(vec![self.handoff_list(&workstream)?]),
            ClientRequest::HandoffCancel {
                workstream,
                handoff_id,
            } => self.handoff_cancel(&workstream, &handoff_id),
            ClientRequest::Status { workstream } => Ok(vec![self.status(&workstream)?]),
            ClientRequest::Events(request) => Ok(vec![self.events(request)?]),
            ClientRequest::Snapshot(request) => self.snapshot(request).await,
            ClientRequest::SummaryInput(request) => self.summary_input(request).await,
            ClientRequest::Recruit(request) => self.recruit(request).await,
        }
    }

    pub fn should_stop(request: &ClientRequest) -> bool {
        matches!(request, ClientRequest::DaemonStop)
    }

    async fn connect(&mut self, request: ConnectRequest) -> anyhow::Result<Vec<Value>> {
        if self.hosts.contains_key(&request.alias) {
            bail!("host alias '{}' is already connected", request.alias);
        }
        let config = SshConfig::parse(&request.ssh_uri)
            .with_context(|| format!("failed to parse ssh uri for '{}'", request.alias))?;
        let handle = config
            .connect()
            .await
            .with_context(|| format!("failed to connect host '{}'", request.alias))?;
        self.hosts.insert(
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

    async fn scan(&mut self, alias: &str) -> anyhow::Result<Vec<Value>> {
        let handle = self.host(alias)?.handle.clone();
        let sessions = handle.list_sessions().await?;
        let tags_by_session = handle
            .list_tags_for_session_infos(tags::PREFIX, &sessions)
            .await?;
        let mut hydrated = 0usize;
        for session in sessions {
            let target = SessionTarget::new(alias, session.name.clone())?;
            let tags = tags_by_session
                .get(&session.id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let parsed = ParsedTags::from_tags(tags);
            if parsed.managed || parsed.workstream.is_some() || parsed.state.is_some() {
                hydrated += 1;
                let state = parsed.state.unwrap_or(AgentState::Idle);
                let record = self.sessions.entry(target.clone()).or_insert_with(|| {
                    SessionRecord::from_target(&target, state, parsed.updated_at)
                });
                record.state = state;
                record.role = parsed.role.clone();
                record.agent = parsed.agent.clone();
                record.workstream = parsed.workstream.clone();
                record.last_report_kind = parsed.last_report_kind.clone();
                record.last_report_summary = parsed.last_report_summary.clone();
                record.context_domains = parsed.context_domains.clone();
                record.context_specialties = parsed.context_specialties.clone();
                record.context_summary = parsed.context_summary.clone();
                record.last_workstream = parsed.last_workstream.clone();
                record.last_workstream_title = parsed.last_workstream_title.clone();

                if let Some(workstream_name) = parsed.workstream {
                    let title = parsed
                        .workstream_title
                        .unwrap_or_else(|| workstream_name.clone());
                    let generation = self.next_generation();
                    let workstream = self
                        .workstreams
                        .entry(workstream_name.clone())
                        .or_insert_with(|| WorkstreamRecord::new(title, None, None, generation));
                    workstream.sessions.insert(target.clone());
                    let _ = handle.start_monitoring_session(target.session_name()).await;
                }
            }
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "scan",
            "host": alias,
            "hydrated_sessions": hydrated,
        })])
    }

    fn open(&mut self, request: OpenRequest) -> anyhow::Result<Vec<Value>> {
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
            if let Some(generation) = new_generation {
                workstream.reopen(generation);
            } else {
                workstream.state = WorkstreamState::Open;
            }
            cursor = workstream.cursor(&request.workstream)?;
        } else {
            let generation = self.next_generation();
            let workstream =
                WorkstreamRecord::new(request.title, request.goal, request.domain, generation);
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

    async fn join(&mut self, request: JoinRequest) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        self.ensure_workstream_open(&request.workstream)?;
        let tmux_target = self.tmux_target(&target).await?;
        self.write_assignment(
            &tmux_target,
            &request.workstream,
            &target,
            &request.role,
            None,
            None,
        )
        .await?;
        self.add_session_to_workstream(
            &request.workstream,
            target.clone(),
            request.role,
            None,
            None,
            AgentState::Busy,
        )?;
        self.host(target.host_alias())?
            .handle
            .start_monitoring_session(target.session_name())
            .await?;
        if let Some(task) = request.task {
            self.send_text_to_target(
                &target,
                &managed_prompt(
                    &request.workstream,
                    &target,
                    self.session_role(&target),
                    None,
                    Some(&task),
                ),
                PasteMode::Bracketed,
                true,
            )
            .await?;
        }
        let cursor = self.record_event(
            &request.workstream,
            EventDraft::new("joined")
                .target(&target)
                .state(AgentState::Busy),
        )?;
        Ok(vec![json!({
            "type": "ok",
            "op": "join",
            "workstream": request.workstream,
            "target": target.to_string(),
            "cursor": cursor,
        })])
    }

    async fn new_session(&mut self, request: NewRequest) -> anyhow::Result<Vec<Value>> {
        if !request.cwd.is_absolute() {
            bail!("--cwd must be an absolute path");
        }
        self.ensure_workstream_open(&request.workstream)?;
        let target: SessionTarget = request.target.parse()?;
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
        let tmux_target = self
            .host(target.host_alias())?
            .handle
            .create_session(target.session_name(), &opts)
            .await?;
        self.write_assignment(
            &tmux_target,
            &request.workstream,
            &target,
            &request.role,
            Some(&request.agent),
            Some(&request.cwd),
        )
        .await?;
        self.add_session_to_workstream(
            &request.workstream,
            target.clone(),
            request.role,
            Some(request.agent),
            Some(request.cwd.clone()),
            AgentState::Busy,
        )?;
        self.host(target.host_alias())?
            .handle
            .start_monitoring_session(target.session_name())
            .await?;
        if let Some(task) = request.task {
            self.send_text_to_target(
                &target,
                &managed_prompt(
                    &request.workstream,
                    &target,
                    self.session_role(&target),
                    Some(&request.cwd),
                    Some(&task),
                ),
                PasteMode::Bracketed,
                true,
            )
            .await?;
        }
        let cursor = self.record_event(
            &request.workstream,
            EventDraft::new("created")
                .target(&target)
                .state(AgentState::Busy),
        )?;
        Ok(vec![json!({
            "type": "ok",
            "op": "new",
            "workstream": request.workstream,
            "target": target.to_string(),
            "cursor": cursor,
        })])
    }

    async fn close(&mut self, request: CloseRequest) -> anyhow::Result<Vec<Value>> {
        let sessions = self.workstream(&request.workstream)?.sessions.clone();
        let title = self.workstream(&request.workstream)?.title.clone();
        for target in &sessions {
            if let Ok(tmux_target) = self.tmux_target(target).await {
                let mut pairs = vec![
                    ("state", tags::state_value(AgentState::Available)),
                    ("last-workstream", request.workstream.clone()),
                    ("last-workstream-title", title.clone()),
                    ("updated-at", tags::now_tag()),
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
            }
            if let Some(record) = self.sessions.get_mut(target) {
                record.state = AgentState::Available;
                record.workstream = None;
                record.last_workstream = Some(request.workstream.clone());
                record.last_workstream_title = Some(title.clone());
                record.updated_at = Utc::now();
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
        let cursor = if let Some(summary) = request.summary.clone() {
            self.record_event(
                &request.workstream,
                EventDraft::new("closed")
                    .text(summary.clone())
                    .summary(summary),
            )?
        } else {
            self.record_event(&request.workstream, EventDraft::new("closed"))?
        };
        let workstream = self.workstream_mut(&request.workstream)?;
        workstream.state = WorkstreamState::Closed;
        workstream.sessions.clear();
        Ok(vec![json!({
            "type": "ok",
            "op": "close",
            "workstream": request.workstream,
            "cursor": cursor,
        })])
    }

    async fn leave(&mut self, request: LeaveRequest) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        let tmux_target = self.tmux_target(&target).await?;
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
        if request.available {
            tags::set_many(
                &tmux_target,
                &[
                    ("state", tags::state_value(AgentState::Available)),
                    ("updated-at", tags::now_tag()),
                ],
            )
            .await?;
        }
        if let Some(record) = self.sessions.get_mut(&target) {
            record.workstream = None;
            if request.available {
                record.state = AgentState::Available;
            }
            record.updated_at = Utc::now();
        }
        let mut event = EventDraft::new("left").target(&target);
        if request.available {
            event = event.state(AgentState::Available);
        }
        let cursor = self.record_event(&request.workstream, event)?;
        let workstream = self.workstream_mut(&request.workstream)?;
        workstream.sessions.remove(&target);
        Ok(vec![json!({
            "type": "ok",
            "op": "leave",
            "workstream": request.workstream,
            "target": target.to_string(),
            "cursor": cursor,
        })])
    }

    async fn kill(&mut self, target: &str) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = target.parse()?;
        let tmux_target = self.tmux_target(&target).await?;
        tmux_target.kill().await?;
        self.sessions.remove(&target);
        for workstream in self.workstreams.values_mut() {
            workstream.sessions.remove(&target);
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "kill",
            "target": target.to_string(),
        })])
    }

    async fn send(&mut self, request: SendRequest) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        self.ensure_target_in_workstream(&request.workstream, &target)?;
        let current_state = self
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
        if request.interrupt_first {
            self.send_interrupt_to_target(&target, InterruptKey::Esc)
                .await?;
            sleep(Duration::from_millis(request.settle_ms)).await;
        }
        self.send_text_to_target(&target, &request.text, request.paste_mode, request.enter)
            .await?;
        if let Some(state) = request.set_state {
            self.set_session_state(&target, state, None).await?;
        }
        let state_after = self
            .sessions
            .get(&target)
            .map(|record| record.state)
            .unwrap_or(current_state);
        let cursor = self.record_event(
            &request.workstream,
            EventDraft::new("message_sent")
                .target(&target)
                .text(request.text)
                .state(state_after),
        )?;
        Ok(vec![json!({
            "type": "ok",
            "op": "send",
            "workstream": request.workstream,
            "target": target.to_string(),
            "target_state": state_after.as_str(),
            "mid_generation_risk": current_state == AgentState::Busy && !request.interrupt_first,
            "paste_mode": request.paste_mode.as_str(),
            "enter": request.enter,
            "cursor": cursor,
        })])
    }

    async fn interrupt(&mut self, request: InterruptRequest) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        self.send_interrupt_to_target(&target, request.key).await?;
        Ok(vec![json!({
            "type": "ok",
            "op": "interrupt",
            "target": target.to_string(),
            "key": request.key.as_str(),
        })])
    }

    async fn broadcast(&mut self, request: BroadcastRequest) -> anyhow::Result<Vec<Value>> {
        let targets = self.workstream(&request.workstream)?.sessions.clone();
        let mut records = Vec::new();
        let mut sent = 0usize;
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
            self.send_text_to_target(&target, &request.text, request.paste_mode, request.enter)
                .await?;
            sent += 1;
            let cursor = self.record_event(
                &request.workstream,
                EventDraft::new("broadcast_sent")
                    .target(&target)
                    .text(request.text.clone())
                    .state(session.state),
            )?;
            records.push(json!({
                "type": "ok",
                "op": "broadcast_sent",
                "workstream": request.workstream,
                "target": target.to_string(),
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

    async fn session_mark(&mut self, request: SessionMarkRequest) -> anyhow::Result<Vec<Value>> {
        let target: SessionTarget = request.target.parse()?;
        self.set_session_state(&target, request.state, Some(&request.summary))
            .await?;
        let workstream = self
            .sessions
            .get(&target)
            .and_then(|record| record.workstream.clone());
        let mut records = Vec::new();
        if let Some(workstream) = workstream {
            let cursor = self.record_event(
                &workstream,
                EventDraft::new(request.state.event_kind().unwrap_or("session_marked"))
                    .target(&target)
                    .state(request.state)
                    .summary(request.summary.clone()),
            )?;
            records.push(json!({
                "type": "event",
                "kind": request.state.event_kind().unwrap_or("session_marked"),
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
        let fired = self.collect_matching_handoffs(&target, request.state);
        for (workstream, handoff_id) in fired {
            records.push(self.fire_handoff(&workstream, &handoff_id).await?);
        }
        Ok(records)
    }

    async fn handoff_arm(&mut self, request: HandoffArmRequest) -> anyhow::Result<Vec<Value>> {
        self.ensure_target_in_workstream(&request.workstream, &request.from.parse()?)?;
        self.ensure_target_in_workstream(&request.workstream, &request.to.parse()?)?;
        let from: SessionTarget = request.from.parse()?;
        let to: SessionTarget = request.to.parse()?;
        let id = format!("h{}", self.next_handoff_id);
        self.next_handoff_id += 1;
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
        self.workstream_mut(&request.workstream)?
            .handoffs
            .insert(id.clone(), handoff);
        let already_met = self
            .sessions
            .get(&from)
            .is_some_and(|record| record.state == request.on);
        if already_met && !request.only_on_transition {
            let record = self.fire_handoff(&request.workstream, &id).await?;
            return Ok(vec![record]);
        }
        Ok(vec![json!({
            "type": "ok",
            "op": "handoff_arm",
            "workstream": request.workstream,
            "handoff_id": id,
            "from": from.to_string(),
            "on": request.on.as_str(),
            "only_on_transition": request.only_on_transition,
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

    async fn snapshot(&self, request: SnapshotRequest) -> anyhow::Result<Vec<Value>> {
        let text = self
            .capture_workstream(&request.workstream, request.max_chars)
            .await?;
        Ok(vec![json!({
            "type": "snapshot",
            "workstream": request.workstream,
            "after": request.after,
            "text": text,
            "ordering": "arrival",
        })])
    }

    async fn summary_input(&self, request: SummaryInputRequest) -> anyhow::Result<Vec<Value>> {
        let text = self
            .capture_workstream(&request.workstream, request.max_chars)
            .await?;
        Ok(vec![json!({
            "type": "summary_input",
            "workstream": request.workstream,
            "since": request.since,
            "text": compact_text(&text, request.max_chars),
            "ordering": "arrival",
        })])
    }

    async fn recruit(&mut self, request: RecruitRequest) -> anyhow::Result<Vec<Value>> {
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
        let selected: Vec<SessionTarget> = candidates.into_iter().take(request.count).collect();
        let mut records = Vec::new();
        for target in selected {
            self.add_session_to_workstream(
                &request.workstream,
                target.clone(),
                request.role.clone(),
                request.agent.clone(),
                None,
                AgentState::Reserved,
            )?;
            self.set_session_state(&target, AgentState::Reserved, None)
                .await?;
            if let Some(task) = &request.task {
                self.send_text_to_target(&target, task, PasteMode::Bracketed, true)
                    .await?;
                self.set_session_state(&target, AgentState::Busy, None)
                    .await?;
            }
            let state = if request.task.is_some() {
                AgentState::Busy
            } else {
                AgentState::Reserved
            };
            let mut event = EventDraft::new("recruited").target(&target).state(state);
            if let Some(goal) = request.goal.clone() {
                event = event.text(goal);
            }
            let cursor = self.record_event(&request.workstream, event)?;
            records.push(json!({
                "type": "ok",
                "op": "recruited",
                "workstream": request.workstream,
                "target": target.to_string(),
                "cursor": cursor,
            }));
        }
        Ok(records)
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
            "cursor": workstream_record.cursor(workstream)?,
            "agents": agents,
        }))
    }

    fn events(&self, request: EventsRequest) -> anyhow::Result<Value> {
        let workstream = self.workstream(&request.workstream)?;
        let from = match request.after {
            Some(cursor) => match PublicCursor::decode(&cursor) {
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
        Ok(json!({
            "type": "events",
            "workstream": request.workstream,
            "events": events,
            "cursor": workstream.cursor(&request.workstream)?,
            "ordering": "arrival",
        }))
    }

    async fn capture_workstream(
        &self,
        workstream: &str,
        max_chars: usize,
    ) -> anyhow::Result<String> {
        let workstream = self.workstream(workstream)?;
        let mut text = String::new();
        for target in &workstream.sessions {
            let capture = match self.tmux_target(target).await {
                Ok(target_handle) => target_handle.capture().await.unwrap_or_default(),
                Err(_) => String::new(),
            };
            text.push_str(&format!("=== {} ===\n", target));
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

    async fn fire_handoff(&mut self, workstream: &str, handoff_id: &str) -> anyhow::Result<Value> {
        let handoff = {
            let workstream_record = self.workstream_mut(workstream)?;
            let Some(handoff) = workstream_record.handoffs.get_mut(handoff_id) else {
                bail!("handoff '{handoff_id}' not found in workstream '{workstream}'");
            };
            if handoff.fired {
                return Ok(json!({
                    "type": "ok",
                    "op": "handoff_already_fired",
                    "workstream": workstream,
                    "handoff_id": handoff_id,
                }));
            }
            handoff.fired = true;
            handoff.clone()
        };
        self.set_session_state(&handoff.to, AgentState::Busy, None)
            .await?;
        self.send_text_to_target(&handoff.to, &handoff.task, PasteMode::Bracketed, true)
            .await?;
        let cursor = self.record_event(
            workstream,
            EventDraft::new("handoff_fired")
                .target(&handoff.to)
                .text(handoff.task)
                .state(AgentState::Busy)
                .handoff_id(handoff.id.clone()),
        )?;
        Ok(json!({
            "type": "event",
            "kind": "handoff_fired",
            "workstream": workstream,
            "handoff_id": handoff.id,
            "from": handoff.from.to_string(),
            "to": handoff.to.to_string(),
            "state": AgentState::Busy.as_str(),
            "cursor": cursor,
        }))
    }

    fn collect_matching_handoffs(
        &self,
        target: &SessionTarget,
        state: AgentState,
    ) -> Vec<(String, String)> {
        let mut handoffs = Vec::new();
        for (workstream_name, workstream) in &self.workstreams {
            for (handoff_id, handoff) in &workstream.handoffs {
                if !handoff.fired && handoff.from == *target && handoff.on == state {
                    handoffs.push((workstream_name.clone(), handoff_id.clone()));
                }
            }
        }
        handoffs
    }

    async fn set_session_state(
        &mut self,
        target: &SessionTarget,
        state: AgentState,
        summary: Option<&str>,
    ) -> anyhow::Result<()> {
        let tmux_target = self.tmux_target(target).await?;
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
        let record = self
            .sessions
            .entry(target.clone())
            .or_insert_with(|| SessionRecord::from_target(target, state, Some(Utc::now())));
        record.state = state;
        record.updated_at = Utc::now();
        if let Some(summary) = summary {
            record.last_report_kind = Some(state.as_str().to_string());
            record.last_report_summary = Some(summary.to_string());
        }
        Ok(())
    }

    async fn send_interrupt_to_target(
        &self,
        target: &SessionTarget,
        key: InterruptKey,
    ) -> anyhow::Result<()> {
        let target = self.tmux_target(target).await?;
        let sequence = match key {
            InterruptKey::Esc => KeySequence::parse("{Escape}")?,
            InterruptKey::CtrlC => KeySequence::parse("{C-c}")?,
        };
        target.send_keys(&sequence).await?;
        Ok(())
    }

    async fn send_text_to_target(
        &self,
        target: &SessionTarget,
        text: &str,
        paste_mode: PasteMode,
        enter: bool,
    ) -> anyhow::Result<()> {
        let target = self.tmux_target(target).await?;
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

    async fn write_assignment(
        &self,
        target: &Target,
        workstream: &str,
        session_target: &SessionTarget,
        role: &str,
        agent: Option<&str>,
        cwd: Option<&Path>,
    ) -> anyhow::Result<()> {
        let workstream_record = self.workstream(workstream)?;
        let mut pairs = vec![
            ("version", "1".to_string()),
            ("managed", "true".to_string()),
            ("workstream", workstream.to_string()),
            ("workstream-title", workstream_record.title.clone()),
            ("workstream-state", "open".to_string()),
            ("role", role.to_string()),
            ("identity", session_target.session_name().to_string()),
            ("state", tags::state_value(AgentState::Busy)),
            ("updated-at", tags::now_tag()),
        ];
        if let Some(goal) = &workstream_record.goal {
            pairs.push(("workstream-goal", goal.clone()));
        }
        if let Some(domain) = &workstream_record.domain {
            pairs.push(("workstream-domain", domain.clone()));
        }
        if let Some(agent) = agent {
            pairs.push(("agent", agent.to_string()));
        }
        if let Some(cwd) = cwd {
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
        while workstream_record.events.len() > EVENT_LIMIT {
            workstream_record.events.pop_front();
        }
        Ok(cursor)
    }

    async fn tmux_target(&self, target: &SessionTarget) -> anyhow::Result<Target> {
        let host = self.host(target.host_alias())?;
        host.handle
            .session(target.session_name())
            .await?
            .with_context(|| format!("session '{}' not found", target))
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

    fn session_role(&self, target: &SessionTarget) -> Option<&str> {
        self.sessions
            .get(target)
            .and_then(|record| record.role.as_deref())
    }

    fn host_matches_selectors(&self, alias: &str, selectors: &BTreeMap<String, String>) -> bool {
        let Some(host) = self.hosts.get(alias) else {
            return false;
        };
        selectors
            .iter()
            .all(|(key, value)| host.labels.get(key) == Some(value))
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
    fn new(title: String, goal: Option<String>, domain: Option<String>, generation: u64) -> Self {
        Self {
            title,
            goal,
            domain,
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
}
