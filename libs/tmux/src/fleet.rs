//! Fleet coordination and routing (DC27, DC9).
//!
//! `Fleet` is a programmatic registry of `HostHandle`s with convenience routing.
//! It owns an `OutputBus`, manages per-host monitoring lifecycle, and provides
//! target alias bindings for cross-host routing.
//!
//! Fleet-level routing helpers are convenience wrappers over normal
//! `HostHandle` / `Target` operations, not a separate action system.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::error::{Error, Result};

use crate::host::{HostHandle, Target};
use crate::keys::KeySequence;
use crate::monitor::{MonitorHandle, MonitorHealth, SessionMonitorHandle};
use crate::sink::{OutputBus, SinkFilter, TimelineMarkerScope, TimelineOptions};
use crate::types::{SessionId, SessionInfo, SessionTag, TargetAddress, TargetLevel, TargetSpec};

// ---------------------------------------------------------------------------
// HostStatus
// ---------------------------------------------------------------------------

/// Per-session monitor status within a Fleet host (DC29, 4.2d).
#[derive(Debug, Clone)]
pub struct SessionMonitorStatus {
    /// Session name.
    pub name: String,
    /// Per-session monitor health (ground truth).
    pub health: MonitorHealth,
}

/// Status of a host within the fleet.
///
/// Per-session monitor health is the ground truth (DC29). Host-level status
/// is derived from per-session states.
#[derive(Debug, Clone)]
pub enum HostStatus {
    /// Registered but not monitoring any sessions.
    Connected,
    /// Monitoring one or more sessions with per-session health.
    Monitoring { sessions: Vec<SessionMonitorStatus> },
    /// Host encountered an error.
    Error(String),
}

// ---------------------------------------------------------------------------
// FleetTargetSpec
// ---------------------------------------------------------------------------

/// Cross-host target spec used by Fleet routing APIs.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct FleetTargetSpec {
    pub host_alias: String,
    pub target: TargetSpec,
}

impl FleetTargetSpec {
    /// Construct a spec from a host alias and tmux target spec.
    pub fn new(host_alias: impl Into<String>, target: TargetSpec) -> Result<Self> {
        let host_alias = host_alias.into();
        validate_fleet_host_alias(&host_alias)?;
        Ok(Self { host_alias, target })
    }

    /// Construct a session-level spec.
    pub fn session(host_alias: impl Into<String>, session: impl Into<String>) -> Result<Self> {
        let session = session.into();
        Self::new(host_alias, TargetSpec::session(&session))
    }

    /// Construct a session-level spec addressed by stable tmux session id.
    pub fn session_id(
        host_alias: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self> {
        Self::new(host_alias, TargetSpec::session_id(session_id)?)
    }

    /// Host alias portion of the spec.
    pub fn host_alias(&self) -> &str {
        &self.host_alias
    }

    /// Target spec portion.
    pub fn target_spec(&self) -> &TargetSpec {
        &self.target
    }

    /// Target's session name.
    pub fn session_name(&self) -> &str {
        self.target.session_name()
    }

    /// Target's stable tmux session id when this spec is id-addressed.
    pub fn session_id_selector(&self) -> Option<&SessionId> {
        self.target.session_id_selector()
    }

    fn level(&self) -> TargetLevel {
        match (self.target.window_selector(), self.target.pane_index()) {
            (None, None) => TargetLevel::Session,
            (Some(_), None) => TargetLevel::Window,
            (Some(_), Some(_)) => TargetLevel::Pane,
            (None, Some(_)) => TargetLevel::Pane,
        }
    }
}

impl FromStr for FleetTargetSpec {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let mut parts = value.split("::");
        let host_alias = parts.next().unwrap_or_default();
        let target = parts.next().ok_or_else(|| {
            Error::Parse("fleet target must use '<host-alias>::<target>'".to_string())
        })?;
        if parts.next().is_some() {
            return Err(Error::Parse(format!(
                "fleet target has ambiguous host delimiter: {value:?}"
            )));
        }
        if target.is_empty() {
            return Err(Error::Parse(
                "fleet target side cannot be empty".to_string(),
            ));
        }
        Self::new(host_alias, TargetSpec::parse(target)?)
    }
}

impl fmt::Display for FleetTargetSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.host_alias, self.target)
    }
}

fn validate_fleet_host_alias(alias: &str) -> Result<()> {
    if alias.is_empty() {
        return Err(Error::Parse("fleet host alias cannot be empty".to_string()));
    }
    if alias.contains("::") {
        return Err(Error::Parse(format!(
            "fleet host alias cannot contain '::': {alias:?}"
        )));
    }
    Ok(())
}

fn ensure_session_level(spec: &FleetTargetSpec) -> Result<()> {
    if spec.level() != TargetLevel::Session {
        return Err(Error::UnsupportedTarget {
            operation: "fleet session monitoring",
            level: spec.level(),
        });
    }
    Ok(())
}

fn session_lookup(spec: &FleetTargetSpec) -> &str {
    spec.session_id_selector()
        .map(SessionId::as_str)
        .unwrap_or_else(|| spec.session_name())
}

/// A cross-host target resolved against the current Fleet registry.
#[derive(Clone)]
pub struct ResolvedFleetTarget {
    /// Stable logical address.
    pub spec: FleetTargetSpec,
    /// Cloned host handle from the fleet registry.
    pub host: HostHandle,
    /// Resolved tmux target.
    pub target: Target,
}

impl ResolvedFleetTarget {
    /// Exact OutputBus filter for this target's host/session.
    pub fn sink_filter(&self) -> SinkFilter {
        SinkFilter::for_host_session(
            self.spec.host_alias(),
            self.target
                .session_id()
                .unwrap_or_else(|| self.target.session_name()),
        )
    }

    /// Exact marker scope for gap/discontinuity markers for this target.
    pub fn marker_scope(&self) -> TimelineMarkerScope {
        match self.target.session_id() {
            Some(id) => TimelineMarkerScope::for_host_session_identity(
                self.spec.host_alias(),
                self.target.session_name(),
                id,
            ),
            None => TimelineMarkerScope::for_host_session(
                self.spec.host_alias(),
                self.target.session_name(),
            ),
        }
    }
}

/// Options for fleet-wide session snapshots.
#[derive(Debug, Clone, Default)]
pub struct FleetSnapshotOptions {
    /// If None, scan all registered hosts. If Some, scan only listed aliases.
    pub hosts: Option<Vec<String>>,
    /// Optional tmux user-option tag prefixes to hydrate per session.
    pub tag_prefixes: Vec<String>,
}

/// One session from a fleet-wide snapshot.
#[derive(Debug, Clone)]
pub struct FleetSessionSnapshot {
    /// Fleet host alias.
    pub host_alias: String,
    /// Tmux session info from that host.
    pub session: SessionInfo,
    /// Cross-host session target for this session.
    pub target: FleetTargetSpec,
    /// Tags grouped by requested prefix.
    pub tags: BTreeMap<String, Vec<SessionTag>>,
}

/// One session listed from a registered fleet host.
#[derive(Debug, Clone)]
pub struct FleetSessionInfo {
    /// Fleet host alias.
    pub host_alias: String,
    /// Tmux session info from that host.
    pub session: SessionInfo,
    /// Tags read for this session under the requested prefix.
    pub tags: Vec<SessionTag>,
}

/// A named binding from a target alias to a resolved target.
#[derive(Debug, Clone)]
struct TargetAliasEntry {
    target: FleetTargetSpec,
}

// ---------------------------------------------------------------------------
// Fleet
// ---------------------------------------------------------------------------

/// Multi-host coordination registry (DC27).
///
/// Provides programmatic host registration, aggregate monitoring lifecycle,
/// target-alias routing, and a shared `OutputBus` that aggregates output
/// from all registered hosts.
pub struct Fleet {
    hosts: HashMap<String, HostHandle>,
    bus: Arc<OutputBus>,
    target_aliases: HashMap<String, TargetAliasEntry>,
    /// Per-host session monitors: alias -> [(session_name, handle)].
    session_monitors: HashMap<String, Vec<(String, SessionMonitorHandle)>>,
    /// Per-host full monitors (all sessions).
    full_monitors: HashMap<String, MonitorHandle>,
}

impl Fleet {
    /// Create an empty fleet with a fresh `OutputBus`.
    pub fn new() -> Self {
        Fleet {
            hosts: HashMap::new(),
            bus: Arc::new(OutputBus::new()),
            target_aliases: HashMap::new(),
            session_monitors: HashMap::new(),
            full_monitors: HashMap::new(),
        }
    }

    /// Register a host by alias.
    ///
    /// The fleet alias must match the host's own alias (`host.host_alias()`)
    /// so that output labels and routing names stay consistent in external-agent
    /// workflows. Returns an error if the alias is already taken, the aliases
    /// mismatch, or the host's output bus was already initialized.
    ///
    /// Injects the fleet's shared `OutputBus` into the host so all monitors
    /// publish to a single aggregation bus (DC27).
    pub fn register(&mut self, alias: &str, host: HostHandle) -> Result<()> {
        if self.hosts.contains_key(alias) {
            return Err(Error::AlreadyExists(format!(
                "alias '{}' already registered",
                alias
            )));
        }
        if host.host_alias() != alias {
            return Err(Error::AlreadyExists(format!(
                "fleet alias '{}' does not match host alias '{}'; \
                 create the host with HostHandle::with_alias() using the fleet alias",
                alias,
                host.host_alias()
            )));
        }
        host.inject_output_bus(self.bus.clone())?;
        self.hosts.insert(alias.to_string(), host);
        Ok(())
    }

    /// Remove a registered host and any Fleet-owned routing or monitor state
    /// associated with it.
    pub fn unregister(&mut self, alias: &str) -> Result<HostHandle> {
        let host = self
            .hosts
            .remove(alias)
            .ok_or_else(|| Error::NotFound(format!("host '{}' not registered", alias)))?;
        self.target_aliases
            .retain(|_, entry| entry.target.host_alias() != alias);
        self.session_monitors.remove(alias);
        self.full_monitors.remove(alias);
        host.stop_monitoring();
        Ok(host)
    }

    /// Look up a host by alias.
    pub fn host(&self, name: &str) -> Option<&HostHandle> {
        self.hosts.get(name)
    }

    /// Iterator over all registered `(alias, HostHandle)` pairs.
    pub fn hosts(&self) -> impl Iterator<Item = (&str, &HostHandle)> {
        self.hosts.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of registered hosts.
    pub fn host_count(&self) -> usize {
        self.hosts.len()
    }

    /// The shared `OutputBus` aggregating output from all hosts.
    pub fn output_bus(&self) -> Arc<OutputBus> {
        self.bus.clone()
    }

    /// Status of a registered host with per-session health (DC29, 4.2d).
    ///
    /// Per-session monitor health is the ground truth. Host status is derived
    /// from per-session states.
    pub fn host_status(&self, alias: &str) -> Option<HostStatus> {
        if !self.hosts.contains_key(alias) {
            return None;
        }
        let mut statuses = Vec::new();

        // Per-session monitors
        if let Some(monitors) = self.session_monitors.get(alias) {
            for (_, handle) in monitors {
                statuses.push(SessionMonitorStatus {
                    name: handle.display_name(),
                    health: handle.health(),
                });
            }
        }
        // Full monitors (aggregate handle) — use all_sessions() so that
        // failed/stopped sessions remain visible in health status (DC29).
        if let Some(monitor) = self.full_monitors.get(alias) {
            for session_name in monitor.all_sessions() {
                if let Some(handle) = monitor.get(&session_name) {
                    statuses.push(SessionMonitorStatus {
                        name: session_name.to_string(),
                        health: handle.health(),
                    });
                }
            }
        }

        if statuses.is_empty() {
            Some(HostStatus::Connected)
        } else {
            Some(HostStatus::Monitoring { sessions: statuses })
        }
    }

    // --- Target resolution ---

    /// Resolve a cross-host target spec.
    pub async fn resolve_target(
        &self,
        spec: &FleetTargetSpec,
    ) -> Result<Option<ResolvedFleetTarget>> {
        let host = self
            .hosts
            .get(spec.host_alias())
            .ok_or_else(|| Error::NotFound(format!("host '{}' not registered", spec.host_alias())))?
            .clone();
        let target = match host.target(spec.target_spec()).await? {
            Some(target) => target,
            None => return Ok(None),
        };
        Ok(Some(ResolvedFleetTarget {
            spec: spec.clone(),
            host,
            target,
        }))
    }

    /// Resolve a cross-host target and return `NotFound` if the tmux target is missing.
    pub async fn require_target(&self, spec: &FleetTargetSpec) -> Result<ResolvedFleetTarget> {
        self.resolve_target(spec).await?.ok_or_else(|| {
            Error::NotFound(format!(
                "target '{}' not found on host '{}'",
                spec.target_spec(),
                spec.host_alias()
            ))
        })
    }

    /// Resolve many cross-host targets.
    pub async fn resolve_targets<I>(&self, specs: I) -> Result<Vec<ResolvedFleetTarget>>
    where
        I: IntoIterator<Item = FleetTargetSpec>,
    {
        let mut out = Vec::new();
        for spec in specs {
            if let Some(resolved) = self.resolve_target(&spec).await? {
                out.push(resolved);
            }
        }
        Ok(out)
    }

    /// Compatibility name for [`require_target`](Self::require_target).
    pub async fn target_by_spec(&self, spec: &FleetTargetSpec) -> Result<Target> {
        Ok(self.require_target(spec).await?.target)
    }

    // --- Monitoring lifecycle ---

    /// Start monitoring a specific session on a host.
    pub async fn start_monitoring_session(&mut self, alias: &str, session: &str) -> Result<()> {
        let host = self
            .hosts
            .get(alias)
            .ok_or_else(|| Error::NotFound(format!("host '{}' not registered", alias)))?
            .clone();

        let target = host
            .target(&TargetSpec::session(session))
            .await?
            .ok_or_else(|| Error::NotFound(format!("session '{}' not found", session)))?;
        let id = target
            .session_id()
            .ok_or_else(|| Error::State(format!("session '{}' has no session id", session)))?
            .to_string();

        if self
            .session_monitors
            .get(alias)
            .is_some_and(|monitors| monitors.iter().any(|(key, _)| key == &id))
        {
            return Ok(());
        }

        if self
            .full_monitors
            .get(alias)
            .is_some_and(|monitor| monitor.get(&id).is_some() || monitor.get(session).is_some())
        {
            return Ok(());
        }

        let monitor = host.start_monitoring_session_by_id(&id).await?;
        self.session_monitors
            .entry(alias.to_string())
            .or_default()
            .push((id, monitor));
        Ok(())
    }

    /// Start monitoring the session addressed by a cross-host target spec.
    ///
    /// Window and pane specs monitor their containing session.
    pub async fn start_monitoring_target(&mut self, spec: &FleetTargetSpec) -> Result<()> {
        if let Some(id) = spec.session_id_selector() {
            let host = self
                .hosts
                .get(spec.host_alias())
                .ok_or_else(|| {
                    Error::NotFound(format!("host '{}' not registered", spec.host_alias()))
                })?
                .clone();
            let id = id.as_str().to_string();
            if self
                .session_monitors
                .get(spec.host_alias())
                .is_some_and(|monitors| monitors.iter().any(|(key, _)| key == &id))
            {
                return Ok(());
            }
            if self
                .full_monitors
                .get(spec.host_alias())
                .is_some_and(|monitor| monitor.get(&id).is_some())
            {
                return Ok(());
            }
            let monitor = host.start_monitoring_session_by_id(&id).await?;
            self.session_monitors
                .entry(spec.host_alias().to_string())
                .or_default()
                .push((id, monitor));
            Ok(())
        } else {
            self.start_monitoring_session(spec.host_alias(), spec.session_name())
                .await
        }
    }

    /// Stop monitoring the session addressed by a cross-host target spec.
    ///
    /// Returns `Ok(())` when the target is already unmonitored.
    pub fn stop_monitoring_target(&mut self, spec: &FleetTargetSpec) -> Result<()> {
        if !self.hosts.contains_key(spec.host_alias()) {
            return Err(Error::NotFound(format!(
                "host '{}' not registered",
                spec.host_alias()
            )));
        }

        if let Some(monitors) = self.session_monitors.get_mut(spec.host_alias()) {
            let lookup = session_lookup(spec);
            if let Some(pos) = monitors.iter().position(|(key, handle)| {
                key == lookup || handle.display_name() == spec.session_name()
            }) {
                monitors.remove(pos);
            }
        }

        if let Some(host) = self.hosts.get(spec.host_alias()) {
            let _ = match spec.session_id_selector() {
                Some(id) => host.stop_monitoring_session_by_id(id.as_str()),
                None => host.stop_monitoring_session(spec.session_name()),
            };
        }
        if let Some(monitor) = self.full_monitors.get_mut(spec.host_alias()) {
            monitor.remove_session(session_lookup(spec));
        }
        Ok(())
    }

    /// Ensure a session-level target is monitored.
    pub async fn ensure_monitoring_session(
        &mut self,
        target: &FleetTargetSpec,
    ) -> Result<SessionMonitorStatus> {
        ensure_session_level(target)?;

        if let Some(status) = self.monitor_status_for_target(target) {
            if !matches!(
                status.health,
                MonitorHealth::Failed | MonitorHealth::Stopped
            ) {
                return Ok(status);
            }
            self.remove_session_monitor(target.host_alias(), session_lookup(target));
        }

        self.start_monitoring_target(target).await?;
        self.monitor_status_for_target(target).ok_or_else(|| {
            Error::State(format!(
                "monitor for target '{}' was not registered after start",
                target
            ))
        })
    }

    /// Ensure several session-level targets are monitored.
    pub async fn ensure_monitoring_sessions<I>(
        &mut self,
        targets: I,
    ) -> Result<Vec<SessionMonitorStatus>>
    where
        I: IntoIterator<Item = FleetTargetSpec>,
    {
        let mut statuses = Vec::new();
        for target in targets {
            statuses.push(self.ensure_monitoring_session(&target).await?);
        }
        Ok(statuses)
    }

    /// Stop monitoring one session-level target.
    pub fn stop_monitoring_session_target(&mut self, target: &FleetTargetSpec) -> Result<()> {
        ensure_session_level(target)?;
        self.stop_monitoring_target(target)
    }

    /// Return monitor status for a session-level target, if tracked.
    pub fn monitor_status_for_target(
        &self,
        target: &FleetTargetSpec,
    ) -> Option<SessionMonitorStatus> {
        if target.level() != TargetLevel::Session {
            return None;
        }

        if let Some(monitors) = self.session_monitors.get(target.host_alias()) {
            let lookup = session_lookup(target);
            if let Some((_, handle)) = monitors.iter().find(|(key, handle)| {
                key == lookup || handle.display_name() == target.session_name()
            }) {
                return Some(SessionMonitorStatus {
                    name: handle.display_name(),
                    health: handle.health(),
                });
            }
        }

        self.full_monitors
            .get(target.host_alias())
            .and_then(|monitor| monitor.get(session_lookup(target)))
            .map(|handle| SessionMonitorStatus {
                name: handle.display_name(),
                health: handle.health(),
            })
    }

    fn remove_session_monitor(&mut self, alias: &str, session: &str) {
        if let Some(monitors) = self.session_monitors.get_mut(alias) {
            monitors.retain(|(name, _)| name != session);
        }
    }

    /// Start monitoring all sessions on a host.
    pub async fn start_monitoring_host(&mut self, alias: &str) -> Result<()> {
        let host = self
            .hosts
            .get(alias)
            .ok_or_else(|| Error::NotFound(format!("host '{}' not registered", alias)))?
            .clone();

        let monitor = host.start_monitoring(None).await?;
        self.full_monitors.insert(alias.to_string(), monitor);
        Ok(())
    }

    /// Stop all monitoring on a specific host.
    pub fn stop_monitoring_host(&mut self, alias: &str) -> Result<()> {
        if !self.hosts.contains_key(alias) {
            return Err(Error::NotFound(format!("host '{}' not registered", alias)));
        }
        // Stop session monitors
        self.session_monitors.remove(alias);
        // Stop full monitor
        self.full_monitors.remove(alias);
        // Tell the host to stop
        if let Some(host) = self.hosts.get(alias) {
            host.stop_monitoring();
        }
        Ok(())
    }

    /// Shutdown the fleet: stop all monitoring and close the bus.
    pub fn shutdown(&mut self) {
        self.session_monitors.clear();
        self.full_monitors.clear();
        for host in self.hosts.values() {
            host.stop_monitoring();
        }
        self.bus.shutdown();
    }

    // --- Session inventory ---

    /// List sessions across all registered hosts, grouped by host alias.
    pub async fn list_sessions_by_host(&self) -> Result<BTreeMap<String, Vec<SessionInfo>>> {
        let mut by_host = BTreeMap::new();
        for (alias, host) in &self.hosts {
            by_host.insert(alias.clone(), host.list_sessions().await?);
        }
        Ok(by_host)
    }

    /// List sessions across all registered hosts with tags under `tag_prefix`.
    pub async fn list_sessions_with_tags(&self, tag_prefix: &str) -> Result<Vec<FleetSessionInfo>> {
        let mut out = Vec::new();
        for (alias, host) in &self.hosts {
            let sessions = host.list_sessions().await?;
            let tags_by_id = host
                .list_tags_for_session_infos(tag_prefix, &sessions)
                .await?;
            for session in sessions {
                let tags = tags_by_id.get(&session.id).cloned().unwrap_or_default();
                out.push(FleetSessionInfo {
                    host_alias: alias.clone(),
                    session,
                    tags,
                });
            }
        }
        Ok(out)
    }

    /// Snapshot sessions across registered hosts with optional tag hydration.
    pub async fn snapshot_sessions(
        &self,
        opts: FleetSnapshotOptions,
    ) -> Result<Vec<FleetSessionSnapshot>> {
        let hosts = self.snapshot_hosts(opts.hosts)?;
        let mut snapshots = Vec::new();

        for (alias, host) in hosts {
            let sessions = host.list_sessions().await?;
            let mut tags_by_prefix = BTreeMap::<String, HashMap<SessionId, Vec<SessionTag>>>::new();

            for prefix in &opts.tag_prefixes {
                tags_by_prefix.insert(
                    prefix.clone(),
                    host.list_tags_for_session_infos(prefix, &sessions).await?,
                );
            }

            for session in sessions {
                let mut tags = BTreeMap::new();
                for prefix in &opts.tag_prefixes {
                    let prefix_tags = tags_by_prefix
                        .get(prefix)
                        .and_then(|by_session| by_session.get(&session.id))
                        .cloned()
                        .unwrap_or_default();
                    tags.insert(prefix.clone(), prefix_tags);
                }

                let target = FleetTargetSpec::session_id(alias.clone(), session.id.as_str())?;
                snapshots.push(FleetSessionSnapshot {
                    host_alias: alias.clone(),
                    session,
                    target,
                    tags,
                });
            }
        }

        Ok(snapshots)
    }

    fn snapshot_hosts(&self, aliases: Option<Vec<String>>) -> Result<Vec<(String, HostHandle)>> {
        match aliases {
            Some(aliases) => aliases
                .into_iter()
                .map(|alias| {
                    let host = self.hosts.get(&alias).ok_or_else(|| {
                        Error::NotFound(format!("host '{}' not registered", alias))
                    })?;
                    Ok((alias, host.clone()))
                })
                .collect(),
            None => Ok(self
                .hosts
                .iter()
                .map(|(alias, host)| (alias.clone(), host.clone()))
                .collect()),
        }
    }

    // --- Target alias bindings ---

    /// Bind an arbitrary alias to a cross-host target.
    pub fn bind_target_alias(&mut self, alias: &str, target: FleetTargetSpec) -> Result<()> {
        if !self.hosts.contains_key(target.host_alias()) {
            return Err(Error::NotFound(format!(
                "cannot bind target alias '{}': host '{}' not registered",
                alias,
                target.host_alias()
            )));
        }
        self.target_aliases
            .insert(alias.to_string(), TargetAliasEntry { target });
        Ok(())
    }

    /// Remove a target alias binding.
    pub fn unbind_target_alias(&mut self, alias: &str) -> Result<()> {
        self.target_aliases
            .remove(alias)
            .map(|_| ())
            .ok_or_else(|| Error::NotFound(format!("target alias '{}' not found", alias)))
    }

    /// Resolve a target alias to a resolved target.
    pub async fn resolve_target_alias(&self, alias: &str) -> Result<Option<ResolvedFleetTarget>> {
        let entry = self
            .target_aliases
            .get(alias)
            .ok_or_else(|| Error::NotFound(format!("target alias '{}' not bound", alias)))?;
        self.resolve_target(&entry.target).await
    }

    /// Resolve a target alias and return `NotFound` if the target is missing.
    pub async fn require_target_alias(&self, alias: &str) -> Result<ResolvedFleetTarget> {
        self.resolve_target_alias(alias)
            .await?
            .ok_or_else(|| Error::NotFound(format!("target alias '{}' target not found", alias)))
    }

    /// Iterator over all target alias names.
    pub fn target_aliases(&self) -> impl Iterator<Item = &str> {
        self.target_aliases.keys().map(|k| k.as_str())
    }

    // --- Timeline helpers ---

    /// Build a `TimelineOptions` whose source-routing filter is derived from a
    /// `FleetTargetSpec`. The remaining timeline options keep their defaults;
    /// callers can adjust them before opening the timeline on the bus.
    pub fn timeline_options_for_spec(&self, spec: &FleetTargetSpec) -> Result<TimelineOptions> {
        if !self.hosts.contains_key(spec.host_alias()) {
            return Err(Error::NotFound(format!(
                "host '{}' not registered",
                spec.host_alias()
            )));
        }

        Ok(TimelineOptions {
            filters: vec![SinkFilter::for_host_session(
                spec.host_alias(),
                spec.session_name(),
            )],
            ..Default::default()
        })
    }

    /// Marker scope for timeline events associated with a target.
    pub fn timeline_marker_scope(&self, spec: &FleetTargetSpec) -> Result<TimelineMarkerScope> {
        if !self.hosts.contains_key(spec.host_alias()) {
            return Err(Error::NotFound(format!(
                "host '{}' not registered",
                spec.host_alias()
            )));
        }
        Ok(match spec.session_id_selector() {
            Some(id) => TimelineMarkerScope::for_host_session_id(spec.host_alias(), id.as_str()),
            None => TimelineMarkerScope::for_host_session(spec.host_alias(), spec.session_name()),
        })
    }

    /// Build timeline options with filters for the provided resolved targets.
    pub fn timeline_options_for_targets(
        &self,
        targets: &[ResolvedFleetTarget],
        mut base: TimelineOptions,
    ) -> TimelineOptions {
        base.filters = targets
            .iter()
            .map(ResolvedFleetTarget::sink_filter)
            .collect();
        base
    }

    // --- Convenience routed actions ---

    /// Send text to a target alias.
    pub async fn send_text(&self, alias: &str, text: &str) -> Result<()> {
        self.require_target_alias(alias)
            .await?
            .target
            .send_text(text)
            .await
    }

    /// Send keys to a target alias.
    pub async fn send_keys(&self, alias: &str, keys: &KeySequence) -> Result<()> {
        self.require_target_alias(alias)
            .await?
            .target
            .send_keys(keys)
            .await
    }

    /// Capture the current content of a target alias.
    pub async fn capture(&self, alias: &str) -> Result<String> {
        self.require_target_alias(alias)
            .await?
            .target
            .capture()
            .await
    }

    /// Resolve a target alias to its `Target` handle.
    pub async fn target(&self, alias: &str) -> Result<Target> {
        Ok(self.require_target_alias(alias).await?.target)
    }

    /// Send text to a target spec without first creating an alias.
    pub async fn send_text_to(&self, spec: &FleetTargetSpec, text: &str) -> Result<()> {
        self.target_by_spec(spec).await?.send_text(text).await
    }

    /// Capture a target spec without first creating an alias.
    pub async fn capture_target(&self, spec: &FleetTargetSpec) -> Result<String> {
        self.target_by_spec(spec).await?.capture().await
    }

    /// Build a fleet target spec from a resolved target.
    pub fn spec_for_target(&self, host_alias: &str, target: &Target) -> Result<FleetTargetSpec> {
        if !self.hosts.contains_key(host_alias) {
            return Err(Error::NotFound(format!(
                "host '{}' not registered",
                host_alias
            )));
        }
        let spec = match target.address() {
            TargetAddress::Session(session) => TargetSpec::session_id(session.id.as_str())?,
            TargetAddress::Window(window) => {
                TargetSpec::session_id(window.session_id.clone())?.window(window.index)
            }
            TargetAddress::Pane(pane) => match pane.session_id.as_ref() {
                Some(id) => TargetSpec::session_id(id.as_str())?
                    .window(pane.window)
                    .pane(pane.pane)?,
                None => TargetSpec::session(&pane.session)
                    .window(pane.window)
                    .pane(pane.pane)?,
            },
        };
        FleetTargetSpec::new(host_alias, spec)
    }
}

impl Default for Fleet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::SinkEvent;
    use crate::types::{SessionTag, TargetSpec};
    use std::time::Duration;

    fn local_host() -> HostHandle {
        HostHandle::local()
    }

    fn local_host_aliased(alias: &str) -> HostHandle {
        HostHandle::with_alias(
            crate::transport::TransportKind::Local(crate::transport::LocalTransport::new()),
            None,
            alias,
        )
    }

    fn mock_host_aliased(alias: &str, mock: crate::transport::MockTransport) -> HostHandle {
        HostHandle::with_alias(crate::transport::TransportKind::Mock(mock), None, alias)
    }

    #[test]
    fn fleet_target_spec_parse_display_roundtrips() {
        let spec: FleetTargetSpec = "web-1::build:0.1".parse().unwrap();

        assert_eq!(spec.host_alias(), "web-1");
        assert_eq!(spec.session_name(), "build");
        assert_eq!(spec.target_spec().to_string(), "build:0.1");
        assert_eq!(spec.to_string(), "web-1::build:0.1");
    }

    #[test]
    fn fleet_target_spec_parse_rejects_invalid_forms() {
        assert!("missing".parse::<FleetTargetSpec>().is_err());
        assert!("::build".parse::<FleetTargetSpec>().is_err());
        assert!("web-1::".parse::<FleetTargetSpec>().is_err());
        assert!("a::b::c".parse::<FleetTargetSpec>().is_err());
    }

    #[tokio::test]
    async fn fleet_resolves_cross_host_target() {
        let mock = crate::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n");
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        let spec = FleetTargetSpec::session("web-1", "build").unwrap();

        let resolved = fleet.require_target(&spec).await.unwrap();

        assert_eq!(resolved.spec, spec);
        assert_eq!(resolved.host.host_alias(), "web-1");
        assert_eq!(resolved.target.session_name(), "build");
    }

    #[test]
    fn fleet_unregister_removes_host_and_target_aliases() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        fleet
            .bind_target_alias(
                "builder",
                FleetTargetSpec::session("web-1", "build").unwrap(),
            )
            .unwrap();

        let removed = fleet.unregister("web-1").unwrap();

        assert_eq!(removed.host_alias(), "web-1");
        assert!(fleet.host("web-1").is_none());
        assert_eq!(fleet.target_aliases().count(), 0);
        assert!(fleet.unregister("web-1").is_err());
    }

    #[tokio::test]
    async fn resolved_target_timeline_filter_matches_host_session_only() {
        let mock = crate::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n");
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        let spec = FleetTargetSpec::session("web-1", "build").unwrap();
        let resolved = fleet.require_target(&spec).await.unwrap();

        let options = fleet.timeline_options_for_targets(
            std::slice::from_ref(&resolved),
            TimelineOptions::default(),
        );
        let filter = crate::sink::CompiledSinkFilter::compile(&options.filters[0]).unwrap();
        let mut output = crate::sink::TargetOutput {
            source: resolved.target.address().clone(),
            host: "web-1".to_string(),
            content: "ok".to_string(),
            raw_content: None,
            sequence: 1,
            fidelity: crate::types::OutputFidelity::default(),
            timestamp: std::time::Instant::now(),
        };

        assert!(filter.matches(&output));
        output.host = "db-1".to_string();
        assert!(!filter.matches(&output));
    }

    #[tokio::test]
    async fn fleet_timeline_helpers_for_id_spec_match_live_name_output_and_markers() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        let spec = FleetTargetSpec::session_id("web-1", "$1").unwrap();
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline("id-target", fleet.timeline_options_for_spec(&spec).unwrap())
            .unwrap();

        bus.publish(crate::sink::TargetOutput {
            source: TargetAddress::Pane(crate::types::PaneAddress {
                pane_id: "%5".to_string(),
                session_id: Some(SessionId::new("$1").unwrap()),
                session: "build".to_string(),
                window: 0,
                pane: 0,
            }),
            host: "web-1".to_string(),
            content: "live output\n".to_string(),
            raw_content: None,
            sequence: 1,
            fidelity: crate::types::OutputFidelity::default(),
            timestamp: std::time::Instant::now(),
        });
        bus.publish_discontinuity_for(
            TimelineMarkerScope::for_host_session_identity("web-1", "build", "$1"),
            "live marker",
        );
        bus.publish_discontinuity_for(fleet.timeline_marker_scope(&spec).unwrap(), "manual marker");

        let page = timeline
            .render_after(
                crate::sink::TimelineCursor::default(),
                crate::sink::RenderOptions::default(),
            )
            .await
            .unwrap();

        assert!(page.text.contains("live output"), "{}", page.text);
        assert!(page.text.contains("live marker"), "{}", page.text);
        assert!(page.text.contains("manual marker"), "{}", page.text);
    }

    #[tokio::test]
    async fn fleet_resolve_target_returns_none_for_missing_tmux_target() {
        let mock = crate::transport::MockTransport::new().with_response("list-sessions", "");
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        let spec = FleetTargetSpec::session("web-1", "missing").unwrap();

        assert!(fleet.resolve_target(&spec).await.unwrap().is_none());
        assert!(fleet.require_target(&spec).await.is_err());
    }

    #[test]
    fn fleet_register_and_lookup() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        fleet.register("db-1", local_host_aliased("db-1")).unwrap();

        assert_eq!(fleet.host_count(), 2);
        assert!(fleet.host("web-1").is_some());
        assert!(fleet.host("db-1").is_some());
        assert!(fleet.host("missing").is_none());
    }

    #[test]
    fn fleet_alias_conflict_detection() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        let err = fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn fleet_alias_mismatch_rejected() {
        let mut fleet = Fleet::new();
        let err = fleet.register("web-1", local_host()).unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn fleet_hosts_iterator() {
        let mut fleet = Fleet::new();
        fleet.register("a", local_host_aliased("a")).unwrap();
        fleet.register("b", local_host_aliased("b")).unwrap();

        let mut aliases: Vec<&str> = fleet.hosts().map(|(k, _)| k).collect();
        aliases.sort();
        assert_eq!(aliases, vec!["a", "b"]);
    }

    #[test]
    fn fleet_target_alias_bind_find_unbind() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();

        let spec = FleetTargetSpec::session("web-1", "build").unwrap();
        fleet.bind_target_alias("primary", spec.clone()).unwrap();

        let aliases: Vec<&str> = fleet.target_aliases().collect();
        assert_eq!(aliases, vec!["primary"]);
        assert_eq!(fleet.target_aliases.get("primary").unwrap().target, spec);

        fleet.unbind_target_alias("primary").unwrap();
        assert!(fleet.target_aliases().next().is_none());
    }

    #[test]
    fn fleet_timeline_options_are_scoped_to_host_session() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        let spec = FleetTargetSpec::session("web-1", "build").unwrap();

        let options = fleet.timeline_options_for_spec(&spec).unwrap();
        assert_eq!(options.filters.len(), 1);
        assert_eq!(options.filters[0].host.as_deref(), Some("^web\\-1$"));
        assert_eq!(options.filters[0].session.as_deref(), Some("^build$"));

        let scope = fleet.timeline_marker_scope(&spec).unwrap();
        assert_eq!(scope.host.as_deref(), Some("web-1"));
        assert_eq!(scope.session.as_deref(), Some("build"));
    }

    #[test]
    fn fleet_monitor_status_rejects_non_session_targets() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        let spec = FleetTargetSpec::new("web-1", TargetSpec::session("build").window(0)).unwrap();

        let err = fleet.stop_monitoring_session_target(&spec).unwrap_err();

        assert!(matches!(err, Error::UnsupportedTarget { .. }));
        assert!(fleet.monitor_status_for_target(&spec).is_none());
    }

    #[tokio::test]
    async fn fleet_stop_monitoring_session_target_removes_full_monitor_entry() {
        let mock = crate::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n")
            .with_shell_data(vec![b"%output %5 active\n".to_vec()]);
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        let spec = FleetTargetSpec::session("web-1", "build").unwrap();

        fleet.start_monitoring_host("web-1").await.unwrap();
        assert!(fleet.monitor_status_for_target(&spec).is_some());

        fleet.stop_monitoring_session_target(&spec).unwrap();

        assert!(fleet.monitor_status_for_target(&spec).is_none());
        assert!(fleet
            .full_monitors
            .get("web-1")
            .is_none_or(|monitor| monitor.get("build").is_none()));
        assert!(matches!(
            fleet.host_status("web-1"),
            Some(HostStatus::Connected)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn fleet_monitor_recovery_publishes_resume_and_snapshot_markers() {
        let mock = crate::transport::MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n")
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n")
            .with_response("list-sessions", "build $1 100 0 1  200\n")
            .with_response("list-panes", "%5 $1 build 0 0 title bash 123 80 24 1\n")
            .with_response("capture-pane", "snapshot\n")
            .with_shell_sequence(vec![b"%output %5 before\n".to_vec()])
            .with_shell_sequence(vec![b"%output %5 after\n".to_vec()]);
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        let mut rx = fleet
            .output_bus()
            .subscribe(vec![], 16)
            .unwrap()
            .into_receiver();

        fleet
            .start_monitoring_session("web-1", "build")
            .await
            .unwrap();

        match rx.recv().await.unwrap() {
            SinkEvent::Data(output) => assert_eq!(output.content, "before"),
            other => panic!("expected initial data, got {:?}", other),
        }
        match rx.recv().await.unwrap() {
            SinkEvent::Discontinuity { reason } => {
                assert!(reason.contains("stream interrupted"));
            }
            other => panic!("expected interrupt marker, got {:?}", other),
        }

        tokio::time::advance(Duration::from_secs(1)).await;
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }

        let mut saw_resume = false;
        let mut saw_snapshot_marker = false;
        let mut saw_post_reconnect_output = false;
        let mut observed = Vec::new();
        for _ in 0..5 {
            match rx.recv().await.unwrap() {
                SinkEvent::Data(output) if output.content == "after" => {
                    observed.push(format!("data:{}", output.content));
                    saw_post_reconnect_output = true;
                }
                SinkEvent::Data(output) => {
                    observed.push(format!("data:{}", output.content));
                }
                SinkEvent::Discontinuity { reason } => {
                    observed.push(format!("discontinuity:{reason}"));
                    saw_resume |= reason.contains("stream resumed");
                    saw_snapshot_marker |= reason.contains("stream snapshot");
                }
                SinkEvent::Gap { dropped, .. } => {
                    observed.push(format!("gap:{dropped}"));
                }
            }
        }

        assert!(
            saw_resume,
            "expected reconnect resume marker, observed {observed:?}"
        );
        assert!(
            saw_snapshot_marker,
            "expected reconnect snapshot marker, observed {observed:?}"
        );
        assert!(
            saw_post_reconnect_output,
            "expected output from reopened control channel, observed {observed:?}"
        );

        let spec = FleetTargetSpec::session("web-1", "build").unwrap();
        fleet.stop_monitoring_session_target(&spec).unwrap();
    }

    #[tokio::test]
    async fn fleet_lists_sessions_with_tags_across_hosts() {
        let mock = crate::transport::MockTransport::new()
            .with_response("__MOTLIE_TAGS__", "__MOTLIE_TAGS__ $1\n@app/role worker\n")
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n");
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();

        let sessions = fleet.list_sessions_with_tags("app").await.unwrap();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].host_alias, "web-1");
        assert_eq!(sessions[0].session.name, "build");
        assert_eq!(
            sessions[0].tags,
            vec![SessionTag::new("app", "role", "worker").unwrap()]
        );
    }

    #[tokio::test]
    async fn fleet_snapshot_sessions_filters_hosts_and_hydrates_tags() {
        let mock = crate::transport::MockTransport::new()
            .with_response("__MOTLIE_TAGS__", "__MOTLIE_TAGS__ $1\n@app/role worker\n")
            .with_response("list-sessions", "__MOTLIE_S__ build $1 100 0 1  200\n");
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", mock_host_aliased("web-1", mock))
            .unwrap();
        fleet.register("db-1", local_host_aliased("db-1")).unwrap();

        let snapshots = fleet
            .snapshot_sessions(FleetSnapshotOptions {
                hosts: Some(vec!["web-1".to_string()]),
                tag_prefixes: vec!["app".to_string()],
            })
            .await
            .unwrap();

        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].host_alias, "web-1");
        assert_eq!(
            snapshots[0].target,
            FleetTargetSpec::session_id("web-1", "$1").unwrap()
        );
        assert_eq!(
            snapshots[0].tags.get("app").unwrap(),
            &vec![SessionTag::new("app", "role", "worker").unwrap()]
        );

        let err = fleet
            .snapshot_sessions(FleetSnapshotOptions {
                hosts: Some(vec!["missing".to_string()]),
                tag_prefixes: vec![],
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not registered"));
    }

    #[test]
    fn fleet_bind_target_alias_rejects_unknown_host() {
        let mut fleet = Fleet::new();
        let err = fleet
            .bind_target_alias(
                "ci",
                FleetTargetSpec::session("nonexistent", "build").unwrap(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("not registered"));
    }

    #[test]
    fn fleet_unbind_target_alias_rejects_unknown_alias() {
        let mut fleet = Fleet::new();
        let err = fleet.unbind_target_alias("nonexistent").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn fleet_host_status_connected() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        match fleet.host_status("web-1") {
            Some(HostStatus::Connected) => {}
            other => panic!("expected Connected, got {:?}", other),
        }
    }

    #[test]
    fn fleet_host_status_missing() {
        let fleet = Fleet::new();
        assert!(fleet.host_status("missing").is_none());
    }

    #[test]
    fn fleet_output_bus_is_shared() {
        let fleet = Fleet::new();
        let bus1 = fleet.output_bus();
        let bus2 = fleet.output_bus();
        assert!(Arc::ptr_eq(&bus1, &bus2));
    }

    #[test]
    fn fleet_shutdown_clears_state() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();
        fleet
            .bind_target_alias("ci", FleetTargetSpec::session("web-1", "build").unwrap())
            .unwrap();

        fleet.shutdown();
        // Bus should be shutdown
        assert_eq!(fleet.output_bus().subscriber_count(), 0);
    }

    #[test]
    fn fleet_bus_injected_into_hosts() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();

        // The host's output_bus() should return the fleet's shared bus
        let fleet_bus = fleet.output_bus();
        let host_bus = fleet.host("web-1").unwrap().output_bus();
        assert!(Arc::ptr_eq(&fleet_bus, &host_bus));
    }
}
