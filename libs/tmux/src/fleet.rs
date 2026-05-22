//! Fleet coordination and routing (DC27, DC9).
//!
//! `Fleet` is a programmatic registry of `HostHandle`s with convenience routing.
//! It owns an `OutputBus`, manages per-host monitoring lifecycle, and provides
//! workstream bindings for alias-based targeting.
//!
//! Fleet-level routing helpers are convenience wrappers over normal
//! `HostHandle` / `Target` operations, not a separate action system.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};

use crate::host::{HostHandle, Target};
use crate::keys::KeySequence;
use crate::monitor::{MonitorHandle, MonitorHealth, SessionMonitorHandle};
use crate::sink::{OutputBus, TimelineHandle, TimelineOptions};
use crate::types::TargetSpec;

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
// WorkstreamEntry
// ---------------------------------------------------------------------------

/// A named binding from a workstream alias to a resolved target.
#[derive(Debug, Clone)]
struct WorkstreamEntry {
    host_alias: String,
    target_spec: TargetSpec,
}

// ---------------------------------------------------------------------------
// Fleet
// ---------------------------------------------------------------------------

/// Multi-host coordination registry (DC27).
///
/// Provides programmatic host registration, aggregate monitoring lifecycle,
/// workstream-based routing, and a shared `OutputBus` that aggregates output
/// from all registered hosts.
pub struct Fleet {
    hosts: HashMap<String, HostHandle>,
    bus: Arc<OutputBus>,
    workstreams: HashMap<String, WorkstreamEntry>,
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
            workstreams: HashMap::new(),
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

    /// Create a named timeline on the shared `OutputBus`.
    pub fn create_timeline(
        &self,
        name: impl Into<String>,
        opts: TimelineOptions,
    ) -> Result<TimelineHandle> {
        self.bus.create_timeline(name, opts)
    }

    /// Return an existing timeline or create it on the shared `OutputBus`.
    pub fn create_or_get_timeline(
        &self,
        name: impl Into<String>,
        opts: TimelineOptions,
    ) -> Result<TimelineHandle> {
        self.bus.create_or_get_timeline(name, opts)
    }

    /// Look up a named timeline on the shared `OutputBus`.
    pub fn timeline(&self, name: &str) -> Result<Option<TimelineHandle>> {
        self.bus.timeline(name)
    }

    /// Remove a named timeline from the shared `OutputBus`.
    pub fn remove_timeline(&self, name: &str) -> Result<()> {
        self.bus.remove_timeline(name)
    }

    /// Remove idle timelines from the shared `OutputBus`.
    pub fn remove_idle_timelines(&self, idle_for: std::time::Duration) -> Result<Vec<String>> {
        self.bus.remove_idle_timelines(idle_for)
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
            for (name, handle) in monitors {
                statuses.push(SessionMonitorStatus {
                    name: name.clone(),
                    health: handle.health(),
                });
            }
        }
        // Full monitors (aggregate handle) — use all_sessions() so that
        // failed/stopped sessions remain visible in health status (DC29).
        if let Some(monitor) = self.full_monitors.get(alias) {
            for session_name in monitor.all_sessions() {
                if let Some(handle) = monitor.get(session_name) {
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

    // --- Monitoring lifecycle ---

    /// Start monitoring a specific session on a host.
    pub async fn start_monitoring_session(&mut self, alias: &str, session: &str) -> Result<()> {
        let host = self
            .hosts
            .get(alias)
            .ok_or_else(|| Error::NotFound(format!("host '{}' not registered", alias)))?
            .clone();

        let monitor = host.start_monitoring_session(session).await?;
        self.session_monitors
            .entry(alias.to_string())
            .or_default()
            .push((session.to_string(), monitor));
        Ok(())
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

    // --- Workstream bindings ---

    /// Bind a workstream name to a host alias + target spec.
    pub fn bind(&mut self, workstream: &str, host_alias: &str, target: TargetSpec) -> Result<()> {
        if !self.hosts.contains_key(host_alias) {
            return Err(Error::NotFound(format!(
                "cannot bind workstream '{}': host '{}' not registered",
                workstream, host_alias
            )));
        }
        self.workstreams.insert(
            workstream.to_string(),
            WorkstreamEntry {
                host_alias: host_alias.to_string(),
                target_spec: target,
            },
        );
        Ok(())
    }

    /// Remove a workstream binding.
    pub fn unbind(&mut self, workstream: &str) -> Result<()> {
        self.workstreams
            .remove(workstream)
            .map(|_| ())
            .ok_or_else(|| Error::NotFound(format!("workstream '{}' not found", workstream)))
    }

    /// Resolve a workstream name to a `Target`. Returns `None` if the
    /// workstream is not bound or the target cannot be resolved.
    pub async fn find(&self, workstream: &str) -> Result<Option<Target>> {
        let entry = self
            .workstreams
            .get(workstream)
            .ok_or_else(|| Error::NotFound(format!("workstream '{}' not bound", workstream)))?;
        let host = self.hosts.get(&entry.host_alias).ok_or_else(|| {
            Error::NotFound(format!("host '{}' not registered", entry.host_alias))
        })?;
        host.target(&entry.target_spec).await
    }

    /// Iterator over all workstream names.
    pub fn workstreams(&self) -> impl Iterator<Item = &str> {
        self.workstreams.keys().map(|k| k.as_str())
    }

    // --- Convenience routed actions ---

    /// Send text to a workstream target.
    pub async fn send_text(&self, workstream: &str, text: &str) -> Result<()> {
        let target = self.find(workstream).await?.ok_or_else(|| {
            Error::NotFound(format!("workstream '{}' target not found", workstream))
        })?;
        target.send_text(text).await
    }

    /// Send keys to a workstream target.
    pub async fn send_keys(&self, workstream: &str, keys: &KeySequence) -> Result<()> {
        let target = self.find(workstream).await?.ok_or_else(|| {
            Error::NotFound(format!("workstream '{}' target not found", workstream))
        })?;
        target.send_keys(keys).await
    }

    /// Capture the current content of a workstream target.
    pub async fn capture(&self, workstream: &str) -> Result<String> {
        let target = self.find(workstream).await?.ok_or_else(|| {
            Error::NotFound(format!("workstream '{}' target not found", workstream))
        })?;
        target.capture().await
    }

    /// Resolve a workstream to its `Target` handle.
    pub async fn target(&self, workstream: &str) -> Result<Target> {
        self.find(workstream)
            .await?
            .ok_or_else(|| Error::NotFound(format!("workstream '{}' target not found", workstream)))
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
    use crate::types::TargetSpec;

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
    fn fleet_workstream_bind_find_unbind() {
        let mut fleet = Fleet::new();
        fleet
            .register("web-1", local_host_aliased("web-1"))
            .unwrap();

        let spec = TargetSpec::session("build");
        fleet.bind("ci", "web-1", spec).unwrap();

        let mut ws: Vec<&str> = fleet.workstreams().collect();
        ws.sort();
        assert_eq!(ws, vec!["ci"]);

        fleet.unbind("ci").unwrap();
        assert!(fleet.workstreams().next().is_none());
    }

    #[test]
    fn fleet_bind_rejects_unknown_host() {
        let mut fleet = Fleet::new();
        let err = fleet
            .bind("ci", "nonexistent", TargetSpec::session("build"))
            .unwrap_err();
        assert!(err.to_string().contains("not registered"));
    }

    #[test]
    fn fleet_unbind_rejects_unknown_workstream() {
        let mut fleet = Fleet::new();
        let err = fleet.unbind("nonexistent").unwrap_err();
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
            .bind("ci", "web-1", TargetSpec::session("build"))
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
