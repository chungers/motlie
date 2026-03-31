use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use tokio::time::Duration;

use crate::capture;
use crate::control;
use crate::discovery;
use crate::keys::KeySequence;
use crate::monitor::{
    MonitorExitReason, MonitorHandle, MonitorHealth, SessionMonitor, SessionMonitorHandle,
};
use crate::sink::{OutputBus, TargetOutput};
use crate::transport::TransportKind;
use crate::types::*;

/// Internal state shared via Arc between HostHandle and Target.
struct HostHandleInner {
    transport: TransportKind,
    socket: Option<TmuxSocket>,
    /// Per-pane exec locks keyed by resolved `pane_id` (DC19).
    exec_locks: std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Host alias for TargetOutput.host field (e.g. "localhost", "web-1").
    host_alias: String,
    /// Shared output bus (DC24). Lazily created on first `start_monitoring`.
    output_bus: std::sync::Mutex<Option<Arc<OutputBus>>>,
    /// Per-host tracking of active monitor stop signals (DC13).
    /// Keyed by session name. The `watch::Sender<bool>` fires the stop signal;
    /// the actual `JoinHandle` is owned by the returned `SessionMonitorHandle`.
    monitor_signals: std::sync::Mutex<HashMap<String, tokio::sync::watch::Sender<bool>>>,
    /// Per-pane active exec state tracking (DC31).
    /// Keyed by `session_name:pane_id` for session-scoped discontinuity invalidation.
    /// Each entry holds Arc refs to ExecState for running execs.
    active_execs: std::sync::Mutex<HashMap<String, Vec<Arc<std::sync::Mutex<ExecState>>>>>,
    /// Resolved tmux binary path (e.g. "/opt/homebrew/bin/tmux").
    /// Set once after connection via `resolve_tmux_bin()`.
    tmux_bin: std::sync::Mutex<Option<String>>,
}

impl HostHandleInner {
    /// Resolve the tmux binary path on the remote host.
    /// Uses `command -v tmux` through a login shell to get the full PATH.
    /// Caches the result. Falls back to bare `"tmux"` on localhost/mock.
    async fn resolve_tmux_bin(&self) -> String {
        {
            let cached = self.tmux_bin.lock().expect("tmux_bin lock poisoned");
            if let Some(ref bin) = *cached {
                return bin.clone();
            }
        }

        // Use login shell to get the user's full PATH.
        // `command -v` returns exit code 1 when not found (unlike `which`
        // on macOS zsh which returns "not found" text with exit 0).
        // $SHELL is set by sshd from the user's passwd entry, so
        // `$SHELL -lc` works regardless of which shell the user has.
        let probes: &[&str] = &[
            "command -v tmux",
            "$SHELL -lc 'command -v tmux' 2>/dev/null",
        ];

        let mut resolved = "tmux".to_string();
        for probe in probes {
            if let Ok(out) = self.transport.exec(probe).await {
                let out = out.trim().to_string();
                if !out.is_empty() && out.starts_with('/') {
                    resolved = out;
                    break;
                }
            }
        }

        let mut cached = self.tmux_bin.lock().expect("tmux_bin lock poisoned");
        *cached = Some(resolved.clone());
        resolved
    }

    /// Build the tmux command prefix using the resolved binary path.
    async fn tmux_prefix(&self) -> String {
        let bin = self.resolve_tmux_bin().await;
        crate::transport::tmux_prefix_with_bin(&bin, self.socket.as_ref())
    }
    /// Get or create an exec lock for a given pane identity key.
    fn exec_lock(&self, key: &str) -> Arc<Mutex<()>> {
        let mut locks = self.exec_locks.lock().expect("exec_locks poisoned");
        locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Transition Running exec states to Unknown for a specific session (DC31).
    ///
    /// Called on connection loss / discontinuity. Only affects execs whose
    /// `active_execs` key starts with `session_name:`, preserving execs in
    /// unrelated sessions. Leaves Completed and already-Unknown states unchanged.
    fn notify_exec_discontinuity(&self, session_name: &str, reason: &str) {
        let prefix = format!("{}:", session_name);
        let active = self.active_execs.lock().expect("active_execs poisoned");
        for (key, states) in active.iter() {
            if !key.starts_with(&prefix) {
                continue;
            }
            for state_arc in states {
                let mut state = state_arc.lock().expect("exec state poisoned");
                if matches!(*state, ExecState::Running) {
                    *state = ExecState::Unknown {
                        reason: reason.to_string(),
                    };
                }
            }
        }
    }
}

/// Handle to a tmux host (local or remote).
#[derive(Clone)]
pub struct HostHandle {
    inner: Arc<HostHandleInner>,
}

impl HostHandle {
    /// Create a HostHandle for localhost.
    pub fn local() -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner {
                transport: TransportKind::Local(crate::transport::LocalTransport::new()),
                socket: None,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
                host_alias: "localhost".to_string(),
                output_bus: std::sync::Mutex::new(None),
                monitor_signals: std::sync::Mutex::new(HashMap::new()),
                active_execs: std::sync::Mutex::new(HashMap::new()),
                tmux_bin: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Create a HostHandle for localhost with a custom transport timeout.
    ///
    /// The default `local()` uses a 10-second timeout. Use this when
    /// commands may take longer (e.g., slow CI machines) or when you
    /// need a shorter timeout for responsiveness.
    pub fn local_with_timeout(timeout: Duration) -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner {
                transport: TransportKind::Local(crate::transport::LocalTransport::with_timeout(
                    timeout,
                )),
                socket: None,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
                host_alias: "localhost".to_string(),
                output_bus: std::sync::Mutex::new(None),
                monitor_signals: std::sync::Mutex::new(HashMap::new()),
                active_execs: std::sync::Mutex::new(HashMap::new()),
                tmux_bin: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Create a HostHandle with a specific transport and socket.
    pub fn new(transport: TransportKind, socket: Option<TmuxSocket>) -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner {
                transport,
                socket,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
                host_alias: "localhost".to_string(),
                output_bus: std::sync::Mutex::new(None),
                monitor_signals: std::sync::Mutex::new(HashMap::new()),
                active_execs: std::sync::Mutex::new(HashMap::new()),
                tmux_bin: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Create a HostHandle with a specific transport, socket, and host alias.
    pub fn with_alias(transport: TransportKind, socket: Option<TmuxSocket>, alias: &str) -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner {
                transport,
                socket,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
                host_alias: alias.to_string(),
                output_bus: std::sync::Mutex::new(None),
                monitor_signals: std::sync::Mutex::new(HashMap::new()),
                active_execs: std::sync::Mutex::new(HashMap::new()),
                tmux_bin: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Access the transport kind for crate tests.
    ///
    /// Kept test-only so the transport abstraction stays internal per DC21
    /// while still allowing URI/connect tests to assert localhost transport
    /// selection without relying on behavioral side effects.
    #[cfg(test)]
    pub(crate) fn transport_kind(&self) -> &TransportKind {
        &self.inner.transport
    }

    /// Ensure the tmux server is running for this host's configured socket (DC30).
    ///
    /// Runs `tmux start-server` (with socket args if configured). Idempotent —
    /// no-op if the server is already running. Call this before creating sessions
    /// on a dedicated automation socket to ensure the server exists.
    pub async fn ensure_socket_server(&self) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        let cmd = format!("{} start-server", prefix);
        self.inner.transport.exec(&cmd).await?;
        Ok(())
    }

    /// List all tmux sessions on this host.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let prefix = self.inner.tmux_prefix().await;
        discovery::list_sessions_with_prefix(&self.inner.transport, &prefix).await
    }

    /// List attached clients on this host (DC20, Phase 1.9b).
    pub async fn list_clients(&self) -> Result<Vec<ClientInfo>> {
        let prefix = self.inner.tmux_prefix().await;
        discovery::list_clients_with_prefix(&self.inner.transport, &prefix).await
    }

    /// Set global `history-limit` (DC20, Phase 1.9b).
    ///
    /// **Must be set before creating sessions/panes.** Existing panes
    /// retain their creation-time limit.
    pub async fn set_global_history_limit(&self, limit: u32) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        control::set_history_limit_with_prefix(&self.inner.transport, &prefix, None, limit).await
    }

    /// Query the global `history-limit` default.
    pub async fn get_global_history_limit(&self) -> Result<u32> {
        let prefix = self.inner.tmux_prefix().await;
        control::get_history_limit_with_prefix(&self.inner.transport, &prefix, None).await
    }

    /// Create a new tmux session. Returns a Target at session level (DC22).
    ///
    /// Use `CreateSessionOptions` to set window size, history limit, etc.
    /// `CreateSessionOptions::default()` preserves pre-DC22 behavior.
    pub async fn create_session(&self, name: &str, opts: &CreateSessionOptions) -> Result<Target> {
        let prefix = self.inner.tmux_prefix().await;
        control::create_session_with_prefix(&self.inner.transport, &prefix, name, opts).await?;

        // Query the created session to get full info
        let sessions = self.list_sessions().await?;
        let info = sessions
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow!("session '{}' created but not found in list", name))?;

        Ok(Target {
            inner: self.inner.clone(),
            address: TargetAddress::Session(info),
        })
    }

    /// Get a Target for an existing session by name.
    pub async fn session(&self, name: &str) -> Result<Option<Target>> {
        let sessions = self.list_sessions().await?;
        Ok(sessions
            .into_iter()
            .find(|s| s.name == name)
            .map(|info| Target {
                inner: self.inner.clone(),
                address: TargetAddress::Session(info),
            }))
    }

    /// Get a Target from a TargetSpec. Verifies the entity exists.
    pub async fn target(&self, spec: &TargetSpec) -> Result<Option<Target>> {
        let prefix = self.inner.tmux_prefix().await;
        // Check session exists
        let sessions = self.list_sessions().await?;
        let session_info = match sessions.into_iter().find(|s| s.name == spec.session_name()) {
            Some(s) => s,
            None => return Ok(None),
        };

        match (spec.window_selector(), spec.pane_index()) {
            (None, Some(_)) => {
                return Err(anyhow!(
                    "invalid TargetSpec: pane requires window (got session='{}', window=None, pane=Some)",
                    spec.session_name()
                ));
            }
            (None, None) => Ok(Some(Target {
                inner: self.inner.clone(),
                address: TargetAddress::Session(session_info),
            })),
            (Some(window_str), None) => {
                let windows = discovery::list_windows_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    spec.session_name(),
                )
                .await?;
                let win = windows
                    .into_iter()
                    .find(|w| w.index.to_string() == *window_str || w.name == *window_str);
                Ok(win.map(|w| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Window(w),
                }))
            }
            (Some(window_str), Some(pane_idx)) => {
                // Resolve window by index or name → actual window index
                let windows = discovery::list_windows_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    spec.session_name(),
                )
                .await?;
                let resolved_window = windows
                    .into_iter()
                    .find(|w| w.index.to_string() == *window_str || w.name == *window_str);
                let window_index = match resolved_window {
                    Some(w) => w.index,
                    None => return Ok(None),
                };

                let panes = discovery::list_panes_in_session_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    spec.session_name(),
                )
                .await?;
                let pane = panes
                    .into_iter()
                    .find(|p| p.address.window == window_index && p.address.pane == pane_idx);
                Ok(pane.map(|p| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Pane(p.address),
                }))
            }
        }
    }

    /// Upload a file or directory to the host (DC23).
    ///
    /// For SSH-backed hosts, uses SFTP. For localhost, uses filesystem copy.
    /// Directory placement follows `cp -r` semantics. See `TransferOptions`
    /// for overwrite and recursive behavior.
    pub async fn upload(
        &self,
        local_path: &std::path::Path,
        remote_path: &std::path::Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        self.inner
            .transport
            .upload(local_path, remote_path, opts)
            .await
    }

    /// Download a file or directory from the host (DC23).
    ///
    /// For SSH-backed hosts, uses SFTP. For localhost, uses filesystem copy.
    /// Directory placement follows `cp -r` semantics. See `TransferOptions`
    /// for overwrite and recursive behavior.
    pub async fn download(
        &self,
        remote_path: &std::path::Path,
        local_path: &std::path::Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        self.inner
            .transport
            .download(remote_path, local_path, opts)
            .await
    }

    // --- Monitoring lifecycle (2a.4a, DC13, DC24) ---

    /// Get or create the shared OutputBus for this host.
    pub fn output_bus(&self) -> Arc<OutputBus> {
        let mut bus_opt = self
            .inner
            .output_bus
            .lock()
            .expect("output_bus lock poisoned");
        if let Some(bus) = bus_opt.as_ref() {
            return bus.clone();
        }
        let bus = Arc::new(OutputBus::new());
        *bus_opt = Some(bus.clone());
        bus
    }

    /// Inject a shared OutputBus before monitoring starts.
    ///
    /// Used by `Fleet` to wire all hosts to a single aggregation bus (DC27).
    /// Must be called before `output_bus()` or any `start_monitoring*` method;
    /// returns an error if a bus has already been created.
    pub fn inject_output_bus(&self, bus: Arc<OutputBus>) -> Result<()> {
        let mut bus_opt = self
            .inner
            .output_bus
            .lock()
            .expect("output_bus lock poisoned");
        if bus_opt.is_some() {
            return Err(anyhow!(
                "output bus already initialized for host '{}'; inject before first use",
                self.inner.host_alias
            ));
        }
        *bus_opt = Some(bus);
        Ok(())
    }

    /// Start monitoring a single session with reconnect supervision (DC29, 4.2a).
    ///
    /// Opens a control mode connection, parses `%output` frames, and publishes
    /// to the OutputBus. On unexpected EOF, the monitor attempts bounded
    /// reconnect with exponential backoff. Intentional stop (via shutdown) does
    /// not trigger reconnect.
    ///
    /// Returns a `SessionMonitorHandle` for lifecycle control and health inspection.
    pub async fn start_monitoring_session(
        &self,
        session_name: &str,
    ) -> Result<SessionMonitorHandle> {
        let bus = self.output_bus();
        let host_alias = self.inner.host_alias.clone();
        let session = session_name.to_string();

        // Resolve the session first — fail before any registration
        let sessions = self.list_sessions().await?;
        let session_info = sessions
            .into_iter()
            .find(|s| s.name == session)
            .ok_or_else(|| anyhow!("session '{}' not found", session))?;
        let target = Target {
            inner: self.inner.clone(),
            address: TargetAddress::Session(session_info),
        };

        // Open the initial shell channel for control mode
        let shell = self.inner.transport.open_shell(80, 24).await?;

        // Create stop signal and register for host-level tracking (DC13).
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        {
            let mut signals = self
                .inner
                .monitor_signals
                .lock()
                .expect("monitor_signals lock poisoned");
            signals.insert(session_name.to_string(), stop_tx.clone());
        }

        // Shared health state (DC29, 4.2d)
        let health = Arc::new(std::sync::Mutex::new(MonitorHealth::Streaming));
        let health_task = health.clone();
        let (startup_ready_tx, startup_ready_rx) = oneshot::channel();

        // Resolve tmux binary once before spawning the monitor task
        let resolved_tmux_bin = self.inner.resolve_tmux_bin().await;

        // Spawn the supervised monitor task (4.2a)
        let inner_ref = self.inner.clone();
        let session_for_cleanup = session_name.to_string();
        let task = tokio::spawn(async move {
            let max_retries = 5u32;
            let mut attempt = 0u32;
            let mut shell = shell;
            let mut startup_ready = Some(startup_ready_tx);

            let set_health = |h: MonitorHealth| {
                *health_task.lock().expect("health lock poisoned") = h;
            };

            loop {
                let mut monitor = SessionMonitor::new(session.clone(), host_alias.clone())
                    .with_socket(inner_ref.socket.clone())
                    .with_tmux_bin(Some(resolved_tmux_bin.clone()));

                let exit = monitor
                    .run(&mut shell, &bus, stop_rx.clone(), &mut startup_ready)
                    .await?;

                match exit {
                    MonitorExitReason::Stopped => {
                        set_health(MonitorHealth::Stopped);
                        if let Ok(mut signals) = inner_ref.monitor_signals.lock() {
                            signals.remove(&session_for_cleanup);
                        }
                        return Ok(MonitorExitReason::Stopped);
                    }
                    MonitorExitReason::ConnectionLost => {
                        // Transition Running execs in this session to Unknown (DC31)
                        inner_ref.notify_exec_discontinuity(
                            &session,
                            &format!("connection lost for {}:{}", host_alias, session),
                        );

                        // Unexpected EOF — emit discontinuity and attempt reconnect
                        bus.publish_discontinuity(&format!(
                            "stream interrupted: control channel lost for {}:{}",
                            host_alias, session
                        ));
                        set_health(MonitorHealth::Reconnecting);

                        attempt += 1;
                        if attempt > max_retries {
                            set_health(MonitorHealth::Failed);
                            if let Ok(mut signals) = inner_ref.monitor_signals.lock() {
                                signals.remove(&session_for_cleanup);
                            }
                            return Err(anyhow!(
                                "monitor for '{}' exhausted {} reconnect attempts",
                                session,
                                max_retries
                            ));
                        }

                        // Exponential backoff: 1s, 2s, 4s, 8s, 16s (capped at 30s)
                        let delay_ms = std::cmp::min(
                            1000u64.saturating_mul(2u64.saturating_pow(attempt - 1)),
                            30_000,
                        );
                        let mut stop_clone = stop_rx.clone();
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                            _ = stop_clone.changed() => {
                                if *stop_rx.borrow() {
                                    set_health(MonitorHealth::Stopped);
                                    if let Ok(mut signals) = inner_ref.monitor_signals.lock() {
                                        signals.remove(&session_for_cleanup);
                                    }
                                    return Ok(MonitorExitReason::Stopped);
                                }
                            }
                        }

                        // Check session still exists before reconnecting
                        let reconnect_prefix = inner_ref.tmux_prefix().await;
                        let reconnect_cmd = format!(
                            "{} list-sessions -F '{}'",
                            reconnect_prefix,
                            discovery::LIST_SESSIONS_FMT
                        );
                        let sessions = match inner_ref
                            .transport
                            .exec(&reconnect_cmd)
                            .await
                            .and_then(|o| discovery::parse_sessions(&o))
                        {
                            Ok(s) => s,
                            Err(_) => continue, // Can't reach host, retry after next backoff
                        };

                        if !sessions.iter().any(|s| s.name == session) {
                            // Session gone — permanent failure (DC29 session identity)
                            bus.publish_discontinuity(&format!(
                                "stream failed: session '{}' no longer exists on {}",
                                session, host_alias
                            ));
                            set_health(MonitorHealth::Failed);
                            if let Ok(mut signals) = inner_ref.monitor_signals.lock() {
                                signals.remove(&session_for_cleanup);
                            }
                            return Err(anyhow!(
                                "session '{}' no longer exists after reconnect",
                                session
                            ));
                        }

                        // Reopen shell channel
                        match inner_ref.transport.open_shell(80, 24).await {
                            Ok(new_shell) => {
                                shell = new_shell;

                                // Fresh snapshot anchoring (4.2c): capture each pane's
                                // visible content and publish as TargetOutput so
                                // downstream consumers (history, subscribers) get
                                // re-anchored with current screen state.
                                bus.publish_discontinuity(&format!(
                                    "stream resumed: reattached after reconnect for {}:{}",
                                    host_alias, session
                                ));

                                let mut snapshot_panes = 0usize;
                                let mut snapshot_failed = false;
                                let reconnect_pfx = inner_ref.tmux_prefix().await;
                                match discovery::list_panes_in_session_with_prefix(
                                    &inner_ref.transport,
                                    &reconnect_pfx,
                                    &session,
                                )
                                .await
                                {
                                    Ok(panes) => {
                                        for pane in &panes {
                                            let target = pane.address.to_string();
                                            if let Ok(content) = capture::capture_pane_with_prefix(
                                                &inner_ref.transport,
                                                &reconnect_pfx,
                                                &target,
                                            )
                                            .await
                                            {
                                                if !content.is_empty() {
                                                    bus.publish(TargetOutput {
                                                        source: TargetAddress::Pane(
                                                            pane.address.clone(),
                                                        ),
                                                        host: host_alias.clone(),
                                                        content,
                                                        raw_content: None,
                                                        sequence: 0,
                                                        fidelity: OutputFidelity::clean(),
                                                        timestamp: std::time::Instant::now(),
                                                    });
                                                    snapshot_panes += 1;
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        snapshot_failed = true;
                                    }
                                }

                                // Explicit snapshot-outcome marker (DC29 4.2c):
                                // report what actually happened so transcript is
                                // truthful about recovery state.
                                let snapshot_msg = if snapshot_failed {
                                    format!(
                                        "stream snapshot failed: could not discover panes \
                                         after reconnect for {}:{}; \
                                         intermediate output may be missing",
                                        host_alias, session
                                    )
                                } else if snapshot_panes == 0 {
                                    format!(
                                        "stream snapshot empty: no pane content captured \
                                         after reconnect for {}:{}; \
                                         intermediate output may be missing",
                                        host_alias, session
                                    )
                                } else {
                                    format!(
                                        "stream snapshot: captured current screen state \
                                         after reconnect for {}:{} ({} pane{}); \
                                         intermediate output may be missing",
                                        host_alias,
                                        session,
                                        snapshot_panes,
                                        if snapshot_panes == 1 { "" } else { "s" }
                                    )
                                };
                                bus.publish_discontinuity(&snapshot_msg);

                                set_health(MonitorHealth::Streaming);
                                attempt = 0; // Reset on successful reconnect
                            }
                            Err(_) => continue, // Shell open failed, retry
                        }
                    }
                }
            }
        });

        startup_ready_rx.await.map_err(|_| {
            anyhow!(
                "monitor for '{}' exited before control mode became ready",
                session_name
            )
        })?;

        Ok(SessionMonitorHandle::new(target, stop_tx, task, health))
    }

    /// Start monitoring all sessions (optionally filtered).
    /// Returns a `MonitorHandle` for aggregate lifecycle control.
    pub async fn start_monitoring(&self, filter: Option<&regex::Regex>) -> Result<MonitorHandle> {
        let sessions = self.list_sessions().await?;
        let mut handles = HashMap::new();

        for session in sessions {
            if let Some(re) = filter {
                if !re.is_match(&session.name) {
                    continue;
                }
            }
            let name = session.name.clone();
            let handle = self.start_monitoring_session(&name).await?;
            handles.insert(name, handle);
        }

        Ok(MonitorHandle::new(handles))
    }

    /// Signal a specific session's monitor to stop (fire-and-forget).
    ///
    /// This only sends the stop signal — it does **not** await task completion.
    /// The monitor task will exit asynchronously and clean up its registration.
    /// For awaited teardown guarantees, use `SessionMonitorHandle::shutdown()`.
    pub fn stop_monitoring_session(&self, session_name: &str) -> Result<()> {
        let signals = self
            .inner
            .monitor_signals
            .lock()
            .expect("monitor_signals lock poisoned");
        let tx = signals
            .get(session_name)
            .ok_or_else(|| anyhow!("session '{}' not being monitored", session_name))?;
        let _ = tx.send(true);
        Ok(())
    }

    /// Signal all session monitors to stop (fire-and-forget).
    ///
    /// Sends the stop signal to every active monitor. Does **not** await task
    /// completion. For awaited teardown, use `MonitorHandle::shutdown()`.
    pub fn stop_monitoring(&self) {
        let signals = self
            .inner
            .monitor_signals
            .lock()
            .expect("monitor_signals lock poisoned");
        for (_, tx) in signals.iter() {
            let _ = tx.send(true);
        }
    }

    /// List currently monitored session names.
    pub fn monitored_sessions(&self) -> Vec<String> {
        let signals = self
            .inner
            .monitor_signals
            .lock()
            .expect("monitor_signals lock poisoned");
        signals.keys().cloned().collect()
    }

    pub fn host_alias(&self) -> &str {
        &self.inner.host_alias
    }
}

/// Handle to a tracked command execution (DC31).
///
/// Returned by `Target::start_exec()`. Provides non-blocking status inspection
/// via `status()` and async completion via `wait()`.
pub struct ExecHandle {
    id: ExecId,
    state: Arc<std::sync::Mutex<ExecState>>,
    task: tokio::task::JoinHandle<Result<ExecState>>,
}

impl ExecHandle {
    /// The execution identity.
    pub fn id(&self) -> ExecId {
        self.id
    }

    /// Non-blocking snapshot of current execution state.
    pub fn status(&self) -> ExecState {
        self.state.lock().expect("exec state lock poisoned").clone()
    }

    /// Await completion, consuming the handle. Returns the final `ExecState`.
    pub async fn wait(self) -> Result<ExecState> {
        self.task
            .await
            .map_err(|e| anyhow!("exec task panicked: {}", e))?
    }
}

/// Unified target at any hierarchy level (DC16).
pub struct Target {
    inner: Arc<HostHandleInner>,
    address: TargetAddress,
}

impl Target {
    /// Resolve the effective pane_id for exec lock keying (DC19).
    ///
    /// For pane targets, returns the known `pane_id` directly.
    /// For session/window targets, queries tmux to resolve the active pane
    /// via `display-message -p '#{pane_id}'`. Falls back to `target_string()`
    /// if resolution fails.
    async fn resolve_pane_id(&self) -> String {
        match &self.address {
            TargetAddress::Pane(p) => p.pane_id.clone(),
            _ => {
                let prefix = self.inner.tmux_prefix().await;
                let cmd = format!(
                    "{} display-message -p -t {} '#{{pane_id}}'",
                    prefix,
                    crate::control::shell_escape(&self.target_string())
                );
                match self.inner.transport.exec(&cmd).await {
                    Ok(output) => {
                        let id = output.trim().to_string();
                        if id.starts_with('%') {
                            id
                        } else {
                            self.target_string()
                        }
                    }
                    Err(_) => self.target_string(),
                }
            }
        }
    }
}

impl Target {
    /// What level this target is at.
    pub fn level(&self) -> TargetLevel {
        match &self.address {
            TargetAddress::Session(_) => TargetLevel::Session,
            TargetAddress::Window(_) => TargetLevel::Window,
            TargetAddress::Pane(_) => TargetLevel::Pane,
        }
    }

    /// The tmux target string for commands.
    pub fn target_string(&self) -> String {
        match &self.address {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => p.to_tmux_target(),
        }
    }

    /// Full session info (only available at session level).
    /// For cross-level session name access, use `session_name()`.
    pub fn session_info(&self) -> Option<&SessionInfo> {
        match &self.address {
            TargetAddress::Session(s) => Some(s),
            _ => None,
        }
    }

    /// Session name (available at any level).
    pub fn session_name(&self) -> &str {
        match &self.address {
            TargetAddress::Session(s) => &s.name,
            TargetAddress::Window(w) => &w.session_name,
            TargetAddress::Pane(p) => &p.session,
        }
    }

    /// Full window info (only available at window level).
    /// Pane targets carry window index via `pane_address().window` but not
    /// full `WindowInfo`. Use `children()` or `window()` to get window info.
    pub fn window_info(&self) -> Option<&WindowInfo> {
        match &self.address {
            TargetAddress::Window(w) => Some(w),
            _ => None,
        }
    }

    /// Pane address (available at pane level only).
    pub fn pane_address(&self) -> Option<&PaneAddress> {
        match &self.address {
            TargetAddress::Pane(p) => Some(p),
            _ => None,
        }
    }

    /// The underlying TargetAddress.
    pub fn address(&self) -> &TargetAddress {
        &self.address
    }

    // --- Navigation ---

    /// List children one level down.
    pub async fn children(&self) -> Result<Vec<Target>> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(s) => {
                let windows =
                    discovery::list_windows_with_prefix(&self.inner.transport, &prefix, &s.name)
                        .await?;
                Ok(windows
                    .into_iter()
                    .map(|w| Target {
                        inner: self.inner.clone(),
                        address: TargetAddress::Window(w),
                    })
                    .collect())
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &w.session_name,
                )
                .await?;
                Ok(panes
                    .into_iter()
                    .filter(|p| p.address.window == w.index)
                    .map(|p| Target {
                        inner: self.inner.clone(),
                        address: TargetAddress::Pane(p.address),
                    })
                    .collect())
            }
            TargetAddress::Pane(_) => Ok(Vec::new()),
        }
    }

    /// Navigate to a window by index (from session level).
    pub async fn window(&self, index: u32) -> Result<Option<Target>> {
        let prefix = self.inner.tmux_prefix().await;
        let session_name = self.session_name().to_string();
        let windows =
            discovery::list_windows_with_prefix(&self.inner.transport, &prefix, &session_name)
                .await?;
        Ok(windows
            .into_iter()
            .find(|w| w.index == index)
            .map(|w| Target {
                inner: self.inner.clone(),
                address: TargetAddress::Window(w),
            }))
    }

    /// Navigate to a pane by index.
    ///
    /// At session level: resolves via the active window (the window with
    /// `window_active==1`), then filters panes by that window + pane index.
    /// **Active-window drift**: the active window is resolved at call time,
    /// so successive calls may return different panes if `select-window`
    /// changes the active window between calls. For stable pane addressing,
    /// use `window(idx).pane(idx)` with an explicit window index.
    ///
    /// At window level: filters panes within this window.
    /// At pane level: only matches if the current pane has the given index.
    pub async fn pane(&self, index: u32) -> Result<Option<Target>> {
        let prefix = self.inner.tmux_prefix().await;
        let session_name = self.session_name().to_string();

        let window_filter = match &self.address {
            TargetAddress::Session(_) => {
                // Resolve active window for session-level targets
                let windows = discovery::list_windows_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &session_name,
                )
                .await?;
                let active_window = windows.into_iter().find(|w| w.active);
                match active_window {
                    Some(w) => Some(w.index),
                    None => return Ok(None),
                }
            }
            TargetAddress::Window(w) => Some(w.index),
            TargetAddress::Pane(p) => {
                // At pane level, just check if this pane has the requested index
                if p.pane == index {
                    return Ok(Some(Target {
                        inner: self.inner.clone(),
                        address: TargetAddress::Pane(p.clone()),
                    }));
                } else {
                    return Ok(None);
                }
            }
        };

        let panes = discovery::list_panes_in_session_with_prefix(
            &self.inner.transport,
            &prefix,
            &session_name,
        )
        .await?;

        let pane = panes.into_iter().find(|p| {
            p.address.pane == index && window_filter.map_or(true, |wi| p.address.window == wi)
        });

        Ok(pane.map(|p| Target {
            inner: self.inner.clone(),
            address: TargetAddress::Pane(p.address),
        }))
    }

    /// Navigate to a pane by PaneAddress.
    pub fn pane_by_address(&self, address: &PaneAddress) -> Target {
        Target {
            inner: self.inner.clone(),
            address: TargetAddress::Pane(address.clone()),
        }
    }

    /// Create a new child window from a session target (DC25).
    pub async fn new_window(&self, opts: &CreateWindowOptions) -> Result<Target> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(s) => {
                let window =
                    control::new_window_with_prefix(&self.inner.transport, &prefix, &s.name, opts)
                        .await?;
                Ok(Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Window(window),
                })
            }
            TargetAddress::Window(_) => Err(anyhow!(
                "new_window() requires a session target, got window"
            )),
            TargetAddress::Pane(_) => {
                Err(anyhow!("new_window() requires a session target, got pane"))
            }
        }
    }

    /// Split a new child pane from a window or pane target (DC25).
    pub async fn split_pane(&self, opts: &SplitPaneOptions) -> Result<Target> {
        match &self.address {
            TargetAddress::Session(_) => Err(anyhow!(
                "split_pane() requires a window or pane target, got session"
            )),
            TargetAddress::Window(_) | TargetAddress::Pane(_) => {
                let prefix = self.inner.tmux_prefix().await;
                let pane = control::split_pane_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &self.target_string(),
                    opts,
                )
                .await?;
                Ok(Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Pane(pane),
                })
            }
        }
    }

    // --- I/O ---

    /// Send literal text.
    pub async fn send_text(&self, text: &str) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        control::send_text_with_prefix(&self.inner.transport, &prefix, &self.target_string(), text)
            .await
    }

    /// Send a key sequence.
    pub async fn send_keys(&self, keys: &KeySequence) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        control::send_keys_with_prefix(&self.inner.transport, &prefix, &self.target_string(), keys)
            .await
    }

    /// Capture visible pane content.
    ///
    /// At **session or window level**, this captures the **active pane only**
    /// (the pane tmux resolves for the target string). To capture all panes,
    /// use [`capture_all()`](Self::capture_all).
    pub async fn capture(&self) -> Result<String> {
        let prefix = self.inner.tmux_prefix().await;
        capture::capture_pane_with_prefix(&self.inner.transport, &prefix, &self.target_string())
            .await
    }

    /// Capture with scrollback history. `start` is negative for scrollback lines.
    pub async fn capture_with_history(&self, start: i32) -> Result<String> {
        let prefix = self.inner.tmux_prefix().await;
        capture::capture_pane_history_with_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
            start,
        )
        .await
    }

    /// Sample recent scrollback.
    pub async fn sample_text(&self, query: &ScrollbackQuery) -> Result<String> {
        let prefix = self.inner.tmux_prefix().await;
        capture::sample_text_with_tmux_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
            query,
        )
        .await
    }

    /// Capture with options, returning `CaptureResult` with fidelity metadata (DC20).
    pub async fn capture_with_options(&self, opts: &CaptureOptions) -> Result<CaptureResult> {
        let prefix = self.inner.tmux_prefix().await;
        capture::capture_pane_with_options_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
            opts,
        )
        .await
    }

    /// Sample scrollback with options, returning `CaptureResult` (DC20).
    ///
    /// When `previous_text` is provided and `opts.overlap_lines >= 2`,
    /// performs overlap-aware dedup between the previous and new capture.
    pub async fn sample_text_with_options(
        &self,
        query: &ScrollbackQuery,
        opts: &CaptureOptions,
        previous_text: Option<&str>,
    ) -> Result<CaptureResult> {
        let prefix = self.inner.tmux_prefix().await;
        capture::sample_text_with_options_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
            query,
            opts,
            previous_text,
        )
        .await
    }

    /// Capture all panes under this target.
    /// Session: all panes in all windows. Window: all panes in that window.
    /// Pane: single-entry map.
    pub async fn capture_all(&self) -> Result<HashMap<PaneAddress, String>> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(_) => {
                capture::capture_session_with_tmux_prefix(
                    &self.inner.transport,
                    &prefix,
                    self.session_name(),
                )
                .await
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &w.session_name,
                )
                .await?;
                let mut result = HashMap::new();
                for pane in panes.into_iter().filter(|p| p.address.window == w.index) {
                    let target = pane.address.to_tmux_target();
                    let content =
                        capture::capture_pane_with_prefix(&self.inner.transport, &prefix, &target)
                            .await?;
                    result.insert(pane.address, content);
                }
                Ok(result)
            }
            TargetAddress::Pane(p) => {
                let content = capture::capture_pane_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &p.to_tmux_target(),
                )
                .await?;
                let mut result = HashMap::new();
                result.insert(p.clone(), content);
                Ok(result)
            }
        }
    }

    /// Capture all panes with options, returning `CaptureResult` per pane (DC20).
    pub async fn capture_all_with_options(
        &self,
        opts: &CaptureOptions,
    ) -> Result<HashMap<PaneAddress, CaptureResult>> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(_) => {
                capture::capture_session_with_options_prefix(
                    &self.inner.transport,
                    &prefix,
                    self.session_name(),
                    opts,
                )
                .await
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &w.session_name,
                )
                .await?;
                let mut result = HashMap::new();
                for pane in panes.into_iter().filter(|p| p.address.window == w.index) {
                    let target = pane.address.to_tmux_target();
                    let cr = capture::capture_pane_with_options_prefix(
                        &self.inner.transport,
                        &prefix,
                        &target,
                        opts,
                    )
                    .await?;
                    result.insert(pane.address, cr);
                }
                Ok(result)
            }
            TargetAddress::Pane(p) => {
                let cr = capture::capture_pane_with_options_prefix(
                    &self.inner.transport,
                    &prefix,
                    &p.to_tmux_target(),
                    opts,
                )
                .await?;
                let mut result = HashMap::new();
                result.insert(p.clone(), cr);
                Ok(result)
            }
        }
    }

    // --- Lifecycle ---

    /// Kill this entity.
    pub async fn kill(&self) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(s) => {
                control::kill_session_with_prefix(&self.inner.transport, &prefix, &s.name).await
            }
            TargetAddress::Window(_) => {
                control::kill_window_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &self.target_string(),
                )
                .await
            }
            TargetAddress::Pane(_) => {
                control::kill_pane_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &self.target_string(),
                )
                .await
            }
        }
    }

    /// Rename this entity and return a new `Target` with the updated address.
    ///
    /// **Session rename** is a correctness concern: the old `Target` holds a
    /// stale session name, so all subsequent commands (which use `target_string()`)
    /// would fail. Always use the returned `Target` after renaming a session.
    ///
    /// **Window rename** is metadata-only: tmux windows are addressed by index,
    /// not name, so the old `Target` continues to work. The returned `Target`
    /// carries the updated display name.
    ///
    /// Pane-level rename is not supported by tmux and returns `Err`.
    pub async fn rename(&self, new_name: &str) -> Result<Target> {
        let prefix = self.inner.tmux_prefix().await;
        match &self.address {
            TargetAddress::Session(s) => {
                control::rename_session_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &s.name,
                    new_name,
                )
                .await?;
                let mut new_info = s.clone();
                new_info.name = new_name.to_string();
                Ok(Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Session(new_info),
                })
            }
            TargetAddress::Window(w) => {
                control::rename_window_with_prefix(
                    &self.inner.transport,
                    &prefix,
                    &w.session_name,
                    w.index,
                    new_name,
                )
                .await?;
                let mut new_info = w.clone();
                new_info.name = new_name.to_string();
                Ok(Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Window(new_info),
                })
            }
            TargetAddress::Pane(_) => Err(anyhow!("cannot rename a pane")),
        }
    }

    // --- Monitoring (DC13, 2a.4a) ---

    /// Start monitoring this session's output via control mode.
    ///
    /// Session-level only — returns error for window/pane targets.
    /// Opens a control mode connection, parses `%output` frames, and
    /// publishes `TargetOutput` to the host's `OutputBus`.
    pub async fn start_monitoring(&self) -> Result<SessionMonitorHandle> {
        if !matches!(self.address, TargetAddress::Session(_)) {
            return Err(anyhow!(
                "start_monitoring() is session-level only; target '{}' is {:?}",
                self.target_string(),
                self.level()
            ));
        }
        let host = HostHandle {
            inner: self.inner.clone(),
        };
        host.start_monitoring_session(self.session_name()).await
    }

    /// Signal this session's monitor to stop (fire-and-forget).
    ///
    /// Session-level only — returns error for window/pane targets.
    /// Sends the stop signal but does **not** await task completion.
    /// For awaited teardown, use the `SessionMonitorHandle` returned by
    /// `start_monitoring()` and call `shutdown()` on it.
    pub fn stop_monitoring(&self) -> Result<()> {
        if !matches!(self.address, TargetAddress::Session(_)) {
            return Err(anyhow!(
                "stop_monitoring() is session-level only; target '{}' is {:?}",
                self.target_string(),
                self.level()
            ));
        }
        let host = HostHandle {
            inner: self.inner.clone(),
        };
        host.stop_monitoring_session(self.session_name())
    }

    // --- Geometry & history (DC20, Phase 1.9b) ---

    /// Query the current pane geometry and scrollback state.
    pub async fn pane_geometry(&self) -> Result<PaneGeometry> {
        let prefix = self.inner.tmux_prefix().await;
        discovery::query_pane_geometry_with_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
        )
        .await
    }

    /// Take a full geometry snapshot (clients + pane state).
    pub async fn geometry_snapshot(&self) -> Result<GeometrySnapshot> {
        let prefix = self.inner.tmux_prefix().await;
        discovery::take_geometry_snapshot_with_prefix(
            &self.inner.transport,
            &prefix,
            &self.target_string(),
        )
        .await
    }

    /// Set `history-limit` for this target's session (DC20, Phase 1.9b).
    ///
    /// **Must be called before creating panes** — existing panes keep their
    /// creation-time limit. Only meaningful at session level.
    pub async fn set_history_limit(&self, limit: u32) -> Result<()> {
        let prefix = self.inner.tmux_prefix().await;
        control::set_history_limit_with_prefix(
            &self.inner.transport,
            &prefix,
            Some(self.session_name()),
            limit,
        )
        .await
    }

    /// Query the current `history-limit` for this target's session.
    ///
    /// Returns the *configured* limit, not the effective limit of existing
    /// panes. Existing panes retain their creation-time limit. See
    /// [`set_history_limit`](Self::set_history_limit) for details.
    pub async fn get_history_limit(&self) -> Result<u32> {
        let prefix = self.inner.tmux_prefix().await;
        control::get_history_limit_with_prefix(
            &self.inner.transport,
            &prefix,
            Some(self.session_name()),
        )
        .await
    }

    // --- exec (DC19 sentinel mechanism) ---

    /// Launch a tracked command execution in the target pane (DC31).
    ///
    /// Returns an `ExecHandle` immediately. The command runs in a background
    /// tokio task that acquires the per-pane exec lock (DC19 serialization),
    /// sends the command with a sentinel, and polls scrollback.
    ///
    /// Use `ExecHandle::status()` for non-blocking inspection and
    /// `ExecHandle::wait()` for async completion.
    pub async fn start_exec(&self, command: &str, timeout: Duration) -> Result<ExecHandle> {
        let exec_id = ExecId::new();
        let state = Arc::new(std::sync::Mutex::new(ExecState::Running));
        let state_task = state.clone();

        let pane_id = self.resolve_pane_id().await;
        let lock = self.inner.exec_lock(&pane_id);

        let target_str = if pane_id.starts_with('%') {
            pane_id.clone()
        } else {
            self.target_string()
        };
        let marker = format!("__ML{}__", exec_id.short_hex());
        let command = command.to_string();
        let session_name = self.session_name().to_string();

        // Key includes session for session-scoped discontinuity invalidation
        let active_key = format!("{}:{}", session_name, pane_id);

        // Register in active_execs before spawning
        {
            let mut active = self
                .inner
                .active_execs
                .lock()
                .expect("active_execs poisoned");
            let pane_states = active.entry(active_key.clone()).or_default();
            pane_states.push(state.clone());
        }

        let inner_ref = self.inner.clone();
        let active_key_cleanup = active_key;
        let state_cleanup = state.clone();

        let task = tokio::spawn(async move {
            // Acquire pane lock inside the task — start_exec returns immediately
            let _guard = lock.lock().await;

            // Helper: finalize state and deregister from active_execs.
            // Called on all exit paths (success, error, timeout).
            let finalize = |inner: &HostHandleInner,
                            state_arc: &Arc<std::sync::Mutex<ExecState>>,
                            cleanup_arc: &Arc<std::sync::Mutex<ExecState>>,
                            key: &str,
                            new_state: ExecState| {
                // Only update if still Running — do not overwrite a discontinuity-set
                // Unknown (DC31 truthfulness: once continuity is broken, the task
                // must not restore certainty).
                {
                    let mut guard = state_arc.lock().expect("exec state poisoned");
                    if matches!(*guard, ExecState::Running) {
                        *guard = new_state.clone();
                    }
                }

                // Deregister from active_execs
                let mut active = inner.active_execs.lock().expect("active_execs poisoned");
                if let Some(pane_states) = active.get_mut(key) {
                    pane_states.retain(|s| !Arc::ptr_eq(s, cleanup_arc));
                    if pane_states.is_empty() {
                        active.remove(key);
                    }
                }

                // Return the actual state (may differ from new_state if discontinuity fired)
                state_arc.lock().expect("exec state poisoned").clone()
            };

            let transport = &inner_ref.transport;
            let exec_prefix = inner_ref.tmux_prefix().await;

            // Detect shell for exit code variable
            let exit_var = detect_exit_var(transport, &exec_prefix, &target_str).await;

            // Send command with sentinel
            let sentinel_cmd = format!("{} ; echo \"{} {}\"", command, marker, exit_var);
            let keys = KeySequence::literal(&sentinel_cmd).then_enter();
            if let Err(e) =
                control::send_keys_with_prefix(transport, &exec_prefix, &target_str, &keys).await
            {
                let err_state = ExecState::Unknown {
                    reason: format!("send_keys failed: {}", e),
                };
                let final_state = finalize(
                    &inner_ref,
                    &state_task,
                    &state_cleanup,
                    &active_key_cleanup,
                    err_state,
                );
                return Ok(final_state);
            }

            // Poll scrollback for sentinel
            let start_time = tokio::time::Instant::now();
            let poll_interval = Duration::from_millis(100);

            let poll_result = loop {
                // Check if discontinuity already set state to Unknown
                {
                    let guard = state_task.lock().expect("exec state poisoned");
                    if !matches!(*guard, ExecState::Running) {
                        // Discontinuity or external transition — stop polling
                        break guard.clone();
                    }
                }

                if start_time.elapsed() > timeout {
                    break ExecState::Unknown {
                        reason: format!("timed out after {:?} waiting for sentinel", timeout),
                    };
                }

                match capture::capture_pane_escape_history_with_prefix(
                    transport,
                    &exec_prefix,
                    &target_str,
                    -500,
                )
                .await
                {
                    Ok(raw_content) => {
                        let content = capture::strip_ansi(&raw_content);
                        if let Some(result) = parse_sentinel_output(&content, &marker) {
                            break ExecState::Completed(result);
                        }
                    }
                    Err(e) => {
                        break ExecState::Unknown {
                            reason: format!("capture failed: {}", e),
                        };
                    }
                }

                tokio::time::sleep(poll_interval).await;
            };

            let final_state = finalize(
                &inner_ref,
                &state_task,
                &state_cleanup,
                &active_key_cleanup,
                poll_result,
            );
            Ok(final_state)
        });

        Ok(ExecHandle {
            id: exec_id,
            state,
            task,
        })
    }

    /// Execute a shell command in the target pane and capture output.
    ///
    /// Sends command with a UUID sentinel suffix, polls scrollback until
    /// the sentinel appears (or timeout), and extracts stdout + exit code.
    ///
    /// **Two independent timeout knobs**: the `timeout` parameter here is
    /// the sentinel-poll timeout — how long to wait for the command's output
    /// to appear in scrollback. The *transport* timeout (`SshConfig::timeout`
    /// or `LocalTransport::timeout`) bounds each individual `tmux` command
    /// execution (e.g., `capture-pane`, `send-keys`). These are independent:
    /// a long-running user command needs a large `exec` timeout, while the
    /// transport timeout should remain short to catch hung connections.
    pub async fn exec(&self, command: &str, timeout: Duration) -> Result<ExecOutput> {
        let handle = self.start_exec(command, timeout).await?;
        match handle.wait().await? {
            ExecState::Completed(output) => Ok(output),
            ExecState::Unknown { reason } => Err(anyhow!("exec result unknown: {}", reason)),
            ExecState::Running => Err(anyhow!("internal error: Running after wait")),
        }
    }
}

/// Detect the shell exit code variable using a caller-provided prefix.
async fn detect_exit_var(transport: &TransportKind, prefix: &str, target: &str) -> &'static str {
    // Try to detect fish shell
    let cmd = format!(
        "{} display-message -p -t '{}' '#{{pane_current_command}}'",
        prefix,
        crate::transport::shell_escape_arg(target)
    );
    if let Ok(shell) = transport.exec(&cmd).await {
        let shell = shell.trim().to_lowercase();
        if shell.contains("fish") {
            return "$status";
        }
    }
    "$?"
}

/// Parse sentinel output from captured scrollback, tolerating line wraps.
///
/// The sentinel pattern is `<marker> <exit_code>`. In narrow panes, this may
/// wrap across multiple lines. The parser joins all lines and searches for the
/// marker pattern in the concatenated text, then maps back to line boundaries
/// for stdout extraction.
fn parse_sentinel_output(content: &str, marker: &str) -> Option<ExecOutput> {
    let lines: Vec<&str> = content.lines().collect();

    // First try single-line match (fast path for normal-width panes)
    let mut sentinel_idx = None;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with(marker) && !trimmed.contains("echo") {
            sentinel_idx = Some(i);
            break;
        }
    }

    if let Some(idx) = sentinel_idx {
        // Single-line sentinel found — try to extract exit code from this line
        let sentinel_line = lines[idx].trim();
        let after_marker = sentinel_line.strip_prefix(marker).unwrap_or("").trim();

        // Only use fast path if exit code is on the same line
        if let Ok(exit_code) = after_marker.parse::<i32>() {
            // Find command echo line (before sentinel, contains marker in a command)
            let mut cmd_echo_idx = None;
            for i in (0..idx).rev() {
                if lines[i].contains(marker) {
                    cmd_echo_idx = Some(i);
                    break;
                }
            }

            let start = cmd_echo_idx.map_or(0, |i| i + 1);
            let stdout = lines[start..idx].join("\n");
            let stdout = stdout.trim_end().to_string();
            return Some(ExecOutput { stdout, exit_code });
        }
        // Exit code not on same line — fall through to wrap-tolerant path
    }

    // Wrap-tolerant path: join lines and search for marker in concatenated text.
    // This handles the case where the sentinel wraps across lines in narrow panes.
    // Build a map of (cumulative_offset, line_index) for mapping back.
    let mut offsets: Vec<usize> = Vec::with_capacity(lines.len());
    let mut joined = String::new();
    for (i, line) in lines.iter().enumerate() {
        offsets.push(joined.len());
        if i > 0 {
            joined.push('\n');
        }
        joined.push_str(line.trim());
    }

    // Find the sentinel output (not the echo command) in joined text.
    // The echo command line contains "echo" before the marker.
    // The sentinel output has the marker at a position not preceded by "echo".
    let mut search_start = 0;
    let sentinel_pos = loop {
        let pos = match joined[search_start..].find(marker) {
            Some(p) => search_start + p,
            None => return None,
        };

        // Check this isn't the echo command (look back for "echo")
        let before = &joined[..pos];
        let lookback = {
            let mut b = before.len().saturating_sub(20);
            while b > 0 && !before.is_char_boundary(b) {
                b -= 1;
            }
            b
        };
        let is_echo_cmd = before.len() >= 4 && before[lookback..].contains("echo");

        if !is_echo_cmd {
            break pos;
        }

        search_start = pos + marker.len();
        if search_start >= joined.len() {
            return None;
        }
    };

    // Extract exit code from text after marker
    let after_marker = &joined[sentinel_pos + marker.len()..];
    let after_trimmed = after_marker.trim_start();
    let exit_code_str: String = after_trimmed
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    let exit_code = exit_code_str.parse::<i32>().unwrap_or(-1);

    // Map sentinel position back to line index for stdout extraction
    let sentinel_line_idx = offsets
        .iter()
        .rposition(|&off| off <= sentinel_pos)
        .unwrap_or(0);

    // Find command echo line (contains marker before sentinel)
    let mut cmd_echo_idx = None;
    for i in (0..sentinel_line_idx).rev() {
        if lines[i].contains(marker) {
            cmd_echo_idx = Some(i);
            break;
        }
    }
    // Also check if the sentinel_line_idx itself is the echo line (sentinel wrapped from echo)
    if cmd_echo_idx.is_none() && sentinel_line_idx > 0 {
        // The echo and sentinel merged in joined text; check lines above
        for i in (0..sentinel_line_idx).rev() {
            if lines[i].contains("echo") {
                cmd_echo_idx = Some(i);
                break;
            }
        }
    }

    let start = cmd_echo_idx.map_or(0, |i| i + 1);
    let stdout = lines[start..sentinel_line_idx].join("\n");
    let stdout = stdout.trim_end().to_string();
    Some(ExecOutput { stdout, exit_code })
}

#[cfg(test)]
impl HostHandle {
    /// Create a Target for testing without querying tmux.
    pub fn create_target_for_test(&self, session_name: &str) -> Target {
        Target {
            inner: self.inner.clone(),
            address: TargetAddress::Session(SessionInfo {
                name: session_name.to_string(),
                id: "$0".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    fn mock_host(mock: MockTransport) -> HostHandle {
        HostHandle::new(TransportKind::Mock(mock), None)
    }

    #[tokio::test]
    async fn ensure_socket_server_sends_start_server() {
        // Mock verifies the correct command is sent
        let mock = MockTransport::new().with_response("start-server", "");
        let host = mock_host(mock);
        host.ensure_socket_server().await.unwrap();
    }

    #[tokio::test]
    async fn ensure_socket_server_with_named_socket() {
        let mock = MockTransport::new().with_response("start-server", "");
        let socket = TmuxSocket::automation("ci").unwrap();
        let host = HostHandle::new(TransportKind::Mock(mock), Some(socket));
        // Should succeed — the mock matches "start-server" in the command
        host.ensure_socket_server().await.unwrap();
    }

    #[tokio::test]
    async fn create_session_returns_target() {
        let mock = MockTransport::new()
            .with_default("")
            .with_response("list-sessions", "test $0 1700000000 0 1 \n");
        let host = mock_host(mock);
        let target = host
            .create_session("test", &Default::default())
            .await
            .unwrap();
        assert_eq!(target.level(), TargetLevel::Session);
        assert_eq!(target.target_string(), "test");
        assert_eq!(target.session_name(), "test");
    }

    #[tokio::test]
    async fn session_not_found() {
        let mock = MockTransport::new().with_response("list-sessions", "other $0 0 0 1 \n");
        let host = mock_host(mock);
        let result = host.session("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn session_found() {
        let mock = MockTransport::new().with_response("list-sessions", "build $0 0 1 2 \n");
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap();
        assert!(target.is_some());
        let t = target.unwrap();
        assert_eq!(t.level(), TargetLevel::Session);
        assert_eq!(t.target_string(), "build");
    }

    #[tokio::test]
    async fn target_spec_session_level() {
        let mock = MockTransport::new().with_response("list-sessions", "build $0 0 0 1 \n");
        let host = mock_host(mock);
        let spec = TargetSpec::session("build");
        let t = host.target(&spec).await.unwrap();
        assert!(t.is_some());
        assert_eq!(t.unwrap().level(), TargetLevel::Session);
    }

    #[tokio::test]
    async fn children_session_lists_windows() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build $0 0 0 2 \n")
            .with_response(
                "list-windows",
                "$0 build 0 main 1 1 layout\n$0 build 1 editor 0 1 layout\n",
            );
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap().unwrap();
        let children = target.children().await.unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].level(), TargetLevel::Window);
        assert_eq!(children[0].target_string(), "build:0");
        assert_eq!(children[1].target_string(), "build:1");
    }

    #[tokio::test]
    async fn new_window_returns_window_target() {
        let expected = "new-window -P";
        let mock = MockTransport::new().with_response(expected, "$0 build 1 editor 1 1 layout");
        let host = mock_host(mock);
        let target = host.create_target_for_test("build");

        let window = target
            .new_window(&CreateWindowOptions {
                name: Some("editor".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(window.level(), TargetLevel::Window);
        let addr = window.window_info().unwrap();
        assert_eq!(addr.session_name, "build");
        assert_eq!(addr.index, 1);
        assert_eq!(addr.name, "editor");
    }

    #[tokio::test]
    async fn new_window_rejects_non_session_targets() {
        let host = mock_host(MockTransport::new().with_default(""));
        let window_target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Window(WindowInfo {
                session_id: "$0".to_string(),
                session_name: "build".to_string(),
                index: 0,
                name: "main".to_string(),
                active: true,
                pane_count: 1,
                layout: "layout".to_string(),
            }),
        };
        let pane_target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(PaneAddress {
                pane_id: "%1".to_string(),
                session: "build".to_string(),
                window: 0,
                pane: 0,
            }),
        };

        assert!(window_target.new_window(&Default::default()).await.is_err());
        assert!(pane_target.new_window(&Default::default()).await.is_err());
    }

    #[tokio::test]
    async fn split_pane_returns_pane_target_from_window() {
        let mock = MockTransport::new().with_response("split-window -P", "%9 build 0 1");
        let host = mock_host(mock);
        let window_target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Window(WindowInfo {
                session_id: "$0".to_string(),
                session_name: "build".to_string(),
                index: 0,
                name: "main".to_string(),
                active: true,
                pane_count: 1,
                layout: "layout".to_string(),
            }),
        };

        let pane = window_target.split_pane(&Default::default()).await.unwrap();
        assert_eq!(pane.level(), TargetLevel::Pane);
        let addr = pane.pane_address().unwrap();
        assert_eq!(addr.pane_id, "%9");
        assert_eq!(addr.session, "build");
        assert_eq!(addr.window, 0);
        assert_eq!(addr.pane, 1);
    }

    #[tokio::test]
    async fn split_pane_rejects_session_targets() {
        let host = mock_host(MockTransport::new().with_default(""));
        let session_target = host.create_target_for_test("build");
        let err = session_target
            .split_pane(&Default::default())
            .await
            .err()
            .unwrap();
        assert!(err.to_string().contains("requires a window or pane"));
    }

    #[tokio::test]
    async fn rename_pane_fails() {
        let addr = PaneAddress {
            pane_id: "%0".to_string(),
            session: "test".to_string(),
            window: 0,
            pane: 0,
        };
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        let target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(addr),
        };
        assert!(target.rename("new_name").await.is_err());
    }

    #[tokio::test]
    async fn rename_session_returns_updated_target() {
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        let target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Session(SessionInfo {
                name: "old".to_string(),
                id: "$0".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            }),
        };
        let new_target = target.rename("new").await.unwrap();
        assert_eq!(new_target.session_name(), "new");
        assert_eq!(new_target.target_string(), "new");
    }

    #[test]
    fn parse_sentinel_basic() {
        let marker = "__MLabc123__";
        let content = format!(
            "$ echo hello ; echo \"{} $?\"\nhello\n{} 0\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker);
        assert!(result.is_some());
        let out = result.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "hello");
    }

    #[test]
    fn parse_sentinel_nonzero_exit() {
        let marker = "__MLxyz789__";
        let content = format!("$ false ; echo \"{} $?\"\n{} 1\n$", marker, marker);
        let result = parse_sentinel_output(&content, marker);
        assert!(result.is_some());
        assert_eq!(result.unwrap().exit_code, 1);
    }

    #[test]
    fn parse_sentinel_not_found() {
        let result = parse_sentinel_output("just some output\n", "__MLnope__");
        assert!(result.is_none());
    }

    #[test]
    fn parse_sentinel_multiline_output() {
        let marker = "__MLtest42__";
        let content = format!(
            "$ ls ; echo \"{} $?\"\nfile1\nfile2\nfile3\n{} 0\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker).unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout, "file1\nfile2\nfile3");
    }

    #[test]
    fn parse_sentinel_wrapped_across_lines() {
        // Simulate a narrow pane (e.g., 20 cols) where the sentinel wraps
        let marker = "__MLabc123__";
        // The echo output wraps: marker starts on one line, exit code on next
        let content = format!("$ cmd ; echo \"{} $?\"\noutput\n{}", marker, marker);
        // Marker split across lines: "__MLabc123__" then " 0" on next line
        let wrapped = format!("{}\n 0\n$", content);
        let result = parse_sentinel_output(&wrapped, marker);
        assert!(result.is_some());
        let out = result.unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("output"));
    }

    #[test]
    fn parse_sentinel_wrapped_marker_split() {
        // Even more extreme: marker itself wraps mid-token in joined text
        // In practice the marker is short (~16 chars) so this is rare,
        // but the joined-text search should still find it
        let marker = "__MLxyz789__";
        let content = format!("$ echo \"{} $?\"\nhello world\n{} 42\n$", marker, marker);
        let result = parse_sentinel_output(&content, marker).unwrap();
        assert_eq!(result.exit_code, 42);
        assert_eq!(result.stdout, "hello world");
    }

    #[tokio::test]
    async fn pane_session_level_resolves_active_window() {
        // Session with 2 windows: window 0 (inactive) and window 1 (active).
        // Both have pane 0. Session-level pane(0) should return window 1's pane.
        let mock = MockTransport::new()
            .with_response("list-sessions", "build $0 0 0 2 \n")
            .with_response(
                "list-windows",
                "$0 build 0 main 0 1 layout\n$0 build 1 editor 1 1 layout\n",
            )
            .with_response(
                "list-panes",
                "%0 build 0 0  bash 100 80 24 1\n%1 build 1 0  vim 101 80 24 1\n",
            );
        let host = mock_host(mock);
        let session = host.session("build").await.unwrap().unwrap();
        let pane = session.pane(0).await.unwrap();
        assert!(pane.is_some());
        let p = pane.unwrap();
        // Should resolve to window 1 (active), pane 0 → pane_id %1
        assert_eq!(p.pane_address().unwrap().pane_id, "%1");
        assert_eq!(p.pane_address().unwrap().window, 1);
    }

    #[tokio::test]
    async fn exec_lock_shared_across_pane_handles() {
        // Two independently obtained Target handles for the same pane
        // should share the same exec lock via HostHandleInner.
        let addr = PaneAddress {
            pane_id: "%5".to_string(),
            session: "test".to_string(),
            window: 0,
            pane: 0,
        };
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        let t1 = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(addr.clone()),
        };
        let t2 = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(addr),
        };

        // Pane targets resolve directly to pane_id
        let key1 = t1.resolve_pane_id().await;
        let key2 = t2.resolve_pane_id().await;
        assert_eq!(key1, "%5");
        assert_eq!(key2, "%5");

        // Both should get the same lock from HostHandleInner
        let lock1 = t1.inner.exec_lock(&key1);
        let lock2 = t2.inner.exec_lock(&key2);
        assert!(Arc::ptr_eq(&lock1, &lock2));
    }

    #[tokio::test]
    async fn exec_lock_different_panes_are_independent() {
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        let t1 = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(PaneAddress {
                pane_id: "%5".to_string(),
                session: "test".to_string(),
                window: 0,
                pane: 0,
            }),
        };
        let t2 = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(PaneAddress {
                pane_id: "%6".to_string(),
                session: "test".to_string(),
                window: 0,
                pane: 1,
            }),
        };

        let key1 = t1.resolve_pane_id().await;
        let key2 = t2.resolve_pane_id().await;
        let lock1 = t1.inner.exec_lock(&key1);
        let lock2 = t2.inner.exec_lock(&key2);
        assert!(!Arc::ptr_eq(&lock1, &lock2));
    }

    // --- start_exec / ExecHandle tests (DC31) ---

    #[tokio::test]
    async fn start_exec_returns_running_immediately() {
        let marker_re = "__ML";
        let mock = MockTransport::new()
            .with_default("")
            .with_response("display-message", "%5\n")
            // send-keys always succeeds
            .with_response("send-keys", "")
            // capture returns sentinel on second call
            .with_response("capture-pane", "")
            .with_response(
                "capture-pane",
                format!(
                    "$ cmd ; echo \"{}test1234__ $?\"\nhello\n{}test1234__ 0\n$",
                    marker_re, marker_re
                )
                .as_str(),
            );
        let host = mock_host(mock);
        let target = host.create_target_for_test("build");
        let handle = target
            .start_exec("echo hello", Duration::from_secs(5))
            .await
            .unwrap();

        // Handle should be immediately available with Running state
        assert_eq!(handle.id().short_hex().len(), 8);
        // Wait for completion
        let state = handle.wait().await.unwrap();
        assert!(state.is_terminal());
    }

    #[tokio::test]
    async fn start_exec_timeout_produces_unknown() {
        let mock = MockTransport::new()
            .with_default("")
            .with_response("display-message", "%5\n")
            .with_response("send-keys", "")
            // capture-pane never returns sentinel
            .with_response("capture-pane", "just some output\n");
        let host = mock_host(mock);
        let target = host.create_target_for_test("test");
        let handle = target
            .start_exec("sleep 100", Duration::from_millis(200))
            .await
            .unwrap();

        let state = handle.wait().await.unwrap();
        match state {
            ExecState::Unknown { reason } => {
                assert!(reason.contains("timed out"));
            }
            _ => panic!("expected Unknown state, got {:?}", state),
        }
    }

    #[tokio::test]
    async fn start_exec_deregisters_from_active_execs() {
        let mock = MockTransport::new()
            .with_default("")
            .with_response("display-message", "%5\n")
            .with_response("send-keys", "")
            .with_response("capture-pane", "just some output\n");
        let host = mock_host(mock);
        let target = host.create_target_for_test("test");
        let handle = target
            .start_exec("cmd", Duration::from_millis(200))
            .await
            .unwrap();

        // Wait for completion (will timeout → Unknown)
        let _ = handle.wait().await;

        // active_execs should be empty after completion
        let active = host.inner.active_execs.lock().unwrap();
        assert!(
            active.is_empty(),
            "active_execs should be empty after completion"
        );
    }

    // --- Discontinuity notification tests (DC31) ---

    #[tokio::test]
    async fn notify_exec_discontinuity_transitions_running_to_unknown() {
        let state = Arc::new(std::sync::Mutex::new(ExecState::Running));
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        {
            let mut active = host.inner.active_execs.lock().unwrap();
            active
                .entry("test-session:test-pane".to_string())
                .or_default()
                .push(state.clone());
        }
        host.inner
            .notify_exec_discontinuity("test-session", "connection lost");
        let guard = state.lock().unwrap();
        match &*guard {
            ExecState::Unknown { reason } => assert!(reason.contains("connection lost")),
            other => panic!("expected Unknown, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn notify_exec_discontinuity_leaves_completed_unchanged() {
        let state = Arc::new(std::sync::Mutex::new(ExecState::Completed(ExecOutput {
            stdout: "ok".to_string(),
            exit_code: 0,
        })));
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        {
            let mut active = host.inner.active_execs.lock().unwrap();
            active
                .entry("test-session:test-pane".to_string())
                .or_default()
                .push(state.clone());
        }
        host.inner
            .notify_exec_discontinuity("test-session", "connection lost");
        let guard = state.lock().unwrap();
        match &*guard {
            ExecState::Completed(out) => assert_eq!(out.stdout, "ok"),
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn notify_exec_discontinuity_session_scoped() {
        // Session A loss should not affect session B execs
        let state_a = Arc::new(std::sync::Mutex::new(ExecState::Running));
        let state_b = Arc::new(std::sync::Mutex::new(ExecState::Running));
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        {
            let mut active = host.inner.active_execs.lock().unwrap();
            active
                .entry("session-a:pane1".to_string())
                .or_default()
                .push(state_a.clone());
            active
                .entry("session-b:pane2".to_string())
                .or_default()
                .push(state_b.clone());
        }
        // Only session-a loses connection
        host.inner
            .notify_exec_discontinuity("session-a", "connection lost");

        let guard_a = state_a.lock().unwrap();
        assert!(
            matches!(&*guard_a, ExecState::Unknown { .. }),
            "session-a exec should be Unknown"
        );
        drop(guard_a);

        let guard_b = state_b.lock().unwrap();
        assert!(
            matches!(&*guard_b, ExecState::Running),
            "session-b exec should still be Running"
        );
    }

    #[tokio::test]
    async fn exec_lock_session_resolves_pane_id() {
        // A session-level target should resolve its active pane_id via
        // display-message, so it shares a lock with a pane-level target
        // pointing to the same pane.
        let mock = MockTransport::new()
            .with_default("")
            .with_response("display-message", "%5\n");
        let host = mock_host(mock);
        let session_target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Session(SessionInfo {
                name: "build".to_string(),
                id: "$0".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            }),
        };
        let pane_target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(PaneAddress {
                pane_id: "%5".to_string(),
                session: "build".to_string(),
                window: 0,
                pane: 0,
            }),
        };

        let session_key = session_target.resolve_pane_id().await;
        let pane_key = pane_target.resolve_pane_id().await;

        // Session target should resolve to the same pane_id
        assert_eq!(session_key, "%5");
        assert_eq!(pane_key, "%5");

        // Same lock
        let lock1 = host.inner.exec_lock(&session_key);
        let lock2 = host.inner.exec_lock(&pane_key);
        assert!(Arc::ptr_eq(&lock1, &lock2));
    }
}
