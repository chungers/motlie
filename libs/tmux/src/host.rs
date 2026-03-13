use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

use crate::capture;
use crate::control;
use crate::discovery;
use crate::keys::KeySequence;
use crate::transport::TransportKind;
use crate::types::*;

/// Internal state shared via Arc between HostHandle and Target.
struct HostHandleInner {
    transport: TransportKind,
    socket: Option<TmuxSocket>,
    /// Per-pane exec locks keyed by resolved `pane_id` (DC19).
    /// All target levels resolve their effective pane via
    /// `display-message -p '#{pane_id}'` before lock acquisition,
    /// ensuring session/window/pane handles to the same active pane
    /// share one lock.
    /// Uses `std::sync::Mutex` for the map (held briefly, no await),
    /// inner `tokio::sync::Mutex` for the per-pane exec serialization.
    exec_locks: std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl HostHandleInner {
    /// Get or create an exec lock for a given pane identity key.
    fn exec_lock(&self, key: &str) -> Arc<Mutex<()>> {
        let mut locks = self.exec_locks.lock().expect("exec_locks poisoned");
        locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
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
                transport: TransportKind::Local(
                    crate::transport::LocalTransport::new(),
                ),
                socket: None,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
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
                transport: TransportKind::Local(
                    crate::transport::LocalTransport::with_timeout(timeout),
                ),
                socket: None,
                exec_locks: std::sync::Mutex::new(HashMap::new()),
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
            }),
        }
    }

    /// Access the transport kind (crate-internal, not public API).
    ///
    /// Scoped to `pub(crate)` so the transport abstraction stays internal
    /// per DC21. Enables unit tests within the crate to assert transport
    /// selection without relying on behavioral side effects.
    pub(crate) fn transport_kind(&self) -> &TransportKind {
        &self.inner.transport
    }

    /// List all tmux sessions on this host.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref()).await
    }

    /// List attached clients on this host (DC20, Phase 1.9b).
    pub async fn list_clients(&self) -> Result<Vec<ClientInfo>> {
        discovery::list_clients(&self.inner.transport, self.inner.socket.as_ref()).await
    }

    /// Set global `history-limit` (DC20, Phase 1.9b).
    ///
    /// **Must be set before creating sessions/panes.** Existing panes
    /// retain their creation-time limit.
    pub async fn set_global_history_limit(&self, limit: u32) -> Result<()> {
        control::set_history_limit(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            None,
            limit,
        )
        .await
    }

    /// Query the global `history-limit` default.
    pub async fn get_global_history_limit(&self) -> Result<u32> {
        control::get_history_limit(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            None,
        )
        .await
    }

    /// Create a new tmux session. Returns a Target at session level.
    pub async fn create_session(
        &self,
        name: &str,
        window_name: Option<&str>,
        command: Option<&str>,
    ) -> Result<Target> {
        control::create_session(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            name,
            window_name,
            command,
        )
        .await?;

        // Query the created session to get full info
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
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
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
        Ok(sessions.into_iter().find(|s| s.name == name).map(|info| {
            Target {
                inner: self.inner.clone(),
                address: TargetAddress::Session(info),
    
            }
        }))
    }

    /// Get a Target from a TargetSpec. Verifies the entity exists.
    pub async fn target(&self, spec: &TargetSpec) -> Result<Option<Target>> {
        // Check session exists
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
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
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    spec.session_name(),
                )
                .await?;
                let win = windows.into_iter().find(|w| {
                    w.index.to_string() == *window_str || w.name == *window_str
                });
                Ok(win.map(|w| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Window(w),
        
                }))
            }
            (Some(window_str), Some(pane_idx)) => {
                // Resolve window by index or name → actual window index
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    spec.session_name(),
                )
                .await?;
                let resolved_window = windows.into_iter().find(|w| {
                    w.index.to_string() == *window_str || w.name == *window_str
                });
                let window_index = match resolved_window {
                    Some(w) => w.index,
                    None => return Ok(None),
                };

                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    spec.session_name(),
                )
                .await?;
                let pane = panes.into_iter().find(|p| {
                    p.address.window == window_index && p.address.pane == pane_idx
                });
                Ok(pane.map(|p| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Pane(p.address),
        
                }))
            }
        }
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
                let prefix = crate::transport::tmux_prefix(self.inner.socket.as_ref());
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
        match &self.address {
            TargetAddress::Session(s) => {
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &s.name,
                )
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
                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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
        let session_name = self.session_name().to_string();
        let windows = discovery::list_windows(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &session_name,
        )
        .await?;
        Ok(windows.into_iter().find(|w| w.index == index).map(|w| {
            Target {
                inner: self.inner.clone(),
                address: TargetAddress::Window(w),
    
            }
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
        let session_name = self.session_name().to_string();

        let window_filter = match &self.address {
            TargetAddress::Session(_) => {
                // Resolve active window for session-level targets
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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

        let panes = discovery::list_panes_in_session(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &session_name,
        )
        .await?;

        let pane = panes.into_iter().find(|p| {
            p.address.pane == index
                && window_filter.map_or(true, |wi| p.address.window == wi)
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

    // --- I/O ---

    /// Send literal text.
    pub async fn send_text(&self, text: &str) -> Result<()> {
        control::send_text(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            text,
        )
        .await
    }

    /// Send a key sequence.
    pub async fn send_keys(&self, keys: &KeySequence) -> Result<()> {
        control::send_keys(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            keys,
        )
        .await
    }

    /// Capture visible pane content.
    ///
    /// At **session or window level**, this captures the **active pane only**
    /// (the pane tmux resolves for the target string). To capture all panes,
    /// use [`capture_all()`](Self::capture_all).
    pub async fn capture(&self) -> Result<String> {
        capture::capture_pane(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
        )
        .await
    }

    /// Capture with scrollback history. `start` is negative for scrollback lines.
    pub async fn capture_with_history(&self, start: i32) -> Result<String> {
        capture::capture_pane_history(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            start,
        )
        .await
    }

    /// Sample recent scrollback.
    pub async fn sample_text(&self, query: &ScrollbackQuery) -> Result<String> {
        capture::sample_text(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            query,
        )
        .await
    }

    /// Capture with options, returning `CaptureResult` with fidelity metadata (DC20).
    pub async fn capture_with_options(
        &self,
        opts: &CaptureOptions,
    ) -> Result<CaptureResult> {
        capture::capture_pane_with_options(
            &self.inner.transport,
            self.inner.socket.as_ref(),
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
        capture::sample_text_with_options(
            &self.inner.transport,
            self.inner.socket.as_ref(),
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
        match &self.address {
            TargetAddress::Session(_) => {
                capture::capture_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    self.session_name(),
                )
                .await
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &w.session_name,
                )
                .await?;
                let mut result = HashMap::new();
                for pane in panes.into_iter().filter(|p| p.address.window == w.index) {
                    let target = pane.address.to_tmux_target();
                    let content = capture::capture_pane(
                        &self.inner.transport,
                        self.inner.socket.as_ref(),
                        &target,
                    )
                    .await?;
                    result.insert(pane.address, content);
                }
                Ok(result)
            }
            TargetAddress::Pane(p) => {
                let content = capture::capture_pane(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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
        match &self.address {
            TargetAddress::Session(_) => {
                capture::capture_session_with_options(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    self.session_name(),
                    opts,
                )
                .await
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &w.session_name,
                )
                .await?;
                let mut result = HashMap::new();
                for pane in panes.into_iter().filter(|p| p.address.window == w.index) {
                    let target = pane.address.to_tmux_target();
                    let cr = capture::capture_pane_with_options(
                        &self.inner.transport,
                        self.inner.socket.as_ref(),
                        &target,
                        opts,
                    )
                    .await?;
                    result.insert(pane.address, cr);
                }
                Ok(result)
            }
            TargetAddress::Pane(p) => {
                let cr = capture::capture_pane_with_options(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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
        match &self.address {
            TargetAddress::Session(s) => {
                control::kill_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &s.name,
                )
                .await
            }
            TargetAddress::Window(_) => {
                control::kill_window(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &self.target_string(),
                )
                .await
            }
            TargetAddress::Pane(_) => {
                control::kill_pane(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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
        match &self.address {
            TargetAddress::Session(s) => {
                control::rename_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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
                control::rename_window(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
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

    // --- Geometry & history (DC20, Phase 1.9b) ---

    /// Query the current pane geometry and scrollback state.
    pub async fn pane_geometry(&self) -> Result<PaneGeometry> {
        discovery::query_pane_geometry(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
        )
        .await
    }

    /// Take a full geometry snapshot (clients + pane state).
    pub async fn geometry_snapshot(&self) -> Result<GeometrySnapshot> {
        discovery::take_geometry_snapshot(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
        )
        .await
    }

    /// Set `history-limit` for this target's session (DC20, Phase 1.9b).
    ///
    /// **Must be called before creating panes** — existing panes keep their
    /// creation-time limit. Only meaningful at session level.
    pub async fn set_history_limit(&self, limit: u32) -> Result<()> {
        control::set_history_limit(
            &self.inner.transport,
            self.inner.socket.as_ref(),
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
        control::get_history_limit(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            Some(self.session_name()),
        )
        .await
    }

    // --- exec (DC19 sentinel mechanism) ---

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
        let pane_id = self.resolve_pane_id().await;
        let lock = self.inner.exec_lock(&pane_id);
        let _guard = lock.lock().await;

        // Use a short hex ID to avoid line wrapping in narrow panes
        let id = &uuid::Uuid::new_v4().to_string()[..8];
        let marker = format!("__ML{}__", id);
        // Use resolved pane_id as tmux target when available (starts with %).
        // This ensures lock key and execution target are the same pane,
        // preventing divergence if the active pane changes after lock acquisition.
        let target = if pane_id.starts_with('%') {
            pane_id.clone()
        } else {
            self.target_string()
        };
        let socket = self.inner.socket.as_ref();
        let transport = &self.inner.transport;

        // Detect shell for exit code variable
        let exit_var = detect_exit_var(transport, socket, &target).await;

        // Send command with sentinel.
        // The sentinel echo must NOT be inside single quotes so $? expands.
        // Format: <command> ; echo "<marker> $?"
        let sentinel_cmd = format!(
            "{} ; echo \"{} {}\"",
            command, marker, exit_var
        );
        let keys = KeySequence::literal(&sentinel_cmd).then_enter();
        control::send_keys(transport, socket, &target, &keys).await?;

        // Poll scrollback for sentinel
        let start_time = tokio::time::Instant::now();
        let poll_interval = Duration::from_millis(100);

        loop {
            if start_time.elapsed() > timeout {
                return Err(anyhow!(
                    "exec timed out after {:?} waiting for sentinel",
                    timeout
                ));
            }

            // Poll with `-ep` scrollback for wrap-tolerant sentinel detection (DC20).
            // The escape-mode capture preserves ANSI sequences which we strip
            // before parsing, so line-wrapping artifacts from width changes
            // don't break sentinel matching.
            let raw_content =
                capture::capture_pane_escape_history(transport, socket, &target, -500)
                    .await?;
            let content = capture::strip_ansi(&raw_content);

            if let Some(result) = parse_sentinel_output(&content, &marker) {
                return Ok(result);
            }

            tokio::time::sleep(poll_interval).await;
        }
    }
}

/// Detect the shell exit code variable.
async fn detect_exit_var(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> &'static str {
    // Try to detect fish shell
    let prefix = crate::transport::tmux_prefix(socket);
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
        let is_echo_cmd = before.len() >= 4
            && before[before.len().saturating_sub(20)..].contains("echo");

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
    let exit_code_str: String = after_trimmed.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
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
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    fn mock_host(mock: MockTransport) -> HostHandle {
        HostHandle::new(TransportKind::Mock(mock), None)
    }

    #[tokio::test]
    async fn create_session_returns_target() {
        let mock = MockTransport::new()
            .with_default("")
            .with_response("list-sessions", "test\t$0\t1700000000\t0\t1\t\n");
        let host = mock_host(mock);
        let target = host.create_session("test", None, None).await.unwrap();
        assert_eq!(target.level(), TargetLevel::Session);
        assert_eq!(target.target_string(), "test");
        assert_eq!(target.session_name(), "test");
    }

    #[tokio::test]
    async fn session_not_found() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "other\t$0\t0\t0\t1\t\n");
        let host = mock_host(mock);
        let result = host.session("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn session_found() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t1\t2\t\n");
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap();
        assert!(target.is_some());
        let t = target.unwrap();
        assert_eq!(t.level(), TargetLevel::Session);
        assert_eq!(t.target_string(), "build");
    }

    #[tokio::test]
    async fn target_spec_session_level() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t0\t1\t\n");
        let host = mock_host(mock);
        let spec = TargetSpec::session("build");
        let t = host.target(&spec).await.unwrap();
        assert!(t.is_some());
        assert_eq!(t.unwrap().level(), TargetLevel::Session);
    }

    #[tokio::test]
    async fn children_session_lists_windows() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t0\t2\t\n")
            .with_response("list-windows", "$0\tbuild\t0\tmain\t1\t1\tlayout\n$0\tbuild\t1\teditor\t0\t1\tlayout\n");
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap().unwrap();
        let children = target.children().await.unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].level(), TargetLevel::Window);
        assert_eq!(children[0].target_string(), "build:0");
        assert_eq!(children[1].target_string(), "build:1");
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
        let content = format!(
            "$ false ; echo \"{} $?\"\n{} 1\n$",
            marker, marker
        );
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
        let content = format!(
            "$ cmd ; echo \"{} $?\"\noutput\n{}",
            marker, marker
        );
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
        let content = format!(
            "$ echo \"{} $?\"\nhello world\n{} 42\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker).unwrap();
        assert_eq!(result.exit_code, 42);
        assert_eq!(result.stdout, "hello world");
    }

    #[tokio::test]
    async fn pane_session_level_resolves_active_window() {
        // Session with 2 windows: window 0 (inactive) and window 1 (active).
        // Both have pane 0. Session-level pane(0) should return window 1's pane.
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t0\t2\t\n")
            .with_response(
                "list-windows",
                "$0\tbuild\t0\tmain\t0\t1\tlayout\n$0\tbuild\t1\teditor\t1\t1\tlayout\n",
            )
            .with_response(
                "list-panes",
                "%0\tbuild:0.0\t\tbash\t100\t80\t24\t1\n%1\tbuild:1.0\t\tvim\t101\t80\t24\t1\n",
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
