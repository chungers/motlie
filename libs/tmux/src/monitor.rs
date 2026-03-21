//! SessionMonitor: control mode stream parser and monitor handle wiring
//! (2a.2a, 2a.4a, DC10, DC24).
//!
//! The monitor parses tmux control mode `%output` frames, assembles per-pane
//! state, and publishes `TargetOutput` to the `OutputBus`. The monitor does
//! **not** evaluate rules — rule evaluation is a consumer concern via
//! Subscription adapters (DC24).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::watch;

// ---------------------------------------------------------------------------
// Per-session monitor health (DC29, 4.2a/4.2d)
// ---------------------------------------------------------------------------

/// Exit reason from `SessionMonitor::run()`, used by the supervision loop
/// to decide whether to reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorExitReason {
    /// Intentional stop via shutdown signal.
    Stopped,
    /// Connection closed unexpectedly (EOF or None from shell).
    ConnectionLost,
}

/// Per-session monitor health state (DC29).
///
/// This is the ground truth for streaming health. Host/Fleet health is derived
/// from per-session states rather than flattened into a single coarse status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorHealth {
    /// Actively streaming control-mode output.
    Streaming,
    /// Connection lost, attempting reconnect with backoff.
    Reconnecting,
    /// Permanently failed (session gone or retries exhausted).
    Failed,
    /// Intentionally stopped by caller.
    Stopped,
}

use crate::capture;
use crate::host::Target;
use crate::sink::{OutputBus, TargetOutput};
use crate::transport::{tmux_prefix, ShellChannelKind, ShellEvent};
use crate::types::*;

// ---------------------------------------------------------------------------
// Control mode parser (2a.2a)
// ---------------------------------------------------------------------------

/// Parsed control mode message from tmux.
#[derive(Debug, PartialEq)]
pub(crate) enum ControlModeMessage {
    /// `%output %<pane_id> <data>`
    Output { pane_id: String, data: String },
    /// `%begin`, `%end`, `%error` — command response lifecycle
    CommandResponse(String),
    /// `%session-changed`, `%window-renamed`, etc. — lifecycle notifications
    Notification(String),
    /// Unrecognized line
    Unknown(String),
}

/// Parse a single control mode line.
pub(crate) fn parse_control_line(line: &str) -> ControlModeMessage {
    if let Some(rest) = line.strip_prefix("%output ") {
        // Format: %output %<pane_id> <data>
        // Find the pane_id (starts with %, ends at next space)
        if let Some(space_idx) = rest.find(' ') {
            let pane_id = rest[..space_idx].to_string();
            let data = decode_octal_escapes(&rest[space_idx + 1..]);
            ControlModeMessage::Output { pane_id, data }
        } else {
            // Pane ID with no data
            let pane_id = rest.to_string();
            ControlModeMessage::Output {
                pane_id,
                data: String::new(),
            }
        }
    } else if line.starts_with("%begin ")
        || line.starts_with("%end ")
        || line.starts_with("%error ")
    {
        ControlModeMessage::CommandResponse(line.to_string())
    } else if line.starts_with('%') {
        ControlModeMessage::Notification(line.to_string())
    } else {
        ControlModeMessage::Unknown(line.to_string())
    }
}

/// Decode tmux control mode octal escapes (`\ooo`) to actual bytes.
///
/// Tmux encodes non-printable and non-ASCII bytes as `\ooo` octal sequences.
/// Multi-byte UTF-8 characters produce multiple consecutive escapes (e.g.,
/// `é` → `\303\251`, bytes 0xC3 0xA9). This decoder collects raw bytes and
/// converts to UTF-8 at the end, preserving multi-byte sequences correctly.
pub(crate) fn decode_octal_escapes(input: &str) -> String {
    let mut buf: Vec<u8> = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() {
            let d1 = bytes[i + 1];
            let d2 = bytes[i + 2];
            let d3 = bytes[i + 3];
            if (b'0'..=b'7').contains(&d1)
                && (b'0'..=b'7').contains(&d2)
                && (b'0'..=b'7').contains(&d3)
            {
                let val = (d1 - b'0') * 64 + (d2 - b'0') * 8 + (d3 - b'0');
                buf.push(val);
                i += 4;
                continue;
            }
        }
        // Also handle \\  -> backslash
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
            buf.push(b'\\');
            i += 2;
            continue;
        }
        buf.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Per-pane stream assembly state.
struct PaneAssemblyState {
    sequence: u64,
    address: PaneAddress,
}

impl PaneAssemblyState {
    fn new(pane_id: &str, session: &str) -> Self {
        PaneAssemblyState {
            sequence: 0,
            address: PaneAddress {
                pane_id: pane_id.to_string(),
                session: session.to_string(),
                // Window/pane indices unknown from control mode — use 0.
                // These are display-only; pane_id is authoritative.
                window: 0,
                pane: 0,
            },
        }
    }
}

/// Monitors a single tmux session via control mode.
pub struct SessionMonitor {
    session_name: String,
    host_alias: String,
    socket: Option<TmuxSocket>,
    normalize: CaptureNormalizeMode,
    pane_states: HashMap<String, PaneAssemblyState>,
}

impl SessionMonitor {
    pub fn new(session_name: String, host_alias: String) -> Self {
        SessionMonitor {
            session_name,
            host_alias,
            socket: None,
            normalize: CaptureNormalizeMode::Raw,
            pane_states: HashMap::new(),
        }
    }

    /// Create a monitor with a specific normalization mode.
    pub fn with_normalize(
        session_name: String,
        host_alias: String,
        normalize: CaptureNormalizeMode,
    ) -> Self {
        SessionMonitor {
            session_name,
            host_alias,
            socket: None,
            normalize,
            pane_states: HashMap::new(),
        }
    }

    /// Attach a tmux socket context for control-mode monitoring.
    pub fn with_socket(mut self, socket: Option<TmuxSocket>) -> Self {
        self.socket = socket;
        self
    }

    fn attach_command(&self) -> String {
        build_attach_command(&self.session_name, self.socket.as_ref())
    }

    /// Process a parsed %output frame: return a TargetOutput.
    ///
    /// Applies the configured normalization mode (1.9a):
    /// - `Raw`: pass decoded data unchanged (ANSI preserved)
    /// - `ScreenStable`: normalize line endings + trim, ANSI preserved;
    ///   `raw_content` holds original decoded data
    /// - `PlainText`: strip ANSI + normalize; `raw_content` holds original
    fn process_output(&mut self, pane_id: &str, data: &str) -> TargetOutput {
        let state = self
            .pane_states
            .entry(pane_id.to_string())
            .or_insert_with(|| PaneAssemblyState::new(pane_id, &self.session_name));

        state.sequence += 1;

        let (content, raw_content) = match self.normalize {
            CaptureNormalizeMode::Raw => (data.to_string(), None),
            CaptureNormalizeMode::ScreenStable => {
                let normalized = capture::normalize_screen_stable(data);
                let raw = if normalized != data {
                    Some(data.to_string())
                } else {
                    None
                };
                (normalized, raw)
            }
            CaptureNormalizeMode::PlainText => {
                let normalized = capture::normalize_plain_text(data);
                let raw = if normalized != data {
                    Some(data.to_string())
                } else {
                    None
                };
                (normalized, raw)
            }
        };

        TargetOutput {
            source: TargetAddress::Pane(state.address.clone()),
            host: self.host_alias.clone(),
            content,
            raw_content,
            sequence: state.sequence,
            fidelity: OutputFidelity::clean(),
            timestamp: Instant::now(),
        }
    }

    /// Main monitor loop.
    ///
    /// Opens `tmux -C attach-session -t <session>` on the configured socket,
    /// reads control mode output,
    /// parses `%output` frames, publishes `TargetOutput` to the bus.
    /// Returns `Stopped` when the stop signal fires, `ConnectionLost` on EOF.
    pub async fn run(
        &mut self,
        shell: &mut ShellChannelKind,
        bus: &OutputBus,
        mut stop: watch::Receiver<bool>,
    ) -> Result<MonitorExitReason> {
        // Send the tmux control mode attach command
        let attach_cmd = self.attach_command();
        shell.write(attach_cmd.as_bytes()).await?;

        // Line-buffered reader over raw shell bytes
        let mut line_buf = String::new();

        loop {
            tokio::select! {
                _ = stop.changed() => {
                    if *stop.borrow() {
                        return Ok(MonitorExitReason::Stopped);
                    }
                }
                event = shell.read() => {
                    match event {
                        Some(ShellEvent::Data(bytes)) => {
                            let text = String::from_utf8_lossy(&bytes);
                            line_buf.push_str(&text);

                            // Process complete lines
                            while let Some(newline_pos) = line_buf.find('\n') {
                                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                                line_buf = line_buf[newline_pos + 1..].to_string();

                                if line.is_empty() {
                                    continue;
                                }

                                match parse_control_line(&line) {
                                    ControlModeMessage::Output { pane_id, data } => {
                                        let output = self.process_output(&pane_id, &data);
                                        bus.publish(output);
                                    }
                                    ControlModeMessage::CommandResponse(_) => {
                                        // Ignore command responses for now
                                    }
                                    ControlModeMessage::Notification(_) => {
                                        // Ignore notifications for now
                                    }
                                    ControlModeMessage::Unknown(line) => {
                                        tracing::warn!(
                                            session = %self.session_name,
                                            "unrecognized control mode line: {}",
                                            line
                                        );
                                    }
                                }
                            }
                        }
                        Some(ShellEvent::Eof) | None => {
                            tracing::info!(session = %self.session_name, "control mode connection closed");
                            return Ok(MonitorExitReason::ConnectionLost);
                        }
                    }
                }
            }
        }
    }
}

fn build_attach_command(session_name: &str, socket: Option<&TmuxSocket>) -> String {
    format!(
        "{} -C attach-session -t {}\n",
        tmux_prefix(socket),
        crate::control::shell_escape(session_name)
    )
}

// ---------------------------------------------------------------------------
// 2a.4a — Monitor handle wiring
// ---------------------------------------------------------------------------

/// Handle to a single monitored session (DC16, DC29).
/// Provides lifecycle control, health inspection, and `Deref` to the
/// underlying `Target`.
pub struct SessionMonitorHandle {
    target: Target,
    stop_tx: watch::Sender<bool>,
    task: std::sync::Mutex<Option<tokio::task::JoinHandle<Result<MonitorExitReason>>>>,
    health: Arc<std::sync::Mutex<MonitorHealth>>,
}

impl SessionMonitorHandle {
    /// Create a new handle. Called internally by HostHandle::start_monitoring_session.
    pub(crate) fn new(
        target: Target,
        stop_tx: watch::Sender<bool>,
        task: tokio::task::JoinHandle<Result<MonitorExitReason>>,
        health: Arc<std::sync::Mutex<MonitorHealth>>,
    ) -> Self {
        SessionMonitorHandle {
            target,
            stop_tx,
            task: std::sync::Mutex::new(Some(task)),
            health,
        }
    }

    /// Current health state of this session monitor (DC29).
    pub fn health(&self) -> MonitorHealth {
        *self.health.lock().expect("health lock poisoned")
    }

    /// Signal stop and wait for the monitor task to finish.
    pub async fn shutdown(&self) -> Result<()> {
        let _ = self.stop_tx.send(true);
        let task = self.task.lock().expect("task lock poisoned").take();
        if let Some(task) = task {
            task.await
                .map_err(|e| anyhow!("monitor task panicked: {}", e))?
                .map(|_| ())?;
        }
        Ok(())
    }

    /// Whether the monitor task is still running.
    pub fn is_active(&self) -> bool {
        let guard = self.task.lock().expect("task lock poisoned");
        match &*guard {
            Some(task) => !task.is_finished(),
            None => false,
        }
    }

    /// Get the underlying target.
    pub fn target(&self) -> &Target {
        &self.target
    }
}

impl std::ops::Deref for SessionMonitorHandle {
    type Target = crate::host::Target;
    fn deref(&self) -> &crate::host::Target {
        &self.target
    }
}

/// Aggregate handle over all monitored sessions on a host.
pub struct MonitorHandle {
    sessions: HashMap<String, SessionMonitorHandle>,
}

impl MonitorHandle {
    pub(crate) fn new(sessions: HashMap<String, SessionMonitorHandle>) -> Self {
        MonitorHandle { sessions }
    }

    /// Shutdown all monitored sessions.
    pub async fn shutdown(&mut self) -> Result<()> {
        for (_, handle) in self.sessions.drain() {
            handle.shutdown().await?;
        }
        Ok(())
    }

    /// Get a session monitor handle by name.
    pub fn get(&self, session: &str) -> Option<&SessionMonitorHandle> {
        self.sessions.get(session)
    }

    /// Stop and remove a specific session's monitor.
    pub async fn stop_session(&mut self, name: &str) -> Result<()> {
        let handle = self
            .sessions
            .remove(name)
            .ok_or_else(|| anyhow!("session '{}' not being monitored", name))?;
        handle.shutdown().await
    }

    /// Get a session monitor handle by TargetSpec.
    /// Matches on the spec's session name.
    pub fn get_by_spec(&self, spec: &crate::TargetSpec) -> Option<&SessionMonitorHandle> {
        self.sessions.get(spec.session_name())
    }

    /// List currently active session names.
    pub fn active_sessions(&self) -> Vec<&str> {
        self.sessions
            .iter()
            .filter(|(_, h)| h.is_active())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Number of monitored sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_control_line tests ---

    #[test]
    fn parse_output_frame() {
        let msg = parse_control_line("%output %5 hello world");
        assert_eq!(
            msg,
            ControlModeMessage::Output {
                pane_id: "%5".to_string(),
                data: "hello world".to_string(),
            }
        );
    }

    #[test]
    fn parse_output_empty_data() {
        let msg = parse_control_line("%output %5");
        assert_eq!(
            msg,
            ControlModeMessage::Output {
                pane_id: "%5".to_string(),
                data: String::new(),
            }
        );
    }

    #[test]
    fn parse_output_with_octal() {
        let msg = parse_control_line("%output %5 hello\\012world");
        assert_eq!(
            msg,
            ControlModeMessage::Output {
                pane_id: "%5".to_string(),
                data: "hello\nworld".to_string(),
            }
        );
    }

    #[test]
    fn parse_begin() {
        let msg = parse_control_line("%begin 1234 1 0");
        assert!(matches!(msg, ControlModeMessage::CommandResponse(_)));
    }

    #[test]
    fn parse_end() {
        let msg = parse_control_line("%end 1234 1 0");
        assert!(matches!(msg, ControlModeMessage::CommandResponse(_)));
    }

    #[test]
    fn parse_error() {
        let msg = parse_control_line("%error 1234 1 0");
        assert!(matches!(msg, ControlModeMessage::CommandResponse(_)));
    }

    #[test]
    fn parse_notification() {
        let msg = parse_control_line("%session-changed $1 mysession");
        assert!(matches!(msg, ControlModeMessage::Notification(_)));
    }

    #[test]
    fn parse_unknown() {
        let msg = parse_control_line("some random output");
        assert!(matches!(msg, ControlModeMessage::Unknown(_)));
    }

    // --- decode_octal_escapes tests ---

    #[test]
    fn decode_newline() {
        assert_eq!(decode_octal_escapes("hello\\012world"), "hello\nworld");
    }

    #[test]
    fn decode_tab() {
        assert_eq!(decode_octal_escapes("hello\\011world"), "hello\tworld");
    }

    #[test]
    fn decode_space() {
        assert_eq!(decode_octal_escapes("\\040"), " ");
    }

    #[test]
    fn decode_backslash() {
        assert_eq!(decode_octal_escapes("hello\\\\world"), "hello\\world");
    }

    #[test]
    fn decode_no_escapes() {
        assert_eq!(decode_octal_escapes("hello world"), "hello world");
    }

    #[test]
    fn decode_multiple_escapes() {
        assert_eq!(
            decode_octal_escapes("a\\012b\\011c\\040d"),
            "a\nb\tc d"
        );
    }

    #[test]
    fn decode_partial_escape_at_end() {
        // Not enough digits for a full octal escape
        assert_eq!(decode_octal_escapes("hello\\01"), "hello\\01");
    }

    #[test]
    fn decode_utf8_multibyte() {
        // é is U+00E9, encoded in UTF-8 as bytes 0xC3 0xA9 (octal 303 251)
        assert_eq!(decode_octal_escapes("caf\\303\\251"), "café");
    }

    #[test]
    fn decode_utf8_three_byte() {
        // € is U+20AC, encoded in UTF-8 as bytes 0xE2 0x82 0xAC (octal 342 202 254)
        assert_eq!(decode_octal_escapes("\\342\\202\\254100"), "€100");
    }

    #[test]
    fn decode_utf8_four_byte() {
        // 🎉 is U+1F389, encoded as 0xF0 0x9F 0x8E 0x89 (octal 360 237 216 211)
        assert_eq!(
            decode_octal_escapes("\\360\\237\\216\\211"),
            "🎉"
        );
    }

    #[test]
    fn attach_command_default_socket() {
        assert_eq!(
            build_attach_command("build", None),
            "tmux -C attach-session -t 'build'\n"
        );
    }

    #[test]
    fn attach_command_named_socket() {
        assert_eq!(
            build_attach_command("build", Some(&TmuxSocket::Name("sock".into()))),
            "tmux -L 'sock' -C attach-session -t 'build'\n"
        );
    }

    #[test]
    fn attach_command_path_socket() {
        assert_eq!(
            build_attach_command("build", Some(&TmuxSocket::Path("/tmp/tmux.sock".into()))),
            "tmux -S '/tmp/tmux.sock' -C attach-session -t 'build'\n"
        );
    }

    // --- SessionMonitor.process_output tests ---

    #[test]
    fn process_output_sequence_monotonic() {
        let mut monitor = SessionMonitor::new("build".into(), "localhost".into());

        let out1 = monitor.process_output("%5", "hello");
        assert_eq!(out1.sequence, 1);
        assert_eq!(out1.content, "hello");
        assert_eq!(out1.host, "localhost");
        assert_eq!(out1.session_name(), "build");
        assert_eq!(out1.pane_id(), Some("%5"));

        let out2 = monitor.process_output("%5", "world");
        assert_eq!(out2.sequence, 2);
    }

    #[test]
    fn process_output_independent_panes() {
        let mut monitor = SessionMonitor::new("build".into(), "localhost".into());

        let out_a1 = monitor.process_output("%5", "a1");
        assert_eq!(out_a1.sequence, 1);

        let out_b1 = monitor.process_output("%6", "b1");
        assert_eq!(out_b1.sequence, 1); // Independent sequence

        let out_a2 = monitor.process_output("%5", "a2");
        assert_eq!(out_a2.sequence, 2);
    }

    // --- process_output normalization tests ---

    #[test]
    fn process_output_raw_preserves_ansi() {
        let mut monitor = SessionMonitor::new("build".into(), "localhost".into());
        let data = "hello \x1b[31mred\x1b[0m world\r\n";
        let out = monitor.process_output("%5", data);
        assert_eq!(out.content, data);
        assert!(out.raw_content.is_none());
    }

    #[test]
    fn process_output_screen_stable_normalizes() {
        let mut monitor = SessionMonitor::with_normalize(
            "build".into(),
            "localhost".into(),
            CaptureNormalizeMode::ScreenStable,
        );
        let data = "hello world  \r\n";
        let out = monitor.process_output("%5", data);
        // ScreenStable trims trailing whitespace and normalizes \r\n → \n
        assert_eq!(out.content, "hello world\n");
        // raw_content holds original when normalization changed the content
        assert_eq!(out.raw_content, Some(data.to_string()));
    }

    #[test]
    fn process_output_plain_text_strips_ansi() {
        let mut monitor = SessionMonitor::with_normalize(
            "build".into(),
            "localhost".into(),
            CaptureNormalizeMode::PlainText,
        );
        let data = "\x1b[32mgreen\x1b[0m text";
        let out = monitor.process_output("%5", data);
        assert_eq!(out.content, "green text\n");
        assert_eq!(out.raw_content, Some(data.to_string()));
    }

    #[test]
    fn process_output_raw_content_none_when_unchanged() {
        let mut monitor = SessionMonitor::with_normalize(
            "build".into(),
            "localhost".into(),
            CaptureNormalizeMode::PlainText,
        );
        // Content that normalizes to itself (already plain, with trailing \n)
        let data = "plain text\n";
        let out = monitor.process_output("%5", data);
        assert_eq!(out.content, "plain text\n");
        assert!(out.raw_content.is_none());
    }

    // --- MonitorHandle.stop_session / get_by_spec tests ---

    fn default_health() -> Arc<std::sync::Mutex<MonitorHealth>> {
        Arc::new(std::sync::Mutex::new(MonitorHealth::Streaming))
    }

    #[tokio::test]
    async fn monitor_handle_stop_session() {
        let target1 = crate::host::HostHandle::local().create_target_for_test("s1");
        let target2 = crate::host::HostHandle::local().create_target_for_test("s2");
        let (tx1, _) = watch::channel(false);
        let (tx2, _) = watch::channel(false);
        let task1 = tokio::spawn(async { Ok::<_, anyhow::Error>(MonitorExitReason::Stopped) });
        let task2 = tokio::spawn(async { Ok::<_, anyhow::Error>(MonitorExitReason::Stopped) });

        let mut sessions = HashMap::new();
        sessions.insert(
            "s1".to_string(),
            SessionMonitorHandle::new(target1, tx1, task1, default_health()),
        );
        sessions.insert(
            "s2".to_string(),
            SessionMonitorHandle::new(target2, tx2, task2, default_health()),
        );
        let mut handle = MonitorHandle::new(sessions);

        assert_eq!(handle.session_count(), 2);
        handle.stop_session("s1").await.unwrap();
        assert_eq!(handle.session_count(), 1);
        assert!(handle.get("s1").is_none());
        assert!(handle.get("s2").is_some());

        // Stopping non-existent session returns error
        assert!(handle.stop_session("s1").await.is_err());
    }

    #[test]
    fn monitor_handle_get_by_spec() {
        let target = crate::host::HostHandle::local().create_target_for_test("build");
        let (tx, _) = watch::channel(false);
        let task = tokio::runtime::Runtime::new().unwrap().spawn(async { Ok(MonitorExitReason::Stopped) });
        let mut sessions = HashMap::new();
        sessions.insert(
            "build".to_string(),
            SessionMonitorHandle::new(target, tx, task, default_health()),
        );
        let handle = MonitorHandle::new(sessions);

        let spec = crate::TargetSpec::session("build");
        assert!(handle.get_by_spec(&spec).is_some());

        let spec2 = crate::TargetSpec::session("nonexistent");
        assert!(handle.get_by_spec(&spec2).is_none());
    }

    // --- SessionMonitorHandle tests ---

    #[tokio::test]
    async fn monitor_handle_shutdown() {
        let target = crate::host::HostHandle::local().create_target_for_test(
            "test_session",
        );
        let (stop_tx, _stop_rx) = watch::channel(false);
        let task = tokio::spawn(async { Ok::<_, anyhow::Error>(MonitorExitReason::Stopped) });

        let handle = SessionMonitorHandle::new(target, stop_tx, task, default_health());
        // Shutdown waits for the task to finish
        handle.shutdown().await.unwrap();
        assert!(!handle.is_active());
    }

    // --- Pipeline integration test (2c.4a) ---

    #[tokio::test]
    async fn monitor_run_publishes_to_bus() {
        use crate::sink::{OutputBus, SinkEvent};
        use crate::transport::MockTransport;

        // Simulate control mode output: two %output frames then EOF
        let control_output = b"%output %5 hello world\n%output %6 second pane\n";
        let mock = MockTransport::new()
            .with_shell_data(vec![control_output.to_vec()]);
        let mut shell = mock.open_shell_for_test().await;

        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut rx = sub.into_receiver();

        let (_stop_tx, stop_rx) = watch::channel(false);
        let mut monitor = SessionMonitor::new("test".into(), "localhost".into());

        // Run monitor — it will process the canned data then hit EOF
        monitor.run(&mut shell, &bus, stop_rx).await.unwrap();

        // Verify two TargetOutput events were published
        match rx.try_recv().unwrap() {
            SinkEvent::Data(out) => {
                assert_eq!(out.content, "hello world");
                assert_eq!(out.pane_id(), Some("%5"));
                assert_eq!(out.session_name(), "test");
                assert_eq!(out.host, "localhost");
                assert_eq!(out.sequence, 1);
            }
            _ => panic!("expected Data"),
        }
        match rx.try_recv().unwrap() {
            SinkEvent::Data(out) => {
                assert_eq!(out.content, "second pane");
                assert_eq!(out.pane_id(), Some("%6"));
                assert_eq!(out.sequence, 1); // Independent pane, own sequence
            }
            _ => panic!("expected Data"),
        }
        // No more events
        assert!(rx.try_recv().is_err());
    }

    // --- MonitorHealth tests (DC29, 4.2a) ---

    #[tokio::test]
    async fn session_monitor_health_tracking() {
        let health = Arc::new(std::sync::Mutex::new(MonitorHealth::Streaming));
        let target = crate::host::HostHandle::local().create_target_for_test("test");
        let (stop_tx, _) = watch::channel(false);
        let task = tokio::spawn(async { Ok::<_, anyhow::Error>(MonitorExitReason::Stopped) });

        let handle = SessionMonitorHandle::new(target, stop_tx, task, health.clone());
        assert_eq!(handle.health(), MonitorHealth::Streaming);

        // Simulate health transition
        *health.lock().unwrap() = MonitorHealth::Reconnecting;
        assert_eq!(handle.health(), MonitorHealth::Reconnecting);

        *health.lock().unwrap() = MonitorHealth::Failed;
        assert_eq!(handle.health(), MonitorHealth::Failed);
    }

    #[tokio::test]
    async fn monitor_returns_connection_lost_on_eof() {
        use crate::sink::{OutputBus, SinkEvent};
        use crate::transport::MockTransport;

        let control_output = b"%output %5 hello\n";
        let mock = MockTransport::new()
            .with_shell_data(vec![control_output.to_vec()]);
        let mut shell = mock.open_shell_for_test().await;

        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut rx = sub.into_receiver();

        let (_stop_tx, stop_rx) = watch::channel(false);
        let mut monitor = SessionMonitor::new("test".into(), "localhost".into());

        let result = monitor.run(&mut shell, &bus, stop_rx).await.unwrap();
        assert_eq!(result, MonitorExitReason::ConnectionLost);

        // Verify data was still published before EOF
        match rx.try_recv().unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "hello"),
            other => panic!("expected Data, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn monitor_returns_stopped_on_signal() {
        use crate::sink::OutputBus;
        use crate::transport::MockTransport;

        // Shell that never EOFs (large data keeps flowing)
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(format!("%output %5 line{}\n", i).into_bytes());
        }
        let mock = MockTransport::new().with_shell_data(data);
        let mut shell = mock.open_shell_for_test().await;

        let bus = OutputBus::new();
        let _sub = bus.subscribe(vec![], 64).unwrap();

        let (stop_tx, stop_rx) = watch::channel(false);
        let mut monitor = SessionMonitor::new("test".into(), "localhost".into());

        // Stop immediately
        stop_tx.send(true).unwrap();

        let result = monitor.run(&mut shell, &bus, stop_rx).await.unwrap();
        assert_eq!(result, MonitorExitReason::Stopped);
    }
}
