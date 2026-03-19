//! Output sink pipeline: TargetOutput, SinkEvent, SinkFilter, OutputBus,
//! Subscription, JoinedStream, SinkKind (DC12, DC24).
//!
//! The pipeline has three layers:
//! - **Bus**: source routing (host/session/pane), fan-out, backpressure/gap tracking
//! - **Subscription adapters**: joining, piping to sinks (Track A); content matching,
//!   reacting (Track B, future)
//! - **Terminal consumers**: SinkKind (stdio, callback), custom code via into_receiver()

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::types::{OutputFidelity, TargetAddress};

// ---------------------------------------------------------------------------
// 2c.1a — Core sink types
// ---------------------------------------------------------------------------

/// Opaque subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SinkId(u64);

/// The unit of output flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct TargetOutput {
    /// Source entity — typically pane-level for control-mode output.
    pub source: TargetAddress,
    /// Host alias (e.g. "localhost", "web-1").
    pub host: String,
    /// Canonical content after normalization.
    pub content: String,
    /// Exact tmux capture before normalization, when requested.
    pub raw_content: Option<String>,
    /// Monotonic per-source sequence for gap detection.
    pub sequence: u64,
    /// Fidelity metadata.
    pub fidelity: OutputFidelity,
    /// Timestamp of emission.
    pub timestamp: Instant,
}

impl TargetOutput {
    /// Session name — available at any source level.
    pub fn session_name(&self) -> &str {
        match &self.source {
            TargetAddress::Session(s) => &s.name,
            TargetAddress::Window(w) => &w.session_name,
            TargetAddress::Pane(p) => &p.session,
        }
    }

    /// Pane ID — available when source is pane-level.
    pub fn pane_id(&self) -> Option<&str> {
        match &self.source {
            TargetAddress::Pane(p) => Some(&p.pane_id),
            _ => None,
        }
    }

    /// Tmux target string for the source.
    ///
    /// For pane-level sources, returns `pane_id` (e.g. `%5`) — the canonical
    /// identity. Window/pane indices may be synthetic for control-mode output.
    pub fn target_string(&self) -> String {
        match &self.source {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => p.pane_id.clone(),
        }
    }

    /// True when fidelity was degraded.
    pub fn degraded(&self) -> bool {
        self.fidelity.degraded
    }
}

/// Events delivered to subscribers.
#[derive(Debug)]
pub enum SinkEvent {
    /// Normal output data.
    Data(TargetOutput),
    /// Backpressure marker: this subscriber missed `dropped` events.
    Gap { dropped: usize, timestamp: Instant },
}

/// Source-routing filter (routing only, no content matching — DC24).
#[derive(Debug, Clone, Default)]
pub struct SinkFilter {
    /// Regex pattern against host alias.
    pub host: Option<String>,
    /// Regex pattern against session name.
    pub session: Option<String>,
    /// Regex pattern against "session:window_index".
    pub window: Option<String>,
    /// Regex pattern against pane_id or "session:window.pane".
    pub pane: Option<String>,
}

/// Compiled form of SinkFilter — regexes compiled once at subscribe() time.
pub struct CompiledSinkFilter {
    host: Option<regex::Regex>,
    session: Option<regex::Regex>,
    window: Option<regex::Regex>,
    pane: Option<regex::Regex>,
}

impl CompiledSinkFilter {
    /// Compile a SinkFilter into regexes.
    pub fn compile(filter: &SinkFilter) -> Result<Self> {
        let compile_opt = |opt: &Option<String>| -> Result<Option<regex::Regex>> {
            match opt {
                Some(pat) => Ok(Some(
                    regex::Regex::new(pat)
                        .map_err(|e| anyhow!("invalid filter regex '{}': {}", pat, e))?,
                )),
                None => Ok(None),
            }
        };
        Ok(CompiledSinkFilter {
            host: compile_opt(&filter.host)?,
            session: compile_opt(&filter.session)?,
            window: compile_opt(&filter.window)?,
            pane: compile_opt(&filter.pane)?,
        })
    }

    /// Returns true if output matches ALL non-None fields (AND within filter).
    /// None fields are wildcards (match everything).
    pub fn matches(&self, output: &TargetOutput) -> bool {
        if let Some(ref re) = self.host {
            if !re.is_match(&output.host) {
                return false;
            }
        }
        if let Some(ref re) = self.session {
            if !re.is_match(output.session_name()) {
                return false;
            }
        }
        if let Some(ref re) = self.window {
            let window_str = match &output.source {
                TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
                TargetAddress::Pane(p) => format!("{}:{}", p.session, p.window),
                TargetAddress::Session(s) => s.name.clone(),
            };
            if !re.is_match(&window_str) {
                return false;
            }
        }
        if let Some(ref re) = self.pane {
            let matched = match &output.source {
                TargetAddress::Pane(p) => {
                    // Match against pane_id (e.g. "%5") OR tmux target string
                    // (e.g. "build:0.1"). pane_id is the canonical identity from
                    // control mode; target string is available when indices are known.
                    re.is_match(&p.pane_id) || re.is_match(&p.to_tmux_target())
                }
                TargetAddress::Window(w) => {
                    re.is_match(&format!("{}:{}", w.session_name, w.index))
                }
                TargetAddress::Session(s) => re.is_match(&s.name),
            };
            if !matched {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// 2c.2a — SinkKind, Subscription, JoinedStream
// ---------------------------------------------------------------------------

/// Closed enum of all sink types. Static dispatch on the hot path (DC14).
/// Sinks are terminal consumers — they do not own routing filters (DC24).
/// Routing is owned by `OutputBus::subscribe(filters) -> Subscription`;
/// sinks are attached via `Subscription::pipe(SinkKind)`.
pub enum SinkKind {
    /// Writes to stdout with configurable formatting.
    Stdio(crate::sinks::stdio::StdioSink),
    /// User-provided sink via callback with explicit state.
    Callback(CallbackSink),
}

impl SinkKind {
    /// Human-readable name for logging.
    pub fn name(&self) -> &str {
        match self {
            SinkKind::Stdio(s) => s.name(),
            SinkKind::Callback(s) => &s.name,
        }
    }

    /// Process one sink event.
    pub async fn write(&mut self, event: SinkEvent) -> Result<()> {
        match self {
            SinkKind::Stdio(s) => s.write(event).await,
            SinkKind::Callback(s) => {
                (s.on_output)(&s.state, event)?;
                Ok(())
            }
        }
    }

    /// Flush internal buffers on shutdown.
    pub async fn flush(&mut self) -> Result<()> {
        match self {
            SinkKind::Stdio(s) => s.flush().await,
            SinkKind::Callback(s) => {
                if let Some(on_flush) = s.on_flush {
                    on_flush(&s.state).await?;
                }
                Ok(())
            }
        }
    }
}

/// User-provided sink via callback with explicit state.
pub struct CallbackSink {
    pub name: String,
    /// Shared state passed to callbacks.
    pub state: std::sync::Arc<dyn Any + Send + Sync>,
    /// Synchronous callback for each output event.
    pub on_output: fn(state: &std::sync::Arc<dyn Any + Send + Sync>, event: SinkEvent) -> Result<()>,
    /// Called on bus shutdown. Returns a boxed future.
    pub on_flush: Option<
        fn(
            state: &std::sync::Arc<dyn Any + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>,
    >,
}

// ---------------------------------------------------------------------------
// Subscription — the composable seam (DC24)
// ---------------------------------------------------------------------------

/// A subscription from the OutputBus. All consumer composition is layered
/// on this type via adapter methods.
pub struct Subscription {
    id: SinkId,
    rx: mpsc::Receiver<SinkEvent>,
}

impl Subscription {
    /// The subscription's unique identifier.
    pub fn id(&self) -> SinkId {
        self.id
    }

    /// Consume the subscription and return the raw receiver.
    pub fn into_receiver(self) -> mpsc::Receiver<SinkEvent> {
        self.rx
    }

    /// Pipe all events to a SinkKind. Spawns a task that drives the sink.
    /// Consumes the subscription. Returns a JoinHandle for the sink task.
    pub fn pipe(self, mut sink: SinkKind) -> JoinHandle<()> {
        let mut rx = self.rx;
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(e) = sink.write(event).await {
                    tracing::warn!(sink = sink.name(), "sink write error: {}", e);
                }
            }
            if let Err(e) = sink.flush().await {
                tracing::warn!(sink = sink.name(), "sink flush error: {}", e);
            }
        })
    }

    /// Create a JoinedStream that merges events with source attribution.
    /// Consumes the subscription.
    pub fn joined(self, label_format: LabelFormat) -> JoinedStream {
        JoinedStream {
            rx: self.rx,
            label_format,
            last_source: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JoinedStream — multi-source consolidated view (DC15, DC24)
// ---------------------------------------------------------------------------

/// Identifies the source of a chunk in a joined stream.
#[derive(Debug, Clone)]
pub struct SourceLabel {
    pub host: String,
    pub target: TargetAddress,
}

impl SourceLabel {
    pub fn from_output(output: &TargetOutput) -> Self {
        SourceLabel {
            host: output.host.clone(),
            target: output.source.clone(),
        }
    }

    /// "web-1:build(%5)" (pane) or "web-1:build" (session).
    ///
    /// Pane-level sources use `pane_id` as the canonical identity — control
    /// mode only provides `%<id>`, not window/pane indices. Using `pane_id`
    /// ensures distinct panes in the same session are distinguishable.
    pub fn short(&self) -> String {
        match &self.target {
            TargetAddress::Session(s) => format!("{}:{}", self.host, s.name),
            TargetAddress::Window(w) => format!("{}:{}:{}", self.host, w.session_name, w.index),
            TargetAddress::Pane(p) => format!("{}:{}({})", self.host, p.session, p.pane_id),
        }
    }

    /// "build(%5)" or "build" (no host prefix).
    pub fn minimal(&self) -> String {
        match &self.target {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => format!("{}({})", p.session, p.pane_id),
        }
    }

    /// Compare labels for source coalescing.
    ///
    /// For pane-level targets, compares using `pane_id` (the authoritative
    /// identity from tmux) rather than display indices which may be synthetic.
    fn same_source(&self, other: &SourceLabel) -> bool {
        if self.host != other.host {
            return false;
        }
        match (&self.target, &other.target) {
            (TargetAddress::Pane(a), TargetAddress::Pane(b)) => {
                a.session == b.session && a.pane_id == b.pane_id
            }
            _ => self.short() == other.short(),
        }
    }
}

/// A chunk in the joined stream.
pub struct StreamChunk {
    pub source: SourceLabel,
    pub output: TargetOutput,
    /// True when the source differs from the previous chunk's source.
    pub source_changed: bool,
}

/// Label format for JoinedStream rendering.
pub enum LabelFormat {
    /// "[web-1:build:0.1] content"
    Bracketed,
    /// "web-1:build:0.1> content"
    Prompt,
    /// Custom formatting function.
    Custom(fn(&SourceLabel, &str) -> String),
}

/// Multi-source consolidated view returned by `Subscription::joined()`.
pub struct JoinedStream {
    rx: mpsc::Receiver<SinkEvent>,
    label_format: LabelFormat,
    last_source: Option<SourceLabel>,
}

impl JoinedStream {
    /// Receive the next chunk. Returns None when the subscription closes.
    /// Gap events are skipped (they are a bus-level concern).
    pub async fn next(&mut self) -> Option<StreamChunk> {
        loop {
            match self.rx.recv().await {
                Some(SinkEvent::Data(output)) => {
                    let source = SourceLabel::from_output(&output);
                    let source_changed = match &self.last_source {
                        Some(prev) => !prev.same_source(&source),
                        None => true,
                    };
                    self.last_source = Some(source.clone());
                    return Some(StreamChunk {
                        source,
                        output,
                        source_changed,
                    });
                }
                Some(SinkEvent::Gap { .. }) => continue,
                None => return None,
            }
        }
    }

    /// Format a chunk with the configured label format.
    pub fn format(&self, chunk: &StreamChunk) -> String {
        let label = chunk.source.short();
        let content = &chunk.output.content;
        match &self.label_format {
            LabelFormat::Bracketed => format!("[{}] {}", label, content),
            LabelFormat::Prompt => format!("{}> {}", label, content),
            LabelFormat::Custom(f) => f(&chunk.source, content),
        }
    }
}

// ---------------------------------------------------------------------------
// 2c.3 — OutputBus
// ---------------------------------------------------------------------------

/// Internal subscriber entry.
struct SubEntry {
    id: SinkId,
    #[allow(dead_code)]
    name: String,
    tx: mpsc::Sender<SinkEvent>,
    filters: Vec<CompiledSinkFilter>,
    /// Events dropped due to channel backpressure since last successful send.
    dropped: usize,
}

/// Central fan-out dispatcher (DC12, DC24).
///
/// Uses interior mutability so it can be shared via `Arc<OutputBus>`.
/// All methods take `&self`.
pub struct OutputBus {
    subscribers: std::sync::Mutex<Vec<SubEntry>>,
    next_id: AtomicU64,
}

impl OutputBus {
    pub fn new() -> Self {
        OutputBus {
            subscribers: std::sync::Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Subscribe with source-routing filters. Returns a Subscription.
    /// Empty filters vec means "match all output".
    pub fn subscribe(
        &self,
        filters: Vec<SinkFilter>,
        channel_capacity: usize,
    ) -> Result<Subscription> {
        let compiled: Vec<CompiledSinkFilter> = filters
            .iter()
            .map(CompiledSinkFilter::compile)
            .collect::<Result<Vec<_>>>()?;

        let id = SinkId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::channel(channel_capacity);
        let name = format!("sub-{}", id.0);

        let entry = SubEntry {
            id,
            name,
            tx,
            filters: compiled,
            dropped: 0,
        };

        self.subscribers.lock().expect("bus lock poisoned").push(entry);
        Ok(Subscription { id, rx })
    }

    /// Remove a subscription by id.
    ///
    /// Drops the sender, closing the channel. If a piped task was spawned via
    /// `Subscription::pipe()`, it will drain remaining buffered events and exit
    /// on its own — this method does **not** await the task. Callers that need
    /// join semantics should hold the `JoinHandle` returned by `pipe()`.
    pub fn unsubscribe(&self, id: SinkId) -> Result<()> {
        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        let len_before = subs.len();
        subs.retain(|s| s.id != id);
        if subs.len() == len_before {
            return Err(anyhow!("subscription {:?} not found", id));
        }
        Ok(())
    }

    /// Fan out to all matching subscribers. Non-blocking (try_send).
    pub fn publish(&self, output: TargetOutput) {
        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        for sub in subs.iter_mut() {
            // Check filters: empty filters = match all; else OR across filters
            let matched = sub.filters.is_empty()
                || sub.filters.iter().any(|f| f.matches(&output));
            if !matched {
                continue;
            }

            // If there were prior drops, send a Gap event first
            if sub.dropped > 0 {
                let gap = SinkEvent::Gap {
                    dropped: sub.dropped,
                    timestamp: Instant::now(),
                };
                match sub.tx.try_send(gap) {
                    Ok(()) => {
                        sub.dropped = 0;
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        // Can't even send the gap — count continues accumulating
                        sub.dropped += 1;
                        continue;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => continue,
                }
            }

            // Send the data event
            match sub.tx.try_send(SinkEvent::Data(output.clone())) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    sub.dropped += 1;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {}
            }
        }
    }

    /// Drop all senders, closing all receiver channels.
    ///
    /// Piped tasks (spawned via `Subscription::pipe()`) will drain remaining
    /// buffered events and exit on their own — this method does **not** track
    /// or await those tasks. Callers that need flush-and-join semantics should
    /// hold the `JoinHandle` returned by `pipe()` and await it after calling
    /// `shutdown()`.
    pub fn shutdown(&self) {
        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        subs.clear();
    }

    /// Number of active subscribers (for diagnostics).
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.lock().expect("bus lock poisoned").len()
    }
}

impl Default for OutputBus {
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
    use crate::types::{OutputFidelity, PaneAddress, SessionInfo};
    use std::sync::Arc;

    fn make_output(host: &str, session: &str, pane_id: &str, content: &str, seq: u64) -> TargetOutput {
        TargetOutput {
            source: TargetAddress::Pane(PaneAddress {
                pane_id: pane_id.to_string(),
                session: session.to_string(),
                window: 0,
                pane: 0,
            }),
            host: host.to_string(),
            content: content.to_string(),
            raw_content: None,
            sequence: seq,
            fidelity: OutputFidelity::clean(),
            timestamp: Instant::now(),
        }
    }

    fn make_session_output(host: &str, session: &str, content: &str) -> TargetOutput {
        TargetOutput {
            source: TargetAddress::Session(SessionInfo {
                name: session.to_string(),
                id: "$0".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            }),
            host: host.to_string(),
            content: content.to_string(),
            raw_content: None,
            sequence: 1,
            fidelity: OutputFidelity::clean(),
            timestamp: Instant::now(),
        }
    }

    // --- TargetOutput accessor tests ---

    #[test]
    fn target_output_pane_accessors() {
        let out = make_output("web-1", "build", "%5", "hello", 1);
        assert_eq!(out.session_name(), "build");
        assert_eq!(out.pane_id(), Some("%5"));
        assert_eq!(out.target_string(), "%5");
        assert!(!out.degraded());
    }

    #[test]
    fn target_output_session_accessors() {
        let out = make_session_output("web-1", "build", "hello");
        assert_eq!(out.session_name(), "build");
        assert_eq!(out.pane_id(), None);
        assert_eq!(out.target_string(), "build");
    }

    // --- SinkFilter / CompiledSinkFilter tests ---

    #[test]
    fn compiled_filter_matches_all_when_empty() {
        let filter = SinkFilter::default();
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        let out = make_output("web-1", "build", "%5", "hello", 1);
        assert!(compiled.matches(&out));
    }

    #[test]
    fn compiled_filter_host_match() {
        let filter = SinkFilter {
            host: Some("web-.*".to_string()),
            ..Default::default()
        };
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("web-1", "build", "%5", "hello", 1)));
        assert!(!compiled.matches(&make_output("db-1", "build", "%5", "hello", 1)));
    }

    #[test]
    fn compiled_filter_session_match() {
        let filter = SinkFilter {
            session: Some("build".to_string()),
            ..Default::default()
        };
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("web-1", "build", "%5", "hello", 1)));
        assert!(!compiled.matches(&make_output("web-1", "deploy", "%5", "hello", 1)));
    }

    #[test]
    fn compiled_filter_and_semantics() {
        let filter = SinkFilter {
            host: Some("web-1".to_string()),
            session: Some("build".to_string()),
            ..Default::default()
        };
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        // Both match
        assert!(compiled.matches(&make_output("web-1", "build", "%5", "hello", 1)));
        // Host doesn't match
        assert!(!compiled.matches(&make_output("db-1", "build", "%5", "hello", 1)));
        // Session doesn't match
        assert!(!compiled.matches(&make_output("web-1", "deploy", "%5", "hello", 1)));
    }

    #[test]
    fn compiled_filter_invalid_regex() {
        let filter = SinkFilter {
            host: Some("[invalid".to_string()),
            ..Default::default()
        };
        assert!(CompiledSinkFilter::compile(&filter).is_err());
    }

    // --- OutputBus tests ---

    #[tokio::test]
    async fn bus_fanout_to_multiple_subscribers() {
        let bus = OutputBus::new();
        let sub1 = bus.subscribe(vec![], 16).unwrap();
        let sub2 = bus.subscribe(vec![], 16).unwrap();
        let sub3 = bus.subscribe(vec![], 16).unwrap();

        bus.publish(make_output("h", "s", "%1", "hello", 1));

        let mut rx1 = sub1.into_receiver();
        let mut rx2 = sub2.into_receiver();
        let mut rx3 = sub3.into_receiver();

        assert!(matches!(rx1.recv().await, Some(SinkEvent::Data(_))));
        assert!(matches!(rx2.recv().await, Some(SinkEvent::Data(_))));
        assert!(matches!(rx3.recv().await, Some(SinkEvent::Data(_))));
    }

    #[tokio::test]
    async fn bus_source_routing_filter() {
        let bus = OutputBus::new();
        let sub = bus
            .subscribe(
                vec![SinkFilter {
                    host: Some("web-1".to_string()),
                    ..Default::default()
                }],
                16,
            )
            .unwrap();

        bus.publish(make_output("web-1", "s", "%1", "yes", 1));
        bus.publish(make_output("db-1", "s", "%1", "no", 2));

        let mut rx = sub.into_receiver();
        let event = rx.recv().await.unwrap();
        match event {
            SinkEvent::Data(out) => assert_eq!(out.content, "yes"),
            _ => panic!("expected Data"),
        }
        // Channel should be empty — db-1 was filtered
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn bus_or_across_filters() {
        let bus = OutputBus::new();
        let sub = bus
            .subscribe(
                vec![
                    SinkFilter {
                        host: Some("web-1".to_string()),
                        ..Default::default()
                    },
                    SinkFilter {
                        session: Some("build".to_string()),
                        ..Default::default()
                    },
                ],
                16,
            )
            .unwrap();

        // Matches first filter (host)
        bus.publish(make_output("web-1", "deploy", "%1", "a", 1));
        // Matches second filter (session)
        bus.publish(make_output("db-1", "build", "%1", "b", 2));
        // Matches neither
        bus.publish(make_output("db-1", "deploy", "%1", "c", 3));

        let mut rx = sub.into_receiver();
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "a"),
            _ => panic!("expected Data"),
        }
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "b"),
            _ => panic!("expected Data"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn bus_gap_on_backpressure() {
        let bus = OutputBus::new();
        // Capacity 2 so we can receive Gap + Data after draining
        let sub = bus.subscribe(vec![], 2).unwrap();

        // Fill the channel (capacity 2)
        bus.publish(make_output("h", "s", "%1", "first", 1));
        bus.publish(make_output("h", "s", "%1", "second", 2));
        // These should be dropped (channel full)
        bus.publish(make_output("h", "s", "%1", "dropped1", 3));
        bus.publish(make_output("h", "s", "%1", "dropped2", 4));

        let mut rx = sub.into_receiver();
        // Drain both slots
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "first"),
            _ => panic!("expected Data"),
        }
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "second"),
            _ => panic!("expected Data"),
        }

        // Now there's room — next publish should send Gap then Data
        bus.publish(make_output("h", "s", "%1", "after", 5));

        match rx.recv().await.unwrap() {
            SinkEvent::Gap { dropped, .. } => assert_eq!(dropped, 2),
            other => panic!("expected Gap, got {:?}", other),
        }
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "after"),
            _ => panic!("expected Data after Gap"),
        }
    }

    #[tokio::test]
    async fn bus_unsubscribe() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let id = sub.id();
        assert_eq!(bus.subscriber_count(), 1);

        bus.unsubscribe(id).unwrap();
        assert_eq!(bus.subscriber_count(), 0);

        // Receiver should be closed
        let mut rx = sub.into_receiver();
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn bus_shutdown_closes_all() {
        let bus = OutputBus::new();
        let sub1 = bus.subscribe(vec![], 16).unwrap();
        let sub2 = bus.subscribe(vec![], 16).unwrap();

        bus.shutdown();
        assert_eq!(bus.subscriber_count(), 0);

        let mut rx1 = sub1.into_receiver();
        let mut rx2 = sub2.into_receiver();
        assert!(rx1.recv().await.is_none());
        assert!(rx2.recv().await.is_none());
    }

    // --- JoinedStream tests ---

    #[tokio::test]
    async fn joined_stream_source_changed() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut joined = sub.joined(LabelFormat::Bracketed);

        // Two outputs from same source
        bus.publish(make_output("h", "s", "%1", "a", 1));
        bus.publish(make_output("h", "s", "%1", "b", 2));
        // Output from different source (different session)
        bus.publish(make_output("h", "other", "%2", "c", 3));

        let c1 = joined.next().await.unwrap();
        assert!(c1.source_changed); // first chunk always "changed"
        assert_eq!(c1.output.content, "a");

        let c2 = joined.next().await.unwrap();
        assert!(!c2.source_changed); // same source
        assert_eq!(c2.output.content, "b");

        let c3 = joined.next().await.unwrap();
        assert!(c3.source_changed); // different pane
        assert_eq!(c3.output.content, "c");
    }

    #[tokio::test]
    async fn joined_stream_format() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut joined = sub.joined(LabelFormat::Bracketed);

        bus.publish(make_output("web-1", "build", "%5", "hello", 1));
        let chunk = joined.next().await.unwrap();
        let formatted = joined.format(&chunk);
        assert_eq!(formatted, "[web-1:build(%5)] hello");
    }

    #[tokio::test]
    async fn joined_stream_prompt_format() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut joined = sub.joined(LabelFormat::Prompt);

        bus.publish(make_output("web-1", "build", "%5", "hello", 1));
        let chunk = joined.next().await.unwrap();
        let formatted = joined.format(&chunk);
        assert_eq!(formatted, "web-1:build(%5)> hello");
    }

    #[tokio::test]
    async fn joined_stream_closes_on_bus_shutdown() {
        let bus = Arc::new(OutputBus::new());
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut joined = sub.joined(LabelFormat::Bracketed);

        bus.shutdown();
        assert!(joined.next().await.is_none());
    }

    // --- SourceLabel tests ---

    #[test]
    fn source_label_short_pane() {
        let label = SourceLabel {
            host: "web-1".to_string(),
            target: TargetAddress::Pane(PaneAddress {
                pane_id: "%5".to_string(),
                session: "build".to_string(),
                window: 0,
                pane: 1,
            }),
        };
        assert_eq!(label.short(), "web-1:build(%5)");
        assert_eq!(label.minimal(), "build(%5)");
    }

    #[test]
    fn source_label_short_session() {
        let label = SourceLabel {
            host: "web-1".to_string(),
            target: TargetAddress::Session(SessionInfo {
                name: "build".to_string(),
                id: "$0".to_string(),
                created: 0,
                attached: false,
                window_count: 1,
                group: None,
            }),
        };
        assert_eq!(label.short(), "web-1:build");
        assert_eq!(label.minimal(), "build");
    }

    // --- Pane identity tests ---

    #[test]
    fn source_label_same_source_distinguishes_panes() {
        // Two panes in the same session with different pane_ids but same
        // synthetic indices (as control mode produces) must be distinct.
        let label_a = SourceLabel {
            host: "h".to_string(),
            target: TargetAddress::Pane(PaneAddress {
                pane_id: "%5".to_string(),
                session: "build".to_string(),
                window: 0,
                pane: 0,
            }),
        };
        let label_b = SourceLabel {
            host: "h".to_string(),
            target: TargetAddress::Pane(PaneAddress {
                pane_id: "%6".to_string(),
                session: "build".to_string(),
                window: 0,
                pane: 0,
            }),
        };
        assert!(!label_a.same_source(&label_b));
        assert!(label_a.same_source(&label_a));
    }

    #[test]
    fn compiled_filter_pane_id_match() {
        // Filter using pane_id pattern should match monitor-originated output
        let filter = SinkFilter {
            pane: Some("%5".to_string()),
            ..Default::default()
        };
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("h", "build", "%5", "hello", 1)));
        assert!(!compiled.matches(&make_output("h", "build", "%6", "hello", 1)));
    }

    #[tokio::test]
    async fn joined_stream_source_changed_distinct_panes() {
        // Two panes with same session but different pane_ids must trigger
        // source_changed, even with identical synthetic window/pane indices.
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let mut joined = sub.joined(LabelFormat::Bracketed);

        bus.publish(make_output("h", "s", "%5", "a", 1));
        bus.publish(make_output("h", "s", "%6", "b", 1));

        let c1 = joined.next().await.unwrap();
        assert!(c1.source_changed);

        let c2 = joined.next().await.unwrap();
        assert!(c2.source_changed); // different pane_id = different source
    }

    // --- Subscription::pipe test ---

    #[tokio::test]
    async fn subscription_pipe_drives_sink() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();

        let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone: Arc<dyn Any + Send + Sync> = events.clone();

        let sink = SinkKind::Callback(CallbackSink {
            name: "test".to_string(),
            state: events_clone,
            on_output: |state, event| {
                let events = state
                    .downcast_ref::<std::sync::Mutex<Vec<String>>>()
                    .unwrap();
                if let SinkEvent::Data(out) = event {
                    events.lock().unwrap().push(out.content.clone());
                }
                Ok(())
            },
            on_flush: None,
        });

        let handle = sub.pipe(sink);

        bus.publish(make_output("h", "s", "%1", "msg1", 1));
        bus.publish(make_output("h", "s", "%1", "msg2", 2));

        // Shutdown bus to close the channel, causing pipe task to finish
        bus.shutdown();
        handle.await.unwrap();

        let collected = events.lock().unwrap();
        assert_eq!(*collected, vec!["msg1", "msg2"]);
    }
}
