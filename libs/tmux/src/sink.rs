//! Output sink pipeline: TargetOutput, SinkEvent, SinkFilter, OutputBus,
//! Subscription, JoinedStream, HistoryHandle, SinkKind (DC12, DC24, DC28).
//!
//! The pipeline has three layers:
//! - **Bus**: source routing (host/session/pane), fan-out, backpressure/gap tracking
//! - **Subscription adapters**: joining, piping to sinks (Track A); `filter_fn` predicate
//!   filtering, `history()` rolling transcript/history (Track B, DC28)
//! - **Terminal consumers**: SinkKind (stdio, callback), custom code via into_receiver()

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime};

use crate::error::{Error, Result};
use tokio::sync::{mpsc, Mutex};
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
    /// Per-source sequence for gap detection. Monotonically increasing within
    /// a continuous stream segment. A `SinkEvent::Discontinuity` resets the
    /// sequence epoch — consumers must not assume monotonicity across
    /// discontinuity boundaries.
    pub sequence: u64,
    /// Fidelity metadata.
    pub fidelity: OutputFidelity,
    /// Daemon-side receipt timestamp for this output frame.
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

    /// Canonical source identity for bus routing, filter matching, and source
    /// coalescing. Returns `pane_id` for pane sources (the authoritative
    /// identity from tmux control mode), session name for sessions, and
    /// `session:window_index` for windows.
    pub fn source_key(&self) -> String {
        match &self.source {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => p.pane_id.clone(),
        }
    }

    /// Human-readable tmux target string for display/logging.
    ///
    /// Returns `session:window.pane` format using the display indices stored
    /// in `PaneAddress`. For control-mode output where window/pane indices are
    /// synthetic (0/0), prefer `source_key()` or `pane_id()` for identity.
    pub fn target_string(&self) -> String {
        match &self.source {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => p.to_tmux_target(),
        }
    }

    /// True when fidelity was degraded.
    pub fn degraded(&self) -> bool {
        self.fidelity.degraded
    }
}

/// Events delivered to subscribers.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SinkEvent {
    /// Normal output data.
    Data(TargetOutput),
    /// Backpressure marker: this subscriber missed `dropped` events.
    Gap { dropped: usize, timestamp: Instant },
    /// Upstream monitor discontinuity (DC29). The monitor/transport continuity
    /// was broken (e.g., control-mode EOF, SSH disconnect, tmux server restart).
    /// Distinct from `Gap` which is subscriber-local backpressure.
    Discontinuity { reason: String },
}

/// Policy for events that arrive outside a timestamp-merge reorder window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateEventPolicy {
    /// Append the event and mark it as late.
    AppendWithMarker,
}

/// Ordering policy for bus-owned timelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimelineOrdering {
    /// Preserve bus ingest order.
    #[default]
    Arrival,
    /// Insert output by daemon-side receipt timestamp within a bounded reorder window.
    TimestampMerge {
        /// Maximum age behind the newest observed receipt timestamp before an
        /// event is considered late.
        reorder_window: Duration,
        /// How late arrivals are represented.
        late_event_policy: LateEventPolicy,
    },
}

/// Options for a named [`OutputBus`] timeline.
#[derive(Debug, Clone)]
pub struct TimelineOptions {
    /// Source-routing filters. Empty means "all output".
    pub filters: Vec<SinkFilter>,
    /// Maximum entries retained in the ring buffer.
    pub max_entries: usize,
    /// Maximum rendered characters retained. `0` disables character trimming.
    pub max_render_chars: usize,
    /// Ordering policy.
    pub ordering: TimelineOrdering,
    /// Rendering mode for prompt-ready text.
    pub render_mode: RenderMode,
    /// Source label format for rendering.
    pub label_format: LabelFormat,
    /// Include omission markers when rendering retained windows.
    pub include_omission_marker: bool,
}

impl Default for TimelineOptions {
    fn default() -> Self {
        TimelineOptions {
            filters: Vec::new(),
            max_entries: 10_000,
            max_render_chars: 0,
            ordering: TimelineOrdering::Arrival,
            render_mode: RenderMode::Interleaved,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
        }
    }
}

/// Source identity for timeline gap and discontinuity markers.
///
/// Empty scope is a global marker and only matches unfiltered timelines. Scoped
/// markers match timelines using the same [`SinkFilter`] rules as output data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimelineMarkerScope {
    /// Host alias associated with the marker.
    pub host: Option<String>,
    /// Session name associated with the marker.
    pub session: Option<String>,
    /// Window target string associated with the marker.
    pub window: Option<String>,
    /// Pane id or target string associated with the marker.
    pub pane: Option<String>,
}

impl TimelineMarkerScope {
    /// Global marker for unfiltered timelines.
    pub fn global() -> Self {
        Self::default()
    }

    /// Marker scoped to a host alias.
    pub fn for_host(host: &str) -> Self {
        Self {
            host: Some(host.to_string()),
            ..Default::default()
        }
    }

    /// Marker scoped to a session on any host.
    pub fn for_session(session: &str) -> Self {
        Self {
            session: Some(session.to_string()),
            ..Default::default()
        }
    }

    /// Marker scoped to a host/session pair.
    pub fn for_host_session(host: &str, session: &str) -> Self {
        Self {
            host: Some(host.to_string()),
            session: Some(session.to_string()),
            ..Default::default()
        }
    }

    /// Marker scoped to a pane id or target string.
    pub fn for_pane(pane: &str) -> Self {
        Self {
            pane: Some(pane.to_string()),
            ..Default::default()
        }
    }
}

/// Stable cursor for incremental timeline polling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineCursor {
    /// Lowest timeline sequence number that has not been fully drained.
    pub next_sequence: u64,
    /// Out-of-order sequences already returned while lower retained sequences
    /// are still pending. This lets timestamp-ordered pages resume without
    /// replaying or skipping entries.
    pub seen_sequences: Vec<u64>,
}

impl Default for TimelineCursor {
    fn default() -> Self {
        TimelineCursor {
            next_sequence: 1,
            seen_sequences: Vec::new(),
        }
    }
}

/// Options for rendering a timeline window.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    /// Maximum characters to return. `0` uses the timeline's configured budget.
    pub max_chars: usize,
}

/// Kind of event retained by a timeline.
#[derive(Debug, Clone)]
pub enum TimelineEntryKind {
    /// Normal output data.
    Output,
    /// Backpressure or synthetic gap marker.
    Gap { dropped_events: usize },
    /// Upstream monitor discontinuity marker.
    Discontinuity { reason: String },
}

/// A retained timeline entry with source metadata and continuity markers.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// Monotonic sequence assigned by the timeline.
    pub sequence: u64,
    /// Continuity epoch. Incremented after discontinuity markers.
    pub discontinuity_epoch: u64,
    /// Entry kind.
    pub kind: TimelineEntryKind,
    /// Source label for output entries.
    pub source: Option<SourceLabel>,
    /// Full target identity for output entries.
    pub target: Option<TargetAddress>,
    /// Host alias for output entries.
    pub host: Option<String>,
    /// Session name for output entries.
    pub session: Option<String>,
    /// Pane id for output entries when pane-level.
    pub pane_id: Option<String>,
    /// Output content for data entries.
    pub content: Option<String>,
    /// Per-source output sequence for data entries.
    pub output_sequence: Option<u64>,
    /// Daemon-side receipt timestamp from `TargetOutput`.
    pub received_at: Option<Instant>,
    /// Estimated wall-clock time corresponding to `received_at`.
    pub received_at_wall: Option<SystemTime>,
    /// Bus ingest timestamp.
    pub ingested_at: Instant,
    /// Wall-clock bus ingest timestamp for JSONL-friendly consumers.
    pub ingested_at_wall: SystemTime,
    /// True when timestamp ordering observed this entry outside the reorder window.
    pub late: bool,
}

impl TimelineEntry {
    fn rendered_chars(&self, label_format: &LabelFormat) -> usize {
        self.render(label_format).chars().count()
    }

    fn render_source_key(&self) -> String {
        self.source
            .as_ref()
            .map(SourceLabel::short)
            .unwrap_or_else(|| "__system__".to_string())
    }

    fn render(&self, label_format: &LabelFormat) -> String {
        match &self.kind {
            TimelineEntryKind::Output => {
                let Some(source) = &self.source else {
                    return String::new();
                };
                let content = self.content.as_deref().unwrap_or_default();
                let marker = if self.late { " [late]" } else { "" };
                match label_format {
                    LabelFormat::Bracketed => {
                        format!("[{}{}] {}\n", source.short(), marker, content)
                    }
                    LabelFormat::Prompt => format!("{}{}> {}\n", source.short(), marker, content),
                    LabelFormat::Custom(f) => {
                        let rendered = f(source, content);
                        if self.late {
                            format!("[late] {}\n", rendered)
                        } else {
                            format!("{}\n", rendered)
                        }
                    }
                }
            }
            TimelineEntryKind::Gap { dropped_events } => {
                format!("[gap: {} event(s) dropped]\n", dropped_events)
            }
            TimelineEntryKind::Discontinuity { reason } => format!("[{}]\n", reason),
        }
    }
}

/// Page of timeline entries returned by incremental queries.
#[derive(Debug, Clone)]
pub struct TimelinePage {
    pub entries: Vec<TimelineEntry>,
    pub cursor: TimelineCursor,
    pub omitted_entries: usize,
}

/// Rendered timeline text plus the next cursor.
#[derive(Debug, Clone)]
pub struct TimelineRenderPage {
    pub text: String,
    pub cursor: TimelineCursor,
    pub omitted_entries: usize,
}

type TimelineStateHandle = Arc<std::sync::Mutex<TimelineState>>;
type TimelineRegistry = Arc<std::sync::Mutex<HashMap<String, TimelineStateHandle>>>;
type WeakTimelineRegistry = Weak<std::sync::Mutex<HashMap<String, TimelineStateHandle>>>;

fn timeline_lock_error() -> Error {
    Error::State("timeline lock poisoned".to_string())
}

fn timeline_stale_error(name: &str, generation: u64) -> Error {
    Error::State(format!(
        "timeline '{}' generation {} is detached",
        name, generation
    ))
}

fn compile_timeline_filters(filters: &[SinkFilter]) -> Result<Vec<CompiledSinkFilter>> {
    filters
        .iter()
        .map(CompiledSinkFilter::compile)
        .collect::<Result<Vec<_>>>()
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

fn truncate_to_chars(text: &mut String, max_chars: usize) {
    if max_chars == 0 {
        return;
    }
    if let Some((byte_idx, _)) = text.char_indices().nth(max_chars) {
        text.truncate(byte_idx);
    }
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

impl SinkFilter {
    /// Filter matching a specific host (exact match).
    pub fn for_host(host: &str) -> Self {
        SinkFilter {
            host: Some(format!("^{}$", regex::escape(host))),
            ..Default::default()
        }
    }

    /// Filter matching a specific session (exact match).
    pub fn for_session(session: &str) -> Self {
        SinkFilter {
            session: Some(format!("^{}$", regex::escape(session))),
            ..Default::default()
        }
    }

    /// Filter matching a specific pane by pane_id (exact match, e.g. `%5`).
    pub fn for_pane(pane_id: &str) -> Self {
        SinkFilter {
            pane: Some(format!("^{}$", regex::escape(pane_id))),
            ..Default::default()
        }
    }

    /// Filter matching a specific host and session (exact match on both).
    pub fn for_host_session(host: &str, session: &str) -> Self {
        SinkFilter {
            host: Some(format!("^{}$", regex::escape(host))),
            session: Some(format!("^{}$", regex::escape(session))),
            ..Default::default()
        }
    }
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
                Some(pat) => Ok(Some(regex::Regex::new(pat).map_err(|e| {
                    Error::Parse(format!("invalid filter regex '{}': {}", pat, e))
                })?)),
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
                TargetAddress::Window(w) => re.is_match(&format!("{}:{}", w.session_name, w.index)),
                TargetAddress::Session(s) => re.is_match(&s.name),
            };
            if !matched {
                return false;
            }
        }
        true
    }

    fn matches_marker_scope(&self, scope: &TimelineMarkerScope) -> bool {
        if let Some(ref re) = self.host {
            match scope.host.as_deref() {
                Some(host) if re.is_match(host) => {}
                _ => return false,
            }
        }
        if let Some(ref re) = self.session {
            match scope.session.as_deref() {
                Some(session) if re.is_match(session) => {}
                _ => return false,
            }
        }
        if let Some(ref re) = self.window {
            let value = scope.window.as_deref().or(scope.session.as_deref());
            match value {
                Some(window) if re.is_match(window) => {}
                _ => return false,
            }
        }
        if let Some(ref re) = self.pane {
            let value = scope
                .pane
                .as_deref()
                .or(scope.window.as_deref())
                .or(scope.session.as_deref());
            match value {
                Some(pane) if re.is_match(pane) => {}
                _ => return false,
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
    pub state: CallbackState,
    /// Synchronous callback for each output event.
    pub on_output: fn(state: &CallbackState, event: SinkEvent) -> Result<()>,
    /// Called on bus shutdown. Returns a boxed future.
    pub on_flush: Option<CallbackFlush>,
}

pub type CallbackState = std::sync::Arc<dyn Any + Send + Sync>;
pub type CallbackFlush =
    fn(state: &CallbackState) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;

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
    /// Consumes the subscription. Returns a `PipeHandle` that combines
    /// the subscription id (for bus control) with the task handle (for
    /// flush/join semantics) in a single ownership unit.
    pub fn pipe(self, mut sink: SinkKind) -> PipeHandle {
        let id = self.id;
        let mut rx = self.rx;
        let task = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(e) = sink.write(event).await {
                    tracing::warn!(sink = sink.name(), "sink write error: {}", e);
                }
            }
            if let Err(e) = sink.flush().await {
                tracing::warn!(sink = sink.name(), "sink flush error: {}", e);
            }
        });
        PipeHandle { id, task }
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

    /// Consumer-owned predicate filtering. Spawns a forwarding task that
    /// passes only events whose `TargetOutput` satisfies `predicate`.
    /// Gap events are always forwarded. Consumes the subscription.
    pub fn filter_fn(self, predicate: fn(&TargetOutput) -> bool) -> Subscription {
        let id = self.id;
        let mut rx = self.rx;
        let (tx, new_rx) = mpsc::channel(256);
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let should_forward = match &event {
                    SinkEvent::Data(output) => predicate(output),
                    SinkEvent::Gap { .. } => true,
                    SinkEvent::Discontinuity { .. } => true,
                };
                if should_forward && tx.send(event).await.is_err() {
                    break;
                }
            }
        });
        Subscription { id, rx: new_rx }
    }
}

/// Lifecycle handle for a piped subscription.
///
/// Combines the subscription id (for bus-level `unsubscribe()`) with the
/// spawned task handle (for flush/join semantics) in a single unit.
pub struct PipeHandle {
    id: SinkId,
    task: JoinHandle<()>,
}

impl PipeHandle {
    /// The subscription id for bus control (e.g. `bus.unsubscribe(handle.id())`).
    pub fn id(&self) -> SinkId {
        self.id
    }

    /// Await the piped task to completion. The task finishes when the bus
    /// drops the subscription's sender (via `unsubscribe()` or `shutdown()`),
    /// draining remaining buffered events and flushing the sink.
    pub async fn join(self) -> Result<()> {
        self.task.await.map_err(Error::JoinError)
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

/// Label format for JoinedStream and history rendering.
#[derive(Clone, Copy)]
pub enum LabelFormat {
    /// "[web-1:build:0.1] content"
    Bracketed,
    /// "web-1:build:0.1> content"
    Prompt,
    /// Custom formatting function.
    Custom(fn(&SourceLabel, &str) -> String),
}

impl std::fmt::Debug for LabelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LabelFormat::Bracketed => write!(f, "Bracketed"),
            LabelFormat::Prompt => write!(f, "Prompt"),
            LabelFormat::Custom(_) => write!(f, "Custom(fn)"),
        }
    }
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
                Some(SinkEvent::Discontinuity { .. }) => {
                    // Reset source tracking so next output is source_changed = true
                    self.last_source = None;
                    continue;
                }
                None => return None,
            }
        }
    }

    /// Format a chunk with the configured label format.
    ///
    /// This is a consumer-side convenience for interactive multi-pane views.
    /// Terminal sinks (`StdioSink`) own their own presentation formatting
    /// at a different layer — they consume `TargetOutput` directly, not
    /// `StreamChunk`. The two formatting paths serve different consumers
    /// and should not be conflated.
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
// 2b.1 — Transcript/History for external LLM context (DC28)
// ---------------------------------------------------------------------------

/// Rendering mode for history output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Entries in arrival order with source labels on source transitions.
    #[default]
    Interleaved,
    /// Group entries by source, render each source as a labeled section.
    PerSource,
}

/// Options for configuring a rolling transcript/history handle.
#[derive(Debug, Clone)]
pub struct HistoryOptions {
    /// Maximum number of logical entries to retain.
    pub max_entries: usize,
    /// Maximum rendered characters. Oldest entries are trimmed first to stay
    /// within budget. `0` means no character limit.
    pub max_render_chars: usize,
    /// Label format used for the underlying `JoinedStream` source attribution.
    pub label_format: LabelFormat,
    /// When true, `render_text()` prepends an omission marker if entries were
    /// trimmed (e.g. `[... 37 earlier entries omitted ...]`).
    pub include_omission_marker: bool,
    /// Rendering mode: `Interleaved` (default) or `PerSource` for grouped sections.
    pub render_mode: RenderMode,
    /// Global character cap across all sources (Phase 3, DC33). When > 0 and
    /// `render_mode == PerSource`, after per-source trimming the largest
    /// source windows are trimmed until total rendered chars is within budget.
    /// `0` means no global cap. Ignored when `render_mode == Interleaved`.
    pub global_max_render_chars: usize,
}

impl Default for HistoryOptions {
    fn default() -> Self {
        HistoryOptions {
            max_entries: 500,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
            render_mode: RenderMode::Interleaved,
            global_max_render_chars: 0,
        }
    }
}

/// A single entry in the transcript history.
#[derive(Debug, Clone)]
pub enum HistoryEntry {
    /// Source-labeled output burst.
    Output {
        /// Source label from `JoinedStream` coalescing.
        source: SourceLabel,
        /// Content text.
        text: String,
        /// True when the source changed from the previous entry.
        source_changed: bool,
    },
    /// Explicit gap marker from bus backpressure.
    Gap {
        /// Number of events that were dropped.
        dropped_events: usize,
    },
    /// Upstream monitor discontinuity (DC29).
    Discontinuity {
        /// Human-readable reason for the discontinuity.
        reason: String,
    },
}

impl HistoryEntry {
    /// Source key for per-source grouping (Phase 2, DC33).
    ///
    /// Returns the source's `short()` label for Output entries, or a synthetic
    /// key for Gap/Discontinuity entries so they can be handled uniformly.
    pub fn source_key(&self) -> String {
        match self {
            HistoryEntry::Output { source, .. } => source.short(),
            HistoryEntry::Gap { .. } => "__system__".to_string(),
            HistoryEntry::Discontinuity { .. } => "__system__".to_string(),
        }
    }

    /// Plain text content without source labels (for per-source rendering).
    fn text_content(&self) -> String {
        match self {
            HistoryEntry::Output { text, .. } => format!("{}\n", text),
            HistoryEntry::Gap { dropped_events } => {
                format!("[gap: {} event(s) dropped]\n", dropped_events)
            }
            HistoryEntry::Discontinuity { reason } => {
                format!("[{}]\n", reason)
            }
        }
    }

    /// Rendered character count for budget trimming.
    ///
    /// Measures the actual rendered string length to ensure accurate budget
    /// enforcement for all label formats including `LabelFormat::Custom`.
    fn rendered_chars(&self, label_format: &LabelFormat) -> usize {
        let source_changed = match self {
            HistoryEntry::Output { source_changed, .. } => *source_changed,
            HistoryEntry::Gap { .. } => true,
            HistoryEntry::Discontinuity { .. } => true,
        };
        self.render(label_format, source_changed).len()
    }

    /// Render this entry as text.
    fn render(&self, label_format: &LabelFormat, source_changed: bool) -> String {
        match self {
            HistoryEntry::Output { source, text, .. } => {
                if source_changed {
                    match label_format {
                        LabelFormat::Bracketed => format!("[{}] {}\n", source.short(), text),
                        LabelFormat::Prompt => format!("{}> {}\n", source.short(), text),
                        LabelFormat::Custom(f) => format!("{}\n", f(source, text)),
                    }
                } else {
                    format!("{}\n", text)
                }
            }
            HistoryEntry::Gap { dropped_events } => {
                format!("[gap: {} event(s) dropped]\n", dropped_events)
            }
            HistoryEntry::Discontinuity { reason } => {
                format!("[{}]\n", reason)
            }
        }
    }
}

/// Snapshot of the rolling transcript at a point in time.
#[derive(Debug, Clone)]
pub struct HistorySnapshot {
    /// Entries in chronological order (oldest first).
    pub entries: Vec<HistoryEntry>,
    /// Total rendered characters in this snapshot.
    pub rendered_chars: usize,
    /// Number of older entries that were trimmed.
    pub omitted_entries: usize,
}

/// Internal entry for `PollHistory` with optional source tagging (Phase 2, DC33).
#[derive(Debug, Clone)]
struct PollHistoryEntry {
    source: Option<String>,
    text: String,
    char_count: usize,
}

/// Bounded rolling text history for poll-based consumers.
///
/// This mirrors the trimming and omission-marker behavior of `HistoryHandle`
/// without requiring an `OutputBus` subscription. It is intended for examples
/// and higher-level polling workflows that already have rendered text.
#[derive(Debug, Clone)]
pub struct PollHistory {
    entries: VecDeque<PollHistoryEntry>,
    max_entries: usize,
    max_render_chars: usize,
    rendered_chars: usize,
    omitted_entries: usize,
    include_omission_marker: bool,
    render_mode: RenderMode,
}

impl PollHistory {
    pub fn new(max_entries: usize, max_render_chars: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_render_chars,
            rendered_chars: 0,
            omitted_entries: 0,
            include_omission_marker: true,
            render_mode: RenderMode::Interleaved,
        }
    }

    pub fn with_omission_marker(mut self, include: bool) -> Self {
        self.include_omission_marker = include;
        self
    }

    /// Set the render mode (Phase 2, DC33).
    pub fn with_render_mode(mut self, mode: RenderMode) -> Self {
        self.render_mode = mode;
        self
    }

    pub fn push_text(&mut self, entry: String) {
        let chars = entry.len();
        self.entries.push_back(PollHistoryEntry {
            source: None,
            text: entry,
            char_count: chars,
        });
        self.rendered_chars += chars;
        self.trim();
    }

    /// Push a text entry tagged with a source key (Phase 2, DC33).
    ///
    /// When `render_mode == PerSource`, entries with the same source are
    /// grouped together in `render_text()`.
    pub fn push_text_for_source(&mut self, source: &str, text: String) {
        let chars = text.len();
        self.entries.push_back(PollHistoryEntry {
            source: Some(source.to_string()),
            text,
            char_count: chars,
        });
        self.rendered_chars += chars;
        self.trim();
    }

    pub fn render_text(&self) -> String {
        match self.render_mode {
            RenderMode::Interleaved => self.render_interleaved(),
            RenderMode::PerSource => self.render_per_source(),
        }
    }

    fn render_interleaved(&self) -> String {
        let mut result = String::with_capacity(self.rendered_chars + 64);
        if self.include_omission_marker && self.omitted_entries > 0 {
            result.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            ));
        }
        for entry in &self.entries {
            result.push_str(&entry.text);
        }
        result
    }

    fn render_per_source(&self) -> String {
        let mut sections: Vec<(String, Vec<&PollHistoryEntry>)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();

        for entry in &self.entries {
            let key = entry
                .source
                .clone()
                .unwrap_or_else(|| "__unsourced__".to_string());
            if let Some(&idx) = index.get(&key) {
                sections[idx].1.push(entry);
            } else {
                index.insert(key.clone(), sections.len());
                sections.push((key, vec![entry]));
            }
        }

        let mut result = String::new();
        if self.include_omission_marker && self.omitted_entries > 0 {
            result.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            ));
        }
        for (source, entries) in &sections {
            result.push_str(&format!("=== {} ===\n", source));
            for entry in entries {
                result.push_str(&entry.text);
            }
            result.push('\n');
        }
        result
    }

    pub fn rendered_chars(&self) -> usize {
        self.rendered_chars
    }

    pub fn omitted_entries(&self) -> usize {
        self.omitted_entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn trim(&mut self) {
        while self.entries.len() > self.max_entries {
            if let Some(removed) = self.entries.pop_front() {
                self.rendered_chars = self.rendered_chars.saturating_sub(removed.char_count);
                self.omitted_entries += 1;
            }
        }
        if self.max_render_chars > 0 {
            while self.rendered_chars > self.max_render_chars && !self.entries.is_empty() {
                if let Some(removed) = self.entries.pop_front() {
                    self.rendered_chars = self.rendered_chars.saturating_sub(removed.char_count);
                    self.omitted_entries += 1;
                }
            }
        }
    }
}

/// Per-source entry window for Phase 3 (DC33).
///
/// Holds entries for a single source with independent budget trimming.
struct SourceWindow {
    entries: VecDeque<HistoryEntry>,
    rendered_chars: usize,
    omitted_entries: usize,
}

impl SourceWindow {
    fn new() -> Self {
        SourceWindow {
            entries: VecDeque::new(),
            rendered_chars: 0,
            omitted_entries: 0,
        }
    }

    fn push(&mut self, entry: HistoryEntry, label_format: &LabelFormat) {
        let entry_chars = entry.rendered_chars(label_format);
        self.entries.push_back(entry);
        self.rendered_chars += entry_chars;
    }

    fn append_to_last(&mut self, text: &str, label_format: &LabelFormat) -> bool {
        if !matches!(self.entries.back(), Some(HistoryEntry::Output { .. })) {
            return false;
        }
        // Entry is guaranteed to exist and be Output after the guard above.
        let old_chars = self
            .entries
            .back()
            .map(|e| e.rendered_chars(label_format))
            .unwrap_or(0);
        if let Some(HistoryEntry::Output {
            text: ref mut existing,
            ..
        }) = self.entries.back_mut()
        {
            existing.push_str(text);
        }
        let new_chars = self
            .entries
            .back()
            .map(|e| e.rendered_chars(label_format))
            .unwrap_or(0);
        self.rendered_chars = self.rendered_chars.saturating_sub(old_chars) + new_chars;
        true
    }

    fn trim(&mut self, max_entries: usize, max_render_chars: usize, label_format: &LabelFormat) {
        while self.entries.len() > max_entries {
            if let Some(removed) = self.entries.pop_front() {
                let chars = removed.rendered_chars(label_format);
                self.rendered_chars = self.rendered_chars.saturating_sub(chars);
                self.omitted_entries += 1;
            }
        }
        if max_render_chars > 0 {
            while self.rendered_chars > max_render_chars && !self.entries.is_empty() {
                if let Some(removed) = self.entries.pop_front() {
                    let chars = removed.rendered_chars(label_format);
                    self.rendered_chars = self.rendered_chars.saturating_sub(chars);
                    self.omitted_entries += 1;
                }
            }
        }
    }
}

/// Shared state for the history accumulator.
struct HistoryState {
    // --- Interleaved mode storage (existing behavior) ---
    entries: VecDeque<HistoryEntry>,
    rendered_chars: usize,
    omitted_entries: usize,

    // --- Per-source mode storage (Phase 3, DC33) ---
    per_source: HashMap<String, SourceWindow>,
    source_order: Vec<String>,
    /// Tracks the last source key that was written to, for correct append targeting.
    last_written_source: Option<String>,

    // --- Configuration ---
    max_entries: usize,
    max_render_chars: usize,
    global_max_render_chars: usize,
    label_format: LabelFormat,
    include_omission_marker: bool,
    render_mode: RenderMode,
}

impl HistoryState {
    fn push(&mut self, entry: HistoryEntry) {
        match self.render_mode {
            RenderMode::Interleaved => self.push_interleaved(entry),
            RenderMode::PerSource => self.push_per_source(entry),
        }
    }

    fn push_interleaved(&mut self, entry: HistoryEntry) {
        let entry_chars = entry.rendered_chars(&self.label_format);
        self.entries.push_back(entry);
        self.rendered_chars += entry_chars;
        self.trim_interleaved();
    }

    fn push_per_source(&mut self, entry: HistoryEntry) {
        let key = entry.source_key();

        // Gap and Discontinuity go to all existing source windows
        match &entry {
            HistoryEntry::Gap { .. } | HistoryEntry::Discontinuity { .. } => {
                // Add to a __system__ window for rendering
                if !self.source_order.contains(&key) {
                    self.source_order.push(key.clone());
                }
                let window = self.per_source.entry(key).or_insert_with(SourceWindow::new);
                window.push(entry, &self.label_format);
                window.trim(self.max_entries, self.max_render_chars, &self.label_format);
            }
            HistoryEntry::Output { .. } => {
                if !self.source_order.contains(&key) {
                    self.source_order.push(key.clone());
                }
                self.last_written_source = Some(key.clone());
                let label_format = self.label_format;
                let max_entries = self.max_entries;
                let max_render_chars = self.max_render_chars;
                let window = self.per_source.entry(key).or_insert_with(SourceWindow::new);
                window.push(entry, &label_format);
                window.trim(max_entries, max_render_chars, &label_format);
            }
        }

        self.trim_global();
    }

    /// Phase 1 (DC33): Append text to the last entry if it's the same source.
    /// Returns `true` if the append succeeded, `false` if the caller should
    /// fall back to pushing a new entry (e.g. the last entry is a Gap).
    fn append_to_last(&mut self, text: &str) -> bool {
        match self.render_mode {
            RenderMode::Interleaved => self.append_to_last_interleaved(text),
            RenderMode::PerSource => self.append_to_last_per_source(text),
        }
    }

    fn append_to_last_interleaved(&mut self, text: &str) -> bool {
        if !matches!(self.entries.back(), Some(HistoryEntry::Output { .. })) {
            return false;
        }
        // Entry is guaranteed to exist and be Output after the guard above.
        let old_chars = self
            .entries
            .back()
            .map(|e| e.rendered_chars(&self.label_format))
            .unwrap_or(0);
        if let Some(HistoryEntry::Output {
            text: ref mut existing,
            ..
        }) = self.entries.back_mut()
        {
            existing.push_str(text);
        }
        let new_chars = self
            .entries
            .back()
            .map(|e| e.rendered_chars(&self.label_format))
            .unwrap_or(0);
        self.rendered_chars = self.rendered_chars.saturating_sub(old_chars) + new_chars;
        true
    }

    fn append_to_last_per_source(&mut self, text: &str) -> bool {
        // Find the last source that was actually written to (not first-seen order)
        if let Some(last_key) = self.last_written_source.clone() {
            if let Some(window) = self.per_source.get_mut(&last_key) {
                if !window.append_to_last(text, &self.label_format) {
                    return false;
                }
                let max_entries = self.max_entries;
                let max_render_chars = self.max_render_chars;
                let label_format = self.label_format;
                window.trim(max_entries, max_render_chars, &label_format);
                self.trim_global();
                return true;
            }
        }
        false
    }

    fn trim_interleaved(&mut self) {
        while self.entries.len() > self.max_entries {
            if let Some(removed) = self.entries.pop_front() {
                let chars = removed.rendered_chars(&self.label_format);
                self.rendered_chars = self.rendered_chars.saturating_sub(chars);
                self.omitted_entries += 1;
            }
        }
        if self.max_render_chars > 0 {
            while self.rendered_chars > self.max_render_chars && !self.entries.is_empty() {
                if let Some(removed) = self.entries.pop_front() {
                    let chars = removed.rendered_chars(&self.label_format);
                    self.rendered_chars = self.rendered_chars.saturating_sub(chars);
                    self.omitted_entries += 1;
                }
            }
        }
    }

    /// Phase 3 (DC33): Trim across all source windows until total rendered
    /// chars is within the global cap. Trims from the source with the most
    /// rendered chars first.
    fn trim_global(&mut self) {
        if self.global_max_render_chars == 0 {
            return;
        }
        let total: usize = self.per_source.values().map(|w| w.rendered_chars).sum();
        if total <= self.global_max_render_chars {
            return;
        }

        let mut to_trim = total - self.global_max_render_chars;
        while to_trim > 0 {
            // Find the source window with the most rendered chars
            let largest_key = self
                .per_source
                .iter()
                .filter(|(_, w)| !w.entries.is_empty())
                .max_by_key(|(_, w)| w.rendered_chars)
                .map(|(k, _)| k.clone());

            match largest_key {
                Some(key) => {
                    if let Some(window) = self.per_source.get_mut(&key) {
                        if let Some(removed) = window.entries.pop_front() {
                            let chars = removed.rendered_chars(&self.label_format);
                            window.rendered_chars = window.rendered_chars.saturating_sub(chars);
                            window.omitted_entries += 1;
                            to_trim = to_trim.saturating_sub(chars);
                        } else {
                            break;
                        }
                    }
                }
                None => break,
            }
        }
    }

    fn snapshot(&self) -> HistorySnapshot {
        match self.render_mode {
            RenderMode::Interleaved => HistorySnapshot {
                entries: self.entries.iter().cloned().collect(),
                rendered_chars: self.rendered_chars,
                omitted_entries: self.omitted_entries,
            },
            RenderMode::PerSource => {
                // Flatten per-source windows in source_order
                let mut entries = Vec::new();
                let mut total_rendered = 0usize;
                let mut total_omitted = 0usize;
                for key in &self.source_order {
                    if let Some(window) = self.per_source.get(key) {
                        entries.extend(window.entries.iter().cloned());
                        total_rendered += window.rendered_chars;
                        total_omitted += window.omitted_entries;
                    }
                }
                HistorySnapshot {
                    entries,
                    rendered_chars: total_rendered,
                    omitted_entries: total_omitted,
                }
            }
        }
    }

    fn render_text(&self) -> String {
        match self.render_mode {
            RenderMode::Interleaved => self.render_interleaved(),
            RenderMode::PerSource => self.render_per_source(),
        }
    }

    fn render_interleaved(&self) -> String {
        let mut result = String::with_capacity(self.rendered_chars + 64);

        if self.include_omission_marker && self.omitted_entries > 0 {
            result.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            ));
        }

        for entry in &self.entries {
            let source_changed = match entry {
                HistoryEntry::Output { source_changed, .. } => *source_changed,
                HistoryEntry::Gap { .. } => true,
                HistoryEntry::Discontinuity { .. } => true,
            };
            result.push_str(&entry.render(&self.label_format, source_changed));
        }

        result
    }

    fn render_per_source(&self) -> String {
        let mut result = String::new();

        // Collect total omissions for the marker
        let total_omitted: usize = self.per_source.values().map(|w| w.omitted_entries).sum();
        if self.include_omission_marker && total_omitted > 0 {
            result.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                total_omitted
            ));
        }

        for key in &self.source_order {
            if let Some(window) = self.per_source.get(key) {
                if window.entries.is_empty() {
                    continue;
                }
                result.push_str(&format!("=== {} ===\n", key));
                for entry in &window.entries {
                    result.push_str(&entry.text_content());
                }
                result.push('\n');
            }
        }
        result
    }
}

/// Handle to a rolling transcript/history accumulator (DC28).
///
/// Created by `Subscription::history(opts)`. Spawns a background task that
/// consumes `JoinedStream` chunks and accumulates them into a bounded,
/// source-labeled transcript. Consumers poll via `snapshot()` for structured
/// access or `render_text()` for prompt-ready rolling context.
pub struct HistoryHandle {
    id: SinkId,
    state: Arc<Mutex<HistoryState>>,
    task: JoinHandle<()>,
}

impl HistoryHandle {
    /// The subscription id for bus control.
    pub fn id(&self) -> SinkId {
        self.id
    }

    /// Take a snapshot of the current transcript state.
    pub async fn snapshot(&self) -> HistorySnapshot {
        self.state.lock().await.snapshot()
    }

    /// Render the current transcript as prompt-ready text.
    pub async fn render_text(&self) -> String {
        self.state.lock().await.render_text()
    }

    /// Await the background accumulator task to completion and return the
    /// final snapshot. The task finishes when the subscription channel closes
    /// (via `unsubscribe()` or bus `shutdown()`), guaranteeing all buffered
    /// events have been drained into the snapshot.
    pub async fn join(self) -> Result<HistorySnapshot> {
        self.task.await.map_err(Error::JoinError)?;
        Ok(self.state.lock().await.snapshot())
    }
}

impl Subscription {
    /// Create a rolling transcript/history handle (DC28).
    ///
    /// Spawns a background task that consumes events, applies source coalescing
    /// (same logic as `JoinedStream`), and accumulates entries into a bounded
    /// in-memory transcript. Unlike `joined()`, Gap events are preserved as
    /// explicit history entries instead of being skipped.
    ///
    /// Consumes the subscription.
    pub fn history(self, opts: HistoryOptions) -> HistoryHandle {
        let id = self.id;
        let mut rx = self.rx;

        let state = Arc::new(Mutex::new(HistoryState {
            entries: VecDeque::new(),
            rendered_chars: 0,
            omitted_entries: 0,
            per_source: HashMap::new(),
            source_order: Vec::new(),
            last_written_source: None,
            max_entries: opts.max_entries,
            max_render_chars: opts.max_render_chars,
            global_max_render_chars: opts.global_max_render_chars,
            label_format: opts.label_format,
            include_omission_marker: opts.include_omission_marker,
            render_mode: opts.render_mode,
        }));

        let state_clone = state.clone();
        let task = tokio::spawn(async move {
            // Source tracking replicates JoinedStream logic so we can also
            // capture Gap events (which JoinedStream::next() skips).
            let mut last_source: Option<SourceLabel> = None;

            while let Some(event) = rx.recv().await {
                match event {
                    SinkEvent::Data(output) => {
                        let source = SourceLabel::from_output(&output);
                        let source_changed = match &last_source {
                            Some(prev) => !prev.same_source(&source),
                            None => true,
                        };
                        last_source = Some(source.clone());

                        // Phase 1 (DC33): coalesce consecutive same-source
                        // chunks by appending to the last entry. Falls back
                        // to pushing a new entry if the last entry is not
                        // Output (e.g. a Gap or Discontinuity intervened).
                        if !source_changed {
                            let appended = state_clone.lock().await.append_to_last(&output.content);
                            if !appended {
                                let entry = HistoryEntry::Output {
                                    source,
                                    text: output.content.clone(),
                                    source_changed: false,
                                };
                                state_clone.lock().await.push(entry);
                            }
                        } else {
                            let entry = HistoryEntry::Output {
                                source,
                                text: output.content.clone(),
                                source_changed,
                            };
                            state_clone.lock().await.push(entry);
                        }
                    }
                    SinkEvent::Gap { dropped, .. } => {
                        let entry = HistoryEntry::Gap {
                            dropped_events: dropped,
                        };
                        state_clone.lock().await.push(entry);
                    }
                    SinkEvent::Discontinuity { reason } => {
                        // Reset source tracking — next output is source_changed = true
                        last_source = None;
                        let entry = HistoryEntry::Discontinuity { reason };
                        state_clone.lock().await.push(entry);
                    }
                }
            }
        });

        HistoryHandle { id, state, task }
    }
}

// ---------------------------------------------------------------------------
// DC33 Phase 4/5 — FlushPolicy + SourceAccumulator
// ---------------------------------------------------------------------------

/// Flush policy for [`SourceAccumulator`] — controls when accumulated lines
/// are committed as a history entry.
#[derive(Debug, Clone)]
pub enum FlushPolicy {
    /// Flush after `min_lines` accumulate, or after `max_wait` with any content.
    /// Good for build output, log tailing, high-throughput streams.
    LineCount {
        min_lines: usize,
        max_wait: std::time::Duration,
    },

    /// Flush when content stops changing for `idle_duration`.
    /// Good for CI jobs, test runners, deploy scripts — output arrives in bursts.
    Idle {
        idle_duration: std::time::Duration,
        max_wait: std::time::Duration,
    },

    /// Flush when a prompt line appears after content, indicating the agent
    /// finished its turn. Falls back to `max_wait` if no prompt is detected.
    /// Best for interactive agent sessions (Claude Code, Codex, shells).
    PromptBoundary {
        max_wait: std::time::Duration,
        min_content_lines: usize,
    },
}

impl FlushPolicy {
    /// Convenience: `LineCount { min_lines: 3, max_wait: 10s }`.
    pub fn line_count(min_lines: usize, max_wait: std::time::Duration) -> Self {
        FlushPolicy::LineCount {
            min_lines,
            max_wait,
        }
    }

    /// Convenience: `Idle { idle_duration: 3s, max_wait: 15s }`.
    pub fn idle(idle_duration: std::time::Duration, max_wait: std::time::Duration) -> Self {
        FlushPolicy::Idle {
            idle_duration,
            max_wait,
        }
    }

    /// Convenience: `PromptBoundary { max_wait: 30s, min_content_lines: 1 }`.
    pub fn prompt_boundary(max_wait: std::time::Duration, min_content_lines: usize) -> Self {
        FlushPolicy::PromptBoundary {
            max_wait,
            min_content_lines,
        }
    }

    /// Evaluate whether a flush should happen given the current state.
    pub fn should_flush(
        &self,
        pending: &[String],
        time_since_flush: std::time::Duration,
        time_since_last_change: std::time::Duration,
        saw_prompt: bool,
    ) -> bool {
        if pending.is_empty() {
            return false;
        }
        match self {
            FlushPolicy::LineCount {
                min_lines,
                max_wait,
            } => pending.len() >= *min_lines || time_since_flush >= *max_wait,
            FlushPolicy::Idle {
                idle_duration,
                max_wait,
            } => time_since_last_change >= *idle_duration || time_since_flush >= *max_wait,
            FlushPolicy::PromptBoundary {
                max_wait,
                min_content_lines,
            } => {
                if saw_prompt && pending.len() >= *min_content_lines {
                    return true;
                }
                time_since_flush >= *max_wait
            }
        }
    }
}

/// Per-source buffered content collection with pluggable filter and flush policy.
///
/// Each source (tmux session/pane) gets its own accumulator. The caller polls
/// pane content via `Target::capture_all()` and feeds it to `ingest()`. The
/// accumulator diffs against the previous capture, applies the content filter,
/// and flushes accumulated lines according to the flush policy.
///
/// ```rust,ignore
/// use motlie_tmux::{AgentTuiFilter, FlushPolicy, SourceAccumulator};
///
/// let baseline = target.capture_all().await?;
/// let mut acc = SourceAccumulator::new(
///     "my-session",
///     baseline,
///     Box::new(AgentTuiFilter::codex()),
///     FlushPolicy::prompt_boundary(Duration::from_secs(30), 1),
/// );
///
/// // In polling loop:
/// let current = target.capture_all().await?;
/// if let Some(chunk) = acc.ingest(&current) {
///     history.push_text_for_source("my-session", chunk);
/// }
/// ```
pub struct SourceAccumulator {
    #[allow(dead_code)]
    name: String,
    previous: std::collections::HashMap<crate::types::PaneAddress, String>,
    pending_lines: Vec<String>,
    last_flush: std::time::Instant,
    last_change: std::time::Instant,
    saw_prompt_since_flush: bool,
    filter: Box<dyn crate::filter::ContentFilter>,
    policy: FlushPolicy,
}

impl SourceAccumulator {
    /// Create a new accumulator for a source.
    ///
    /// - `name`: label for this source (used for diagnostics)
    /// - `baseline`: initial pane content from `capture_all()` (used as diff base)
    /// - `filter`: content filter for this source
    /// - `policy`: flush policy controlling when to commit
    pub fn new(
        name: &str,
        baseline: std::collections::HashMap<crate::types::PaneAddress, String>,
        filter: Box<dyn crate::filter::ContentFilter>,
        policy: FlushPolicy,
    ) -> Self {
        let now = std::time::Instant::now();
        Self {
            name: name.to_string(),
            previous: baseline,
            pending_lines: Vec::new(),
            last_flush: now,
            last_change: now,
            saw_prompt_since_flush: false,
            filter,
            policy,
        }
    }

    /// Feed new pane content. Diffs against previous, filters, accumulates.
    /// Returns a flushed chunk if the policy triggers.
    pub fn ingest(
        &mut self,
        current: &std::collections::HashMap<crate::types::PaneAddress, String>,
    ) -> Option<String> {
        if current == &self.previous {
            return self.maybe_flush();
        }

        let mut pane_list: Vec<_> = current.iter().collect();
        pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

        for (addr, content) in pane_list {
            let prev_content = self.previous.get(addr).map(String::as_str).unwrap_or("");
            let new_lines = crate::filter::diff_new_lines(prev_content, content, &*self.filter);
            for line in &new_lines {
                if self.filter.is_prompt(line) {
                    self.saw_prompt_since_flush = true;
                }
            }
            if !new_lines.is_empty() {
                self.pending_lines.extend(new_lines);
                self.last_change = std::time::Instant::now();
            }
        }

        self.previous = current.clone();
        self.maybe_flush()
    }

    /// Force flush any remaining pending content.
    pub fn flush_remaining(&mut self) -> Option<String> {
        if self.pending_lines.is_empty() {
            return None;
        }
        let chunk = self.pending_lines.join("\n");
        self.pending_lines.clear();
        self.last_flush = std::time::Instant::now();
        self.saw_prompt_since_flush = false;
        if chunk.trim().is_empty() {
            None
        } else {
            Some(format!("{}\n", chunk))
        }
    }

    fn maybe_flush(&mut self) -> Option<String> {
        if self.pending_lines.is_empty() {
            return None;
        }

        let should = self.policy.should_flush(
            &self.pending_lines,
            self.last_flush.elapsed(),
            self.last_change.elapsed(),
            self.saw_prompt_since_flush,
        );

        if !should {
            return None;
        }

        if !self.filter.is_meaningful_batch(&self.pending_lines) {
            self.pending_lines.clear();
            self.last_flush = std::time::Instant::now();
            self.saw_prompt_since_flush = false;
            return None;
        }

        let chunk = self.pending_lines.join("\n");
        self.pending_lines.clear();
        self.last_flush = std::time::Instant::now();
        self.saw_prompt_since_flush = false;

        if chunk.trim().is_empty() {
            None
        } else {
            Some(format!("{}\n", chunk))
        }
    }
}

// ---------------------------------------------------------------------------
// Bus-owned timelines
// ---------------------------------------------------------------------------

struct TimelineState {
    entries: VecDeque<TimelineEntry>,
    filters: Vec<CompiledSinkFilter>,
    options: TimelineOptions,
    rendered_chars: usize,
    omitted_entries: usize,
    next_sequence: u64,
    discontinuity_epoch: u64,
    newest_received_at: Option<Instant>,
    generation: u64,
    detached: bool,
    last_accessed_at: Instant,
}

impl TimelineState {
    fn new(options: TimelineOptions, generation: u64) -> Result<Self> {
        let filters = compile_timeline_filters(&options.filters)?;
        Ok(TimelineState {
            entries: VecDeque::new(),
            filters,
            options,
            rendered_chars: 0,
            omitted_entries: 0,
            next_sequence: 1,
            discontinuity_epoch: 0,
            newest_received_at: None,
            generation,
            detached: false,
            last_accessed_at: Instant::now(),
        })
    }

    fn touch(&mut self) {
        self.last_accessed_at = Instant::now();
    }

    fn detach(&mut self) {
        self.detached = true;
    }

    fn matches(&self, output: &TargetOutput) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| f.matches(output))
    }

    fn matches_marker_scope(&self, scope: &TimelineMarkerScope) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| f.matches_marker_scope(scope))
    }

    fn set_filters(&mut self, filters: Vec<SinkFilter>) -> Result<()> {
        let compiled = compile_timeline_filters(&filters)?;
        self.options.filters = filters;
        self.filters = compiled;
        self.touch();
        Ok(())
    }

    fn add_filter(&mut self, filter: SinkFilter) -> Result<()> {
        let compiled = CompiledSinkFilter::compile(&filter)?;
        self.options.filters.push(filter);
        self.filters.push(compiled);
        self.touch();
        Ok(())
    }

    fn push_output(&mut self, output: &TargetOutput, ingested_at: Instant) {
        if !self.matches(output) {
            return;
        }

        let received_at = output.timestamp;
        let ingested_at_wall = SystemTime::now();
        let received_at_wall = ingested_at
            .checked_duration_since(received_at)
            .and_then(|age| ingested_at_wall.checked_sub(age))
            .unwrap_or(ingested_at_wall);
        let mut late = false;
        if let Some(newest) = self.newest_received_at {
            if received_at > newest {
                self.newest_received_at = Some(received_at);
            } else if let TimelineOrdering::TimestampMerge { reorder_window, .. } =
                self.options.ordering
            {
                late = newest
                    .checked_duration_since(received_at)
                    .is_some_and(|age| age > reorder_window);
            }
        } else {
            self.newest_received_at = Some(received_at);
        }

        let entry = TimelineEntry {
            sequence: self.next_sequence,
            discontinuity_epoch: self.discontinuity_epoch,
            kind: TimelineEntryKind::Output,
            source: Some(SourceLabel::from_output(output)),
            target: Some(output.source.clone()),
            host: Some(output.host.clone()),
            session: Some(output.session_name().to_string()),
            pane_id: output.pane_id().map(str::to_string),
            content: Some(output.content.clone()),
            output_sequence: Some(output.sequence),
            received_at: Some(received_at),
            received_at_wall: Some(received_at_wall),
            ingested_at,
            ingested_at_wall,
            late,
        };
        self.next_sequence += 1;
        self.insert_entry(entry);
    }

    fn push_gap(&mut self, dropped_events: usize, ingested_at: Instant) {
        let ingested_at_wall = SystemTime::now();
        let entry = TimelineEntry {
            sequence: self.next_sequence,
            discontinuity_epoch: self.discontinuity_epoch,
            kind: TimelineEntryKind::Gap { dropped_events },
            source: None,
            target: None,
            host: None,
            session: None,
            pane_id: None,
            content: None,
            output_sequence: None,
            received_at: None,
            received_at_wall: None,
            ingested_at,
            ingested_at_wall,
            late: false,
        };
        self.next_sequence += 1;
        self.insert_entry(entry);
    }

    fn push_discontinuity(&mut self, reason: &str, ingested_at: Instant) {
        let ingested_at_wall = SystemTime::now();
        let entry = TimelineEntry {
            sequence: self.next_sequence,
            discontinuity_epoch: self.discontinuity_epoch,
            kind: TimelineEntryKind::Discontinuity {
                reason: reason.to_string(),
            },
            source: None,
            target: None,
            host: None,
            session: None,
            pane_id: None,
            content: None,
            output_sequence: None,
            received_at: None,
            received_at_wall: None,
            ingested_at,
            ingested_at_wall,
            late: false,
        };
        self.next_sequence += 1;
        self.discontinuity_epoch += 1;
        self.insert_entry(entry);
    }

    fn ingest_historical(&mut self, entries: Vec<TimelineEntry>) -> TimelinePage {
        let cursor = TimelineCursor {
            next_sequence: self.next_sequence,
            seen_sequences: Vec::new(),
        };
        let mut accepted = Vec::new();
        for mut entry in entries {
            entry.sequence = self.next_sequence;
            entry.discontinuity_epoch = self.discontinuity_epoch;
            entry.received_at = None;
            self.next_sequence += 1;
            if matches!(entry.kind, TimelineEntryKind::Discontinuity { .. }) {
                self.discontinuity_epoch += 1;
            }
            self.append_entry(entry.clone());
            accepted.push(entry);
        }
        let cursor = self.advance_cursor(&cursor, &accepted);
        TimelinePage {
            entries: accepted,
            cursor,
            omitted_entries: self.omitted_entries,
        }
    }

    fn append_entry(&mut self, entry: TimelineEntry) {
        let chars = entry.rendered_chars(&self.options.label_format);
        self.entries.push_back(entry);
        self.rendered_chars += chars;
        self.trim();
        self.touch();
    }

    fn insert_entry(&mut self, entry: TimelineEntry) {
        let chars = entry.rendered_chars(&self.options.label_format);
        match self.options.ordering {
            TimelineOrdering::Arrival => self.entries.push_back(entry),
            TimelineOrdering::TimestampMerge { .. } => {
                if entry.late || !matches!(entry.kind, TimelineEntryKind::Output) {
                    self.entries.push_back(entry);
                } else if let Some(received_at) = entry.received_at {
                    let pos = self.timestamp_insert_position(received_at);
                    if pos == self.entries.len() {
                        self.entries.push_back(entry);
                    } else {
                        self.entries.insert(pos, entry);
                    }
                } else {
                    self.entries.push_back(entry);
                }
            }
        }
        self.rendered_chars += chars;
        self.trim();
        self.touch();
    }

    fn timestamp_insert_position(&self, received_at: Instant) -> usize {
        if let Some(back) = self.entries.back() {
            if back.received_at.is_none_or(|ts| ts <= received_at) {
                return self.entries.len();
            }
        }

        for (idx, existing) in self.entries.iter().enumerate().rev() {
            if existing.received_at.is_some_and(|ts| ts <= received_at) {
                return idx + 1;
            }
        }

        self.entries
            .iter()
            .position(|entry| entry.received_at.is_some())
            .unwrap_or(self.entries.len())
    }

    fn trim(&mut self) {
        while self.entries.len() > self.options.max_entries {
            if let Some(removed) = self.entries.pop_front() {
                self.rendered_chars = self
                    .rendered_chars
                    .saturating_sub(removed.rendered_chars(&self.options.label_format));
                self.omitted_entries += 1;
            }
        }
        if self.options.max_render_chars > 0 {
            while self.rendered_chars > self.options.max_render_chars && !self.entries.is_empty() {
                if let Some(removed) = self.entries.pop_front() {
                    self.rendered_chars = self
                        .rendered_chars
                        .saturating_sub(removed.rendered_chars(&self.options.label_format));
                    self.omitted_entries += 1;
                }
            }
        }
    }

    fn pending_entries_after(&self, cursor: &TimelineCursor) -> Vec<TimelineEntry> {
        let seen_sequences: HashSet<u64> = cursor.seen_sequences.iter().copied().collect();
        self.entries
            .iter()
            .filter(|entry| {
                entry.sequence >= cursor.next_sequence && !seen_sequences.contains(&entry.sequence)
            })
            .cloned()
            .collect()
    }

    fn advance_cursor(
        &self,
        cursor: &TimelineCursor,
        returned: &[TimelineEntry],
    ) -> TimelineCursor {
        if returned.is_empty() {
            return cursor.clone();
        }

        let mut seen_set: HashSet<u64> = cursor.seen_sequences.iter().copied().collect();
        seen_set.extend(returned.iter().map(|entry| entry.sequence));

        let pending_next = self
            .entries
            .iter()
            .filter(|entry| {
                entry.sequence >= cursor.next_sequence && !seen_set.contains(&entry.sequence)
            })
            .map(|entry| entry.sequence)
            .min();

        let mut seen_sequences: Vec<u64> = seen_set.into_iter().collect();
        seen_sequences.sort_unstable();

        if let Some(next_sequence) = pending_next {
            let seen_sequences = seen_sequences
                .into_iter()
                .filter(|sequence| *sequence >= next_sequence)
                .collect();
            TimelineCursor {
                next_sequence,
                seen_sequences,
            }
        } else {
            TimelineCursor {
                next_sequence: self.next_sequence,
                seen_sequences: Vec::new(),
            }
        }
    }

    fn entries_after(&self, cursor: TimelineCursor, limit: usize) -> TimelinePage {
        let mut entries = self.pending_entries_after(&cursor);
        if limit > 0 && entries.len() > limit {
            entries.truncate(limit);
        }
        let next_cursor = self.advance_cursor(&cursor, &entries);
        TimelinePage {
            entries,
            cursor: next_cursor,
            omitted_entries: self.omitted_entries,
        }
    }

    fn latest(&self, limit: usize) -> TimelinePage {
        let len = self.entries.len();
        let start = if limit == 0 || limit >= len {
            0
        } else {
            len - limit
        };
        let entries: Vec<TimelineEntry> = self.entries.iter().skip(start).cloned().collect();
        TimelinePage {
            entries,
            cursor: TimelineCursor {
                next_sequence: self.next_sequence,
                seen_sequences: Vec::new(),
            },
            omitted_entries: self.omitted_entries,
        }
    }

    fn rendered_entries_after(
        &self,
        cursor: TimelineCursor,
        opts: RenderOptions,
    ) -> (Vec<TimelineEntry>, TimelineCursor) {
        let entries = self.pending_entries_after(&cursor);
        let cap = self.render_cap(opts);
        if cap == 0 {
            let next_cursor = self.advance_cursor(&cursor, &entries);
            return (entries, next_cursor);
        }

        let selected = match self.options.render_mode {
            RenderMode::Interleaved => self.select_interleaved_entries(entries, cap),
            RenderMode::PerSource => self.select_per_source_entries(entries, cap),
        };
        let next_cursor = self.advance_cursor(&cursor, &selected);
        (selected, next_cursor)
    }

    fn select_interleaved_entries(
        &self,
        entries: Vec<TimelineEntry>,
        cap: usize,
    ) -> Vec<TimelineEntry> {
        let mut selected = Vec::new();
        let mut rendered_chars = self.omission_marker_chars();
        for entry in entries {
            let entry_chars = entry.rendered_chars(&self.options.label_format);
            let candidate_chars = rendered_chars.saturating_add(entry_chars);
            if candidate_chars > cap && !selected.is_empty() {
                break;
            }
            selected.push(entry);
            rendered_chars = candidate_chars;
            if rendered_chars >= cap {
                break;
            }
        }
        selected
    }

    fn select_per_source_entries(
        &self,
        entries: Vec<TimelineEntry>,
        cap: usize,
    ) -> Vec<TimelineEntry> {
        let mut selected = Vec::new();
        let mut rendered_chars = self.omission_marker_chars();
        let mut seen_sources = HashSet::new();
        for entry in entries {
            let source = entry.render_source_key();
            let source_overhead = if seen_sources.contains(&source) {
                0
            } else {
                format!("=== {} ===\n", source).chars().count() + 1
            };
            let entry_chars = entry.rendered_chars(&self.options.label_format);
            let candidate_chars = rendered_chars
                .saturating_add(source_overhead)
                .saturating_add(entry_chars);
            if candidate_chars > cap && !selected.is_empty() {
                break;
            }
            seen_sources.insert(source);
            selected.push(entry);
            rendered_chars = candidate_chars;
            if rendered_chars >= cap {
                break;
            }
        }
        selected
    }

    fn omission_marker_chars(&self) -> usize {
        if self.options.include_omission_marker && self.omitted_entries > 0 {
            format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            )
            .chars()
            .count()
        } else {
            0
        }
    }

    fn render_entries(&self, entries: &[TimelineEntry], opts: RenderOptions) -> String {
        match self.options.render_mode {
            RenderMode::Interleaved => self.render_interleaved(entries, opts),
            RenderMode::PerSource => self.render_per_source(entries, opts),
        }
    }

    fn render_interleaved(&self, entries: &[TimelineEntry], opts: RenderOptions) -> String {
        let mut text = String::new();
        if self.options.include_omission_marker && self.omitted_entries > 0 {
            text.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            ));
        }
        for entry in entries {
            text.push_str(&entry.render(&self.options.label_format));
            self.enforce_render_cap(&mut text, opts);
        }
        text
    }

    fn render_per_source(&self, entries: &[TimelineEntry], opts: RenderOptions) -> String {
        let mut sections: Vec<(String, Vec<&TimelineEntry>)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            let key = entry.render_source_key();
            if let Some(idx) = index.get(&key).copied() {
                sections[idx].1.push(entry);
            } else {
                index.insert(key.clone(), sections.len());
                sections.push((key, vec![entry]));
            }
        }

        let mut text = String::new();
        if self.options.include_omission_marker && self.omitted_entries > 0 {
            text.push_str(&format!(
                "[... {} earlier entries omitted ...]\n",
                self.omitted_entries
            ));
        }
        for (source, entries) in sections {
            text.push_str(&format!("=== {} ===\n", source));
            for entry in entries {
                text.push_str(&entry.render(&self.options.label_format));
                self.enforce_render_cap(&mut text, opts);
            }
            text.push('\n');
        }
        text
    }

    fn render_cap(&self, opts: RenderOptions) -> usize {
        if opts.max_chars > 0 {
            opts.max_chars
        } else {
            self.options.max_render_chars
        }
    }

    fn enforce_render_cap(&self, text: &mut String, opts: RenderOptions) {
        let cap = self.render_cap(opts);
        if cap > 0 && char_count(text) > cap {
            truncate_to_chars(text, cap);
        }
    }
}

/// Handle to a named bus-owned timeline.
#[derive(Clone)]
pub struct TimelineHandle {
    name: String,
    generation: u64,
    state: TimelineStateHandle,
    registry: WeakTimelineRegistry,
}

impl TimelineHandle {
    /// Timeline name/key.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Timeline generation assigned by the owning bus.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, TimelineState>> {
        let state = self.state.lock().map_err(|_| timeline_lock_error())?;
        if state.detached || state.generation != self.generation {
            return Err(timeline_stale_error(&self.name, self.generation));
        }
        Ok(state)
    }

    /// Replace source-routing filters without dropping retained entries.
    pub async fn set_filters(&self, filters: Vec<SinkFilter>) -> Result<()> {
        let mut state = self.lock_state()?;
        state.set_filters(filters)
    }

    /// Add one source-routing filter without dropping retained entries.
    pub async fn add_filter(&self, filter: SinkFilter) -> Result<()> {
        let mut state = self.lock_state()?;
        state.add_filter(filter)
    }

    /// Ingest already reconstructed history into this timeline.
    ///
    /// Supplied entries are appended in caller order with fresh timeline
    /// sequence numbers. Process-local `received_at` instants are cleared during
    /// backfill; use wall-clock fields for persisted history metadata.
    pub async fn ingest_historical(&self, entries: Vec<TimelineEntry>) -> Result<TimelinePage> {
        let mut state = self.lock_state()?;
        Ok(state.ingest_historical(entries))
    }

    /// Detach this timeline from its owning bus and mark this handle stale.
    pub async fn detach(&self) -> Result<()> {
        if let Some(registry) = self.registry.upgrade() {
            let mut timelines = registry.lock().map_err(|_| timeline_lock_error())?;
            if timelines
                .get(&self.name)
                .is_some_and(|state| Arc::ptr_eq(state, &self.state))
            {
                timelines.remove(&self.name);
            }
        }
        let mut state = self.state.lock().map_err(|_| timeline_lock_error())?;
        state.detach();
        Ok(())
    }

    /// Fetch entries after a cursor. `limit == 0` returns all retained matches.
    pub async fn entries_after(
        &self,
        cursor: TimelineCursor,
        limit: usize,
    ) -> Result<TimelinePage> {
        let mut state = self.lock_state()?;
        state.touch();
        Ok(state.entries_after(cursor, limit))
    }

    /// Fetch the latest retained entries. `limit == 0` returns all retained entries.
    pub async fn latest(&self, limit: usize) -> Result<TimelinePage> {
        let mut state = self.lock_state()?;
        state.touch();
        Ok(state.latest(limit))
    }

    /// Render a prompt-ready window after a cursor.
    pub async fn render_after(
        &self,
        cursor: TimelineCursor,
        opts: RenderOptions,
    ) -> Result<TimelineRenderPage> {
        let mut state = self.lock_state()?;
        state.touch();
        let (entries, next_cursor) = state.rendered_entries_after(cursor, opts);
        let text = state.render_entries(&entries, opts);
        Ok(TimelineRenderPage {
            text,
            cursor: next_cursor,
            omitted_entries: state.omitted_entries,
        })
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
    /// Number of `Discontinuity` events lost due to backpressure. When > 0,
    /// the next successful delivery will synthesize a discontinuity-missed
    /// event so consumers know continuity was broken even if they missed the
    /// original signal.
    missed_discontinuities: usize,
}

/// Central fan-out dispatcher (DC12, DC24).
///
/// Uses interior mutability so it can be shared via `Arc<OutputBus>`.
/// All methods take `&self`.
pub struct OutputBus {
    subscribers: std::sync::Mutex<Vec<SubEntry>>,
    timelines: TimelineRegistry,
    next_id: AtomicU64,
    next_timeline_generation: AtomicU64,
}

impl OutputBus {
    pub fn new() -> Self {
        OutputBus {
            subscribers: std::sync::Mutex::new(Vec::new()),
            timelines: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            next_timeline_generation: AtomicU64::new(1),
        }
    }

    fn make_timeline_handle(
        &self,
        name: String,
        state: TimelineStateHandle,
    ) -> Result<TimelineHandle> {
        let generation = state.lock().map_err(|_| timeline_lock_error())?.generation;
        Ok(TimelineHandle {
            name,
            generation,
            state,
            registry: Arc::downgrade(&self.timelines),
        })
    }

    /// Create a named bus-owned timeline.
    pub fn create_timeline(
        &self,
        name: impl Into<String>,
        opts: TimelineOptions,
    ) -> Result<TimelineHandle> {
        let name = name.into();
        let generation = self
            .next_timeline_generation
            .fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(std::sync::Mutex::new(TimelineState::new(opts, generation)?));
        let mut timelines = self.timelines.lock().map_err(|_| timeline_lock_error())?;
        if timelines.contains_key(&name) {
            return Err(Error::AlreadyExists(format!(
                "timeline '{}' already exists",
                name
            )));
        }
        timelines.insert(name.clone(), state.clone());
        self.make_timeline_handle(name, state)
    }

    /// Open a named timeline — return the existing one or create it if missing.
    ///
    /// When the timeline already exists, `opts` are ignored and the existing
    /// generation is returned unchanged.
    pub fn open_timeline(
        &self,
        name: impl Into<String>,
        opts: TimelineOptions,
    ) -> Result<TimelineHandle> {
        let name = name.into();
        if let Some(handle) = self.timeline(&name)? {
            return Ok(handle);
        }

        let generation = self
            .next_timeline_generation
            .fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(std::sync::Mutex::new(TimelineState::new(opts, generation)?));
        let mut timelines = self.timelines.lock().map_err(|_| timeline_lock_error())?;
        if let Some(existing) = timelines.get(&name).cloned() {
            drop(timelines);
            return self.make_timeline_handle(name, existing);
        }
        timelines.insert(name.clone(), state.clone());
        self.make_timeline_handle(name, state)
    }

    /// Look up a named timeline.
    pub fn timeline(&self, name: &str) -> Result<Option<TimelineHandle>> {
        let state = self
            .timelines
            .lock()
            .map_err(|_| timeline_lock_error())?
            .get(name)
            .cloned();
        state
            .map(|state| self.make_timeline_handle(name.to_string(), state))
            .transpose()
    }

    /// Remove a named timeline and mark outstanding handles stale.
    pub fn remove_timeline(&self, name: &str) -> Result<()> {
        let state = self
            .timelines
            .lock()
            .map_err(|_| timeline_lock_error())?
            .remove(name)
            .ok_or_else(|| Error::NotFound(format!("timeline '{}' not found", name)))?;
        state.lock().map_err(|_| timeline_lock_error())?.detach();
        Ok(())
    }

    /// Remove timelines that have not been accessed for at least `idle_for`.
    pub fn remove_idle_timelines(&self, idle_for: Duration) -> Result<Vec<String>> {
        let now = Instant::now();
        let mut timelines = self.timelines.lock().map_err(|_| timeline_lock_error())?;
        let mut removed = Vec::new();
        let mut names = Vec::new();
        for (name, state) in timelines.iter() {
            let state = state.lock().map_err(|_| timeline_lock_error())?;
            if now.duration_since(state.last_accessed_at) >= idle_for {
                names.push(name.clone());
            }
        }
        for name in names {
            if let Some(state) = timelines.remove(&name) {
                state.lock().map_err(|_| timeline_lock_error())?.detach();
                removed.push(name);
            }
        }
        removed.sort();
        Ok(removed)
    }

    /// List timeline names.
    pub fn timelines(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self
            .timelines
            .lock()
            .map_err(|_| timeline_lock_error())?
            .keys()
            .cloned()
            .collect();
        names.sort();
        Ok(names)
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
            missed_discontinuities: 0,
        };

        self.subscribers
            .lock()
            .expect("bus lock poisoned")
            .push(entry);
        Ok(Subscription { id, rx })
    }

    /// Remove a subscription by id.
    ///
    /// Drops the sender, closing the channel. If a piped task was spawned via
    /// `Subscription::pipe()`, it will drain remaining buffered events and exit
    /// on its own — this method does **not** await the task. Callers that need
    /// join semantics should hold the `PipeHandle` returned by `pipe()`.
    pub fn unsubscribe(&self, id: SinkId) -> Result<()> {
        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        let len_before = subs.len();
        subs.retain(|s| s.id != id);
        if subs.len() == len_before {
            return Err(Error::NotFound(format!("subscription {:?} not found", id)));
        }
        Ok(())
    }

    fn timeline_states(&self) -> Vec<TimelineStateHandle> {
        self.timelines
            .lock()
            .map(|timelines| timelines.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Fan out to all matching subscribers. Non-blocking (try_send).
    pub fn publish(&self, output: TargetOutput) {
        let ingested_at = Instant::now();
        for state in self.timeline_states() {
            if let Ok(mut state) = state.lock() {
                state.push_output(&output, ingested_at);
            }
        }

        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        for sub in subs.iter_mut() {
            // Check filters: empty filters = match all; else OR across filters
            let matched = sub.filters.is_empty() || sub.filters.iter().any(|f| f.matches(&output));
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

            // If discontinuity events were missed, synthesize one so the
            // consumer always learns continuity was broken (DC29).
            if sub.missed_discontinuities > 0 {
                let missed = sub.missed_discontinuities;
                let event = SinkEvent::Discontinuity {
                    reason: format!(
                        "missed {} discontinuity event(s) due to backpressure",
                        missed
                    ),
                };
                match sub.tx.try_send(event) {
                    Ok(()) => {
                        sub.missed_discontinuities = 0;
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
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

    fn publish_timeline_gap(&self, scope: &TimelineMarkerScope, dropped_events: usize) {
        let ingested_at = Instant::now();
        for state in self.timeline_states() {
            if let Ok(mut state) = state.lock() {
                if state.matches_marker_scope(scope) {
                    state.push_gap(dropped_events, ingested_at);
                }
            }
        }
    }

    fn publish_timeline_discontinuity(&self, scope: &TimelineMarkerScope, reason: &str) {
        let ingested_at = Instant::now();
        for state in self.timeline_states() {
            if let Ok(mut state) = state.lock() {
                if state.matches_marker_scope(scope) {
                    state.push_discontinuity(reason, ingested_at);
                }
            }
        }
    }

    /// Record a global gap marker in unfiltered timelines.
    ///
    /// Subscriber-local backpressure still flows through `SinkEvent::Gap`; this
    /// method is for callers that need a bus-level retained gap marker.
    pub fn publish_gap(&self, dropped_events: usize) {
        self.publish_gap_for(TimelineMarkerScope::global(), dropped_events);
    }

    /// Record a scoped gap marker in timelines whose filters match the scope.
    pub fn publish_gap_for(&self, scope: TimelineMarkerScope, dropped_events: usize) {
        self.publish_timeline_gap(&scope, dropped_events);
    }

    /// Broadcast a global discontinuity event to all subscribers (DC29) and to
    /// unfiltered retained timelines.
    ///
    /// Unlike `publish()`, subscriber discontinuity events bypass source-routing
    /// filters because they are system-level signals, not content. Scoped
    /// timeline markers should use [`OutputBus::publish_discontinuity_for`].
    pub fn publish_discontinuity(&self, reason: &str) {
        self.publish_discontinuity_for(TimelineMarkerScope::global(), reason);
    }

    /// Broadcast a discontinuity event and retain it in timelines whose filters
    /// match the supplied scope.
    pub fn publish_discontinuity_for(&self, scope: TimelineMarkerScope, reason: &str) {
        self.publish_timeline_discontinuity(&scope, reason);

        let mut subs = self.subscribers.lock().expect("bus lock poisoned");
        for sub in subs.iter_mut() {
            let event = SinkEvent::Discontinuity {
                reason: reason.to_string(),
            };
            match sub.tx.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    sub.missed_discontinuities += 1;
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
    /// hold the `PipeHandle` returned by `pipe()` and call `join()` after
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
    use crate::types::{OutputFidelity, PaneAddress, SessionId, SessionInfo};
    use std::sync::Arc;

    fn make_output(
        host: &str,
        session: &str,
        pane_id: &str,
        content: &str,
        seq: u64,
    ) -> TargetOutput {
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
                id: SessionId::for_test("$0"),
                created: 0,
                attached_count: 0,
                window_count: 1,
                group: None,
                activity: 0,
            }),
            host: host.to_string(),
            content: content.to_string(),
            raw_content: None,
            sequence: 1,
            fidelity: OutputFidelity::clean(),
            timestamp: Instant::now(),
        }
    }

    fn make_timeline_entry(
        host: &str,
        session: &str,
        pane_id: &str,
        content: &str,
    ) -> TimelineEntry {
        let output = make_output(host, session, pane_id, content, 1);
        let wall = SystemTime::now();
        TimelineEntry {
            sequence: 999,
            discontinuity_epoch: 0,
            kind: TimelineEntryKind::Output,
            source: Some(SourceLabel::from_output(&output)),
            target: Some(output.source.clone()),
            host: Some(output.host.clone()),
            session: Some(output.session_name().to_string()),
            pane_id: output.pane_id().map(str::to_string),
            content: Some(output.content.clone()),
            output_sequence: Some(output.sequence),
            received_at: Some(output.timestamp),
            received_at_wall: Some(wall),
            ingested_at: Instant::now(),
            ingested_at_wall: wall,
            late: false,
        }
    }

    // --- TargetOutput accessor tests ---

    #[test]
    fn target_output_pane_accessors() {
        let out = make_output("web-1", "build", "%5", "hello", 1);
        assert_eq!(out.session_name(), "build");
        assert_eq!(out.pane_id(), Some("%5"));
        assert_eq!(out.source_key(), "%5");
        assert_eq!(out.target_string(), "build:0.0");
        assert!(!out.degraded());
    }

    #[test]
    fn target_output_source_key_vs_target_string() {
        let out = make_output("web-1", "build", "%5", "hello", 1);
        // source_key returns canonical identity (pane_id for panes)
        assert_eq!(out.source_key(), "%5");
        // target_string returns display format (session:window.pane)
        assert_eq!(out.target_string(), "build:0.0");
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

    #[tokio::test]
    async fn bus_timeline_filters_and_incremental_cursor() {
        let bus = OutputBus::new();
        let all = bus
            .create_timeline(
                "all",
                TimelineOptions {
                    max_entries: 10,
                    ..Default::default()
                },
            )
            .unwrap();
        let build = bus
            .create_timeline(
                "build",
                TimelineOptions {
                    filters: vec![SinkFilter::for_session("build")],
                    max_entries: 10,
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(
            bus.timelines().unwrap(),
            vec!["all".to_string(), "build".to_string()]
        );
        bus.publish(make_output("h", "build", "%1", "compile", 1));
        bus.publish(make_output("h", "review", "%2", "check", 1));

        let all_page = all
            .entries_after(TimelineCursor::default(), 0)
            .await
            .unwrap();
        assert_eq!(all_page.entries.len(), 2);
        assert_eq!(all_page.cursor.next_sequence, 3);

        let build_page = build
            .entries_after(TimelineCursor::default(), 0)
            .await
            .unwrap();
        assert_eq!(build_page.entries.len(), 1);
        assert_eq!(build_page.entries[0].session.as_deref(), Some("build"));
        assert_eq!(build_page.entries[0].content.as_deref(), Some("compile"));

        let none = build.entries_after(build_page.cursor, 0).await.unwrap();
        assert!(none.entries.is_empty());
    }

    #[tokio::test]
    async fn bus_timeline_timestamp_merge_and_late_marker() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "merged",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    label_format: LabelFormat::Prompt,
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);
        let mut late = make_output("h", "a", "%1", "late", 2);
        late.timestamp = base - Duration::from_millis(200);

        bus.publish(b);
        bus.publish(a);
        bus.publish(late);

        let page = timeline.latest(0).await.unwrap();
        let contents: Vec<&str> = page
            .entries
            .iter()
            .filter_map(|entry| entry.content.as_deref())
            .collect();
        assert_eq!(contents, vec!["first", "second", "late"]);
        assert!(page.entries[2].late);

        let rendered = timeline
            .render_after(TimelineCursor::default(), RenderOptions { max_chars: 0 })
            .await
            .unwrap();
        assert!(rendered.text.contains("[late]"));
    }

    #[tokio::test]
    async fn bus_timeline_timestamp_merge_cursor_uses_max_sequence() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "cursor",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);

        bus.publish(b);
        bus.publish(a);

        let page = timeline
            .entries_after(TimelineCursor::default(), 0)
            .await
            .unwrap();
        let sequences: Vec<u64> = page.entries.iter().map(|entry| entry.sequence).collect();
        assert_eq!(sequences, vec![2, 1]);
        assert_eq!(page.cursor.next_sequence, 3);

        let next = timeline.entries_after(page.cursor, 0).await.unwrap();
        assert!(next.entries.is_empty());
    }

    #[tokio::test]
    async fn bus_timeline_timestamp_merge_limit_cursor_does_not_skip_lower_sequence() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "bounded-cursor",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);

        bus.publish(b);
        bus.publish(a);

        let first = timeline
            .entries_after(TimelineCursor::default(), 1)
            .await
            .unwrap();
        assert_eq!(first.entries.len(), 1);
        assert_eq!(first.entries[0].sequence, 2);
        assert_eq!(first.cursor.next_sequence, 1);
        assert_eq!(first.cursor.seen_sequences, vec![2]);

        let second = timeline.entries_after(first.cursor, 1).await.unwrap();
        assert_eq!(second.entries.len(), 1);
        assert_eq!(second.entries[0].sequence, 1);
        assert_eq!(second.cursor.next_sequence, 3);
        assert!(second.cursor.seen_sequences.is_empty());

        let done = timeline.entries_after(second.cursor, 1).await.unwrap();
        assert!(done.entries.is_empty());
    }

    #[tokio::test]
    async fn bus_timeline_latest_cursor_uses_max_sequence() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "latest-cursor",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);

        bus.publish(b);
        bus.publish(a);

        let page = timeline.latest(0).await.unwrap();
        let sequences: Vec<u64> = page.entries.iter().map(|entry| entry.sequence).collect();
        assert_eq!(sequences, vec![2, 1]);
        assert_eq!(page.cursor.next_sequence, 3);
    }

    #[tokio::test]
    async fn bus_timeline_timestamp_merge_latest_limit_cursor_does_not_replay_snapshot() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "latest-bounded-cursor",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);

        bus.publish(b);
        bus.publish(a);

        let latest = timeline.latest(1).await.unwrap();
        let sequences: Vec<u64> = latest.entries.iter().map(|entry| entry.sequence).collect();
        assert_eq!(sequences, vec![1]);
        assert_eq!(latest.cursor.next_sequence, 3);

        let next = timeline
            .entries_after(latest.cursor.clone(), 0)
            .await
            .unwrap();
        assert!(next.entries.is_empty());

        let mut c = make_output("h", "c", "%3", "third", 1);
        c.timestamp = base + Duration::from_millis(30);
        bus.publish(c);

        let next = timeline.entries_after(latest.cursor, 0).await.unwrap();
        assert_eq!(next.entries.len(), 1);
        assert_eq!(next.entries[0].sequence, 3);
        assert_eq!(next.entries[0].content.as_deref(), Some("third"));
    }

    #[tokio::test]
    async fn bus_timeline_render_after_cursor_stops_at_rendered_entries() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "render-cursor",
                TimelineOptions {
                    label_format: LabelFormat::Prompt,
                    ..Default::default()
                },
            )
            .unwrap();

        bus.publish(make_output("h", "s", "%1", "one", 1));
        bus.publish(make_output("h", "s", "%1", "two", 2));
        bus.publish(make_output("h", "s", "%1", "three", 3));

        let first_entry_chars =
            timeline.latest(0).await.unwrap().entries[0].rendered_chars(&LabelFormat::Prompt);
        let page = timeline
            .render_after(
                TimelineCursor::default(),
                RenderOptions {
                    max_chars: first_entry_chars,
                },
            )
            .await
            .unwrap();
        assert!(page.text.contains("one"));
        assert!(!page.text.contains("two"));
        assert_eq!(page.cursor.next_sequence, 2);

        let next = timeline
            .render_after(page.cursor, RenderOptions::default())
            .await
            .unwrap();
        assert!(next.text.contains("two"));
        assert!(next.text.contains("three"));
        assert_eq!(next.cursor.next_sequence, 4);
    }

    #[tokio::test]
    async fn bus_timeline_timestamp_merge_render_cursor_does_not_skip_lower_sequence() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "render-bounded-cursor",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    label_format: LabelFormat::Prompt,
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut b = make_output("h", "b", "%2", "second", 1);
        b.timestamp = base + Duration::from_millis(20);
        let mut a = make_output("h", "a", "%1", "first", 1);
        a.timestamp = base + Duration::from_millis(10);

        bus.publish(b);
        bus.publish(a);

        let ordered = timeline.latest(0).await.unwrap();
        assert_eq!(ordered.entries[0].sequence, 2);
        let first_entry_chars = ordered.entries[0].rendered_chars(&LabelFormat::Prompt);

        let first = timeline
            .render_after(
                TimelineCursor::default(),
                RenderOptions {
                    max_chars: first_entry_chars,
                },
            )
            .await
            .unwrap();
        assert!(first.text.contains("first"));
        assert!(!first.text.contains("second"));
        assert_eq!(first.cursor.next_sequence, 1);
        assert_eq!(first.cursor.seen_sequences, vec![2]);

        let second = timeline
            .render_after(first.cursor, RenderOptions::default())
            .await
            .unwrap();
        assert!(second.text.contains("second"));
        assert_eq!(second.cursor.next_sequence, 3);
        assert!(second.cursor.seen_sequences.is_empty());
    }

    #[tokio::test]
    async fn bus_timeline_render_cap_handles_non_ascii_without_panic() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "unicode-render",
                TimelineOptions {
                    label_format: LabelFormat::Prompt,
                    ..Default::default()
                },
            )
            .unwrap();

        bus.publish(make_output("h", "s", "%1", "éé", 1));

        let full = timeline
            .render_after(TimelineCursor::default(), RenderOptions::default())
            .await
            .unwrap();
        let cap = full
            .text
            .chars()
            .position(|ch| ch == 'é')
            .map(|idx| idx + 1)
            .unwrap();

        let capped = timeline
            .render_after(TimelineCursor::default(), RenderOptions { max_chars: cap })
            .await
            .unwrap();
        assert_eq!(char_count(&capped.text), cap);
        assert!(capped.text.ends_with('é'));
        assert!(!capped.text.ends_with("éé"));
    }

    #[tokio::test]
    async fn bus_timeline_records_discontinuities_and_retention_omissions() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "short",
                TimelineOptions {
                    max_entries: 2,
                    ..Default::default()
                },
            )
            .unwrap();

        bus.publish(make_output("h", "s", "%1", "one", 1));
        bus.publish_discontinuity("stream resumed");
        bus.publish(make_output("h", "s", "%1", "two", 2));

        let page = timeline.latest(0).await.unwrap();
        assert_eq!(page.entries.len(), 2);
        assert_eq!(page.omitted_entries, 1);
        assert!(matches!(
            page.entries[0].kind,
            TimelineEntryKind::Discontinuity { .. }
        ));
        assert_eq!(page.entries[1].discontinuity_epoch, 1);
    }

    #[tokio::test]
    async fn bus_timeline_filters_are_mutable_without_dropping_buffer() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "mutable",
                TimelineOptions {
                    filters: vec![SinkFilter::for_session("build")],
                    ..Default::default()
                },
            )
            .unwrap();

        bus.publish(make_output("h", "build", "%1", "compile", 1));
        timeline
            .set_filters(vec![SinkFilter::for_session("test")])
            .await
            .unwrap();
        bus.publish(make_output("h", "test", "%2", "pass", 2));

        let page = timeline
            .entries_after(TimelineCursor::default(), 0)
            .await
            .unwrap();
        let contents: Vec<&str> = page
            .entries
            .iter()
            .filter_map(|entry| entry.content.as_deref())
            .collect();
        assert_eq!(contents, vec!["compile", "pass"]);

        timeline
            .add_filter(SinkFilter::for_session("deploy"))
            .await
            .unwrap();
        bus.publish(make_output("h", "deploy", "%3", "ship", 3));
        let page = timeline.entries_after(page.cursor, 0).await.unwrap();
        assert_eq!(page.entries[0].content.as_deref(), Some("ship"));
    }

    #[tokio::test]
    async fn bus_timeline_scoped_markers_respect_filters() {
        let bus = OutputBus::new();
        let build = bus
            .create_timeline(
                "build",
                TimelineOptions {
                    filters: vec![SinkFilter::for_host_session("h1", "build")],
                    ..Default::default()
                },
            )
            .unwrap();
        let test = bus
            .create_timeline(
                "test",
                TimelineOptions {
                    filters: vec![SinkFilter::for_host_session("h1", "test")],
                    ..Default::default()
                },
            )
            .unwrap();
        let all = bus
            .create_timeline("all", TimelineOptions::default())
            .unwrap();

        bus.publish_discontinuity_for(
            TimelineMarkerScope::for_host_session("h1", "build"),
            "build reconnect",
        );

        assert!(matches!(
            build.latest(0).await.unwrap().entries[0].kind,
            TimelineEntryKind::Discontinuity { .. }
        ));
        assert!(test.latest(0).await.unwrap().entries.is_empty());
        assert_eq!(all.latest(0).await.unwrap().entries.len(), 1);

        bus.publish_discontinuity("global reconnect");
        assert_eq!(build.latest(0).await.unwrap().entries.len(), 1);
        assert!(test.latest(0).await.unwrap().entries.is_empty());
        assert_eq!(all.latest(0).await.unwrap().entries.len(), 2);

        bus.publish_gap_for(TimelineMarkerScope::for_host_session("h1", "build"), 3);
        let build_page = build.latest(0).await.unwrap();
        assert_eq!(build_page.entries.len(), 2);
        assert!(matches!(
            build_page.entries[1].kind,
            TimelineEntryKind::Gap { dropped_events: 3 }
        ));
        assert!(test.latest(0).await.unwrap().entries.is_empty());
        assert_eq!(all.latest(0).await.unwrap().entries.len(), 3);

        bus.publish_gap(2);
        assert_eq!(build.latest(0).await.unwrap().entries.len(), 2);
        assert!(test.latest(0).await.unwrap().entries.is_empty());
        assert_eq!(all.latest(0).await.unwrap().entries.len(), 4);
    }

    #[tokio::test]
    async fn bus_timeline_create_or_get_detach_and_stale_handles() {
        let bus = OutputBus::new();
        let first = bus
            .open_timeline("workstream", TimelineOptions::default())
            .unwrap();
        let same = bus
            .open_timeline(
                "workstream",
                TimelineOptions {
                    max_entries: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(first.generation(), same.generation());

        first.detach().await.unwrap();
        assert!(bus.timeline("workstream").unwrap().is_none());
        assert!(first.latest(0).await.is_err());

        let replacement = bus
            .open_timeline("workstream", TimelineOptions::default())
            .unwrap();
        assert_ne!(first.generation(), replacement.generation());
        assert!(first.latest(0).await.is_err());

        bus.remove_timeline("workstream").unwrap();
        assert!(replacement.latest(0).await.is_err());
    }

    #[tokio::test]
    async fn bus_timeline_writes_refresh_idle_deadline() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "active",
                TimelineOptions {
                    filters: vec![SinkFilter::for_session("s")],
                    ..Default::default()
                },
            )
            .unwrap();

        {
            let mut state = timeline.state.lock().unwrap();
            state.last_accessed_at = Instant::now() - Duration::from_secs(60);
        }

        bus.publish(make_output("h", "s", "%1", "live", 1));

        let removed = bus.remove_idle_timelines(Duration::from_secs(30)).unwrap();
        assert!(removed.is_empty());
        assert_eq!(timeline.latest(0).await.unwrap().entries.len(), 1);
    }

    #[tokio::test]
    async fn bus_timeline_idle_cleanup_detaches_handles() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline("idle", TimelineOptions::default())
            .unwrap();

        let removed = bus.remove_idle_timelines(Duration::ZERO).unwrap();
        assert_eq!(removed, vec!["idle"]);
        assert!(timeline.latest(0).await.is_err());
    }

    #[tokio::test]
    async fn bus_timeline_historical_ingest_appends_in_caller_order() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline(
                "history-order",
                TimelineOptions {
                    ordering: TimelineOrdering::TimestampMerge {
                        reorder_window: Duration::from_millis(100),
                        late_event_policy: LateEventPolicy::AppendWithMarker,
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        let base = Instant::now();
        let mut first = make_timeline_entry("h", "s", "%1", "first");
        first.received_at = Some(base + Duration::from_millis(20));
        let mut second = make_timeline_entry("h", "s", "%2", "second");
        second.received_at = Some(base + Duration::from_millis(10));

        timeline
            .ingest_historical(vec![first, second])
            .await
            .unwrap();

        let page = timeline.latest(0).await.unwrap();
        let contents: Vec<&str> = page
            .entries
            .iter()
            .filter_map(|entry| entry.content.as_deref())
            .collect();
        assert_eq!(contents, vec!["first", "second"]);
        assert!(page.entries.iter().all(|entry| entry.received_at.is_none()));

        let mut live = make_output("h", "s", "%3", "live", 3);
        live.timestamp = base + Duration::from_millis(5);
        bus.publish(live);

        let page = timeline.latest(0).await.unwrap();
        let contents: Vec<&str> = page
            .entries
            .iter()
            .filter_map(|entry| entry.content.as_deref())
            .collect();
        assert_eq!(contents, vec!["first", "second", "live"]);
    }

    #[tokio::test]
    async fn bus_timeline_ingests_historical_entries_before_live_output() {
        let bus = OutputBus::new();
        let timeline = bus
            .create_timeline("history", TimelineOptions::default())
            .unwrap();

        let page = timeline
            .ingest_historical(vec![make_timeline_entry("h", "s", "%1", "backfill")])
            .await
            .unwrap();
        assert_eq!(page.entries[0].sequence, 1);
        assert_eq!(page.entries[0].content.as_deref(), Some("backfill"));
        assert!(page.entries[0].received_at_wall.is_some());

        bus.publish(make_output("h", "s", "%1", "live", 2));
        let page = timeline.latest(0).await.unwrap();
        let sequences: Vec<u64> = page.entries.iter().map(|entry| entry.sequence).collect();
        assert_eq!(sequences, vec![1, 2]);
        assert!(page.entries[1].received_at_wall.is_some());
        assert!(page.entries[1]
            .ingested_at_wall
            .duration_since(SystemTime::UNIX_EPOCH)
            .is_ok());
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
                id: SessionId::for_test("$0"),
                created: 0,
                attached_count: 0,
                window_count: 1,
                group: None,
                activity: 0,
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

    // --- SinkFilter constructor tests ---

    #[test]
    fn sink_filter_for_session_exact() {
        let filter = SinkFilter::for_session("build");
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("h", "build", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("h", "build2", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("h", "rebuild", "%5", "y", 1)));
    }

    #[test]
    fn sink_filter_for_host_exact() {
        let filter = SinkFilter::for_host("web-1");
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("web-1", "s", "%1", "y", 1)));
        assert!(!compiled.matches(&make_output("web-10", "s", "%1", "y", 1)));
    }

    #[test]
    fn sink_filter_for_pane_exact() {
        let filter = SinkFilter::for_pane("%5");
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("h", "s", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("h", "s", "%50", "y", 1)));
    }

    #[test]
    fn sink_filter_for_host_session_exact() {
        let filter = SinkFilter::for_host_session("web-1", "build");
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("web-1", "build", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("web-1", "deploy", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("web-2", "build", "%5", "y", 1)));
    }

    #[test]
    fn sink_filter_for_session_escapes_regex_chars() {
        // Session name with regex metacharacters should not be interpreted as regex
        let filter = SinkFilter::for_session("my.session+1");
        let compiled = CompiledSinkFilter::compile(&filter).unwrap();
        assert!(compiled.matches(&make_output("h", "my.session+1", "%5", "y", 1)));
        assert!(!compiled.matches(&make_output("h", "myXsessionX1", "%5", "y", 1)));
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

        let events: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
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
        handle.join().await.unwrap();

        let collected = events.lock().unwrap();
        assert_eq!(*collected, vec!["msg1", "msg2"]);
    }

    // --- Subscription::filter_fn tests ---

    #[tokio::test]
    async fn filter_fn_passes_matching_events() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();

        // Filter: only events from session "build"
        let filtered = sub.filter_fn(|out| out.session_name() == "build");
        let mut rx = filtered.into_receiver();

        bus.publish(make_output("h", "build", "%1", "yes", 1));
        bus.publish(make_output("h", "deploy", "%2", "no", 2));
        bus.publish(make_output("h", "build", "%3", "also-yes", 3));

        // Allow forwarding task to process
        tokio::task::yield_now().await;

        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "yes"),
            _ => panic!("expected Data"),
        }
        match rx.recv().await.unwrap() {
            SinkEvent::Data(out) => assert_eq!(out.content, "also-yes"),
            _ => panic!("expected Data"),
        }
    }

    #[tokio::test]
    async fn filter_fn_always_forwards_gaps() {
        let bus = OutputBus::new();
        // Small capacity to trigger backpressure
        let sub = bus.subscribe(vec![], 2).unwrap();

        let filtered = sub.filter_fn(|_| false); // block all data
        let mut rx = filtered.into_receiver();

        // Fill channel to create drops
        bus.publish(make_output("h", "s", "%1", "a", 1));
        bus.publish(make_output("h", "s", "%1", "b", 2));
        bus.publish(make_output("h", "s", "%1", "c", 3));
        bus.publish(make_output("h", "s", "%1", "d", 4));

        // Drain the original sub's channel — the filter_fn reads from
        // the subscription's channel (which is capacity 2), so let's
        // yield to let the forwarder process
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Publish more to trigger gap
        bus.publish(make_output("h", "s", "%1", "e", 5));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // We expect gaps to be forwarded even though predicate blocks Data
        // The gap should arrive since gaps are always forwarded
        bus.shutdown();
        // Drain all events and check that any Gap events made it through
        let mut found_gap = false;
        while let Some(event) = rx.recv().await {
            if matches!(event, SinkEvent::Gap { .. }) {
                found_gap = true;
            }
        }
        assert!(found_gap, "Gap events should be forwarded by filter_fn");
    }

    #[tokio::test]
    async fn filter_fn_composes_with_pipe() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();

        let filtered = sub.filter_fn(|out| out.content.contains("important"));

        let events: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
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

        let handle = filtered.pipe(sink);

        bus.publish(make_output("h", "s", "%1", "noise", 1));
        bus.publish(make_output("h", "s", "%1", "important data", 2));
        bus.publish(make_output("h", "s", "%1", "more noise", 3));
        bus.publish(make_output("h", "s", "%1", "also important!", 4));

        bus.shutdown();
        handle.join().await.unwrap();

        let collected = events.lock().unwrap();
        assert_eq!(*collected, vec!["important data", "also important!"]);
    }

    // --- HistoryHandle tests (DC28) ---

    #[tokio::test]
    async fn history_accumulates_entries() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();

        let history = sub.history(HistoryOptions {
            max_entries: 100,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
            ..Default::default()
        });

        // Two outputs from different sources — each gets its own entry
        bus.publish(make_output("h", "s", "%1", "hello", 1));
        bus.publish(make_output("h", "s", "%2", "world", 2));

        // Let accumulator process
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.omitted_entries, 0);
        match &snap.entries[0] {
            HistoryEntry::Output {
                text,
                source_changed,
                ..
            } => {
                assert_eq!(text, "hello");
                assert!(*source_changed); // first entry
            }
            _ => panic!("expected Output"),
        }
        match &snap.entries[1] {
            HistoryEntry::Output {
                text,
                source_changed,
                ..
            } => {
                assert_eq!(text, "world");
                assert!(*source_changed); // different source
            }
            _ => panic!("expected Output"),
        }

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_source_coalescing() {
        // Phase 1 (DC33): consecutive same-source chunks are coalesced.
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions::default());

        bus.publish(make_output("h", "s", "%1", "a", 1));
        bus.publish(make_output("h", "s", "%2", "b", 2)); // different pane
        bus.publish(make_output("h", "s", "%2", "c", 3)); // same pane — coalesced

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        // Third event coalesced into second — only 2 entries
        assert_eq!(snap.entries.len(), 2);

        // First: source_changed = true (initial)
        match &snap.entries[0] {
            HistoryEntry::Output { source_changed, .. } => assert!(*source_changed),
            _ => panic!("expected Output"),
        }
        // Second: source_changed = true (different pane), text is "bc" (coalesced)
        match &snap.entries[1] {
            HistoryEntry::Output {
                source_changed,
                text,
                ..
            } => {
                assert!(*source_changed);
                assert_eq!(text, "bc");
            }
            _ => panic!("expected Output"),
        }

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_trims_oldest_by_entry_count() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 3,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
            ..Default::default()
        });

        // Use different panes to prevent coalescing
        for i in 0..5 {
            bus.publish(make_output(
                "h",
                "s",
                &format!("%{}", i),
                &format!("msg{}", i),
                i as u64,
            ));
        }

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        assert_eq!(snap.entries.len(), 3);
        assert_eq!(snap.omitted_entries, 2);

        // Oldest remaining should be msg2
        match &snap.entries[0] {
            HistoryEntry::Output { text, .. } => assert_eq!(text, "msg2"),
            _ => panic!("expected Output"),
        }

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_trims_by_render_char_budget() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 1000,
            max_render_chars: 50,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: false,
            ..Default::default()
        });

        // Use different panes so coalescing doesn't merge them.
        // Each entry: "[h:s(%N)] <text>\n" — label is ~12 chars + text + newline
        for i in 0..10 {
            bus.publish(make_output(
                "h",
                "s",
                &format!("%{}", i),
                &format!("line-{:03}", i),
                i as u64,
            ));
        }

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        assert!(
            snap.rendered_chars <= 50,
            "rendered_chars {} > 50",
            snap.rendered_chars
        );
        assert!(snap.omitted_entries > 0, "should have trimmed some entries");

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_gap_propagation() {
        // Test that gap events from the bus are recorded as HistoryEntry::Gap.
        // We inject a gap directly via the channel rather than relying on
        // timing-sensitive backpressure.
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(999),
            rx,
        };
        let history = sub.history(HistoryOptions::default());

        tx.send(SinkEvent::Data(make_output("h", "s", "%1", "before", 1)))
            .await
            .unwrap();
        tx.send(SinkEvent::Gap {
            dropped: 5,
            timestamp: Instant::now(),
        })
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output("h", "s", "%1", "after", 2)))
            .await
            .unwrap();

        // Close sender to let task finish
        drop(tx);
        history.task.await.unwrap();

        // Re-acquire state for assertion (task is done, lock is uncontested)
        let snap = history.state.lock().await.snapshot();
        assert_eq!(snap.entries.len(), 3);
        match &snap.entries[1] {
            HistoryEntry::Gap { dropped_events } => assert_eq!(*dropped_events, 5),
            other => panic!("expected Gap, got {:?}", other),
        }

        let text = history.state.lock().await.render_text();
        assert!(text.contains("[gap: 5 event(s) dropped]"), "got: {}", text);
    }

    #[tokio::test]
    async fn history_render_text_with_omission() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 2,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
            ..Default::default()
        });

        // Use different sources to prevent coalescing
        bus.publish(make_output("h", "build", "%1", "old", 1));
        bus.publish(make_output("h", "build", "%2", "mid", 2));
        bus.publish(make_output("h", "build", "%3", "new", 3));

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let text = history.render_text().await;
        assert!(
            text.contains("[... 1 earlier entries omitted ...]"),
            "expected omission marker, got: {}",
            text
        );
        assert!(text.contains("mid"));
        assert!(text.contains("new"));
        assert!(!text.contains("old")); // trimmed

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_render_text_labels_on_source_change() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 100,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: false,
            ..Default::default()
        });

        bus.publish(make_output("web", "build", "%1", "compiling", 1));
        bus.publish(make_output("web", "build", "%1", "done", 2));
        bus.publish(make_output("web", "deploy", "%2", "deploying", 3));

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let text = history.render_text().await;
        // First entry gets label (source_changed = true)
        assert!(text.contains("[web:build(%1)] compiling"), "got: {}", text);
        // Same source = no label
        assert!(text.contains("done\n"), "got: {}", text);
        // Source change = new label
        assert!(text.contains("[web:deploy(%2)] deploying"), "got: {}", text);

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_rolling_under_sustained_output() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 64).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 5,
            max_render_chars: 0,
            label_format: LabelFormat::Bracketed,
            include_omission_marker: true,
            ..Default::default()
        });

        // Publish 20 events from alternating sources
        for i in 0..20u64 {
            let pane = if i % 2 == 0 { "%1" } else { "%2" };
            bus.publish(make_output("h", "s", pane, &format!("msg{}", i), i));
        }

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let snap = history.snapshot().await;
        assert_eq!(snap.entries.len(), 5);
        assert_eq!(snap.omitted_entries, 15);

        // Should contain only msg15-msg19
        for (idx, entry) in snap.entries.iter().enumerate() {
            match entry {
                HistoryEntry::Output { text, .. } => {
                    assert_eq!(text, &format!("msg{}", 15 + idx));
                }
                _ => panic!("unexpected entry type"),
            }
        }

        let text = history.render_text().await;
        assert!(text.starts_with("[... 15 earlier entries omitted ...]"));

        bus.shutdown();
        history.join().await.unwrap();
    }

    // --- SinkEvent::Discontinuity tests (4.2b, DC29) ---

    #[tokio::test]
    async fn discontinuity_forwarded_by_filter_fn() {
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(100),
            rx,
        };
        let filtered = sub.filter_fn(|_| false); // Block all data
        let mut rx = filtered.into_receiver();

        // Send a discontinuity event
        tx.send(SinkEvent::Discontinuity {
            reason: "test disconnect".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        match rx.recv().await.unwrap() {
            SinkEvent::Discontinuity { reason } => {
                assert_eq!(reason, "test disconnect");
            }
            other => panic!("expected Discontinuity, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn discontinuity_resets_joined_stream_source() {
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(101),
            rx,
        };
        let mut joined = sub.joined(LabelFormat::Prompt);

        // First output from pane A
        tx.send(SinkEvent::Data(make_output("h", "s", "%1", "first", 1)))
            .await
            .unwrap();
        let chunk = joined.next().await.unwrap();
        assert!(chunk.source_changed);

        // Second output from same pane A — source NOT changed
        tx.send(SinkEvent::Data(make_output("h", "s", "%1", "second", 2)))
            .await
            .unwrap();
        let chunk = joined.next().await.unwrap();
        assert!(!chunk.source_changed);

        // Discontinuity resets source tracking
        tx.send(SinkEvent::Discontinuity {
            reason: "connection lost".to_string(),
        })
        .await
        .unwrap();

        // Next output from same pane A — source IS changed (reset by discontinuity)
        tx.send(SinkEvent::Data(make_output("h", "s", "%1", "after", 3)))
            .await
            .unwrap();
        let chunk = joined.next().await.unwrap();
        assert!(
            chunk.source_changed,
            "source should be changed after discontinuity"
        );
    }

    #[tokio::test]
    async fn discontinuity_recorded_in_history() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 10,
            ..Default::default()
        });

        bus.publish(make_output("h", "s", "%1", "before", 1));
        bus.publish_discontinuity("stream interrupted: control channel lost");
        bus.publish(make_output("h", "s", "%1", "after", 2));

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        assert_eq!(snap.entries.len(), 3);

        match &snap.entries[0] {
            HistoryEntry::Output { text, .. } => assert_eq!(text, "before"),
            other => panic!("expected Output, got {:?}", other),
        }
        match &snap.entries[1] {
            HistoryEntry::Discontinuity { reason } => {
                assert_eq!(reason, "stream interrupted: control channel lost");
            }
            other => panic!("expected Discontinuity, got {:?}", other),
        }
        match &snap.entries[2] {
            HistoryEntry::Output {
                text,
                source_changed,
                ..
            } => {
                assert_eq!(text, "after");
                assert!(
                    *source_changed,
                    "source should be reset after discontinuity"
                );
            }
            other => panic!("expected Output, got {:?}", other),
        }

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn discontinuity_counts_against_history_budget() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 2,
            ..Default::default()
        });

        bus.publish(make_output("h", "s", "%1", "first", 1));
        bus.publish_discontinuity("interrupted");
        bus.publish(make_output("h", "s", "%1", "third", 2));

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        // max_entries=2 so oldest entry trimmed
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.omitted_entries, 1);

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn discontinuity_rendered_in_history_text() {
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 10,
            include_omission_marker: false,
            ..Default::default()
        });

        bus.publish(make_output("h", "s", "%1", "hello", 1));
        bus.publish_discontinuity("stream interrupted: ssh lost");
        bus.publish(make_output("h", "s", "%1", "resumed", 2));

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let text = history.render_text().await;
        assert!(
            text.contains("[stream interrupted: ssh lost]"),
            "rendered text should contain discontinuity marker, got: {}",
            text
        );
        assert!(text.contains("hello"));
        assert!(text.contains("resumed"));

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn publish_discontinuity_broadcasts_to_all_subscribers() {
        let bus = OutputBus::new();
        let sub1 = bus.subscribe(vec![], 16).unwrap();
        let sub2 = bus
            .subscribe(vec![SinkFilter::for_session("specific")], 16)
            .unwrap();

        let mut rx1 = sub1.into_receiver();
        let mut rx2 = sub2.into_receiver();

        // Discontinuity bypasses filters
        bus.publish_discontinuity("system event");

        match rx1.recv().await.unwrap() {
            SinkEvent::Discontinuity { reason } => assert_eq!(reason, "system event"),
            other => panic!("sub1: expected Discontinuity, got {:?}", other),
        }
        match rx2.recv().await.unwrap() {
            SinkEvent::Discontinuity { reason } => assert_eq!(reason, "system event"),
            other => panic!("sub2: expected Discontinuity, got {:?}", other),
        }
    }

    // --- 4.2e: Stress + failure injection tests ---

    #[tokio::test]
    async fn bus_stress_high_throughput_with_slow_subscriber() {
        // A slow subscriber (capacity=4) gets Gap events after we drain some
        // events to make room. A fast subscriber (capacity=1000) gets everything.
        let bus = OutputBus::new();
        let fast_sub = bus.subscribe(vec![], 1000).unwrap();
        let slow_sub = bus.subscribe(vec![], 4).unwrap();
        let mut slow_rx = slow_sub.into_receiver();

        // Phase 1: fill the slow channel (4 events fill capacity)
        for i in 0..10u64 {
            bus.publish(make_output("h", "s", "%0", &format!("msg-{}", i), i));
        }
        // Drain the slow channel to make room for Gap delivery
        let mut slow_data = 0u64;
        let mut slow_gaps = 0u64;
        let mut total_dropped = 0usize;
        while let Ok(event) = slow_rx.try_recv() {
            match event {
                SinkEvent::Data(_) => slow_data += 1,
                SinkEvent::Gap { dropped, .. } => {
                    slow_gaps += 1;
                    total_dropped += dropped;
                }
                _ => {}
            }
        }

        // Phase 2: publish more — this should trigger a Gap event for prior drops
        for i in 10..100u64 {
            bus.publish(make_output("h", "s", "%0", &format!("msg-{}", i), i));
        }
        // Drain again
        while let Ok(event) = slow_rx.try_recv() {
            match event {
                SinkEvent::Data(_) => slow_data += 1,
                SinkEvent::Gap { dropped, .. } => {
                    slow_gaps += 1;
                    total_dropped += dropped;
                }
                _ => {}
            }
        }

        let mut fast_rx = fast_sub.into_receiver();
        let mut fast_data = 0u64;
        while let Ok(event) = fast_rx.try_recv() {
            if matches!(event, SinkEvent::Data(_)) {
                fast_data += 1;
            }
        }
        assert_eq!(fast_data, 100, "fast subscriber should get all 100 events");
        assert!(
            slow_data < 100,
            "slow subscriber should have dropped some (got {})",
            slow_data
        );
        assert!(
            slow_gaps > 0,
            "slow subscriber should have at least one Gap"
        );
        assert!(total_dropped > 0, "Gap should report dropped events");
    }

    #[tokio::test]
    async fn history_determinism_under_bursty_multi_source_with_discontinuity() {
        // Verifies transcript determinism: multi-source output interleaved with
        // discontinuity events produces a history where discontinuity markers
        // appear at the correct positions and source labels are re-emitted.
        // Phase 1 (DC33) coalesces consecutive same-source chunks.
        let bus = Arc::new(OutputBus::new());
        let sub = bus.subscribe(vec![], 128).unwrap();
        let history = sub.history(HistoryOptions {
            max_entries: 100,
            max_render_chars: 10_000,
            ..Default::default()
        });

        // Phase 1: output from source A (alpha-1 + alpha-2 coalesced)
        bus.publish(make_output("h", "s", "%0", "alpha-1\n", 1));
        bus.publish(make_output("h", "s", "%0", "alpha-2\n", 2));

        // Phase 2: output from source B
        bus.publish(make_output("h", "s", "%1", "beta-1\n", 1));

        // Discontinuity
        bus.publish_discontinuity("reconnect");

        // Phase 3: output from source A again (after reconnect)
        bus.publish(make_output("h", "s", "%0", "alpha-3\n", 3));

        // Phase 4: output from source B again
        bus.publish(make_output("h", "s", "%1", "beta-2\n", 2));

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        let snap = history.snapshot().await;

        // Verify structure (alpha-1 and alpha-2 coalesced):
        // 0. Data ("alpha-1\nalpha-2\n", source_changed=true) — coalesced
        // 1. Data (beta-1, source_changed=true)
        // 2. Discontinuity
        // 3. Data (alpha-3, source_changed=true — source reset by discontinuity)
        // 4. Data (beta-2, source_changed=true)
        assert_eq!(
            snap.entries.len(),
            5,
            "expected 5 entries, got {}",
            snap.entries.len()
        );

        // Entry 0 should have coalesced alpha-1 and alpha-2
        match &snap.entries[0] {
            HistoryEntry::Output { text, .. } => {
                assert!(
                    text.contains("alpha-1") && text.contains("alpha-2"),
                    "expected coalesced alpha text, got: {}",
                    text
                );
            }
            other => panic!("entry[0] should be Output, got {:?}", other),
        }

        // Entry 2 should be Discontinuity
        assert!(
            matches!(&snap.entries[2], HistoryEntry::Discontinuity { reason } if reason == "reconnect"),
            "entry[2] should be Discontinuity(reconnect), got {:?}",
            snap.entries[2]
        );

        // Entry 3 should have source_changed=true (reset by discontinuity)
        match &snap.entries[3] {
            HistoryEntry::Output { source_changed, .. } => {
                assert!(
                    *source_changed,
                    "first output after discontinuity should have source_changed=true"
                );
            }
            other => panic!("entry[3] should be Output, got {:?}", other),
        }

        // Rendered text should contain the discontinuity marker
        let text = history.render_text().await;
        assert!(
            text.contains("[reconnect]"),
            "rendered text should contain [reconnect]"
        );

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn bus_missed_discontinuity_synthesized_on_next_publish() {
        // When a discontinuity can't be delivered due to backpressure, the
        // next successful publish() must synthesize a Discontinuity event
        // so the consumer always learns continuity was broken (DC29).
        let bus = OutputBus::new();
        // Capacity 3: fill with data, then discontinuity will fail
        let sub = bus.subscribe(vec![], 3).unwrap();
        let mut rx = sub.into_receiver();

        // Fill the channel
        for i in 0..3u64 {
            bus.publish(make_output("h", "s", "%0", &format!("fill-{}", i), i));
        }

        // Discontinuity will be missed (channel full)
        bus.publish_discontinuity("ssh drop");

        // Drain the channel to make room
        let mut drained = 0;
        while let Ok(_) = rx.try_recv() {
            drained += 1;
        }
        assert_eq!(drained, 3, "should have drained 3 data events");

        // Next publish should deliver: Gap (for the missed events) +
        // synthesized Discontinuity + Data
        bus.publish(make_output("h", "s", "%0", "after-reconnect", 10));

        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }

        // Should see Gap, then Discontinuity, then Data
        let has_gap = events.iter().any(|e| matches!(e, SinkEvent::Gap { .. }));
        let has_discontinuity = events
            .iter()
            .any(|e| matches!(e, SinkEvent::Discontinuity { .. }));
        let has_data = events.iter().any(|e| matches!(e, SinkEvent::Data(_)));

        assert!(has_gap, "should have a Gap event for dropped events");
        assert!(
            has_discontinuity,
            "should have a synthesized Discontinuity event for missed signal, events: {:?}",
            events
        );
        assert!(has_data, "should have the new Data event");

        // Verify the discontinuity mentions it was missed
        for ev in &events {
            if let SinkEvent::Discontinuity { reason } = ev {
                assert!(
                    reason.contains("missed"),
                    "reason should mention missed: {}",
                    reason
                );
            }
        }
    }

    #[test]
    fn poll_history_trims_by_entry_and_char_budget() {
        let mut history = PollHistory::new(2, 12);
        history.push_text("first\n".to_string());
        history.push_text("second\n".to_string());
        history.push_text("third\n".to_string());

        assert_eq!(history.len(), 1);
        assert_eq!(history.omitted_entries(), 2);
        assert_eq!(
            history.render_text(),
            "[... 2 earlier entries omitted ...]\nthird\n"
        );
    }

    // --- DC33: Per-source coherent history tests ---

    #[tokio::test]
    async fn history_coalesces_same_source_chunks() {
        // Phase 1 (DC33): 5 rapid same-source events produce fewer than 5 entries.
        let bus = OutputBus::new();
        let sub = bus.subscribe(vec![], 16).unwrap();
        let history = sub.history(HistoryOptions::default());

        for i in 0..5 {
            bus.publish(make_output(
                "h",
                "s",
                "%1",
                &format!("chunk{}", i),
                i as u64,
            ));
        }

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let snap = history.snapshot().await;
        assert!(
            snap.entries.len() < 5,
            "coalescing should reduce 5 same-source events to fewer entries, got {}",
            snap.entries.len()
        );
        // All text should be present in the coalesced entry
        match &snap.entries[0] {
            HistoryEntry::Output { text, .. } => {
                for i in 0..5 {
                    assert!(
                        text.contains(&format!("chunk{}", i)),
                        "missing chunk{} in coalesced text: {}",
                        i,
                        text
                    );
                }
            }
            other => panic!("expected Output, got {:?}", other),
        }

        bus.shutdown();
        history.join().await.unwrap();
    }

    #[tokio::test]
    async fn history_per_source_render_groups_by_source() {
        // Phase 2 (DC33): interleaved events from 2 sources produce grouped sections.
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(200),
            rx,
        };
        let history = sub.history(HistoryOptions {
            max_entries: 100,
            render_mode: RenderMode::PerSource,
            include_omission_marker: false,
            ..Default::default()
        });

        // Interleave source A and B
        tx.send(SinkEvent::Data(make_output(
            "h",
            "build",
            "%1",
            "compiling",
            1,
        )))
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output(
            "h",
            "test",
            "%2",
            "test_a PASS",
            1,
        )))
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output(
            "h", "build", "%1", "linking", 2,
        )))
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output(
            "h",
            "test",
            "%2",
            "test_b PASS",
            2,
        )))
        .await
        .unwrap();

        drop(tx);
        history.task.await.unwrap();

        let text = history.state.lock().await.render_text();
        // Each source should appear as a section
        assert!(
            text.contains("=== h:build(%1) ==="),
            "should have build section header, got: {}",
            text
        );
        assert!(
            text.contains("=== h:test(%2) ==="),
            "should have test section header, got: {}",
            text
        );
        // Build entries should be contiguous
        let build_pos = text.find("compiling").unwrap();
        let linking_pos = text.find("linking").unwrap();
        let test_a_pos = text.find("test_a PASS").unwrap();
        // compiling and linking should both appear before test_a (grouped)
        assert!(
            build_pos < test_a_pos && linking_pos < test_a_pos,
            "build entries should be grouped before test entries"
        );
    }

    #[tokio::test]
    async fn history_per_source_section_order() {
        // Phase 2 (DC33): sections appear in first-seen order.
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(201),
            rx,
        };
        let history = sub.history(HistoryOptions {
            max_entries: 100,
            render_mode: RenderMode::PerSource,
            include_omission_marker: false,
            ..Default::default()
        });

        // Source B appears first, then A, then C
        tx.send(SinkEvent::Data(make_output("h", "beta", "%2", "b1", 1)))
            .await
            .unwrap();
        tx.send(SinkEvent::Data(make_output("h", "alpha", "%1", "a1", 1)))
            .await
            .unwrap();
        tx.send(SinkEvent::Data(make_output("h", "gamma", "%3", "c1", 1)))
            .await
            .unwrap();

        drop(tx);
        history.task.await.unwrap();

        let text = history.state.lock().await.render_text();
        let beta_pos = text.find("=== h:beta(%2) ===").unwrap();
        let alpha_pos = text.find("=== h:alpha(%1) ===").unwrap();
        let gamma_pos = text.find("=== h:gamma(%3) ===").unwrap();

        assert!(
            beta_pos < alpha_pos && alpha_pos < gamma_pos,
            "sections should appear in first-seen order: beta < alpha < gamma, got beta={}, alpha={}, gamma={}",
            beta_pos, alpha_pos, gamma_pos
        );
    }

    #[tokio::test]
    async fn history_per_source_budget_isolates_sources() {
        // Phase 3 (DC33): many events from source A don't evict source B.
        let (tx, rx) = mpsc::channel(128);
        let sub = Subscription {
            id: SinkId(202),
            rx,
        };
        let history = sub.history(HistoryOptions {
            max_entries: 3, // per-source limit
            render_mode: RenderMode::PerSource,
            include_omission_marker: false,
            ..Default::default()
        });

        // Push a few from source B
        tx.send(SinkEvent::Data(make_output("h", "quiet", "%2", "q1", 1)))
            .await
            .unwrap();
        tx.send(SinkEvent::Data(make_output("h", "quiet", "%2", "q2", 2)))
            .await
            .unwrap();

        // Flood source A with many events (different panes to prevent coalescing)
        for i in 0..20 {
            // Alternate back to source A to break coalescing
            tx.send(SinkEvent::Data(make_output(
                "h",
                "noisy",
                "%1",
                &format!("n{}", i),
                i as u64,
            )))
            .await
            .unwrap();
            // Brief switch to another source to break coalescing for the next %1
            if i < 19 {
                tx.send(SinkEvent::Data(make_output(
                    "h",
                    "quiet",
                    "%2",
                    &format!("q_extra_{}", i),
                    100 + i as u64,
                )))
                .await
                .unwrap();
            }
        }

        drop(tx);
        history.task.await.unwrap();

        let text = history.state.lock().await.render_text();
        // Source B should still have entries — not evicted by A's flood
        assert!(
            text.contains("=== h:quiet(%2) ==="),
            "quiet source should still have a section, got: {}",
            text
        );
        // Source B should have some entries (at most max_entries=3)
        let quiet_section_start = text.find("=== h:quiet(%2) ===").unwrap();
        let quiet_section = &text[quiet_section_start..];
        assert!(
            quiet_section.contains("q"),
            "quiet source should have entries, section: {}",
            quiet_section
        );
    }

    #[tokio::test]
    async fn history_interleaved_mode_unchanged() {
        // Verify Interleaved mode produces identical output to pre-change behavior.
        // Same test as history_render_text_labels_on_source_change but explicit
        // about render_mode.
        let (tx, rx) = mpsc::channel(16);
        let sub = Subscription {
            id: SinkId(203),
            rx,
        };
        let history = sub.history(HistoryOptions {
            max_entries: 100,
            render_mode: RenderMode::Interleaved,
            include_omission_marker: false,
            ..Default::default()
        });

        tx.send(SinkEvent::Data(make_output(
            "web",
            "build",
            "%1",
            "compiling",
            1,
        )))
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output(
            "web",
            "deploy",
            "%2",
            "deploying",
            2,
        )))
        .await
        .unwrap();
        tx.send(SinkEvent::Data(make_output(
            "web", "build", "%1", "done", 3,
        )))
        .await
        .unwrap();

        drop(tx);
        history.task.await.unwrap();

        let text = history.state.lock().await.render_text();
        // Interleaved: entries in arrival order with labels on source change
        assert!(text.contains("[web:build(%1)] compiling"), "got: {}", text);
        assert!(text.contains("[web:deploy(%2)] deploying"), "got: {}", text);
        assert!(text.contains("[web:build(%1)] done"), "got: {}", text);
        // Should NOT have per-source section headers
        assert!(
            !text.contains("==="),
            "interleaved mode should not have section headers, got: {}",
            text
        );
    }

    #[test]
    fn poll_history_per_source_render() {
        // Phase 2 (DC33): PollHistory with push_text_for_source and PerSource mode.
        let mut history = PollHistory::new(100, 0)
            .with_render_mode(RenderMode::PerSource)
            .with_omission_marker(false);

        history.push_text_for_source("build", "compiling\n".to_string());
        history.push_text_for_source("test", "test_a PASS\n".to_string());
        history.push_text_for_source("build", "linking\n".to_string());
        history.push_text_for_source("test", "test_b PASS\n".to_string());

        let text = history.render_text();
        assert!(
            text.contains("=== build ==="),
            "should have build section, got: {}",
            text
        );
        assert!(
            text.contains("=== test ==="),
            "should have test section, got: {}",
            text
        );

        // Build entries grouped
        let compile_pos = text.find("compiling").unwrap();
        let linking_pos = text.find("linking").unwrap();
        let test_a_pos = text.find("test_a PASS").unwrap();
        assert!(
            compile_pos < linking_pos && linking_pos < test_a_pos,
            "build entries should be grouped before test entries"
        );
    }

    #[tokio::test]
    async fn history_global_char_cap_trims_across_sources() {
        // Phase 3 (DC33): global cap trims from largest source first.
        let (tx, rx) = mpsc::channel(64);
        let sub = Subscription {
            id: SinkId(204),
            rx,
        };
        let history = sub.history(HistoryOptions {
            max_entries: 100,             // large per-source limit
            max_render_chars: 0,          // no per-source char limit
            global_max_render_chars: 100, // tight global cap
            render_mode: RenderMode::PerSource,
            include_omission_marker: false,
            ..Default::default()
        });

        // Source A: lots of text
        for i in 0..10 {
            tx.send(SinkEvent::Data(make_output(
                "h",
                "big",
                "%1",
                &format!("big-line-{:03}\n", i),
                i as u64,
            )))
            .await
            .unwrap();
            // Break coalescing with B
            tx.send(SinkEvent::Data(make_output(
                "h",
                "small",
                "%2",
                &format!("s{}\n", i),
                i as u64,
            )))
            .await
            .unwrap();
        }

        drop(tx);
        history.task.await.unwrap();

        let snap = history.state.lock().await.snapshot();
        // Total rendered chars should be under the global cap
        let total_chars: usize = snap
            .entries
            .iter()
            .map(|e| e.rendered_chars(&LabelFormat::Bracketed))
            .sum();
        // The rendered_chars from snapshot may not include section headers,
        // but the trimming should have brought entries down.
        assert!(
            snap.omitted_entries > 0,
            "global cap should have trimmed some entries"
        );
        // The small source should still have some entries
        let has_small = snap.entries.iter().any(|e| match e {
            HistoryEntry::Output { source, .. } => source.short().contains("small"),
            _ => false,
        });
        assert!(
            has_small,
            "small source should still have entries after global trimming, total_chars={}, entries={:?}",
            total_chars,
            snap.entries.iter().map(|e| e.source_key()).collect::<Vec<_>>()
        );
    }
}
