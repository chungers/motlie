mod attach;
pub mod capture;
pub mod control;
mod discovery;
pub mod error;
pub mod filter;
pub mod fleet;
pub mod host;
pub mod keys;
pub mod monitor;
pub mod sink;
pub mod sinks;
pub mod transport;
pub mod types;
mod uri;

pub use attach::{AttachExit, AttachOptions};
pub use capture::{
    has_visible_text, normalize_plain_text, normalize_screen_stable, overlap_deduplicate,
    pane_tail_excerpt, strip_ansi,
};
pub use error::{Error, Result};
pub use filter::{
    clean_line, diff_new_lines, is_tui_chrome, AgentTuiFilter, ContentFilter, RawFilter,
    ShellFilter,
};
pub use fleet::{
    Fleet, FleetSessionInfo, FleetSessionSnapshot, FleetSnapshotOptions, FleetTargetSpec,
    HostStatus, ResolvedFleetTarget, SessionMonitorStatus,
};
pub use host::{
    ExecHandle, HostEvent, HostEventStream, HostHandle, PaneTargetTree, SessionEnvironment,
    SessionStatus, SessionStatusOverrides, SessionStatusSnapshot, SessionTags, SessionTargetTree,
    SessionWatchHandle, SessionWatchOptions, SessionWindowStyleOverrides, SessionWindowStyles,
    SessionWindowStylesSnapshot, Target, WindowStyleSnapshot, WindowTargetTree,
};
pub use keys::{KeySequence, SpecialKey};
pub use monitor::{MonitorExitReason, MonitorHandle, MonitorHealth, SessionMonitorHandle};
pub use sink::{
    CallbackSink, FlushPolicy, HistoryEntry, HistoryHandle, HistoryOptions, HistorySnapshot,
    JoinedStream, LabelFormat, LateEventPolicy, OutputBus, PipeHandle, PollHistory, RenderMode,
    RenderOptions, SinkEvent, SinkFilter, SinkId, SinkKind, SourceAccumulator, SourceLabel,
    StreamChunk, Subscription, TargetOutput, TimelineCursor, TimelineEntry, TimelineEntryKind,
    TimelineHandle, TimelineMarkerScope, TimelineOptions, TimelineOrdering, TimelinePage,
    TimelineRenderPage,
};
pub use sinks::stdio::{StdioFormat, StdioSink};
pub use transport::{
    ShellChannelKind, ShellEvent, SshConfig, SshTransport, TransportKind, SSH_DEFAULT_PORT,
};
pub use types::{
    CaptureNormalizeMode, CaptureOptions, CaptureResult, ClientInfo, CreateSessionOptions,
    CreateWindowOptions, ExecId, ExecOutput, ExecState, FidelityIssue, GeometrySnapshot,
    HostKeyPolicy, OutputFidelity, PaneAddress, PaneGeometry, PaneInfo, ScrollbackQuery,
    SessionClientActivity, SessionEnvVar, SessionId, SessionInfo, SessionTag, SplitDirection,
    SplitPaneOptions, SplitSize, StatusLeft, StatusLeftLength, StatusStyle, TargetAddress,
    TargetLevel, TargetSpec, TmuxSocket, TmuxStyle, TransferOptions, WindowInfo, WindowStyle,
    SESSION_ENV_VAR_VALUE_MAX_BYTES, SESSION_TAG_VALUE_MAX_BYTES, STATUS_LEFT_LENGTH_MAX,
    STATUS_LEFT_MAX_BYTES, STATUS_STYLE_MAX_BYTES, TMUX_STYLE_MAX_BYTES,
};
