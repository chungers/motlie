mod attach;
pub mod capture;
pub mod control;
pub mod discovery;
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

pub use attach::AttachExit;
pub use capture::{
    has_visible_text, normalize_plain_text, normalize_screen_stable, overlap_deduplicate,
    pane_tail_excerpt, strip_ansi,
};
pub use error::{Error, Result};
pub use filter::{
    clean_line, diff_new_lines, is_tui_chrome, AgentTuiFilter, ContentFilter, RawFilter,
    ShellFilter,
};
pub use fleet::{Fleet, HostStatus, SessionMonitorStatus};
pub use host::{
    ExecHandle, HostEvent, HostEventStream, HostHandle, PaneTargetTree, SessionTargetTree,
    SessionWatchHandle, SessionWatchOptions, Target, WindowTargetTree,
};
pub use keys::{KeySequence, SpecialKey};
pub use monitor::{MonitorExitReason, MonitorHandle, MonitorHealth, SessionMonitorHandle};
pub use sink::{
    CallbackSink, FlushPolicy, HistoryEntry, HistoryHandle, HistoryOptions, HistorySnapshot,
    JoinedStream, LabelFormat, OutputBus, PipeHandle, PollHistory, RenderMode, SinkEvent,
    SinkFilter, SinkId, SinkKind, SourceAccumulator, SourceLabel, StreamChunk, Subscription,
    TargetOutput,
};
pub use sinks::stdio::{StdioFormat, StdioSink};
pub use transport::{
    ShellChannelKind, ShellEvent, SshConfig, SshTransport, TransportKind, SSH_DEFAULT_PORT,
};
pub use types::{
    CaptureNormalizeMode, CaptureOptions, CaptureResult, ClientInfo, CreateSessionOptions,
    CreateWindowOptions, ExecId, ExecOutput, ExecState, FidelityIssue, GeometrySnapshot,
    HostKeyPolicy, OutputFidelity, PaneAddress, PaneGeometry, PaneInfo, ScrollbackQuery, SessionId,
    SessionInfo, SessionListing, SplitDirection, SplitPaneOptions, SplitSize, TargetAddress,
    TargetLevel, TargetSpec, TmuxSocket, TransferOptions, WindowInfo,
};
