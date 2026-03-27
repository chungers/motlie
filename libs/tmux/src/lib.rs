pub mod capture;
pub mod control;
pub mod discovery;
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

pub use capture::{
    has_visible_text, normalize_plain_text, normalize_screen_stable, overlap_deduplicate,
    pane_tail_excerpt, strip_ansi,
};
pub use filter::{
    AgentTuiFilter, ContentFilter, RawFilter, ShellFilter,
    clean_line, diff_new_lines, is_tui_chrome,
};
pub use fleet::{Fleet, HostStatus, SessionMonitorStatus};
pub use host::{ExecHandle, HostHandle, Target};
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
    MockFsEntry, ShellChannelKind, ShellEvent, SshConfig, SshTransport, TransportKind,
};
pub use types::*;
