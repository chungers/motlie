pub mod types;
pub mod keys;
pub mod transport;
pub mod discovery;
pub mod capture;
pub mod control;
pub mod host;
pub mod sink;
pub mod sinks;
pub mod monitor;
pub mod fleet;
mod uri;

pub use types::*;
pub use keys::{KeySequence, SpecialKey};
pub use transport::{TransportKind, ShellChannelKind, ShellEvent, SshTransport, SshConfig, MockFsEntry};
pub use host::{ExecHandle, HostHandle, Target};
pub use capture::{normalize_screen_stable, normalize_plain_text, strip_ansi, overlap_deduplicate};
pub use sink::{
    TargetOutput, SinkEvent, SinkFilter, SinkId, SinkKind, CallbackSink,
    Subscription, PipeHandle, JoinedStream, StreamChunk, SourceLabel, LabelFormat, OutputBus,
    HistoryHandle, HistoryOptions, HistorySnapshot, HistoryEntry,
};
pub use sinks::stdio::{StdioSink, StdioFormat};
pub use monitor::{SessionMonitorHandle, MonitorHandle, MonitorHealth, MonitorExitReason};
pub use fleet::{Fleet, HostStatus, SessionMonitorStatus};
