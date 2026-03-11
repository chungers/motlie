pub mod types;
pub mod keys;
pub mod transport;
pub mod discovery;
pub mod capture;
pub mod control;
pub mod host;

pub use types::*;
pub use keys::{KeySequence, SpecialKey};
pub use transport::{TransportKind, ShellChannelKind, ShellEvent, SshTransport, SshConfig};
pub use host::{HostHandle, Target};
pub use capture::{normalize_screen_stable, normalize_plain_text, strip_ansi, overlap_deduplicate};
