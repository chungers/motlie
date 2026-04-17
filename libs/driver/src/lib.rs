pub mod clap;
pub mod commands;
pub mod completion;
pub mod engine;
pub mod error;
pub mod history;
pub mod naming;
#[cfg(feature = "repl")]
pub mod repl;
pub mod term;
#[cfg(any(
    all(feature = "repl", feature = "commands-tmux"),
    all(feature = "tui", feature = "commands-tmux")
))]
pub mod tmux_frontend;

pub use crate::completion::{CompletionCandidate, CompletionRequest};
pub use crate::engine::{CommandEffect, CommandEngine, CommandOutput, CommandSet};
pub use crate::error::{DriverError, DriverResult};
pub use crate::history::{HistoryBuffer, HistoryPage, HistoryRecord};
pub use crate::naming::{
    parse_qualified_name, validate_qualified_name, QualifiedName, ResolveName, ResolvedName,
};
#[cfg(feature = "repl")]
pub use crate::repl::ReplFrontend;
