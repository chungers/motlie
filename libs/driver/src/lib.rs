pub mod clap;
pub mod commands;
pub mod completion;
pub mod engine;
pub mod history;
#[cfg(feature = "repl")]
pub mod repl;
pub mod term;
#[cfg(feature = "tui")]
pub mod tui;

pub use crate::completion::{CompletionCandidate, CompletionRequest};
pub use crate::engine::{CommandEffect, CommandEngine, CommandOutput, CommandSet};
pub use crate::history::{HistoryBuffer, HistoryPage, HistoryRecord};
#[cfg(feature = "repl")]
pub use crate::repl::ReplFrontend;
#[cfg(feature = "tui")]
pub use crate::tui::TuiFrontend;
