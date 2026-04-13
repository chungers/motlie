use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DriverError {
    #[error("invalid shell quoting")]
    InvalidShellQuoting,

    #[error("unknown help topic '{0}'")]
    UnknownHelpTopic(String),

    #[error("history capacity must be > 0")]
    InvalidHistoryCapacity,

    #[error("{0}")]
    Message(String),

    #[error("failed to convert rendered help to utf-8")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Clap(#[from] clap::Error),

    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),

    #[cfg(feature = "repl")]
    #[error(transparent)]
    Reedline(#[from] reedline::ReedlineError),

    #[cfg(feature = "commands-tmux")]
    #[error(transparent)]
    Tmux(#[from] motlie_tmux::Error),

    #[cfg(feature = "commands-tmux")]
    #[error(transparent)]
    Regex(#[from] regex::Error),

    #[error("resource not found: {kind} '{name}'")]
    NotFound { kind: &'static str, name: String },

    #[error("invalid argument '{name}': {reason}")]
    InvalidArgument { name: &'static str, reason: String },

    #[error("failed to persist artifact {path}: {reason}")]
    Persist { path: PathBuf, reason: String },

    #[error(transparent)]
    Asciicast(#[from] crate::term::asciicast::AsciicastError),
}

impl DriverError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub fn invalid_argument(name: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidArgument {
            name,
            reason: reason.into(),
        }
    }
}

pub type DriverResult<T> = std::result::Result<T, DriverError>;
