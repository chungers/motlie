use thiserror::Error;

/// Errors produced by the motlie-tmux crate.
#[derive(Debug, Error)]
pub enum Error {
    /// Parse or validation error (URIs, target specs, key sequences, tmux output).
    #[error("{0}")]
    Parse(String),

    /// Requested resource not found (session, window, pane, host, subscription).
    #[error("{0}")]
    NotFound(String),

    /// Resource already exists or is already initialized.
    #[error("{0}")]
    AlreadyExists(String),

    /// Transport-level error (SSH connection, command execution, file transfer).
    #[error("{0}")]
    Transport(String),

    /// Tmux command returned an error or unexpected output.
    #[error("{0}")]
    Command(String),

    /// Invalid state or state transition.
    #[error("{0}")]
    State(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Regex compilation error.
    #[error(transparent)]
    Regex(#[from] regex::Error),

    /// Async task join error.
    #[error("task panicked: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

pub type Result<T> = std::result::Result<T, Error>;
