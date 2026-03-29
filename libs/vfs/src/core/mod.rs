//! Server core: FsServer, MemOverlay, inode management, events, and policy.

pub mod op;
pub mod server;
pub mod inode;
pub mod overlay;
pub mod event;
pub mod policy;

pub use op::{DirEntry, FileAttr, FileType, FsOp, FsResult, FsStats, SetAttrFields};
pub use event::{FsEvent, FsOpKind};
pub use policy::{AllowAll, PolicyFn};

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum VfsError {
    #[error("unknown mount tag: {0}")]
    UnknownTag(String),

    #[error("unknown layer: {0}")]
    UnknownLayer(String),

    #[error("overlay not enabled")]
    OverlayNotEnabled,

    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("internal inconsistency: {0}")]
    Internal(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VfsError>;
