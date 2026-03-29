//! FsEvent, FsOpKind, and EventSender for non-blocking event emission.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Operation kind tag for event emission. One variant per `FsOp` variant,
/// without carrying request data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FsOpKind {
    Lookup,
    Getattr,
    Setattr,
    Readdir,
    Open,
    Read,
    Write,
    Create,
    Mkdir,
    Unlink,
    Rmdir,
    Rename,
    Symlink,
    Readlink,
    Release,
    Fsync,
    Statfs,
}

/// A structured filesystem event emitted by `FsServer` on every operation.
///
/// Events are delivered via `try_send` on a broadcast channel — non-blocking,
/// lossy under backpressure. The library does not persist or transport events;
/// that is the caller's responsibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FsEvent {
    pub timestamp: SystemTime,
    pub tag: String,
    pub op_kind: FsOpKind,
    pub path: String,
    pub bytes: Option<usize>,
}

impl FsOpKind {
    /// Derive the operation kind from an `FsOp` reference.
    pub fn from_op(op: &super::op::FsOp) -> Self {
        use super::op::FsOp;
        match op {
            FsOp::Lookup { .. } => Self::Lookup,
            FsOp::Getattr { .. } => Self::Getattr,
            FsOp::Setattr { .. } => Self::Setattr,
            FsOp::Readdir { .. } => Self::Readdir,
            FsOp::Open { .. } => Self::Open,
            FsOp::Read { .. } => Self::Read,
            FsOp::Write { .. } => Self::Write,
            FsOp::Create { .. } => Self::Create,
            FsOp::Mkdir { .. } => Self::Mkdir,
            FsOp::Unlink { .. } => Self::Unlink,
            FsOp::Rmdir { .. } => Self::Rmdir,
            FsOp::Rename { .. } => Self::Rename,
            FsOp::Symlink { .. } => Self::Symlink,
            FsOp::Readlink { .. } => Self::Readlink,
            FsOp::Release { .. } => Self::Release,
            FsOp::Fsync { .. } => Self::Fsync,
            FsOp::Statfs => Self::Statfs,
        }
    }
}
