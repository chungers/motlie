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
