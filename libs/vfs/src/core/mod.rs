//! Server core: FsServer, MemOverlay, inode management, events, and policy.

pub mod event;
pub mod inode;
pub mod op;
pub mod overlay;
pub mod policy;
pub mod server;

pub use event::{FsEvent, FsOpKind};
pub use op::{DirEntry, FileAttr, FileType, FsOp, FsResult, FsStats, SetAttrFields};
pub use policy::{AllowAll, PolicyFn};
pub use server::{FsAccess, FsObserver, FsServer, FsServerBuilder};
