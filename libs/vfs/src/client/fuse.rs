//! FuseClient: guest-side fuser::Filesystem implementation over a transport.
//!
//! Translates fuser callbacks into FsOp requests, sends them through the
//! transport, and maps FsResult responses back to fuser reply types.

// This module is only compiled when the "client" feature is enabled,
// which brings in the `fuser` dependency.

// Placeholder — FuseClient implementation requires mapping all fuser
// callbacks to FsOp/FsResult. This is a significant amount of boilerplate
// that becomes useful only when mounting a real FUSE filesystem.
// For the vertical slice, the transport layer (VsockClientTransport) is
// the critical path — it enables end-to-end testing over duplex streams
// without requiring a real FUSE mount.
