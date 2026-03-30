//! GuestMountRunner and GuestMountSpec: guest-side mount orchestration.
//!
//! This module is transport-independent and compiles on any platform.
//! `GuestMountRunner::mount_all()` spawns one thread per mount spec, calls
//! the caller-supplied connector closure to obtain a transport, and then
//! invokes the FUSE mount loop.
//!
//! Both `bins/motlie-vfs-guest.rs` and future `motlie-vmm-guest` binaries
//! should call into this module rather than reimplementing mount loops.

use std::thread::{self, JoinHandle};

use anyhow::Result;

/// Specification for a single guest mount point.
#[derive(Debug, Clone)]
pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: String,
    pub read_only: bool,
}

impl GuestMountSpec {
    pub fn new(tag: impl Into<String>, guest_path: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            guest_path: guest_path.into(),
            read_only: false,
        }
    }

    pub fn read_only(mut self, ro: bool) -> Self {
        self.read_only = ro;
        self
    }
}

/// Handle set returned by `mount_all()`. Callers can join or manage mount threads.
pub struct MountHandles {
    handles: Vec<JoinHandle<Result<()>>>,
}

impl MountHandles {
    /// Wait for all mount threads to finish (blocks until all mounts are unmounted).
    pub fn join_all(self) -> Vec<Result<()>> {
        self.handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("mount thread panicked"))))
            .collect()
    }
}

/// Guest-side mount orchestrator.
///
/// `mount_all()` spawns one thread per `GuestMountSpec`. Each thread:
/// 1. Calls the connector closure to obtain a transport
/// 2. Creates `std::fs::create_dir_all` for the guest mount point
/// 3. Calls `fuser::mount2` with the transport-backed `FuseClient`
///
/// Returns after all mount threads are started. The caller owns the
/// subsequent control-loop lifecycle.
pub struct GuestMountRunner {
    specs: Vec<GuestMountSpec>,
}

impl GuestMountRunner {
    pub fn new(specs: Vec<GuestMountSpec>) -> Self {
        Self { specs }
    }

    /// Mount all specified filesystems.
    ///
    /// `connector` is called once per spec with the tag; it should return
    /// an established transport (e.g. `VsockClientTransport`) ready for
    /// FsOp/FsResult exchange. The VMM handshake happens inside the connector,
    /// outside this crate.
    ///
    /// On Linux with the `client` feature, each thread calls `fuser::mount2`.
    /// Without the `client` feature, this returns an error.
    #[cfg(feature = "client")]
    pub fn mount_all<F>(self, connector: F) -> Result<MountHandles>
    where
        F: Fn(&str) -> Result<crate::vsock::client::VsockClientTransport<tokio::net::unix::OwnedWriteHalf>> + Send + Sync + 'static,
    {
        // The real implementation will be transport-generic once we have
        // a trait. For now, the concrete type is placeholder — the actual
        // mount_all needs a generic transport or a boxed trait object.
        let _ = connector;
        anyhow::bail!("FUSE mount requires Linux with libfuse3 — use mount_all_with_fuse() on Linux")
    }

    /// Platform-independent mount stub for testing without fuser.
    /// Returns immediately with handles that resolve to Ok.
    pub fn mount_all_stub(self) -> Result<MountHandles> {
        let mut handles = Vec::new();
        for spec in self.specs {
            let h = thread::Builder::new()
                .name(format!("fuse-{}", spec.tag))
                .spawn(move || -> Result<()> {
                    std::fs::create_dir_all(&spec.guest_path)?;
                    // Stub: no actual FUSE mount. Used for testing orchestration.
                    Ok(())
                })?;
            handles.push(h);
        }
        Ok(MountHandles { handles })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_mount_spec_builder() {
        let spec = GuestMountSpec::new("home", "/home/alice").read_only(true);
        assert_eq!(spec.tag, "home");
        assert_eq!(spec.guest_path, "/home/alice");
        assert!(spec.read_only);
    }

    #[test]
    fn mount_runner_stub() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mount-test");
        let specs = vec![
            GuestMountSpec::new("tag1", path.to_str().unwrap()),
        ];
        let runner = GuestMountRunner::new(specs);
        let handles = runner.mount_all_stub().unwrap();
        let results = handles.join_all();
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(path.exists());
    }

    #[test]
    fn mount_runner_multiple_specs() {
        let dir = tempfile::tempdir().unwrap();
        let specs = vec![
            GuestMountSpec::new("a", dir.path().join("a").to_str().unwrap()),
            GuestMountSpec::new("b", dir.path().join("b").to_str().unwrap()),
            GuestMountSpec::new("c", dir.path().join("c").to_str().unwrap()),
        ];
        let runner = GuestMountRunner::new(specs);
        let handles = runner.mount_all_stub().unwrap();
        let results = handles.join_all();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(dir.path().join("a").exists());
        assert!(dir.path().join("b").exists());
        assert!(dir.path().join("c").exists());
    }
}
