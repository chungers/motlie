//! GuestMountRunner and GuestMountSpec: guest-side mount orchestration.
//!
//! This module compiles on any platform. The real FUSE mount path is gated
//! behind `#[cfg(all(feature = "client", target_os = "linux"))]`.
//!
//! `GuestMountRunner::mount_all()` spawns one thread per mount spec, calls
//! the caller-supplied connector closure to obtain a transport, wraps it in
//! a `FuseClient`, and calls `fuser::mount2()`.
//!
//! The `bins/v1/motlie-vfs-guest.rs`, `bins/v1.1/motlie-vfs-guest.rs`, and
//! future `motlie-vmm-guest` binaries
//! should call into this module rather than reimplementing mount loops.

#[cfg(all(feature = "client", feature = "vsock", target_os = "linux"))]
use std::sync::Arc;
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
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("mount thread panicked")))
            })
            .collect()
    }
}

/// Guest-side mount orchestrator.
pub struct GuestMountRunner {
    specs: Vec<GuestMountSpec>,
}

impl GuestMountRunner {
    pub fn new(specs: Vec<GuestMountSpec>) -> Self {
        Self { specs }
    }

    /// Mount all specified filesystems over FUSE.
    ///
    /// `connector` is called once per spec with the tag. It must return a
    /// connected, handshake-complete `VsockClientTransport` ready for
    /// FsOp/FsResult exchange. The VMM handshake happens inside the
    /// connector closure, outside this crate.
    ///
    /// Each thread:
    /// 1. Calls `connector(tag)` to get a transport
    /// 2. Creates a Tokio runtime for the async transport
    /// 3. Wraps the transport in a blocking `FuseClient` request function
    /// 4. Creates the mount point directory
    /// 5. Calls `fuser::mount2()` with v1 mount options (direct_io, zero-TTL)
    ///
    /// Returns after all mount threads are started. Each thread blocks in
    /// `fuser::mount2()` until the filesystem is unmounted. The caller owns
    /// any subsequent control-loop lifecycle.
    /// Mount all specified filesystems over FUSE.
    ///
    /// `connector` is called once per spec with the tag and a reference to the
    /// thread's Tokio runtime handle. The connector must use this runtime for
    /// any async work (e.g. vsock connect) so that IO resources are registered
    /// with the same reactor that will later drive the FUSE request loop.
    #[cfg(all(feature = "client", feature = "vsock", target_os = "linux"))]
    pub fn mount_all<S, F>(self, connector: F) -> Result<MountHandles>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
        F: Fn(
                &str,
                &tokio::runtime::Runtime,
            ) -> Result<crate::vsock::client::VsockClientTransport<S>>
            + Send
            + Sync
            + 'static,
    {
        let connector = Arc::new(connector);
        let mut handles = Vec::new();

        for spec in self.specs {
            let connector = Arc::clone(&connector);
            let h = thread::Builder::new()
                .name(format!("fuse-{}", spec.tag))
                .spawn(move || -> Result<()> {
                    // 1. Create a single Tokio runtime for this mount thread.
                    //    Both the connector (vsock connect) and the FUSE request
                    //    loop must use the same runtime so IO resources (AsyncFd)
                    //    are registered with the correct reactor.
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;

                    // 2. Connect transport using this thread's runtime
                    let transport = connector(&spec.tag, &rt)?;
                    let transport = Arc::new(transport);

                    // 3. Build the blocking request function for FuseClient
                    let transport_for_fuse = Arc::clone(&transport);
                    let request_fn = move |op: crate::core::op::FsOp| -> crate::core::op::FsResult {
                        let transport = Arc::clone(&transport_for_fuse);
                        match rt.block_on(transport.request(&op)) {
                            Ok(result) => result,
                            Err(_) => crate::core::op::FsResult::Error { errno: libc::EIO },
                        }
                    };

                    // 4. Create mount point
                    std::fs::create_dir_all(&spec.guest_path)?;

                    // 5. Mount via fuser
                    let fuse_client = crate::client::fuse::FuseClient::new(request_fn);
                    let opts = crate::client::fuse::v1_mount_options(spec.read_only);
                    fuser::mount2(fuse_client, &spec.guest_path, &opts)?;

                    Ok(())
                })?;
            handles.push(h);
        }

        Ok(MountHandles { handles })
    }

    /// Platform-independent mount stub for testing without fuser.
    /// Creates mount point directories but does not actually mount FUSE.
    pub fn mount_all_stub(self) -> Result<MountHandles> {
        let mut handles = Vec::new();
        for spec in self.specs {
            let h = thread::Builder::new()
                .name(format!("fuse-{}", spec.tag))
                .spawn(move || -> Result<()> {
                    std::fs::create_dir_all(&spec.guest_path)?;
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
        let specs = vec![GuestMountSpec::new("tag1", path.to_str().unwrap())];
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
    }
}
