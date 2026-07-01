//! Linux local in-process FUSE mount helper.
//!
//! This path uses blocking `fuser::mount2()` in a dedicated thread and keeps
//! vsock out of local embedding daemons.

use anyhow::{Context, Result};
use fuser::{MountOption, Session};
use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::client::fuse::FuseClient;
use crate::core::server::FsServer;

pub struct LocalMount {
    mountpoint: PathBuf,
    join: Option<JoinHandle<io::Result<()>>>,
}

impl LocalMount {
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    pub fn unmount(mut self) -> Result<()> {
        unmount_mountpoint(&self.mountpoint)
            .with_context(|| format!("failed to unmount {}", self.mountpoint.display()))?;
        if let Some(join) = self.join.take() {
            match join.join() {
                Ok(result) => result.with_context(|| {
                    format!("mount thread failed for {}", self.mountpoint.display())
                })?,
                Err(_) => anyhow::bail!("mount thread panicked for {}", self.mountpoint.display()),
            }
        }
        Ok(())
    }
}

pub fn mount_local(server: Arc<FsServer>, tag: &str, mountpoint: &Path) -> Result<LocalMount> {
    recover_stale_mount(mountpoint);
    std::fs::create_dir_all(mountpoint)
        .with_context(|| format!("failed to create mountpoint {}", mountpoint.display()))?;

    let request_server = Arc::clone(&server);
    let request_tag = tag.to_string();
    let client = FuseClient::new(move |op| request_server.handle_op(&request_tag, op));
    let options = local_mount_options();
    let mut session =
        Session::new(client, mountpoint, &options).context("local FUSE mount failed")?;
    let join = thread::Builder::new()
        .name(format!("vfs-local-{tag}"))
        .spawn(move || session.run())
        .context("failed to spawn local FUSE mount thread")?;

    Ok(LocalMount {
        mountpoint: mountpoint.to_path_buf(),
        join: Some(join),
    })
}

pub fn local_mount_options() -> Vec<MountOption> {
    vec![
        MountOption::FSName("motlie-vfs".to_string()),
        MountOption::RO,
        MountOption::NoSuid,
        MountOption::NoDev,
    ]
}

fn recover_stale_mount(mountpoint: &Path) {
    let stale = match std::fs::read_dir(mountpoint) {
        Ok(_) => false,
        Err(err) => err.raw_os_error() == Some(libc::ENOTCONN),
    };
    if stale {
        let _ = unmount_mountpoint(mountpoint);
    }
}

fn unmount_mountpoint(mountpoint: &Path) -> io::Result<()> {
    match Command::new("fusermount3")
        .arg("-u")
        .arg(mountpoint)
        .status()
    {
        Ok(status) if status.success() => return Ok(()),
        Ok(_) | Err(_) => {}
    }

    let mountpoint = CString::new(mountpoint.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "mountpoint contains NUL"))?;
    let result = unsafe { libc::umount2(mountpoint.as_ptr(), libc::MNT_DETACH) };
    if result == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    if matches!(err.raw_os_error(), Some(libc::EINVAL | libc::ENOENT)) {
        Ok(())
    } else {
        Err(err)
    }
}
