use std::path::Path;

use crate::guestfs::{GuestFsError, GuestFsHandle};
use crate::spec::GuestSpec;

#[derive(Debug, Default, Clone, Copy)]
pub struct MotlieVfsBacking;

impl MotlieVfsBacking {
    pub fn new() -> Self {
        Self
    }

    pub async fn provision(&self, guest: &GuestSpec) -> Result<Option<MotlieVfsHandle>, GuestFsError> {
        if guest.mounts.is_empty() {
            return Ok(None);
        }

        Ok(Some(MotlieVfsHandle {
            inner: GuestFsHandle::provision(guest).await?,
        }))
    }
}

pub struct MotlieVfsHandle {
    inner: GuestFsHandle,
}

impl std::fmt::Debug for MotlieVfsHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotlieVfsHandle")
            .field("socket_path", &self.inner.socket_path())
            .finish()
    }
}

impl MotlieVfsHandle {
    pub async fn wait_ready(&self, timeout: std::time::Duration) -> Result<(), GuestFsError> {
        self.inner.wait_until_ready(timeout).await
    }

    pub fn socket_path(&self) -> &Path {
        self.inner.socket_path()
    }

    pub fn required_mount_tags(&self) -> &[String] {
        self.inner.required_mount_tags()
    }

    pub fn shutdown(&self) -> Result<(), GuestFsError> {
        self.inner.shutdown()
    }
}
