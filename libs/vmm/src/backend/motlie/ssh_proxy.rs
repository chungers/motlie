use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::ca::SshCa;
use crate::orchestrator::PreparedGuest;
use crate::ssh::{
    spawn_guest_ssh_bridge, ExecOutput, GuestBridgeHandle, GuestPtySession, GuestRegistry,
    PtyRequest, SshProxyError,
};

#[derive(Clone)]
pub struct MotlieSshProxyBacking {
    ca: Arc<SshCa>,
    registry: GuestRegistry,
}

impl std::fmt::Debug for MotlieSshProxyBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotlieSshProxyBacking")
            .finish_non_exhaustive()
    }
}

impl MotlieSshProxyBacking {
    pub fn new(ca: Arc<SshCa>, registry: GuestRegistry) -> Self {
        Self { ca, registry }
    }

    pub fn provision(
        &self,
        prepared: &PreparedGuest,
    ) -> Result<Option<MotlieSshProxyHandle>, SshProxyError> {
        Ok(Some(MotlieSshProxyHandle {
            inner: spawn_guest_ssh_bridge(
                prepared
                    .runtime_paths
                    .vsock_socket
                    .to_string_lossy()
                    .as_ref(),
                Arc::clone(&self.ca),
                prepared.guest.guest_id.clone(),
                prepared.guest.ssh.clone(),
                Arc::clone(&self.registry),
            )?,
        }))
    }
}

pub struct MotlieSshProxyHandle {
    inner: GuestBridgeHandle,
}

impl std::fmt::Debug for MotlieSshProxyHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotlieSshProxyHandle")
            .finish_non_exhaustive()
    }
}

impl MotlieSshProxyHandle {
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), SshProxyError> {
        self.inner.wait_ready(timeout).await
    }

    pub async fn exec(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<ExecOutput, SshProxyError> {
        self.inner.exec(command, timeout).await
    }

    pub async fn open_pty(
        &self,
        request: PtyRequest,
        timeout: Duration,
    ) -> Result<GuestPtySession, SshProxyError> {
        self.inner.open_pty(request, timeout).await
    }

    pub fn shutdown(&self) -> Result<(), SshProxyError> {
        self.inner.shutdown()
    }

    pub fn bridge_socket_path(&self) -> &Path {
        self.inner.uds_path()
    }
}
