use motlie_vnet::{VnetBackend, VnetConfig, VnetError, VnetHandle};

use crate::backend::BackendError;
use crate::orchestrator::PreparedGuest;

#[derive(Debug, Default, Clone, Copy)]
pub struct MotlieVnetBacking;

impl MotlieVnetBacking {
    pub fn new() -> Self {
        Self
    }

    pub fn provision(
        &self,
        prepared: &PreparedGuest,
    ) -> Result<Option<MotlieVnetHandle>, MotlieVnetProvisionError> {
        if !matches!(
            prepared.network_modes.egress,
            crate::network::EgressNetMode::VhostUser
        ) {
            return Ok(None);
        }

        if let Some(parent) = prepared.runtime_paths.vnet_socket.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                MotlieVnetProvisionError::Backend(BackendError::CreateRuntimeDir {
                    path: parent.to_path_buf(),
                    source,
                })
            })?;
        }

        let config = VnetConfig::builder()
            .socket_path(&prepared.runtime_paths.vnet_socket)
            .guest_ipv4(prepared.net_assignment.egress_ipv4.guest)
            .host_ipv4(prepared.net_assignment.egress_ipv4.host)
            .netmask(prepared.net_assignment.egress_ipv4.netmask)
            .dns_ipv4(prepared.net_assignment.egress_ipv4.dns)
            .mac(prepared.net_assignment.egress_mac)
            .build()?;

        Ok(Some(MotlieVnetHandle {
            inner: VnetBackend::new(config).start()?,
        }))
    }
}

pub struct MotlieVnetHandle {
    inner: VnetHandle,
}

impl std::fmt::Debug for MotlieVnetHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotlieVnetHandle").finish_non_exhaustive()
    }
}

impl MotlieVnetHandle {
    pub fn shutdown(&mut self) -> Result<(), VnetError> {
        self.inner.shutdown()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MotlieVnetProvisionError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Vnet(#[from] VnetError),
}
