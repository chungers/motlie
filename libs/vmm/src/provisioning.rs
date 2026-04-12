use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use thiserror::Error;
use tokio::sync::Mutex as AsyncMutex;

use crate::network::NetworkModes;
use crate::network_alloc::{GuestNetAllocator, GuestNetAllocatorError, GuestNetAssignment};
use crate::observability::VmObservability;
use crate::orchestrator::{
    boot, prepare, LifecycleServices, OrchestratorError, PrepareRequest, ReadinessPolicy,
    ShutdownReport, VmHandle,
};
use crate::spec::{GuestRuntimePaths, GuestSpec, RuntimeNamespace};
use crate::ssh::{ExecOutput, GuestPtySession, PrincipalResolver, PtyRequest, SshProxyError};

type GuestSpecFactory =
    dyn Fn(&ProvisioningGuestRequest) -> Result<GuestSpec, String> + Send + Sync + 'static;
type HostSeedHook = dyn Fn(&GuestSpec) -> Result<(), String> + Send + Sync + 'static;

#[derive(Debug, Clone)]
pub struct ProvisioningGuestRequest {
    pub principal: String,
    pub namespace: RuntimeNamespace,
    pub base_dir: PathBuf,
    pub network_modes: NetworkModes,
    pub net_assignment: GuestNetAssignment,
}

#[derive(Clone)]
pub struct GuestProvisionerConfig {
    pub namespace: RuntimeNamespace,
    pub base_dir: PathBuf,
    pub network_modes: NetworkModes,
    pub readiness_policy: ReadinessPolicy,
    pub services: LifecycleServices,
    pub allocator: GuestNetAllocator,
    pub ssh_ca_pubkey: String,
    pub guest_spec_factory: Arc<GuestSpecFactory>,
    pub host_seed_hook: Option<Arc<HostSeedHook>>,
}

struct ProvisioningState {
    allocator: GuestNetAllocator,
    guests: BTreeMap<String, ProvisionedGuestRecord>,
}

struct ProvisionedGuestRecord {
    spec: GuestSpec,
    net_assignment: GuestNetAssignment,
    handle: Option<Arc<VmHandle>>,
}

#[derive(Clone)]
pub struct GuestProvisioner {
    inner: Arc<GuestProvisionerInner>,
}

struct GuestProvisionerInner {
    namespace: RuntimeNamespace,
    base_dir: PathBuf,
    network_modes: NetworkModes,
    readiness_policy: ReadinessPolicy,
    services: LifecycleServices,
    ssh_ca_pubkey: String,
    guest_spec_factory: Arc<GuestSpecFactory>,
    host_seed_hook: Option<Arc<HostSeedHook>>,
    state: Mutex<ProvisioningState>,
    operation_lock: AsyncMutex<()>,
}

#[derive(Clone)]
pub struct EnsuredGuest {
    pub principal: String,
    pub created: bool,
    pub spec: GuestSpec,
    pub net_assignment: GuestNetAssignment,
    pub handle: Arc<VmHandle>,
}

#[derive(Debug, Clone)]
pub struct ProvisionedGuestSnapshot {
    pub principal: String,
    pub spec: GuestSpec,
    pub net_assignment: GuestNetAssignment,
    pub active: bool,
    pub pid: Option<u32>,
    pub runtime_paths: Option<GuestRuntimePaths>,
}

#[derive(Debug, Error)]
pub enum ProvisioningError {
    #[error("guest principal cannot be empty")]
    EmptyPrincipal,
    #[error("failed to build guest spec for principal '{principal}': {reason}")]
    BuildGuestSpec { principal: String, reason: String },
    #[error("provisioned guest id '{guest_id}' does not match principal '{principal}'")]
    GuestIdMismatch { principal: String, guest_id: String },
    #[error("failed to seed host state for principal '{principal}': {reason}")]
    SeedHostState { principal: String, reason: String },
    #[error(transparent)]
    NetworkAllocation(#[from] GuestNetAllocatorError),
    #[error(transparent)]
    Orchestrator(#[from] OrchestratorError),
    #[error("unknown guest principal '{0}'")]
    UnknownPrincipal(String),
    #[error("guest principal '{0}' is not currently booted")]
    GuestNotBooted(String),
    #[error("internal provisioning state poisoned: {0}")]
    StatePoisoned(&'static str),
}

impl GuestProvisioner {
    pub fn new(config: GuestProvisionerConfig) -> Self {
        Self {
            inner: Arc::new(GuestProvisionerInner {
                namespace: config.namespace,
                base_dir: config.base_dir,
                network_modes: config.network_modes,
                readiness_policy: config.readiness_policy,
                services: config.services,
                ssh_ca_pubkey: config.ssh_ca_pubkey,
                guest_spec_factory: config.guest_spec_factory,
                host_seed_hook: config.host_seed_hook,
                state: Mutex::new(ProvisioningState {
                    allocator: config.allocator,
                    guests: BTreeMap::new(),
                }),
                operation_lock: AsyncMutex::new(()),
            }),
        }
    }

    pub fn ssh_principal_resolver(&self) -> PrincipalResolver {
        let this = self.clone();
        Arc::new(move |principal: String| {
            let this = this.clone();
            Box::pin(async move {
                this.ensure_guest_for_principal(&principal)
                    .await
                    .map(|guest| guest.spec.guest_id.clone())
                    .map_err(|err| SshProxyError::ResolveGuest {
                        principal,
                        reason: err.to_string(),
                    })
            })
        })
    }

    pub async fn ensure_guest_for_principal(
        &self,
        principal: &str,
    ) -> Result<EnsuredGuest, ProvisioningError> {
        if principal.trim().is_empty() {
            return Err(ProvisioningError::EmptyPrincipal);
        }

        let _guard = self.inner.operation_lock.lock().await;
        let principal = principal.to_string();

        if let Some(existing) = self.active_record(&principal)? {
            let handle = existing
                .handle
                .expect("active guest snapshots always include a live handle");
            if handle.ready(&self.inner.readiness_policy).await.is_ok() {
                return Ok(EnsuredGuest {
                    principal,
                    created: false,
                    spec: existing.spec,
                    net_assignment: existing.net_assignment,
                    handle,
                });
            }

            let _ = handle.shutdown().await;
            self.clear_active_handle(&principal)?;
        }

        let (spec, created_record) = match self.record(&principal)? {
            Some(record) => (record.spec, false),
            None => {
                let net_assignment = {
                    let mut state = self.lock_state()?;
                    state.allocator.ensure(&principal)?.clone()
                };
                let request = ProvisioningGuestRequest {
                    principal: principal.clone(),
                    namespace: self.inner.namespace.clone(),
                    base_dir: self.inner.base_dir.clone(),
                    network_modes: self.inner.network_modes,
                    net_assignment,
                };
                let spec = (self.inner.guest_spec_factory)(&request).map_err(|reason| {
                    ProvisioningError::BuildGuestSpec {
                        principal: principal.clone(),
                        reason,
                    }
                })?;
                if spec.guest_id != principal {
                    return Err(ProvisioningError::GuestIdMismatch {
                        principal: principal.clone(),
                        guest_id: spec.guest_id.clone(),
                    });
                }
                if let Some(hook) = self.inner.host_seed_hook.as_ref() {
                    hook(&spec).map_err(|reason| ProvisioningError::SeedHostState {
                        principal: principal.clone(),
                        reason,
                    })?;
                }
                self.upsert_record(&principal, spec.clone(), None)?;
                (spec, true)
            }
        };

        let prepared = {
            let mut state = self.lock_state()?;
            prepare(
                PrepareRequest {
                    guest: spec.clone(),
                    namespace: self.inner.namespace.clone(),
                    network_modes: self.inner.network_modes,
                    base_dir: self.inner.base_dir.clone(),
                    ssh_ca_pubkey: Some(self.inner.ssh_ca_pubkey.clone()),
                },
                &mut state.allocator,
            )?
        };

        let net_assignment = prepared.net_assignment.clone();
        let handle = Arc::new(boot(prepared, self.inner.services.clone()).await?);
        handle.ready(&self.inner.readiness_policy).await?;
        self.upsert_record(&principal, spec.clone(), Some(Arc::clone(&handle)))?;

        Ok(EnsuredGuest {
            principal,
            created: created_record,
            spec,
            net_assignment,
            handle,
        })
    }

    pub async fn shutdown_guest(
        &self,
        principal: &str,
    ) -> Result<Option<ShutdownReport>, ProvisioningError> {
        let _guard = self.inner.operation_lock.lock().await;
        let Some(handle) = self.take_active_handle(principal)? else {
            return Ok(None);
        };
        Ok(Some(handle.shutdown().await?))
    }

    pub async fn ready(&self, principal: &str) -> Result<(), ProvisioningError> {
        let handle = self.active_handle(principal)?;
        handle.ready(&self.inner.readiness_policy).await?;
        Ok(())
    }

    pub async fn exec(
        &self,
        principal: &str,
        command: &str,
        timeout: Duration,
    ) -> Result<ExecOutput, ProvisioningError> {
        let handle = self.active_handle(principal)?;
        Ok(handle.exec(command, timeout).await?)
    }

    pub async fn open_pty(
        &self,
        principal: &str,
        request: PtyRequest,
        timeout: Duration,
    ) -> Result<GuestPtySession, ProvisioningError> {
        let handle = self.active_handle(principal)?;
        Ok(handle.open_pty(request, timeout).await?)
    }

    pub fn active_vm_handle(&self, principal: &str) -> Result<Arc<VmHandle>, ProvisioningError> {
        self.active_handle(principal)
    }

    pub fn observability(&self, principal: &str) -> Result<VmObservability, ProvisioningError> {
        let handle = self.active_handle(principal)?;
        Ok(handle.observability())
    }

    pub fn snapshot(
        &self,
        principal: &str,
    ) -> Result<Option<ProvisionedGuestSnapshot>, ProvisioningError> {
        Ok(self
            .record(principal)?
            .map(|record| ProvisionedGuestSnapshot {
                principal: principal.to_string(),
                spec: record.spec,
                net_assignment: record.net_assignment,
                active: record.handle.is_some(),
                pid: record.handle.as_ref().and_then(|handle| handle.pid),
                runtime_paths: record
                    .handle
                    .as_ref()
                    .map(|handle| handle.runtime_paths.clone()),
            }))
    }

    pub fn guests(&self) -> Result<Vec<ProvisionedGuestSnapshot>, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state
            .guests
            .iter()
            .map(|(principal, record)| snapshot_from_record(principal.clone(), record))
            .collect())
    }

    pub fn capacity(&self) -> Result<u32, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state.allocator.capacity()?)
    }

    pub fn next_slot(&self) -> Result<u32, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state.allocator.next_slot())
    }

    pub fn remaining_capacity(&self) -> Result<u32, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state.allocator.remaining_capacity()?)
    }

    pub fn allocator_config(
        &self,
    ) -> Result<crate::network_alloc::GuestNetAllocatorConfig, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state.allocator.config().clone())
    }

    fn active_handle(&self, principal: &str) -> Result<Arc<VmHandle>, ProvisioningError> {
        let record = self
            .record(principal)?
            .ok_or_else(|| ProvisioningError::UnknownPrincipal(principal.to_string()))?;
        record
            .handle
            .ok_or_else(|| ProvisioningError::GuestNotBooted(principal.to_string()))
    }

    fn active_record(
        &self,
        principal: &str,
    ) -> Result<Option<ProvisionedGuestRecordSnapshot>, ProvisioningError> {
        Ok(self.record(principal)?.and_then(|record| {
            record.handle.as_ref()?;
            Some(record)
        }))
    }

    fn record(
        &self,
        principal: &str,
    ) -> Result<Option<ProvisionedGuestRecordSnapshot>, ProvisioningError> {
        let state = self.lock_state()?;
        Ok(state.guests.get(principal).map(snapshot_record))
    }

    fn upsert_record(
        &self,
        principal: &str,
        spec: GuestSpec,
        handle: Option<Arc<VmHandle>>,
    ) -> Result<(), ProvisioningError> {
        let mut state = self.lock_state()?;
        let net_assignment = state.allocator.ensure(principal)?.clone();
        state.guests.insert(
            principal.to_string(),
            ProvisionedGuestRecord {
                spec,
                net_assignment,
                handle,
            },
        );
        Ok(())
    }

    fn clear_active_handle(&self, principal: &str) -> Result<(), ProvisioningError> {
        let mut state = self.lock_state()?;
        let Some(record) = state.guests.get_mut(principal) else {
            return Ok(());
        };
        record.handle = None;
        Ok(())
    }

    fn take_active_handle(
        &self,
        principal: &str,
    ) -> Result<Option<Arc<VmHandle>>, ProvisioningError> {
        let mut state = self.lock_state()?;
        let Some(record) = state.guests.get_mut(principal) else {
            return Ok(None);
        };
        Ok(record.handle.take())
    }

    fn lock_state(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, ProvisioningState>, ProvisioningError> {
        self.inner
            .state
            .lock()
            .map_err(|_| ProvisioningError::StatePoisoned("provisioning state"))
    }
}

#[derive(Clone)]
struct ProvisionedGuestRecordSnapshot {
    spec: GuestSpec,
    net_assignment: GuestNetAssignment,
    handle: Option<Arc<VmHandle>>,
}

fn snapshot_record(record: &ProvisionedGuestRecord) -> ProvisionedGuestRecordSnapshot {
    ProvisionedGuestRecordSnapshot {
        spec: record.spec.clone(),
        net_assignment: record.net_assignment.clone(),
        handle: record.handle.as_ref().map(Arc::clone),
    }
}

fn snapshot_from_record(
    principal: String,
    record: &ProvisionedGuestRecord,
) -> ProvisionedGuestSnapshot {
    ProvisionedGuestSnapshot {
        principal,
        spec: record.spec.clone(),
        net_assignment: record.net_assignment.clone(),
        active: record.handle.is_some(),
        pid: record.handle.as_ref().and_then(|handle| handle.pid),
        runtime_paths: record
            .handle
            .as_ref()
            .map(|handle| handle.runtime_paths.clone()),
    }
}
