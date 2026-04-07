use thiserror::Error;

use crate::backend::ch::shell::{ChShellBackend, ChShellError, ChShellHandle};
use crate::orchestrator::PreparedGuest;

pub mod ch;
pub mod motlie;
pub mod vz;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    ChShell,
    ChForkExec,
    ChVmmThread,
    Vz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmBackendCapabilities {
    pub same_process_vmm: bool,
    pub supports_api_socket: bool,
    pub supports_event_monitor: bool,
    pub supports_fd_handoff: bool,
    pub supports_memfd_boot_artifacts: bool,
    pub supports_guest_metrics: bool,
}

#[derive(Debug)]
pub enum BackendHandle {
    ChShell(ChShellHandle),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendShutdownOutcome {
    pub api_attempted: bool,
    pub forced: Option<&'static str>,
}

impl BackendHandle {
    pub fn pid(&self) -> Option<u32> {
        match self {
            Self::ChShell(handle) => handle.pid,
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::ChShell(_) => BackendKind::ChShell,
        }
    }

    pub fn has_exited(&self) -> Result<bool, BackendError> {
        match self {
            Self::ChShell(handle) => Ok(handle.has_exited()?),
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to create runtime directory {path}: {source}")]
    CreateRuntimeDir {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    ChShell(#[from] ChShellError),
    #[error("backend {0:?} is not implemented yet")]
    UnsupportedBackend(BackendKind),
    #[error("backend handle {actual:?} does not match expected backend {expected:?}")]
    HandleKindMismatch {
        expected: BackendKind,
        actual: BackendKind,
    },
}

pub trait VmBackend {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> VmBackendCapabilities;
    fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError>;
    fn shutdown(&self, handle: &BackendHandle) -> Result<BackendShutdownOutcome, BackendError>;
}

#[derive(Debug, Clone)]
pub struct BackendSet {
    pub ch_shell: ChShellBackend,
}

impl Default for BackendSet {
    fn default() -> Self {
        Self {
            ch_shell: ChShellBackend::new(),
        }
    }
}

impl BackendSet {
    pub fn boot(
        &self,
        kind: BackendKind,
        prepared: &PreparedGuest,
    ) -> Result<BackendHandle, BackendError> {
        match kind {
            BackendKind::ChShell => self.ch_shell.boot(prepared),
            kind => Err(BackendError::UnsupportedBackend(kind)),
        }
    }

    pub fn shutdown(
        &self,
        kind: BackendKind,
        handle: &BackendHandle,
    ) -> Result<BackendShutdownOutcome, BackendError> {
        match kind {
            BackendKind::ChShell => self.ch_shell.shutdown(handle),
            kind => Err(BackendError::UnsupportedBackend(kind)),
        }
    }
}
