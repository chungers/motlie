use thiserror::Error;

use crate::backend::ch::shell::{ChShellError, ChShellHandle};
use crate::backend::vz::shell::{VzShellError, VzShellHandle};

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
    VzShell(VzShellHandle),
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
            Self::VzShell(handle) => handle.pid,
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::ChShell(_) => BackendKind::ChShell,
            Self::VzShell(_) => BackendKind::Vz,
        }
    }

    pub fn has_exited(&self) -> Result<bool, BackendError> {
        match self {
            Self::ChShell(handle) => Ok(handle.has_exited()?),
            Self::VzShell(handle) => Ok(handle.has_exited()?),
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
    #[error(transparent)]
    VzShell(#[from] VzShellError),
    #[error("backend {0:?} is not implemented yet")]
    UnsupportedBackend(BackendKind),
    #[error("backend handle {actual:?} does not match expected backend {expected:?}")]
    HandleKindMismatch {
        expected: BackendKind,
        actual: BackendKind,
    },
}
