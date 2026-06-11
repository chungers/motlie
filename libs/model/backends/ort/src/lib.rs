use std::path::Path;

#[cfg(feature = "cuda")]
use motlie_model::metrics_runtime::should_force_cpu;
use motlie_model::ModelError;
use ort::session::Session;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrtExecutionTarget {
    Auto,
    CpuOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrtResolvedExecutionTarget {
    Cpu,
    Cuda,
}

pub fn resolved_execution_target(target: OrtExecutionTarget) -> OrtResolvedExecutionTarget {
    #[cfg(feature = "cuda")]
    {
        if matches!(target, OrtExecutionTarget::CpuOnly) || should_force_cpu() {
            OrtResolvedExecutionTarget::Cpu
        } else {
            OrtResolvedExecutionTarget::Cuda
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = target;
        OrtResolvedExecutionTarget::Cpu
    }
}

pub fn build_session(backend: &'static str, model_path: &Path) -> Result<Session, ModelError> {
    build_session_with_target(backend, model_path, OrtExecutionTarget::Auto)
}

pub fn build_session_with_target(
    backend: &'static str,
    model_path: &Path,
    target: OrtExecutionTarget,
) -> Result<Session, ModelError> {
    #[allow(unused_mut)]
    #[cfg(not(feature = "cuda"))]
    let _ = target;

    #[cfg(feature = "cuda")]
    let builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend,
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    #[cfg(not(feature = "cuda"))]
    let mut builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend,
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    #[cfg(feature = "cuda")]
    let mut builder = if resolved_execution_target(target) == OrtResolvedExecutionTarget::Cpu {
        builder
    } else {
        builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(|err| ModelError::BackendInitialization {
                backend,
                message: format!("failed to configure CUDA execution provider: {err}"),
            })?
    };

    builder
        .commit_from_file(model_path)
        .map_err(|err| ModelError::BackendInitialization {
            backend,
            message: format!(
                "failed to load ONNX model from `{}`: {err}",
                model_path.display()
            ),
        })
}
