use std::path::Path;

use motlie_model::ModelError;
#[cfg(feature = "cuda")]
use motlie_model::metrics_runtime::should_force_cpu;
use ort::session::Session;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrtExecutionTarget {
    Auto,
    CpuOnly,
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

    let mut builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend,
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    #[cfg(feature = "cuda")]
    let mut builder = if matches!(target, OrtExecutionTarget::CpuOnly) || should_force_cpu() {
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
