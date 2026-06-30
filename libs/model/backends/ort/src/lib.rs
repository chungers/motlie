use std::path::Path;

#[cfg(feature = "cuda")]
use motlie_model::metrics_runtime::should_force_cpu;
use motlie_model::ModelError;
use ort::session::builder::SessionBuilder;
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

/// Returns true only when the `cuda` feature is enabled AND the ONNX Runtime
/// this process linked against was compiled with the CUDA execution provider.
///
/// Static ORT packages without the CUDA EP (e.g. the k2-fsa `sherpa-onnx`
/// CPU archives) make `ort`'s CUDA registration a silent CPU fallback, so
/// resolution must consult the runtime provider list rather than trusting
/// the Cargo feature alone. See issues #495/#496/#497.
pub fn cuda_ep_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use ort::ep::ExecutionProvider as _;
        ort::ep::CUDA::default().is_available().unwrap_or(false)
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn resolved_execution_target(target: OrtExecutionTarget) -> OrtResolvedExecutionTarget {
    #[cfg(feature = "cuda")]
    {
        if matches!(target, OrtExecutionTarget::CpuOnly)
            || should_force_cpu()
            || !cuda_ep_available()
        {
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

/// Applies the resolved execution target to a session builder. When the
/// target resolves to CUDA the execution provider is registered with
/// `error_on_failure`, so a session can never silently run on CPU while the
/// backend reports CUDA.
pub fn apply_execution_target(
    backend: &'static str,
    builder: SessionBuilder,
    target: OrtExecutionTarget,
) -> Result<SessionBuilder, ModelError> {
    #[cfg(not(feature = "cuda"))]
    let _ = backend;

    match resolved_execution_target(target) {
        OrtResolvedExecutionTarget::Cpu => Ok(builder),
        #[cfg(feature = "cuda")]
        OrtResolvedExecutionTarget::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|err| ModelError::BackendInitialization {
                backend,
                message: format!("failed to configure CUDA execution provider: {err}"),
            }),
        #[cfg(not(feature = "cuda"))]
        OrtResolvedExecutionTarget::Cuda => Ok(builder),
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
    let builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend,
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    let mut builder = apply_execution_target(backend, builder, target)?;

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
