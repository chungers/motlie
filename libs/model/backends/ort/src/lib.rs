use std::path::Path;

#[cfg(feature = "cuda")]
use motlie_model::metrics_runtime::should_force_cpu;
use motlie_model::ModelError;
use ort::session::Session;

pub fn build_session(backend: &'static str, model_path: &Path) -> Result<Session, ModelError> {
    #[allow(unused_mut)]
    let mut builder = Session::builder().map_err(|err| ModelError::BackendInitialization {
        backend,
        message: format!("failed to create ONNX Runtime session builder: {err}"),
    })?;

    #[cfg(feature = "cuda")]
    let mut builder = if should_force_cpu() {
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
