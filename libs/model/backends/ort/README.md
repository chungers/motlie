# motlie-model-ort

Shared ONNX Runtime helpers for Motlie model backends.

All backends that use this crate must follow the Motlie ORT/ONNX policy:
[../../docs/ORT_ONNX_POLICY.md](../../docs/ORT_ONNX_POLICY.md).

Current eval/runbook policy:

- Link ONNX Runtime statically through the workspace-patched
  `libs/model/backends/ort/ort-sys` crate.
- Use `MOTLIE_ORT_SOURCE=sherpa-onnx`; do not use the Pyke binary source for
  DGX CUDA audio evals.
- CPU builds may consume the upstream `sherpa-onnx` static archive selected by
  the patched `ort-sys` build script.
- CUDA builds compile ONNX Runtime v1.24.2 from the submodule at
  `libs/model/backends/ort/vendor/onnxruntime` and statically link the CUDA
  execution provider. The CPU-only `sherpa-onnx` archive is rejected when
  `ort/cuda` is enabled.
- The only accepted CUDA runtime dynamic dependencies are the host CUDA driver
  (`libcuda.so`) and cuDNN 9.
- CUDA sessions are registered with `error_on_failure`, so a compiled but
  unusable CUDA EP fails at session creation instead of silently running on CPU.
