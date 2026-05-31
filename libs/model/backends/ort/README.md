# motlie-model-ort

Shared ONNX Runtime helpers for Motlie model backends.

All backends that use this crate must follow the general Motlie ORT/ONNX policy:
[../../docs/ORT_ONNX_POLICY.md](../../docs/ORT_ONNX_POLICY.md).

In short: link ONNX Runtime statically, keep `ORT_PREFER_DYNAMIC_LINK` unset,
and let the ONNX Runtime source build own its third-party C/C++ dependencies.
