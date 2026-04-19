# motlie-models

Curated model bundle catalog for the Motlie ecosystem.

This crate is the composition layer over `motlie-model`. It will own bundle
descriptors, catalog registration, backend binding, packaging policy, and
release/profile-facing bundle composition.

Notable support artifacts:

- `docs/EXPORT_QWEN3_TTS_ONNX.md`: exact runbook for regenerating the curated Qwen3-TTS ONNX artifacts from upstream Hugging Face safetensors.
- `scripts/export_qwen3_tts_onnx.py`: exporter used by that runbook.
