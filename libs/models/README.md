# motlie-models

Curated model bundle catalog for the Motlie ecosystem.

This crate is the composition layer over `motlie-model`. It will own bundle
descriptors, catalog registration, backend binding, packaging policy, and
release/profile-facing bundle composition.

Notable support artifacts:

- `examples/README.md`: runnable example index, including the chat/tool-use capability ledger by backend and model.
- `docs/BUILD_MODELS.md`: canonical host-prerequisite and build guide for the shipped curated model backends.
- `docs/DESIGN_TTS.md`: current TTS backend decision record, including the removal rationale for the non-functional Qwen3-TTS ONNX path.
