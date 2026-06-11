#[cfg(any(
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "piper-cuda",
    feature = "sherpa-onnx-cuda",
))]
compile_error!(
    "motlie-models example feature matrix: chat/tool examples that use cel-cxx must be built without ORT-backed Piper, Moonshine, or Sherpa features. Build chat/tool examples and ORT-backed ASR/TTS examples as separate cargo invocations."
);
