//! whisper.cpp backend implementations for `motlie-model` ASR contracts.
//!
//! This backend uses ggml-format Whisper model weights via the `whisper-rs`
//! Rust bindings to provide streaming voice-to-text transcription. The first
//! curated artifact is `ggml-base.en.bin` from the `ggerganov/whisper.cpp`
//! Hugging Face repository.

mod common;
mod transcription;

pub use transcription::{
    WhisperCppHandle, WhisperCppStream, WhisperCppTranscriptionAdapter,
    WhisperCppTranscriptionBundle, WhisperCppTranscriptionSpec,
};
