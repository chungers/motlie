//! qwen3-tts.cpp backend implementations for `motlie-model` TTS contracts.
//!
//! This backend wraps the upstream `predict-woo/qwen3-tts.cpp` C API through a
//! pinned git submodule and exposes it via the typed `SpeechSynthesizer` /
//! `SpeechStream` contracts.
//!
//! The runtime expects GGUF artifacts in a model directory:
//! - `qwen3-tts-0.6b-q8_0.gguf` or `qwen3-tts-0.6b-f16.gguf`
//! - `qwen3-tts-tokenizer-f16.gguf`
//!
//! The curated bundle currently points at community GGUF exports because the
//! official Qwen model repository does not ship GGUF artifacts.

mod common;
mod speech;

pub use speech::{
    Qwen3TtsCppHandle, Qwen3TtsCppSpeechAdapter, Qwen3TtsCppSpeechBundle, Qwen3TtsCppSpeechSpec,
    Qwen3TtsCppSpeechStream,
};
