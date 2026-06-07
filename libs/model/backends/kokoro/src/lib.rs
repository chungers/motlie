//! Kokoro-82M ONNX backend integration.

pub mod common;
pub mod speech;

pub use speech::{KokoroHandle, KokoroSpeechAdapter, KokoroSpeechBundle, KokoroSpeechSpec};
