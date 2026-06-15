//! Kokoro-82M ONNX backend integration.

pub mod common;
mod incremental;
pub mod speech;

pub use incremental::KokoroIncrementalSpeechStream;
pub use speech::{KokoroHandle, KokoroSpeechAdapter, KokoroSpeechBundle, KokoroSpeechSpec};
