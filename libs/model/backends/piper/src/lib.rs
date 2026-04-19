//! Piper backend implementations for `motlie-model` speech contracts.

mod common;
mod speech;

pub use speech::{
    PiperHandle, PiperSpeechAdapter, PiperSpeechBundle, PiperSpeechSpec, PiperSpeechStream,
};
