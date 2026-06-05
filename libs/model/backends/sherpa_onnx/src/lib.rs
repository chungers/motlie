//! sherpa-onnx backend implementations for `motlie-model` ASR contracts.
//!
//! This backend targets streaming Zipformer transducer checkpoints exported as
//! ONNX encoder/decoder/joiner graphs. Runtime ownership stays with the
//! upstream `sherpa-onnx` Rust crate, including its online recognizer,
//! endpointing, and static native-library flow. Motlie patches
//! `sherpa-onnx-sys` so ONNX Runtime itself is linked once through the
//! workspace `ort` dependency. This crate only adapts that runtime to Motlie's
//! typed `StreamingTranscriber` /
//! `TranscriptionSession` contract.

extern crate ort as _;

mod common;
mod transcription;

pub use transcription::{
    SherpaOnnxHandle, SherpaOnnxStream, SherpaOnnxStreamingAdapter, SherpaOnnxStreamingBundle,
    SherpaOnnxStreamingSpec,
};
