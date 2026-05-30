//! sherpa-onnx backend implementations for `motlie-model` ASR contracts.
//!
//! This backend targets streaming Zipformer transducer checkpoints exported as
//! ONNX encoder/decoder/joiner graphs. Inference runs through ONNX Runtime and
//! exposes the typed `StreamingTranscriber` / `TranscriptionSession` contract.

mod common;
mod transcription;

pub use transcription::{
    SherpaOnnxHandle, SherpaOnnxStream, SherpaOnnxStreamingAdapter, SherpaOnnxStreamingBundle,
    SherpaOnnxStreamingSpec,
};
