//! sherpa-onnx backend implementations for `motlie-model` ASR contracts.
//!
//! This backend targets streaming Zipformer transducer checkpoints exported as
//! ONNX encoder/decoder/joiner graphs. Inference runs through ONNX Runtime and
//! exposes the existing `TranscriptionModel` / `TranscriptionStream` contract.

mod common;
mod transcription;

pub use transcription::{
    SherpaOnnxStreamingAdapter, SherpaOnnxStreamingBundle, SherpaOnnxStreamingSpec,
};
