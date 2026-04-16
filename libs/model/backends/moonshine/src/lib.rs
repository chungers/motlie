//! Moonshine ASR backend for `motlie-model` transcription contracts.
//!
//! This backend wraps the `transcribe-rs` Moonshine streaming runtime over
//! ONNX Runtime. The current curated slice is intentionally positioned as the
//! secondary batch/offline option behind sherpa-onnx: it uses the shared PCM
//! streaming contract, but buffers audio and emits the committed transcript on
//! final flush instead of targeting telephony-grade incremental latency.

mod common;
mod transcription;

pub use transcription::{
    MoonshineStreamingAdapter, MoonshineStreamingBundle, MoonshineStreamingSpec,
};
