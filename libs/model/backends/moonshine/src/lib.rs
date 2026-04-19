//! Moonshine ASR backend for `motlie-model` transcription contracts.
//!
//! This backend wraps the `transcribe-rs` Moonshine streaming runtime over
//! ONNX Runtime. The current curated slice is intentionally positioned as the
//! secondary option behind sherpa-onnx: it advances the Moonshine state machine
//! on each pushed PCM chunk and can emit interim text, but its measured
//! per-chunk latency still makes it unsuitable for telephony-grade realtime use.

mod common;
mod transcription;

pub use transcription::{
    MoonshineHandle, MoonshineStream, MoonshineStreamingAdapter, MoonshineStreamingBundle,
    MoonshineStreamingSpec,
};
