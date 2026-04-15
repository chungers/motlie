//! Qwen3-TTS ONNX backend for `motlie-model` speech contracts.
//!
//! This backend runs pre-exported ONNX components of the Qwen3-TTS
//! CosyVoice2 pipeline (encoder, decoder, vocoder) via ONNX Runtime.
//! The upstream model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on Hugging Face.
//!
//! ## Prerequisites
//!
//! The curated artifact set expects ONNX-exported model components:
//! - `encoder.onnx` — text/phoneme encoder
//! - `decoder.onnx` — flow-matching mel decoder
//! - `vocoder.onnx` — mel-to-waveform vocoder (BigVGAN-derived)
//! - `config.json` — model configuration and vocabulary
//!
//! See `libs/models/docs/DESIGN_TTS.md` Phase 2 for the export procedure.

mod common;
mod speech;

pub use speech::{Qwen3TtsSpeechAdapter, Qwen3TtsSpeechBundle, Qwen3TtsSpeechSpec};
