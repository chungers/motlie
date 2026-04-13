//! llama.cpp backend implementations for `motlie-model`.
//!
//! This backend uses GGUF-quantized weights via the `llama-cpp-2` Rust bindings
//! to run the same model families (Qwen3, Gemma4) that the `mistralrs` backend
//! supports with safetensors weights. The two weight formats are **not**
//! interchangeable — each backend requires its own artifact set — but they
//! target identical upstream model architectures.

mod common;
mod text;

pub use text::{LlamaCppTextAdapter, LlamaCppTextArch, LlamaCppTextBundle, LlamaCppTextSpec};
