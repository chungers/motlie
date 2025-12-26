# TTS Model Research & Recommendations

This document captures research findings for local Text-to-Speech (TTS) models suitable for running on MacOS (Apple Silicon) and NVIDIA DGX Spark.

## Primary Engine: Fish Speech (fish-speech.rs)

**Fish Speech is the recommended cross-platform TTS engine for motlie.**

### Why Fish Speech?

| Requirement | Fish Speech |
|-------------|-------------|
| **Native Rust** | ✅ Pure Rust via Candle.rs |
| **Cross-platform** | ✅ MacOS (Metal) + Linux (CUDA) + CPU |
| **No Python** | ✅ Single ~15MB static binary |
| **Voice cloning** | ✅ 10-30 second reference audio |
| **Quality** | ✅ State-of-the-art (0.008 WER, 0.004 CER) |
| **OpenAI-compatible API** | ✅ `/v1/audio/speech` endpoint |

### Cross-Platform Performance

| Platform | Build Command | Real-time Factor |
|----------|---------------|------------------|
| **DGX Spark (CUDA)** | `cargo build --release --features cuda` | ~1:7 (7x faster) |
| **MacOS (Apple Silicon)** | `cargo build --release --features metal` | ~1:3 (3x faster) |
| **CPU fallback** | `cargo build --release` | ~1:1 (real-time) |

### Key Correction: MacOS Support

**Fish Speech fully supports MacOS with Metal acceleration via Candle.rs.**

The official Python Fish Speech project notes "macOS coming soon", but this refers to the Python implementation only. The **Rust implementation (fish-speech.rs)** has full Metal support and is recommended for both MacOS and Linux:

```bash
# MacOS (Apple Silicon)
cargo build --release --bin server --features metal

# Linux (NVIDIA GPU)
cargo build --release --bin server --features cuda
```

---

## Model Comparison Summary

| Model | Rust Native | Voice Cloning | Emotion Control | Best For |
|-------|-------------|---------------|-----------------|----------|
| **Fish Speech / OpenAudio** | ✅ Yes (Candle) | ✅ Yes (10-30s) | ❌ No | **Primary choice** |
| **Chatterbox** | ⚠️ ONNX only | ✅ Yes (5-20s) | ✅ Yes | Emotion control only |
| **Kokoro** | ⚠️ ONNX | ❌ No | ❌ No | Lightweight/edge |
| **Piper** | ⚠️ C++ (ONNX) | ❌ No | ❌ No | Embedded, RPi |

**Decision:**
- **Default**: Fish Speech (cross-platform, native Rust, high quality)
- **If emotion control required**: Chatterbox via ONNX (future consideration)

---

## Fish Speech / OpenAudio

Fish Speech has rebranded to **OpenAudio**, introducing advanced TTS models with significant quality improvements.

### Models

| Model | Parameters | Description |
|-------|------------|-------------|
| OpenAudio S1 | 4B | Flagship, highest quality (0.008 WER, 0.004 CER) |
| OpenAudio S1-mini | 0.5B | Distilled, faster inference |
| Fish Speech 1.5 | - | Previous generation, still excellent |

### Key Features

- **Zero-shot voice cloning**: 10-30 second reference audio sample
- **Multilingual**: English, Japanese, Korean, Chinese, French, German, Arabic, Spanish
- **No phoneme dependency**: Handles any language script directly
- **RLHF trained**: Both S1 and S1-mini use online Reinforcement Learning from Human Feedback

### Performance

- Real-time factor: ~1:7 on NVIDIA RTX 4090
- WER: 0.008 (English), CER: 0.004 (English)
- Trained on 1M+ hours of audio data

### Architecture

- Dual Autoregressive (Dual-AR) architecture
- Grouped Finite Scalar Vector Quantization (GFSQ)
- LLM-based linguistic feature extraction (no G2P conversion needed)

### Rust Implementation

**fish-speech.rs** - Pure Rust implementation using Candle.rs

```
GitHub: https://github.com/EndlessReform/fish-speech.rs
```

**Features:**
- Single ~15MB static binary
- No Python environment required
- Supports Fish Speech 1.2, 1.4, 1.5

**Hardware Support:**
| Platform | Build Command |
|----------|---------------|
| NVIDIA CUDA | `cargo build --release --features cuda` |
| Apple Silicon | `cargo build --release --features metal` |
| CPU | `cargo build --release` |

**Performance (fish-speech.rs):**
| Platform | Real-time Factor | Notes |
|----------|------------------|-------|
| RTX 4090 (CUDA) | ~1:7 | 7x faster than real-time playback |
| M2 MacBook Air (Metal) | ~1:3 | 3x faster than real-time playback |
| CPU | ~1:1 | Approximately real-time |

**Candle vs PyTorch (general benchmarks):**
| Metric | Candle | PyTorch |
|--------|--------|---------|
| Inference speed | 35-47% faster | baseline |
| Peak RAM | 3.2 GB | 4.7 GB |
| Memory growth | 18 MB/min | 42 MB/min |

**API Endpoints (OpenAI-compatible):**

```
POST /v1/audio/speech
{
  "input": "Hello world",
  "voice": "default",
  "response_format": "wav",
  "model": "tts-1"
}

POST /v1/audio/encoding  # Voice cloning
GET  /v1/voices          # List available voices
```

**CLI Tools:**
```bash
# Encoder (generate speaker tokens)
cargo run --release --features metal --bin encoder -- -i audio.wav

# LLM Generator (create semantic tokens)
cargo run --release --features metal --bin llama_generate -- \
  --text "Your text" --prompt-tokens speaker.npy

# Vocoder (decode to WAV)
cargo run --release --features metal --bin vocoder -- -i tokens.npy -o output.wav
```

### Resources

- GitHub: https://github.com/fishaudio/fish-speech
- Documentation: https://speech.fish.audio/
- Hugging Face: https://huggingface.co/fishaudio/fish-speech-1.5
- Rust Implementation: https://github.com/EndlessReform/fish-speech.rs
- Paper: https://arxiv.org/abs/2411.01156

---

## Chatterbox (Resemble AI)

State-of-the-art open-source TTS with unique emotion control capabilities.

### Models

| Model | Parameters | Description |
|-------|------------|-------------|
| Chatterbox | 0.5B | Full model with LLaMA backbone |
| Chatterbox-Turbo | 350M | Distilled, single-step decoder |

### Key Features

- **Emotion exaggeration control**: First open-source TTS with this feature
- **Zero-shot voice cloning**: 5-20 second reference audio
- **23 languages**: Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese
- **Paralinguistic tags**: `[cough]`, `[laugh]`, `[chuckle]` for realism
- **Neural watermarking**: Perth watermarker embedded in all outputs

### Performance

- Sub-200ms latency for real-time applications
- Outperformed ElevenLabs with 63.75% user preference in blind evaluations
- 1M+ downloads on Hugging Face, 11K+ GitHub stars

### Hardware Support

| Platform | Support Level |
|----------|---------------|
| NVIDIA CUDA | Full support |
| AMD ROCm | Full support |
| Apple Silicon MPS | Partial (known issues, often falls back to CPU) |
| CPU | Fallback mode |

### Rust Integration Options

**No native Rust binding exists.** Integration options:

#### Option 1: ONNX Runtime (Recommended for Rust)

Official ONNX models available:
- `ResembleAI/chatterbox-turbo-ONNX` - Official, production-ready

Use with `ort` crate:
```toml
[dependencies]
ort = "2.0"
tokenizers = "0.15"
```

```rust
use ort::{Session, Environment, GraphOptimizationLevel};

pub struct ChatterboxOnnx {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}
```

**Challenge:** Tokenizer must be ported to Rust (LLaMA-based BPE tokenizer).

#### Option 2: HTTP Sidecar (Python)

Run Chatterbox as a separate service:
```python
# chatterbox_server.py
from flask import Flask, request, send_file
from chatterbox import ChatterboxModel

app = Flask(__name__)
model = ChatterboxModel()

@app.route("/v1/audio/speech", methods=["POST"])
def synthesize():
    data = request.json
    wav = model.generate(data["input"],
                         audio_prompt_path=data.get("reference"),
                         exaggeration=data.get("emotion", 0.5))
    return send_file(io.BytesIO(wav), mimetype="audio/wav")
```

### Apple Silicon Issues

Current Chatterbox + MPS has known problems:
- "Placeholder storage has not been allocated on MPS device!" errors
- Models saved with CUDA device references cause loading failures
- Community workarounds fall back to CPU mode

**No native MLX port exists.** mlx-audio supports Kokoro, not Chatterbox.

### Resources

- GitHub: https://github.com/resemble-ai/chatterbox
- Official Site: https://www.resemble.ai/chatterbox/
- ONNX Models: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
- PyPI: `pip install chatterbox-tts`
- Demo: https://huggingface.co/spaces/ResembleAI/Chatterbox

---

## Kokoro

Extremely lightweight TTS model ideal for edge devices and browser deployment.

### Specifications

| Attribute | Value |
|-----------|-------|
| Parameters | 82M |
| License | Apache 2.0 |
| Architecture | StyleTTS 2 + ISTFTNet (decoder-only, no diffusion) |

### Key Features

- **Browser-first**: Runs 100% client-side via WebGPU/WebAssembly (Transformers.js)
- **Privacy**: Text never leaves device
- **Multi-language**: English, French, Korean, Japanese, Mandarin Chinese
- **Quality**: Rivals models 10-100x its size

### Limitations

- **No voice cloning** (fine-tuning only)
- Limited to supported languages

### Hardware Requirements

- Runs on CPU, GPU, or browser
- ONNX version available for broad compatibility
- Suitable for Raspberry Pi, mobile, wearables

### Resources

- Hugging Face: https://huggingface.co/hexgrad/Kokoro-82M
- GitHub: https://github.com/hexgrad/kokoro
- Web Demo: https://kokoroweb.app/

---

## Piper

Lightweight neural TTS optimized for embedded and smart-home applications.

### Key Features

- **C++ inference engine**: No Python required
- **ONNX-based**: Uses VITS architecture
- **Raspberry Pi capable**: Runs on extremely low-power devices
- **Home Assistant integration**: Go-to for smart home projects

### Limitations

- No built-in voice cloning
- Trade-off speed for flexibility

### Resources

- GitHub: https://github.com/rhasspy/piper
- Maintained by Open Home Foundation

---

## Other Notable Models

### Higgs Audio V2 (Boson AI)

- **Parameters**: 11.5GB safetensors
- **VRAM**: 16GB minimum, 24GB ideal
- **Features**: Emotions, voice cloning, sound effects, music, singing, multi-speaker, real-time translation
- **Note**: Most feature-rich but heavy resource requirements

### NeuTTS Air (Neuphonic)

- **Parameters**: 0.5B LLM backbone
- **Format**: GGUF/GGML
- **Features**: Instant voice cloning from 3 seconds, CPU/GPU support
- **Note**: First on-device super-realistic TTS with instant cloning

### XTTS-v2 (Coqui - Community Maintained)

- **Voice cloning**: 6-second sample, 17 languages
- **Latency**: <150ms streaming
- **Status**: Original company shut down (2024), now community-maintained by Idiap Research Institute
- **VRAM**: 6-8GB recommended

---

## Rust Integration Architecture

### Integration Strategy: Fish Speech

Fish Speech can be integrated in two ways:

#### Option 1: HTTP Client (Recommended for Initial Implementation)

Run fish-speech.rs as a sidecar server and call its OpenAI-compatible API:

```rust
use reqwest::Client;

pub struct FishSpeechClient {
    client: Client,
    base_url: String,  // e.g., "http://localhost:3000"
}

impl FishSpeechClient {
    pub async fn synthesize(&self, text: &str, voice: &str) -> anyhow::Result<Vec<u8>> {
        let resp = self.client
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&serde_json::json!({
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "model": "tts-1"
            }))
            .send()
            .await?;
        Ok(resp.bytes().await?.to_vec())
    }

    pub async fn clone_voice(&self, name: &str, audio: &[u8], prompt: &str) -> anyhow::Result<()> {
        // POST /v1/audio/encoding with multipart form
        let form = reqwest::multipart::Form::new()
            .text("id", name.to_string())
            .text("prompt", prompt.to_string())
            .part("file", reqwest::multipart::Part::bytes(audio.to_vec()));

        self.client
            .post(format!("{}/v1/audio/encoding", self.base_url))
            .multipart(form)
            .send()
            .await?;
        Ok(())
    }
}
```

#### Option 2: Embedded (Future)

Vendor fish-speech.rs as a library dependency for single-binary deployment.

### Module Structure

```
libs/mcp/src/tts/
├── mod.rs              # Existing TTS module (macOS `say` command)
├── types.rs            # Existing parameter types
├── server.rs           # Existing MCP server
├── fish_speech.rs      # NEW: Fish Speech HTTP client
└── docs/
    ├── MODELS.md       # This document
    └── LINUX.md        # Linux-specific notes
```

### Unified Trait

```rust
#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize text to audio (WAV bytes)
    async fn synthesize(&self, text: &str, voice: &str) -> anyhow::Result<Vec<u8>>;

    /// Clone a voice from reference audio
    async fn clone_voice(&self, name: &str, reference_audio: &[u8], prompt: &str) -> anyhow::Result<String>;

    /// List available voices
    async fn list_voices(&self) -> anyhow::Result<Vec<String>>;
}

pub enum TtsBackendType {
    /// macOS system `say` command (current implementation)
    MacOsSay,
    /// Fish Speech via HTTP (cross-platform)
    FishSpeech { endpoint: String },
}
```

### Platform Deployment

| Platform | Backend | Deployment |
|----------|---------|------------|
| **DGX Spark** | Fish Speech | `fish-speech.rs --features cuda` as sidecar |
| **MacOS** | Fish Speech | `fish-speech.rs --features metal` as sidecar |
| **MacOS (fallback)** | MacOS Say | Built-in, no external dependencies |

---

## Key Rust Crates

| Crate | Purpose | Maturity |
|-------|---------|----------|
| [`ort`](https://crates.io/crates/ort) | ONNX Runtime bindings | Production (Google, SurrealDB, Supabase) |
| [`candle-core`](https://crates.io/crates/candle-core) | ML framework (fish-speech.rs uses this) | Production |
| [`tokenizers`](https://crates.io/crates/tokenizers) | HuggingFace tokenizers | Production |
| [`mlx-rs`](https://crates.io/crates/mlx-rs) | Apple MLX bindings | Development (not ready) |

---

## Decision Matrix

### Use Fish Speech (Default Choice)

Fish Speech is the **primary TTS engine** for motlie because:

- ✅ **Cross-platform**: Same codebase for MacOS (Metal) and DGX Spark (CUDA)
- ✅ **Native Rust**: No Python dependencies, single binary deployment
- ✅ **High quality**: State-of-the-art WER/CER scores
- ✅ **Voice cloning**: Zero-shot cloning with 10-30s reference audio
- ✅ **OpenAI-compatible**: Drop-in API compatibility
- ✅ **Multilingual**: 8 languages supported

### Consider Chatterbox (Future)

Only if emotion control is a hard requirement:

- Unique emotion exaggeration control
- Paralinguistic tags ([laugh], [cough], [chuckle])
- 23 languages
- **Trade-off**: Requires ONNX runtime or Python sidecar (no native Rust)

### Consider Kokoro (Edge/Browser)

For extremely resource-constrained environments:

- 82M parameters (smallest)
- Browser deployment via WebGPU
- **Trade-off**: No voice cloning

---

## References

- Fish Speech Paper: https://arxiv.org/abs/2411.01156
- fish-speech.rs: https://github.com/EndlessReform/fish-speech.rs
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M
- ort crate: https://ort.pyke.io/
- mlx-rs: https://github.com/oxideai/mlx-rs
- DGX Spark: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- Candle vs PyTorch comparison: https://markaicode.com/rust-ai-frameworks-candle-pytorch-comparison-2025/
- metal-candle: https://github.com/GarthDB/metal-candle
