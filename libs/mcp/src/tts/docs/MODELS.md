# TTS Model Research & Recommendations

This document captures research findings for local Text-to-Speech (TTS) models suitable for running on MacOS (Apple Silicon) and NVIDIA DGX Spark.

## Executive Summary

| Model | Rust Native | Voice Cloning | Emotion Control | Best For |
|-------|-------------|---------------|-----------------|----------|
| **Fish Speech / OpenAudio** | Yes (Candle) | Yes (10-30s) | No | DGX Spark, highest quality |
| **Chatterbox** | ONNX only | Yes (5-20s) | Yes | Emotion control, multilingual |
| **Kokoro** | ONNX | No | No | Lightweight/edge devices |
| **Piper** | C++ (ONNX) | No | No | Embedded, Raspberry Pi |

**Recommended:**
- **DGX Spark**: Fish Speech (fish-speech.rs) - native Rust, CUDA optimized
- **MacOS**: Fish Speech (Metal) or Chatterbox (ONNX) depending on emotion control needs

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

### Recommended Structure

```
libs/mcp/src/tts/
├── mod.rs              # TTS trait + unified interface
├── fish_speech.rs      # Fish Speech client (HTTP or embedded)
├── chatterbox_onnx.rs  # Chatterbox via ONNX Runtime
├── config.rs           # Engine configuration
└── docs/
    └── MODELS.md       # This document
```

### Unified Trait

```rust
#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// Synthesize text to audio samples
    async fn synthesize(&self, text: &str, voice: &str) -> anyhow::Result<Vec<u8>>;

    /// Clone a voice from reference audio
    async fn clone_voice(&self, name: &str, reference_audio: &[u8]) -> anyhow::Result<String>;

    /// Check if engine supports emotion control
    fn supports_emotion_control(&self) -> bool;

    /// List available voices
    async fn list_voices(&self) -> anyhow::Result<Vec<String>>;
}

pub enum TtsEngineType {
    FishSpeech,
    ChatterboxOnnx,
}

pub struct TtsConfig {
    pub engine: TtsEngineType,
    pub endpoint: Option<String>,  // For HTTP mode
    pub model_path: Option<String>, // For embedded mode
}
```

### Platform Recommendations

| Platform | Engine | Integration |
|----------|--------|-------------|
| DGX Spark (CUDA) | Fish Speech | fish-speech.rs (embedded) or HTTP sidecar |
| MacOS (Apple Silicon) | Fish Speech | fish-speech.rs with Metal |
| MacOS (needs emotion) | Chatterbox | ONNX via `ort` crate |
| Edge/Embedded | Kokoro | ONNX via `ort` crate |

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

### When to use Fish Speech

- Need highest quality output
- Running on NVIDIA GPU (DGX Spark)
- Want native Rust without Python
- Multilingual requirements
- Voice cloning needed

### When to use Chatterbox

- Need emotion control (unique feature)
- 23-language requirement
- Can tolerate ONNX complexity or Python sidecar
- Want paralinguistic features ([laugh], [cough])

### When to use Kokoro

- Resource-constrained environment
- Browser deployment needed
- Don't need voice cloning
- Prioritize size over features

---

## References

- Fish Speech Paper: https://arxiv.org/abs/2411.01156
- fish-speech.rs: https://github.com/EndlessReform/fish-speech.rs
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M
- ort crate: https://ort.pyke.io/
- mlx-rs: https://github.com/oxideai/mlx-rs
- DGX Spark: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
