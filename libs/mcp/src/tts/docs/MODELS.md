# TTS Model Research & Recommendations

This document captures research findings for local Text-to-Speech (TTS) models suitable for running on MacOS (Apple Silicon) and NVIDIA DGX Spark, with a focus on Rust integration paths and embedded deployment within motlie.

> **Last updated: March 2026**

---

## Evaluation Criteria

Models are evaluated against these requirements for motlie integration:

| Criterion | Description |
|-----------|-------------|
| **Rust integration** | Native Rust (candle/tch), ONNX (ort crate), or GGUF (llama.cpp bindings) |
| **Embedded viability** | Can run as a library inside the motlie process (no sidecar) |
| **Model size** | Disk/memory footprint for deployment |
| **CPU performance** | Real-time or faster on CPU alone |
| **Voice cloning** | Zero-shot cloning from short reference audio |
| **Emotion control** | Tag-based or parametric emotion/prosody control |
| **License** | Apache 2.0 or equivalent permissive license |

---

## Comprehensive Model Comparison

### Full Matrix

| Model | Params | Formats | Rust Path | Clone | Emotion | Languages | CPU RT | Min Size | License |
|-------|--------|---------|-----------|-------|---------|-----------|--------|----------|---------|
| **Sesame CSM** | 1.1B | safetensors, GGUF | candle (csm.rs) | Yes (8-20s) | Yes | EN | Yes (GGUF) | ~1GB (Q4) | Apache 2.0 |
| **Qwen3 TTS** | 0.6B/1.7B | safetensors, GGUF | candle, tch | Yes (3s) | Yes | 10 | Batch only | ~2GB | Apache 2.0 |
| **OuteTTS** | 350M-1B | safetensors, ONNX, GGUF | ort, llama.cpp | Yes | Inherited | 12 | Yes (Q4_K) | ~200MB (Q4) | Apache 2.0 |
| **Orpheus TTS** | 150M-3B | safetensors, GGUF | llama.cpp | Yes | Yes (tags) | 5 | Via llama.cpp | ~600MB (150M) | Apache 2.0 |
| **Fish Speech S1** | 0.5B/4B | safetensors | candle (dormant) | Yes (10-30s) | Yes (50+) | 13+ | No | ~2GB | Apache 2.0 |
| **Kitten TTS Nano** | 14M | ONNX | ort | No | No | EN | Yes (<2s) | 25MB | Apache 2.0 |
| **LuxTTS** | ~123M | ONNX (ZipVoice) | ort | Yes | Yes | EN, ZH | Yes (>1x RT) | ~1GB | Apache 2.0 |
| **Chatterbox** | 350M/500M | safetensors, ONNX | ort | Yes (5-20s) | Yes (param) | 23 | Fallback | ~1GB | MIT |
| **Kokoro** | 82M | ONNX | ort (kokoroxide) | No | No | 9 | Yes | ~200MB | Apache 2.0 |
| **Dia TTS** | 1.6B | safetensors | None | Yes | Yes (tags) | EN | No | ~10GB | Apache 2.0 |
| **Parler-TTS** | 880M/2.3B | safetensors, GGUF | TTS.cpp | No (desc.) | Via text | 8 EU | macOS only | ~1.5GB | Apache 2.0 |
| **Piper** | Various | ONNX (VITS) | subprocess | No | No | 30+ | Yes | ~50MB | MIT |

### Tier Ranking for motlie Embedded Use

**Tier 1 — Production-ready Rust integration:**

| Model | Why |
|-------|-----|
| **Sesame CSM** | csm.rs is a mature candle implementation with GGUF Q4/Q8, Metal + CUDA + CPU, OpenAI-compatible API server |
| **Qwen3 TTS** | Multiple Rust implementations (candle, tch), GGUF via qwen3-tts.cpp, 3-second voice cloning, 10 languages |

**Tier 2 — Viable via GGUF/ONNX with existing Rust crates:**

| Model | Why |
|-------|-----|
| **OuteTTS** | Smallest GGUF TTS with voice cloning (~200MB Q4), runs on CPU, 12 languages |
| **Orpheus TTS** | 150M variant is edge-viable, GGUF works via llama.cpp Rust bindings, emotion tags |
| **Kitten TTS Nano** | 25MB ONNX model, sub-2s CPU synthesis, zero external deps via ort — but no voice cloning |

**Tier 3 — Requires sidecar or future work:**

| Model | Why |
|-------|-----|
| **Fish Speech S1** | Best quality but Rust impl dormant; requires Python sidecar |
| **Chatterbox** | Unique continuous emotion control but ONNX-only, complex tokenizer port |
| **Dia TTS** | Impressive multi-speaker but no ONNX/GGUF/Rust path yet |

---

## Detailed Model Profiles

### Sesame CSM (Conversational Speech Model)

1.1B total (1B backbone + 100M decoder), both Llama architecture. Uses Mimi audio codec.

**Why it's notable:** csm.rs provides a production-ready pure Rust implementation with GGUF quantization, making it the most mature embedded option.

**Rust implementation: [csm.rs](https://github.com/cartesia-one/csm.rs)**

```toml
# csm.rs supports multiple backends
[features]
metal = ["candle-core/metal"]       # macOS
cuda = ["candle-core/cuda"]         # NVIDIA
mkl = ["candle-core/mkl"]           # Intel CPU
accelerate = ["candle-core/accelerate"]  # macOS Accelerate
```

**Embedded integration sketch:**

```rust
use csm::{Model, GenerationConfig};

pub struct CsmBackend {
    model: Model,
}

impl CsmBackend {
    pub fn load(model_path: &str) -> anyhow::Result<Self> {
        // Loads GGUF Q4_K or Q8_0 checkpoint
        let model = Model::load(model_path)?;
        Ok(Self { model })
    }

    pub async fn synthesize(&self, text: &str) -> anyhow::Result<Vec<u8>> {
        let config = GenerationConfig::default();
        let audio = self.model.generate(text, &config)?;
        Ok(audio.to_wav_bytes())
    }

    pub async fn synthesize_with_reference(
        &self,
        text: &str,
        reference_audio: &[u8],
    ) -> anyhow::Result<Vec<u8>> {
        let config = GenerationConfig::default()
            .with_context_audio(reference_audio);
        let audio = self.model.generate(text, &config)?;
        Ok(audio.to_wav_bytes())
    }
}
```

| Attribute | Value |
|-----------|-------|
| Parameters | 1.1B |
| GGUF | Yes (official ggml-org, Q4_K + Q8_0) |
| Voice cloning | Yes (8-20s reference) |
| Emotion | Yes (context-adaptive, whisper/sarcasm/intensity) |
| Languages | English (primarily) |
| CPU | Yes via GGUF |
| GPU RTF | ~0.6x on RTX 4070 Ti |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/SesameAILabs/csm
- csm.rs: https://github.com/cartesia-one/csm.rs
- GGUF: https://huggingface.co/ggml-org/sesame-csm-1b-GGUF

---

### Qwen3 TTS (Alibaba)

0.6B and 1.7B variants with dual-track autoregressive architecture. Trained on 5M+ hours.

**Why it's notable:** Multiple active Rust implementations, GGUF support, 3-second voice cloning, instruction-driven emotion control, and 10 languages.

**Rust implementations:**

```rust
// Option 1: candle-based (TrevorS/qwen3-tts-rs) — pure Rust, no Python/ONNX
use qwen3_tts::Model;
let model = Model::load("Qwen3-TTS-12Hz-0.6B")?;
let audio = model.synthesize("Hello world", None)?;

// Option 2: tch-based (second-state/qwen3_tts_rs) — libtorch backend
use qwen3_tts_rs::TtsModel;
let model = TtsModel::new("model_path", Device::Cuda(0))?;
let wav = model.generate("Hello world")?;

// Option 3: Crane (lucasjinreal/Crane) — candle engine with OpenAI-compatible API
// Supports voice cloning, streaming, HTTP server
// cargo run --release --features cuda -- --model qwen3-tts
```

**GGUF inference (qwen3-tts.cpp):**
```bash
# C++ inference with GGML, F16 and Q8_0 quantization
git clone https://github.com/predict-woo/qwen3-tts.cpp
cd qwen3-tts.cpp && mkdir build && cd build
cmake .. && make -j
./qwen3-tts -m model-q8_0.gguf -t "Hello world" -o output.wav
```

| Attribute | Value |
|-----------|-------|
| Parameters | 0.6B / 1.7B |
| GGUF | Yes (community + qwen3-tts.cpp) |
| ONNX | Partial (decoder only via Crane) |
| Voice cloning | Yes (3-second reference) |
| Emotion | Yes (instruction-driven: timbre, prosody, emotion) |
| Languages | 10 (EN, ZH, JA, KO, DE, FR, RU, PT, ES, IT) |
| CPU | Batch only (not real-time) |
| GPU RTF | ~0.85-1.15x on RTX 4060 (0.6B) |
| Min VRAM | 6GB+ |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/QwenLM/Qwen3-TTS
- Rust (candle): https://github.com/TrevorS/qwen3-tts-rs
- Rust (tch): https://github.com/second-state/qwen3_tts_rs
- Rust (Crane): https://github.com/lucasjinreal/Crane
- C++ GGUF: https://github.com/predict-woo/qwen3-tts.cpp
- Paper: https://arxiv.org/abs/2601.15621

---

### OuteTTS (OuteAI)

350M to 1B variants. Pure LLM approach (Qwen/Llama base) with WavTokenizer.

**Why it's notable:** Smallest GGUF TTS with voice cloning, official ONNX exports with quantization levels, native llama.cpp support. The Q4_K GGUF fits in ~200MB.

**Integration paths:**

```rust
// Path 1: ONNX via ort crate
use ort::Session;

pub struct OuteTtsOnnx {
    session: Session,
}

impl OuteTtsOnnx {
    pub fn load(model_path: &str) -> anyhow::Result<Self> {
        // Available quantizations: fp32, fp16, q8, q4
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }
}

// Path 2: GGUF via llama-cpp-rs
use llama_cpp::LlamaModel;

pub struct OuteTtsGguf {
    model: LlamaModel,
}

impl OuteTtsGguf {
    pub fn load(gguf_path: &str) -> anyhow::Result<Self> {
        let model = LlamaModel::load_from_file(gguf_path, Default::default())?;
        Ok(Self { model })
    }

    pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<i16>> {
        // Generate speech tokens via LLM inference
        // Decode tokens to audio via WavTokenizer
        todo!("WavTokenizer decoding step required")
    }
}
```

**Note:** Both ONNX and GGUF paths require a WavTokenizer decoding step to convert LLM-generated speech tokens into audio waveforms. This codec is not yet available as a standalone Rust crate.

| Attribute | Value |
|-----------|-------|
| Parameters | 350M / 500M / 1B |
| GGUF | Yes (official, Q4_K) |
| ONNX | Yes (community, fp32/fp16/q8/q4) |
| Voice cloning | Yes (style/accent/emotion from reference) |
| Emotion | Inherited from reference audio |
| Languages | 12 (EN, AR, ZH, NL, FR, DE, IT, JA, KO, LT, RU, ES) |
| CPU | Yes (Q4_K GGUF) |
| Min size | ~200MB (Q4 GGUF) |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/edwko/OuteTTS
- GGUF: https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B-GGUF
- ONNX: https://huggingface.co/onnx-community/OuteTTS-0.2-500M

---

### Orpheus TTS (Canopy Labs)

150M to 3B variants built on Llama backbone with SNAC audio codec.

**Why it's notable:** The 150M variant is edge-viable, GGUF support via Ollama/llama.cpp, emotion tags, and the Llama backbone means existing Rust llama.cpp bindings work directly.

**Integration via llama.cpp bindings:**

```rust
use llama_cpp::{LlamaModel, LlamaParams};

pub struct OrpheusTts {
    model: LlamaModel,
}

impl OrpheusTts {
    pub fn load(gguf_path: &str) -> anyhow::Result<Self> {
        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(gguf_path, params)?;
        Ok(Self { model })
    }

    pub fn synthesize(&self, text: &str, emotion: Option<&str>) -> anyhow::Result<Vec<i16>> {
        // Format prompt with optional emotion tag
        let prompt = match emotion {
            Some(e) => format!("<|audio|>{} <{}>{}<|eoa|>", "default", e, text),
            None => format!("<|audio|>default{}<|eoa|>", text),
        };
        // Generate speech tokens, then decode via SNAC codec
        todo!("SNAC decoding step required")
    }
}
```

**Note:** Like OuteTTS, Orpheus requires a codec decoding step (SNAC) to produce audio from generated tokens. A Rust SNAC implementation would be needed for fully embedded use.

| Attribute | Value |
|-----------|-------|
| Parameters | 150M / 400M / 1B / 3B |
| GGUF | Yes (Ollama, community) |
| Voice cloning | Yes (zero-shot) |
| Emotion | Yes (`<laugh>`, `<sigh>`, emotional descriptions) |
| Languages | EN primary + ZH, HI, KO, ES |
| CPU | Via llama.cpp |
| Streaming latency | ~200ms (GPU) |
| Min size | ~600MB (150M) |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/canopyai/Orpheus-TTS
- GGUF: https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF
- Ollama: https://ollama.com/legraphista/Orpheus

---

### Kitten TTS Nano (KittenML)

14M parameters, ~25MB. Designed specifically for edge/embedded deployment.

**Why it's notable:** Smallest viable TTS model. Native ONNX format means direct use via `ort` crate with no codec decoding step — the model produces audio directly.

**Embedded integration:**

```rust
use ort::{Session, Value};
use ndarray::Array2;

pub struct KittenTtsNano {
    session: Session,
}

impl KittenTtsNano {
    pub fn load(onnx_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(onnx_path)?;
        Ok(Self { session })
    }

    pub fn synthesize(&self, text: &str, voice: u8) -> anyhow::Result<Vec<f32>> {
        // voice: 0-7 (4 female: Bella/Luna/Rosie/Kiki, 4 male: Jasper/Bruno/Hugo/Leo)
        let inputs = ort::inputs![
            "text" => Value::from_string(text)?,
            "voice_id" => Value::from_array(&[voice as i64])?,
        ]?;
        let outputs = self.session.run(inputs)?;
        let audio: Array2<f32> = outputs[0].try_extract_tensor()?;
        Ok(audio.into_raw_vec())
    }
}
```

| Attribute | Value |
|-----------|-------|
| Parameters | 14M (nano) / 40M (micro) / 80M (mini) |
| ONNX | Yes (native, primary format) |
| Voice cloning | No (8 fixed voices) |
| Emotion | No |
| Languages | English only |
| CPU | Yes (sub-2s on dual-core) |
| RAM | Under 1GB |
| Min size | 25MB (nano, int8+fp16) |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/KittenML/KittenTTS
- ONNX: https://huggingface.co/onnx-community/kitten-tts-nano-0.1-ONNX
- Web demo: https://clowerweb.github.io/kitten-tts-web-demo/

---

### LuxTTS

~123M parameters based on ZipVoice with flow matching. 48kHz output.

**Why it's notable:** 150x real-time on GPU, faster-than-real-time on CPU, ONNX support via ZipVoice backend, voice cloning with style prompts.

| Attribute | Value |
|-----------|-------|
| Parameters | ~123M |
| ONNX | Yes (ZipVoice backend) |
| Voice cloning | Yes (style prompts) |
| Emotion | Yes (tone/pacing prompts) |
| Languages | EN, ZH |
| CPU | Yes (>1x real-time) |
| Min size | ~1GB |
| License | Apache 2.0 |

**Resources:**
- GitHub: https://github.com/ysharma3501/LuxTTS
- HuggingFace: https://huggingface.co/YatharthS/LuxTTS

---

### Fish Speech / OpenAudio

See previous sections for full details. Summary of current status:

| Attribute | Value |
|-----------|-------|
| Parameters | 0.5B (S1-mini) / 4B (S1) |
| Rust impl | Dormant (fish-speech.rs, candle 0.8.3) |
| Voice cloning | Yes (10-30s) |
| Emotion | Yes (50+ markers, S1/S1-mini) |
| Languages | 13+ |
| Quality | Best in class (0.008 WER) |
| CPU | No (GPU required) |
| License | Apache 2.0 |

> fish-speech.rs has been dormant since February 2025. For Rust integration,
> Sesame CSM (csm.rs) and Qwen3 TTS now offer better-maintained alternatives.

---

### Chatterbox (Resemble AI)

350M (Turbo) / 500M. Unique continuous emotion exaggeration parameter.

| Attribute | Value |
|-----------|-------|
| Parameters | 350M / 500M |
| ONNX | Yes (official Turbo export) |
| Voice cloning | Yes (5-20s) |
| Emotion | Yes (continuous 0.0-1.0 parameter + paralinguistic tags) |
| Languages | 23 |
| CPU | Fallback mode |
| Apple Silicon | Partial (MPS issues, often falls back to CPU) |
| License | MIT |

---

### Kokoro

82M parameters. Browser-first, ONNX-native.

| Attribute | Value |
|-----------|-------|
| Parameters | 82M |
| ONNX | Yes |
| Rust | kokoroxide crate (ort-based) |
| Voice cloning | No |
| Emotion | No |
| Languages | 9 |
| CPU | Yes |
| Min size | ~200MB |
| License | Apache 2.0 |

---

### Other Notable Models

**Dia TTS (Nari Labs)** — 1.6B, multi-speaker dialogue generation in a single pass. Impressive but no ONNX/GGUF/Rust path yet. English only. Apache 2.0.

**Parler-TTS (HuggingFace)** — 880M/2.3B, text-description voice control (no reference audio needed). GGUF via TTS.cpp (macOS only). 8 EU languages. Apache 2.0.

**Piper** — Various small VITS models, ONNX-native, 30+ languages. No voice cloning. MIT. Best for multi-language fixed-voice synthesis.

---

## Integration Architecture for motlie

### Integration Methods Compared

| Method | Latency | Complexity | Models | Deployment |
|--------|---------|------------|--------|------------|
| **Embedded candle** | Lowest | Medium | Sesame CSM, Qwen3 TTS, Fish Speech | Single binary |
| **Embedded ONNX (ort)** | Low | Low | Kitten, OuteTTS, Kokoro, LuxTTS, Chatterbox | Single binary |
| **Embedded GGUF (llama.cpp)** | Low | Medium | OuteTTS, Orpheus, Qwen3 TTS, Sesame CSM | Single binary |
| **HTTP sidecar** | Medium | Low | Any model | Separate process |

### Recommended: Multi-Backend Architecture

```rust
use async_trait::async_trait;

/// Core trait for all TTS backends
#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize text to WAV audio bytes
    async fn synthesize(&self, request: &SynthesizeRequest) -> anyhow::Result<Vec<u8>>;

    /// List available voices
    async fn list_voices(&self) -> anyhow::Result<Vec<VoiceInfo>>;

    /// Clone a voice from reference audio (if supported)
    async fn clone_voice(&self, name: &str, reference: &[u8]) -> anyhow::Result<String> {
        anyhow::bail!("voice cloning not supported by this backend")
    }

    /// Backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

pub struct SynthesizeRequest {
    pub text: String,
    pub voice: String,
    pub emotion: Option<String>,        // e.g., "(happy)" or "0.7"
    pub format: AudioFormat,            // Wav, Pcm16
}

pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub cloned: bool,
}

pub struct BackendCapabilities {
    pub voice_cloning: bool,
    pub emotion_control: bool,
    pub languages: Vec<String>,
    pub streaming: bool,
}

#[derive(Clone)]
pub enum AudioFormat {
    Wav,
    Pcm16 { sample_rate: u32 },
}

/// Backend selector
pub enum TtsBackendType {
    /// macOS system `say` command (current implementation)
    MacOsSay,
    /// Sesame CSM via candle (embedded, GGUF)
    SesameCsm { model_path: String },
    /// Qwen3 TTS via candle (embedded)
    Qwen3Tts { model_path: String },
    /// Kitten TTS Nano via ONNX (embedded, smallest)
    KittenNano { onnx_path: String },
    /// OuteTTS via ONNX or GGUF (embedded, with voice cloning)
    OuteTts { model_path: String },
    /// Any OpenAI-compatible server (sidecar)
    HttpSidecar { endpoint: String },
}
```

### Module Structure

```
libs/mcp/src/tts/
├── mod.rs              # TTS module root, TtsBackend trait
├── types.rs            # Parameter types, SynthesizeRequest, VoiceInfo
├── server.rs           # MCP server (existing)
├── backend/
│   ├── mod.rs          # Backend selector, TtsBackendType
│   ├── macos_say.rs    # macOS `say` command (existing behavior)
│   ├── http.rs         # HTTP client for OpenAI-compatible sidecar
│   ├── onnx.rs         # ONNX Runtime backend (Kitten, OuteTTS, Kokoro)
│   ├── candle.rs       # Candle backend (Sesame CSM, Qwen3 TTS)
│   └── gguf.rs         # GGUF/llama.cpp backend (OuteTTS, Orpheus)
└── docs/
    ├── MODELS.md       # This document
    └── LINUX.md        # Linux-specific notes
```

### Platform Deployment Matrix

| Platform | Recommended Backend | Fallback |
|----------|-------------------|----------|
| **DGX Spark (CUDA)** | Sesame CSM (candle+CUDA) or Qwen3 TTS | HTTP sidecar |
| **MacOS (Apple Silicon)** | Sesame CSM (candle+Metal) | Kitten Nano (ONNX) or macOS Say |
| **Linux (CPU only)** | Kitten Nano (ONNX) or OuteTTS (GGUF Q4) | HTTP sidecar |
| **Edge / minimal** | Kitten Nano (ONNX, 25MB) | Piper (subprocess) |

---

## Decision Matrix

### For initial implementation: Sesame CSM via csm.rs

**Recommended first integration** because:

- ✅ Production-ready Rust implementation (csm.rs)
- ✅ GGUF quantization for CPU deployment
- ✅ Metal + CUDA + CPU backends via candle
- ✅ Voice cloning from reference audio
- ✅ Emotion/tone control
- ✅ OpenAI-compatible API server included
- ⚠️ English only (for now)

### For smallest footprint: Kitten TTS Nano via ort

**Recommended for edge/fallback** because:

- ✅ 25MB model, sub-2s synthesis on dual-core CPU
- ✅ Native ONNX — direct audio output, no codec step
- ✅ Zero external dependencies via ort crate
- ❌ No voice cloning, no emotion control, English only

### For multilingual + voice cloning: Qwen3 TTS

**Recommended when language coverage matters** because:

- ✅ 10 languages with 3-second voice cloning
- ✅ Multiple active Rust implementations
- ✅ GGUF available for CPU inference
- ✅ Instruction-driven emotion control
- ⚠️ 0.6B minimum — not real-time on CPU

### For maximum language coverage: OuteTTS via GGUF

**Recommended for polyglot use cases** because:

- ✅ 12 languages, voice cloning, ~200MB Q4 GGUF
- ✅ Both ONNX and GGUF paths available
- ⚠️ Requires WavTokenizer codec port to Rust

### Previous recommendation (Fish Speech) — demoted

Fish Speech / OpenAudio S1 remains the highest-quality model but:

- ❌ Rust implementation (fish-speech.rs) dormant since Feb 2025
- ❌ Requires Python sidecar for current S1/S1-mini models
- ❌ No GGUF or ONNX exports
- ✅ Still the best option if quality is the only criterion and a sidecar is acceptable

---

## Key Rust Crates

| Crate | Purpose | Maturity |
|-------|---------|----------|
| [`ort`](https://crates.io/crates/ort) | ONNX Runtime bindings | Production (v2.0.0-rc.12) |
| [`candle-core`](https://crates.io/crates/candle-core) | ML framework (csm.rs, Crane use this) | Production (v0.9.x) |
| [`llama-cpp-rs`](https://crates.io/crates/llama-cpp-2) | llama.cpp Rust bindings (GGUF inference) | Production |
| [`tokenizers`](https://crates.io/crates/tokenizers) | HuggingFace tokenizers | Production |
| [`kokoroxide`](https://crates.io/crates/kokoroxide) | Kokoro TTS via ONNX | Early |

---

## Implementation Roadmap

### Phase 1: TtsBackend trait + HTTP sidecar

- Define `TtsBackend` trait and `SynthesizeRequest`/`VoiceInfo` types
- Implement `HttpSidecar` backend (works with any OpenAI-compatible server)
- Preserve existing macOS `say` backend
- Validate with Fish Speech Python server or csm.rs server

### Phase 2: Embedded ONNX backend (Kitten Nano)

- Add `ort` dependency, implement `OnnxBackend`
- Ship Kitten TTS Nano as default fallback (25MB, no GPU)
- No voice cloning — pure synthesis only

### Phase 3: Embedded candle backend (Sesame CSM)

- Add `candle-core` dependency with Metal/CUDA features
- Implement `CandleBackend` wrapping csm.rs model loading
- Enable voice cloning and emotion control
- GGUF loading for CPU-only deployment

### Phase 4: Model selection and voice management

- Runtime model selection based on hardware detection
- Voice registry (cloned voices persisted to disk)
- Streaming synthesis support

---

## References

- Sesame CSM: https://github.com/SesameAILabs/csm
- csm.rs: https://github.com/cartesia-one/csm.rs
- Sesame CSM GGUF: https://huggingface.co/ggml-org/sesame-csm-1b-GGUF
- Qwen3 TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen3 TTS Paper: https://arxiv.org/abs/2601.15621
- qwen3-tts-rs (candle): https://github.com/TrevorS/qwen3-tts-rs
- qwen3-tts.cpp: https://github.com/predict-woo/qwen3-tts.cpp
- Crane (Rust TTS engine): https://github.com/lucasjinreal/Crane
- OuteTTS: https://github.com/edwko/OuteTTS
- OuteTTS ONNX: https://huggingface.co/onnx-community/OuteTTS-0.2-500M
- Orpheus TTS: https://github.com/canopyai/Orpheus-TTS
- Kitten TTS: https://github.com/KittenML/KittenTTS
- LuxTTS: https://github.com/ysharma3501/LuxTTS
- Fish Speech: https://github.com/fishaudio/fish-speech
- fish-speech.rs: https://github.com/EndlessReform/fish-speech.rs
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M
- Dia TTS: https://github.com/nari-labs/dia
- Parler-TTS: https://github.com/huggingface/parler-tts
- Piper: https://github.com/rhasspy/piper
- ort crate: https://ort.pyke.io/
- candle: https://github.com/huggingface/candle
- llama-cpp-2: https://crates.io/crates/llama-cpp-2
