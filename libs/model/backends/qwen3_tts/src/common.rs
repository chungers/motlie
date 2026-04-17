use std::collections::HashMap;
use std::path::{Path, PathBuf};

use motlie_model::{
    ArtifactPolicy, AudioSpec, CheckpointFormat, ModelError, PcmEncoding, ResolvedCheckpoint,
};
use serde::Deserialize;

pub(crate) use motlie_model::metrics_runtime::{
    RuntimeMetricState, lock_metrics, observe_latency, observe_memory,
};

/// Paths to the three ONNX model components, config, and vocabulary.
#[derive(Clone, Debug)]
pub(crate) struct Qwen3TtsArtifactPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub vocoder: PathBuf,
    pub config: PathBuf,
    pub vocab: PathBuf,
}

/// Resolve ONNX artifacts from a checkpoint root directory.
pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
) -> Result<Qwen3TtsArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "qwen3-tts expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    };

    let paths = build_artifact_paths(&root);
    validate_artifacts(&paths)?;
    Ok(paths)
}

/// Resolve artifacts from an artifact policy.
pub(crate) fn configure_artifact_policy(
    policy: ArtifactPolicy,
) -> Result<Qwen3TtsArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    let paths = build_artifact_paths(&root);
    validate_artifacts(&paths)?;
    Ok(paths)
}

fn build_artifact_paths(root: &Path) -> Qwen3TtsArtifactPaths {
    Qwen3TtsArtifactPaths {
        encoder: root.join("encoder.onnx"),
        decoder: root.join("decoder.onnx"),
        vocoder: root.join("vocoder.onnx"),
        config: root.join("config.json"),
        vocab: root.join("vocab.json"),
    }
}

fn validate_artifacts(paths: &Qwen3TtsArtifactPaths) -> Result<(), ModelError> {
    for (label, path) in [
        ("encoder model", &paths.encoder),
        ("decoder model", &paths.decoder),
        ("vocoder model", &paths.vocoder),
        ("model config", &paths.config),
        ("vocabulary", &paths.vocab),
    ] {
        if !path.is_file() {
            return Err(ModelError::InvalidConfiguration(format!(
                "qwen3-tts {label} `{}` does not exist",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Model configuration parsed from config.json.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Qwen3TtsConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_hop_length")]
    pub hop_length: u32,
    #[serde(default = "default_mel_channels")]
    pub mel_channels: u32,
    #[serde(default = "default_fft_size")]
    pub fft_size: usize,
}

fn default_sample_rate() -> u32 {
    // CosyVoice2 / Qwen3-TTS canonical sample rate.
    24_000
}

fn default_hop_length() -> u32 {
    256
}

fn default_mel_channels() -> u32 {
    80
}

fn default_fft_size() -> usize {
    1024
}

impl Qwen3TtsConfig {
    pub(crate) fn from_path(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to open qwen3-tts config `{}`: {err}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_reader(file).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to parse qwen3-tts config `{}`: {err}",
                path.display()
            ))
        })?;

        if config.sample_rate == 0 {
            return Err(ModelError::InvalidConfiguration(format!(
                "qwen3-tts config `{}` declares sample_rate = 0",
                path.display()
            )));
        }

        Ok(config)
    }

    pub(crate) fn audio_spec(&self) -> AudioSpec {
        AudioSpec {
            sample_rate_hz: self.sample_rate,
            channels: 1,
            encoding: PcmEncoding::F32Le,
            preferred_chunk_bytes: 0,
        }
    }
}

/// Vocabulary loaded from vocab.json: maps tokens to integer IDs.
#[derive(Clone, Debug)]
pub(crate) struct Vocabulary {
    token_to_id: HashMap<String, i64>,
    unk_id: i64,
    bos_id: i64,
    eos_id: i64,
    /// Longest token in the vocabulary (in chars), used as the greedy match window.
    max_token_len: usize,
}

impl Vocabulary {
    pub(crate) fn from_path(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to open qwen3-tts vocabulary `{}`: {err}",
                path.display()
            ))
        })?;
        let raw: HashMap<String, i64> = serde_json::from_reader(file).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to parse qwen3-tts vocabulary `{}`: {err}",
                path.display()
            ))
        })?;

        let unk_id = raw.get("<unk>").copied().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "qwen3-tts vocabulary `{}` is missing `<unk>` token",
                path.display()
            ))
        })?;
        let bos_id = raw.get("<bos>").copied().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "qwen3-tts vocabulary `{}` is missing `<bos>` token",
                path.display()
            ))
        })?;
        let eos_id = raw.get("<eos>").copied().ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "qwen3-tts vocabulary `{}` is missing `<eos>` token",
                path.display()
            ))
        })?;

        let max_token_len = raw
            .keys()
            .filter(|k| !k.starts_with('<'))
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(1);

        Ok(Self {
            token_to_id: raw,
            unk_id,
            bos_id,
            eos_id,
            max_token_len,
        })
    }

    /// Build a vocabulary from raw entries. Test-only.
    #[cfg(test)]
    pub(crate) fn from_entries(entries: &[(&str, i64)]) -> Result<Self, ModelError> {
        let mut map = HashMap::new();
        for &(token, id) in entries {
            map.insert(token.to_string(), id);
        }
        let unk_id = map
            .get("<unk>")
            .copied()
            .ok_or_else(|| ModelError::InvalidConfiguration("test vocab missing <unk>".into()))?;
        let bos_id = map
            .get("<bos>")
            .copied()
            .ok_or_else(|| ModelError::InvalidConfiguration("test vocab missing <bos>".into()))?;
        let eos_id = map
            .get("<eos>")
            .copied()
            .ok_or_else(|| ModelError::InvalidConfiguration("test vocab missing <eos>".into()))?;
        let max_token_len = map
            .keys()
            .filter(|k| !k.starts_with('<'))
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(1);
        Ok(Self {
            token_to_id: map,
            unk_id,
            bos_id,
            eos_id,
            max_token_len,
        })
    }

    /// Tokenize text using greedy longest-match against the vocabulary.
    ///
    /// Scans the input left-to-right, greedily matching the longest vocabulary
    /// token at each position (up to `max_token_len` characters). Falls back to
    /// single-character `<unk>` for unmatched positions. Result is wrapped with
    /// BOS/EOS.
    ///
    /// This supports both character-level and subword vocabularies. When
    /// `vocab.json` contains multi-character subword entries (e.g., from the
    /// official `Qwen3-TTS-Tokenizer-12Hz` SentencePiece export), the greedy
    /// match correctly emits subword token IDs. When it only contains single
    /// characters, behavior is equivalent to per-character lookup.
    pub(crate) fn tokenize(&self, text: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        ids.push(self.bos_id);

        let chars: Vec<char> = text.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = self.unk_id;

            // Try longest match first, shrinking to single character.
            let max_end = (pos + self.max_token_len).min(chars.len());
            for end in (pos + 1..=max_end).rev() {
                let candidate: String = chars[pos..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    best_len = end - pos;
                    best_id = id;
                    break;
                }
            }

            ids.push(best_id);
            pos += best_len.max(1); // advance at least one character on unk
        }

        ids.push(self.eos_id);
        ids
    }
}

/// Compute a log-mel spectrogram from mono f32 audio samples.
///
/// Uses a Hann-windowed DFT with the given FFT size, hop length, and mel
/// channel count. This is a correct but not performance-optimized
/// implementation suitable for short reference audio (3-5 seconds).
pub(crate) fn compute_log_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_length: usize,
    mel_channels: usize,
) -> Vec<f32> {
    if samples.is_empty() || fft_size == 0 || hop_length == 0 || mel_channels == 0 {
        return Vec::new();
    }

    let num_frames = (samples.len().saturating_sub(fft_size)) / hop_length + 1;
    let num_bins = fft_size / 2 + 1;

    // Precompute Hann window.
    let window: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos()))
        .collect();

    // Precompute mel filterbank (triangular filters on mel scale).
    let mel_filterbank = build_mel_filterbank(sample_rate, fft_size, num_bins, mel_channels);

    let mut mel_spec = Vec::with_capacity(num_frames * mel_channels);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        // Apply window and compute power spectrum via DFT.
        let power = compute_power_spectrum(samples, start, &window, fft_size, num_bins);

        // Apply mel filterbank and take log.
        for filter in &mel_filterbank {
            let energy: f32 = power.iter().zip(filter.iter()).map(|(p, f)| p * f).sum();
            mel_spec.push((energy + 1e-10).ln());
        }
    }

    mel_spec
}

/// Compute the power spectrum of a windowed frame using naive DFT.
fn compute_power_spectrum(
    samples: &[f32],
    start: usize,
    window: &[f32],
    fft_size: usize,
    num_bins: usize,
) -> Vec<f32> {
    (0..num_bins)
        .map(|k| {
            let mut real = 0.0_f64;
            let mut imag = 0.0_f64;
            for (n, &w) in window.iter().enumerate() {
                let sample_idx = start + n;
                let sample = if sample_idx < samples.len() {
                    samples[sample_idx] as f64 * w as f64
                } else {
                    0.0
                };
                let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / fft_size as f64;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            (real * real + imag * imag) as f32
        })
        .collect()
}

/// Build triangular mel filterbank.
fn build_mel_filterbank(
    sample_rate: u32,
    _fft_size: usize,
    num_bins: usize,
    mel_channels: usize,
) -> Vec<Vec<f32>> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(f_max);
    let mel_min = hz_to_mel(0.0);

    // Uniformly spaced mel points.
    let mel_points: Vec<f32> = (0..mel_channels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (mel_channels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            ((hz / f_max) * (num_bins - 1) as f32)
                .round()
                .clamp(0.0, (num_bins - 1) as f32) as usize
        })
        .collect();

    (0..mel_channels)
        .map(|i| {
            let mut filter = vec![0.0_f32; num_bins];
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            // Rising slope: left → center
            if center > left {
                for (j, val) in filter[left..center].iter_mut().enumerate() {
                    *val = j as f32 / (center - left) as f32;
                }
            }
            // Center peak: always 1.0 (prevents zero-energy on degenerate filters
            // where center == left or center == right).
            if center < num_bins {
                filter[center] = 1.0;
            }
            // Falling slope: center → right
            if right > center {
                for (j, val) in filter[(center + 1)..right].iter_mut().enumerate() {
                    *val = (right - center - 1 - j) as f32 / (right - center) as f32;
                }
            }
            filter
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Resample mono audio from `src_rate` to `dst_rate` using linear interpolation.
pub(crate) fn resample_mono(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };
        output.push(sample as f32);
    }
    output
}

/// Downmix multi-channel audio to mono by averaging channels.
pub(crate) fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Decode raw PCM bytes into f32 samples.
pub(crate) fn decode_pcm_to_f32(pcm: &[u8], encoding: PcmEncoding) -> Result<Vec<f32>, ModelError> {
    match encoding {
        PcmEncoding::S16Le => {
            if !pcm.len().is_multiple_of(2) {
                return Err(ModelError::InvalidConfiguration(
                    "S16Le reference audio length must be even".into(),
                ));
            }
            Ok(pcm
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect())
        }
        PcmEncoding::F32Le => {
            if !pcm.len().is_multiple_of(4) {
                return Err(ModelError::InvalidConfiguration(
                    "F32Le reference audio length must be a multiple of 4".into(),
                ));
            }
            Ok(pcm
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
    }
}

/// Encode f32 audio samples to PCM bytes.
pub(crate) fn encode_pcm(samples: &[f32], encoding: PcmEncoding) -> Vec<u8> {
    match encoding {
        PcmEncoding::S16Le => {
            let mut out = Vec::with_capacity(samples.len() * 2);
            for sample in samples {
                let clamped = sample.clamp(-1.0, 1.0);
                let as_i16 = (clamped * i16::MAX as f32) as i16;
                out.extend_from_slice(&as_i16.to_le_bytes());
            }
            out
        }
        PcmEncoding::F32Le => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for sample in samples {
                out.extend_from_slice(&sample.to_le_bytes());
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_missing_components() {
        let root = std::env::temp_dir().join("motlie-qwen3-tts-missing");
        std::fs::create_dir_all(&root).ok();

        let paths = build_artifact_paths(&root);
        let err = validate_artifacts(&paths).expect_err("missing components should fail");

        assert!(
            matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("does not exist"))
        );
    }

    #[test]
    fn resolve_rejects_wrong_checkpoint_format() {
        let checkpoint = ResolvedCheckpoint {
            checkpoint: motlie_model::ModelCheckpoint {
                format: CheckpointFormat::Gguf,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                },
                include: vec![],
                quantization: None,
            },
            path: PathBuf::from("/tmp/fake"),
        };

        let err = resolve_onnx_artifacts(&checkpoint).expect_err("wrong format should fail");
        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("Onnx")));
    }

    #[test]
    fn tokenizer_longest_match_prefers_longer_tokens() {
        let vocab = Vocabulary::from_entries(&[
            ("<unk>", 0),
            ("<bos>", 1),
            ("<eos>", 2),
            ("a", 10),
            ("ab", 11),
            ("abc", 12),
            ("d", 13),
        ])
        .expect("test vocab");

        let ids = vocab.tokenize("abcd");
        // Expected: BOS, "abc"=12, "d"=13, EOS
        assert_eq!(ids, vec![1, 12, 13, 2]);
    }

    #[test]
    fn tokenizer_falls_back_to_unk_and_advances() {
        let vocab = Vocabulary::from_entries(&[
            ("<unk>", 0),
            ("<bos>", 1),
            ("<eos>", 2),
            ("a", 10),
            ("c", 12),
        ])
        .expect("test vocab");

        let ids = vocab.tokenize("abc");
        // "a"=10, "b"=unk(0), "c"=12
        assert_eq!(ids, vec![1, 10, 0, 12, 2]);
    }

    #[test]
    fn tokenizer_handles_single_char_vocab() {
        let vocab = Vocabulary::from_entries(&[
            ("<unk>", 0),
            ("<bos>", 1),
            ("<eos>", 2),
            ("h", 10),
            ("i", 11),
        ])
        .expect("test vocab");

        let ids = vocab.tokenize("hi");
        assert_eq!(ids, vec![1, 10, 11, 2]);
    }

    #[test]
    fn tokenizer_empty_text_produces_only_bos_eos() {
        let vocab = Vocabulary::from_entries(&[("<unk>", 0), ("<bos>", 1), ("<eos>", 2)])
            .expect("test vocab");

        let ids = vocab.tokenize("");
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn tokenizer_all_unknown_advances_one_char_at_a_time() {
        let vocab = Vocabulary::from_entries(&[("<unk>", 0), ("<bos>", 1), ("<eos>", 2)])
            .expect("test vocab");

        let ids = vocab.tokenize("xyz");
        // Each char is unknown, advances by 1
        assert_eq!(ids, vec![1, 0, 0, 0, 2]);
    }
}
