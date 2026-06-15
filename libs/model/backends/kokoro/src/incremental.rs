use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use motlie_model::typed::{
    IncrementalSpeechCancelToken, IncrementalSpeechChunk, IncrementalSpeechControls,
    IncrementalSpeechStream, IncrementalSpeechSummary, SynthesisRequest,
};
use motlie_model::{
    ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint, SpeechParams,
};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig};
use tokio::sync::mpsc;

const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
const OUTPUT_CHUNK_DURATION_MS: u32 = 40;
const NUM_THREADS: i32 = 2;
const DEFAULT_MAX_BUFFERED_AUDIO_MS: u32 = 80;

pub(crate) const INCREMENTAL_MODEL_FILE: &str = "model.onnx";
pub(crate) const INCREMENTAL_VOICES_FILE: &str = "voices.bin";
pub(crate) const INCREMENTAL_TOKENS_FILE: &str = "tokens.txt";
pub(crate) const INCREMENTAL_DATA_DIR: &str = "espeak-ng-data";

#[derive(Clone, Copy, Debug)]
pub(crate) struct KokoroIncrementalArtifactSpec<'a> {
    pub model: &'a str,
    pub voices: &'a str,
    pub tokens: &'a str,
    pub data_dir: Option<&'a str>,
    pub dict_dir: Option<&'a str>,
    pub lexicon: Option<&'a str>,
    pub lang: Option<&'a str>,
}

impl KokoroIncrementalArtifactSpec<'static> {
    pub(crate) fn kokoro_82m() -> Self {
        Self {
            model: INCREMENTAL_MODEL_FILE,
            voices: INCREMENTAL_VOICES_FILE,
            tokens: INCREMENTAL_TOKENS_FILE,
            data_dir: Some(INCREMENTAL_DATA_DIR),
            dict_dir: None,
            lexicon: None,
            lang: Some("en-us"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct KokoroIncrementalArtifactPaths {
    model: PathBuf,
    voices: PathBuf,
    tokens: PathBuf,
    data_dir: Option<PathBuf>,
    dict_dir: Option<PathBuf>,
    lexicon: Option<PathBuf>,
    lang: Option<String>,
}

pub(crate) struct KokoroIncrementalRuntime {
    tts: Mutex<OfflineTts>,
    sample_rate_hz: u32,
}

impl KokoroIncrementalRuntime {
    pub(crate) fn load(artifacts: &KokoroIncrementalArtifactPaths) -> Result<Self, ModelError> {
        let mut config = OfflineTtsConfig::default();
        config.model.kokoro = OfflineTtsKokoroModelConfig {
            model: Some(path_to_string(&artifacts.model)?),
            voices: Some(path_to_string(&artifacts.voices)?),
            tokens: Some(path_to_string(&artifacts.tokens)?),
            data_dir: optional_path_to_string(artifacts.data_dir.as_deref())?,
            length_scale: 1.0,
            dict_dir: optional_path_to_string(artifacts.dict_dir.as_deref())?,
            lexicon: optional_path_to_string(artifacts.lexicon.as_deref())?,
            lang: artifacts.lang.clone(),
        };
        config.model.num_threads = NUM_THREADS;
        config.model.provider = Some("cpu".to_owned());

        let tts = OfflineTts::create(&config).ok_or_else(|| ModelError::BackendInitialization {
            backend: "kokoro-incremental-tts",
            message: "failed to create Kokoro incremental TTS runtime".into(),
        })?;
        let sample_rate_hz =
            u32::try_from(tts.sample_rate()).map_err(|err| ModelError::BackendInitialization {
                backend: "kokoro-incremental-tts",
                message: format!("invalid Kokoro sample rate reported by runtime: {err}"),
            })?;
        if sample_rate_hz != KOKORO_SAMPLE_RATE_HZ {
            return Err(ModelError::InvalidConfiguration(format!(
                "Kokoro incremental sample rate {sample_rate_hz} does not match expected {KOKORO_SAMPLE_RATE_HZ}"
            )));
        }

        Ok(Self {
            tts: Mutex::new(tts),
            sample_rate_hz,
        })
    }

    pub(crate) async fn synthesize_incremental(
        self: Arc<Self>,
        request: SynthesisRequest,
        mut controls: IncrementalSpeechControls,
    ) -> Result<KokoroIncrementalSpeechStream, ModelError> {
        if controls.max_buffered_audio_ms == 0 {
            controls.max_buffered_audio_ms = DEFAULT_MAX_BUFFERED_AUDIO_MS;
        }
        let capacity = buffered_audio_capacity(controls.max_buffered_audio_ms);
        let (sender, receiver) = mpsc::channel(capacity);
        let worker = tokio::task::spawn_blocking(move || {
            self.synthesize_incremental_blocking(request, controls, sender)
        });

        Ok(KokoroIncrementalSpeechStream { receiver, worker })
    }

    fn synthesize_incremental_blocking(
        &self,
        request: SynthesisRequest,
        controls: IncrementalSpeechControls,
        sender: mpsc::Sender<Result<IncrementalSpeechChunk, ModelError>>,
    ) -> Result<IncrementalSpeechSummary, ModelError> {
        if controls.cancel.is_canceled() {
            return Ok(IncrementalSpeechSummary {
                canceled: true,
                ..IncrementalSpeechSummary::default()
            });
        }

        let text = validated_text(&request)?;
        let config = generation_config(&request.params)?;
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(
            self.sample_rate_hz,
            1,
        )));
        let callback_progress = Arc::clone(&progress);
        let callback_cancel = controls.cancel.clone();
        let callback_sender = sender;

        let tts = self
            .tts
            .lock()
            .map_err(|_| ModelError::Internal("kokoro incremental tts mutex poisoned".into()))?;
        let audio = tts.generate_with_config(
            text,
            &config,
            Some(move |samples: &[f32], progress_value: f32| -> bool {
                send_incremental_delta(
                    samples,
                    progress_value,
                    &callback_progress,
                    &callback_cancel,
                    &callback_sender,
                )
            }),
        );

        let worker_progress = progress
            .lock()
            .map_err(|_| {
                ModelError::Internal("kokoro incremental tts progress mutex poisoned".into())
            })?
            .clone();
        let canceled = controls.cancel.is_canceled();
        let synthesis_completed = audio.is_some() && !canceled;
        if audio.is_none() && !canceled {
            return Err(ModelError::BackendExecution {
                backend: "kokoro-incremental-tts",
                operation: "generate_with_config",
                message: "upstream TTS generation stopped before completion".into(),
            });
        }
        if synthesis_completed && worker_progress.chunks == 0 {
            return Err(ModelError::BackendExecution {
                backend: "kokoro-incremental-tts",
                operation: "generate_with_config",
                message: "generation completed without callback audio chunks".into(),
            });
        }

        Ok(IncrementalSpeechSummary {
            chunks: worker_progress.chunks,
            audio_ms: worker_progress.audio_ms,
            canceled,
            synthesis_completed,
        })
    }
}

pub struct KokoroIncrementalSpeechStream {
    receiver: mpsc::Receiver<Result<IncrementalSpeechChunk, ModelError>>,
    worker: tokio::task::JoinHandle<Result<IncrementalSpeechSummary, ModelError>>,
}

impl IncrementalSpeechStream for KokoroIncrementalSpeechStream {
    async fn next_audio_chunk(&mut self) -> Result<Option<IncrementalSpeechChunk>, ModelError> {
        match self.receiver.recv().await {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }
    }

    async fn finish(mut self) -> Result<IncrementalSpeechSummary, ModelError> {
        while let Some(event) = self.receiver.recv().await {
            event?;
        }
        join_blocking_tts_task(self.worker).await
    }
}

#[derive(Clone, Debug)]
struct IncrementalWorkerProgress {
    sample_rate_hz: u32,
    channels: u16,
    sent_samples: usize,
    chunks: u64,
    audio_ms: u64,
    last_progress: f32,
}

impl IncrementalWorkerProgress {
    fn new(sample_rate_hz: u32, channels: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            sent_samples: 0,
            chunks: 0,
            audio_ms: 0,
            last_progress: 0.0,
        }
    }
}

pub(crate) fn try_resolve_incremental_artifacts(
    checkpoint: &ResolvedCheckpoint,
    spec: KokoroIncrementalArtifactSpec<'_>,
) -> Result<Option<KokoroIncrementalArtifactPaths>, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "Kokoro incremental TTS expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .ok_or_else(|| {
                ModelError::InvalidConfiguration(format!(
                    "onnx checkpoint path `{}` has no parent directory",
                    checkpoint.path.display()
                ))
            })?
            .to_path_buf()
    };

    try_resolve_incremental_paths(&root, spec)
}

pub(crate) fn try_configure_incremental_artifact_policy(
    spec: KokoroIncrementalArtifactSpec<'_>,
    policy: ArtifactPolicy,
) -> Result<Option<KokoroIncrementalArtifactPaths>, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    try_resolve_incremental_paths(&root, spec)
}

fn try_resolve_incremental_paths(
    root: &Path,
    spec: KokoroIncrementalArtifactSpec<'_>,
) -> Result<Option<KokoroIncrementalArtifactPaths>, ModelError> {
    let Some(model) = find_file(root, spec.model)? else {
        return Ok(None);
    };
    let Some(voices) = find_file(root, spec.voices)? else {
        return Ok(None);
    };
    let Some(tokens) = find_file(root, spec.tokens)? else {
        return Ok(None);
    };
    let Some(data_dir) = optional_existing_path(root, spec.data_dir)? else {
        return Ok(None);
    };
    let dict_dir = optional_existing_path(root, spec.dict_dir)?;
    let lexicon = optional_existing_path(root, spec.lexicon)?;

    Ok(Some(KokoroIncrementalArtifactPaths {
        model,
        voices,
        tokens,
        data_dir: Some(data_dir),
        dict_dir,
        lexicon,
        lang: spec.lang.map(str::to_owned),
    }))
}

fn find_file(root: &Path, relative: &str) -> Result<Option<PathBuf>, ModelError> {
    let nested = root.join(relative);
    if nested.is_file() {
        return Ok(Some(nested));
    }

    let basename = Path::new(relative).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("invalid Kokoro artifact path `{relative}`"))
    })?;
    let direct = root.join(basename);
    if direct.is_file() {
        return Ok(Some(direct));
    }

    Ok(None)
}

fn optional_existing_path(
    root: &Path,
    relative: Option<&str>,
) -> Result<Option<PathBuf>, ModelError> {
    let Some(relative) = relative else {
        return Ok(None);
    };
    let path = root.join(relative);
    if path.exists() {
        return Ok(Some(path));
    }
    Ok(None)
}

fn send_incremental_delta(
    samples: &[f32],
    progress_value: f32,
    progress: &Arc<Mutex<IncrementalWorkerProgress>>,
    cancel: &IncrementalSpeechCancelToken,
    sender: &mpsc::Sender<Result<IncrementalSpeechChunk, ModelError>>,
) -> bool {
    if cancel.is_canceled() {
        return false;
    }

    let chunks = {
        let Ok(mut state) = progress.lock() else {
            let _ = sender.blocking_send(Err(ModelError::Internal(
                "kokoro incremental tts progress mutex poisoned".into(),
            )));
            return false;
        };
        state.last_progress = progress_value;
        if samples.len() <= state.sent_samples {
            return true;
        }

        let delta = f32_to_i16_samples(&samples[state.sent_samples..]);
        state.sent_samples = samples.len();
        let chunk_len = samples_per_output_chunk(state.sample_rate_hz, state.channels);
        let mut chunks = delta
            .chunks(chunk_len)
            .filter(|samples| !samples.is_empty())
            .map(|samples| {
                let chunk = IncrementalSpeechChunk {
                    samples_i16: samples.to_vec(),
                    sample_rate_hz: state.sample_rate_hz,
                    channels: state.channels,
                    chunk_index: state.chunks,
                    is_final: false,
                };
                state.chunks = state.chunks.saturating_add(1);
                state.audio_ms = state.audio_ms.saturating_add(chunk.audio_ms());
                chunk
            })
            .collect::<Vec<_>>();
        if progress_value >= 1.0 {
            if let Some(last) = chunks.last_mut() {
                last.is_final = true;
            }
        }
        chunks
    };

    for chunk in chunks {
        if sender.blocking_send(Ok(chunk)).is_err() || cancel.is_canceled() {
            return false;
        }
    }

    true
}

fn samples_per_output_chunk(sample_rate_hz: u32, channels: u16) -> usize {
    let frames = (sample_rate_hz as usize)
        .saturating_mul(OUTPUT_CHUNK_DURATION_MS as usize)
        .div_ceil(1000)
        .max(1);
    frames.saturating_mul(channels.max(1) as usize)
}

async fn join_blocking_tts_task<T>(
    handle: tokio::task::JoinHandle<Result<T, ModelError>>,
) -> Result<T, ModelError> {
    match handle.await {
        Ok(result) => result,
        Err(err) => Err(ModelError::BackendExecution {
            backend: "kokoro-incremental-tts",
            operation: "join_blocking_tts_task",
            message: err.to_string(),
        }),
    }
}

fn validated_text(request: &SynthesisRequest) -> Result<&str, ModelError> {
    let text = request.text.trim();
    if text.is_empty() {
        return Err(ModelError::InvalidConfiguration(
            "speech request requires non-empty text".into(),
        ));
    }
    Ok(text)
}

fn generation_config(params: &SpeechParams) -> Result<GenerationConfig, ModelError> {
    if params.seed.is_some() {
        return Err(ModelError::InvalidConfiguration(
            "Kokoro incremental backend does not support `SpeechParams.seed`".into(),
        ));
    }
    Ok(GenerationConfig {
        speed: synthesis_speed(params)?,
        ..GenerationConfig::default()
    })
}

fn synthesis_speed(params: &SpeechParams) -> Result<f32, ModelError> {
    match params.speaking_rate {
        Some(rate) if !rate.is_finite() || rate <= 0.0 => Err(ModelError::InvalidConfiguration(
            format!("speech speaking_rate must be positive and finite, got {rate}"),
        )),
        Some(rate) => Ok(rate),
        None => Ok(1.0),
    }
}

fn f32_to_i16_samples(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

fn buffered_audio_capacity(max_buffered_audio_ms: u32) -> usize {
    let chunk_ms = OUTPUT_CHUNK_DURATION_MS.max(1);
    let capacity = max_buffered_audio_ms.div_ceil(chunk_ms).max(1);
    capacity as usize
}

fn path_to_string(path: &Path) -> Result<String, ModelError> {
    path.to_str().map(ToOwned::to_owned).ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("non-UTF-8 path: {}", path.display()))
    })
}

fn optional_path_to_string(path: Option<&Path>) -> Result<Option<String>, ModelError> {
    path.map(path_to_string).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_audio_capacity_is_bounded_to_at_least_one_chunk() {
        assert_eq!(buffered_audio_capacity(0), 1);
        assert_eq!(buffered_audio_capacity(1), 1);
        assert_eq!(buffered_audio_capacity(40), 1);
        assert_eq!(buffered_audio_capacity(41), 2);
        assert_eq!(buffered_audio_capacity(80), 2);
    }

    #[test]
    fn delta_callback_sends_only_new_samples() {
        let (sender, mut receiver) = mpsc::channel(4);
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(24_000, 1)));
        let cancel = IncrementalSpeechCancelToken::new();

        assert!(send_incremental_delta(
            &[0.0, 0.25],
            0.2,
            &progress,
            &cancel,
            &sender,
        ));
        assert!(send_incremental_delta(
            &[0.0, 0.25, 0.5],
            1.0,
            &progress,
            &cancel,
            &sender,
        ));
        drop(sender);

        let first = receiver
            .blocking_recv()
            .expect("first event should exist")
            .expect("first event should be ok");
        let second = receiver
            .blocking_recv()
            .expect("second event should exist")
            .expect("second event should be ok");
        assert_eq!(first.samples_i16.len(), 2);
        assert!(!first.is_final);
        assert_eq!(second.samples_i16.len(), 1);
        assert!(second.is_final);
    }

    #[test]
    fn callback_delta_is_split_into_bounded_audio_chunks() {
        let (sender, mut receiver) = mpsc::channel(4);
        let progress = Arc::new(Mutex::new(IncrementalWorkerProgress::new(1_000, 1)));
        let cancel = IncrementalSpeechCancelToken::new();
        let samples = vec![0.0; 100];

        assert!(send_incremental_delta(
            &samples, 1.0, &progress, &cancel, &sender,
        ));
        drop(sender);

        let mut lengths = Vec::new();
        let mut finals = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            let chunk = event.expect("chunk event should be ok");
            lengths.push(chunk.samples_i16.len());
            finals.push(chunk.is_final);
        }

        assert_eq!(lengths, vec![40, 40, 20]);
        assert_eq!(finals, vec![false, false, true]);
    }

    #[test]
    fn invalid_speed_is_rejected() {
        let err = generation_config(&SpeechParams {
            speaking_rate: Some(0.0),
            seed: None,
        })
        .expect_err("zero speed should fail");

        assert!(matches!(
            err,
            ModelError::InvalidConfiguration(message) if message.contains("speaking_rate")
        ));
    }
}
