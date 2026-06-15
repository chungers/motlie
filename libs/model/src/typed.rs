use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{ModelError, SpeechParams, TranscriptionParams, TranscriptionUpdate};

pub trait ChannelLayout: Send + Sync + 'static {
    const COUNT: u16;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mono;

impl ChannelLayout for Mono {
    const COUNT: u16 = 1;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stereo;

impl ChannelLayout for Stereo {
    const COUNT: u16 = 2;
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuf<S, const RATE_HZ: u32, C: ChannelLayout> {
    samples: Vec<S>,
    _channels: PhantomData<C>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynthesisRequest {
    pub text: String,
    pub params: SpeechParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloneReference<const RATE_HZ: u32, C: ChannelLayout> {
    pub audio: AudioBuf<f32, RATE_HZ, C>,
    pub transcript: Option<String>,
}

impl<S, const RATE_HZ: u32, C: ChannelLayout> AudioBuf<S, RATE_HZ, C> {
    pub fn new(samples: Vec<S>) -> Self {
        Self {
            samples,
            _channels: PhantomData,
        }
    }

    pub fn samples(&self) -> &[S] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<S> {
        self.samples
    }

    pub fn sample_rate_hz(&self) -> u32 {
        RATE_HZ
    }

    pub fn channels(&self) -> u16 {
        C::COUNT
    }

    pub fn frame_count(&self) -> usize {
        let channels = C::COUNT.max(1) as usize;
        self.samples.len() / channels
    }
}

pub trait AudioTransform {
    type Input;
    type Output;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, ModelError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Compose<A, B> {
    first: A,
    second: B,
}

impl<A, B> Compose<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> AudioTransform for Compose<A, B>
where
    A: AudioTransform,
    B: AudioTransform<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let mid = self.first.transform(input)?;
        self.second.transform(mid)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityTransform<T>(PhantomData<T>);

impl<T> IdentityTransform<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> AudioTransform for IdentityTransform<T> {
    type Input = T;
    type Output = T;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        Ok(input)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I16ToF32<const RATE_HZ: u32, C: ChannelLayout>(PhantomData<C>);

impl<const RATE_HZ: u32, C: ChannelLayout> I16ToF32<RATE_HZ, C> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const RATE_HZ: u32, C: ChannelLayout> AudioTransform for I16ToF32<RATE_HZ, C> {
    type Input = AudioBuf<i16, RATE_HZ, C>;
    type Output = AudioBuf<f32, RATE_HZ, C>;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        Ok(AudioBuf::new(
            input
                .into_samples()
                .into_iter()
                .map(|sample| sample as f32 / 32768.0)
                .collect(),
        ))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I16MonoResampler<const IN_RATE_HZ: u32, const OUT_RATE_HZ: u32>;

impl<const IN_RATE_HZ: u32, const OUT_RATE_HZ: u32> AudioTransform
    for I16MonoResampler<IN_RATE_HZ, OUT_RATE_HZ>
{
    type Input = AudioBuf<i16, IN_RATE_HZ, Mono>;
    type Output = AudioBuf<i16, OUT_RATE_HZ, Mono>;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        if IN_RATE_HZ == OUT_RATE_HZ {
            return Ok(AudioBuf::new(input.into_samples()));
        }

        let input = input.into_samples();
        if input.is_empty() {
            return Ok(AudioBuf::new(Vec::new()));
        }

        let ratio = IN_RATE_HZ as f64 / OUT_RATE_HZ as f64;
        let out_len =
            ((input.len() as f64) * OUT_RATE_HZ as f64 / IN_RATE_HZ as f64).ceil() as usize;
        let mut output = Vec::with_capacity(out_len.max(1));
        let max_index = input.len().saturating_sub(1);

        for out_idx in 0..out_len {
            let src_pos = out_idx as f64 * ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - idx as f64;
            let left = input[idx.min(max_index)] as f64;
            let right = input[(idx + 1).min(max_index)] as f64;
            let value = left * (1.0 - frac) + right * frac;
            output.push(value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
        }

        Ok(AudioBuf::new(output))
    }
}

pub trait BatchTranscriber: Send + Sync {
    type Input;

    fn transcribe(
        &self,
        audio: Self::Input,
        params: TranscriptionParams,
    ) -> impl Future<Output = Result<TranscriptionUpdate, ModelError>> + Send;
}

pub trait StreamingTranscriber: Send + Sync {
    type Input;
    type Session: TranscriptionSession<Input = Self::Input>;

    fn open_session(
        &self,
        params: TranscriptionParams,
    ) -> impl Future<Output = Result<Self::Session, ModelError>> + Send;
}

pub trait TranscriptionSession: Send {
    type Input;

    fn ingest(
        &mut self,
        audio: Self::Input,
    ) -> impl Future<Output = Result<Option<TranscriptionUpdate>, ModelError>> + Send;

    fn finish(self) -> impl Future<Output = Result<TranscriptionUpdate, ModelError>> + Send;
}

pub trait BufferedSpeechSynthesizer: Send + Sync {
    type Request;
    type Output;

    fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> impl Future<Output = Result<Self::Output, ModelError>> + Send;
}

pub trait SpeechSynthesizer: Send + Sync {
    type Request;
    type Output;
    type Stream: SpeechStream<Chunk = Self::Output>;

    fn synthesize(
        &self,
        request: Self::Request,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}

pub trait BufferedVoiceCloneSynthesizer<const RATE_HZ: u32, C: ChannelLayout>: Send + Sync {
    type Request;
    type Output;

    fn synthesize_with_reference_buffered(
        &self,
        request: Self::Request,
        reference: CloneReference<RATE_HZ, C>,
    ) -> impl Future<Output = Result<Self::Output, ModelError>> + Send;
}

pub trait VoiceCloneSynthesizer<const RATE_HZ: u32, C: ChannelLayout>: Send + Sync {
    type Request;
    type Output;
    type Stream: SpeechStream<Chunk = Self::Output>;

    fn synthesize_with_reference(
        &self,
        request: Self::Request,
        reference: CloneReference<RATE_HZ, C>,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}

pub trait SpeechStream: Send {
    type Chunk;

    fn next_chunk(
        &mut self,
    ) -> impl Future<Output = Result<Option<Self::Chunk>, ModelError>> + Send;
    fn finish(self) -> impl Future<Output = Result<(), ModelError>> + Send;
}

#[derive(Clone, Debug, Default)]
pub struct IncrementalSpeechCancelToken {
    canceled: Arc<AtomicBool>,
}

impl IncrementalSpeechCancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncrementalSpeechRequestLabel(String);

impl IncrementalSpeechRequestLabel {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct IncrementalSpeechControls {
    pub cancel: IncrementalSpeechCancelToken,
    pub request_label: Option<IncrementalSpeechRequestLabel>,
    pub max_buffered_audio_ms: u32,
}

impl Default for IncrementalSpeechControls {
    fn default() -> Self {
        Self {
            cancel: IncrementalSpeechCancelToken::new(),
            request_label: None,
            max_buffered_audio_ms: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncrementalSpeechChunk {
    pub samples_i16: Vec<i16>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub chunk_index: u64,
    pub is_final: bool,
}

impl IncrementalSpeechChunk {
    pub fn audio_ms(&self) -> u64 {
        if self.sample_rate_hz == 0 || self.channels == 0 {
            return 0;
        }
        let frames = self.samples_i16.len() as u64 / self.channels as u64;
        frames.saturating_mul(1000) / self.sample_rate_hz as u64
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IncrementalSpeechSummary {
    pub chunks: u64,
    pub audio_ms: u64,
    pub canceled: bool,
    pub synthesis_completed: bool,
}

pub trait IncrementalSpeechSynthesizer: Send + Sync {
    type Request;
    type Stream: IncrementalSpeechStream;

    fn synthesize_incremental(
        &self,
        request: Self::Request,
        controls: IncrementalSpeechControls,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}

pub trait IncrementalSpeechStream: Send {
    fn next_audio_chunk(
        &mut self,
    ) -> impl Future<Output = Result<Option<IncrementalSpeechChunk>, ModelError>> + Send;

    fn finish(self) -> impl Future<Output = Result<IncrementalSpeechSummary, ModelError>> + Send;
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferedSpeechChunkStream<S, const RATE_HZ: u32, C: ChannelLayout> {
    audio: AudioBuf<S, RATE_HZ, C>,
    next_offset: usize,
    chunk_len_samples: usize,
}

impl<S, const RATE_HZ: u32, C: ChannelLayout> BufferedSpeechChunkStream<S, RATE_HZ, C> {
    pub fn new(audio: AudioBuf<S, RATE_HZ, C>, chunk_duration_ms: u32) -> Self {
        let channels = C::COUNT.max(1) as usize;
        let frames_per_chunk = (((RATE_HZ as u64) * (chunk_duration_ms as u64)) / 1000) as usize;
        let chunk_len_samples = frames_per_chunk.max(1) * channels;

        Self {
            audio,
            next_offset: 0,
            chunk_len_samples,
        }
    }
}

impl<S, const RATE_HZ: u32, C: ChannelLayout> SpeechStream
    for BufferedSpeechChunkStream<S, RATE_HZ, C>
where
    S: Clone + Send + Sync + 'static,
{
    type Chunk = AudioBuf<S, RATE_HZ, C>;

    async fn next_chunk(&mut self) -> Result<Option<Self::Chunk>, ModelError> {
        if self.next_offset >= self.audio.samples().len() {
            return Ok(None);
        }

        let end = (self.next_offset + self.chunk_len_samples).min(self.audio.samples().len());
        let chunk = AudioBuf::new(self.audio.samples()[self.next_offset..end].to_vec());
        self.next_offset = end;

        Ok(Some(chunk))
    }

    async fn finish(self) -> Result<(), ModelError> {
        Ok(())
    }
}

pub async fn stream_speech_into_asr<Tts, Xform, Asr>(
    tts: &Tts,
    request: Tts::Request,
    transform: &Xform,
    asr: &Asr,
    params: TranscriptionParams,
) -> Result<TranscriptionUpdate, ModelError>
where
    Tts: SpeechSynthesizer,
    Xform: AudioTransform<Input = Tts::Output, Output = Asr::Input>,
    Asr: StreamingTranscriber,
{
    let mut stream = tts.synthesize(request).await?;
    let mut session = asr.open_session(params).await?;

    while let Some(chunk) = stream.next_chunk().await? {
        let normalized = transform.transform(chunk)?;
        let _ = session.ingest(normalized).await?;
    }

    stream.finish().await?;
    session.finish().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_buf_tracks_compile_time_layout() {
        let audio = AudioBuf::<i16, 16_000, Mono>::new(vec![1, 2, 3, 4]);
        assert_eq!(audio.sample_rate_hz(), 16_000);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.frame_count(), 4);
    }

    #[test]
    fn i16_to_f32_normalizes_samples() {
        let audio = AudioBuf::<i16, 16_000, Mono>::new(vec![i16::MIN, 0, i16::MAX]);
        let converted = I16ToF32::<16_000, Mono>::new()
            .transform(audio)
            .expect("conversion should succeed");
        let samples = converted.samples();
        assert!((samples[0] + 1.0).abs() < 0.01);
        assert_eq!(samples[1], 0.0);
        assert!(samples[2] > 0.99);
    }

    #[test]
    fn mono_resampler_changes_frame_count() {
        let audio = AudioBuf::<i16, 22_050, Mono>::new((0..2205).map(|v| v as i16).collect());
        let output = I16MonoResampler::<22_050, 16_000>
            .transform(audio)
            .expect("resample should succeed");
        assert!((output.frame_count() as isize - 1600).abs() <= 1);
    }

    #[test]
    fn synthesis_request_defaults_to_empty_text_and_no_optional_controls() {
        let request = SynthesisRequest::default();

        assert!(request.text.is_empty());
        assert_eq!(request.params, SpeechParams::default());
    }

    #[test]
    fn clone_reference_preserves_typed_audio_and_transcript() {
        let reference = CloneReference::<16_000, Mono> {
            audio: AudioBuf::new(vec![0.0, 0.5, -0.25]),
            transcript: Some("hello".into()),
        };

        assert_eq!(reference.audio.sample_rate_hz(), 16_000);
        assert_eq!(reference.audio.channels(), 1);
        assert_eq!(reference.transcript.as_deref(), Some("hello"));
    }
}
