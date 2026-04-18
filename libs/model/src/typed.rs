use std::future::Future;
use std::marker::PhantomData;

use crate::{ModelError, SpeechRequest, TranscriptionParams, TranscriptionUpdate};

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

pub trait SpeechSynthesizer: Send + Sync {
    type Output;
    type Stream: SpeechStream<Chunk = Self::Output>;

    fn synthesize(
        &self,
        request: SpeechRequest,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}

pub trait SpeechStream: Send {
    type Chunk;

    fn next_chunk(
        &mut self,
    ) -> impl Future<Output = Result<Option<Self::Chunk>, ModelError>> + Send;
    fn finish(self) -> impl Future<Output = Result<(), ModelError>> + Send;
}

pub async fn stream_speech_into_asr<Tts, Xform, Asr>(
    tts: &Tts,
    request: SpeechRequest,
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
}
