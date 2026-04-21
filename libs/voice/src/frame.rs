use motlie_model::typed::{AudioBuf, ChannelLayout};

#[derive(Clone, Debug, PartialEq)]
pub struct PcmFrame<const RATE_HZ: u32, C: ChannelLayout, E> {
    audio: AudioBuf<E, RATE_HZ, C>,
}

impl<const RATE_HZ: u32, C: ChannelLayout, E> PcmFrame<RATE_HZ, C, E> {
    pub fn new(samples: Vec<E>) -> Self {
        Self {
            audio: AudioBuf::new(samples),
        }
    }

    pub fn from_audio(audio: AudioBuf<E, RATE_HZ, C>) -> Self {
        Self { audio }
    }

    pub fn audio(&self) -> &AudioBuf<E, RATE_HZ, C> {
        &self.audio
    }

    pub fn into_audio(self) -> AudioBuf<E, RATE_HZ, C> {
        self.audio
    }

    pub fn samples(&self) -> &[E] {
        self.audio.samples()
    }

    pub fn into_samples(self) -> Vec<E> {
        self.audio.into_samples()
    }
}

impl<const RATE_HZ: u32, C: ChannelLayout, E> From<AudioBuf<E, RATE_HZ, C>>
    for PcmFrame<RATE_HZ, C, E>
{
    fn from(audio: AudioBuf<E, RATE_HZ, C>) -> Self {
        Self::from_audio(audio)
    }
}

impl<const RATE_HZ: u32, C: ChannelLayout, E> From<PcmFrame<RATE_HZ, C, E>>
    for AudioBuf<E, RATE_HZ, C>
{
    fn from(frame: PcmFrame<RATE_HZ, C, E>) -> Self {
        frame.into_audio()
    }
}
