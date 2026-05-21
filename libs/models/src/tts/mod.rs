#[cfg(any(
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
))]
use std::fmt;
use std::str::FromStr;

#[cfg(any(
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
))]
use motlie_model::BundleId;

pub const PIPER_EN_US_LJSPEECH_MEDIUM_SELECTOR: &str = "piper/en_us_ljspeech_medium";
pub const QWEN3_TTS_CPP_0_6B_SELECTOR: &str = "qwen/qwen3_tts_cpp_0_6b";

#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
pub mod piper_en_us_ljspeech_medium;
#[cfg(feature = "model-qwen3-tts-cpp")]
pub mod qwen3_tts_cpp;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum TtsModels {
    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    PiperEnUsLjspeechMedium,
    #[cfg(feature = "model-qwen3-tts-cpp")]
    Qwen3TtsCpp0_6B,
}

#[cfg(any(
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
))]
impl TtsModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::SELECTOR,
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => qwen3_tts_cpp::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::descriptor().id,
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => qwen3_tts_cpp::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::descriptor(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => qwen3_tts_cpp::descriptor(),
        }
    }

    pub fn bundle(&self) -> crate::CuratedBundle {
        match self {
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => crate::CuratedBundle::PiperEnUsLjspeechMedium,
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => crate::CuratedBundle::Qwen3TtsCpp0_6B,
        }
    }
}

#[cfg(any(
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
))]
impl fmt::Display for TtsModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for TtsModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            piper_en_us_ljspeech_medium::SELECTOR => Ok(Self::PiperEnUsLjspeechMedium),
            #[cfg(not(feature = "model-piper-en-us-ljspeech-medium"))]
            PIPER_EN_US_LJSPEECH_MEDIUM_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            qwen3_tts_cpp::SELECTOR => Ok(Self::Qwen3TtsCpp0_6B),
            #[cfg(not(feature = "model-qwen3-tts-cpp"))]
            QWEN3_TTS_CPP_0_6B_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownTtsModel {
                selector: other.to_owned(),
            }),
        }
    }
}
