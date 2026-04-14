#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
use std::fmt;
use std::str::FromStr;

#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
use motlie_model::{BundleId, ModelBundle};

pub const PIPER_EN_US_LJSPEECH_MEDIUM_SELECTOR: &str = "piper/en_us_ljspeech_medium";

#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
pub mod piper_en_us_ljspeech_medium;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TtsModels {
    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    PiperEnUsLjspeechMedium,
}

#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
impl TtsModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            Self::PiperEnUsLjspeechMedium => piper_en_us_ljspeech_medium::bundle(),
        }
    }
}

#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
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
            other => Err(crate::ModelsError::UnknownTtsModel {
                selector: other.to_owned(),
            }),
        }
    }
}
