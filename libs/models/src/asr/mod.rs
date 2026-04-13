#[cfg(feature = "model-whisper-base-en")]
use std::fmt;
use std::str::FromStr;

#[cfg(feature = "model-whisper-base-en")]
use motlie_model::{BundleId, ModelBundle};

pub const WHISPER_BASE_EN_SELECTOR: &str = "openai/whisper_base_en";

#[cfg(feature = "model-whisper-base-en")]
pub mod whisper_base_en;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AsrModels {
    #[cfg(feature = "model-whisper-base-en")]
    WhisperBaseEn,
}

#[cfg(feature = "model-whisper-base-en")]
impl AsrModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WhisperBaseEn => whisper_base_en::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            Self::WhisperBaseEn => whisper_base_en::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            Self::WhisperBaseEn => whisper_base_en::descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            Self::WhisperBaseEn => whisper_base_en::bundle(),
        }
    }
}

#[cfg(feature = "model-whisper-base-en")]
impl fmt::Display for AsrModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for AsrModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-whisper-base-en")]
            whisper_base_en::SELECTOR => Ok(Self::WhisperBaseEn),
            #[cfg(not(feature = "model-whisper-base-en"))]
            WHISPER_BASE_EN_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownAsrModel {
                selector: other.to_owned(),
            }),
        }
    }
}
