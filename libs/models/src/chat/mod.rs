#[cfg(any(
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
))]
use std::fmt;
use std::str::FromStr;

#[cfg(any(
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
))]
use motlie_model::{BundleId, ModelBundle};

pub const QWEN3_4B_SELECTOR: &str = "qwen/qwen3_4b";
pub const GEMMA4_E2B_SELECTOR: &str = "google/gemma4_e2b";
pub const QWEN3_4B_GGUF_SELECTOR: &str = "qwen/qwen3_4b_gguf";
pub const GEMMA4_E2B_GGUF_SELECTOR: &str = "google/gemma4_e2b_gguf";

#[cfg(feature = "model-gemma4-e2b")]
pub mod gemma4_e2b;
#[cfg(feature = "model-gemma4-e2b-gguf")]
pub mod gemma4_e2b_gguf;
#[cfg(feature = "model-qwen3-4b")]
pub mod qwen3_4b;
#[cfg(feature = "model-qwen3-4b-gguf")]
pub mod qwen3_4b_gguf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum ChatModels {
    #[cfg(feature = "model-gemma4-e2b")]
    Gemma4E2B,
    #[cfg(feature = "model-gemma4-e2b-gguf")]
    Gemma4E2B_Gguf,
    #[cfg(feature = "model-qwen3-4b")]
    Qwen3_4B,
    #[cfg(feature = "model-qwen3-4b-gguf")]
    Qwen3_4B_Gguf,
}

#[cfg(any(
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
))]
impl ChatModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => gemma4_e2b::SELECTOR,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => gemma4_e2b_gguf::SELECTOR,
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::SELECTOR,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => qwen3_4b_gguf::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => gemma4_e2b::descriptor().id,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => gemma4_e2b_gguf::descriptor().id,
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::descriptor().id,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => qwen3_4b_gguf::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => gemma4_e2b::descriptor(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => gemma4_e2b_gguf::descriptor(),
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::descriptor(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => qwen3_4b_gguf::descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => gemma4_e2b::bundle(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => gemma4_e2b_gguf::bundle(),
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::bundle(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => qwen3_4b_gguf::bundle(),
        }
    }
}

#[cfg(any(
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
))]
impl fmt::Display for ChatModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for ChatModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-gemma4-e2b")]
            gemma4_e2b::SELECTOR => Ok(Self::Gemma4E2B),
            #[cfg(not(feature = "model-gemma4-e2b"))]
            GEMMA4_E2B_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            gemma4_e2b_gguf::SELECTOR => Ok(Self::Gemma4E2B_Gguf),
            #[cfg(not(feature = "model-gemma4-e2b-gguf"))]
            GEMMA4_E2B_GGUF_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(feature = "model-qwen3-4b")]
            qwen3_4b::SELECTOR => Ok(Self::Qwen3_4B),
            #[cfg(not(feature = "model-qwen3-4b"))]
            QWEN3_4B_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            qwen3_4b_gguf::SELECTOR => Ok(Self::Qwen3_4B_Gguf),
            #[cfg(not(feature = "model-qwen3-4b-gguf"))]
            QWEN3_4B_GGUF_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownChatModel {
                selector: other.to_owned(),
            }),
        }
    }
}
