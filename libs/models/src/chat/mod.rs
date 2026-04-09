#[cfg(feature = "model-qwen3-4b")]
use std::fmt;
use std::str::FromStr;

#[cfg(feature = "model-qwen3-4b")]
use motlie_model::{BundleId, ModelBundle};

pub const QWEN3_4B_SELECTOR: &str = "qwen/qwen3_4b";

#[cfg(feature = "model-qwen3-4b")]
pub mod qwen3_4b;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChatModels {
    #[cfg(feature = "model-qwen3-4b")]
    Qwen3_4B,
}

#[cfg(feature = "model-qwen3-4b")]
impl ChatModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Qwen3_4B => qwen3_4b::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            Self::Qwen3_4B => qwen3_4b::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            Self::Qwen3_4B => qwen3_4b::descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            Self::Qwen3_4B => qwen3_4b::bundle(),
        }
    }
}

#[cfg(feature = "model-qwen3-4b")]
impl fmt::Display for ChatModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for ChatModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-qwen3-4b")]
            qwen3_4b::SELECTOR => Ok(Self::Qwen3_4B),
            #[cfg(not(feature = "model-qwen3-4b"))]
            QWEN3_4B_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownChatModel {
                selector: other.to_owned(),
            }),
        }
    }
}
