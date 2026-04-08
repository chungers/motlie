use std::fmt;
use std::str::FromStr;

use motlie_model::{BundleId, ModelBundle};

pub const QWEN3_4B_SELECTOR: &str = "qwen/qwen3_4b";

#[cfg(feature = "model-qwen3-4b")]
pub mod qwen3_4b;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatModels {
    #[cfg(feature = "model-qwen3-4b")]
    Qwen3_4B,
    #[doc(hidden)]
    #[cfg(not(feature = "model-qwen3-4b"))]
    __NoModels,
}

impl ChatModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::SELECTOR,
            #[cfg(not(feature = "model-qwen3-4b"))]
            Self::__NoModels => unreachable!("no chat models are enabled in this build"),
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::descriptor().id,
            #[cfg(not(feature = "model-qwen3-4b"))]
            Self::__NoModels => unreachable!("no chat models are enabled in this build"),
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::descriptor(),
            #[cfg(not(feature = "model-qwen3-4b"))]
            Self::__NoModels => unreachable!("no chat models are enabled in this build"),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => qwen3_4b::bundle(),
            #[cfg(not(feature = "model-qwen3-4b"))]
            Self::__NoModels => unreachable!("no chat models are enabled in this build"),
        }
    }
}

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
