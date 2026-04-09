use std::fmt;
use std::str::FromStr;

use motlie_model::{BundleId, EmbeddingSpec, ModelBundle};

pub const GOOGLE_GEMMA_300M_SELECTOR: &str = "google/embeddinggemma_300m";

#[cfg(feature = "model-google-gemma-300m")]
pub mod google_gemma_300m;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EmbeddingModels {
    #[cfg(feature = "model-google-gemma-300m")]
    GoogleGemma300m,
}

impl EmbeddingModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::SELECTOR,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::descriptor().id,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::descriptor(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::bundle(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn embedding_spec(&self) -> &'static EmbeddingSpec {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::embedding_spec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for EmbeddingModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for EmbeddingModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-google-gemma-300m")]
            google_gemma_300m::SELECTOR => Ok(Self::GoogleGemma300m),
            #[cfg(not(feature = "model-google-gemma-300m"))]
            GOOGLE_GEMMA_300M_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownEmbeddingModel {
                selector: other.to_owned(),
            }),
        }
    }
}
