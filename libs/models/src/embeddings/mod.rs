#[cfg(feature = "model-google-gemma-300m")]
use std::fmt;
use std::str::FromStr;

#[cfg(feature = "model-google-gemma-300m")]
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

#[cfg(feature = "model-google-gemma-300m")]
impl EmbeddingModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GoogleGemma300m => google_gemma_300m::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            Self::GoogleGemma300m => google_gemma_300m::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            Self::GoogleGemma300m => google_gemma_300m::descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            Self::GoogleGemma300m => google_gemma_300m::bundle(),
        }
    }

    pub fn embedding_spec(&self) -> &'static EmbeddingSpec {
        match self {
            Self::GoogleGemma300m => google_gemma_300m::embedding_spec(),
        }
    }
}

#[cfg(feature = "model-google-gemma-300m")]
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
