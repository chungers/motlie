use std::fmt;
use std::str::FromStr;

use motlie_model::{BundleId, EmbeddingSpec, ModelBundle};

pub mod google_gemma_300m;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingModels {
    GoogleGemma300m,
}

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

impl fmt::Display for EmbeddingModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for EmbeddingModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            google_gemma_300m::SELECTOR => Ok(Self::GoogleGemma300m),
            other => Err(crate::ModelsError::UnknownEmbeddingModel {
                selector: other.to_owned(),
            }),
        }
    }
}
