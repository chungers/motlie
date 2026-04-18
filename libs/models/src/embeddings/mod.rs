#[cfg(any(
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b"
))]
use std::fmt;
use std::str::FromStr;

#[cfg(any(
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b"
))]
use motlie_model::{BundleId, EmbeddingSpec};

pub const GOOGLE_GEMMA_300M_SELECTOR: &str = "google/embeddinggemma_300m";
pub const QWEN3_EMBEDDING_06B_SELECTOR: &str = "qwen/qwen3_embedding_06b";

#[cfg(feature = "model-google-gemma-300m")]
pub mod google_gemma_300m;
#[cfg(feature = "model-qwen3-embedding-06b")]
pub mod qwen3_embedding_06b;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EmbeddingModels {
    #[cfg(feature = "model-google-gemma-300m")]
    GoogleGemma300m,
    #[cfg(feature = "model-qwen3-embedding-06b")]
    Qwen3Embedding06B,
}

#[cfg(any(
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b"
))]
impl EmbeddingModels {
    pub fn only_enabled() -> crate::Result<Self> {
        let enabled = [
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B,
        ];
        if enabled.len() == 1 {
            Ok(enabled[0])
        } else {
            Err(crate::ModelsError::AmbiguousEmbeddingModelSelection {
                count: enabled.len(),
            })
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::SELECTOR,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => qwen3_embedding_06b::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::descriptor().id,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => qwen3_embedding_06b::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::descriptor(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => qwen3_embedding_06b::descriptor(),
        }
    }

    pub fn bundle(&self) -> crate::CuratedBundle {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::bundle(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => qwen3_embedding_06b::bundle(),
        }
    }

    pub fn embedding_spec(&self) -> &'static EmbeddingSpec {
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::GoogleGemma300m => google_gemma_300m::embedding_spec(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => qwen3_embedding_06b::embedding_spec(),
        }
    }
}

#[cfg(any(
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b"
))]
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
            #[cfg(feature = "model-qwen3-embedding-06b")]
            qwen3_embedding_06b::SELECTOR => Ok(Self::Qwen3Embedding06B),
            #[cfg(not(feature = "model-google-gemma-300m"))]
            GOOGLE_GEMMA_300M_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(not(feature = "model-qwen3-embedding-06b"))]
            QWEN3_EMBEDDING_06B_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            other => Err(crate::ModelsError::UnknownEmbeddingModel {
                selector: other.to_owned(),
            }),
        }
    }
}
