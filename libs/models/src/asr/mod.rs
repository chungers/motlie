#[cfg(any(
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en"
))]
use std::fmt;
use std::str::FromStr;

#[cfg(any(
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en"
))]
use motlie_model::BundleId;

pub const MOONSHINE_STREAMING_SELECTOR: &str = "moonshine/streaming_en";
pub const SHERPA_ONNX_STREAMING_SELECTOR: &str = "sherpa-onnx/streaming_zipformer_en";
pub const SHERPA_ONNX_STREAMING_KROKO_2025_SELECTOR: &str =
    "sherpa-onnx/streaming_zipformer_en_kroko_2025";
pub const WHISPER_BASE_EN_SELECTOR: &str = "openai/whisper_base_en";

#[cfg(feature = "model-moonshine-streaming")]
pub mod moonshine_streaming_en;
#[cfg(feature = "model-sherpa-onnx-streaming")]
pub mod sherpa_onnx_streaming_en;
#[cfg(feature = "model-sherpa-onnx-streaming")]
pub mod sherpa_onnx_streaming_en_kroko_2025;
#[cfg(feature = "model-whisper-base-en")]
pub mod whisper_base_en;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AsrModels {
    #[cfg(feature = "model-moonshine-streaming")]
    MoonshineStreamingEn,
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingEn,
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingEnKroko2025,
    #[cfg(feature = "model-whisper-base-en")]
    WhisperBaseEn,
}

#[cfg(any(
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en"
))]
impl AsrModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => moonshine_streaming_en::SELECTOR,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEn => sherpa_onnx_streaming_en::SELECTOR,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEnKroko2025 => sherpa_onnx_streaming_en_kroko_2025::SELECTOR,
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => whisper_base_en::SELECTOR,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => moonshine_streaming_en::descriptor().id,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEn => sherpa_onnx_streaming_en::descriptor().id,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEnKroko2025 => {
                sherpa_onnx_streaming_en_kroko_2025::descriptor().id
            }
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => whisper_base_en::descriptor().id,
        }
    }

    pub fn descriptor(&self) -> crate::BundleDescriptor {
        match self {
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => moonshine_streaming_en::descriptor(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEn => sherpa_onnx_streaming_en::descriptor(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEnKroko2025 => {
                sherpa_onnx_streaming_en_kroko_2025::descriptor()
            }
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => whisper_base_en::descriptor(),
        }
    }

    pub fn bundle(&self) -> crate::CuratedBundle {
        match self {
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => crate::CuratedBundle::MoonshineStreamingEn,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEn => crate::CuratedBundle::SherpaOnnxStreamingEn,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingEnKroko2025 => {
                crate::CuratedBundle::SherpaOnnxStreamingEnKroko2025
            }
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => crate::CuratedBundle::WhisperBaseEn,
        }
    }
}

#[cfg(any(
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en"
))]
impl fmt::Display for AsrModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl FromStr for AsrModels {
    type Err = crate::ModelsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            #[cfg(feature = "model-moonshine-streaming")]
            moonshine_streaming_en::SELECTOR => Ok(Self::MoonshineStreamingEn),
            #[cfg(not(feature = "model-moonshine-streaming"))]
            MOONSHINE_STREAMING_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            sherpa_onnx_streaming_en::SELECTOR => Ok(Self::SherpaOnnxStreamingEn),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            sherpa_onnx_streaming_en_kroko_2025::SELECTOR => {
                Ok(Self::SherpaOnnxStreamingEnKroko2025)
            }
            #[cfg(not(feature = "model-sherpa-onnx-streaming"))]
            SHERPA_ONNX_STREAMING_SELECTOR => Err(crate::ModelsError::ModelUnavailable {
                selector: value.to_owned(),
            }),
            #[cfg(not(feature = "model-sherpa-onnx-streaming"))]
            SHERPA_ONNX_STREAMING_KROKO_2025_SELECTOR => {
                Err(crate::ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                })
            }
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
