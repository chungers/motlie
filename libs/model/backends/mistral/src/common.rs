use std::path::PathBuf;

use mistralrs::core::StopTokens;
use mistralrs::{IsqBits, RequestBuilder, SamplingParams};
use motlie_model::{ArtifactPolicy, ChatRole, GenerationParams, ModelError, QuantizationBits};

pub(crate) struct ConfiguredBuilder {
    pub(crate) model_target: String,
    pub(crate) hf_cache_root: Option<PathBuf>,
}

pub(crate) fn configure_artifact_policy(
    model_id: &str,
    policy: ArtifactPolicy,
) -> Result<ConfiguredBuilder, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => Ok(ConfiguredBuilder {
            model_target: model_id.to_owned(),
            hf_cache_root: root,
        }),
        ArtifactPolicy::LocalOnly { root } => Ok(ConfiguredBuilder {
            model_target: root.display().to_string(),
            hf_cache_root: None,
        }),
    }
}

pub(crate) fn map_quantization_bits(bits: QuantizationBits) -> IsqBits {
    match bits {
        QuantizationBits::Four => IsqBits::Four,
        QuantizationBits::Eight => IsqBits::Eight,
    }
}

pub(crate) fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn map_chat_role(role: ChatRole) -> mistralrs::TextMessageRole {
    match role {
        ChatRole::System => mistralrs::TextMessageRole::System,
        ChatRole::User => mistralrs::TextMessageRole::User,
        ChatRole::Assistant => mistralrs::TextMessageRole::Assistant,
    }
}

pub(crate) fn apply_generation_params(
    builder: RequestBuilder,
    params: &GenerationParams,
) -> RequestBuilder {
    let mut sampling = SamplingParams::deterministic();
    if let Some(temperature) = params.temperature {
        sampling.temperature = Some(temperature as f64);
        sampling.top_k = None;
    }
    if let Some(top_p) = params.top_p {
        sampling.top_p = Some(top_p as f64);
    }
    if let Some(max_tokens) = params.max_tokens {
        sampling.max_len = Some(max_tokens as usize);
    }
    if !params.stop_sequences.is_empty() {
        sampling.stop_toks = Some(StopTokens::Seqs(params.stop_sequences.clone()));
    }
    builder.set_sampling(sampling)
}
