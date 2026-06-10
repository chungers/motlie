use crate::units::{Bytes, Milliseconds, Tokens, TokensPerSecond};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelMetricSnapshot {
    pub runtime: Option<RuntimeMetrics>,
    pub text_generation: Option<TextGenerationMetrics>,
    pub embeddings: Option<EmbeddingMetrics>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeAcceleratorObservation {
    pub backend_mode: String,
    pub offload: Option<String>,
    pub selected_device: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeMetrics {
    pub resident_memory: Option<Bytes>,
    pub peak_resident_memory: Option<Bytes>,
    pub request_count: Option<u64>,
    pub last_latency: Option<Milliseconds>,
    pub max_latency: Option<Milliseconds>,
    pub avg_latency: Option<Milliseconds>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextGenerationMetrics {
    pub total_prompt_tokens: Option<Tokens>,
    pub total_generated_tokens: Option<Tokens>,
    pub total_tokens: Option<Tokens>,
    pub avg_prompt_tokens_per_sec: Option<TokensPerSecond>,
    pub avg_generated_tokens_per_sec: Option<TokensPerSecond>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EmbeddingMetrics {
    pub request_count: Option<u64>,
    pub input_count: Option<u64>,
}
