use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities, ChatMessage,
    ChatModel, ChatRequest, ChatResponse, CheckpointFormat, CompletionModel, CompletionRequest,
    CompletionResponse, EmbeddingModel, GenerationTiming, LoadedBundleDescriptor, ModelBundle,
    ModelError, ModelIdentity, ModelMetricSnapshot, QuantizationBits, QuantizationSupport,
    ResolvedCheckpoint, StartOptions, UnsupportedCompletion, UnsupportedEmbeddings,
};

use crate::common::{
    chat_request_to_builder, lock_metrics, mistral_response_to_chat_response, observe_latency,
    observe_memory, observe_text_usage, resolve_local_checkpoint, snapshot_text_metrics,
    MistralMessageParts, RuntimeMetricState, TextMetricState,
};

#[derive(Clone, Debug, Default)]
pub struct MistralChatMetrics {
    pub(crate) runtime: RuntimeMetricState,
    pub(crate) text: TextMetricState,
}

pub trait MistralProfile: Sized + Send + Sync + 'static {
    type Arch: Copy + Eq + std::fmt::Debug + Send + Sync + 'static;
    type Completion: CompletionModel;
    type Embeddings: EmbeddingModel;

    const FORMATS: &'static [CheckpointFormat];
    const START_METRIC_CONTEXT: &'static str;
    const CHAT_METRIC_CONTEXT: &'static str;
    const SNAPSHOT_METRIC_CONTEXT: &'static str;

    fn build_model(
        model_id: &str,
        arch: Self::Arch,
        resolved_quantization: Option<QuantizationBits>,
        options: StartOptions,
    ) -> impl Future<Output = Result<mistralrs::Model, ModelError>> + Send;

    fn collect_message(message: &ChatMessage) -> Result<MistralMessageParts, ModelError>;
    fn completion(handle: &MistralHandle<Self>) -> Result<&Self::Completion, ModelError>;
    fn embeddings(handle: &MistralHandle<Self>) -> Result<&Self::Embeddings, ModelError>;
}

#[derive(Clone, Debug)]
pub struct MistralAdapter<P: MistralProfile> {
    arch: P::Arch,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    _profile: PhantomData<P>,
}

impl<P: MistralProfile> MistralAdapter<P> {
    pub(crate) fn from_parts(
        arch: P::Arch,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
    ) -> Self {
        Self {
            arch,
            capabilities,
            quantization,
            _profile: PhantomData,
        }
    }
}

#[async_trait]
impl<P: MistralProfile> BackendAdapter for MistralAdapter<P> {
    type Handle = MistralHandle<P>;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        P::FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::MistralRs
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn quantization(&self) -> &QuantizationSupport {
        &self.quantization
    }

    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .quantization
            .resolve(options.quantization, &identity.id)?;
        let (model_id, options) =
            resolve_local_checkpoint(checkpoint, CheckpointFormat::Safetensors, options)?;
        let model = P::build_model(model_id, self.arch, resolved_quantization, options).await?;

        Ok(MistralHandle::real(
            identity.id.clone(),
            identity.display_name.clone(),
            self.capabilities.clone(),
            self.quantization.clone(),
            resolved_quantization,
            model,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct MistralBundle<P: MistralProfile> {
    metadata: BundleMetadata,
    arch: P::Arch,
    model_id: &'static str,
    _profile: PhantomData<P>,
}

impl<P: MistralProfile> MistralBundle<P> {
    pub(crate) fn from_parts(
        id: BundleId,
        display_name: &'static str,
        model_id: &'static str,
        arch: P::Arch,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
    ) -> Self {
        Self {
            metadata: BundleMetadata {
                id,
                display_name: display_name.into(),
                capabilities,
                quantization,
            },
            arch,
            model_id,
            _profile: PhantomData,
        }
    }
}

#[async_trait]
impl<P: MistralProfile> ModelBundle for MistralBundle<P> {
    type Handle = MistralHandle<P>;

    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(&self, options: StartOptions) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;
        let model =
            P::build_model(self.model_id, self.arch, resolved_quantization, options).await?;

        Ok(MistralHandle::real(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            resolved_quantization,
            model,
        ))
    }
}

pub struct MistralHandle<P: MistralProfile> {
    descriptor: LoadedBundleDescriptor,
    runtime: MistralRuntime<P>,
    metrics: Arc<Mutex<MistralChatMetrics>>,
    unsupported_completion: UnsupportedCompletion,
    unsupported_embeddings: UnsupportedEmbeddings,
}

impl<P: MistralProfile> MistralHandle<P> {
    fn real(
        id: BundleId,
        display_name: String,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
        resolved_quantization: Option<QuantizationBits>,
        model: mistralrs::Model,
    ) -> Self {
        Self::new(
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
            MistralRuntime::real(model),
        )
    }

    fn new(
        id: BundleId,
        display_name: String,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
        resolved_quantization: Option<QuantizationBits>,
        runtime: MistralRuntime<P>,
    ) -> Self {
        let metrics = Arc::new(Mutex::new(MistralChatMetrics::default()));
        {
            let mut metrics = lock_metrics(&metrics, P::START_METRIC_CONTEXT);
            observe_memory(&mut metrics.runtime);
        }

        Self {
            descriptor: LoadedBundleDescriptor {
                id,
                display_name,
                capabilities,
                quantization,
                resolved_quantization,
            },
            runtime: runtime.with_metrics(Arc::clone(&metrics)),
            metrics,
            unsupported_completion: UnsupportedCompletion,
            unsupported_embeddings: UnsupportedEmbeddings,
        }
    }

    #[cfg(test)]
    pub(crate) fn stub(
        id: BundleId,
        display_name: String,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
        resolved_quantization: Option<QuantizationBits>,
        kind: MistralStubKind,
    ) -> Self {
        Self::new(
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
            MistralRuntime::stub(kind),
        )
    }

    pub(crate) fn unsupported_completion(&self) -> &UnsupportedCompletion {
        &self.unsupported_completion
    }

    pub(crate) fn unsupported_embeddings(&self) -> &UnsupportedEmbeddings {
        &self.unsupported_embeddings
    }
}

#[async_trait]
impl<P: MistralProfile> BundleHandle for MistralHandle<P> {
    type Chat = Self;
    type Completion = P::Completion;
    type Embeddings = P::Embeddings;

    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, P::SNAPSHOT_METRIC_CONTEXT).clone();
        Some(snapshot_text_metrics(&metrics.runtime, &metrics.text))
    }

    fn accelerator_observation(&self) -> Option<motlie_model::RuntimeAcceleratorObservation> {
        Some(crate::common::accelerator_observation())
    }

    fn chat(&self) -> Result<&Self::Chat, ModelError> {
        Ok(self)
    }

    fn completion(&self) -> Result<&Self::Completion, ModelError> {
        P::completion(self)
    }

    fn embeddings(&self) -> Result<&Self::Embeddings, ModelError> {
        P::embeddings(self)
    }

    async fn shutdown(self) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl<P: MistralProfile> ChatModel for MistralHandle<P> {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

#[async_trait]
impl<P> CompletionModel for MistralHandle<P>
where
    P: MistralProfile<Completion = MistralHandle<P>>,
{
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        self.runtime.complete(request).await
    }
}

struct RealChatRuntime {
    model: mistralrs::Model,
    metrics: Arc<Mutex<MistralChatMetrics>>,
}

enum MistralRuntimeInner {
    Real(RealChatRuntime),
    #[cfg(test)]
    Stub(MistralStubKind),
}

struct MistralRuntime<P: MistralProfile> {
    inner: MistralRuntimeInner,
    _profile: PhantomData<P>,
}

impl<P: MistralProfile> MistralRuntime<P> {
    fn real(model: mistralrs::Model) -> Self {
        Self {
            inner: MistralRuntimeInner::Real(RealChatRuntime {
                model,
                metrics: Arc::new(Mutex::new(MistralChatMetrics::default())),
            }),
            _profile: PhantomData,
        }
    }

    #[cfg(test)]
    fn stub(kind: MistralStubKind) -> Self {
        Self {
            inner: MistralRuntimeInner::Stub(kind),
            _profile: PhantomData,
        }
    }

    fn with_metrics(mut self, metrics: Arc<Mutex<MistralChatMetrics>>) -> Self {
        match &mut self.inner {
            MistralRuntimeInner::Real(runtime) => runtime.metrics = metrics,
            #[cfg(test)]
            MistralRuntimeInner::Stub(_) => {}
        }
        self
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        match &self.inner {
            MistralRuntimeInner::Real(runtime) => runtime.chat::<P>(request).await,
            #[cfg(test)]
            MistralRuntimeInner::Stub(kind) => kind.chat(request),
        }
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        match &self.inner {
            MistralRuntimeInner::Real(runtime) => {
                let chat_request = ChatRequest {
                    messages: vec![motlie_model::ChatMessage::new(
                        motlie_model::ChatRole::User,
                        request.prompt,
                    )],
                    params: request.params,
                    ..Default::default()
                };
                let chat_response = runtime.chat::<P>(chat_request).await?;
                Ok(CompletionResponse {
                    content: chat_response.content,
                })
            }
            #[cfg(test)]
            MistralRuntimeInner::Stub(kind) => kind.complete(request),
        }
    }
}

impl RealChatRuntime {
    async fn chat<P: MistralProfile>(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, ModelError> {
        let builder = chat_request_to_builder(&request, P::collect_message)?;
        let request_at = Instant::now();

        let response = self.model.send_chat_request(builder).await.map_err(|err| {
            ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "send_chat_request",
                message: err.to_string(),
            }
        })?;
        let response_at = Instant::now();
        let elapsed = response_at.duration_since(request_at);

        let usage = response.usage.clone();
        let choice =
            response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| ModelError::BackendExecution {
                    backend: "mistralrs",
                    operation: "send_chat_request",
                    message: "response contained no choices".into(),
                })?;
        let timing = generation_timing_from_usage(request_at, response_at, &usage, &choice.message);
        let response = mistral_response_to_chat_response(
            choice.message,
            choice.finish_reason,
            &usage,
            Some(timing),
        )?;

        {
            let mut metrics = lock_metrics(&self.metrics, P::CHAT_METRIC_CONTEXT);
            observe_latency(&mut metrics.runtime, elapsed);
            observe_text_usage(&mut metrics.text, &usage);
        }

        Ok(response)
    }
}

fn generation_timing_from_usage(
    request_at: Instant,
    response_at: Instant,
    usage: &mistralrs::core::Usage,
    message: &mistralrs::ResponseMessage,
) -> GenerationTiming {
    let first_token_at = usage_duration(usage.total_prompt_time_sec)
        .and_then(|duration| request_at.checked_add(duration))
        .map(|instant| instant.min(response_at))
        .unwrap_or(response_at);
    let completion_duration = usage_duration(usage.total_completion_time_sec);
    let last_token_at = completion_duration
        .and_then(|duration| first_token_at.checked_add(duration))
        .map(|instant| instant.min(response_at))
        .unwrap_or(response_at);
    let first_answer_token_at = if message
        .reasoning_content
        .as_deref()
        .is_some_and(|reasoning| !reasoning.trim().is_empty())
        && message
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty())
    {
        last_token_at
    } else {
        first_token_at
    };

    GenerationTiming {
        request_at,
        first_token_at: Some(first_token_at),
        first_answer_token_at: Some(first_answer_token_at),
        last_token_at: Some(last_token_at),
        generated_tokens: usage_count_to_u32(usage.completion_tokens),
        // mistral.rs reports aggregate prompt/completion durations, not
        // per-token timings, so the reasoning token count at the think→answer
        // boundary is not available here.
        tokens_before_answer: None,
    }
}

fn usage_duration(seconds: f32) -> Option<Duration> {
    if seconds.is_finite() && seconds >= 0.0 {
        Some(Duration::from_secs_f64(f64::from(seconds)))
    } else {
        None
    }
}

fn usage_count_to_u32(count: usize) -> u32 {
    count.min(u32::MAX as usize) as u32
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum MistralStubKind {
    Text,
    Multimodal,
}

#[cfg(test)]
impl MistralStubKind {
    fn chat(self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        match self {
            Self::Text => {
                let prompt = request
                    .messages
                    .last()
                    .and_then(|m| m.content.first())
                    .and_then(|part| match part {
                        motlie_model::ContentPart::Text(text) => Some(text.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                Ok(ChatResponse::text(format!("stub response to: {prompt}")))
            }
            Self::Multimodal => {
                let last_text = request
                    .messages
                    .last()
                    .map(|m| {
                        m.content
                            .iter()
                            .filter_map(|part| match part {
                                motlie_model::ContentPart::Text(text) => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<String>()
                    })
                    .unwrap_or_default();
                Ok(ChatResponse::text(format!(
                    "multimodal stub response to: {last_text}"
                )))
            }
        }
    }

    fn complete(self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        match self {
            Self::Text => Ok(CompletionResponse {
                content: format!("stub completion of: {}", request.prompt),
            }),
            Self::Multimodal => Err(ModelError::UnsupportedCapability(
                motlie_model::CapabilityKind::Completion,
            )),
        }
    }
}
