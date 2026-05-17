use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities, ChatMessage,
    ChatModel, ChatRequest, ChatResponse, CheckpointFormat, CompletionModel, CompletionRequest,
    CompletionResponse, EmbeddingModel, LoadedBundleDescriptor, ModelBundle, ModelError,
    ModelIdentity, ModelMetricSnapshot, QuantizationBits, QuantizationSupport, ResolvedCheckpoint,
    StartOptions, UnsupportedCompletion, UnsupportedEmbeddings,
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
        let started_at = Instant::now();

        let response = self.model.send_chat_request(builder).await.map_err(|err| {
            ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "send_chat_request",
                message: err.to_string(),
            }
        })?;
        let elapsed = started_at.elapsed();

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
        let response =
            mistral_response_to_chat_response(choice.message, choice.finish_reason, &usage)?;

        {
            let mut metrics = lock_metrics(&self.metrics, P::CHAT_METRIC_CONTEXT);
            observe_latency(&mut metrics.runtime, elapsed);
            observe_text_usage(&mut metrics.text, &usage);
        }

        Ok(response)
    }
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
