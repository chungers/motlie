use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::future::Future;
use std::marker::PhantomData;

use async_trait::async_trait;
use motlie_model::{
    ChatMessage, Tool, ToolArgumentError, ToolArguments, ToolCall, ToolName, ToolSchemaError,
    ToolSpec,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool `{0}` is already registered")]
    DuplicateTool(ToolName),
    #[error("tool execution failed: {0}")]
    Execution(#[source] Box<dyn StdError + Send + Sync>),
    #[error(transparent)]
    InvalidArguments(#[from] ToolArgumentError),
    #[error("failed to serialize tool output: {0}")]
    OutputSerialization(serde_json::Error),
    #[error(transparent)]
    Schema(#[from] ToolSchemaError),
    #[error("tool `{0}` is not registered")]
    UnknownTool(ToolName),
}

impl ToolError {
    pub fn execution(message: impl Into<String>) -> Self {
        Self::Execution(Box::new(ToolExecutionMessage(message.into())))
    }

    pub fn execution_source(error: impl StdError + Send + Sync + 'static) -> Self {
        Self::Execution(Box::new(error))
    }
}

#[derive(Debug)]
struct ToolExecutionMessage(String);

impl std::fmt::Display for ToolExecutionMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for ToolExecutionMessage {}

#[async_trait]
trait ErasedTool: Send + Sync {
    fn spec(&self) -> &ToolSpec;
    async fn call_json(&self, args: ToolArguments) -> Result<String, ToolError>;
}

struct ToolAdapter<T: Tool> {
    tool: T,
    spec: ToolSpec,
}

impl<T: Tool> ToolAdapter<T> {
    fn new(tool: T) -> Result<Self, ToolError> {
        let spec = tool.spec()?;
        Ok(Self { tool, spec })
    }
}

#[async_trait]
impl<T: Tool> ErasedTool for ToolAdapter<T> {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call_json(&self, args: ToolArguments) -> Result<String, ToolError> {
        let args = args.parse::<T::Args>()?;
        let output = self
            .tool
            .call(args)
            .await
            .map_err(ToolError::execution_source)?;
        serde_json::to_string(&output).map_err(ToolError::OutputSerialization)
    }
}

struct FunctionTool<Args, Output, E, F> {
    name: &'static str,
    description: &'static str,
    f: F,
    _marker: PhantomData<fn(Args) -> Result<Output, E>>,
}

impl<Args, Output, E, F> FunctionTool<Args, Output, E, F> {
    fn new(name: &'static str, description: &'static str, f: F) -> Self {
        Self {
            name,
            description,
            f,
            _marker: PhantomData,
        }
    }
}

impl<Args, Output, E, F, Fut> Tool for FunctionTool<Args, Output, E, F>
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    E: StdError + Send + Sync + 'static,
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Output, E>> + Send + 'static,
{
    type Args = Args;
    type Output = Output;
    type Error = E;

    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        (self.f)(args)
    }
}

/// Caller-owned registry for examples and applications that execute Rust tools.
///
/// The core `motlie-model` crate stays statically dispatched and only defines
/// portable tool-call contracts. This curated/examples crate owns the
/// runtime-extensible registry because heterogeneous Rust tools require type
/// erasure at the execution boundary.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<ToolName, Box<dyn ErasedTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T>(&mut self, tool: T) -> Result<&mut Self, ToolError>
    where
        T: Tool,
    {
        let adapter = ToolAdapter::new(tool)?;
        self.insert_erased(Box::new(adapter))
    }

    pub fn insert_fn<Args, Output, E, F, Fut>(
        &mut self,
        name: &'static str,
        description: &'static str,
        f: F,
    ) -> Result<&mut Self, ToolError>
    where
        Args: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + Send + 'static,
        E: StdError + Send + Sync + 'static,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output, E>> + Send + 'static,
    {
        self.insert(FunctionTool::<Args, Output, E, F>::new(
            name,
            description,
            f,
        ))
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| tool.spec().clone())
            .collect()
    }

    pub async fn call_to_message(&self, call: ToolCall) -> Result<ChatMessage, ToolError> {
        let id = call.id.clone();
        let name = call.name.clone();
        let tool = self
            .tools
            .get(&name)
            .ok_or_else(|| ToolError::UnknownTool(name.clone()))?;
        let content = tool.call_json(call.arguments).await?;
        Ok(ChatMessage::tool_result_parts(id, name, content))
    }

    fn insert_erased(&mut self, tool: Box<dyn ErasedTool>) -> Result<&mut Self, ToolError> {
        let name = tool.spec().name.clone();
        if self.tools.contains_key(&name) {
            return Err(ToolError::DuplicateTool(name));
        }

        self.tools.insert(name, tool);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct AddArgs {
        left: i64,
        right: i64,
    }

    #[derive(Debug, serde::Serialize)]
    struct AddOutput {
        value: i64,
    }

    async fn add(args: AddArgs) -> Result<AddOutput, ToolError> {
        Ok(AddOutput {
            value: args.left + args.right,
        })
    }

    #[tokio::test]
    async fn registry_executes_existing_function_binding() {
        let mut registry = ToolRegistry::new();
        registry
            .insert_fn("add", "Add two integers.", add)
            .expect("function should register");

        let call =
            ToolCall::from_serializable_args("call-1", "add", &AddArgs { left: 2, right: 3 })
                .expect("args should serialize");
        let message = registry
            .call_to_message(call)
            .await
            .expect("tool should run");

        assert_eq!(message.name.as_ref().map(ToolName::as_str), Some("add"));
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(
            message.content,
            vec![motlie_model::ContentPart::Text(r#"{"value":5}"#.into())]
        );
    }

    #[tokio::test]
    async fn registry_executes_closure_binding() {
        let mut registry = ToolRegistry::new();
        registry
            .insert_fn("add", "Add two integers.", |args: AddArgs| async move {
                Ok::<_, ToolError>(AddOutput {
                    value: args.left + args.right,
                })
            })
            .expect("closure should register");

        let call = ToolCall::from_json_args("call-1", "add", r#"{"left":4,"right":5}"#)
            .expect("args should validate");
        let message = registry
            .call_to_message(call)
            .await
            .expect("tool should run");

        assert_eq!(
            message.content,
            vec![motlie_model::ContentPart::Text(r#"{"value":9}"#.into())]
        );
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let mut registry = ToolRegistry::new();
        registry
            .insert_fn("add", "Add two integers.", add)
            .expect("first insert should succeed");
        let error = match registry.insert_fn("add", "Add two integers.", add) {
            Ok(_) => panic!("duplicate insert should fail"),
            Err(error) => error,
        };

        assert!(matches!(error, ToolError::DuplicateTool(name) if name.as_str() == "add"));
    }
}
