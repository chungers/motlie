use std::error::Error as StdError;
use std::future::Future;

use motlie_model::{
    ChatMessage, Tool, ToolArgumentError, ToolArguments, ToolCall, ToolName, ToolSchemaError,
    ToolSpec,
};
use thiserror::Error;

/// Result of asking a statically-typed tool list to handle a model tool call.
///
/// `NotMine` is deliberate: callers can compose local Rust tools with later
/// data-routed providers such as MCP servers without treating fallthrough as an
/// execution failure.
#[derive(Debug)]
pub enum ToolDispatch {
    Handled(ChatMessage),
    NotMine(ToolCall),
}

impl ToolDispatch {
    pub fn handled(self) -> Option<ChatMessage> {
        match self {
            Self::Handled(message) => Some(message),
            Self::NotMine(_) => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum ToolListError {
    #[error(transparent)]
    InvalidArguments(#[from] ToolArgumentError),
    #[error("failed to serialize tool output: {0}")]
    OutputSerialization(#[source] serde_json::Error),
    #[error(transparent)]
    Schema(#[from] ToolSchemaError),
    #[error("tool execution failed: {0}")]
    ToolFailed(#[source] Box<dyn StdError + Send + Sync>),
}

impl ToolListError {
    pub fn tool_failed(error: impl StdError + Send + Sync + 'static) -> Self {
        Self::ToolFailed(Box::new(error))
    }
}

/// Static-dispatched collection of typed Rust tools.
///
/// Implementations are provided for `()` and recursive `(Tool, ToolList)`
/// tuples. Use [`tool_list!`] for ergonomic construction.
pub trait ToolList: Send + Sync {
    fn collect_specs(&self, out: &mut Vec<ToolSpec>) -> Result<(), ToolListError>;

    fn dispatch(
        &self,
        call: ToolCall,
    ) -> impl Future<Output = Result<ToolDispatch, ToolListError>> + Send + '_;

    fn specs(&self) -> Result<Vec<ToolSpec>, ToolListError> {
        let mut specs = Vec::new();
        self.collect_specs(&mut specs)?;
        Ok(specs)
    }
}

impl ToolList for () {
    fn collect_specs(&self, _out: &mut Vec<ToolSpec>) -> Result<(), ToolListError> {
        Ok(())
    }

    fn dispatch(
        &self,
        call: ToolCall,
    ) -> impl Future<Output = Result<ToolDispatch, ToolListError>> + Send + '_ {
        async move { Ok(ToolDispatch::NotMine(call)) }
    }
}

impl<T, R> ToolList for (T, R)
where
    T: Tool,
    R: ToolList,
{
    fn collect_specs(&self, out: &mut Vec<ToolSpec>) -> Result<(), ToolListError> {
        out.push(self.0.spec()?);
        self.1.collect_specs(out)
    }

    fn dispatch(
        &self,
        call: ToolCall,
    ) -> impl Future<Output = Result<ToolDispatch, ToolListError>> + Send + '_ {
        async move {
            let ToolCall {
                id,
                name,
                arguments,
            } = call;

            if name.as_str() != self.0.name() {
                return self
                    .1
                    .dispatch(ToolCall {
                        id,
                        name,
                        arguments,
                    })
                    .await;
            }

            let args = arguments.parse::<T::Args>()?;
            let output = self
                .0
                .call(args)
                .await
                .map_err(ToolListError::tool_failed)?;
            let content =
                serde_json::to_string(&output).map_err(ToolListError::OutputSerialization)?;

            Ok(ToolDispatch::Handled(ChatMessage::tool_result_parts(
                id, name, content,
            )))
        }
    }
}

#[macro_export]
macro_rules! tool_list {
    () => {
        ()
    };
    ($head:expr $(, $tail:expr)* $(,)?) => {
        ($head, $crate::tool_list!($($tail),*))
    };
}

/// Scaffold for future Model Context Protocol data-routed tool integration.
///
/// PR #279 only defines the concrete type and host composition shape. Concrete
/// transports, JSON-RPC framing, initialize/shutdown lifecycle, and capability
/// negotiation are tracked by GitHub issue #284 and should land in a follow-up
/// PR.
#[derive(Debug)]
pub struct Mcp {
    server_name: String,
    transport: McpTransport,
    catalog: Vec<ToolSpec>,
}

#[derive(Debug)]
pub enum McpTransport {
    Unimplemented,
}

impl Mcp {
    pub fn new_unimplemented(server_name: impl Into<String>, catalog: Vec<ToolSpec>) -> Self {
        Self {
            server_name: server_name.into(),
            transport: McpTransport::Unimplemented,
            catalog,
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn transport(&self) -> &McpTransport {
        &self.transport
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.catalog
    }

    pub fn owns(&self, name: &ToolName) -> bool {
        let prefixed = format!("{}.", self.server_name);
        name.as_str().starts_with(&prefixed) || self.catalog.iter().any(|spec| spec.name == *name)
    }

    pub async fn call(&self, _name: &ToolName, _args: &ToolArguments) -> Result<String, McpError> {
        match self.transport {
            McpTransport::Unimplemented => Err(McpError::NotImplemented),
        }
    }
}

#[derive(Debug, Error)]
pub enum McpError {
    #[error("MCP transports are not implemented in this PR; see GitHub issue #284")]
    NotImplemented,
    #[error("MCP transport failure: {message}")]
    Transport { message: String },
    #[error("MCP JSON-RPC error {code}: {message}")]
    Rpc { code: i64, message: String },
    #[error("MCP tool `{name}` failed: {content}")]
    ToolFailed { name: ToolName, content: String },
    #[error("MCP tool `{0}` is not in the cached catalog")]
    UnknownTool(ToolName),
    #[error("MCP capability `{0}` was not negotiated")]
    CapabilityUnavailable(String),
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

    #[derive(Debug, Error)]
    #[error("add failed")]
    struct AddError;

    struct AddTool;

    impl Tool for AddTool {
        type Args = AddArgs;
        type Output = AddOutput;
        type Error = AddError;

        fn name(&self) -> &'static str {
            "add"
        }

        fn description(&self) -> &'static str {
            "Add two integers."
        }

        fn call(
            &self,
            args: Self::Args,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            async move {
                Ok(AddOutput {
                    value: args.left + args.right,
                })
            }
        }
    }

    #[tokio::test]
    async fn tool_list_executes_matching_tool() {
        let tools = tool_list!(AddTool);
        let call =
            ToolCall::from_serializable_args("call-1", "add", &AddArgs { left: 2, right: 3 })
                .expect("args should serialize");

        let message = match tools.dispatch(call).await.expect("tool should run") {
            ToolDispatch::Handled(message) => message,
            ToolDispatch::NotMine(call) => panic!("unexpected unknown tool: {}", call.name),
        };

        assert_eq!(message.name.as_ref().map(ToolName::as_str), Some("add"));
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(
            message.content,
            vec![motlie_model::ContentPart::Text(r#"{"value":5}"#.into())]
        );
    }

    #[tokio::test]
    async fn tool_list_falls_through_unknown_tool() {
        let tools = tool_list!(AddTool);
        let call =
            ToolCall::from_serializable_args("call-1", "other", &AddArgs { left: 2, right: 3 })
                .expect("args should serialize");

        let dispatch = tools
            .dispatch(call)
            .await
            .expect("fallthrough is not an error");

        assert!(matches!(dispatch, ToolDispatch::NotMine(call) if call.name.as_str() == "other"));
    }

    #[test]
    fn tool_list_collects_specs() {
        let tools = tool_list!(AddTool);
        let specs = tools.specs().expect("specs should collect");

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name.as_str(), "add");
    }

    #[test]
    fn mcp_owns_catalog_tool() {
        let spec = ToolSpec::from_args::<AddArgs>("add", "Add two integers.").expect("valid spec");
        let mcp = Mcp::new_unimplemented("math_server", vec![spec]);
        let add = ToolName::new("add").expect("valid name");
        let other = ToolName::new("other").expect("valid name");

        assert!(mcp.owns(&add));
        assert!(!mcp.owns(&other));
    }
}
