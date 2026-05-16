use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::value::RawValue;
use thiserror::Error;

/// Stable model-visible name for a callable tool.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToolName(String);

const MAX_TOOL_NAME_LEN: usize = 64;

impl ToolName {
    /// Create an OpenAI-compatible tool name.
    ///
    /// The shared contract keeps this validation in the common type because
    /// both llama.cpp's OpenAI-compatible templates and mistral.rs reject names
    /// outside this shape.
    pub fn new(value: impl Into<String>) -> Result<Self, ToolNameError> {
        let value = value.into();
        validate_tool_name(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for ToolName {
    type Error = ToolNameError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for ToolName {
    type Error = ToolNameError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Deref for ToolName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ToolNameError {
    #[error("tool name cannot be empty")]
    Empty,
    #[error("tool name `{0}` exceeds {MAX_TOOL_NAME_LEN} characters")]
    TooLong(String),
    #[error(
        "tool name `{0}` contains invalid character `{1}`; only ASCII letters, digits, underscores, and hyphens are allowed"
    )]
    InvalidCharacter(String, char),
}

fn validate_tool_name(value: &str) -> Result<(), ToolNameError> {
    if value.is_empty() {
        return Err(ToolNameError::Empty);
    }
    if value.len() > MAX_TOOL_NAME_LEN {
        return Err(ToolNameError::TooLong(value.to_owned()));
    }
    if let Some(ch) = value
        .chars()
        .find(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')))
    {
        return Err(ToolNameError::InvalidCharacter(value.to_owned(), ch));
    }

    Ok(())
}

/// Stable correlation id for a model-requested tool call.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(value: impl Into<String>) -> Result<Self, ToolCallIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ToolCallIdError::Empty);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for ToolCallId {
    type Error = ToolCallIdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for ToolCallId {
    type Error = ToolCallIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Deref for ToolCallId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ToolCallIdError {
    #[error("tool call id cannot be empty")]
    Empty,
}

/// Validated JSON Schema for a tool argument object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolInputSchema {
    raw_json_schema: String,
}

impl ToolInputSchema {
    pub fn from_args<T>() -> Result<Self, ToolSchemaError>
    where
        T: JsonSchema,
    {
        let schema = schemars::schema_for!(T);
        let value = serde_json::to_value(schema).map_err(ToolSchemaError::SerializeJson)?;
        Self::from_json_value(value)
    }

    pub fn from_json_schema(raw_json_schema: impl AsRef<str>) -> Result<Self, ToolSchemaError> {
        let value =
            serde_json::from_str(raw_json_schema.as_ref()).map_err(ToolSchemaError::InvalidJson)?;
        Self::from_json_value(value)
    }

    pub fn from_json_value(value: serde_json::Value) -> Result<Self, ToolSchemaError> {
        if !value.is_object() {
            return Err(ToolSchemaError::NotJsonObject);
        }

        let describes_object = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind == "object")
            || value
                .get("properties")
                .is_some_and(serde_json::Value::is_object);

        if !describes_object {
            return Err(ToolSchemaError::NotObjectSchema);
        }

        let raw_json_schema =
            serde_json::to_string(&value).map_err(ToolSchemaError::SerializeJson)?;
        Ok(Self { raw_json_schema })
    }

    pub fn as_json_str(&self) -> &str {
        &self.raw_json_schema
    }

    pub fn to_json_value(&self) -> Result<serde_json::Value, ToolSchemaError> {
        serde_json::from_str(&self.raw_json_schema).map_err(ToolSchemaError::InvalidJson)
    }
}

/// Model-facing schema for one callable tool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: String,
    pub input_schema: ToolInputSchema,
}

impl ToolSpec {
    pub fn from_args<T>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Self, ToolSchemaError>
    where
        T: JsonSchema,
    {
        Ok(Self {
            name: ToolName::new(name)?,
            description: description.into(),
            input_schema: ToolInputSchema::from_args::<T>()?,
        })
    }

    pub fn from_json_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        raw_json_schema: impl AsRef<str>,
    ) -> Result<Self, ToolSchemaError> {
        Ok(Self {
            name: ToolName::new(name)?,
            description: description.into(),
            input_schema: ToolInputSchema::from_json_schema(raw_json_schema)?,
        })
    }
}

/// Lossless JSON argument payload emitted by a model for a tool call.
#[derive(Clone, Debug)]
pub struct ToolArguments {
    raw_json: Box<RawValue>,
}

impl PartialEq for ToolArguments {
    fn eq(&self, other: &Self) -> bool {
        self.raw_json.get() == other.raw_json.get()
    }
}

impl ToolArguments {
    pub fn from_json_str(raw_json: impl Into<String>) -> Result<Self, ToolArgumentError> {
        let raw_json = raw_json.into();
        let value: serde_json::Value =
            serde_json::from_str(&raw_json).map_err(ToolArgumentError::InvalidJson)?;
        if !value.is_object() {
            return Err(ToolArgumentError::NotObject);
        }

        let raw_json = RawValue::from_string(raw_json).map_err(ToolArgumentError::InvalidJson)?;
        Ok(Self { raw_json })
    }

    pub fn from_serializable<T>(value: &T) -> Result<Self, ToolArgumentError>
    where
        T: Serialize,
    {
        let raw_json = serde_json::to_string(value).map_err(ToolArgumentError::Serialize)?;
        Self::from_json_str(raw_json)
    }

    pub fn parse<T>(&self) -> Result<T, ToolArgumentError>
    where
        T: DeserializeOwned,
    {
        serde_json::from_str(self.raw_json.get()).map_err(ToolArgumentError::Deserialize)
    }

    pub fn raw_json(&self) -> &RawValue {
        &self.raw_json
    }

    pub fn raw_json_str(&self) -> &str {
        self.raw_json.get()
    }
}

/// One tool call requested by an assistant turn.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: ToolArguments,
}

impl ToolCall {
    pub fn new(id: ToolCallId, name: ToolName, arguments: ToolArguments) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }

    pub fn from_json_args(
        id: impl Into<String>,
        name: impl Into<String>,
        raw_json: impl Into<String>,
    ) -> Result<Self, ToolCallError> {
        Ok(Self::new(
            ToolCallId::new(id)?,
            ToolName::new(name)?,
            ToolArguments::from_json_str(raw_json)?,
        ))
    }

    pub fn from_serializable_args<T>(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: &T,
    ) -> Result<Self, ToolCallError>
    where
        T: Serialize,
    {
        Ok(Self::new(
            ToolCallId::new(id)?,
            ToolName::new(name)?,
            ToolArguments::from_serializable(arguments)?,
        ))
    }
}

/// Tool-selection policy for a chat generation request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    Auto,
    Named(ToolName),
    None,
    Required,
}

impl ToolChoice {
    pub fn named(name: impl Into<String>) -> Result<Self, ToolNameError> {
        Ok(Self::Named(ToolName::new(name)?))
    }
}

#[derive(Debug, Error)]
pub enum ToolSchemaError {
    #[error(transparent)]
    InvalidName(#[from] ToolNameError),
    #[error("tool input schema must be a JSON object")]
    NotJsonObject,
    #[error("tool input schema must describe a JSON object argument payload")]
    NotObjectSchema,
    #[error("invalid tool input schema JSON: {0}")]
    InvalidJson(serde_json::Error),
    #[error("failed to serialize tool input schema: {0}")]
    SerializeJson(serde_json::Error),
}

#[derive(Debug, Error)]
pub enum ToolArgumentError {
    #[error("invalid tool argument JSON: {0}")]
    InvalidJson(serde_json::Error),
    #[error("tool arguments must be a JSON object")]
    NotObject,
    #[error("failed to deserialize tool arguments: {0}")]
    Deserialize(serde_json::Error),
    #[error("failed to serialize tool arguments: {0}")]
    Serialize(serde_json::Error),
}

#[derive(Debug, Error)]
pub enum ToolCallError {
    #[error(transparent)]
    InvalidId(#[from] ToolCallIdError),
    #[error(transparent)]
    InvalidName(#[from] ToolNameError),
    #[error(transparent)]
    InvalidArguments(#[from] ToolArgumentError),
}

/// Typed Rust binding for one callable tool.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    type Args: DeserializeOwned + JsonSchema + Send + 'static;
    type Output: Serialize + Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;

    fn spec(&self) -> Result<ToolSpec, ToolSchemaError> {
        ToolSpec::from_args::<Self::Args>(self.name(), self.description())
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

    #[derive(Debug, Error)]
    #[error("add failed")]
    struct AddError;

    struct AddTool;

    #[async_trait]
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

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(AddOutput {
                value: args.left + args.right,
            })
        }
    }

    #[test]
    fn tool_spec_from_args_produces_object_schema() {
        let spec = ToolSpec::from_args::<AddArgs>("add", "Add two integers.")
            .expect("schema should build");

        assert_eq!(spec.name.as_str(), "add");
        assert_eq!(spec.description, "Add two integers.");
        assert!(spec.input_schema.as_json_str().contains("\"left\""));
    }

    #[test]
    fn tool_name_validates_openai_compatible_shape() {
        assert!(ToolName::new("add_numbers-1").is_ok());
        assert!(matches!(ToolName::new(""), Err(ToolNameError::Empty)));
        assert!(matches!(
            ToolName::new("bad name"),
            Err(ToolNameError::InvalidCharacter(_, ' '))
        ));
        assert!(matches!(
            ToolName::new("x".repeat(MAX_TOOL_NAME_LEN + 1)),
            Err(ToolNameError::TooLong(_))
        ));
    }

    #[test]
    fn named_tool_choice_validates_name() {
        assert!(matches!(
            ToolChoice::named("bad name"),
            Err(ToolNameError::InvalidCharacter(_, ' '))
        ));
        assert_eq!(
            ToolChoice::named("add").expect("tool name should validate"),
            ToolChoice::Named(ToolName::new("add").expect("tool name should validate"))
        );
    }

    #[test]
    fn tool_arguments_parse_typed_values() {
        let args =
            ToolArguments::from_json_str(r#"{"left":2,"right":3}"#).expect("json should validate");
        let parsed: AddArgs = args.parse().expect("args should parse");

        assert_eq!(parsed.left, 2);
        assert_eq!(parsed.right, 3);
    }

    #[test]
    fn tool_arguments_reject_non_object_json() {
        let error =
            ToolArguments::from_json_str(r#""not an object""#).expect_err("string args fail");

        assert!(matches!(error, ToolArgumentError::NotObject));
    }

    #[test]
    fn tool_call_rejects_invalid_ids_and_names() {
        let error =
            ToolCall::from_json_args("", "add", r#"{"left":1,"right":2}"#).expect_err("id fails");
        assert!(matches!(
            error,
            ToolCallError::InvalidId(ToolCallIdError::Empty)
        ));

        let error = ToolCall::from_json_args("call-1", "bad name", r#"{"left":1,"right":2}"#)
            .expect_err("name fails");
        assert!(matches!(
            error,
            ToolCallError::InvalidName(ToolNameError::InvalidCharacter(_, ' '))
        ));
    }

    #[test]
    fn tool_spec_rejects_non_object_argument_schema() {
        let error = ToolSpec::from_json_schema("bad", "Bad schema.", r#"{"type":"string"}"#)
            .expect_err("non-object schema should fail");

        assert!(matches!(error, ToolSchemaError::NotObjectSchema));
    }

    #[tokio::test]
    async fn tool_trait_exposes_spec_and_typed_call() {
        let tool = AddTool;
        let spec = tool.spec().expect("schema should build");
        let output = tool
            .call(AddArgs { left: 2, right: 3 })
            .await
            .expect("tool should run");

        assert_eq!(spec.name.as_str(), "add");
        assert_eq!(output.value, 5);
    }
}
