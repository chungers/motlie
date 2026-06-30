use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use motlie_model::{ChatMessage, ToolCall, ToolInputSchema, ToolName, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const WEATHER_TOOL_NAME: &str = "get_weather";
pub const CEL_TOOL_NAME: &str = "evaluate_cel_expression";

pub trait EvalTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn spec(&self) -> Result<ToolSpec, EvalToolError>;
    fn execute(&self, arguments: &Value) -> Result<Value, EvalToolError>;
}

#[derive(Clone)]
pub struct EvalToolRegistry {
    handlers: BTreeMap<String, Arc<dyn EvalTool>>,
}

impl fmt::Debug for EvalToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalToolRegistry")
            .field("tools", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl EvalToolRegistry {
    pub fn empty() -> Self {
        Self {
            handlers: BTreeMap::new(),
        }
    }

    pub fn with_default_tools() -> Self {
        let mut registry = Self::empty();
        registry.register(WeatherTool);
        registry.register(CelExpressionTool);
        registry
    }

    pub fn with_tools<'a>(tools: impl IntoIterator<Item = &'a str>) -> Result<Self, EvalToolError> {
        let available = Self::with_default_tools();
        let mut registry = Self::empty();
        for tool in tools {
            let Some(handler) = available.handlers.get(tool) else {
                return Err(EvalToolError::UnknownTool(tool.to_owned()));
            };
            registry
                .handlers
                .insert(tool.to_owned(), Arc::clone(handler));
        }
        Ok(registry)
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: EvalTool + 'static,
    {
        self.handlers.insert(tool.name().to_owned(), Arc::new(tool));
    }

    pub fn specs(&self) -> Result<Vec<ToolSpec>, EvalToolError> {
        self.handlers
            .values()
            .map(|handler| handler.spec())
            .collect()
    }

    pub fn execute(&self, call: &ToolCall) -> Result<ToolExecution, EvalToolError> {
        let name = call.name.as_str();
        let Some(handler) = self.handlers.get(name) else {
            return Err(EvalToolError::UnknownTool(name.to_owned()));
        };
        let arguments = serde_json::from_str::<Value>(call.arguments.raw_json_str())?;
        let output = handler.execute(&arguments)?;
        let output_json = serde_json::to_string(&output)?;
        Ok(ToolExecution {
            call_id: call.id.as_str().to_owned(),
            name: name.to_owned(),
            arguments,
            output,
            output_json,
        })
    }

    pub fn execute_message(
        &self,
        call: &ToolCall,
    ) -> Result<(ToolExecution, ChatMessage), EvalToolError> {
        let execution = self.execute(call)?;
        let message = ChatMessage::tool_result_parts(
            call.id.clone(),
            call.name.clone(),
            execution.output_json.clone(),
        );
        Ok((execution, message))
    }
}

impl Default for EvalToolRegistry {
    fn default() -> Self {
        Self::with_default_tools()
    }
}

struct WeatherTool;

impl EvalTool for WeatherTool {
    fn name(&self) -> &'static str {
        WEATHER_TOOL_NAME
    }

    fn spec(&self) -> Result<ToolSpec, EvalToolError> {
        Ok(ToolSpec {
            name: ToolName::new(WEATHER_TOOL_NAME)?,
            description: "Return a deterministic weather summary for a city.".to_owned(),
            input_schema: ToolInputSchema::from_json_schema(
                r#"{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["fahrenheit","celsius"]}},"required":["city"]}"#,
            )?,
        })
    }

    fn execute(&self, arguments: &Value) -> Result<Value, EvalToolError> {
        let args = serde_json::from_value::<WeatherArgs>(arguments.clone())?;
        Ok(serde_json::to_value(get_weather(args))?)
    }
}

struct CelExpressionTool;

impl EvalTool for CelExpressionTool {
    fn name(&self) -> &'static str {
        CEL_TOOL_NAME
    }

    fn spec(&self) -> Result<ToolSpec, EvalToolError> {
        Ok(ToolSpec {
            name: ToolName::new(CEL_TOOL_NAME)?,
            description: "Evaluate a deterministic CEL arithmetic expression.".to_owned(),
            input_schema: ToolInputSchema::from_json_schema(
                r#"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#,
            )?,
        })
    }

    fn execute(&self, arguments: &Value) -> Result<Value, EvalToolError> {
        let args = serde_json::from_value::<CelExpressionArgs>(arguments.clone())?;
        Ok(serde_json::to_value(evaluate_cel_expression(args)?)?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolExecution {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub output: Value,
    pub output_json: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolTranscript {
    #[serde(default)]
    pub invocations: Vec<ToolExecution>,
    #[serde(default)]
    pub final_response: Option<String>,
    #[serde(default)]
    pub rounds: u32,
    #[serde(default)]
    pub tool_call_errors: Vec<String>,
}

impl ToolTranscript {
    pub fn tool_call_count(&self, tool: Option<&str>) -> usize {
        self.invocations
            .iter()
            .filter(|invocation| tool.is_none_or(|tool| invocation.name == tool))
            .count()
    }

    pub fn called(&self, tool: &str) -> bool {
        self.tool_call_count(Some(tool)) > 0
    }

    pub fn final_contains(&self, needle: &str) -> bool {
        self.final_response
            .as_deref()
            .is_some_and(|response| contains_case_insensitive(response, needle))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CelAssertionResult {
    pub expression: String,
    pub passed: bool,
    pub message: String,
}

pub fn evaluate_cel_assertion(
    expression: &str,
    transcript: &ToolTranscript,
) -> Result<CelAssertionResult, EvalToolError> {
    let mut messages = Vec::new();
    let mut passed = true;
    for clause in split_conjunctions(expression) {
        let clause_passed = evaluate_cel_clause(&clause, transcript)?;
        messages.push(format!("{clause}={clause_passed}"));
        passed &= clause_passed;
    }
    Ok(CelAssertionResult {
        expression: expression.to_owned(),
        passed,
        message: messages.join(", "),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeatherArgs {
    pub city: String,
    #[serde(default)]
    pub units: TemperatureUnits,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureUnits {
    Celsius,
    #[default]
    Fahrenheit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeatherOutput {
    pub city: String,
    pub temperature: f64,
    pub units: TemperatureUnits,
    pub summary: String,
}

pub fn get_weather(args: WeatherArgs) -> WeatherOutput {
    let temperature_f = match args.city.to_ascii_lowercase().as_str() {
        "seattle" => 72.0,
        "portland" => 68.0,
        "san francisco" => 64.0,
        "new york" | "new york city" => 75.0,
        "chicago" => 70.0,
        "austin" => 84.0,
        _ => 71.0,
    };
    let temperature = match args.units {
        TemperatureUnits::Celsius => (temperature_f - 32.0) * 5.0 / 9.0,
        TemperatureUnits::Fahrenheit => temperature_f,
    };
    WeatherOutput {
        city: args.city,
        temperature,
        units: args.units,
        summary: "clear".to_owned(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CelExpressionArgs {
    pub expression: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CelExpressionOutput {
    pub expression: String,
    pub value: f64,
    pub formatted: String,
    pub engine: &'static str,
}

pub fn evaluate_cel_expression(
    args: CelExpressionArgs,
) -> Result<CelExpressionOutput, EvalToolError> {
    let expression = args.expression.trim();
    if expression.is_empty() {
        return Err(EvalToolError::Expression(
            "expression cannot be empty".to_owned(),
        ));
    }
    if expression.len() > 512 {
        return Err(EvalToolError::Expression(
            "expression is longer than the 512 byte eval limit".to_owned(),
        ));
    }
    let value = ArithmeticParser::new(expression).parse()?;
    if !value.is_finite() {
        return Err(EvalToolError::Expression(
            "expression evaluated to a non-finite value".to_owned(),
        ));
    }
    Ok(CelExpressionOutput {
        expression: expression.to_owned(),
        value,
        formatted: trim_float(format!("{value:.6}")),
        engine: "motlie-eval-tools-cel-subset",
    })
}

#[derive(Debug, Error)]
pub enum EvalToolError {
    #[error("unknown eval tool `{0}`")]
    UnknownTool(String),
    #[error("invalid tool schema: {0}")]
    ToolSchema(#[from] motlie_model::ToolSchemaError),
    #[error("invalid tool arguments: {0}")]
    ToolArguments(#[from] motlie_model::ToolArgumentError),
    #[error("invalid tool JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("CEL expression failed: {0}")]
    Expression(String),
    #[error("CEL assertion failed: {0}")]
    Assertion(String),
    #[error("invalid eval tool name: {0}")]
    ToolName(#[from] motlie_model::ToolNameError),
}

fn split_conjunctions(expression: &str) -> Vec<String> {
    split_on_token_outside_quotes(expression, "&&")
        .into_iter()
        .map(|clause| clause.trim().to_owned())
        .filter(|clause| !clause.is_empty())
        .collect()
}

fn split_on_token_outside_quotes(input: &str, token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            }
            continue;
        }
        if quote.is_none() && input[index..].starts_with(token) {
            parts.push(input[start..index].to_owned());
            start = index + token.len();
        }
    }
    parts.push(input[start..].to_owned());
    parts
}

fn split_on_char_outside_quotes(input: &str, delimiter: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            }
            continue;
        }
        if quote.is_none() && ch == delimiter {
            parts.push(input[start..index].to_owned());
            start = index + ch.len_utf8();
        }
    }
    parts.push(input[start..].to_owned());
    parts
}

fn evaluate_cel_clause(clause: &str, transcript: &ToolTranscript) -> Result<bool, EvalToolError> {
    if let Some(args) = parse_function_call(clause, "tool_called") {
        let [tool] = one_arg(args, clause)?;
        return Ok(transcript.called(&tool));
    }
    if let Some(args) = parse_function_call(clause, "final_contains") {
        let [needle] = one_arg(args, clause)?;
        return Ok(transcript.final_contains(&needle));
    }
    if let Some(args) = parse_function_call(clause, "argument_equals") {
        let [tool, key, expected] = three_args(args, clause)?;
        return Ok(transcript.invocations.iter().any(|invocation| {
            invocation.name == tool
                && invocation
                    .arguments
                    .get(key.as_str())
                    .and_then(Value::as_str)
                    .is_some_and(|actual| actual.eq_ignore_ascii_case(&expected))
        }));
    }
    if let Some(args) = parse_function_call(clause, "round_trip_success") {
        if !args.trim().is_empty() {
            return Err(EvalToolError::Assertion(format!(
                "`{clause}` does not accept arguments"
            )));
        }
        return Ok(transcript.final_response.is_some() && !transcript.invocations.is_empty());
    }
    if let Some(result) = evaluate_field_equality(clause, transcript)? {
        return Ok(result);
    }
    if let Some(result) = evaluate_numeric_comparison(clause, transcript)? {
        return Ok(result);
    }
    Err(EvalToolError::Assertion(format!(
        "unsupported CEL assertion clause `{clause}`"
    )))
}

fn evaluate_field_equality(
    clause: &str,
    transcript: &ToolTranscript,
) -> Result<Option<bool>, EvalToolError> {
    let Some((left, right)) = split_once_operator_outside_quotes(clause, "==") else {
        return Ok(None);
    };
    let Some(actual) = tool_call_field_value(left.trim(), transcript) else {
        return Ok(None);
    };
    let expected = parse_quoted_string(right.trim()).ok_or_else(|| {
        EvalToolError::Assertion(format!(
            "CEL assertion right side `{}` must be a quoted string",
            right.trim()
        ))
    })?;
    Ok(Some(actual.eq_ignore_ascii_case(&expected)))
}

fn split_once_operator_outside_quotes<'a>(
    input: &'a str,
    operator: &str,
) -> Option<(&'a str, &'a str)> {
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            }
            continue;
        }
        if quote.is_none() && input[index..].starts_with(operator) {
            return Some((&input[..index], &input[index + operator.len()..]));
        }
    }
    None
}

fn tool_call_field_value(path: &str, transcript: &ToolTranscript) -> Option<String> {
    let rest = path.strip_prefix("tool_calls[")?;
    let (index, field) = rest.split_once(']')?;
    let index = index.parse::<usize>().ok()?;
    let field = field.strip_prefix('.')?;
    let invocation = transcript.invocations.get(index)?;
    if field == "name" {
        return Some(invocation.name.clone());
    }
    let key = field.strip_prefix("args.")?;
    invocation.arguments.get(key).and_then(json_value_to_string)
}

fn json_value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn evaluate_numeric_comparison(
    clause: &str,
    transcript: &ToolTranscript,
) -> Result<Option<bool>, EvalToolError> {
    for operator in [">=", "<=", "==", ">", "<"] {
        if let Some((left, right)) = clause.split_once(operator) {
            let left = numeric_term(left.trim(), transcript)?;
            let right = right.trim().parse::<f64>().map_err(|_| {
                EvalToolError::Assertion(format!("invalid numeric literal `{right}`"))
            })?;
            return Ok(Some(match operator {
                ">=" => left >= right,
                "<=" => left <= right,
                "==" => (left - right).abs() < f64::EPSILON,
                ">" => left > right,
                "<" => left < right,
                _ => unreachable!(),
            }));
        }
    }
    Ok(None)
}

fn numeric_term(term: &str, transcript: &ToolTranscript) -> Result<f64, EvalToolError> {
    if let Some(args) = parse_function_call(term, "tool_call_count") {
        if args.trim().is_empty() {
            return Ok(transcript.tool_call_count(None) as f64);
        }
        let [tool] = one_arg(args, term)?;
        return Ok(transcript.tool_call_count(Some(&tool)) as f64);
    }
    if let Some(args) = parse_function_call(term, "tool_precision") {
        if !args.trim().is_empty() {
            return Err(EvalToolError::Assertion(format!(
                "`{term}` does not accept arguments"
            )));
        }
        let total = transcript.invocations.len() + transcript.tool_call_errors.len();
        if total == 0 {
            return Ok(0.0);
        }
        return Ok(transcript.invocations.len() as f64 / total as f64);
    }
    term.parse::<f64>()
        .map_err(|_| EvalToolError::Assertion(format!("unsupported numeric term `{term}`")))
}

fn parse_function_call<'a>(clause: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}(");
    clause
        .strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(')'))
}

fn one_arg(args: &str, clause: &str) -> Result<[String; 1], EvalToolError> {
    let args = parse_string_args(args)?;
    match args.as_slice() {
        [one] => Ok([one.clone()]),
        _ => Err(EvalToolError::Assertion(format!(
            "`{clause}` expected one string argument"
        ))),
    }
}

fn three_args(args: &str, clause: &str) -> Result<[String; 3], EvalToolError> {
    let args = parse_string_args(args)?;
    match args.as_slice() {
        [one, two, three] => Ok([one.clone(), two.clone(), three.clone()]),
        _ => Err(EvalToolError::Assertion(format!(
            "`{clause}` expected three string arguments"
        ))),
    }
}

fn parse_string_args(args: &str) -> Result<Vec<String>, EvalToolError> {
    let mut parsed = Vec::new();
    if args.trim().is_empty() {
        return Ok(parsed);
    }
    for raw in split_on_char_outside_quotes(args, ',') {
        let raw = raw.trim();
        let Some(unquoted) = parse_quoted_string(raw) else {
            return Err(EvalToolError::Assertion(format!(
                "CEL assertion argument `{raw}` must be quoted"
            )));
        };
        parsed.push(unquoted);
    }
    Ok(parsed)
}

fn parse_quoted_string(raw: &str) -> Option<String> {
    let quote = raw.chars().next()?;
    if !matches!(quote, '\'' | '"') || !raw.ends_with(quote) || raw.len() < 2 {
        return None;
    }
    let inner = &raw[quote.len_utf8()..raw.len() - quote.len_utf8()];
    let mut out = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        out.push('\\');
    }
    Some(out)
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn trim_float(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

struct ArithmeticParser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> ArithmeticParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            offset: 0,
        }
    }

    fn parse(mut self) -> Result<f64, EvalToolError> {
        let value = self.expression()?;
        self.skip_ws();
        if self.offset != self.input.len() {
            return Err(EvalToolError::Expression(format!(
                "unexpected token near byte {}",
                self.offset
            )));
        }
        Ok(value)
    }

    fn expression(&mut self) -> Result<f64, EvalToolError> {
        let mut value = self.term()?;
        loop {
            self.skip_ws();
            if self.take(b'+') {
                value += self.term()?;
            } else if self.take(b'-') {
                value -= self.term()?;
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn term(&mut self) -> Result<f64, EvalToolError> {
        let mut value = self.factor()?;
        loop {
            self.skip_ws();
            if self.take(b'*') {
                value *= self.factor()?;
            } else if self.take(b'/') {
                value /= self.factor()?;
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn factor(&mut self) -> Result<f64, EvalToolError> {
        self.skip_ws();
        if self.take(b'(') {
            let value = self.expression()?;
            self.skip_ws();
            if !self.take(b')') {
                return Err(EvalToolError::Expression("missing closing ')'".to_owned()));
            }
            return Ok(value);
        }
        if self.take(b'-') {
            return Ok(-self.factor()?);
        }
        self.number()
    }

    fn number(&mut self) -> Result<f64, EvalToolError> {
        self.skip_ws();
        let start = self.offset;
        while self
            .input
            .get(self.offset)
            .is_some_and(|ch| ch.is_ascii_digit() || *ch == b'.')
        {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(EvalToolError::Expression(format!(
                "expected number near byte {}",
                self.offset
            )));
        }
        std::str::from_utf8(&self.input[start..self.offset])
            .map_err(|err| EvalToolError::Expression(err.to_string()))?
            .parse::<f64>()
            .map_err(|err| EvalToolError::Expression(err.to_string()))
    }

    fn skip_ws(&mut self) {
        while self
            .input
            .get(self.offset)
            .is_some_and(|ch| ch.is_ascii_whitespace())
        {
            self.offset += 1;
        }
    }

    fn take(&mut self, ch: u8) -> bool {
        if self.input.get(self.offset).copied() == Some(ch) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::ToolCall;

    #[test]
    fn registry_executes_weather() {
        let registry = EvalToolRegistry::with_default_tools();
        let call =
            ToolCall::from_json_args("call-1", WEATHER_TOOL_NAME, r#"{"city":"Seattle"}"#).unwrap();

        let execution = registry.execute(&call).unwrap();

        assert_eq!(execution.name, WEATHER_TOOL_NAME);
        assert_eq!(execution.output["temperature"], 72.0);
    }

    #[test]
    fn cel_expression_evaluates_arithmetic_subset() {
        let output = evaluate_cel_expression(CelExpressionArgs {
            expression: "(72.0 + 68.0 + 64.0) / 3.0".to_owned(),
        })
        .unwrap();

        assert_eq!(output.formatted, "68");
    }

    #[test]
    fn cel_assertion_checks_tool_round_trip() {
        let registry = EvalToolRegistry::with_default_tools();
        let call =
            ToolCall::from_json_args("call-1", WEATHER_TOOL_NAME, r#"{"city":"Seattle"}"#).unwrap();
        let execution = registry.execute(&call).unwrap();
        let transcript = ToolTranscript {
            invocations: vec![execution],
            final_response: Some("Seattle is clear.".to_owned()),
            rounds: 1,
            tool_call_errors: Vec::new(),
        };

        let result = evaluate_cel_assertion(
            "tool_called('get_weather') && argument_equals('get_weather','city','Seattle') && final_contains('seattle') && tool_precision() >= 1.0",
            &transcript,
        )
        .unwrap();

        assert!(result.passed);
    }

    #[test]
    fn cel_assertion_accepts_tool_calls_field_paths() {
        let registry = EvalToolRegistry::with_default_tools();
        let call =
            ToolCall::from_json_args("call-1", WEATHER_TOOL_NAME, r#"{"city":"Seattle"}"#).unwrap();
        let execution = registry.execute(&call).unwrap();
        let transcript = ToolTranscript {
            invocations: vec![execution],
            final_response: Some("Seattle is clear.".to_owned()),
            rounds: 1,
            tool_call_errors: Vec::new(),
        };

        let result = evaluate_cel_assertion(
            r#"tool_calls[0].name == "get_weather" && tool_calls[0].args.city == "Seattle""#,
            &transcript,
        )
        .unwrap();

        assert!(result.passed);
    }

    #[test]
    fn cel_parser_keeps_quoted_commas_and_conjunctions() {
        let transcript = ToolTranscript {
            invocations: vec![ToolExecution {
                call_id: "call-1".to_owned(),
                name: WEATHER_TOOL_NAME.to_owned(),
                arguments: serde_json::json!({ "city": "Seattle, WA" }),
                output: serde_json::json!({}),
                output_json: "{}".to_owned(),
            }],
            final_response: Some("literal a && b".to_owned()),
            rounds: 1,
            tool_call_errors: Vec::new(),
        };

        let result = evaluate_cel_assertion(
            r#"argument_equals('get_weather','city','Seattle, WA') && final_contains("a && b")"#,
            &transcript,
        )
        .unwrap();

        assert!(result.passed);
    }
}
