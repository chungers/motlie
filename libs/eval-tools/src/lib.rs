use std::collections::BTreeSet;

use motlie_model::{ChatMessage, ToolCall, ToolInputSchema, ToolName, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const WEATHER_TOOL_NAME: &str = "get_weather";
pub const CEL_TOOL_NAME: &str = "evaluate_cel_expression";

#[derive(Clone, Debug)]
pub struct EvalToolRegistry {
    enabled: BTreeSet<String>,
}

impl EvalToolRegistry {
    pub fn with_default_tools() -> Self {
        Self {
            enabled: [WEATHER_TOOL_NAME.to_owned(), CEL_TOOL_NAME.to_owned()]
                .into_iter()
                .collect(),
        }
    }

    pub fn with_tools<'a>(tools: impl IntoIterator<Item = &'a str>) -> Result<Self, EvalToolError> {
        let mut enabled = BTreeSet::new();
        for tool in tools {
            if !is_builtin_tool(tool) {
                return Err(EvalToolError::UnknownTool(tool.to_owned()));
            }
            enabled.insert(tool.to_owned());
        }
        Ok(Self { enabled })
    }

    pub fn specs(&self) -> Result<Vec<ToolSpec>, EvalToolError> {
        let mut specs = Vec::new();
        for tool in &self.enabled {
            specs.push(spec_for_builtin(tool)?);
        }
        Ok(specs)
    }

    pub fn execute(&self, call: &ToolCall) -> Result<ToolExecution, EvalToolError> {
        let name = call.name.as_str();
        if !self.enabled.contains(name) {
            return Err(EvalToolError::UnknownTool(name.to_owned()));
        }
        let arguments = serde_json::from_str::<Value>(call.arguments.raw_json_str())?;
        let output = match name {
            WEATHER_TOOL_NAME => {
                let args = call.arguments.parse::<WeatherArgs>()?;
                serde_json::to_value(get_weather(args))?
            }
            CEL_TOOL_NAME => {
                let args = call.arguments.parse::<CelExpressionArgs>()?;
                serde_json::to_value(evaluate_cel_expression(args)?)?
            }
            other => return Err(EvalToolError::UnknownTool(other.to_owned())),
        };
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
        let clause_passed = evaluate_cel_clause(clause, transcript)?;
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

fn is_builtin_tool(tool: &str) -> bool {
    matches!(tool, WEATHER_TOOL_NAME | CEL_TOOL_NAME)
}

fn spec_for_builtin(tool: &str) -> Result<ToolSpec, EvalToolError> {
    match tool {
        WEATHER_TOOL_NAME => Ok(ToolSpec {
            name: ToolName::new(WEATHER_TOOL_NAME)?,
            description: "Return a deterministic weather summary for a city.".to_owned(),
            input_schema: ToolInputSchema::from_json_schema(
                r#"{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["fahrenheit","celsius"]}},"required":["city"]}"#,
            )?,
        }),
        CEL_TOOL_NAME => Ok(ToolSpec {
            name: ToolName::new(CEL_TOOL_NAME)?,
            description: "Evaluate a deterministic CEL arithmetic expression.".to_owned(),
            input_schema: ToolInputSchema::from_json_schema(
                r#"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#,
            )?,
        }),
        other => Err(EvalToolError::UnknownTool(other.to_owned())),
    }
}

fn split_conjunctions(expression: &str) -> Vec<&str> {
    expression
        .split("&&")
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
        .collect()
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
    if let Some(result) = evaluate_numeric_comparison(clause, transcript)? {
        return Ok(result);
    }
    Err(EvalToolError::Assertion(format!(
        "unsupported CEL assertion clause `{clause}`"
    )))
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
    for raw in args.split(',') {
        let raw = raw.trim();
        let Some(unquoted) = raw
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                raw.strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            })
        else {
            return Err(EvalToolError::Assertion(format!(
                "CEL assertion argument `{raw}` must be quoted"
            )));
        };
        parsed.push(unquoted.to_owned());
    }
    Ok(parsed)
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
}
