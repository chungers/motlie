use anyhow::{bail, ensure, Context, Result};
use cel_cxx::{Activation, Env, Value};
use motlie_model::{
    ChatMessage, ChatModel, ChatRequest, ChatRole, ContentPart, GenerationParams, ThinkingMode,
    Tool, ToolChoice, ToolName,
};
use motlie_models::{tool_list, ToolDispatch, ToolList};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::future::Future;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeatherArgs {
    pub city: String,
    /// Temperature unit. Prefer lowercase `fahrenheit` or `celsius`.
    pub units: TemperatureUnits,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureUnits {
    #[serde(alias = "Celsius", alias = "C", alias = "c")]
    Celsius,
    #[serde(alias = "Fahrenheit", alias = "F", alias = "f")]
    Fahrenheit,
}

#[derive(Debug, Serialize)]
pub struct WeatherOutput {
    pub city: String,
    pub temperature: f32,
    pub units: TemperatureUnits,
    pub summary: String,
}

#[derive(Debug, thiserror::Error)]
#[error("weather lookup failed")]
pub struct WeatherError;

pub async fn get_weather(args: WeatherArgs) -> std::result::Result<WeatherOutput, WeatherError> {
    let temperature_fahrenheit = match args.city.to_ascii_lowercase().as_str() {
        "seattle" => 72.0,
        "portland" => 68.0,
        "san francisco" => 64.0,
        "new york" | "new york city" => 75.0,
        "chicago" => 70.0,
        "austin" => 84.0,
        _ => 71.0,
    };
    let temperature = match args.units {
        TemperatureUnits::Celsius => (temperature_fahrenheit - 32.0) * 5.0 / 9.0,
        TemperatureUnits::Fahrenheit => temperature_fahrenheit,
    };

    Ok(WeatherOutput {
        city: args.city,
        temperature,
        units: args.units,
        summary: "clear".to_string(),
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeatherTool;

impl Tool for WeatherTool {
    type Args = WeatherArgs;
    type Output = WeatherOutput;
    type Error = WeatherError;

    fn name(&self) -> &'static str {
        "get_weather"
    }

    fn description(&self) -> &'static str {
        "Return a current weather summary for a city."
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        get_weather(args)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MathExpressionArgs {
    /// CEL arithmetic expression to evaluate. Supports parentheses, numeric
    /// operators, comparisons, conditionals, and math.* extension functions.
    /// Use matching numeric types in division; for averages of decimal values,
    /// use a decimal divisor such as 3.0.
    pub expression: String,
}

#[derive(Debug, Serialize)]
pub struct MathExpressionOutput {
    pub expression: String,
    pub value: f64,
    pub formatted: String,
    pub engine: &'static str,
}

#[derive(Debug, thiserror::Error)]
pub enum MathExpressionError {
    #[error("expression cannot be empty")]
    Empty,
    #[error("expression is too long; keep examples under 512 bytes")]
    TooLong,
    #[error("failed to build CEL environment: {0}")]
    BuildEnvironment(#[source] cel_cxx::Error),
    #[error("failed to compile CEL expression: {0}")]
    CompileExpression(#[source] cel_cxx::Error),
    #[error("failed to evaluate CEL expression: {0}")]
    EvaluateExpression(#[source] cel_cxx::Error),
    #[error("CEL expression evaluated to non-numeric value: {0}")]
    NonNumeric(String),
    #[error("expression evaluated to a non-finite number")]
    NonFinite,
}

pub async fn evaluate_math_expression(
    args: MathExpressionArgs,
) -> std::result::Result<MathExpressionOutput, MathExpressionError> {
    let expression = args.expression.trim();
    if expression.is_empty() {
        return Err(MathExpressionError::Empty);
    }
    if expression.len() > 512 {
        return Err(MathExpressionError::TooLong);
    }

    let env = Env::builder()
        .with_ext_math(true)
        .build()
        .map_err(MathExpressionError::BuildEnvironment)?;
    let program = env
        .compile(expression)
        .map_err(MathExpressionError::CompileExpression)?;
    let value = program
        .evaluate(&Activation::new())
        .map_err(MathExpressionError::EvaluateExpression)?;
    let value = match value {
        Value::Int(value) => value as f64,
        Value::Uint(value) => value as f64,
        Value::Double(value) => value,
        other => {
            return Err(MathExpressionError::NonNumeric(other.to_string()));
        }
    };
    if !value.is_finite() {
        return Err(MathExpressionError::NonFinite);
    }

    Ok(MathExpressionOutput {
        expression: expression.to_string(),
        value,
        formatted: trim_float(format!("{value:.6}")),
        engine: "cel-cxx",
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvaluateMathExpressionTool;

impl Tool for EvaluateMathExpressionTool {
    type Args = MathExpressionArgs;
    type Output = MathExpressionOutput;
    type Error = MathExpressionError;

    fn name(&self) -> &'static str {
        "evaluate_math_expression"
    }

    fn description(&self) -> &'static str {
        "Evaluate a CEL arithmetic expression with parentheses, numeric operators, conditionals, and math.* functions. Use matching numeric types in division, for example divide decimal values by 3.0."
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        evaluate_math_expression(args)
    }
}

pub fn demo_tools() -> impl ToolList {
    tool_list!(WeatherTool, EvaluateMathExpressionTool)
}

#[derive(Clone, Copy, Debug)]
pub struct ToolDemoOptions<'a> {
    pub generation_defaults: &'a GenerationParams,
    pub system_prompt: Option<&'a str>,
    pub thinking: Option<ThinkingMode>,
}

pub async fn run_tool_demo_with_options(
    chat: &impl ChatModel,
    options: ToolDemoOptions<'_>,
) -> Result<()> {
    println!("\n--- tool calling ---");

    let tools = demo_tools();
    let tool_specs = tools.specs().context("collect demo tool specs")?;
    let generation_params = tool_demo_generation_params(options.generation_defaults);
    println!("tool-demo-effective-params: {generation_params:?}");
    println!("tool-demo-thinking: {:?}", options.thinking);

    let mut system_prompt = String::new();
    if let Some(prompt) = options.system_prompt {
        system_prompt.push_str(prompt);
        system_prompt.push_str("\n\n");
    }
    system_prompt.push_str(
        "Use tools when they are relevant. Make exactly one tool call per assistant turn. Start by calling get_weather for Seattle only. For get_weather, pass units as the lowercase string `fahrenheit`. After each weather result, call get_weather for the next city until Seattle, Portland, and San Francisco are complete. Then call evaluate_math_expression for the average. Do not calculate averages mentally. CEL requires matching numeric types, so divide decimal temperature values by 3.0, not 3. After all tool results are available, answer in one concise sentence.",
    );
    let mut messages = vec![
        ChatMessage::new(ChatRole::System, system_prompt),
        ChatMessage::new(
            ChatRole::User,
            "Calculate the average current fahrenheit temperature for Seattle, Portland, and San Francisco. Use get_weather once for each city. After the weather results are available, call evaluate_math_expression to calculate the average with a CEL expression that uses 3.0 as the divisor.",
        ),
    ];

    let mut seen_tools = BTreeSet::new();
    let mut final_response = None;

    for round in 1..=4 {
        let tool_request_started_at = Instant::now();
        let response = chat
            .generate(ChatRequest {
                messages: messages.clone(),
                params: generation_params.clone(),
                tools: tool_specs.clone(),
                tool_choice: Some(ToolChoice::Auto),
                thinking: options.thinking,
            })
            .await
            .with_context(|| format!("tool-call round {round} generation should succeed"))?;
        let tool_request_latency = tool_request_started_at.elapsed();

        println!("tool-round: {round}");
        println!("tool-request-response: {}", response.content);
        print_thinking_trace("tool-request", &response.reasoning);
        println!("tool-call-count: {}", response.tool_calls.len());
        println!(
            "tool-request-latency-ms: {:.2}",
            tool_request_latency.as_secs_f64() * 1000.0
        );

        if response.tool_calls.is_empty() {
            final_response = Some(response);
            break;
        }

        messages.push(ChatMessage::assistant_tool_calls(
            response.tool_calls.clone(),
        ));
        for call in response.tool_calls {
            println!("tool-call-id: {}", call.id);
            println!("tool-call-name: {}", call.name);
            println!("tool-call-args: {}", call.arguments.raw_json_str());
            seen_tools.insert(call.name.clone());

            let tool_message = match tools
                .dispatch(call)
                .await
                .context("execute model-requested tool")?
            {
                ToolDispatch::Handled(message) => message,
                ToolDispatch::NotMine(call) => {
                    // Future per #284: iterate mcp_servers here.
                    // for server in &mcp_servers { if server.owns(&call.name) { ... } }
                    bail!("unknown tool: {}", call.name);
                }
            };
            for part in &tool_message.content {
                if let ContentPart::Text(text) = part {
                    println!("tool-result: {text}");
                }
            }
            messages.push(tool_message);
        }
    }

    ensure!(
        seen_tools.contains(&ToolName::new("get_weather").expect("valid tool name")),
        "model did not call get_weather"
    );
    ensure!(
        seen_tools.contains(&ToolName::new("evaluate_math_expression").expect("valid tool name")),
        "model did not call evaluate_math_expression"
    );

    let final_response = match final_response {
        Some(response) => response,
        None => {
            let final_started_at = Instant::now();
            let response = chat
                .generate(ChatRequest {
                    messages,
                    params: generation_params,
                    tools: tool_specs,
                    tool_choice: Some(ToolChoice::None),
                    thinking: options.thinking,
                })
                .await
                .context("final answer after tool results should succeed")?;
            let final_latency = final_started_at.elapsed();
            println!(
                "tool-final-latency-ms: {:.2}",
                final_latency.as_secs_f64() * 1000.0
            );
            response
        }
    };

    if !final_response.tool_calls.is_empty() {
        bail!("model returned tool calls after the tool-call round limit");
    }

    println!("tool-final-response: {}", final_response.content);
    print_thinking_trace("tool-final", &final_response.reasoning);

    Ok(())
}

pub fn tool_demo_generation_params(defaults: &GenerationParams) -> GenerationParams {
    let mut params = GenerationParams {
        max_tokens: Some(192),
        ..Default::default()
    }
    .with_defaults(defaults);
    if params.temperature.is_none() {
        params.temperature = Some(0.2);
    }
    params
}

fn print_thinking_trace(label: &str, reasoning: &Option<String>) {
    match reasoning
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        Some(trace) => println!("{label}-thinking-trace: {trace}"),
        None => println!("{label}-thinking-trace: none"),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn evaluates_medium_complexity_expression() {
        let output = evaluate_math_expression(MathExpressionArgs {
            expression: "(48.0 * 7.0 - 13.0) / 5.0 + math.sqrt(144)".to_string(),
        })
        .await
        .expect("expression should evaluate");

        assert_eq!(output.formatted, "76.6");
        assert!((output.value - 76.6).abs() < f64::EPSILON);
    }
}
