use anyhow::{bail, ensure, Context, Result};
use cel_cxx::{Activation, Env, Value};
use motlie_model::{
    ChatMessage, ChatModel, ChatRequest, ChatRole, ContentPart, GenerationParams, ToolChoice,
    ToolName,
};
use motlie_models::{ToolError, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeatherArgs {
    pub city: String,
    pub units: TemperatureUnits,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureUnits {
    Celsius,
    Fahrenheit,
}

#[derive(Debug, Serialize)]
pub struct WeatherOutput {
    pub city: String,
    pub temperature: f32,
    pub units: TemperatureUnits,
    pub summary: String,
}

pub async fn get_weather(args: WeatherArgs) -> std::result::Result<WeatherOutput, ToolError> {
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

pub async fn evaluate_math_expression(
    args: MathExpressionArgs,
) -> std::result::Result<MathExpressionOutput, ToolError> {
    let expression = args.expression.trim();
    if expression.is_empty() {
        return Err(ToolError::execution("expression cannot be empty"));
    }
    if expression.len() > 512 {
        return Err(ToolError::execution(
            "expression is too long; keep examples under 512 bytes",
        ));
    }

    let env = Env::builder()
        .with_ext_math(true)
        .build()
        .map_err(|err| ToolError::execution(format!("failed to build CEL environment: {err}")))?;
    let program = env
        .compile(expression)
        .map_err(|err| ToolError::execution(format!("failed to compile CEL expression: {err}")))?;
    let value = program
        .evaluate(&Activation::new())
        .map_err(|err| ToolError::execution(format!("failed to evaluate CEL expression: {err}")))?;
    let value = match value {
        Value::Int(value) => value as f64,
        Value::Uint(value) => value as f64,
        Value::Double(value) => value,
        other => {
            return Err(ToolError::execution(format!(
                "CEL expression evaluated to non-numeric value: {other}"
            )));
        }
    };
    if !value.is_finite() {
        return Err(ToolError::execution(
            "expression evaluated to a non-finite number",
        ));
    }

    Ok(MathExpressionOutput {
        expression: expression.to_string(),
        value,
        formatted: trim_float(format!("{value:.6}")),
        engine: "cel-cxx",
    })
}

pub fn register_demo_tools(registry: &mut ToolRegistry) -> std::result::Result<(), ToolError> {
    registry
        .insert_fn(
            "get_weather",
            "Return a current weather summary for a city.",
            get_weather,
        )?
        .insert_fn(
            "evaluate_math_expression",
            "Evaluate a CEL arithmetic expression with parentheses, numeric operators, conditionals, and math.* functions. Use matching numeric types in division, for example divide decimal values by 3.0.",
            evaluate_math_expression,
        )?;
    Ok(())
}

pub async fn run_tool_demo(chat: &impl ChatModel) -> Result<()> {
    println!("\n--- tool calling ---");

    let mut registry = ToolRegistry::new();
    register_demo_tools(&mut registry).context("register demo tools")?;

    let tools = registry.specs();
    let mut messages = vec![
        ChatMessage::new(
            ChatRole::System,
            "Use tools when they are relevant. Make exactly one tool call per assistant turn. Start by calling get_weather for Seattle only. After each weather result, call get_weather for the next city until Seattle, Portland, and San Francisco are complete. Then call evaluate_math_expression for the average. Do not calculate averages mentally. CEL requires matching numeric types, so divide decimal temperature values by 3.0, not 3. After all tool results are available, answer in one concise sentence.",
        ),
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
                params: tool_demo_generation_params(),
                tools: tools.clone(),
                tool_choice: Some(ToolChoice::Auto),
                ..Default::default()
            })
            .await
            .with_context(|| format!("tool-call round {round} generation should succeed"))?;
        let tool_request_latency = tool_request_started_at.elapsed();

        println!("tool-round: {round}");
        println!("tool-request-response: {}", response.content);
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

            let tool_message = registry
                .call_to_message(call)
                .await
                .context("execute model-requested tool")?;
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
                    params: tool_demo_generation_params(),
                    tools,
                    tool_choice: Some(ToolChoice::None),
                    ..Default::default()
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

    Ok(())
}

pub fn tool_demo_generation_params() -> GenerationParams {
    GenerationParams {
        max_tokens: Some(192),
        temperature: Some(0.2),
        ..Default::default()
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
