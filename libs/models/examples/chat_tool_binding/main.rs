use anyhow::{Context, Result};
use motlie_model::{ChatMessage, ChatRequest, ChatRole, ContentPart, ToolCall, ToolChoice};
use motlie_models::ToolRegistry;

#[allow(dead_code)]
#[path = "../tool_demo_support.rs"]
mod tool_demo_support;

#[tokio::main]
async fn main() -> Result<()> {
    let mut registry = ToolRegistry::new();
    registry
        .insert_fn(
            "get_weather",
            "Return a current weather summary for a city.",
            tool_demo_support::get_weather,
        )
        .context("register existing function tool")?
        .insert_fn(
            "evaluate_math_expression",
            "Evaluate a CEL arithmetic expression with parentheses, numeric operators, conditionals, and math.* functions.",
            |args: tool_demo_support::MathExpressionArgs| async move {
                tool_demo_support::evaluate_math_expression(args).await
            },
        )
        .context("register closure tool")?;

    let request = ChatRequest {
        messages: vec![ChatMessage::text(
            ChatRole::User,
            "What is Rust, and what is the average fahrenheit temperature across Seattle, Portland, and San Francisco?",
        )],
        tools: registry.specs(),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    println!("registered-tools: {}", request.tools.len());
    for tool in &request.tools {
        println!("tool: {}", tool.name);
        println!("schema: {}", tool.input_schema.as_json_str());
    }

    let model_calls = vec![
        ToolCall::from_serializable_args(
            "call-weather-seattle",
            "get_weather",
            &tool_demo_support::WeatherArgs {
                city: "Seattle".to_string(),
                units: tool_demo_support::TemperatureUnits::Fahrenheit,
            },
        )
        .context("serialize Seattle weather tool call")?,
        ToolCall::from_serializable_args(
            "call-weather-portland",
            "get_weather",
            &tool_demo_support::WeatherArgs {
                city: "Portland".to_string(),
                units: tool_demo_support::TemperatureUnits::Fahrenheit,
            },
        )
        .context("serialize Portland weather tool call")?,
        ToolCall::from_serializable_args(
            "call-weather-san-francisco",
            "get_weather",
            &tool_demo_support::WeatherArgs {
                city: "San Francisco".to_string(),
                units: tool_demo_support::TemperatureUnits::Fahrenheit,
            },
        )
        .context("serialize San Francisco weather tool call")?,
        ToolCall::from_serializable_args(
            "call-average",
            "evaluate_math_expression",
            &tool_demo_support::MathExpressionArgs {
                expression: "(72.0 + 68.0 + 64.0) / 3.0".to_string(),
            },
        )
        .context("serialize average math tool call")?,
    ];

    for call in model_calls {
        let assistant_turn = ChatMessage::assistant_tool_calls(vec![call.clone()]);
        let tool_turn = registry
            .call_to_message(call)
            .await
            .context("execute tool call")?;

        print_message("assistant", &assistant_turn);
        print_message("tool", &tool_turn);
    }

    Ok(())
}

fn print_message(label: &str, message: &ChatMessage) {
    println!("{}-role: {:?}", label, message.role);

    for call in &message.tool_calls {
        println!("{}-call-id: {}", label, call.id);
        println!("{}-call-name: {}", label, call.name);
        println!("{}-call-args: {}", label, call.arguments.raw_json_str());
    }

    for part in &message.content {
        if let ContentPart::Text(text) = part {
            println!("{}-content: {}", label, text);
        }
    }
}
