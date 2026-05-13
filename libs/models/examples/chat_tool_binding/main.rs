use anyhow::{Context, Result};
use motlie_model::{
    ChatMessage, ChatRequest, ChatRole, ContentPart, ToolCall, ToolChoice, ToolError, ToolRegistry,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
    units: TemperatureUnits,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum TemperatureUnits {
    Celsius,
    Fahrenheit,
}

#[derive(Debug, Serialize)]
struct WeatherOutput {
    city: String,
    temperature: f32,
    units: TemperatureUnits,
    summary: String,
}

async fn get_weather(args: WeatherArgs) -> Result<WeatherOutput, ToolError> {
    Ok(WeatherOutput {
        city: args.city,
        temperature: match args.units {
            TemperatureUnits::Celsius => 22.0,
            TemperatureUnits::Fahrenheit => 72.0,
        },
        units: args.units,
        summary: "clear".to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AddArgs {
    left: i64,
    right: i64,
}

#[derive(Debug, Serialize)]
struct AddOutput {
    value: i64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut registry = ToolRegistry::new();
    registry
        .insert_fn(
            "get_weather",
            "Return a current weather summary for a city.",
            get_weather,
        )
        .context("register existing function tool")?
        .insert_fn(
            "add",
            "Add two signed integers.",
            |args: AddArgs| async move {
                Ok(AddOutput {
                    value: args.left + args.right,
                })
            },
        )
        .context("register closure tool")?;

    let request = ChatRequest {
        messages: vec![ChatMessage::text(
            ChatRole::User,
            "What is the weather in Seattle, and what is 40 + 2?",
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
            "call-weather",
            "get_weather",
            &WeatherArgs {
                city: "Seattle".to_string(),
                units: TemperatureUnits::Fahrenheit,
            },
        )
        .context("serialize weather tool call")?,
        ToolCall::from_serializable_args("call-add", "add", &AddArgs { left: 40, right: 2 })
            .context("serialize add tool call")?,
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
