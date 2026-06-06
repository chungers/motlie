use anyhow::{Context, Result, bail};
use motlie_model::{
    ChatMessage, ChatRequest, ChatRole, ContentPart, GenerationParams, ToolCall, ToolChoice,
};
use motlie_models::{ToolDispatch, ToolList, tool_list};

#[allow(dead_code)]
#[path = "../tool_demo_support.rs"]
mod tool_demo_support;

#[tokio::main]
async fn main() -> Result<()> {
    let model_name = parse_model_arg()?;
    let tools = tool_list!(
        tool_demo_support::WeatherTool,
        tool_demo_support::EvaluateMathExpressionTool,
    );
    let recommended_params = exercise_spec_recommendations(model_name.as_deref())?;

    let request = ChatRequest {
        messages: vec![ChatMessage::text(
            ChatRole::User,
            "What is Rust, and what is the average fahrenheit temperature across Seattle, Portland, and San Francisco?",
        )],
        params: recommended_params,
        tools: tools.specs().context("collect tool specs")?,
        tool_choice: Some(ToolChoice::Auto),
        thinking: None,
    };

    println!("registered-tools: {}", request.tools.len());
    println!("request-generation-params: {:?}", request.params);
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
        let tool_turn = match tools.dispatch(call).await.context("execute tool call")? {
            ToolDispatch::Handled(message) => message,
            ToolDispatch::NotMine(call) => {
                // Future per #284: iterate mcp_servers here.
                // for server in &mcp_servers { if server.owns(&call.name) { ... } }
                anyhow::bail!("unknown tool: {}", call.name);
            }
        };

        print_message("assistant", &assistant_turn);
        print_message("tool", &tool_turn);
    }

    Ok(())
}

fn exercise_spec_recommendations(model_name: Option<&str>) -> Result<GenerationParams> {
    let spec = DemoRecommendedChatSpec::for_model(model_name)?;
    let effective = GenerationParams::default().with_defaults(&spec.recommended_generation_params);

    assert_eq!(effective.temperature, Some(1.0));
    assert_eq!(effective.top_p, Some(0.95));
    assert_eq!(
        spec.recommended_system_prompt,
        Some("You are Gemma, a helpful assistant.")
    );

    println!("spec-recommendation-source: {}", spec.source);
    println!("spec-recommended-temperature: {:?}", effective.temperature);
    println!("spec-recommended-top-p: {:?}", effective.top_p);
    println!(
        "spec-recommended-system-prompt: {:?}",
        spec.recommended_system_prompt
    );

    Ok(effective)
}

fn parse_model_arg() -> Result<Option<String>> {
    let mut model_name = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model" {
            model_name = Some(args.next().context("--model requires a model name")?);
        } else if let Some(model) = arg.strip_prefix("--model=") {
            model_name = Some(model.to_owned());
        } else if arg == "--help" || arg == "-h" {
            println!(
                "usage: cargo run -p motlie-models --example chat_tool_binding -- [--model=gemma4-e4b]"
            );
            std::process::exit(0);
        } else {
            bail!("unknown argument `{arg}`; use --model=gemma4-e4b");
        }
    }
    Ok(model_name)
}

struct DemoRecommendedChatSpec {
    source: &'static str,
    recommended_generation_params: GenerationParams,
    recommended_system_prompt: Option<&'static str>,
}

impl DemoRecommendedChatSpec {
    fn for_model(model_name: Option<&str>) -> Result<Self> {
        let requested = model_name.unwrap_or("gemma4-e4b");
        let normalized = requested.to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "e4b" | "gemma4-e4b" | "google/gemma4-e4b" => Ok(Self::gemma4_e4b()),
            other => bail!("unknown spec model `{other}`; use gemma4-e4b"),
        }
    }

    #[cfg(feature = "model-gemma4-e4b-gguf")]
    fn gemma4_e4b() -> Self {
        let spec = motlie_model_llama_cpp::LlamaCppTextSpec::gemma4_e4b();
        Self {
            source: "motlie_model_llama_cpp::LlamaCppTextSpec::gemma4_e4b",
            recommended_generation_params: spec.recommended_generation_params,
            recommended_system_prompt: spec.recommended_system_prompt,
        }
    }

    #[cfg(all(not(feature = "model-gemma4-e4b-gguf"), feature = "model-gemma4-e4b"))]
    fn gemma4_e4b() -> Self {
        let spec = motlie_model_mistral::MistralMultimodalSpec::gemma4_e4b();
        Self {
            source: "motlie_model_mistral::MistralMultimodalSpec::gemma4_e4b",
            recommended_generation_params: spec.recommended_generation_params,
            recommended_system_prompt: spec.recommended_system_prompt,
        }
    }

    #[cfg(all(
        not(feature = "model-gemma4-e4b-gguf"),
        not(feature = "model-gemma4-e4b")
    ))]
    fn gemma4_e4b() -> Self {
        Self {
            source: "local Gemma 4 E4B recommendation witness",
            recommended_generation_params: GenerationParams {
                temperature: Some(1.0),
                top_p: Some(0.95),
                ..Default::default()
            },
            recommended_system_prompt: Some("You are Gemma, a helpful assistant."),
        }
    }
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
