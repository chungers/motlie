use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::result::EvalDepth;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSummary {
    pub id: String,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub schema_version: u32,
    pub id: String,
    pub summary: String,
    #[serde(default = "default_depth")]
    pub depth: EvalDepth,
    pub bundle_filter: BundleFilter,
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileConfig>,
    #[serde(flatten)]
    pub kind: ScenarioKind,
}

impl Scenario {
    pub fn capability(&self) -> CapabilityName {
        self.kind.capability()
    }

    pub fn embeddings(&self) -> Option<&EmbeddingsScenario> {
        match &self.kind {
            ScenarioKind::Embeddings(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn chat(&self) -> Option<&ChatScenario> {
        match &self.kind {
            ScenarioKind::Chat(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn tool_use(&self) -> Option<&ToolUseScenario> {
        match &self.kind {
            ScenarioKind::ToolUse(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn asr(&self) -> Option<&AsrScenario> {
        match &self.kind {
            ScenarioKind::Asr(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn tts(&self) -> Option<&TtsScenario> {
        match &self.kind {
            ScenarioKind::Tts(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn perf(&self) -> Option<&PerfScenario> {
        match &self.kind {
            ScenarioKind::Perf(scenario) => Some(scenario),
            _ => None,
        }
    }

    pub fn gates_for_profile(&self, profile_name: &str) -> Option<&ProfileGates> {
        self.profiles
            .get(profile_name)
            .and_then(|profile| profile.gates.as_ref())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "capability", rename_all = "snake_case")]
pub enum ScenarioKind {
    Embeddings(EmbeddingsScenario),
    Chat(ChatScenario),
    ToolUse(ToolUseScenario),
    Asr(AsrScenario),
    Tts(TtsScenario),
    Perf(PerfScenario),
}

impl ScenarioKind {
    pub fn capability(&self) -> CapabilityName {
        match self {
            Self::Embeddings(_) => CapabilityName::Embeddings,
            Self::Chat(_) => CapabilityName::Chat,
            Self::ToolUse(_) => CapabilityName::ToolUse,
            Self::Asr(_) => CapabilityName::Asr,
            Self::Tts(_) => CapabilityName::Tts,
            Self::Perf(_) => CapabilityName::Perf,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityName {
    Embeddings,
    Chat,
    ToolUse,
    Asr,
    Tts,
    Perf,
}

impl CapabilityName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::Chat => "chat",
            Self::ToolUse => "tool_use",
            Self::Asr => "asr",
            Self::Tts => "tts",
            Self::Perf => "perf",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapabilityName {
    Chat,
    Completion,
    Embeddings,
    Speech,
    ToolUse,
    Transcription,
    Vision,
    VoiceClone,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BundleFilter {
    pub capability: CapabilityName,
    #[serde(default)]
    pub required_capabilities: Vec<ModelCapabilityName>,
    #[serde(default)]
    pub backend: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsScenario {
    pub input: EmbeddingsInput,
    pub assertions: EmbeddingsAssertions,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsInput {
    pub custom_text: String,
    pub similar_a: String,
    pub similar_b: String,
    pub dissimilar_a: String,
    pub dissimilar_b: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsAssertions {
    pub min_embedding_dimensions: usize,
    pub similarity_order: SimilarityOrder,
    pub min_similarity_gap: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityOrder {
    SimilarGtDissimilar,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatScenario {
    pub input: ChatInput,
    pub assertions: ChatAssertions,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatInput {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub followup_prompt: Option<String>,
    pub completion_prompt: Option<String>,
    pub tool_prompt: Option<String>,
    pub tool_name: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatAssertions {
    pub min_response_chars: Option<usize>,
    pub min_followup_response_chars: Option<usize>,
    pub min_completion_chars: Option<usize>,
    pub min_tool_calls: Option<usize>,
    #[serde(default)]
    pub required_substrings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolUseScenario {
    pub input: ToolUseInput,
    pub assertions: ToolUseAssertions,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolUseInput {
    pub prompt: String,
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    pub expected_tool: Option<String>,
    pub expected_argument_key: Option<String>,
    pub expected_argument_value: Option<String>,
    #[serde(default = "default_tool_rounds")]
    pub max_rounds: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

fn default_tool_rounds() -> u32 {
    2
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolUseAssertions {
    pub min_tool_calls: Option<usize>,
    #[serde(default)]
    pub required_tools: Vec<String>,
    #[serde(default)]
    pub required_final_substrings: Vec<String>,
    #[serde(default)]
    pub cel: Vec<CelAssertionConfig>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CelAssertionConfig {
    pub name: String,
    pub expression: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AsrScenario {
    pub input: AsrInput,
    pub assertions: AsrAssertions,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AsrInput {
    pub audio: String,
    pub reference_transcript: Option<String>,
    pub language: Option<String>,
    pub streaming_chunk_ms: Option<u64>,
    #[serde(default = "default_perf_iterations")]
    pub iterations: u64,
    #[serde(default)]
    pub warmup_iterations: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AsrAssertions {
    pub min_transcript_chars: Option<usize>,
    pub max_word_error_rate: Option<f64>,
    #[serde(default)]
    pub required_substrings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TtsScenario {
    pub input: TtsInput,
    pub assertions: TtsAssertions,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsInput {
    pub text: String,
    pub speaking_rate: Option<f32>,
    #[serde(default = "default_perf_iterations")]
    pub iterations: u64,
    #[serde(default)]
    pub warmup_iterations: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TtsAssertions {
    pub min_audio_duration_ms: Option<u64>,
    pub min_sample_count: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerfScenario {
    pub input: PerfInput,
    pub assertions: PerfAssertions,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerfInput {
    pub workload: String,
    pub prompt: Option<String>,
    pub dataset: Option<String>,
    #[serde(default = "default_perf_iterations")]
    pub iterations: u64,
    #[serde(default)]
    pub warmup_iterations: u64,
}

fn default_perf_iterations() -> u64 {
    5
}

fn default_depth() -> EvalDepth {
    EvalDepth::Smoke
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfAssertions {
    pub min_successful_iterations: Option<u64>,
    pub max_mean_latency_ms: Option<f64>,
    pub max_p95_latency_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetricsConfig {
    #[serde(default)]
    pub capture_startup_ms: bool,
    #[serde(default)]
    pub capture_request_latency: bool,
    #[serde(default)]
    pub capture_embedding_dimensions: bool,
    #[serde(default)]
    pub capture_vectors_per_second: bool,
    #[serde(default)]
    pub capture_peak_rss: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub gates: Option<ProfileGates>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileGates {
    pub max_process_swap_delta_bytes: Option<u64>,
}

pub fn list_scenarios(eval_root: &Path) -> Result<Vec<ScenarioSummary>> {
    let scenario_dir = eval_root.join("scenarios");
    let mut scenarios = Vec::new();

    for entry in fs::read_dir(&scenario_dir)
        .with_context(|| format!("failed to read `{}`", scenario_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        scenarios.push(ScenarioSummary {
            id: id.to_owned(),
            path: path.display().to_string(),
        });
    }

    scenarios.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(scenarios)
}

pub fn load_scenario(eval_root: &Path, scenario_id: &str) -> Result<Scenario> {
    let path = scenario_path(eval_root, scenario_id);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read `{}`", path.display()))?;
    let scenario = toml::from_str::<Scenario>(&raw)
        .with_context(|| format!("failed to parse `{}`", path.display()))?;
    Ok(scenario)
}

fn scenario_path(eval_root: &Path, scenario_id: &str) -> PathBuf {
    eval_root
        .join("scenarios")
        .join(format!("{scenario_id}.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_repo_scenarios() {
        let eval_root = repo_eval_root();

        let scenarios = list_scenarios(&eval_root).unwrap();

        assert!(scenarios
            .iter()
            .any(|scenario| scenario.id == "embeddings_similarity"));
    }

    #[test]
    fn parses_embeddings_similarity() {
        let scenario = load_scenario(&repo_eval_root(), "embeddings_similarity").unwrap();

        assert_eq!(scenario.capability(), CapabilityName::Embeddings);
        assert_eq!(scenario.depth, EvalDepth::Smoke);
        assert_eq!(
            scenario.bundle_filter.capability,
            CapabilityName::Embeddings
        );
        let embeddings = scenario.embeddings().unwrap();
        assert_eq!(
            embeddings.assertions.similarity_order,
            SimilarityOrder::SimilarGtDissimilar
        );
        assert!(scenario.profiles.contains_key("apple-metal"));
        assert!(scenario.profiles.contains_key("dgx-spark"));
        assert!(scenario.profiles.contains_key("cuda-workstation"));
        assert_eq!(
            scenario
                .profiles
                .get("apple-metal")
                .and_then(|profile| profile.gates.as_ref())
                .and_then(|gates| gates.max_process_swap_delta_bytes),
            None
        );
    }

    #[test]
    fn parses_chat_scenario_shape() {
        let raw = r#"
schema_version = 1
id = "chat_smoke"
capability = "chat"
summary = "Minimal chat smoke scenario."

[bundle_filter]
capability = "chat"
backend = ["MistralRs"]
required_capabilities = ["completion", "tool_use"]

[input]
prompt = "Say hello."
completion_prompt = "Complete this sentence."
tool_prompt = "Call get_weather."

[assertions]
min_response_chars = 1
min_completion_chars = 1
min_tool_calls = 1

[metrics]
capture_startup_ms = true
capture_request_latency = true
"#;

        let scenario = toml::from_str::<Scenario>(raw).unwrap();

        assert_eq!(scenario.capability(), CapabilityName::Chat);
        assert_eq!(
            scenario.bundle_filter.required_capabilities,
            [
                ModelCapabilityName::Completion,
                ModelCapabilityName::ToolUse
            ]
        );
        match scenario.kind {
            ScenarioKind::Chat(chat) => {
                assert_eq!(chat.input.prompt, "Say hello.");
                assert_eq!(chat.assertions.min_response_chars, Some(1));
                assert_eq!(chat.assertions.min_completion_chars, Some(1));
            }
            other => panic!("expected chat scenario, got {other:?}"),
        }
    }

    #[test]
    fn parses_tool_use_scenario_shape() {
        let raw = r#"
schema_version = 1
id = "tool_use_smoke"
capability = "tool_use"
depth = "smoke"
summary = "Tool-use smoke scenario."

[bundle_filter]
capability = "tool_use"
required_capabilities = ["chat", "tool_use"]

[input]
prompt = "Use get_weather for Seattle."
tools = ["get_weather"]
expected_tool = "get_weather"
expected_argument_key = "city"
expected_argument_value = "Seattle"

[assertions]
min_tool_calls = 1
required_tools = ["get_weather"]
required_final_substrings = ["Seattle"]
[[assertions.cel]]
name = "weather_called"
expression = "tool_called('get_weather')"

[metrics]
capture_startup_ms = true
capture_request_latency = true
"#;

        let scenario = toml::from_str::<Scenario>(raw).unwrap();

        assert_eq!(scenario.capability(), CapabilityName::ToolUse);
        assert_eq!(scenario.depth, EvalDepth::Smoke);
        match scenario.kind {
            ScenarioKind::ToolUse(tool_use) => {
                assert_eq!(tool_use.input.expected_tool.as_deref(), Some("get_weather"));
                assert_eq!(tool_use.assertions.cel.len(), 1);
            }
            other => panic!("expected tool_use scenario, got {other:?}"),
        }
    }

    #[test]
    fn parses_perf_scenario_shape() {
        let raw = r#"
schema_version = 1
id = "bench_chat_startup"
capability = "perf"
summary = "Chat startup and steady-state latency benchmark."

[bundle_filter]
capability = "chat"

[input]
workload = "chat_generation"
prompt = "Say hello."
iterations = 2

[assertions]
min_successful_iterations = 2

[metrics]
capture_startup_ms = true
capture_request_latency = true
"#;

        let scenario = toml::from_str::<Scenario>(raw).unwrap();

        assert_eq!(scenario.capability(), CapabilityName::Perf);
        match scenario.kind {
            ScenarioKind::Perf(perf) => {
                assert_eq!(perf.input.iterations, 2);
                assert_eq!(perf.assertions.min_successful_iterations, Some(2));
            }
            other => panic!("expected perf scenario, got {other:?}"),
        }
    }

    #[test]
    fn parses_asr_scenario_shape() {
        let scenario = load_scenario(&repo_eval_root(), "asr_short_transcription").unwrap();

        assert_eq!(scenario.capability(), CapabilityName::Asr);
        match scenario.kind {
            ScenarioKind::Asr(asr) => {
                assert_eq!(asr.assertions.min_transcript_chars, Some(1));
                assert_eq!(asr.input.language.as_deref(), Some("en"));
                assert_eq!(asr.input.iterations, 3);
                assert_eq!(asr.input.warmup_iterations, 1);
            }
            other => panic!("expected ASR scenario, got {other:?}"),
        }
    }

    #[test]
    fn parses_tts_scenario_shape() {
        let scenario = load_scenario(&repo_eval_root(), "tts_synthesis_smoke").unwrap();

        assert_eq!(scenario.capability(), CapabilityName::Tts);
        match scenario.kind {
            ScenarioKind::Tts(tts) => {
                assert_eq!(tts.assertions.min_sample_count, Some(1));
                assert_eq!(tts.input.text, "Hello from Motlie.");
                assert_eq!(tts.input.iterations, 3);
                assert_eq!(tts.input.warmup_iterations, 1);
            }
            other => panic!("expected TTS scenario, got {other:?}"),
        }
    }

    fn repo_eval_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("bins/evals should live two levels below the repo root")
            .join("evals")
    }
}
