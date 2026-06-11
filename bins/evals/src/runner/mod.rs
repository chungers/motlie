pub mod asr;
pub mod chat;
pub mod embeddings;
pub mod perf;
pub mod support;
pub mod tool_use;
pub mod tts;

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;

use crate::metrics::MetricsSampler;
use crate::platform::PlatformCollector;
use crate::report::OutputSink;
use crate::result::{AcceleratorSection, ChildBuildSection, CoverageSection, ResultRecord};
use crate::scenario::Scenario;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleSelection {
    pub bundle_id: String,
    pub selector: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileSelection {
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeFlags {
    pub command_line: Vec<String>,
    pub download_artifacts: bool,
    pub precision: Option<String>,
    pub artifact_quantization: Option<String>,
    pub quiet_backend_logs: bool,
    pub run_id: Option<String>,
}

pub struct RunContext {
    pub scenario: Scenario,
    pub bundle_selection: BundleSelection,
    pub profile: ProfileSelection,
    pub artifact_root: PathBuf,
    pub runtime_flags: RuntimeFlags,
    pub coverage: Option<CoverageSection>,
    pub accelerator: Option<AcceleratorSection>,
    pub child_build: Option<ChildBuildSection>,
    pub platform_collector: PlatformCollector,
    pub metrics_sampler: MetricsSampler,
    pub output_sink: OutputSink,
}

#[async_trait]
pub trait ScenarioRunner {
    async fn run(&self, context: RunContext) -> Result<ResultRecord>;
}
