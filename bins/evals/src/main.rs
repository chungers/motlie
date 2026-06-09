pub mod accelerator;
pub mod cli;
pub mod driver;
pub mod metrics;
pub mod platform;
pub mod report;
pub mod result;
pub mod runner;
pub mod scenario;
pub mod snapshot;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run(std::env::args()).await
}
