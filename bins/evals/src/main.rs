pub mod cli;
pub mod metrics;
pub mod platform;
pub mod report;
pub mod result;
pub mod runner;
pub mod scenario;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run(std::env::args()).await
}
