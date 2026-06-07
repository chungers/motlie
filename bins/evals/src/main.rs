mod cli;
pub mod metrics;
pub mod platform;
pub mod report;
pub mod result;
pub mod runner;
pub mod scenario;

fn main() -> anyhow::Result<()> {
    cli::run(std::env::args())
}
