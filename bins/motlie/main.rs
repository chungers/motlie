use clap::{Parser, Subcommand};
use tracing_subscriber::FmtSubscriber;

mod db;

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(Parser)]
#[clap(author = "chunger", version, about = "Motlie CLI utility")]
#[clap(propagate_version = true)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Database inspection commands
    Db(db::Command),
}

fn main() {
    let subscriber = FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber).expect("setting the subscriber failed");

    let cli = Cli::parse();

    tracing::info!("starting");

    match cli.command {
        Commands::Db(args) => {
            db::run(&args);
        }
    }
}
