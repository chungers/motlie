mod cli;
mod daemon;
mod jsonl;
mod protocol;
mod state;
mod tags;
mod target;
mod timeline;

use clap::Parser;

use cli::{Cli, Command, DaemonCommand};

#[tokio::main]
async fn main() {
    let code = match run().await {
        Ok(code) => code,
        Err(err) => {
            let record = jsonl::error("fatal", err.to_string());
            let _ = jsonl::print_records(&[record]);
            1
        }
    };
    std::process::exit(code);
}

async fn run() -> anyhow::Result<i32> {
    let cli = Cli::parse();
    let socket = cli.socket_path();

    match cli.command {
        Command::Daemon(DaemonCommand::Start(args)) => {
            if args.foreground {
                daemon::run_foreground(socket).await?;
            } else {
                let records = daemon::start_background(&socket).await?;
                jsonl::print_records(&records)?;
            }
        }
        command => {
            let request = command.into_request()?;
            match daemon::send_request(&socket, &request).await {
                Ok(records) => jsonl::print_records(&records)?,
                Err(err) => {
                    let record = jsonl::error(
                        "daemon_unreachable",
                        format!("{err}; start the daemon or provide --socket"),
                    );
                    jsonl::print_records(&[record])?;
                    return Ok(1);
                }
            }
        }
    }

    Ok(0)
}
