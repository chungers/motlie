mod attach;
mod build_info;
mod cli;
mod client;
mod daemon;
mod jsonl;
mod protocol;
mod skills;
mod state;
mod tags;
mod timeline;

use clap::Parser;
use serde_json::Value;

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
        Command::Attach(args) => return attach::run(&socket, args).await,
        Command::Daemon(DaemonCommand::Start(args)) => {
            let foreground = args.foreground;
            let mount = args.mount;
            if foreground {
                daemon::run_foreground(socket, mount).await?;
            } else {
                let records = daemon::start_background(&socket, mount.as_deref()).await?;
                jsonl::print_records(&records)?;
            }
        }
        command => {
            let request = command.into_request()?;
            match client::send_request(&socket, &request).await {
                Ok(records) => print_client_records(&records)?,
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

fn print_client_records(records: &[Value]) -> anyhow::Result<()> {
    for record in records {
        if record.get("type").and_then(Value::as_str) == Some("events_readable") {
            if let Some(text) = record.get("text").and_then(Value::as_str) {
                println!("{text}");
                continue;
            }
        }
        println!("{}", serde_json::to_string(record)?);
    }
    Ok(())
}
