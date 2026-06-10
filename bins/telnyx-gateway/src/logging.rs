use std::fs::OpenOptions;
use std::path::Path;

use anyhow::Context;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

pub struct LoggingGuard {
    _guard: Option<WorkerGuard>,
}

pub fn init(log_file: Option<&Path>) -> anyhow::Result<LoggingGuard> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,motlie_telnyx_gateway=debug"));

    if let Some(path) = log_file {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create log directory {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open log file {}", path.display()))?;
        let (writer, guard) = tracing_appender::non_blocking(file);
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_ansi(false)
            .with_writer(writer)
            .try_init();
        return Ok(LoggingGuard {
            _guard: Some(guard),
        });
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
    Ok(LoggingGuard { _guard: None })
}
