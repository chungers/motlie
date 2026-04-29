use std::process::Command;

use anyhow::{Context, Result};

pub(crate) fn maybe_run_forcecommand_bypass() -> Result<Option<i32>> {
    let original = match std::env::var("SSH_ORIGINAL_COMMAND") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return Ok(None),
    };

    if std::env::var("MOTLIE_MMUX_BYPASS").ok().as_deref() == Some("1") {
        let status = Command::new("sh")
            .arg("-lc")
            .arg(original)
            .status()
            .context("run SSH_ORIGINAL_COMMAND bypass")?;
        return Ok(Some(shell_status(&status)));
    }

    eprintln!(
        "mmux: SSH_ORIGINAL_COMMAND is disabled for this account; set MOTLIE_MMUX_BYPASS=1 to delegate explicitly"
    );
    Ok(Some(126))
}

fn shell_status(status: &std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(code) = status.code() {
            code
        } else if let Some(signal) = status.signal() {
            128 + signal
        } else {
            1
        }
    }
    #[cfg(not(unix))]
    {
        status.code().unwrap_or(1)
    }
}
