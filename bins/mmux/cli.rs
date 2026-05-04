use clap::Parser;
use crossterm::terminal::size as terminal_size;

use crate::model::LayoutMode;

#[derive(Debug, Clone, Parser)]
#[command(name = "mmux")]
#[command(about = "Select, preview, monitor, and attach tmux sessions")]
pub(crate) struct Cli {
    /// Force portrait layout instead of auto-detecting from the current PTY.
    #[arg(short = 'p', long, conflicts_with = "landscape")]
    pub(crate) portrait: bool,
    /// Force landscape layout instead of auto-detecting from the current PTY.
    #[arg(short = 'l', long, conflicts_with = "portrait")]
    pub(crate) landscape: bool,
    /// Print the selected session name for shell-script integration instead of attaching.
    #[arg(long)]
    pub(crate) script: bool,
    /// Optional additional SSH URI target(s). Localhost is always included;
    /// any URI activates multi-host mode with an aggregated activity-sorted
    /// list across hosts and a per-row host marker.
    pub(crate) ssh_uris: Vec<String>,
}

impl Cli {
    pub(crate) fn forced_layout(&self) -> Option<LayoutMode> {
        if self.portrait {
            Some(LayoutMode::Portrait)
        } else if self.landscape {
            Some(LayoutMode::Normal)
        } else {
            None
        }
    }
}

pub(crate) fn select_layout(force: Option<LayoutMode>) -> LayoutMode {
    if let Some(layout) = force {
        return layout;
    }
    match terminal_size() {
        Ok((columns, rows)) if is_portrait_pty(columns, rows) => LayoutMode::Portrait,
        _ => LayoutMode::Normal,
    }
}

pub(crate) fn is_portrait_pty(columns: u16, rows: u16) -> bool {
    rows > 0 && (columns as u32) <= (rows as u32).saturating_mul(4)
}
