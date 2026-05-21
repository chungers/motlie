use clap::Parser;
use crossterm::terminal::size as terminal_size;

use crate::model::LayoutMode;

#[derive(Debug, Clone, Parser)]
#[command(name = "mmux")]
#[command(about = "Select, live-preview, and attach tmux sessions")]
#[command(version)]
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
    /// Optional comma-separated host label overrides in configured-host order.
    /// Position 0 is localhost; SSH URI i uses position i+1. Empty entries
    /// keep the default discovered label.
    #[arg(long)]
    pub(crate) alias: Option<String>,
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

    pub(crate) fn host_alias_override(&self, host_index: usize) -> Option<&str> {
        self.alias
            .as_deref()?
            .split(',')
            .nth(host_index)
            .filter(|value| !value.is_empty())
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
