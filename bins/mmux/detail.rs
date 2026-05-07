use anyhow::{Context, Result};
use motlie_tmux::{CaptureNormalizeMode, CaptureOptions, HostHandle, ScrollbackQuery};

use crate::model::SelectedSession;

pub(crate) async fn render_live_preview(
    host: &HostHandle,
    session: &SelectedSession,
) -> Result<String> {
    let Some(target) = host.session_by_id(session.id()).await? else {
        return Ok(format!("session {} disappeared", session.name()));
    };
    target
        .capture_with_options(&CaptureOptions::with_mode(
            CaptureNormalizeMode::ScreenStable,
        ))
        .await
        .map(|capture| capture.text)
        .context("capture selected session active pane")
}

pub(crate) async fn fetch_older_lines(
    host: &HostHandle,
    session: &SelectedSession,
    older_than_lines: usize,
    count: usize,
) -> Result<String> {
    let Some(target) = host.session_by_id(session.id()).await? else {
        return Ok(String::new());
    };
    target
        .sample_text_with_options(
            &ScrollbackQuery::LinesRange {
                older_than_lines,
                count,
            },
            &CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable),
            None,
        )
        .await
        .map(|capture| capture.text)
        .context("fetch older detail lines")
}
