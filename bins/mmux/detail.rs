use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use motlie_tmux::{
    has_visible_text, CaptureNormalizeMode, CaptureOptions, CaptureResult, HostHandle, PaneAddress,
    ScrollbackQuery,
};

use crate::consts::DEFAULT_DETAIL_LINES;
use crate::model::{HostId, SelectedSession};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetailMode {
    Sample,
    Monitor,
}

pub(crate) trait SessionDetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()>;

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String>;

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String>;

    async fn deactivate(&mut self) -> Result<()>;

    fn mode(&self) -> DetailMode;
}

#[derive(Debug, Default)]
pub(crate) struct SampleDetailSource;

impl SessionDetailSource for SampleDetailSource {
    async fn activate(&mut self, _host: &HostHandle, _session: &SelectedSession) -> Result<()> {
        Ok(())
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        let Some(target) = host.session_by_id(session.id()).await? else {
            return Ok(format!("session {} disappeared", session.name()));
        };
        target
            .sample_text_with_options(
                &ScrollbackQuery::LastLines(DEFAULT_DETAIL_LINES),
                &CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable),
                None,
            )
            .await
            .map(|capture| capture.text)
            .context("sample selected session")
    }

    async fn fetch_older(
        &mut self,
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
            .context("fetch older sample lines")
    }

    async fn deactivate(&mut self) -> Result<()> {
        Ok(())
    }

    fn mode(&self) -> DetailMode {
        DetailMode::Sample
    }
}

pub(crate) struct MonitorDetailSource {
    pub(crate) session_id: Option<String>,
    pub(crate) host_id: Option<HostId>,
}

impl MonitorDetailSource {
    pub(crate) fn new() -> Self {
        Self {
            session_id: None,
            host_id: None,
        }
    }
}

impl SessionDetailSource for MonitorDetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()> {
        self.deactivate().await?;
        if host.session_by_id(session.id()).await?.is_none() {
            return Err(anyhow!("session {} disappeared", session.name()));
        }
        self.session_id = Some(session.id().to_string());
        self.host_id = Some(session.host_id.clone());
        Ok(())
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        let Some(target) = host.session_by_id(session.id()).await? else {
            return Ok(format!("session {} disappeared", session.name()));
        };
        let panes = target
            .capture_all_with_options(&CaptureOptions::with_mode(
                CaptureNormalizeMode::ScreenStable,
            ))
            .await
            .context("capture monitored session screen")?;
        Ok(render_screen_capture(session.name(), panes))
    }

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String> {
        let mut sample = SampleDetailSource;
        sample
            .fetch_older(host, session, older_than_lines, count)
            .await
    }

    async fn deactivate(&mut self) -> Result<()> {
        self.session_id = None;
        self.host_id = None;
        Ok(())
    }

    fn mode(&self) -> DetailMode {
        DetailMode::Monitor
    }
}

fn render_screen_capture(session_name: &str, panes: HashMap<PaneAddress, CaptureResult>) -> String {
    if panes.is_empty() {
        return "(no visible content)\n".to_string();
    }

    let mut pane_list: Vec<_> = panes.into_iter().collect();
    pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

    if pane_list.len() == 1 {
        let text = pane_list.remove(0).1.text;
        if has_visible_text(&text) {
            return text;
        }
        return "(no visible content)\n".to_string();
    }

    let mut rendered = String::new();
    for (addr, result) in pane_list {
        if !has_visible_text(&result.text) {
            continue;
        }
        rendered.push_str(&format!("--- {}({}) ---\n", session_name, addr.pane_id));
        rendered.push_str(&result.text);
        if !result.text.ends_with('\n') {
            rendered.push('\n');
        }
    }

    if rendered.is_empty() {
        "(no visible content)\n".to_string()
    } else {
        rendered
    }
}

pub(crate) enum DetailSource {
    Sample(SampleDetailSource),
    Monitor(Box<MonitorDetailSource>),
}

impl DetailSource {
    pub(crate) fn sample() -> Self {
        Self::Sample(SampleDetailSource)
    }

    pub(crate) fn monitor() -> Self {
        Self::Monitor(Box::new(MonitorDetailSource::new()))
    }

    pub(crate) fn monitored_session_id(&self) -> Option<&str> {
        match self {
            DetailSource::Sample(_) => None,
            DetailSource::Monitor(source) => source.session_id.as_deref(),
        }
    }

    pub(crate) fn monitored_host_id(&self) -> Option<&HostId> {
        match self {
            DetailSource::Sample(_) => None,
            DetailSource::Monitor(source) => source.host_id.as_ref(),
        }
    }
}

impl SessionDetailSource for DetailSource {
    async fn activate(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<()> {
        match self {
            DetailSource::Sample(source) => source.activate(host, session).await,
            DetailSource::Monitor(source) => source.activate(host, session).await,
        }
    }

    async fn render(&mut self, host: &HostHandle, session: &SelectedSession) -> Result<String> {
        match self {
            DetailSource::Sample(source) => source.render(host, session).await,
            DetailSource::Monitor(source) => source.render(host, session).await,
        }
    }

    async fn fetch_older(
        &mut self,
        host: &HostHandle,
        session: &SelectedSession,
        older_than_lines: usize,
        count: usize,
    ) -> Result<String> {
        match self {
            DetailSource::Sample(source) => {
                source
                    .fetch_older(host, session, older_than_lines, count)
                    .await
            }
            DetailSource::Monitor(source) => {
                source
                    .fetch_older(host, session, older_than_lines, count)
                    .await
            }
        }
    }

    async fn deactivate(&mut self) -> Result<()> {
        match self {
            DetailSource::Sample(source) => source.deactivate().await,
            DetailSource::Monitor(source) => source.deactivate().await,
        }
    }

    fn mode(&self) -> DetailMode {
        match self {
            DetailSource::Sample(source) => source.mode(),
            DetailSource::Monitor(source) => source.mode(),
        }
    }
}
