use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use motlie_tmux::{
    CaptureNormalizeMode, HistoryOptions, HostHandle, KeySequence, SessionWatchHandle,
    SessionWatchOptions, Target,
};
use tokio::sync::Mutex;
use tokio::time::{self, Instant};

#[derive(Clone)]
pub struct TmuxBridge {
    target: Target,
    watch: Arc<Mutex<SessionWatchHandle>>,
    reply_timeout: Duration,
}

impl TmuxBridge {
    pub async fn new(target_spec: &str, reply_timeout: Duration) -> anyhow::Result<Self> {
        let normalized = normalize_target_spec(target_spec);
        let session_name = session_name_from_target(&normalized);
        let host = HostHandle::local();
        let watch = host
            .watch_session(
                &session_name,
                &SessionWatchOptions {
                    queue_capacity: 128,
                    history: HistoryOptions {
                        max_entries: 800,
                        max_render_chars: 80_000,
                        ..HistoryOptions::default()
                    },
                    normalize: CaptureNormalizeMode::ScreenStable,
                },
            )
            .await
            .with_context(|| format!("start motlie-tmux monitor for session {session_name}"))?;
        let target = host
            .resolve_target_str(&normalized)
            .await
            .with_context(|| format!("resolve tmux target {normalized}"))?
            .with_context(|| format!("tmux target not found: {normalized}"))?;
        Ok(Self {
            target,
            watch: Arc::new(Mutex::new(watch)),
            reply_timeout,
        })
    }

    pub async fn send_turn(&self, turn_id: &str, caller_text: &str) -> anyhow::Result<String> {
        let marker = end_marker(turn_id);
        let prompt = format!(
            "[motlie-call turn={turn_id}] Caller: {caller_text}\nReply once, then end with {marker}"
        );
        let baseline = self.watch.lock().await.render_text().await;
        let keys = KeySequence::literal(&prompt).then_enter();
        self.target
            .send_keys(&keys)
            .await
            .context("send caller turn to tmux")?;

        let deadline = Instant::now() + self.reply_timeout;
        loop {
            let rendered = self.watch.lock().await.render_text().await;
            if let Some(reply) =
                extract_reply_from_rendered_delta(&baseline, &rendered, &prompt, &marker)
            {
                return Ok(reply);
            }
            if Instant::now() >= deadline {
                anyhow::bail!("timed out waiting for tmux reply marker {marker}");
            }
            time::sleep(Duration::from_millis(250)).await;
        }
    }
}

pub fn normalize_target_spec(target_spec: &str) -> String {
    target_spec
        .strip_prefix("local:")
        .unwrap_or(target_spec)
        .to_string()
}

fn session_name_from_target(target: &str) -> String {
    target.split(':').next().unwrap_or(target).to_string()
}

fn end_marker(turn_id: &str) -> String {
    format!("[[motlie-call:end turn={turn_id}]]")
}

pub fn extract_reply_from_rendered_delta(
    baseline: &str,
    rendered: &str,
    prompt: &str,
    marker: &str,
) -> Option<String> {
    let delta = rendered.strip_prefix(baseline).unwrap_or(rendered);
    let delta = delta.replace(prompt, "");
    let marker_index = delta.find(marker)?;
    let before_marker = &delta[..marker_index];
    let reply = before_marker
        .lines()
        .filter(|line| !line.contains("[motlie-call turn="))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if reply.is_empty() {
        None
    } else {
        Some(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_marker_delimited_reply_from_history_delta() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!("{baseline}{prompt}\nThe answer is yes.\n{marker}\n");
        let reply = extract_reply_from_rendered_delta(baseline, &rendered, prompt, marker);
        assert_eq!(reply.as_deref(), Some("The answer is yes."));
    }
}
