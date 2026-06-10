use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use motlie_agent::{
    Channel, ChannelConfig, ChannelManager, CoalescePolicy, DedupPolicy, EnqueueOptions,
    ManagedMessage, MessageSource, PasteMode, QueuedSubmission, QuietGuardPolicy, ResolvedSession,
    SessionKey, SubmitPolicy,
};
use motlie_tmux::{
    CaptureNormalizeMode, HistoryOptions, HostHandle, SessionWatchHandle, SessionWatchOptions,
};
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::time::{self, Instant};

const DEFAULT_INPUT_QUIET_FOR: Duration = Duration::from_secs(10);
const DEFAULT_INPUT_DELIVERY_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_INPUT_BACKOFF_INITIAL: Duration = Duration::from_millis(250);
const DEFAULT_INPUT_BACKOFF_MAX: Duration = Duration::from_secs(5);
const DEFAULT_TRAILING_ENTER_DELAY: Duration = Duration::from_millis(750);

#[derive(Clone)]
pub struct TmuxBridge {
    watch: Arc<Mutex<SessionWatchHandle>>,
    channel: Channel,
    reply_timeout: Duration,
    injection: KeystrokeInjectionConfig,
    send_lock: Arc<Mutex<()>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeystrokeInjectionConfig {
    pub input_quiet_for: Duration,
    pub input_delivery_timeout: Duration,
    pub input_backoff_initial: Duration,
    pub input_backoff_max: Duration,
    pub trailing_enter: bool,
    pub trailing_enter_delay: Duration,
}

impl Default for KeystrokeInjectionConfig {
    fn default() -> Self {
        Self {
            input_quiet_for: DEFAULT_INPUT_QUIET_FOR,
            input_delivery_timeout: DEFAULT_INPUT_DELIVERY_TIMEOUT,
            input_backoff_initial: DEFAULT_INPUT_BACKOFF_INITIAL,
            input_backoff_max: DEFAULT_INPUT_BACKOFF_MAX,
            trailing_enter: true,
            trailing_enter_delay: DEFAULT_TRAILING_ENTER_DELAY,
        }
    }
}

impl KeystrokeInjectionConfig {
    pub fn new(
        input_quiet_for: Duration,
        input_delivery_timeout: Duration,
        input_backoff_initial: Duration,
        input_backoff_max: Duration,
        trailing_enter: bool,
        trailing_enter_delay: Duration,
    ) -> Self {
        let input_backoff_initial = input_backoff_initial.max(Duration::from_millis(1));
        let input_backoff_max = input_backoff_max.max(input_backoff_initial);
        Self {
            input_quiet_for,
            input_delivery_timeout,
            input_backoff_initial,
            input_backoff_max,
            trailing_enter,
            trailing_enter_delay,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BridgeAbortToken {
    canceled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl BridgeAbortToken {
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }

    pub async fn canceled(&self) {
        if self.is_canceled() {
            return;
        }
        self.notify.notified().await;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BridgeTurnEvent {
    Partial(String),
    Final(String),
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct StableLineEmission {
    required_captures: usize,
    emitted_lines: usize,
    emitted_content: std::collections::BTreeSet<String>,
    last_lines: Vec<String>,
    stable_counts: Vec<usize>,
}

impl StableLineEmission {
    fn new(required_captures: usize) -> Self {
        Self {
            required_captures: required_captures.max(1),
            emitted_lines: 0,
            emitted_content: std::collections::BTreeSet::new(),
            last_lines: Vec::new(),
            stable_counts: Vec::new(),
        }
    }

    fn update(&mut self, lines: Vec<String>, final_marker_seen: bool) -> Vec<String> {
        let mut counts = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            let count = if self.last_lines.get(index) == Some(line) {
                self.stable_counts
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(1)
            } else {
                1
            };
            counts.push(count);
        }
        self.last_lines = lines.clone();
        self.stable_counts = counts;

        let stable_limit = if final_marker_seen {
            lines.len()
        } else {
            lines.len().saturating_sub(1)
        };
        let mut out = Vec::new();
        while self.emitted_lines < stable_limit
            && self
                .stable_counts
                .get(self.emitted_lines)
                .copied()
                .unwrap_or(0)
                >= self.required_captures
        {
            let line = lines[self.emitted_lines].clone();
            if self.emitted_content.insert(line.clone()) {
                out.push(line);
            }
            self.emitted_lines += 1;
        }
        out
    }

    fn remaining_text(&self) -> String {
        self.last_lines
            .iter()
            .skip(self.emitted_lines)
            .cloned()
            .collect::<Vec<_>>()
            .join(
                "
",
            )
            .trim()
            .to_string()
    }
}

impl TmuxBridge {
    pub async fn new_with_config(
        target_spec: &str,
        reply_timeout: Duration,
        injection: KeystrokeInjectionConfig,
    ) -> anyhow::Result<Self> {
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
        let channel_config = ChannelConfig {
            input_quiet_for: injection.input_quiet_for,
            ..ChannelConfig::default()
        };
        let channel_manager = ChannelManager::new(channel_config);
        let key = SessionKey::from_target("local", "local", &target);
        let channel =
            channel_manager.get_or_bind(ResolvedSession::new(key, host.clone(), target.clone()))?;
        Ok(Self {
            watch: Arc::new(Mutex::new(watch)),
            channel,
            reply_timeout,
            injection,
            send_lock: Arc::new(Mutex::new(())),
        })
    }

    pub async fn send_turn_streaming(
        &self,
        turn_id: &str,
        caller_text: &str,
        events: mpsc::Sender<BridgeTurnEvent>,
        abort: BridgeAbortToken,
    ) -> anyhow::Result<()> {
        let _send_guard = self.send_lock.lock().await;
        if abort.is_canceled() {
            anyhow::bail!("bridge turn canceled before delivery");
        }
        let marker = end_marker(turn_id);
        let prompt = format!(
            "[motlie-call turn={turn_id}] Caller: {caller_text}\nReply once, then end with {marker}"
        );
        let message = ManagedMessage::new(
            MessageSource::human(format!("telnyx.turn.{turn_id}")),
            prompt.clone(),
        )
        .with_paste_mode(PasteMode::Bracketed)
        .with_dedup(DedupPolicy::Unique)
        .with_coalesce(CoalescePolicy::Disabled);
        let submit = if self.injection.trailing_enter {
            SubmitPolicy {
                prompt_submit: true,
                settle: self.injection.trailing_enter_delay,
                retries: 1,
                retry_delay: Duration::from_millis(250),
                require_verification: false,
            }
        } else {
            SubmitPolicy::typing_only()
        };
        let queued = self
            .channel
            .enqueue_with_submission(
                message,
                EnqueueOptions {
                    submit,
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
            .context("enqueue caller turn through motlie-agent Channel")?;
        wait_for_channel_submission(
            &self.channel,
            queued,
            &abort,
            self.injection.input_delivery_timeout,
        )
        .await
        .context("deliver caller turn through motlie-agent Channel")?;

        time::sleep(Duration::from_millis(50)).await;
        let baseline = self.watch.lock().await.render_text().await;
        let deadline = Instant::now() + self.reply_timeout;
        let mut stable = StableLineEmission::new(2);
        loop {
            if abort.is_canceled() {
                anyhow::bail!("bridge turn canceled while waiting for reply");
            }
            let rendered = self.watch.lock().await.render_text().await;
            let final_marker_seen = rendered.lines().any(|line| line.trim() == marker);
            let lines = reply_lines_from_rendered_delta(&baseline, &rendered, &prompt, &marker);
            for line in stable.update(lines, final_marker_seen) {
                let text = line.trim();
                if !text.is_empty() {
                    let _ = events
                        .send(BridgeTurnEvent::Partial(format!("{text}\n")))
                        .await;
                }
            }
            if final_marker_seen {
                let final_text = stable.remaining_text();
                let _ = events.send(BridgeTurnEvent::Final(final_text)).await;
                return Ok(());
            }
            if Instant::now() >= deadline {
                anyhow::bail!("timed out waiting for tmux reply marker {marker}");
            }
            tokio::select! {
                _ = abort.canceled() => {
                    anyhow::bail!("bridge turn canceled while waiting for reply");
                }
                _ = time::sleep(Duration::from_millis(250)) => {}
            }
        }
    }
}

async fn wait_for_channel_submission(
    channel: &Channel,
    queued: QueuedSubmission,
    abort: &BridgeAbortToken,
    delivery_timeout: Duration,
) -> anyhow::Result<()> {
    let message_id = queued.message_id();
    let wait = queued.wait();
    if delivery_timeout.is_zero() {
        tokio::select! {
            _ = abort.canceled() => {
                channel.cancel_pending(message_id, "telnyx turn canceled").await;
                anyhow::bail!("bridge turn canceled before Channel submission");
            }
            result = wait => {
                result?;
                Ok(())
            }
        }
    } else {
        tokio::select! {
            _ = abort.canceled() => {
                channel.cancel_pending(message_id, "telnyx turn canceled").await;
                anyhow::bail!("bridge turn canceled before Channel submission");
            }
            result = time::timeout(delivery_timeout, wait) => {
                match result {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(error)) => Err(error.into()),
                    Err(_elapsed) => {
                        channel.cancel_pending(message_id, "delivery deadline exceeded").await;
                        anyhow::bail!(
                            "timed out delivering caller turn to tmux after {} ms",
                            delivery_timeout.as_millis()
                        );
                    }
                }
            }
        }
    }
}

fn reply_lines_from_rendered_delta(
    baseline: &str,
    rendered: &str,
    prompt: &str,
    marker: &str,
) -> Vec<String> {
    let delta = rendered_delta(baseline, rendered);
    let lines = delta.lines().collect::<Vec<_>>();
    let end = lines
        .iter()
        .position(|line| line.trim() == marker)
        .unwrap_or(lines.len());
    let turn_id = turn_id_from_marker(marker);
    let start = reply_start_index(&lines[..end], marker, turn_id.as_deref());
    lines[start..end]
        .iter()
        .copied()
        .filter(|line| !is_prompt_echo_line(line, prompt, marker))
        .map(|line| line.trim_end().to_string())
        .filter(|line| !line.trim().is_empty())
        .collect()
}

fn rendered_delta<'a>(baseline: &str, rendered: &'a str) -> &'a str {
    if let Some(delta) = rendered.strip_prefix(baseline) {
        return delta;
    }
    let Some(anchor) = baseline.lines().rev().find(|line| !line.trim().is_empty()) else {
        return rendered;
    };
    let mut offset = 0;
    let mut best_offset = None;
    for line in rendered.split_inclusive('\n') {
        if line.trim_end() == anchor.trim_end() {
            best_offset = Some(offset + line.len());
        }
        offset += line.len();
    }
    best_offset.map_or(rendered, |start| &rendered[start..])
}

fn reply_start_index(lines: &[&str], marker: &str, turn_id: Option<&str>) -> usize {
    let turn_tag = turn_id.map(|turn_id| format!("[motlie-call turn={turn_id}]"));
    let turn_fragment = turn_id.map(|turn_id| format!("turn={turn_id}"));
    lines
        .iter()
        .rposition(|line| {
            let trimmed = line.trim();
            trimmed.contains(marker)
                || trimmed.contains("Reply once, then end with")
                || trimmed.contains("[[motlie-call:end")
                || turn_tag.as_deref().is_some_and(|tag| trimmed.contains(tag))
                || turn_fragment.as_deref().is_some_and(|fragment| {
                    trimmed.contains(fragment) && trimmed.contains("motlie-call")
                })
        })
        .map_or(0, |index| index + 1)
}

fn turn_id_from_marker(marker: &str) -> Option<String> {
    marker
        .strip_prefix("[[motlie-call:end turn=")
        .and_then(|rest| rest.strip_suffix("]]"))
        .map(ToOwned::to_owned)
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

fn is_prompt_echo_line(line: &str, prompt: &str, marker: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed == prompt.trim()
        || trimmed == marker
        || trimmed.contains("[motlie-call turn=")
        || trimmed.contains("Reply once, then end with")
        || trimmed.contains("[[motlie-call:end")
        || (trimmed.starts_with("turn=") && trimmed.ends_with("]]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_line_emission_withholds_tail_until_later_line_or_final() {
        let mut stable = StableLineEmission::new(2);
        assert!(stable.update(vec!["draft".to_string()], false).is_empty());
        assert!(stable.update(vec!["draft".to_string()], false).is_empty());
        assert_eq!(
            stable.update(vec!["draft".to_string(), "next".to_string()], false),
            vec!["draft".to_string()]
        );
        assert_eq!(
            stable.update(vec!["draft".to_string(), "next".to_string()], true),
            vec!["next".to_string()]
        );
    }

    #[test]
    fn stable_line_emission_does_not_reemit_content_when_line_indices_shift() {
        let mut stable = StableLineEmission::new(1);
        assert_eq!(
            stable.update(vec!["first".to_string(), "second".to_string()], false),
            vec!["first".to_string()]
        );
        assert_eq!(
            stable.update(
                vec![
                    "old context".to_string(),
                    "first".to_string(),
                    "second".to_string(),
                    "third".to_string(),
                ],
                false,
            ),
            vec!["second".to_string()]
        );
        assert_eq!(
            stable.update(
                vec![
                    "old context".to_string(),
                    "first".to_string(),
                    "second".to_string(),
                    "third".to_string(),
                ],
                true,
            ),
            vec!["third".to_string()]
        );
    }

    #[test]
    fn reply_lines_extract_marker_delimited_reply() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!("{baseline}{prompt}\nThe answer is yes.\n{marker}\n");
        let lines = reply_lines_from_rendered_delta(baseline, &rendered, prompt, marker);
        assert_eq!(lines, vec!["The answer is yes.".to_string()]);
    }

    #[test]
    fn reply_lines_ignore_wrapped_prompt_echo() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: status and report back\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!(
            "{baseline}[motlie-call turn=turn-test] Caller: status and\nreport back\nReply once, then end with\n[[motlie-call:end\nturn=turn-test]]\nThe actual reply.\n{marker}\n"
        );

        let lines = reply_lines_from_rendered_delta(baseline, &rendered, prompt, marker);

        assert_eq!(lines, vec!["The actual reply.".to_string()]);
    }

    #[test]
    fn reply_lines_use_baseline_tail_when_prefix_is_evicted() {
        let baseline = "older context\nagent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!("agent> ready\n{prompt}\nFirst reply line.\n{marker}\n");

        let lines = reply_lines_from_rendered_delta(baseline, &rendered, prompt, marker);

        assert_eq!(lines, vec!["First reply line.".to_string()]);
    }

    #[test]
    fn reply_lines_do_not_return_instruction_echo_as_reply() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!(
            "{baseline}[motlie-call turn=turn-test] Caller: hello\nReply once, then end with\n{marker}\n"
        );

        let lines = reply_lines_from_rendered_delta(baseline, &rendered, prompt, marker);

        assert!(lines.is_empty());
    }
}
