use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use motlie_tmux::{
    CaptureNormalizeMode, HistoryOptions, HostHandle, KeySequence, SessionWatchHandle,
    SessionWatchOptions, Target,
};
use tokio::sync::Mutex;
use tokio::time::{self, Instant};

const DEFAULT_INPUT_QUIET_FOR: Duration = Duration::from_secs(10);
const DEFAULT_INPUT_DELIVERY_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_INPUT_BACKOFF_INITIAL: Duration = Duration::from_millis(250);
const DEFAULT_INPUT_BACKOFF_MAX: Duration = Duration::from_secs(5);
const DEFAULT_TRAILING_ENTER_DELAY: Duration = Duration::from_millis(750);

#[derive(Clone)]
pub struct TmuxBridge {
    target: Target,
    host: HostHandle,
    session_name: String,
    watch: Arc<Mutex<SessionWatchHandle>>,
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

#[derive(Clone, Debug, Eq, PartialEq)]
enum TargetActivity {
    Idle,
    Active { reason: &'static str },
}

impl TargetActivity {
    fn reason(&self) -> Option<&'static str> {
        match self {
            Self::Idle => None,
            Self::Active { reason } => Some(reason),
        }
    }
}

type BoxResultFuture<'a, T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send + 'a>>;

trait ActivityProbe {
    fn check<'a>(&'a mut self) -> BoxResultFuture<'a, TargetActivity>;
}

trait HistoryRenderer {
    fn render<'a>(&'a mut self) -> BoxResultFuture<'a, String>;
}

trait KeySender {
    fn send<'a>(&'a mut self, keys: KeySequence) -> BoxResultFuture<'a, ()>;
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
        let session_name = target.session_name().to_string();
        Ok(Self {
            target,
            host,
            session_name,
            watch: Arc::new(Mutex::new(watch)),
            reply_timeout,
            injection,
            send_lock: Arc::new(Mutex::new(())),
        })
    }

    pub async fn send_turn(&self, turn_id: &str, caller_text: &str) -> anyhow::Result<String> {
        let _send_guard = self.send_lock.lock().await;
        let marker = end_marker(turn_id);
        let prompt = format!(
            "[motlie-call turn={turn_id}] Caller: {caller_text}\nReply once, then end with {marker}"
        );
        let mut activity = LiveActivityProbe {
            host: &self.host,
            watch: self.watch.clone(),
            session_name: &self.session_name,
            input_quiet_for: self.injection.input_quiet_for,
        };
        let mut history = LiveHistoryRenderer {
            watch: self.watch.clone(),
        };
        let mut sender = LiveKeySender {
            target: &self.target,
        };
        let baseline = deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            &prompt,
            &self.injection,
        )
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

struct LiveActivityProbe<'a> {
    host: &'a HostHandle,
    watch: Arc<Mutex<SessionWatchHandle>>,
    session_name: &'a str,
    input_quiet_for: Duration,
}

impl ActivityProbe for LiveActivityProbe<'_> {
    fn check<'a>(&'a mut self) -> BoxResultFuture<'a, TargetActivity> {
        Box::pin(async move {
            evaluate_live_target_activity(
                self.host,
                self.watch.clone(),
                self.session_name,
                self.input_quiet_for,
            )
            .await
        })
    }
}

struct LiveHistoryRenderer {
    watch: Arc<Mutex<SessionWatchHandle>>,
}

impl HistoryRenderer for LiveHistoryRenderer {
    fn render<'a>(&'a mut self) -> BoxResultFuture<'a, String> {
        Box::pin(async move { Ok(self.watch.lock().await.render_text().await) })
    }
}

struct LiveKeySender<'a> {
    target: &'a Target,
}

impl KeySender for LiveKeySender<'_> {
    fn send<'a>(&'a mut self, keys: KeySequence) -> BoxResultFuture<'a, ()> {
        Box::pin(async move { self.target.send_keys(&keys).await.map_err(Into::into) })
    }
}

async fn deliver_prompt_after_idle<A, H, S>(
    activity: &mut A,
    history: &mut H,
    sender: &mut S,
    prompt: &str,
    config: &KeystrokeInjectionConfig,
) -> anyhow::Result<String>
where
    A: ActivityProbe,
    H: HistoryRenderer,
    S: KeySender,
{
    wait_for_idle(activity, config).await?;
    let baseline = history.render().await?;
    sender
        .send(KeySequence::literal(prompt).then_enter())
        .await?;
    if config.trailing_enter {
        time::sleep(config.trailing_enter_delay).await;
        sender.send(KeySequence::parse("{Enter}")?).await?;
    }
    Ok(baseline)
}

async fn wait_for_idle<A>(activity: &mut A, config: &KeystrokeInjectionConfig) -> anyhow::Result<()>
where
    A: ActivityProbe,
{
    let mut backoff = config.input_backoff_initial;
    let started_at = Instant::now();
    loop {
        let target_activity = activity.check().await?;
        if matches!(target_activity, TargetActivity::Idle) {
            return Ok(());
        }
        if config.input_delivery_timeout > Duration::ZERO
            && started_at.elapsed() >= config.input_delivery_timeout
        {
            anyhow::bail!(
                "timed out waiting for tmux target to become idle after {} ms; last activity reason: {}",
                config.input_delivery_timeout.as_millis(),
                target_activity.reason().unwrap_or("active")
            );
        }
        let sleep_for = if config.input_delivery_timeout > Duration::ZERO {
            let remaining = config
                .input_delivery_timeout
                .saturating_sub(started_at.elapsed());
            backoff.min(remaining.max(Duration::from_millis(1)))
        } else {
            backoff
        };
        tracing::info!(
            reason = target_activity.reason().unwrap_or("active"),
            retry_delay_ms = sleep_for.as_millis() as u64,
            delivery_wait_ms = started_at.elapsed().as_millis() as u64,
            "telnyx_agent.tmux_input.deferred"
        );
        time::sleep(sleep_for).await;
        backoff = next_backoff(backoff, config.input_backoff_max);
    }
}

async fn evaluate_live_target_activity(
    host: &HostHandle,
    watch: Arc<Mutex<SessionWatchHandle>>,
    session_name: &str,
    input_quiet_for: Duration,
) -> anyhow::Result<TargetActivity> {
    if input_quiet_for > Duration::ZERO {
        let activity = host.session_client_activity(session_name).await?;
        if let Some(latest) = activity.latest_client_activity {
            let age = latest_client_activity_age(latest, unix_now_secs());
            if age < input_quiet_for {
                return Ok(TargetActivity::Active {
                    reason: "recent_client_input",
                });
            }
        }
    }

    let rendered = watch.lock().await.render_text().await;
    if has_uncommitted_composer_text(&rendered) {
        return Ok(TargetActivity::Active {
            reason: "composer_text",
        });
    }
    Ok(TargetActivity::Idle)
}

fn latest_client_activity_age(latest_client_activity_secs: u64, now_secs: u64) -> Duration {
    Duration::from_secs(now_secs.saturating_sub(latest_client_activity_secs))
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.saturating_mul(2).min(max)
}

fn has_uncommitted_composer_text(rendered: &str) -> bool {
    let Some(last_line) = rendered.lines().rev().find(|line| !line.trim().is_empty()) else {
        return false;
    };
    let trimmed = last_line.trim();
    const PROMPT_PREFIXES: &[&str] = &["> ", "$ ", "# "];
    PROMPT_PREFIXES.iter().any(|prefix| {
        trimmed
            .strip_prefix(prefix)
            .is_some_and(|rest| !rest.trim().is_empty())
    })
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
    let lines = delta.lines().collect::<Vec<_>>();
    let marker_line_index = lines.iter().rposition(|line| line.trim() == marker)?;
    let reply_start = lines[..marker_line_index]
        .iter()
        .rposition(|line| line.contains(marker))
        .map_or(0, |index| index + 1);
    let reply = lines[reply_start..marker_line_index]
        .iter()
        .copied()
        .filter(|line| !is_prompt_echo_line(line, prompt, marker))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if reply.is_empty() { None } else { Some(reply) }
}

fn is_prompt_echo_line(line: &str, prompt: &str, marker: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed == prompt.trim()
        || trimmed == marker
        || trimmed.contains("[motlie-call turn=")
        || trimmed.contains("Reply once, then end with")
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

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

    #[test]
    fn extractor_ignores_wrapped_prompt_echo_marker() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello from caller\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!(
            "{baseline}[motlie-call turn=turn-test] Caller: hello from\ncaller\nReply once, then end with\n{marker}\nThe actual reply.\n{marker}\n"
        );

        let reply = extract_reply_from_rendered_delta(baseline, &rendered, prompt, marker);

        assert_eq!(reply.as_deref(), Some("The actual reply."));
    }

    #[test]
    fn extractor_does_not_return_instruction_echo_as_reply() {
        let baseline = "agent> ready\n";
        let prompt = "[motlie-call turn=turn-test] Caller: hello\nReply once, then end with [[motlie-call:end turn=turn-test]]";
        let marker = "[[motlie-call:end turn=turn-test]]";
        let rendered = format!(
            "{baseline}[motlie-call turn=turn-test] Caller: hello\nReply once, then end with\n{marker}\n"
        );

        let reply = extract_reply_from_rendered_delta(baseline, &rendered, prompt, marker);

        assert_eq!(reply, None);
    }

    #[tokio::test]
    async fn active_target_times_out_when_delivery_deadline_expires() {
        let mut activity = FakeActivityProbe::new(vec![
            TargetActivity::Active {
                reason: "recent_client_input",
            },
            TargetActivity::Active {
                reason: "recent_client_input",
            },
            TargetActivity::Active {
                reason: "recent_client_input",
            },
        ]);
        let mut history = FakeHistoryRenderer::new("agent> ready\n");
        let mut sender = FakeKeySender::default();
        let mut config = test_config(false, Duration::ZERO);
        config.input_delivery_timeout = Duration::from_millis(25);

        let error = deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            "queued transcription",
            &config,
        )
        .await
        .expect_err("delivery should fail when target stays active past the deadline");

        assert!(format!("{error:#}").contains("timed out waiting for tmux target to become idle"));
        assert_eq!(history.renders, 0);
        assert!(sender.sent.is_empty());
    }

    #[tokio::test]
    async fn active_target_queues_with_backoff_then_flushes() {
        let mut activity = FakeActivityProbe::new(vec![
            TargetActivity::Active {
                reason: "recent_client_input",
            },
            TargetActivity::Active {
                reason: "composer_text",
            },
            TargetActivity::Idle,
        ]);
        let mut history = FakeHistoryRenderer::new("agent> ready\n");
        let mut sender = FakeKeySender::default();
        let config = test_config(false, Duration::ZERO);
        let start = Instant::now();

        let baseline = deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            "queued transcription",
            &config,
        )
        .await
        .expect("delivery should flush after target becomes idle");

        assert_eq!(baseline, "agent> ready\n");
        assert_eq!(activity.checks, 3);
        assert_eq!(history.renders, 1);
        assert!(Instant::now().duration_since(start) >= Duration::from_millis(30));
        assert_eq!(sender.sent.len(), 1);
        assert!(sender.sent[0].iter().any(|command| {
            command
                .last()
                .is_some_and(|arg| arg == "queued transcription")
        }));
    }

    #[tokio::test]
    async fn idle_target_sends_immediately() {
        let mut activity = FakeActivityProbe::new(vec![TargetActivity::Idle]);
        let mut history = FakeHistoryRenderer::new("agent> ready\n");
        let mut sender = FakeKeySender::default();
        let config = test_config(false, Duration::ZERO);
        deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            "immediate transcription",
            &config,
        )
        .await
        .expect("idle target should send");

        assert_eq!(activity.checks, 1);
        assert_eq!(history.renders, 1);
        assert_eq!(sender.sent.len(), 1);
    }

    #[tokio::test]
    async fn trailing_enter_is_sent_after_configured_delay() {
        let mut activity = FakeActivityProbe::new(vec![TargetActivity::Idle]);
        let mut history = FakeHistoryRenderer::new("agent> ready\n");
        let mut sender = FakeKeySender::default();
        let config = test_config(true, Duration::from_millis(75));
        let start = Instant::now();

        deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            "submit transcription",
            &config,
        )
        .await
        .expect("trailing enter should send");

        assert!(Instant::now().duration_since(start) >= Duration::from_millis(75));
        assert_eq!(sender.sent.len(), 2);
        assert_eq!(
            sender.sent[1],
            vec![vec!["send-keys", "-t", "%target", "Enter"]]
        );
    }

    #[tokio::test]
    async fn trailing_enter_disabled_is_honored() {
        let mut activity = FakeActivityProbe::new(vec![TargetActivity::Idle]);
        let mut history = FakeHistoryRenderer::new("agent> ready\n");
        let mut sender = FakeKeySender::default();
        let config = test_config(false, Duration::from_millis(75));

        deliver_prompt_after_idle(
            &mut activity,
            &mut history,
            &mut sender,
            "no trailing enter",
            &config,
        )
        .await
        .expect("delivery should send without trailing enter");

        assert_eq!(sender.sent.len(), 1);
        assert_eq!(
            sender.sent[0].len(),
            2,
            "prompt send includes literal plus first submit Enter"
        );
    }

    #[test]
    fn composer_text_detector_uses_rendered_history_prompt_line() {
        assert!(has_uncommitted_composer_text("last output\n> unsent"));
        assert!(has_uncommitted_composer_text("$ pending command"));
        assert!(!has_uncommitted_composer_text("agent> ready\n"));
        assert!(!has_uncommitted_composer_text("last output\n"));
    }

    #[test]
    fn recent_client_activity_age_saturates_future_timestamps() {
        assert_eq!(latest_client_activity_age(120, 100), Duration::ZERO);
        assert_eq!(latest_client_activity_age(90, 100), Duration::from_secs(10));
    }

    fn test_config(
        trailing_enter: bool,
        trailing_enter_delay: Duration,
    ) -> KeystrokeInjectionConfig {
        KeystrokeInjectionConfig::new(
            Duration::from_secs(10),
            Duration::from_secs(2),
            Duration::from_millis(10),
            Duration::from_millis(40),
            trailing_enter,
            trailing_enter_delay,
        )
    }

    struct FakeActivityProbe {
        states: VecDeque<TargetActivity>,
        checks: usize,
    }

    impl FakeActivityProbe {
        fn new(states: Vec<TargetActivity>) -> Self {
            Self {
                states: states.into(),
                checks: 0,
            }
        }
    }

    impl ActivityProbe for FakeActivityProbe {
        fn check<'a>(&'a mut self) -> BoxResultFuture<'a, TargetActivity> {
            Box::pin(async move {
                self.checks += 1;
                Ok(self.states.pop_front().unwrap_or(TargetActivity::Idle))
            })
        }
    }

    struct FakeHistoryRenderer {
        rendered: String,
        renders: usize,
    }

    impl FakeHistoryRenderer {
        fn new(rendered: &str) -> Self {
            Self {
                rendered: rendered.to_string(),
                renders: 0,
            }
        }
    }

    impl HistoryRenderer for FakeHistoryRenderer {
        fn render<'a>(&'a mut self) -> BoxResultFuture<'a, String> {
            Box::pin(async move {
                self.renders += 1;
                Ok(self.rendered.clone())
            })
        }
    }

    #[derive(Default)]
    struct FakeKeySender {
        sent: Vec<Vec<Vec<String>>>,
    }

    impl KeySender for FakeKeySender {
        fn send<'a>(&'a mut self, keys: KeySequence) -> BoxResultFuture<'a, ()> {
            Box::pin(async move {
                self.sent.push(keys.to_tmux_args("%target"));
                Ok(())
            })
        }
    }
}
