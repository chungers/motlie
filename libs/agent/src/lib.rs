//! Agent interaction policy over `motlie-tmux`.
//!
//! The first surface is managed prompt delivery through a process-local
//! [`Channel`] keyed by a stable tmux session identity.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use motlie_tmux::{HostHandle, KeySequence, SessionClientActivity, Target};
use thiserror::Error;
use tokio::sync::{broadcast, oneshot, Mutex};
use tokio::time::{sleep, timeout, Instant as TokioInstant};

const DEFAULT_EVENT_CAPACITY: usize = 1024;
const COMPOSER_SEPARATOR: &str = "\n\n---\n\n";

fn lock_or_recover<T>(mutex: &StdMutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A multi-consumer, lossy delivery event stream.
///
/// Internally this is a Tokio broadcast receiver. Slow consumers may observe
/// `Lagged` receive errors and should reconcile with [`Channel::status`].
pub type DeliveryEvents = broadcast::Receiver<DeliveryEvent>;

/// Process-local identity for one resolved tmux session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub host_alias: String,
    pub host_connection_id: String,
    pub tmux_session_id: Option<String>,
    pub tmux_session_name: String,
    pub tmux_session_created: Option<u64>,
}

impl SessionKey {
    /// Build a key from a resolved tmux target.
    pub fn from_target(
        host_alias: impl Into<String>,
        host_connection_id: impl Into<String>,
        target: &Target,
    ) -> Self {
        Self {
            host_alias: host_alias.into(),
            host_connection_id: host_connection_id.into(),
            tmux_session_id: target.session_id().map(ToOwned::to_owned),
            tmux_session_name: target.session_name().to_string(),
            tmux_session_created: target.session_info().map(|session| session.created),
        }
    }
}

/// A resolved tmux destination plus optional agent UI profile.
#[derive(Clone)]
pub struct ResolvedSession {
    pub key: SessionKey,
    pub host: HostHandle,
    pub target: Target,
    pub ui_profile: Option<UiProfile>,
}

impl ResolvedSession {
    pub fn new(key: SessionKey, host: HostHandle, target: Target) -> Self {
        Self {
            key,
            host,
            target,
            ui_profile: None,
        }
    }

    pub fn with_ui_profile(mut self, profile: UiProfile) -> Self {
        self.ui_profile = Some(profile);
        self
    }
}

/// Process-wide owner for channels keyed by [`SessionKey`].
#[derive(Clone)]
pub struct ChannelManager {
    inner: Arc<ChannelManagerInner>,
}

struct ChannelManagerInner {
    config: ChannelConfig,
    channels: StdMutex<HashMap<SessionKey, Channel>>,
    next_message_id: Arc<AtomicU64>,
    event_tx: broadcast::Sender<DeliveryEvent>,
}

impl ChannelManager {
    pub fn new(config: ChannelConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.event_capacity.max(1));
        Self {
            inner: Arc::new(ChannelManagerInner {
                config,
                channels: StdMutex::new(HashMap::new()),
                next_message_id: Arc::new(AtomicU64::new(1)),
                event_tx,
            }),
        }
    }

    /// Return a cheap-clone channel handle for this resolved session.
    pub fn get_or_bind(&self, resolved: ResolvedSession) -> Result<Channel, DeliveryError> {
        let mut channels = lock_or_recover(&self.inner.channels);
        if let Some(channel) = channels.get(&resolved.key) {
            let new_target = resolved.target.target_string();
            if channel.inner.target_string != new_target {
                return Err(DeliveryError::TargetIdentityMismatch {
                    key: Box::new(resolved.key),
                    existing_target: channel.inner.target_string.clone(),
                    new_target,
                });
            }
            return Ok(channel.clone());
        }

        let key = resolved.key.clone();
        let channel = Channel {
            inner: Arc::new(ChannelInner {
                key: key.clone(),
                host: resolved.host,
                target_string: resolved.target.target_string(),
                target: resolved.target,
                config: self.inner.config.clone(),
                ui_profile: resolved
                    .ui_profile
                    .unwrap_or(self.inner.config.default_ui_profile),
                queue: Mutex::new(QueueState::default()),
                status: StdMutex::new(ChannelStatus::new(key.clone())),
                next_message_id: Arc::clone(&self.inner.next_message_id),
                next_waiter_id: AtomicU64::new(1),
                event_tx: self.inner.event_tx.clone(),
            }),
        };
        channels.insert(key, channel.clone());
        Ok(channel)
    }

    pub fn remove(&self, key: &SessionKey) -> Option<Channel> {
        lock_or_recover(&self.inner.channels).remove(key)
    }

    pub fn remove_where(&self, mut predicate: impl FnMut(&SessionKey) -> bool) -> usize {
        let mut channels = lock_or_recover(&self.inner.channels);
        let keys = channels
            .keys()
            .filter(|key| predicate(key))
            .cloned()
            .collect::<Vec<_>>();
        let removed = keys.len();
        for key in keys {
            channels.remove(&key);
        }
        removed
    }

    pub fn subscribe(&self) -> DeliveryEvents {
        self.inner.event_tx.subscribe()
    }
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new(ChannelConfig::default())
    }
}

/// Managed delivery configuration shared by channels from one manager.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub input_quiet_for: Duration,
    pub coalesce_window: Duration,
    pub coalesce_max_wait: Duration,
    pub default_submit: SubmitPolicy,
    pub default_ui_profile: UiProfile,
    pub event_capacity: usize,
    pub preserve_composer: bool,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            input_quiet_for: Duration::from_secs(10),
            coalesce_window: Duration::from_millis(500),
            coalesce_max_wait: Duration::from_millis(1500),
            default_submit: SubmitPolicy::default(),
            default_ui_profile: UiProfile::Generic,
            event_capacity: DEFAULT_EVENT_CAPACITY,
            preserve_composer: true,
        }
    }
}

/// A process-local managed conduit to one resolved agent tmux session.
#[derive(Clone)]
pub struct Channel {
    inner: Arc<ChannelInner>,
}

struct ChannelInner {
    key: SessionKey,
    host: HostHandle,
    target_string: String,
    target: Target,
    config: ChannelConfig,
    ui_profile: UiProfile,
    queue: Mutex<QueueState>,
    status: StdMutex<ChannelStatus>,
    next_message_id: Arc<AtomicU64>,
    next_waiter_id: AtomicU64,
    event_tx: broadcast::Sender<DeliveryEvent>,
}

impl Channel {
    /// Synchronously deliver a message and wait for prompt-window submission.
    pub async fn send(
        &self,
        message: ManagedMessage,
        options: SendOptions,
    ) -> Result<SubmissionOutcome, DeliveryError> {
        let (waiter_tx, waiter_rx) = oneshot::channel();
        let waiter_id = WaiterId(self.inner.next_waiter_id.fetch_add(1, Ordering::Relaxed));
        let message_id = self
            .accept(
                message,
                options.submit,
                QuietGuardPolicy::Default,
                Some(Waiter {
                    id: waiter_id,
                    tx: waiter_tx,
                }),
            )
            .await?;

        match timeout(options.timeout, waiter_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_closed)) => Err(DeliveryError::ChannelClosed),
            Err(_elapsed) => Err(DeliveryError::Timeout {
                message_id,
                timeout: options.timeout,
                still_pending: self.cancel_waiter(message_id, waiter_id).await,
            }),
        }
    }

    /// Accept a message for asynchronous, fire-and-forget delivery.
    pub async fn enqueue(
        &self,
        message: ManagedMessage,
        options: EnqueueOptions,
    ) -> Result<QueuedDelivery, DeliveryError> {
        let accepted_at = Instant::now();
        let message_id = self
            .accept(message, options.submit, options.quiet_guard, None)
            .await?;
        Ok(QueuedDelivery {
            message_id,
            target: self.inner.key.clone(),
            accepted_at,
        })
    }

    pub fn subscribe(&self) -> DeliveryEvents {
        self.inner.event_tx.subscribe()
    }

    pub fn status(&self) -> ChannelStatus {
        lock_or_recover(&self.inner.status).clone()
    }

    pub async fn cancel_pending(&self, message_id: MessageId, reason: impl Into<String>) -> bool {
        let reason = reason.into();
        let (segment, pending_count) = {
            let mut queue = self.inner.queue.lock().await;
            let Some(position) = queue
                .pending
                .iter()
                .position(|segment| segment.message_id == message_id)
            else {
                return false;
            };
            let segment = queue.pending.remove(position).expect("position exists");
            let pending_count = queue.pending.len();
            queue.accept_generation = queue.accept_generation.wrapping_add(1);
            if queue.pending.is_empty() {
                queue.coalesce_started_at = None;
            }
            (segment, pending_count)
        };
        self.update_status(|status| {
            status.pending_count = pending_count;
            status.flushing = pending_count > 0;
        });
        let error = DeliveryError::Canceled {
            message_id,
            reason: reason.clone(),
        };
        self.emit(DeliveryEvent::Failed {
            target: self.inner.key.clone(),
            message_ids: vec![message_id],
            error: error.to_string(),
        });
        for waiter in segment.waiters {
            let _ = waiter.tx.send(Err(error.clone()));
        }
        true
    }

    async fn accept(
        &self,
        message: ManagedMessage,
        submit: SubmitPolicy,
        quiet_guard: QuietGuardPolicy,
        waiter: Option<Waiter>,
    ) -> Result<MessageId, DeliveryError> {
        if message.body.is_empty() {
            return Err(DeliveryError::InvalidMessage(
                "message body cannot be empty",
            ));
        }

        let source = message.source.clone();
        let dedup_key = match message.dedup {
            DedupPolicy::Body => Some(normalize_body(&message.body)),
            DedupPolicy::Unique => None,
        };
        let coalesce = message.coalesce;
        let mut start_worker = false;
        let has_async_source = waiter.is_none();
        let message_id;
        let pending_count;
        {
            let mut queue = self.inner.queue.lock().await;
            if let Some(segment) = dedup_key.as_ref().and_then(|dedup_key| {
                queue
                    .pending
                    .iter_mut()
                    .find(|segment| segment.dedup_key == *dedup_key)
            }) {
                message_id = segment.message_id;
                segment.add_source(source.clone());
                segment.submit = segment.submit.merged(submit);
                segment.paste_mode = segment.paste_mode.merged(message.paste_mode);
                segment.quiet_guard = segment.quiet_guard.merged(quiet_guard);
                segment.coalesce = segment.coalesce.merged(coalesce);
                if let Some(waiter) = waiter {
                    segment.waiters.push(waiter);
                }
                if has_async_source {
                    segment.has_async_source = true;
                }
            } else {
                message_id = MessageId(self.inner.next_message_id.fetch_add(1, Ordering::Relaxed));
                queue.pending.push_back(PendingSegment {
                    message_id,
                    dedup_key: dedup_key.unwrap_or_else(|| format!("__unique:{message_id}")),
                    body: message.body,
                    paste_mode: message.paste_mode,
                    sources: vec![source.clone()],
                    submit,
                    quiet_guard,
                    coalesce,
                    waiters: waiter.into_iter().collect(),
                    has_async_source,
                    deferred_once: false,
                });
            }

            pending_count = queue.pending.len();
            queue.accept_generation = queue.accept_generation.wrapping_add(1);
            if !queue.flushing {
                queue.flushing = true;
                start_worker = true;
            }
        }

        self.update_status(|status| {
            status.pending_count = pending_count;
            status.flushing = true;
            status.last_error = None;
        });
        self.emit(DeliveryEvent::Accepted {
            message_id,
            target: self.inner.key.clone(),
            source,
            accepted_at: Instant::now(),
        });
        if start_worker {
            let channel = self.clone();
            tokio::spawn(async move {
                channel.flush_loop().await;
            });
        }
        Ok(message_id)
    }

    async fn cancel_waiter(&self, message_id: MessageId, waiter_id: WaiterId) -> bool {
        let pending_count;
        let still_pending;
        {
            let mut queue = self.inner.queue.lock().await;
            let Some(position) = queue
                .pending
                .iter()
                .position(|segment| segment.message_id == message_id)
            else {
                return false;
            };
            let segment = &mut queue.pending[position];
            segment.waiters.retain(|waiter| waiter.id != waiter_id);
            if segment.waiters.is_empty() && !segment.has_async_source {
                queue.pending.remove(position);
                still_pending = false;
            } else {
                still_pending = true;
            }
            if queue.pending.is_empty() {
                queue.coalesce_started_at = None;
            }
            pending_count = queue.pending.len();
        }
        self.update_status(|status| {
            status.pending_count = pending_count;
        });
        still_pending
    }

    async fn flush_loop(self) {
        loop {
            if let Err(error) = self.apply_quiet_guard().await {
                let batch = self.drain_pending().await;
                self.fail_batch(batch, error).await;
                self.finish_if_empty().await;
                return;
            }

            let mut batch = match self.next_batch().await {
                Some(batch) => batch,
                None => return,
            };

            match self.submit_batch(&mut batch).await {
                Ok(BatchSubmitAttempt::Submitted(result)) => {
                    if batch.len() > 1 {
                        self.emit(DeliveryEvent::Coalesced {
                            target: self.inner.key.clone(),
                            message_ids: batch.iter().map(|segment| segment.message_id).collect(),
                            segment_count: batch.len(),
                        });
                    }
                    self.complete_batch(batch, result).await;
                }
                Ok(BatchSubmitAttempt::Deferred {
                    message_ids,
                    reason,
                    retry_after,
                }) => {
                    self.requeue_front(batch).await;
                    self.update_status(|status| {
                        status.last_defer_reason = Some(reason.clone());
                    });
                    self.emit(DeliveryEvent::Deferred {
                        target: self.inner.key.clone(),
                        message_ids,
                        reason,
                        retry_after,
                    });
                    sleep(retry_after).await;
                }
                Err(error) => self.fail_batch(batch, error).await,
            }
        }
    }

    async fn apply_quiet_guard(&self) -> Result<(), DeliveryError> {
        if self.inner.config.input_quiet_for.is_zero() {
            return Ok(());
        }
        let should_guard = {
            let queue = self.inner.queue.lock().await;
            if queue.pending.is_empty() {
                return Ok(());
            }
            queue
                .pending
                .iter()
                .any(|segment| segment.quiet_guard == QuietGuardPolicy::Default)
        };
        if !should_guard {
            return Ok(());
        }

        loop {
            let activity = self.quiet_guard_activity().await?;
            let Some(latest) = activity.latest_writable_client_activity else {
                self.trace_quiet_guard_activity(&activity, None);
                return Ok(());
            };
            let age = seconds_since_epoch(latest).unwrap_or(0);
            self.trace_quiet_guard_activity(&activity, Some(age));
            if age >= self.inner.config.input_quiet_for.as_secs() {
                return Ok(());
            }
            let retry_after =
                Duration::from_secs((self.inner.config.input_quiet_for.as_secs() - age).max(1));
            let message_ids = self.mark_deferred().await;
            if message_ids.is_empty() {
                return Ok(());
            }
            let reason = DeferReason::RecentWritableClientActivity {
                latest_client_activity: latest,
                latest_client_activity_age_secs: age,
            };
            self.update_status(|status| {
                status.last_deferred_at = Some(Instant::now());
                status.last_defer_reason = Some(reason.clone());
                status.pending_count = message_ids.len();
                status.flushing = true;
            });
            self.emit(DeliveryEvent::Deferred {
                target: self.inner.key.clone(),
                message_ids,
                reason,
                retry_after,
            });
            sleep(retry_after).await;
        }
    }

    async fn quiet_guard_activity(&self) -> Result<SessionClientActivity, DeliveryError> {
        let session_name = self.inner.target.session_name();
        let primary_selector = self.inner.target.session_id().unwrap_or(session_name);
        let mut activity = self
            .inner
            .host
            .session_client_activity(primary_selector)
            .await
            .map_err(|err| DeliveryError::tmux("session_client_activity", err))?;

        if primary_selector != session_name {
            let by_name = self
                .inner
                .host
                .session_client_activity(session_name)
                .await
                .map_err(|err| DeliveryError::tmux("session_client_activity", err))?;
            merge_client_activity(&mut activity, by_name);
        }

        Ok(activity)
    }

    fn trace_quiet_guard_activity(&self, activity: &SessionClientActivity, age: Option<u64>) {
        if std::env::var_os("MOTLIE_AGENT_QUIET_GUARD_TRACE").is_none() {
            return;
        }
        eprintln!(
            "motlie-agent quiet_guard target={} session={} attached={} writable={} latest={:?} latest_writable={:?} age={:?} quiet_for_secs={}",
            self.inner.target_string,
            activity.session,
            activity.attached_clients,
            activity.writable_clients,
            activity.latest_client_activity,
            activity.latest_writable_client_activity,
            age,
            self.inner.config.input_quiet_for.as_secs(),
        );
    }

    async fn mark_deferred(&self) -> Vec<MessageId> {
        let mut queue = self.inner.queue.lock().await;
        for segment in &mut queue.pending {
            if segment.quiet_guard == QuietGuardPolicy::Default {
                segment.deferred_once = true;
            }
        }
        queue
            .pending
            .iter()
            .map(|segment| segment.message_id)
            .collect()
    }

    async fn next_batch(&self) -> Option<Vec<PendingSegment>> {
        loop {
            let (observed_generation, sleep_for) = {
                let mut queue = self.inner.queue.lock().await;
                if queue.pending.is_empty() {
                    queue.flushing = false;
                    queue.coalesce_started_at = None;
                    self.update_status(|status| {
                        status.pending_count = 0;
                        status.flushing = false;
                    });
                    return None;
                }
                if let Some(segment) = queue
                    .pending
                    .pop_front_if(|segment| segment.coalesce == CoalescePolicy::Disabled)
                {
                    queue.coalesce_started_at = None;
                    let pending_count = queue.pending.len();
                    self.update_status(|status| {
                        status.pending_count = pending_count;
                        status.flushing = true;
                    });
                    return Some(vec![segment]);
                }
                if self.inner.config.coalesce_window.is_zero() {
                    queue.coalesce_started_at = None;
                    let batch = queue.pending.drain(..).collect::<Vec<_>>();
                    self.update_status(|status| {
                        status.pending_count = 0;
                        status.flushing = true;
                    });
                    return Some(batch);
                }

                let started_at = *queue
                    .coalesce_started_at
                    .get_or_insert_with(TokioInstant::now);
                let max_wait = self.effective_coalesce_max_wait();
                let elapsed = started_at.elapsed();
                if elapsed >= max_wait {
                    queue.coalesce_started_at = None;
                    let batch = queue.pending.drain(..).collect::<Vec<_>>();
                    self.update_status(|status| {
                        status.pending_count = 0;
                        status.flushing = true;
                    });
                    return Some(batch);
                }

                let sleep_for = self
                    .inner
                    .config
                    .coalesce_window
                    .min(max_wait.saturating_sub(elapsed));
                (queue.accept_generation, sleep_for)
            };

            sleep(sleep_for).await;

            let mut queue = self.inner.queue.lock().await;
            if queue.pending.is_empty() {
                queue.flushing = false;
                queue.coalesce_started_at = None;
                self.update_status(|status| {
                    status.pending_count = 0;
                    status.flushing = false;
                });
                return None;
            }

            let max_wait = self.effective_coalesce_max_wait();
            let cap_elapsed = queue
                .coalesce_started_at
                .map(|started_at| started_at.elapsed() >= max_wait)
                .unwrap_or(false);
            if queue.accept_generation != observed_generation && !cap_elapsed {
                continue;
            }
            queue.coalesce_started_at = None;
            let batch = queue.pending.drain(..).collect::<Vec<_>>();
            self.update_status(|status| {
                status.pending_count = 0;
                status.flushing = true;
            });
            return Some(batch);
        }
    }

    fn effective_coalesce_max_wait(&self) -> Duration {
        self.inner
            .config
            .coalesce_max_wait
            .max(self.inner.config.coalesce_window)
    }

    async fn quiet_guard_defer_decision_for_batch(
        &self,
        batch: &mut [PendingSegment],
    ) -> Result<Option<(Vec<MessageId>, DeferReason, Duration)>, DeliveryError> {
        if self.inner.config.input_quiet_for.is_zero()
            || !batch
                .iter()
                .any(|segment| segment.quiet_guard == QuietGuardPolicy::Default)
        {
            return Ok(None);
        }

        let activity = self.quiet_guard_activity().await?;
        let Some(latest) = activity.latest_writable_client_activity else {
            self.trace_quiet_guard_activity(&activity, None);
            return Ok(None);
        };
        let age = seconds_since_epoch(latest).unwrap_or(0);
        self.trace_quiet_guard_activity(&activity, Some(age));
        if age >= self.inner.config.input_quiet_for.as_secs() {
            return Ok(None);
        }

        for segment in batch.iter_mut() {
            if segment.quiet_guard == QuietGuardPolicy::Default {
                segment.deferred_once = true;
            }
        }
        let message_ids = batch
            .iter()
            .map(|segment| segment.message_id)
            .collect::<Vec<_>>();
        let retry_after =
            Duration::from_secs((self.inner.config.input_quiet_for.as_secs() - age).max(1));
        let reason = DeferReason::RecentWritableClientActivity {
            latest_client_activity: latest,
            latest_client_activity_age_secs: age,
        };
        Ok(Some((message_ids, reason, retry_after)))
    }

    async fn requeue_front(&self, batch: Vec<PendingSegment>) {
        if batch.is_empty() {
            return;
        }
        let pending_count;
        {
            let mut queue = self.inner.queue.lock().await;
            for segment in batch.into_iter().rev() {
                queue.pending.push_front(segment);
            }
            queue.coalesce_started_at = None;
            queue.accept_generation = queue.accept_generation.wrapping_add(1);
            queue.flushing = true;
            pending_count = queue.pending.len();
        }
        self.update_status(|status| {
            status.pending_count = pending_count;
            status.flushing = true;
            status.last_deferred_at = Some(Instant::now());
        });
    }

    async fn drain_pending(&self) -> Vec<PendingSegment> {
        let mut queue = self.inner.queue.lock().await;
        let batch = queue.pending.drain(..).collect::<Vec<_>>();
        queue.coalesce_started_at = None;
        queue.flushing = false;
        self.update_status(|status| {
            status.pending_count = 0;
            status.flushing = false;
        });
        batch
    }

    async fn finish_if_empty(&self) {
        let mut queue = self.inner.queue.lock().await;
        if queue.pending.is_empty() {
            queue.coalesce_started_at = None;
            queue.flushing = false;
            self.update_status(|status| {
                status.pending_count = 0;
                status.flushing = false;
            });
        }
    }

    async fn submit_batch(
        &self,
        batch: &mut [PendingSegment],
    ) -> Result<BatchSubmitAttempt, DeliveryError> {
        let body = self.assemble_body(batch).await?;
        let paste_mode = batch.iter().fold(PasteMode::Literal, |mode, segment| {
            mode.merged(segment.paste_mode)
        });
        let submit = batch
            .iter()
            .fold(SubmitPolicy::typing_only(), |policy, segment| {
                policy.merged(segment.submit)
            });

        if let Some((message_ids, reason, retry_after)) =
            self.quiet_guard_defer_decision_for_batch(batch).await?
        {
            return Ok(BatchSubmitAttempt::Deferred {
                message_ids,
                reason,
                retry_after,
            });
        }

        let payload = paste_mode.wrap_payload(&body);
        let literal = KeySequence::literal(&payload);
        self.inner
            .target
            .send_keys(&literal)
            .await
            .map_err(|err| DeliveryError::tmux("send_payload", err))?;

        if !submit.prompt_submit {
            return Ok(BatchSubmitAttempt::Submitted(BatchSubmitResult {
                submitted_at: Instant::now(),
                verified: false,
            }));
        }

        if !submit.settle.is_zero() {
            sleep(submit.settle).await;
        }

        let enter =
            KeySequence::parse("{Enter}").map_err(|err| DeliveryError::tmux("parse_enter", err))?;
        for attempt in 0..=submit.retries {
            self.inner
                .target
                .send_keys(&enter)
                .await
                .map_err(|err| DeliveryError::tmux("send_enter", err))?;
            let submitted_at = Instant::now();
            match self
                .inner
                .ui_profile
                .verify_submitted(&self.inner.target)
                .await?
            {
                SubmitVerification::Submitted => {
                    return Ok(BatchSubmitAttempt::Submitted(BatchSubmitResult {
                        submitted_at,
                        verified: true,
                    }));
                }
                SubmitVerification::Unknown => {
                    if attempt == submit.retries {
                        if submit.require_verification {
                            return Err(DeliveryError::VerificationUnavailable {
                                message_ids: batch
                                    .iter()
                                    .map(|segment| segment.message_id)
                                    .collect(),
                            });
                        }
                        return Ok(BatchSubmitAttempt::Submitted(BatchSubmitResult {
                            submitted_at,
                            verified: false,
                        }));
                    }
                    if !submit.retry_delay.is_zero() {
                        sleep(submit.retry_delay).await;
                    }
                }
                SubmitVerification::StillComposed => {
                    if attempt == submit.retries {
                        return Err(DeliveryError::SubmitNotConfirmed {
                            message_ids: batch.iter().map(|segment| segment.message_id).collect(),
                        });
                    }
                    if !submit.retry_delay.is_zero() {
                        sleep(submit.retry_delay).await;
                    }
                }
            }
        }

        Err(DeliveryError::SubmitNotConfirmed {
            message_ids: batch.iter().map(|segment| segment.message_id).collect(),
        })
    }

    async fn assemble_body(&self, batch: &[PendingSegment]) -> Result<String, DeliveryError> {
        let mut body = format_segments(batch);
        if self.inner.config.preserve_composer {
            let should_separate = match self
                .inner
                .ui_profile
                .composer_state(&self.inner.target)
                .await?
            {
                ComposerState::Empty => false,
                ComposerState::NonEmpty => true,
                ComposerState::Unknown => batch.iter().any(|segment| segment.deferred_once),
            };
            if should_separate {
                body = format!("{COMPOSER_SEPARATOR}{body}");
            }
        }
        Ok(body)
    }

    async fn complete_batch(&self, batch: Vec<PendingSegment>, result: BatchSubmitResult) {
        let ids = batch
            .iter()
            .map(|segment| segment.message_id)
            .collect::<Vec<_>>();
        self.emit(DeliveryEvent::Submitted {
            target: self.inner.key.clone(),
            message_ids: ids,
            submitted_at: result.submitted_at,
            verified: result.verified,
        });
        for segment in batch {
            let outcome = if result.verified {
                SubmissionOutcome::SubmittedVerified {
                    message_id: segment.message_id,
                    submitted_at: result.submitted_at,
                }
            } else {
                SubmissionOutcome::SubmittedUnverified {
                    message_id: segment.message_id,
                    submitted_at: result.submitted_at,
                }
            };
            for waiter in segment.waiters {
                let _ = waiter.tx.send(Ok(outcome.clone()));
            }
        }
        self.update_status(|status| {
            status.last_error = None;
        });
    }

    async fn fail_batch(&self, batch: Vec<PendingSegment>, error: DeliveryError) {
        if batch.is_empty() {
            return;
        }
        let ids = batch
            .iter()
            .map(|segment| segment.message_id)
            .collect::<Vec<_>>();
        let error_text = error.to_string();
        self.emit(DeliveryEvent::Failed {
            target: self.inner.key.clone(),
            message_ids: ids,
            error: error_text.clone(),
        });
        for segment in batch {
            for waiter in segment.waiters {
                let _ = waiter.tx.send(Err(error.clone()));
            }
        }
        self.update_status(|status| {
            status.last_error = Some(error_text);
        });
    }

    fn emit(&self, event: DeliveryEvent) {
        let _ = self.inner.event_tx.send(event);
    }

    fn update_status(&self, update: impl FnOnce(&mut ChannelStatus)) {
        let mut status = lock_or_recover(&self.inner.status);
        update(&mut status);
    }
}

#[derive(Default)]
struct QueueState {
    pending: VecDeque<PendingSegment>,
    flushing: bool,
    accept_generation: u64,
    coalesce_started_at: Option<TokioInstant>,
}

struct PendingSegment {
    message_id: MessageId,
    dedup_key: String,
    body: String,
    paste_mode: PasteMode,
    sources: Vec<MessageSource>,
    submit: SubmitPolicy,
    quiet_guard: QuietGuardPolicy,
    coalesce: CoalescePolicy,
    waiters: Vec<Waiter>,
    has_async_source: bool,
    deferred_once: bool,
}

impl PendingSegment {
    fn add_source(&mut self, source: MessageSource) {
        if !self
            .sources
            .iter()
            .any(|existing| existing.label == source.label)
        {
            self.sources.push(source);
        }
    }
}

enum BatchSubmitAttempt {
    Submitted(BatchSubmitResult),
    Deferred {
        message_ids: Vec<MessageId>,
        reason: DeferReason,
        retry_after: Duration,
    },
}

struct BatchSubmitResult {
    submitted_at: Instant,
    verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WaiterId(u64);

struct Waiter {
    id: WaiterId,
    tx: oneshot::Sender<Result<SubmissionOutcome, DeliveryError>>,
}

/// Message accepted by a channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedMessage {
    pub source: MessageSource,
    pub body: String,
    pub paste_mode: PasteMode,
    pub dedup: DedupPolicy,
    pub coalesce: CoalescePolicy,
}

impl ManagedMessage {
    pub fn new(source: MessageSource, body: impl Into<String>) -> Self {
        Self {
            source,
            body: body.into(),
            paste_mode: PasteMode::Bracketed,
            dedup: DedupPolicy::Body,
            coalesce: CoalescePolicy::Enabled,
        }
    }

    pub fn with_paste_mode(mut self, paste_mode: PasteMode) -> Self {
        self.paste_mode = paste_mode;
        self
    }

    pub fn with_dedup(mut self, dedup: DedupPolicy) -> Self {
        self.dedup = dedup;
        self
    }

    pub fn with_coalesce(mut self, coalesce: CoalescePolicy) -> Self {
        self.coalesce = coalesce;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupPolicy {
    Body,
    Unique,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalescePolicy {
    Enabled,
    Disabled,
}

impl CoalescePolicy {
    fn merged(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disabled, _) | (_, Self::Disabled) => Self::Disabled,
            (Self::Enabled, Self::Enabled) => Self::Enabled,
        }
    }
}

/// Human-readable source attribution for coalesced prompt bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    Human,
    Timer,
    Broadcast,
    External,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageSource {
    label: String,
    kind: SourceKind,
}

impl MessageSource {
    pub fn human(label: impl Into<String>) -> Self {
        Self::new(label, SourceKind::Human)
    }

    pub fn timer(label: impl Into<String>) -> Self {
        Self::new(label, SourceKind::Timer)
    }

    pub fn broadcast(label: impl Into<String>) -> Self {
        Self::new(label, SourceKind::Broadcast)
    }

    pub fn external(label: impl Into<String>) -> Self {
        Self::new(label, SourceKind::External)
    }

    pub fn attributed(label: impl Into<String>) -> Self {
        Self::external(label)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn kind(&self) -> SourceKind {
        self.kind
    }

    fn new(label: impl Into<String>, kind: SourceKind) -> Self {
        Self {
            label: label.into(),
            kind,
        }
    }

    fn omit_header_when_single(&self) -> bool {
        self.kind == SourceKind::Human
    }
}

/// Payload typing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PasteMode {
    Literal,
    Bracketed,
}

impl PasteMode {
    fn merged(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bracketed, _) | (_, Self::Bracketed) => Self::Bracketed,
            (Self::Literal, Self::Literal) => Self::Literal,
        }
    }

    fn wrap_payload(self, body: &str) -> String {
        match self {
            Self::Bracketed if body.contains('\n') => format!("\x1b[200~{body}\x1b[201~"),
            _ => body.to_string(),
        }
    }
}

/// Prompt submit behavior for one channel operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitPolicy {
    pub prompt_submit: bool,
    pub settle: Duration,
    pub retries: u8,
    pub retry_delay: Duration,
    pub require_verification: bool,
}

impl SubmitPolicy {
    pub fn typing_only() -> Self {
        Self {
            prompt_submit: false,
            settle: Duration::ZERO,
            retries: 0,
            retry_delay: Duration::ZERO,
            require_verification: false,
        }
    }

    fn merged(self, other: Self) -> Self {
        Self {
            prompt_submit: self.prompt_submit || other.prompt_submit,
            settle: self.settle.max(other.settle),
            retries: self.retries.max(other.retries),
            retry_delay: self.retry_delay.max(other.retry_delay),
            require_verification: self.require_verification || other.require_verification,
        }
    }
}

impl Default for SubmitPolicy {
    fn default() -> Self {
        Self {
            prompt_submit: true,
            settle: Duration::from_millis(500),
            retries: 1,
            retry_delay: Duration::from_millis(750),
            require_verification: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendOptions {
    pub submit: SubmitPolicy,
    pub timeout: Duration,
}

impl Default for SendOptions {
    fn default() -> Self {
        Self {
            submit: SubmitPolicy::default(),
            timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnqueueOptions {
    pub submit: SubmitPolicy,
    pub quiet_guard: QuietGuardPolicy,
}

impl Default for EnqueueOptions {
    fn default() -> Self {
        Self {
            submit: SubmitPolicy::default(),
            quiet_guard: QuietGuardPolicy::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuietGuardPolicy {
    #[default]
    Default,
    Disabled,
}

impl QuietGuardPolicy {
    fn merged(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disabled, _) | (_, Self::Disabled) => Self::Disabled,
            (Self::Default, Self::Default) => Self::Default,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionOutcome {
    SubmittedVerified {
        message_id: MessageId,
        submitted_at: Instant,
    },
    SubmittedUnverified {
        message_id: MessageId,
        submitted_at: Instant,
    },
}

impl SubmissionOutcome {
    pub fn message_id(&self) -> MessageId {
        match self {
            Self::SubmittedVerified { message_id, .. }
            | Self::SubmittedUnverified { message_id, .. } => *message_id,
        }
    }

    pub fn verified(&self) -> bool {
        matches!(self, Self::SubmittedVerified { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedDelivery {
    pub message_id: MessageId,
    pub target: SessionKey,
    pub accepted_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageId(pub u64);

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "msg-{}", self.0)
    }
}

/// Agent UI profile used for composer state and submit verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiProfile {
    #[default]
    Generic,
    Codex,
    Claude,
}

impl UiProfile {
    pub async fn composer_state(&self, _target: &Target) -> Result<ComposerState, DeliveryError> {
        Ok(ComposerState::Unknown)
    }

    pub async fn verify_submitted(
        &self,
        _target: &Target,
    ) -> Result<SubmitVerification, DeliveryError> {
        Ok(SubmitVerification::Unknown)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposerState {
    Empty,
    NonEmpty,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitVerification {
    Submitted,
    StillComposed,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelStatus {
    pub target: SessionKey,
    pub pending_count: usize,
    pub flushing: bool,
    pub last_deferred_at: Option<Instant>,
    pub last_defer_reason: Option<DeferReason>,
    pub last_error: Option<String>,
}

impl ChannelStatus {
    fn new(target: SessionKey) -> Self {
        Self {
            target,
            pending_count: 0,
            flushing: false,
            last_deferred_at: None,
            last_defer_reason: None,
            last_error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeferReason {
    RecentWritableClientActivity {
        latest_client_activity: u64,
        latest_client_activity_age_secs: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryEvent {
    Accepted {
        message_id: MessageId,
        target: SessionKey,
        source: MessageSource,
        accepted_at: Instant,
    },
    Deferred {
        target: SessionKey,
        message_ids: Vec<MessageId>,
        reason: DeferReason,
        retry_after: Duration,
    },
    Coalesced {
        target: SessionKey,
        message_ids: Vec<MessageId>,
        segment_count: usize,
    },
    Submitted {
        target: SessionKey,
        message_ids: Vec<MessageId>,
        submitted_at: Instant,
        verified: bool,
    },
    Failed {
        target: SessionKey,
        message_ids: Vec<MessageId>,
        error: String,
    },
}

#[derive(Debug, Clone, Error)]
pub enum DeliveryError {
    #[error("target identity mismatch for {key:?}: existing target {existing_target}, new target {new_target}")]
    TargetIdentityMismatch {
        key: Box<SessionKey>,
        existing_target: String,
        new_target: String,
    },
    #[error("target unresolved: {target}")]
    TargetUnresolved { target: String },
    #[error("target stale: {target}: {reason}")]
    TargetStale { target: String, reason: String },
    #[error("tmux {operation} failed: {source}")]
    Tmux {
        operation: &'static str,
        #[source]
        source: TmuxErrorSource,
    },
    #[error("submit verification unavailable for {message_ids:?}")]
    VerificationUnavailable { message_ids: Vec<MessageId> },
    #[error("submit not confirmed for {message_ids:?}")]
    SubmitNotConfirmed { message_ids: Vec<MessageId> },
    #[error(
        "delivery timed out after {timeout:?} for {message_id}; still_pending={still_pending}"
    )]
    Timeout {
        message_id: MessageId,
        timeout: Duration,
        still_pending: bool,
    },
    #[error("channel closed")]
    ChannelClosed,
    #[error("delivery canceled for {message_id}: {reason}")]
    Canceled {
        message_id: MessageId,
        reason: String,
    },
    #[error("invalid message: {0}")]
    InvalidMessage(&'static str),
}

impl DeliveryError {
    pub fn tmux(operation: &'static str, source: motlie_tmux::Error) -> Self {
        Self::Tmux {
            operation,
            source: TmuxErrorSource(Arc::new(source)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TmuxErrorSource(Arc<motlie_tmux::Error>);

impl fmt::Display for TmuxErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.as_ref(), f)
    }
}

impl StdError for TmuxErrorSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

fn normalize_body(body: &str) -> String {
    body.replace("\r\n", "\n").replace('\r', "\n")
}

fn format_segments(batch: &[PendingSegment]) -> String {
    if batch.len() == 1 {
        let segment = &batch[0];
        if segment.sources.len() == 1 && segment.sources[0].omit_header_when_single() {
            return segment.body.clone();
        }
    }

    batch
        .iter()
        .map(|segment| {
            let source_labels = segment
                .sources
                .iter()
                .map(|source| source.label.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("[from: {source_labels}]\n{}", segment.body)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn merge_client_activity(activity: &mut SessionClientActivity, other: SessionClientActivity) {
    activity.attached_clients += other.attached_clients;
    activity.writable_clients += other.writable_clients;
    activity.latest_client_activity = max_option(
        activity.latest_client_activity,
        other.latest_client_activity,
    );
    activity.latest_writable_client_activity = max_option(
        activity.latest_writable_client_activity,
        other.latest_writable_client_activity,
    );
}

fn max_option(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn seconds_since_epoch(ts: u64) -> Option<u64> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(now.saturating_sub(ts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_tmux::transport::MockTransport;
    use motlie_tmux::{HostHandle, TransportKind};
    use std::sync::Mutex as TestMutex;

    fn config() -> ChannelConfig {
        ChannelConfig {
            input_quiet_for: Duration::ZERO,
            coalesce_window: Duration::ZERO,
            coalesce_max_wait: Duration::ZERO,
            default_submit: SubmitPolicy::default(),
            default_ui_profile: UiProfile::Generic,
            event_capacity: 64,
            preserve_composer: true,
        }
    }

    async fn mock_channel(
        mock: MockTransport,
        config: ChannelConfig,
    ) -> (Channel, Arc<TestMutex<Vec<String>>>) {
        let log = mock.command_log();
        let host = HostHandle::new(TransportKind::Mock(mock), None);
        let spec = motlie_tmux::TargetSpec::session("build");
        let target = host.target(&spec).await.unwrap().unwrap();
        let key = SessionKey::from_target("local", "local", &target);
        let manager = ChannelManager::new(config);
        let channel = manager
            .get_or_bind(ResolvedSession::new(key, host, target))
            .unwrap();
        (channel, log)
    }

    fn session_response() -> &'static str {
        "__MOTLIE_S__ build $0 100 0 1  200\n"
    }

    #[test]
    fn attributed_coalescing_uses_natural_headers() {
        let batch = vec![
            PendingSegment {
                message_id: MessageId(1),
                dedup_key: "first".to_string(),
                body: "first".to_string(),
                paste_mode: PasteMode::Bracketed,
                sources: vec![MessageSource::broadcast("broadcast")],
                submit: SubmitPolicy::default(),
                quiet_guard: QuietGuardPolicy::Default,
                coalesce: CoalescePolicy::Enabled,
                waiters: Vec::new(),
                has_async_source: false,
                deferred_once: false,
            },
            PendingSegment {
                message_id: MessageId(2),
                dedup_key: "second".to_string(),
                body: "second".to_string(),
                paste_mode: PasteMode::Bracketed,
                sources: vec![MessageSource::timer("timer:poll")],
                submit: SubmitPolicy::default(),
                quiet_guard: QuietGuardPolicy::Default,
                coalesce: CoalescePolicy::Enabled,
                waiters: Vec::new(),
                has_async_source: false,
                deferred_once: false,
            },
        ];

        assert_eq!(
            format_segments(&batch),
            "[from: broadcast]\nfirst\n\n[from: timer:poll]\nsecond"
        );
    }

    #[test]
    fn source_constructors_preserve_kind() {
        assert_eq!(MessageSource::human("h").kind(), SourceKind::Human);
        assert_eq!(MessageSource::timer("t").kind(), SourceKind::Timer);
        assert_eq!(MessageSource::broadcast("b").kind(), SourceKind::Broadcast);
        assert_eq!(MessageSource::external("e").kind(), SourceKind::External);
    }

    #[test]
    fn duplicate_sources_share_one_segment_header() {
        let mut segment = PendingSegment {
            message_id: MessageId(1),
            dedup_key: "same".to_string(),
            body: "same".to_string(),
            paste_mode: PasteMode::Bracketed,
            sources: vec![MessageSource::broadcast("broadcast")],
            submit: SubmitPolicy::default(),
            quiet_guard: QuietGuardPolicy::Default,
            coalesce: CoalescePolicy::Enabled,
            waiters: Vec::new(),
            deferred_once: false,
            has_async_source: false,
        };
        segment.add_source(MessageSource::timer("timer:poll"));
        segment.add_source(MessageSource::timer("timer:poll"));

        assert_eq!(
            format_segments(&[segment]),
            "[from: broadcast, timer:poll]\nsame"
        );
    }

    #[tokio::test]
    async fn send_uses_separate_payload_and_enter() {
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("send-keys", "")
            .with_response("send-keys", "");
        let (channel, log) = mock_channel(mock, config()).await;

        let outcome = channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "hello"),
                SendOptions {
                    submit: SubmitPolicy {
                        settle: Duration::ZERO,
                        retries: 0,
                        retry_delay: Duration::ZERO,
                        require_verification: false,
                        prompt_submit: true,
                    },
                    timeout: Duration::from_secs(1),
                },
            )
            .await
            .unwrap();

        assert!(!outcome.verified());
        let commands = log.lock().unwrap();
        let sends = commands
            .iter()
            .filter(|command| command.contains("send-keys"))
            .collect::<Vec<_>>();
        assert_eq!(sends.len(), 2);
        assert!(sends[0].contains("hello"));
        assert!(sends[1].contains("Enter"));
    }

    #[tokio::test]
    async fn generic_unknown_verification_runs_retries_before_unverified_success() {
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("send-keys", "")
            .with_response("send-keys", "")
            .with_response("send-keys", "")
            .with_response("send-keys", "");
        let (channel, log) = mock_channel(mock, config()).await;

        let outcome = channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "hello"),
                SendOptions {
                    submit: SubmitPolicy {
                        settle: Duration::ZERO,
                        retries: 2,
                        retry_delay: Duration::ZERO,
                        require_verification: false,
                        prompt_submit: true,
                    },
                    timeout: Duration::from_secs(1),
                },
            )
            .await
            .unwrap();

        assert!(!outcome.verified());
        let commands = log.lock().unwrap();
        let enter_sends = commands
            .iter()
            .filter(|command| command.contains("send-keys") && command.contains("Enter"))
            .count();
        assert_eq!(enter_sends, 3);
    }

    #[tokio::test]
    async fn identical_sync_sends_share_pending_segment_and_notify_both_waiters() {
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("send-keys", "")
            .with_response("send-keys", "");
        let mut cfg = config();
        cfg.coalesce_window = Duration::from_millis(20);
        let (channel, _log) = mock_channel(mock, cfg).await;
        let options = SendOptions {
            submit: SubmitPolicy {
                settle: Duration::ZERO,
                retries: 0,
                retry_delay: Duration::ZERO,
                require_verification: false,
                prompt_submit: true,
            },
            timeout: Duration::from_secs(1),
        };

        let first_channel = channel.clone();
        let second_channel = channel.clone();
        let first = first_channel.send(
            ManagedMessage::new(MessageSource::human("mstream.send"), "same"),
            options,
        );
        let second = second_channel.send(
            ManagedMessage::new(MessageSource::human("mstream.send"), "same"),
            options,
        );
        let (first, second) = tokio::join!(first, second);
        let first = first.unwrap();
        let second = second.unwrap();

        assert_eq!(first.message_id(), second.message_id());
    }

    #[tokio::test]
    async fn unique_dedup_keeps_same_body_as_separate_pending_segments() {
        let mock = MockTransport::new().with_response("list-sessions", session_response());
        let mut cfg = config();
        cfg.coalesce_window = Duration::from_secs(60);
        cfg.coalesce_max_wait = Duration::from_secs(60);
        let (channel, _log) = mock_channel(mock, cfg).await;

        let first = channel
            .enqueue(
                ManagedMessage::new(MessageSource::human("telnyx.turn.one"), "same")
                    .with_dedup(DedupPolicy::Unique),
                EnqueueOptions::default(),
            )
            .await
            .expect("first unique message accepted");
        let second = channel
            .enqueue(
                ManagedMessage::new(MessageSource::human("telnyx.turn.two"), "same")
                    .with_dedup(DedupPolicy::Unique),
                EnqueueOptions::default(),
            )
            .await
            .expect("second unique message accepted");

        assert_ne!(first.message_id, second.message_id);
        assert_eq!(channel.status().pending_count, 2);
        assert!(
            channel
                .cancel_pending(first.message_id, "test cleanup")
                .await
        );
        assert!(
            channel
                .cancel_pending(second.message_id, "test cleanup")
                .await
        );
    }

    #[tokio::test]
    async fn coalesce_disabled_bypasses_coalesce_window() {
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("send-keys", "");
        let mut cfg = config();
        cfg.coalesce_window = Duration::from_secs(60);
        cfg.coalesce_max_wait = Duration::from_secs(60);
        let (channel, _log) = mock_channel(mock, cfg).await;
        let mut events = channel.subscribe();

        let queued = channel
            .enqueue(
                ManagedMessage::new(MessageSource::human("telnyx.turn"), "send now")
                    .with_coalesce(CoalescePolicy::Disabled),
                EnqueueOptions {
                    submit: SubmitPolicy::typing_only(),
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
            .expect("message accepted");

        let submitted = timeout(Duration::from_secs(1), async {
            loop {
                if let DeliveryEvent::Submitted { message_ids, .. } = events.recv().await.unwrap() {
                    if message_ids.contains(&queued.message_id) {
                        return true;
                    }
                }
            }
        })
        .await
        .expect("coalesce-disabled message should submit without waiting");
        assert!(submitted);
    }

    #[tokio::test]
    async fn cancel_pending_removes_segment_and_emits_failed() {
        let mock = MockTransport::new().with_response("list-sessions", session_response());
        let mut cfg = config();
        cfg.coalesce_window = Duration::from_secs(60);
        cfg.coalesce_max_wait = Duration::from_secs(60);
        let (channel, _log) = mock_channel(mock, cfg).await;
        let mut events = channel.subscribe();
        let queued = channel
            .enqueue(
                ManagedMessage::new(MessageSource::human("telnyx.turn"), "cancel me"),
                EnqueueOptions::default(),
            )
            .await
            .expect("message accepted");

        assert!(channel.cancel_pending(queued.message_id, "barge-in").await);
        assert_eq!(channel.status().pending_count, 0);
        let failed = timeout(Duration::from_secs(1), async {
            loop {
                if let DeliveryEvent::Failed {
                    message_ids, error, ..
                } = events.recv().await.unwrap()
                {
                    if message_ids.contains(&queued.message_id) {
                        return error;
                    }
                }
            }
        })
        .await
        .expect("cancel should emit failed event");
        assert!(failed.contains("barge-in"));
    }

    #[tokio::test]
    async fn read_only_client_activity_does_not_defer_delivery() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let clients = format!("200 50 build $0 {now} 1 /dev/ttys001\n");
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("list-clients", &clients)
            .with_response("send-keys", "")
            .with_response("send-keys", "");
        let mut cfg = config();
        cfg.input_quiet_for = Duration::from_secs(10);
        let (channel, _log) = mock_channel(mock, cfg).await;

        let outcome = channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "hello"),
                SendOptions {
                    submit: SubmitPolicy {
                        settle: Duration::ZERO,
                        retries: 0,
                        retry_delay: Duration::ZERO,
                        require_verification: false,
                        prompt_submit: true,
                    },
                    timeout: Duration::from_secs(1),
                },
            )
            .await
            .unwrap();

        assert_eq!(outcome.message_id(), MessageId(1));
    }

    #[tokio::test]
    async fn recent_writable_activity_times_out_pure_sync_send_and_drops_pending() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let clients = format!("200 50 build $0 {now} 0 /dev/ttys001\n");
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("list-clients", &clients);
        let mut cfg = config();
        cfg.input_quiet_for = Duration::from_secs(10);
        let (channel, _log) = mock_channel(mock, cfg).await;

        let err = channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "hello"),
                SendOptions {
                    submit: SubmitPolicy::default(),
                    timeout: Duration::from_millis(10),
                },
            )
            .await
            .unwrap_err();

        match err {
            DeliveryError::Timeout { still_pending, .. } => assert!(!still_pending),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(channel.status().pending_count, 0);
    }

    #[tokio::test]
    async fn timed_out_sync_send_preserves_async_pending_work() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let clients = format!("200 50 build $0 {now} 0 /dev/ttys001\n");
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("list-clients", &clients);
        let mut cfg = config();
        cfg.input_quiet_for = Duration::from_secs(10);
        let (channel, _log) = mock_channel(mock, cfg).await;

        let send_channel = channel.clone();
        let send_task = tokio::spawn(async move {
            send_channel
                .send(
                    ManagedMessage::new(MessageSource::human("mstream.send"), "shared"),
                    SendOptions {
                        submit: SubmitPolicy::default(),
                        timeout: Duration::from_millis(50),
                    },
                )
                .await
        });
        tokio::task::yield_now().await;

        let queued = channel
            .enqueue(
                ManagedMessage::new(MessageSource::timer("mstream.timer"), "shared"),
                EnqueueOptions {
                    submit: SubmitPolicy::default(),
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
            .unwrap();
        let err = send_task.await.unwrap().unwrap_err();

        match err {
            DeliveryError::Timeout { still_pending, .. } => assert!(still_pending),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(queued.message_id, MessageId(1));
        assert_eq!(channel.status().pending_count, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn mixed_sync_async_collision_dedups_coalesces_and_notifies_waiter() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let old = now.saturating_sub(20);
        let recent_clients = format!("200 50 build $0 {now} 0 /dev/ttys001\n");
        let old_clients = format!("200 50 build $0 {old} 0 /dev/ttys001\n");
        let mock = MockTransport::new()
            .with_response("list-sessions", session_response())
            .with_response("list-clients", &recent_clients)
            .with_response("list-clients", &old_clients)
            .with_response("send-keys", "")
            .with_response("send-keys", "");
        let mut cfg = config();
        cfg.input_quiet_for = Duration::from_secs(10);
        let (channel, log) = mock_channel(mock, cfg).await;
        let mut events = channel.subscribe();

        let send_channel = channel.clone();
        let send_task = tokio::spawn(async move {
            send_channel
                .send(
                    ManagedMessage::new(MessageSource::human("mstream.send"), "shared"),
                    SendOptions {
                        submit: SubmitPolicy {
                            settle: Duration::ZERO,
                            retries: 0,
                            retry_delay: Duration::ZERO,
                            require_verification: false,
                            prompt_submit: true,
                        },
                        timeout: Duration::from_secs(20),
                    },
                )
                .await
        });

        loop {
            match events.recv().await.unwrap() {
                DeliveryEvent::Deferred { message_ids, .. } => {
                    assert_eq!(message_ids, vec![MessageId(1)]);
                    break;
                }
                _ => {}
            }
        }

        let deduped = channel
            .enqueue(
                ManagedMessage::new(MessageSource::timer("mstream.timer"), "shared"),
                EnqueueOptions {
                    submit: SubmitPolicy::default(),
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
            .unwrap();
        let other = channel
            .enqueue(
                ManagedMessage::new(MessageSource::broadcast("mstream.broadcast"), "other"),
                EnqueueOptions {
                    submit: SubmitPolicy::default(),
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
            .unwrap();

        assert_eq!(deduped.message_id, MessageId(1));
        assert_eq!(other.message_id, MessageId(2));

        tokio::time::advance(Duration::from_secs(10)).await;
        let outcome = send_task.await.unwrap().unwrap();
        assert_eq!(outcome.message_id(), MessageId(1));

        let mut saw_coalesced = false;
        let mut saw_submitted = false;
        while !saw_submitted {
            match events.recv().await.unwrap() {
                DeliveryEvent::Coalesced {
                    message_ids,
                    segment_count,
                    ..
                } => {
                    if message_ids == vec![MessageId(1), MessageId(2)] {
                        assert_eq!(segment_count, 2);
                        saw_coalesced = true;
                    }
                }
                DeliveryEvent::Submitted { message_ids, .. } => {
                    assert_eq!(message_ids, vec![MessageId(1), MessageId(2)]);
                    saw_submitted = true;
                }
                _ => {}
            }
        }
        assert!(saw_coalesced);

        let commands = log.lock().unwrap();
        let rendered = commands.join("\n");
        assert!(rendered.contains("mstream.send, mstream.timer"));
        assert!(rendered.contains("mstream.broadcast"));
        assert!(rendered.contains("shared"));
        assert!(rendered.contains("other"));
    }
}
