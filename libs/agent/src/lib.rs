//! Agent interaction policy over `motlie-tmux`.
//!
//! The first surface is managed prompt delivery through a process-local
//! [`Channel`] keyed by a stable tmux session identity.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use motlie_tmux::{HostHandle, KeySequence, Target};
use thiserror::Error;
use tokio::sync::{broadcast, oneshot, Mutex};
use tokio::time::{sleep, timeout};

const DEFAULT_EVENT_CAPACITY: usize = 1024;
const COMPOSER_SEPARATOR: &str = "\n\n---\n\n";

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
        let mut channels = self
            .inner
            .channels
            .lock()
            .expect("channel manager poisoned");
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
                event_tx: self.inner.event_tx.clone(),
            }),
        };
        channels.insert(key, channel.clone());
        Ok(channel)
    }

    pub fn remove(&self, key: &SessionKey) -> Option<Channel> {
        self.inner
            .channels
            .lock()
            .expect("channel manager poisoned")
            .remove(key)
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
        let message_id = self
            .accept(
                message,
                options.submit,
                QuietGuardPolicy::Default,
                Some(waiter_tx),
            )
            .await?;

        match timeout(options.timeout, waiter_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_closed)) => Err(DeliveryError::ChannelClosed),
            Err(_elapsed) => Err(DeliveryError::Timeout {
                message_id,
                timeout: options.timeout,
                still_pending: self.is_pending(message_id).await,
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
        self.inner
            .status
            .lock()
            .expect("channel status poisoned")
            .clone()
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
        let dedup_key = normalize_body(&message.body);
        let mut start_worker = false;
        let mut coalesced = false;
        let message_id;
        let pending_count;
        {
            let mut queue = self.inner.queue.lock().await;
            if let Some(segment) = queue
                .pending
                .iter_mut()
                .find(|segment| segment.dedup_key == dedup_key)
            {
                message_id = segment.message_id;
                segment.add_source(source.clone());
                segment.submit = segment.submit.merged(submit);
                segment.paste_mode = segment.paste_mode.merged(message.paste_mode);
                segment.quiet_guard = segment.quiet_guard.merged(quiet_guard);
                if let Some(waiter) = waiter {
                    segment.waiters.push(waiter);
                }
                coalesced = true;
            } else {
                message_id = MessageId(self.inner.next_message_id.fetch_add(1, Ordering::Relaxed));
                queue.pending.push_back(PendingSegment {
                    message_id,
                    dedup_key,
                    body: message.body,
                    paste_mode: message.paste_mode,
                    sources: vec![source.clone()],
                    submit,
                    quiet_guard,
                    waiters: waiter.into_iter().collect(),
                    deferred_once: false,
                });
            }

            pending_count = queue.pending.len();
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
        if coalesced {
            self.emit(DeliveryEvent::Coalesced {
                target: self.inner.key.clone(),
                message_ids: vec![message_id],
                segment_count: 1,
            });
        }
        if start_worker {
            let channel = self.clone();
            tokio::spawn(async move {
                channel.flush_loop().await;
            });
        }
        Ok(message_id)
    }

    async fn is_pending(&self, message_id: MessageId) -> bool {
        self.inner
            .queue
            .lock()
            .await
            .pending
            .iter()
            .any(|segment| segment.message_id == message_id)
    }

    async fn flush_loop(self) {
        loop {
            if !self.inner.config.coalesce_window.is_zero() {
                sleep(self.inner.config.coalesce_window).await;
            }

            if let Err(error) = self.apply_quiet_guard().await {
                let batch = self.drain_pending().await;
                self.fail_batch(batch, error).await;
                self.finish_if_empty().await;
                return;
            }

            let batch = match self.next_batch().await {
                Some(batch) => batch,
                None => return,
            };

            if batch.len() > 1 {
                self.emit(DeliveryEvent::Coalesced {
                    target: self.inner.key.clone(),
                    message_ids: batch.iter().map(|segment| segment.message_id).collect(),
                    segment_count: batch.len(),
                });
            }

            match self.submit_batch(&batch).await {
                Ok(result) => self.complete_batch(batch, result).await,
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
            let activity = self
                .inner
                .host
                .session_client_activity(self.inner.target.session_name())
                .await
                .map_err(|err| DeliveryError::tmux("session_client_activity", err))?;
            let Some(latest) = activity.latest_writable_client_activity else {
                return Ok(());
            };
            let age = seconds_since_epoch(latest).unwrap_or(0);
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
        let mut queue = self.inner.queue.lock().await;
        if queue.pending.is_empty() {
            queue.flushing = false;
            self.update_status(|status| {
                status.pending_count = 0;
                status.flushing = false;
            });
            return None;
        }
        let batch = queue.pending.drain(..).collect::<Vec<_>>();
        self.update_status(|status| {
            status.pending_count = 0;
            status.flushing = true;
        });
        Some(batch)
    }

    async fn drain_pending(&self) -> Vec<PendingSegment> {
        let mut queue = self.inner.queue.lock().await;
        let batch = queue.pending.drain(..).collect::<Vec<_>>();
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
            queue.flushing = false;
            self.update_status(|status| {
                status.pending_count = 0;
                status.flushing = false;
            });
        }
    }

    async fn submit_batch(
        &self,
        batch: &[PendingSegment],
    ) -> Result<BatchSubmitResult, DeliveryError> {
        let body = self.assemble_body(batch).await?;
        let paste_mode = batch.iter().fold(PasteMode::Literal, |mode, segment| {
            mode.merged(segment.paste_mode)
        });
        let submit = batch
            .iter()
            .fold(SubmitPolicy::typing_only(), |policy, segment| {
                policy.merged(segment.submit)
            });

        let payload = paste_mode.wrap_payload(&body);
        let literal = KeySequence::literal(&payload);
        self.inner
            .target
            .send_keys(&literal)
            .await
            .map_err(|err| DeliveryError::tmux("send_payload", err))?;

        if !submit.prompt_submit {
            return Ok(BatchSubmitResult {
                submitted_at: Instant::now(),
                verified: false,
            });
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
                    return Ok(BatchSubmitResult {
                        submitted_at,
                        verified: true,
                    });
                }
                SubmitVerification::Unknown if !submit.require_verification => {
                    return Ok(BatchSubmitResult {
                        submitted_at,
                        verified: false,
                    });
                }
                SubmitVerification::Unknown => {
                    if attempt == submit.retries {
                        return Err(DeliveryError::VerificationUnavailable {
                            message_ids: batch.iter().map(|segment| segment.message_id).collect(),
                        });
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
                let _ = waiter.send(Ok(outcome.clone()));
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
                let _ = waiter.send(Err(error.clone()));
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
        let mut status = self.inner.status.lock().expect("channel status poisoned");
        update(&mut status);
    }
}

#[derive(Default)]
struct QueueState {
    pending: VecDeque<PendingSegment>,
    flushing: bool,
}

struct PendingSegment {
    message_id: MessageId,
    dedup_key: String,
    body: String,
    paste_mode: PasteMode,
    sources: Vec<MessageSource>,
    submit: SubmitPolicy,
    quiet_guard: QuietGuardPolicy,
    waiters: Vec<Waiter>,
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

struct BatchSubmitResult {
    submitted_at: Instant,
    verified: bool,
}

type Waiter = oneshot::Sender<Result<SubmissionOutcome, DeliveryError>>;

/// Message accepted by a channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedMessage {
    pub source: MessageSource,
    pub body: String,
    pub paste_mode: PasteMode,
}

impl ManagedMessage {
    pub fn new(source: MessageSource, body: impl Into<String>) -> Self {
        Self {
            source,
            body: body.into(),
            paste_mode: PasteMode::Bracketed,
        }
    }

    pub fn with_paste_mode(mut self, paste_mode: PasteMode) -> Self {
        self.paste_mode = paste_mode;
        self
    }
}

/// Human-readable source attribution for coalesced prompt bodies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageSource {
    label: String,
    omit_header_when_single: bool,
}

impl MessageSource {
    pub fn human(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            omit_header_when_single: true,
        }
    }

    pub fn timer(label: impl Into<String>) -> Self {
        Self::attributed(label)
    }

    pub fn broadcast(label: impl Into<String>) -> Self {
        Self::attributed(label)
    }

    pub fn external(label: impl Into<String>) -> Self {
        Self::attributed(label)
    }

    pub fn attributed(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            omit_header_when_single: false,
        }
    }

    pub fn label(&self) -> &str {
        &self.label
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

#[derive(Debug, Clone, Error, PartialEq, Eq)]
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
    #[error("tmux {operation} failed: {message}")]
    Tmux {
        operation: &'static str,
        message: String,
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
    #[error("invalid message: {0}")]
    InvalidMessage(&'static str),
}

impl DeliveryError {
    pub fn tmux(operation: &'static str, error: impl fmt::Display) -> Self {
        Self::Tmux {
            operation,
            message: error.to_string(),
        }
    }
}

fn normalize_body(body: &str) -> String {
    body.replace("\r\n", "\n").replace('\r', "\n")
}

fn format_segments(batch: &[PendingSegment]) -> String {
    if batch.len() == 1 {
        let segment = &batch[0];
        if segment.sources.len() == 1 && segment.sources[0].omit_header_when_single {
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
                waiters: Vec::new(),
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
                waiters: Vec::new(),
                deferred_once: false,
            },
        ];

        assert_eq!(
            format_segments(&batch),
            "[from: broadcast]\nfirst\n\n[from: timer:poll]\nsecond"
        );
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
            waiters: Vec::new(),
            deferred_once: false,
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
    async fn read_only_client_activity_does_not_defer_delivery() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let clients = format!("200 50 build {now} 1 /dev/ttys001\n");
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
    async fn recent_writable_activity_times_out_sync_send_but_keeps_pending() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let clients = format!("200 50 build {now} 0 /dev/ttys001\n");
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
            DeliveryError::Timeout { still_pending, .. } => assert!(still_pending),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(channel.status().pending_count, 1);
    }
}
