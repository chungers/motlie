use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::oneshot;

/// Metadata for request payloads carried by RequestEnvelope.
pub trait RequestMeta: Send + 'static {
    type Reply: Send + 'static;
    type Options: Default + Send + 'static;

    /// Stable label for tracing/metrics.
    fn request_kind(&self) -> &'static str;
}

/// Generic envelope for request/response over async channels.
pub struct RequestEnvelope<T: RequestMeta> {
    pub payload: T,
    pub options: T::Options,
    pub reply: Option<oneshot::Sender<anyhow::Result<T::Reply>>>,
    pub timeout: Option<Duration>,
    pub request_id: u64,
    pub created_at: Instant,
}

impl<T: RequestMeta> RequestEnvelope<T> {
    pub fn effective_timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub fn kind(&self) -> &'static str {
        self.payload.request_kind()
    }

    pub fn respond(&mut self, result: anyhow::Result<T::Reply>) {
        if let Some(reply) = self.reply.take() {
            let _ = reply.send(result);
        }
    }

    pub fn elapsed_nanos(&self) -> u64 {
        self.created_at.elapsed().as_nanos() as u64
    }
}

/// Generic response envelope for request/reply APIs.
#[derive(Debug, Clone)]
pub struct ReplyEnvelope<T> {
    pub request_id: u64,
    pub elapsed_time: u64,
    pub payload: T,
}

impl<T> ReplyEnvelope<T> {
    pub fn new(request_id: u64, elapsed_time: u64, payload: T) -> Self {
        Self {
            request_id,
            elapsed_time,
            payload,
        }
    }
}

pub fn new_request_id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
