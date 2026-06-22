use std::collections::VecDeque;
use std::time::Instant;

use crate::quality::{ConversationPolicyConfig, PendingOutputOrder};

#[derive(Clone, Debug)]
pub struct PendingPolicyOutput<T> {
    pub sequence: u64,
    pub enqueued_at: Instant,
    pub payload: T,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PendingPolicyEnqueueOutcome {
    pub sequence: u64,
    pub pending_len: usize,
    pub dropped_count: usize,
    pub order: PendingOutputOrder,
}

#[derive(Debug)]
pub struct ConversationPolicyQueue<T> {
    next_sequence: u64,
    drain_running: bool,
    outputs: VecDeque<PendingPolicyOutput<T>>,
}

impl<T> Default for ConversationPolicyQueue<T> {
    fn default() -> Self {
        Self {
            next_sequence: 0,
            drain_running: false,
            outputs: VecDeque::new(),
        }
    }
}

impl<T> ConversationPolicyQueue<T> {
    pub fn enqueue(
        &mut self,
        config: &ConversationPolicyConfig,
        payload: T,
    ) -> PendingPolicyEnqueueOutcome {
        self.next_sequence = self.next_sequence.saturating_add(1);
        let sequence = self.next_sequence;
        let mut dropped_count = 0;
        let max_pending = config.max_pending_outputs.max(1);

        match config.pending_output_order {
            PendingOutputOrder::LatestOnly => {
                dropped_count = self.outputs.len();
                self.outputs.clear();
            }
            PendingOutputOrder::Fifo => {
                while self.outputs.len() >= max_pending {
                    self.outputs.pop_front();
                    dropped_count += 1;
                }
            }
        }

        self.outputs.push_back(PendingPolicyOutput {
            sequence,
            enqueued_at: Instant::now(),
            payload,
        });

        PendingPolicyEnqueueOutcome {
            sequence,
            pending_len: self.outputs.len(),
            dropped_count,
            order: config.pending_output_order,
        }
    }

    pub fn front(&self) -> Option<&PendingPolicyOutput<T>> {
        self.outputs.front()
    }

    pub fn take_next(&mut self) -> Option<PendingPolicyOutput<T>> {
        self.outputs.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub fn drain_running(&self) -> bool {
        self.drain_running
    }

    pub fn set_drain_running(&mut self, running: bool) {
        self.drain_running = running;
    }

    pub fn clear(&mut self) -> usize {
        let dropped = self.outputs.len();
        self.outputs.clear();
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{ConversationPolicyMode, PendingOutputOrder};

    fn config(order: PendingOutputOrder, max_pending_outputs: usize) -> ConversationPolicyConfig {
        ConversationPolicyConfig {
            mode: ConversationPolicyMode::NoBargeInBoundedPending,
            active_playback_hold_ms: 1_000,
            max_pending_outputs,
            pending_output_order: order,
            post_barge_in_silence_ms: 1_200,
        }
    }

    #[test]
    fn latest_only_replaces_existing_pending_output() {
        let mut queue = ConversationPolicyQueue::default();
        let config = config(PendingOutputOrder::LatestOnly, 3);

        assert_eq!(queue.enqueue(&config, "first").dropped_count, 0);
        let outcome = queue.enqueue(&config, "second");

        assert_eq!(outcome.dropped_count, 1);
        assert_eq!(outcome.pending_len, 1);
        assert_eq!(queue.take_next().expect("pending output").payload, "second");
        assert!(queue.is_empty());
    }

    #[test]
    fn fifo_drops_oldest_when_bounded_queue_is_full() {
        let mut queue = ConversationPolicyQueue::default();
        let config = config(PendingOutputOrder::Fifo, 2);

        queue.enqueue(&config, "first");
        queue.enqueue(&config, "second");
        let outcome = queue.enqueue(&config, "third");

        assert_eq!(outcome.dropped_count, 1);
        assert_eq!(outcome.pending_len, 2);
        assert_eq!(queue.take_next().expect("second output").payload, "second");
        assert_eq!(queue.take_next().expect("third output").payload, "third");
        assert!(queue.is_empty());
    }
}
