use serde::{Deserialize, Serialize};

const DEFAULT_BATCH_SIZE: usize = 1;
const DEFAULT_MAX_BATCH_WAIT_MS: u64 = 0;
const DEFAULT_MAX_IDLE_WAIT_MS: u64 = 0;
const DEFAULT_JOIN_SEPARATOR: &str = "\n";

/// Placement-neutral caller turn input for turn batching.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub epoch: u64,
}

impl Turn {
    pub fn new(turn_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            turn_id: turn_id.into(),
            utterance_id: None,
            text: text.into(),
            sequence: 0,
            epoch: 0,
        }
    }
}

/// Placement-neutral prompt emitted when a turn batch is complete.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub batch_id: String,
    pub epoch: u64,
    pub response_turn_id: String,
    pub source_turn_ids: Vec<String>,
    pub text: String,
}

/// Accumulation state emitted while the batcher is intentionally withholding a prompt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Accumulating {
    pub batch_id: String,
    pub epoch: u64,
    pub source_turn_ids: Vec<String>,
    pub deadline_ms: u64,
}

/// Reset state for hosts that need to publish an ordered turn batch reset.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TurnBatchReset {
    pub reason: TurnBatchResetReason,
    pub epoch: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnBatchResetReason {
    BargeIn,
    FinalTurnSuperseded,
    StaleGeneration,
    SessionEnd,
    Manual,
}

impl TurnBatchResetReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BargeIn => "barge_in",
            Self::FinalTurnSuperseded => "final_turn_superseded",
            Self::StaleGeneration => "stale_generation",
            Self::SessionEnd => "session_end",
            Self::Manual => "manual",
        }
    }
}

/// Output contract for the sync turn batcher.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BatchDecision {
    Accumulating(Accumulating),
    PromptComplete(Prompt),
    Reset(TurnBatchReset),
}

/// Pure, sync turn batching contract. It must remain independent of gateway-only timing,
/// transport, model, and async runtime types so it can run in either the gateway or daemon.
pub trait TurnBatcher {
    fn observe(&mut self, turn: Turn) -> BatchDecision;
    fn reset(&mut self, reason: TurnBatchResetReason) -> BatchDecision;
    fn epoch(&self) -> u64;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IdentityTurnBatcherConfig {
    pub fixed_batch_size: usize,
    pub max_batch_turns: usize,
    pub max_batch_wait_ms: u64,
    pub max_idle_wait_ms: u64,
    pub join_separator: String,
}

impl Default for IdentityTurnBatcherConfig {
    fn default() -> Self {
        Self {
            fixed_batch_size: DEFAULT_BATCH_SIZE,
            max_batch_turns: DEFAULT_BATCH_SIZE,
            max_batch_wait_ms: DEFAULT_MAX_BATCH_WAIT_MS,
            max_idle_wait_ms: DEFAULT_MAX_IDLE_WAIT_MS,
            join_separator: DEFAULT_JOIN_SEPARATOR.to_string(),
        }
    }
}

impl IdentityTurnBatcherConfig {
    pub fn batch_of_one() -> Self {
        Self::default()
    }

    pub fn fixed_batch_size(size: usize) -> Self {
        let size = size.max(1);
        Self {
            fixed_batch_size: size,
            max_batch_turns: size,
            ..Self::default()
        }
    }

    pub fn with_max_batch_turns(mut self, max_batch_turns: usize) -> Self {
        self.max_batch_turns = max_batch_turns.max(1);
        self
    }

    pub fn with_max_batch_wait_ms(mut self, max_batch_wait_ms: u64) -> Self {
        self.max_batch_wait_ms = max_batch_wait_ms;
        self
    }

    pub fn with_max_idle_wait_ms(mut self, max_idle_wait_ms: u64) -> Self {
        self.max_idle_wait_ms = max_idle_wait_ms;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityTurnBatcher {
    config: IdentityTurnBatcherConfig,
    epoch: u64,
    next_batch_index: u64,
    pending_batch_id: Option<String>,
    pending_turns: Vec<Turn>,
}

impl Default for IdentityTurnBatcher {
    fn default() -> Self {
        Self::new(IdentityTurnBatcherConfig::default())
    }
}

impl IdentityTurnBatcher {
    pub fn new(config: IdentityTurnBatcherConfig) -> Self {
        let fixed_batch_size = config.fixed_batch_size.max(1);
        let max_batch_turns = config.max_batch_turns.max(1);
        Self {
            config: IdentityTurnBatcherConfig {
                fixed_batch_size,
                max_batch_turns,
                ..config
            },
            epoch: 0,
            next_batch_index: 0,
            pending_batch_id: None,
            pending_turns: Vec::new(),
        }
    }

    pub fn has_pending_turns(&self) -> bool {
        !self.pending_turns.is_empty()
    }

    pub fn pending_batch_id(&self) -> Option<&str> {
        self.pending_batch_id.as_deref()
    }

    pub fn complete_pending(&mut self) -> Option<Prompt> {
        if self.pending_turns.is_empty() {
            return None;
        }
        Some(self.complete_current_batch())
    }

    fn batch_id(&mut self) -> String {
        if let Some(batch_id) = self.pending_batch_id.clone() {
            batch_id
        } else {
            let batch_id = format!("turn-batch-{}-{}", self.epoch, self.next_batch_index);
            self.next_batch_index = self.next_batch_index.saturating_add(1);
            self.pending_batch_id = Some(batch_id.clone());
            batch_id
        }
    }

    fn accumulating(&mut self) -> BatchDecision {
        let batch_id = self.batch_id();
        BatchDecision::Accumulating(Accumulating {
            batch_id,
            epoch: self.epoch,
            source_turn_ids: self
                .pending_turns
                .iter()
                .map(|turn| turn.turn_id.clone())
                .collect(),
            deadline_ms: if self.pending_turns.len() <= 1 {
                self.config.max_batch_wait_ms
            } else {
                self.config
                    .max_idle_wait_ms
                    .max(self.config.max_batch_wait_ms)
            },
        })
    }

    fn complete_current_batch(&mut self) -> Prompt {
        let batch_id = self.batch_id();
        let source_turn_ids = self
            .pending_turns
            .iter()
            .map(|turn| turn.turn_id.clone())
            .collect::<Vec<_>>();
        let response_turn_id = source_turn_ids
            .last()
            .cloned()
            .unwrap_or_else(|| batch_id.clone());
        let text = self
            .pending_turns
            .iter()
            .map(|turn| turn.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(&self.config.join_separator);
        self.pending_turns.clear();
        self.pending_batch_id = None;
        Prompt {
            batch_id,
            epoch: self.epoch,
            response_turn_id,
            source_turn_ids,
            text,
        }
    }
}

impl TurnBatcher for IdentityTurnBatcher {
    fn observe(&mut self, mut turn: Turn) -> BatchDecision {
        turn.epoch = self.epoch;
        self.pending_turns.push(turn);
        if self.pending_turns.len() >= self.config.fixed_batch_size
            || self.pending_turns.len() >= self.config.max_batch_turns
        {
            BatchDecision::PromptComplete(self.complete_current_batch())
        } else {
            self.accumulating()
        }
    }

    fn reset(&mut self, reason: TurnBatchResetReason) -> BatchDecision {
        let batch_id = self.pending_batch_id.take();
        self.pending_turns.clear();
        self.epoch = self.epoch.saturating_add(1);
        BatchDecision::Reset(TurnBatchReset {
            reason,
            epoch: self.epoch,
            batch_id,
        })
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(id: &str, text: &str) -> Turn {
        Turn::new(id, text)
    }

    #[test]
    fn identity_turn_batcher_batch_of_one_completes_immediately() {
        let mut handler = IdentityTurnBatcher::default();

        let decision = handler.observe(turn("turn-1", "hello"));

        assert_eq!(
            decision,
            BatchDecision::PromptComplete(Prompt {
                batch_id: "turn-batch-0-0".to_string(),
                epoch: 0,
                response_turn_id: "turn-1".to_string(),
                source_turn_ids: vec!["turn-1".to_string()],
                text: "hello".to_string(),
            })
        );
    }

    #[test]
    fn identity_turn_batcher_fixed_n_accumulates_then_completes() {
        let mut handler = IdentityTurnBatcher::new(
            IdentityTurnBatcherConfig::fixed_batch_size(2).with_max_batch_wait_ms(250),
        );

        assert_eq!(
            handler.observe(turn("turn-1", "first")),
            BatchDecision::Accumulating(Accumulating {
                batch_id: "turn-batch-0-0".to_string(),
                epoch: 0,
                source_turn_ids: vec!["turn-1".to_string()],
                deadline_ms: 250,
            })
        );
        assert_eq!(
            handler.observe(turn("turn-2", "second")),
            BatchDecision::PromptComplete(Prompt {
                batch_id: "turn-batch-0-0".to_string(),
                epoch: 0,
                response_turn_id: "turn-2".to_string(),
                source_turn_ids: vec!["turn-1".to_string(), "turn-2".to_string()],
                text: "first\nsecond".to_string(),
            })
        );
    }

    #[test]
    fn identity_turn_batcher_reset_drops_pending_and_advances_epoch() {
        let mut handler = IdentityTurnBatcher::new(IdentityTurnBatcherConfig::fixed_batch_size(2));
        let _ = handler.observe(turn("turn-1", "first"));

        assert_eq!(
            handler.reset(TurnBatchResetReason::BargeIn),
            BatchDecision::Reset(TurnBatchReset {
                reason: TurnBatchResetReason::BargeIn,
                epoch: 1,
                batch_id: Some("turn-batch-0-0".to_string()),
            })
        );
        assert_eq!(handler.epoch(), 1);
        assert_eq!(
            handler.observe(turn("turn-2", "second")),
            BatchDecision::Accumulating(Accumulating {
                batch_id: "turn-batch-1-1".to_string(),
                epoch: 1,
                source_turn_ids: vec!["turn-2".to_string()],
                deadline_ms: 0,
            })
        );
    }

    #[test]
    fn identity_turn_batcher_complete_pending_enforces_latency_fallback() {
        let mut handler = IdentityTurnBatcher::new(IdentityTurnBatcherConfig::fixed_batch_size(3));
        let _ = handler.observe(turn("turn-1", "first"));
        let _ = handler.observe(turn("turn-2", "second"));

        assert_eq!(
            handler.complete_pending(),
            Some(Prompt {
                batch_id: "turn-batch-0-0".to_string(),
                epoch: 0,
                response_turn_id: "turn-2".to_string(),
                source_turn_ids: vec!["turn-1".to_string(), "turn-2".to_string()],
                text: "first\nsecond".to_string(),
            })
        );
        assert!(!handler.has_pending_turns());
    }

    #[test]
    fn identity_turn_batcher_is_placement_neutral_for_same_turn_sequence() {
        let config = IdentityTurnBatcherConfig::fixed_batch_size(2);
        let mut gateway_host = IdentityTurnBatcher::new(config.clone());
        let mut daemon_host = IdentityTurnBatcher::new(config);
        let turns = [turn("turn-1", "first"), turn("turn-2", "second")];

        let gateway_decisions = turns
            .iter()
            .cloned()
            .map(|turn| gateway_host.observe(turn))
            .collect::<Vec<_>>();
        let daemon_decisions = turns
            .iter()
            .cloned()
            .map(|turn| daemon_host.observe(turn))
            .collect::<Vec<_>>();

        assert_eq!(gateway_decisions, daemon_decisions);
    }

    #[test]
    fn turn_and_prompt_are_serializable_without_gateway_timing() {
        let turn = Turn {
            turn_id: "turn-1".to_string(),
            utterance_id: Some("utt-1".to_string()),
            text: "hello".to_string(),
            sequence: 7,
            epoch: 3,
        };
        let encoded = serde_json::to_value(&turn).expect("turn serializes");
        assert_eq!(encoded["turn_id"], "turn-1");
        assert!(encoded.get("finalized_at").is_none());
        assert!(encoded.get("caller_turn_sent_at").is_none());

        let prompt = Prompt {
            batch_id: "turn-batch-3-0".to_string(),
            epoch: 3,
            response_turn_id: "turn-1".to_string(),
            source_turn_ids: vec!["turn-1".to_string()],
            text: "hello".to_string(),
        };
        let encoded = serde_json::to_value(&prompt).expect("prompt serializes");
        assert_eq!(encoded["text"], "hello");
        assert!(encoded.get("TextCallTurnTiming").is_none());
    }
}
