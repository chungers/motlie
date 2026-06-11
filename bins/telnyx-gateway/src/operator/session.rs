use crate::adapter::LiveAsrBackend;
use crate::operator::state::{CallSession, CallStatus, GatewayState};
use crate::tts::LiveTtsBackend;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorSession {
    pub selected_call: Option<String>,
    pub detail_scroll: u16,
    pub next_asr_backend: LiveAsrBackend,
    pub next_tts_backend: LiveTtsBackend,
}

impl Default for OperatorSession {
    fn default() -> Self {
        Self::new(LiveAsrBackend::default())
    }
}

impl OperatorSession {
    pub fn new(next_asr_backend: LiveAsrBackend) -> Self {
        Self {
            selected_call: None,
            detail_scroll: 0,
            next_asr_backend,
            next_tts_backend: LiveTtsBackend::default(),
        }
    }
}

impl OperatorSession {
    pub fn select_first(&mut self, state: &GatewayState) -> Option<String> {
        let selected = ordered_call_ids(state).into_iter().next()?;
        self.selected_call = Some(selected.clone());
        Some(selected)
    }

    pub fn ensure_valid_selection(&mut self, state: &GatewayState) {
        if self
            .selected_call
            .as_ref()
            .is_some_and(|call_id| state.calls.contains_key(call_id))
        {
            return;
        }
        self.selected_call = ordered_call_ids(state).into_iter().next();
        self.detail_scroll = 0;
    }

    pub fn select_call(&mut self, state: &mut GatewayState, call_id: &str) -> bool {
        if let Some(call) = state.calls.get_mut(call_id) {
            call.unread_events = 0;
            self.selected_call = Some(call_id.to_string());
            self.detail_scroll = 0;
            true
        } else {
            false
        }
    }

    pub fn move_selection(&mut self, state: &mut GatewayState, delta: isize) -> Option<String> {
        let call_ids = ordered_call_ids(state);
        if call_ids.is_empty() {
            self.selected_call = None;
            self.detail_scroll = 0;
            return None;
        }

        let current_index = self
            .selected_call
            .as_ref()
            .and_then(|selected| call_ids.iter().position(|call_id| call_id == selected))
            .unwrap_or(0);
        let next_index = if delta < 0 {
            current_index.saturating_sub(delta.unsigned_abs())
        } else {
            current_index
                .saturating_add(delta as usize)
                .min(call_ids.len() - 1)
        };
        let selected = call_ids[next_index].clone();
        let _ = self.select_call(state, &selected);
        Some(selected)
    }

    pub fn scroll_detail(&mut self, delta: i16) {
        if delta < 0 {
            self.detail_scroll = self.detail_scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.detail_scroll = self.detail_scroll.saturating_add(delta as u16);
        }
    }

    pub fn selected_call<'a>(&self, state: &'a GatewayState) -> Option<&'a CallSession> {
        let call_id = self.selected_call.as_ref()?;
        state.calls.get(call_id)
    }
}

pub fn ordered_call_ids(state: &GatewayState) -> Vec<String> {
    let mut calls = state.calls.values().collect::<Vec<_>>();
    calls.sort_by(|left, right| {
        call_sort_bucket(left.status)
            .cmp(&call_sort_bucket(right.status))
            .then_with(|| right.updated_at().cmp(&left.updated_at()))
            .then_with(|| left.gateway_call_id.cmp(&right.gateway_call_id))
    });
    calls
        .into_iter()
        .map(|call| call.gateway_call_id.clone())
        .collect()
}

fn call_sort_bucket(status: CallStatus) -> u8 {
    match status {
        CallStatus::PendingInbound => 0,
        CallStatus::Dialing
        | CallStatus::Answering
        | CallStatus::Answered
        | CallStatus::MediaStarted
        | CallStatus::Transcribing
        | CallStatus::Speaking => 1,
        CallStatus::Failed => 2,
        CallStatus::IgnoredInbound | CallStatus::Ended => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::state::{CallStatus, GatewayState, TelnyxIds};

    #[test]
    fn ordered_call_ids_puts_waiting_and_recent_calls_first() {
        let mut state = GatewayState::new("127.0.0.1:0".parse().expect("valid addr"));
        let ended = add_call(&mut state, "ended", CallStatus::Ended);
        let waiting_old = add_call(&mut state, "waiting-old", CallStatus::PendingInbound);
        let waiting_new = add_call(&mut state, "waiting-new", CallStatus::PendingInbound);
        state
            .calls
            .get_mut(&waiting_new)
            .expect("waiting call should exist")
            .push_timeline("newer activity");

        let ordered = ordered_call_ids(&state);

        assert_eq!(ordered[0], waiting_new);
        assert_eq!(ordered[1], waiting_old);
        assert_eq!(ordered[2], ended);
    }

    fn add_call(state: &mut GatewayState, control_id: &str, status: CallStatus) -> String {
        state.add_or_update_inbound_call(
            TelnyxIds {
                call_control_id: control_id.to_string(),
                call_session_id: None,
                call_leg_id: None,
                stream_id: None,
            },
            None,
            None,
            status,
        )
    }
}
