use chrono::{DateTime, Utc};

use crate::protocol::AgentState;

pub const PREFIX: &str = "mstream";

pub fn now_tag() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn state_value(state: AgentState) -> String {
    state.as_str().to_string()
}

pub fn parse_updated_at(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}
