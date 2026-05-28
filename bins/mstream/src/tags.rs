use chrono::{DateTime, Utc};
use motlie_tmux::Target;

use crate::protocol::AgentState;

pub const PREFIX: &str = "mstream";

pub async fn set_many(target: &Target, pairs: &[(&str, String)]) -> anyhow::Result<()> {
    let tags = target.tags(PREFIX).await?;
    for (key, value) in pairs {
        tags.set(key, value).await?;
    }
    Ok(())
}

pub async fn unset_many(target: &Target, keys: &[&str]) -> anyhow::Result<()> {
    let tags = target.tags(PREFIX).await?;
    for key in keys {
        tags.unset(key).await?;
    }
    Ok(())
}

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
