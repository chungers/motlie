use std::fmt;
use std::str::FromStr;

use motlie_tmux::TargetSpec;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TargetParseError {
    #[error("target must use '<host-alias>::<tmux-session-name>'")]
    MissingDelimiter,
    #[error("target host alias cannot be empty")]
    EmptyHost,
    #[error("target session name cannot be empty")]
    EmptySession,
    #[error("target must contain exactly one '::' delimiter")]
    TooManyDelimiters,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionTarget {
    host_alias: String,
    session_name: String,
}

impl SessionTarget {
    pub fn new(
        host_alias: impl Into<String>,
        session_name: impl Into<String>,
    ) -> Result<Self, TargetParseError> {
        let host_alias = host_alias.into();
        let session_name = session_name.into();
        if host_alias.is_empty() {
            return Err(TargetParseError::EmptyHost);
        }
        if session_name.is_empty() {
            return Err(TargetParseError::EmptySession);
        }
        Ok(Self {
            host_alias,
            session_name,
        })
    }

    pub fn host_alias(&self) -> &str {
        &self.host_alias
    }

    pub fn session_name(&self) -> &str {
        &self.session_name
    }

    pub fn target_spec(&self) -> TargetSpec {
        TargetSpec::session(&self.session_name)
    }
}

impl fmt::Display for SessionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.host_alias, self.session_name)
    }
}

impl FromStr for SessionTarget {
    type Err = TargetParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let Some((host, session)) = value.split_once("::") else {
            return Err(TargetParseError::MissingDelimiter);
        };
        if session.contains("::") {
            return Err(TargetParseError::TooManyDelimiters);
        }
        Self::new(host, session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_target() {
        let target: SessionTarget = "amd1::codex-reviewer".parse().expect("valid target");
        assert_eq!(target.host_alias(), "amd1");
        assert_eq!(target.session_name(), "codex-reviewer");
        assert_eq!(target.target_spec().session_name(), "codex-reviewer");
        assert_eq!(target.to_string(), "amd1::codex-reviewer");
    }

    #[test]
    fn rejects_malformed_targets() {
        assert_eq!(
            "missing".parse::<SessionTarget>().unwrap_err(),
            TargetParseError::MissingDelimiter
        );
        assert_eq!(
            "::session".parse::<SessionTarget>().unwrap_err(),
            TargetParseError::EmptyHost
        );
        assert_eq!(
            "host::".parse::<SessionTarget>().unwrap_err(),
            TargetParseError::EmptySession
        );
        assert_eq!(
            "a::b::c".parse::<SessionTarget>().unwrap_err(),
            TargetParseError::TooManyDelimiters
        );
    }
}
