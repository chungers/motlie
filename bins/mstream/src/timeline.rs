use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicCursor {
    pub version: u8,
    pub workstream: String,
    pub generation: u64,
    pub next_sequence: u64,
}

#[derive(Debug, Error)]
pub enum CursorError {
    #[error("cursor is not valid base64")]
    Base64,
    #[error("cursor is not valid JSON")]
    Json,
    #[error("cursor version {0} is not supported")]
    UnsupportedVersion(u8),
}

impl PublicCursor {
    pub fn new(workstream: impl Into<String>, generation: u64, next_sequence: u64) -> Self {
        Self {
            version: 1,
            workstream: workstream.into(),
            generation,
            next_sequence,
        }
    }

    pub fn encode(&self) -> anyhow::Result<String> {
        let bytes = serde_json::to_vec(self)?;
        Ok(URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn decode(value: &str) -> Result<Self, CursorError> {
        let bytes = URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| CursorError::Base64)?;
        let cursor: PublicCursor = serde_json::from_slice(&bytes).map_err(|_| CursorError::Json)?;
        if cursor.version != 1 {
            return Err(CursorError::UnsupportedVersion(cursor.version));
        }
        Ok(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_round_trips_generation() {
        let cursor = PublicCursor::new("pr-324", 7, 42);
        let encoded = cursor.encode().expect("encode");
        let decoded = PublicCursor::decode(&encoded).expect("decode");
        assert_eq!(decoded, cursor);
    }
}
