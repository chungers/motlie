//! Parameter types and ToolCall implementations for TTS tools.
//!
//! Each parameter type implements the `ToolCall` trait, binding it to its
//! execution logic using the persistent TTS worker process.

use super::TtsResource;
use crate::ToolCall;
use async_trait::async_trait;
use rmcp::{model::*, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;

/// Parameters for speaking text aloud.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SayParams {
    /// Array of text strings to speak in sequence
    #[schemars(description = "Array of text strings to speak, each will be spoken in order")]
    pub phrases: Vec<String>,

    /// Optional voice name (e.g., "Alex", "Samantha", "Daniel")
    #[schemars(
        description = "Optional voice name. Use list_voices tool to see available voices on your system"
    )]
    pub voice: Option<String>,

    /// Optional speech rate in words per minute (default: ~175-200)
    #[schemars(
        description = "Speech rate in words per minute. Default is system default (~175-200 wpm)"
    )]
    pub rate: Option<u32>,
}

#[async_trait]
impl ToolCall for SayParams {
    type Resource = TtsResource;

    async fn call(self, res: &TtsResource) -> Result<CallToolResult, McpError> {
        let engine = res.engine().await?;

        if self.phrases.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                json!({
                    "success": true,
                    "message": "No phrases to speak",
                    "spoken_count": 0
                })
                .to_string(),
            )]));
        }

        let mut queued_count = 0;
        let mut errors: Vec<String> = Vec::new();

        for phrase in &self.phrases {
            // Skip empty phrases
            if phrase.trim().is_empty() {
                continue;
            }

            // Queue the phrase to the persistent worker
            match engine
                .say(phrase, self.voice.as_deref(), self.rate)
                .await
            {
                Ok(()) => {
                    queued_count += 1;
                    tracing::info!("Queued phrase: \"{}\"", truncate_for_log(phrase, 50));
                }
                Err(e) => {
                    let msg = format!("Failed to queue phrase: {}", e);
                    tracing::error!("{}", msg);
                    errors.push(msg);
                }
            }
        }

        let success = errors.is_empty();
        let message = if success {
            format!(
                "Successfully queued {} phrase(s) for speaking",
                queued_count
            )
        } else {
            format!(
                "Queued {} of {} phrase(s) with {} error(s)",
                queued_count,
                self.phrases.len(),
                errors.len()
            )
        };

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": success,
                "message": message,
                "queued_count": queued_count,
                "total_phrases": self.phrases.len(),
                "errors": errors
            })
            .to_string(),
        )]))
    }
}

/// Parameters for listing available voices.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListVoicesParams {
    /// Optional filter to match voice names (case-insensitive substring match)
    #[schemars(description = "Optional filter to match voice names (case-insensitive)")]
    pub filter: Option<String>,
}

#[async_trait]
impl ToolCall for ListVoicesParams {
    type Resource = TtsResource;

    async fn call(self, res: &TtsResource) -> Result<CallToolResult, McpError> {
        let engine = res.engine().await?;

        let output = Command::new(engine.say_path())
            .arg("-v")
            .arg("?")
            .output()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to list voices: {}", e), None)
            })?;

        if !output.status.success() {
            return Err(McpError::internal_error(
                format!(
                    "Failed to list voices: 'say -v ?' returned status {}",
                    output.status
                ),
                None,
            ));
        }

        let voices_output = String::from_utf8_lossy(&output.stdout);

        // Parse voice list
        // Format: "Voice Name    language_code    # Description"
        let mut voices: Vec<VoiceInfo> = voices_output
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }

                // Split on multiple spaces to separate name from language
                let parts: Vec<&str> = line.splitn(2, "  ").collect();
                let name = parts.first()?.trim().to_string();

                if name.is_empty() {
                    return None;
                }

                // Extract language code if present
                let language = parts.get(1).and_then(|rest| {
                    let rest = rest.trim();
                    rest.split_whitespace().next().map(String::from)
                });

                Some(VoiceInfo { name, language })
            })
            .collect();

        // Apply filter if specified
        if let Some(ref filter) = self.filter {
            let filter_lower = filter.to_lowercase();
            voices.retain(|v| v.name.to_lowercase().contains(&filter_lower));
        }

        tracing::info!("Listed {} voices", voices.len());

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "voices": voices,
                "count": voices.len(),
                "filter": self.filter
            })
            .to_string(),
        )]))
    }
}

/// Information about an available voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoiceInfo {
    name: String,
    language: Option<String>,
}

/// Truncate a string for logging purposes (handles multi-byte UTF-8 correctly).
fn truncate_for_log(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}
