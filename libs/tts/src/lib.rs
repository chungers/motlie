//! Text-to-speech synthesis library for motlie.
//!
//! This crate provides the core TTS engine for macOS speech synthesis using
//! the system `say` command. It manages a persistent shell worker process
//! that speaks text sequentially, surviving even if the parent process exits.
//!
//! # Platform Support
//!
//! The `TtsEngine` is only available on macOS (`#[cfg(target_os = "macos")]`).
//! On other platforms, the engine type is not compiled. Cross-platform utilities
//! (`INSTRUCTIONS`, `truncate_for_log`) remain available for future backends.
//!
//! # Architecture
//!
//! The engine spawns a persistent shell worker that reads phrases from stdin
//! and speaks them using `/usr/bin/say`. This design ensures speech completes
//! even if the parent MCP server process exits (e.g., when stdin closes in
//! stdio transport mode).
//!
//! # Example
//!
//! ```ignore
//! use motlie_tts::TtsEngine;
//!
//! let engine = TtsEngine::new()?;
//! engine.say("Hello, world!", None, None).await?;
//!
//! // Graceful shutdown - waits for speech to complete
//! engine.close().await?;
//! ```

// =============================================================================
// macOS-only: TtsEngine and worker process management
// =============================================================================

#[cfg(target_os = "macos")]
mod macos {
    use std::process::Stdio;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tokio::process::{Child, ChildStdin};
    use tokio::sync::Mutex;

    /// The inline shell script that runs as a persistent worker.
    ///
    /// This script:
    /// 1. Reads tab-separated lines: phrase\tvoice\trate
    /// 2. Builds the appropriate `say` command arguments
    /// 3. Executes `say` and waits for completion
    /// 4. Repeats until stdin is closed (EOF)
    ///
    /// Using tab as delimiter allows phrases to contain spaces without escaping.
    const TTS_WORKER_SCRIPT: &str = r#"
while IFS=$'\t' read -r phrase voice rate; do
    args=()
    [ -n "$voice" ] && args+=(-v "$voice")
    [ -n "$rate" ] && args+=(-r "$rate")
    /usr/bin/say "${args[@]}" -- "$phrase"
done
"#;

    /// TTS engine handle for macOS speech synthesis.
    ///
    /// This struct manages a persistent shell worker process that handles
    /// text-to-speech requests. The worker survives even if the parent
    /// process exits, ensuring speech completes.
    ///
    /// # Worker Process
    ///
    /// On creation, `TtsEngine` spawns a shell process running an inline script
    /// that reads phrases from stdin and speaks them using the macOS `say` command.
    /// Phrases are sent as tab-separated values: `phrase\tvoice\trate\n`
    ///
    /// # Shutdown
    ///
    /// When `close()` is called (or stdin is dropped), the worker finishes
    /// speaking any queued phrases and exits gracefully.
    ///
    /// # Platform
    ///
    /// Only available on macOS (`#[cfg(target_os = "macos")]`).
    pub struct TtsEngine {
        /// Path to the 'say' command (for list_voices)
        say_path: String,
        /// Stdin pipe to send phrases to the worker
        worker_stdin: Mutex<ChildStdin>,
        /// Handle to the worker process
        worker_handle: Mutex<Child>,
    }

    impl TtsEngine {
        /// Create a new TTS engine, spawning the persistent worker process.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - The `say` command is not found at `/usr/bin/say`
        /// - Failed to spawn the worker process
        pub fn new() -> anyhow::Result<Self> {
            // Verify 'say' command exists
            let say_path = "/usr/bin/say".to_string();
            if !std::path::Path::new(&say_path).exists() {
                anyhow::bail!(
                    "macOS 'say' command not found at {}. Ensure you're running on a standard macOS installation.",
                    say_path
                );
            }

            // Spawn the persistent worker process using tokio
            let mut child = tokio::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(TTS_WORKER_SCRIPT)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| anyhow::anyhow!("Failed to spawn TTS worker: {}", e))?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("Failed to get stdin handle for TTS worker"))?;

            tracing::info!(
                "TTS engine initialized with persistent worker (pid: {:?})",
                child.id()
            );

            Ok(Self {
                say_path,
                worker_stdin: Mutex::new(stdin),
                worker_handle: Mutex::new(child),
            })
        }

        /// Get the path to the 'say' command.
        pub fn say_path(&self) -> &str {
            &self.say_path
        }

        /// Send a phrase to the worker for speaking.
        ///
        /// The phrase is queued and will be spoken after any previously queued phrases.
        /// This method returns immediately after queueing; it does not wait for speech
        /// to complete.
        ///
        /// # Arguments
        ///
        /// * `phrase` - The text to speak
        /// * `voice` - Optional voice name (e.g., "Alex", "Samantha")
        /// * `rate` - Optional speech rate in words per minute
        ///
        /// # Errors
        ///
        /// Returns an error if writing to the worker's stdin fails (e.g., worker crashed).
        pub async fn say(
            &self,
            phrase: &str,
            voice: Option<&str>,
            rate: Option<u32>,
        ) -> anyhow::Result<()> {
            // Format: phrase\tvoice\trate\n
            // Empty strings for optional fields
            let line = format!(
                "{}\t{}\t{}\n",
                phrase,
                voice.unwrap_or(""),
                rate.map(|r| r.to_string()).unwrap_or_default()
            );

            let mut stdin = self.worker_stdin.lock().await;
            stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to write to TTS worker: {}", e))?;
            stdin
                .flush()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to flush TTS worker stdin: {}", e))?;

            tracing::debug!(
                "Queued phrase for speaking: {:?}",
                super::truncate_for_log(phrase, 50)
            );
            Ok(())
        }

        /// Shut down the TTS engine gracefully.
        ///
        /// This method:
        /// 1. Closes the worker's stdin (signals EOF)
        /// 2. Waits for the worker to finish speaking and exit (with timeout)
        ///
        /// The worker will complete any phrases it has already started before exiting.
        ///
        /// Named `close()` (rather than `shutdown()`) to avoid ambiguity when
        /// `ResourceLifecycle::shutdown()` is implemented for this type in `motlie-mcp`.
        pub async fn close(self) -> anyhow::Result<()> {
            tracing::info!("TTS engine shutting down, waiting for worker to finish...");

            // Drop stdin to signal EOF to the worker
            // The worker will finish its current phrase and exit
            drop(self.worker_stdin);

            // Wait for worker to exit with a timeout
            let mut handle = self.worker_handle.into_inner();
            match tokio::time::timeout(Duration::from_secs(120), handle.wait()).await {
                Ok(Ok(status)) => {
                    if status.success() {
                        tracing::info!("TTS worker exited successfully");
                    } else {
                        tracing::warn!("TTS worker exited with status: {}", status);
                    }
                }
                Ok(Err(e)) => {
                    tracing::error!("Error waiting for TTS worker: {}", e);
                }
                Err(_) => {
                    tracing::warn!("TTS worker shutdown timeout, killing process");
                    let _ = handle.kill().await;
                }
            }

            tracing::info!("TTS engine shutdown complete");
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos::TtsEngine;

// =============================================================================
// Cross-platform utilities
// =============================================================================

/// Instructions for AI assistants using TTS tools.
pub const INSTRUCTIONS: &str = r#"
Text-to-speech tools using macOS speech synthesis.

## Available Tools

- **say**: Speak text aloud using the system voice
  - Supports multiple phrases spoken in sequence
  - Optional voice selection (use list_voices to see available voices)
  - Optional speech rate control (words per minute)

- **list_voices**: List all available system voices
  - Returns voice names that can be used with the 'say' tool

## Platform Support

These tools are only available on macOS systems. On other platforms,
the TTS module is not compiled.

## Example Usage

To speak a greeting:
```json
{"phrases": ["Hello, world!", "How are you today?"]}
```

To speak with a specific voice:
```json
{"phrases": ["Hello"], "voice": "Samantha"}
```

To speak slowly:
```json
{"phrases": ["This is spoken slowly"], "rate": 100}
```
"#;

/// Truncate a string for logging purposes.
pub fn truncate_for_log(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
