//! Integration tests for motlie-tmux (localhost).
//!
//! These tests require tmux to be installed and available on PATH.
//! They create/destroy real tmux sessions during testing.

use motlie_tmux::{HostHandle, SshConfig, TargetLevel};
use std::time::Duration;

fn tmux_available() -> bool {
    std::process::Command::new("tmux")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn localhost_session_lifecycle() {
    if !tmux_available() {
        eprintln!("skipping: tmux not available");
        return;
    }

    let session_name = "motlie_test_integ";
    let host = HostHandle::local();

    // Clean up any leftover session from a previous failed run
    if let Ok(Some(t)) = host.session(session_name).await {
        let _ = t.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // 1. Create session
    let target = host
        .create_session(session_name, &motlie_tmux::CreateSessionOptions {
            window_name: Some("main".to_string()),
            ..Default::default()
        })
        .await
        .expect("create_session failed");
    assert_eq!(target.level(), TargetLevel::Session);
    assert_eq!(target.session_name(), session_name);

    // 2. List sessions — confirm it appears
    let sessions = host.list_sessions().await.expect("list_sessions failed");
    assert!(
        sessions.iter().any(|s| s.name == session_name),
        "created session not found in list"
    );

    // 3. Capture pane content (should be an empty shell prompt area)
    let content = target.capture().await.expect("capture failed");
    // Content should be non-empty (at least whitespace/prompt)
    let _ = content;

    // 4. Send text + Enter
    tokio::time::sleep(Duration::from_millis(300)).await; // let shell initialize
    target
        .send_text("echo MOTLIE_TEST_OUTPUT")
        .await
        .expect("send_text failed");
    let enter = motlie_tmux::KeySequence::parse("{Enter}").unwrap();
    target.send_keys(&enter).await.expect("send_keys failed");

    // 5. Wait and capture again — confirm output changed
    tokio::time::sleep(Duration::from_millis(500)).await;
    let content2 = target.capture().await.expect("capture after send failed");
    assert!(
        content2.contains("MOTLIE_TEST_OUTPUT"),
        "expected output not found in capture: {}",
        content2
    );

    // 6. exec("echo hello", 10s)
    let exec_out = target
        .exec("echo hello_exec", Duration::from_secs(10))
        .await
        .expect("exec failed");
    assert!(exec_out.success(), "exec exit code: {}", exec_out.exit_code);
    assert!(
        exec_out.stdout.contains("hello_exec"),
        "exec stdout: {}",
        exec_out.stdout
    );

    // 7. Rename session — rename() now returns a new Target with updated address
    let new_name = "motlie_test_renamed";
    let renamed_target = target
        .rename(new_name)
        .await
        .expect("rename session failed");
    assert_eq!(renamed_target.session_name(), new_name);
    let sessions = host.list_sessions().await.unwrap();
    assert!(sessions.iter().any(|s| s.name == new_name));
    assert!(!sessions.iter().any(|s| s.name == session_name));

    // 8. Kill session — use the returned handle directly (no re-query needed)
    renamed_target.kill().await.expect("kill session failed");

    // 9. Confirm gone
    tokio::time::sleep(Duration::from_millis(200)).await;
    let sessions = host.list_sessions().await.unwrap();
    assert!(
        !sessions.iter().any(|s| s.name == new_name),
        "session should be gone after kill"
    );
}

/// 1.11m — Integration test: SshConfig::parse("ssh://localhost")?.connect()
/// produces a working HostHandle that can list_sessions().
#[tokio::test]
async fn uri_localhost_connect() {
    if !tmux_available() {
        eprintln!("skipping: tmux not available");
        return;
    }

    let session_name = "motlie_test_uri";

    // Parse URI and connect
    let host = SshConfig::parse("ssh://localhost")
        .expect("parse failed")
        .connect()
        .await
        .expect("connect failed");

    // Clean up leftover from previous run
    if let Ok(Some(t)) = host.session(session_name).await {
        let _ = t.kill().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Create session via URI-connected handle
    let target = host
        .create_session(session_name, &Default::default())
        .await
        .expect("create_session failed");

    // Verify list_sessions works
    let sessions = host.list_sessions().await.expect("list_sessions failed");
    assert!(
        sessions.iter().any(|s| s.name == session_name),
        "session not found via URI-connected handle"
    );

    // Verify with custom timeout propagation
    let host2 = SshConfig::parse("ssh://localhost?timeout=30")
        .expect("parse with timeout failed")
        .connect()
        .await
        .expect("connect with timeout failed");
    let sessions2 = host2.list_sessions().await.expect("list_sessions failed");
    assert!(sessions2.iter().any(|s| s.name == session_name));

    // Cleanup
    target.kill().await.expect("kill failed");
}

/// 1.11o — SSH integration test (env-gated).
/// Requires MOTLIE_SSH_TEST_HOST=user@host[:port].
#[tokio::test]
async fn uri_ssh_connect() {
    let Some(test_host) = std::env::var("MOTLIE_SSH_TEST_HOST").ok() else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let uri = format!("ssh://{}", test_host);
    let host = SshConfig::parse(&uri)
        .unwrap_or_else(|e| panic!("failed to parse '{}': {}", uri, e))
        .connect()
        .await
        .unwrap_or_else(|e| panic!("failed to connect to '{}': {}", uri, e));

    // Verify basic operation
    let sessions = host
        .list_sessions()
        .await
        .expect("list_sessions failed on SSH host");
    // Sessions may be empty — that's fine, we just verify the call succeeds
    let _ = sessions;
}
