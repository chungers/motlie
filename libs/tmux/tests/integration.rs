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

// ---------------------------------------------------------------------------
// File transfer integration tests (DC23, Phase 1.13h)
// These run unconditionally — no tmux gate needed.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn localhost_file_upload_download_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let content: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let src = tmp.path().join("test.bin");
    std::fs::write(&src, &content).unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions::default();

    // Upload
    let remote = tmp.path().join("remote.bin");
    host.upload(&src, &remote, &opts).await.unwrap();
    assert_eq!(std::fs::read(&remote).unwrap(), content);

    // Download back
    let restored = tmp.path().join("restored.bin");
    host.download(&remote, &restored, &opts).await.unwrap();
    assert_eq!(std::fs::read(&restored).unwrap(), content);
}

#[tokio::test]
async fn localhost_dir_upload_download_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();

    // Build nested source tree
    let src = tmp.path().join("project");
    std::fs::create_dir_all(src.join("src").join("nested")).unwrap();
    std::fs::write(src.join("Cargo.toml"), b"[package]").unwrap();
    std::fs::write(src.join("src").join("main.rs"), b"fn main() {}").unwrap();
    std::fs::write(src.join("src").join("nested").join("mod.rs"), b"// mod").unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Upload (copy-as: dest doesn't exist)
    let remote = tmp.path().join("deployed");
    host.upload(&src, &remote, &opts).await.unwrap();
    assert_eq!(std::fs::read(remote.join("Cargo.toml")).unwrap(), b"[package]");
    assert_eq!(
        std::fs::read(remote.join("src").join("nested").join("mod.rs")).unwrap(),
        b"// mod"
    );

    // Download back
    let restored = tmp.path().join("restored");
    host.download(&remote, &restored, &opts).await.unwrap();
    assert_eq!(
        std::fs::read(restored.join("src").join("main.rs")).unwrap(),
        b"fn main() {}"
    );
}

#[tokio::test]
async fn localhost_copy_into_vs_copy_as() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("app");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("f.txt"), b"data").unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Copy-as: dest doesn't exist → copied AS that path
    let dest1 = tmp.path().join("new_app");
    host.upload(&src, &dest1, &opts).await.unwrap();
    assert!(dest1.join("f.txt").exists());

    // Copy-into: dest exists as dir → copied INTO it
    let dest2 = tmp.path().join("existing");
    std::fs::create_dir(&dest2).unwrap();
    host.upload(&src, &dest2, &opts).await.unwrap();
    assert!(dest2.join("app").join("f.txt").exists());
}

#[tokio::test]
async fn localhost_dir_merge_overwrite() {
    let tmp = tempfile::tempdir().unwrap();

    // Source
    let src = tmp.path().join("target_dir");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("a.txt"), b"updated").unwrap();
    std::fs::write(src.join("b.txt"), b"new").unwrap();

    // Existing destination with same basename
    let parent = tmp.path().join("parent");
    std::fs::create_dir(&parent).unwrap();
    let existing = parent.join("target_dir");
    std::fs::create_dir(&existing).unwrap();
    std::fs::write(existing.join("a.txt"), b"original").unwrap();
    std::fs::write(existing.join("c.txt"), b"extra").unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Upload → parent (copy-into → parent/target_dir, merge)
    host.upload(&src, &parent, &opts).await.unwrap();
    assert_eq!(std::fs::read(existing.join("a.txt")).unwrap(), b"updated");
    assert_eq!(std::fs::read(existing.join("b.txt")).unwrap(), b"new");
    assert_eq!(std::fs::read(existing.join("c.txt")).unwrap(), b"extra");
}

#[tokio::test]
async fn localhost_overwrite_false_error() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src.txt");
    std::fs::write(&src, b"data").unwrap();
    let dst = tmp.path().join("dst.txt");
    std::fs::write(&dst, b"existing").unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions {
        overwrite: false,
        recursive: false,
    };
    let result = host.upload(&src, &dst, &opts).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("overwrite=false"));
}

#[tokio::test]
async fn localhost_recursive_false_error() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("dir");
    std::fs::create_dir(&src).unwrap();

    let host = HostHandle::local();
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: false,
    };
    let result = host.upload(&src, tmp.path().join("dst").as_path(), &opts).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("recursive=false"));
}

#[tokio::test]
async fn localhost_symlink_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real.txt");
    std::fs::write(&real, b"data").unwrap();
    let link = tmp.path().join("link.txt");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let host = HostHandle::local();
    let result = host
        .upload(&link, tmp.path().join("dst.txt").as_path(), &motlie_tmux::TransferOptions::default())
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("symlink"));
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

// ---------------------------------------------------------------------------
// SSH file transfer integration tests (DC23, Phase 1.13h)
// Gated on MOTLIE_SSH_TEST_HOST — skip when not set.
// ---------------------------------------------------------------------------

/// Helper: connect to the SSH test host, returning (host, remote_tmp_dir).
/// Creates a temporary directory on the remote host for test isolation.
async fn ssh_test_host_with_tmp() -> Option<(HostHandle, String)> {
    let test_host = std::env::var("MOTLIE_SSH_TEST_HOST").ok()?;
    let uri = format!("ssh://{}", test_host);
    let host = SshConfig::parse(&uri)
        .unwrap_or_else(|e| panic!("parse failed: {}", e))
        .connect()
        .await
        .unwrap_or_else(|e| panic!("connect failed: {}", e));

    // Create a remote temp directory
    let output = host.list_sessions().await.ok(); // verify connection works
    let _ = output;

    // Create a remote temp directory via the public exec API.
    let remote_tmp = format!("/tmp/motlie_test_{}", std::process::id());
    // Clean up from any previous failed run, then create
    let _ = host.exec(&format!("rm -rf '{}'", remote_tmp)).await;
    host.exec(&format!("mkdir -p '{}'", remote_tmp))
        .await
        .unwrap_or_else(|e| panic!("failed to create remote tmp dir: {}", e));

    Some((host, remote_tmp))
}

/// Helper: cleanup remote temp directory.
async fn ssh_cleanup(host: &HostHandle, remote_tmp: &str) {
    let _ = host.exec(&format!("rm -rf '{}'", remote_tmp)).await;
}

#[tokio::test]
async fn ssh_file_upload_download_roundtrip() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let content: Vec<u8> = (0..=255).cycle().take(4096).collect();
    let src = tmp.path().join("test.bin");
    std::fs::write(&src, &content).unwrap();

    let remote_file = std::path::PathBuf::from(&format!("{}/test.bin", remote_tmp));
    let opts = motlie_tmux::TransferOptions::default();

    // Upload
    host.upload(&src, &remote_file, &opts)
        .await
        .expect("SSH upload failed");

    // Download back
    let restored = tmp.path().join("restored.bin");
    host.download(&remote_file, &restored, &opts)
        .await
        .expect("SSH download failed");
    assert_eq!(std::fs::read(&restored).unwrap(), content, "byte mismatch after SSH round-trip");

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_dir_upload_download_roundtrip() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("project");
    std::fs::create_dir_all(src.join("src").join("inner")).unwrap();
    std::fs::write(src.join("README.md"), b"# Hello").unwrap();
    std::fs::write(src.join("src").join("main.rs"), b"fn main() {}").unwrap();
    std::fs::write(src.join("src").join("inner").join("mod.rs"), b"// mod").unwrap();

    let remote_dest = std::path::PathBuf::from(&format!("{}/deployed", remote_tmp));
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Upload (copy-as: remote_dest doesn't exist)
    host.upload(&src, &remote_dest, &opts)
        .await
        .expect("SSH dir upload failed");

    // Download back
    let restored = tmp.path().join("restored");
    host.download(&remote_dest, &restored, &opts)
        .await
        .expect("SSH dir download failed");

    assert_eq!(std::fs::read(restored.join("README.md")).unwrap(), b"# Hello");
    assert_eq!(
        std::fs::read(restored.join("src").join("main.rs")).unwrap(),
        b"fn main() {}"
    );
    assert_eq!(
        std::fs::read(restored.join("src").join("inner").join("mod.rs")).unwrap(),
        b"// mod"
    );

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_copy_into_vs_copy_as() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("app");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("f.txt"), b"data").unwrap();

    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Copy-as: dest doesn't exist → copied AS that path
    let dest1 = std::path::PathBuf::from(&format!("{}/new_app", remote_tmp));
    host.upload(&src, &dest1, &opts).await.expect("copy-as upload failed");

    // Verify by downloading
    let dl1 = tmp.path().join("dl1");
    host.download(&dest1, &dl1, &opts).await.expect("copy-as download failed");
    assert!(dl1.join("f.txt").exists(), "copy-as: f.txt should exist at top level");

    // Copy-into: dest exists as dir → copied INTO it
    let dest2 = std::path::PathBuf::from(&format!("{}/existing", remote_tmp));
    host.exec(&format!("mkdir -p '{}'", dest2.display()))
        .await
        .unwrap();
    host.upload(&src, &dest2, &opts).await.expect("copy-into upload failed");

    // Verify: should be existing/app/f.txt
    let dl2 = tmp.path().join("dl2");
    let app_in_existing = std::path::PathBuf::from(&format!("{}/existing/app", remote_tmp));
    host.download(&app_in_existing, &dl2, &opts).await.expect("copy-into download failed");
    assert!(dl2.join("f.txt").exists(), "copy-into: app/f.txt should exist");

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_dir_merge_overwrite() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();

    // Create remote existing dir with a.txt (old) and c.txt (extra)
    let remote_parent = format!("{}/parent", remote_tmp);
    let remote_target = format!("{}/target_dir", remote_parent);
    host.exec(&format!("mkdir -p '{}'", remote_target)).await.unwrap();
    host.exec(&format!("echo -n original > '{}/a.txt'", remote_target)).await.unwrap();
    host.exec(&format!("echo -n extra > '{}/c.txt'", remote_target)).await.unwrap();

    // Source: has a.txt (updated) and b.txt (new)
    let src = tmp.path().join("target_dir");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("a.txt"), b"updated").unwrap();
    std::fs::write(src.join("b.txt"), b"new").unwrap();

    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: true,
    };

    // Upload target_dir → parent (copy-into → parent/target_dir, merge)
    host.upload(&src, &std::path::PathBuf::from(&remote_parent), &opts)
        .await
        .expect("merge upload failed");

    // Download merged result
    let dl = tmp.path().join("merged");
    host.download(&std::path::PathBuf::from(&remote_target), &dl, &opts)
        .await
        .expect("merge download failed");

    assert_eq!(std::fs::read(dl.join("a.txt")).unwrap(), b"updated");
    assert_eq!(std::fs::read(dl.join("b.txt")).unwrap(), b"new");
    assert_eq!(std::fs::read(dl.join("c.txt")).unwrap(), b"extra");

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_overwrite_false_error() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src.txt");
    std::fs::write(&src, b"data").unwrap();

    // Create remote file first
    let remote_file = std::path::PathBuf::from(&format!("{}/existing.txt", remote_tmp));
    host.upload(&src, &remote_file, &motlie_tmux::TransferOptions::default())
        .await
        .unwrap();

    // Try upload with overwrite=false
    let opts = motlie_tmux::TransferOptions {
        overwrite: false,
        recursive: false,
    };
    let result = host.upload(&src, &remote_file, &opts).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("overwrite=false"));

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_recursive_false_error() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("dir");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("f.txt"), b"x").unwrap();

    let remote_dest = std::path::PathBuf::from(&format!("{}/dest", remote_tmp));
    let opts = motlie_tmux::TransferOptions {
        overwrite: true,
        recursive: false,
    };
    let result = host.upload(&src, &remote_dest, &opts).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("recursive=false"));

    ssh_cleanup(&host, &remote_tmp).await;
}

#[tokio::test]
async fn ssh_symlink_rejected() {
    let Some((host, remote_tmp)) = ssh_test_host_with_tmp().await else {
        eprintln!("skipping: MOTLIE_SSH_TEST_HOST not set");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real.txt");
    std::fs::write(&real, b"data").unwrap();
    let link = tmp.path().join("link.txt");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let remote_dest = std::path::PathBuf::from(&format!("{}/link.txt", remote_tmp));
    let result = host
        .upload(&link, &remote_dest, &motlie_tmux::TransferOptions::default())
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("symlink"));

    ssh_cleanup(&host, &remote_tmp).await;
}
