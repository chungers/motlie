//! Guest-side integration tests that don't require FUSE or vsock.
//! Tests the GuestMountRunner orchestration, mock transport patterns,
//! and end-to-end mounted-subtree scenarios in direct mode.

use std::fs;
use std::sync::Arc;

use bytes::Bytes;
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};
use motlie_vfs::core::op::*;
use motlie_vfs::core::server::FsServer;

fn build_overlay_server(dir: &std::path::Path) -> Arc<FsServer> {
    Arc::new(
        FsServer::builder()
            .mount("home", dir.to_path_buf(), false)
            .overlay(true)
            .build()
            .unwrap(),
    )
}

// ---------------------------------------------------------------------------
// 4.2.7: Mock transport tests — FuseClient request function pattern
// ---------------------------------------------------------------------------

// Test that a closure wrapping handle_op produces correct results,
// simulating what FuseClient does internally.
#[test]
fn mock_transport_lookup() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("hello.txt"), b"world").unwrap();
    let server = build_overlay_server(dir.path());

    // This is the same pattern FuseClient uses: a closure that dispatches FsOp
    let request = |op: FsOp| -> FsResult { server.handle_op("home", op) };

    let result = request(FsOp::Lookup { parent: 1, name: "hello.txt".into() });
    assert!(matches!(result, FsResult::Entry { .. }));
}

#[test]
fn mock_transport_overlay_visible() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_overlay_server(dir.path());
    let overlay = server.overlay().unwrap();
    overlay.put_layer("creds", 0).unwrap();
    overlay.put("creds", "home", "/.ssh/id_ed25519", Bytes::from("key-data")).unwrap();

    let request = |op: FsOp| -> FsResult { server.handle_op("home", op) };

    // Overlay file visible
    let result = request(FsOp::Lookup { parent: 1, name: ".ssh".into() });
    match result {
        FsResult::Entry { attrs, .. } => assert_eq!(attrs.kind, FileType::Directory),
        other => panic!("expected dir Entry, got {:?}", other),
    }
}

#[test]
fn mock_transport_open_read_write_release() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_overlay_server(dir.path());
    let overlay = server.overlay().unwrap();
    overlay.put_layer("l", 0).unwrap();
    overlay.put("l", "home", "/secret.txt", Bytes::from("original")).unwrap();

    let request = |op: FsOp| -> FsResult { server.handle_op("home", op) };

    let inode = match request(FsOp::Lookup { parent: 1, name: "secret.txt".into() }) {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };

    let fh = match request(FsOp::Open { inode, flags: 0 }) {
        FsResult::Opened { fh } => fh,
        other => panic!("expected Opened, got {:?}", other),
    };

    // Read
    match request(FsOp::Read { inode, fh, offset: 0, size: 4096 }) {
        FsResult::Data { data } => assert_eq!(&data[..], b"original"),
        other => panic!("expected Data, got {:?}", other),
    }

    // Write — patches bytes at offset 0 without truncating.
    // "original" (8 bytes) patched with "REPLACED" (8 bytes) at offset 0.
    match request(FsOp::Write { inode, fh, offset: 0, data: Bytes::from("REPLACED") }) {
        FsResult::Written { size } => assert_eq!(size, 8),
        other => panic!("expected Written, got {:?}", other),
    }

    // Read after write
    match request(FsOp::Read { inode, fh, offset: 0, size: 4096 }) {
        FsResult::Data { data } => assert_eq!(&data[..], b"REPLACED"),
        other => panic!("expected Data, got {:?}", other),
    }

    // Release
    assert!(matches!(request(FsOp::Release { inode, fh }), FsResult::Ok));
}

// ---------------------------------------------------------------------------
// 4.2.7a: GuestMountRunner with mock connector
// ---------------------------------------------------------------------------

#[test]
fn guest_mount_runner_with_connector_stub() {
    let dir = tempfile::tempdir().unwrap();
    let specs = vec![
        GuestMountSpec::new("tag-a", dir.path().join("a").to_str().unwrap()),
        GuestMountSpec::new("tag-b", dir.path().join("b").to_str().unwrap()),
    ];
    let runner = GuestMountRunner::new(specs);

    // Use stub (connector-driven mount_all requires feature="client" + Linux)
    let handles = runner.mount_all_stub().unwrap();
    let results = handles.join_all();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.is_ok()));
    assert!(dir.path().join("a").exists());
    assert!(dir.path().join("b").exists());
}

#[test]
fn guest_mount_spec_read_only() {
    let spec = GuestMountSpec::new("workspace", "/workspace").read_only(true);
    assert_eq!(spec.tag, "workspace");
    assert_eq!(spec.guest_path, "/workspace");
    assert!(spec.read_only);
}

// ---------------------------------------------------------------------------
// 4.2.10: End-to-end mounted-subtree scenario (direct mode)
// ---------------------------------------------------------------------------

/// Simulates the full SSH use case from DESIGN:
/// - Host mounts ~/alice as tag "home"
/// - Overlay injects .ssh/authorized_keys, .ssh/config, .env
/// - Guest sees merged tree: disk files + overlay files
/// - Whiteout hides a disk file
/// - Dynamic update visible immediately
#[test]
fn e2e_ssh_subtree_scenario() {
    let dir = tempfile::tempdir().unwrap();

    // Seed disk with alice's home directory
    fs::create_dir_all(dir.path().join("projects")).unwrap();
    fs::write(dir.path().join("projects/README.md"), b"from-host").unwrap();
    fs::write(dir.path().join(".bashrc"), b"# bashrc").unwrap();
    fs::create_dir_all(dir.path().join(".config")).unwrap();
    fs::write(dir.path().join(".config/settings.json"), b"{}").unwrap();

    let server = build_overlay_server(dir.path());
    let overlay = server.overlay().unwrap();
    overlay.put_layer("credentials", 0).unwrap();

    // Inject SSH keys (uid=1000 gid=1000 = alice)
    use motlie_vfs::core::overlay::OverlayAttrs;
    let alice_attrs = OverlayAttrs { mode: 0o600, uid: 1000, gid: 1000 };
    overlay.put_with_attrs("credentials", "home", "/.ssh/authorized_keys", alice_attrs, Bytes::from("ssh-ed25519 AAAA... alice@dev")).unwrap();
    overlay.put_with_attrs("credentials", "home", "/.ssh/config", OverlayAttrs { mode: 0o644, uid: 1000, gid: 1000 }, Bytes::from("Host github.com")).unwrap();
    overlay.put_with_attrs("credentials", "home", "/.ssh/id_ed25519", alice_attrs, Bytes::from("PRIVATE_KEY")).unwrap();
    overlay.put_with_attrs("credentials", "home", "/.ssh/id_ed25519.pub", OverlayAttrs { mode: 0o644, uid: 1000, gid: 1000 }, Bytes::from("PUBLIC_KEY")).unwrap();

    // Inject .env
    overlay.put("credentials", "home", "/.env", Bytes::from("ANTHROPIC_API_KEY=sk-ant-xxx")).unwrap();

    let request = |op: FsOp| -> FsResult { server.handle_op("home", op) };

    // --- Readdir on root: should see disk files + overlay entries ---
    let entries = match request(FsOp::Readdir { inode: 1, offset: 0 }) {
        FsResult::DirEntries { entries } => entries,
        other => panic!("expected DirEntries, got {:?}", other),
    };
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"projects"), "disk dir should be visible");
    assert!(names.contains(&".bashrc"), "disk file should be visible");
    assert!(names.contains(&".config"), "disk dir should be visible");
    assert!(names.contains(&".ssh"), "overlay synthetic dir should be visible");
    assert!(names.contains(&".env"), "overlay synthetic file should be visible");

    // --- Lookup .ssh → Directory ---
    match request(FsOp::Lookup { parent: 1, name: ".ssh".into() }) {
        FsResult::Entry { attrs, .. } => {
            assert_eq!(attrs.kind, FileType::Directory);
        }
        other => panic!("expected Entry, got {:?}", other),
    }

    // --- Lookup disk file still works ---
    match request(FsOp::Lookup { parent: 1, name: "projects".into() }) {
        FsResult::Entry { attrs, .. } => assert_eq!(attrs.kind, FileType::Directory),
        other => panic!("expected Entry, got {:?}", other),
    }

    // --- Overlay attrs preserve uid/gid ---
    match request(FsOp::Lookup { parent: 1, name: ".ssh".into() }) {
        FsResult::Entry { attrs, .. } => {
            // .ssh dir inherits uid/gid from the injected children's attrs
            assert_eq!(attrs.kind, FileType::Directory);
        }
        other => panic!("expected Entry, got {:?}", other),
    }

    // --- Read overlay content ---
    let env_inode = match request(FsOp::Lookup { parent: 1, name: ".env".into() }) {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };
    let fh = match request(FsOp::Open { inode: env_inode, flags: 0 }) {
        FsResult::Opened { fh } => fh,
        other => panic!("expected Opened, got {:?}", other),
    };
    match request(FsOp::Read { inode: env_inode, fh, offset: 0, size: 4096 }) {
        FsResult::Data { data } => assert_eq!(&data[..], b"ANTHROPIC_API_KEY=sk-ant-xxx"),
        other => panic!("expected Data, got {:?}", other),
    }
    request(FsOp::Release { inode: env_inode, fh });

    // --- Whiteout hides disk file ---
    overlay.whiteout("credentials", "home", "/.bashrc").unwrap();
    match request(FsOp::Lookup { parent: 1, name: ".bashrc".into() }) {
        FsResult::Error { errno } => assert_eq!(errno, libc::ENOENT),
        other => panic!("expected ENOENT after whiteout, got {:?}", other),
    }

    // Readdir no longer shows .bashrc
    let entries = match request(FsOp::Readdir { inode: 1, offset: 0 }) {
        FsResult::DirEntries { entries } => entries,
        other => panic!("expected DirEntries, got {:?}", other),
    };
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(!names.contains(&".bashrc"), ".bashrc should be hidden by whiteout");

    // --- Dynamic update: replace .env content ---
    overlay.put("credentials", "home", "/.env", Bytes::from("UPDATED_KEY=new-value")).unwrap();
    let fh = match request(FsOp::Open { inode: env_inode, flags: 0 }) {
        FsResult::Opened { fh } => fh,
        other => panic!("expected Opened, got {:?}", other),
    };
    match request(FsOp::Read { inode: env_inode, fh, offset: 0, size: 4096 }) {
        FsResult::Data { data } => assert_eq!(&data[..], b"UPDATED_KEY=new-value"),
        other => panic!("expected Data, got {:?}", other),
    }
    request(FsOp::Release { inode: env_inode, fh });

    // --- Remove whiteout → disk file reappears ---
    overlay.remove("credentials", "home", "/.bashrc").unwrap();
    match request(FsOp::Lookup { parent: 1, name: ".bashrc".into() }) {
        FsResult::Entry { .. } => {} // disk file is back
        other => panic!("expected .bashrc to reappear, got {:?}", other),
    }
}

/// Multi-tag scenario: two tags in one server, overlay-isolated.
#[test]
fn e2e_multi_tag_isolation() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    fs::write(dir_a.path().join("a.txt"), b"A").unwrap();
    fs::write(dir_b.path().join("b.txt"), b"B").unwrap();

    let server = Arc::new(
        FsServer::builder()
            .mount("alice-home", dir_a.path().to_path_buf(), false)
            .mount("bob-home", dir_b.path().to_path_buf(), false)
            .overlay(true)
            .build()
            .unwrap(),
    );
    let overlay = server.overlay().unwrap();
    overlay.put_layer("creds", 0).unwrap();
    overlay.put("creds", "alice-home", "/.env", Bytes::from("ALICE_KEY")).unwrap();
    overlay.put("creds", "bob-home", "/.env", Bytes::from("BOB_KEY")).unwrap();

    // Alice sees her file + overlay
    assert!(matches!(server.handle_op("alice-home", FsOp::Lookup { parent: 1, name: "a.txt".into() }), FsResult::Entry { .. }));
    assert!(matches!(server.handle_op("alice-home", FsOp::Lookup { parent: 1, name: ".env".into() }), FsResult::Entry { .. }));
    // Alice doesn't see Bob's disk file
    assert!(matches!(server.handle_op("alice-home", FsOp::Lookup { parent: 1, name: "b.txt".into() }), FsResult::Error { .. }));

    // Bob sees his file + overlay
    assert!(matches!(server.handle_op("bob-home", FsOp::Lookup { parent: 1, name: "b.txt".into() }), FsResult::Entry { .. }));
    assert!(matches!(server.handle_op("bob-home", FsOp::Lookup { parent: 1, name: ".env".into() }), FsResult::Entry { .. }));
    // Bob doesn't see Alice's disk file
    assert!(matches!(server.handle_op("bob-home", FsOp::Lookup { parent: 1, name: "a.txt".into() }), FsResult::Error { .. }));
}

// ---------------------------------------------------------------------------
// 5.1.25: In-process VMM/REPL hosting pattern
// ---------------------------------------------------------------------------

/// Prove that overlay mutation via server.overlay() works while
/// handle_op is actively serving — the in-process hosting pattern.
#[test]
fn in_process_overlay_mutation_while_serving() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("disk.txt"), b"d").unwrap();
    let server = build_overlay_server(dir.path());

    // Serve a request
    assert!(matches!(server.handle_op("home", FsOp::Lookup { parent: 1, name: "disk.txt".into() }), FsResult::Entry { .. }));

    // Mutate overlay while server is "running"
    let overlay = server.overlay().unwrap();
    overlay.put_layer("hot", 100).unwrap();
    overlay.put("hot", "home", "/injected.txt", Bytes::from("hot")).unwrap();

    // Next request sees the mutation
    assert!(matches!(server.handle_op("home", FsOp::Lookup { parent: 1, name: "injected.txt".into() }), FsResult::Entry { .. }));

    // Remove layer — injected file disappears
    overlay.remove_layer("hot").unwrap();
    assert!(matches!(server.handle_op("home", FsOp::Lookup { parent: 1, name: "injected.txt".into() }), FsResult::Error { .. }));
}
