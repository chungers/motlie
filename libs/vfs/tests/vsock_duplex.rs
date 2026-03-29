//! End-to-end duplex transport tests: VsockConnectionHandler + VsockClientTransport
//! over tokio::io::duplex, proving the full vertical slice works without a real
//! vsock device or FUSE mount.
#![cfg(feature = "vsock")]

use std::fs;
use std::sync::Arc;

use bytes::Bytes;
use motlie_vfs::core::op::*;
use motlie_vfs::core::server::FsServer;
use motlie_vfs::vsock::client::VsockClientTransport;
use motlie_vfs::vsock::handler::VsockConnectionHandler;

extern crate libc;

fn build_server(dir: &std::path::Path) -> Arc<FsServer> {
    Arc::new(
        FsServer::builder()
            .mount("test", dir.to_path_buf(), false)
            .overlay(true)
            .events(64)
            .build()
            .unwrap(),
    )
}

/// Helper: start handler on one half of a duplex, return client transport on the other.
fn duplex_pair(server: Arc<FsServer>, tag: &str) -> VsockClientTransport<tokio::io::DuplexStream> {
    let (client_stream, server_stream) = tokio::io::duplex(256 * 1024);
    let handler = VsockConnectionHandler::new(server, tag);
    tokio::spawn(async move {
        let _ = handler.serve(server_stream).await;
    });
    VsockClientTransport::new(client_stream, tag)
}

#[tokio::test]
async fn lookup_existing_file_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("hello.txt"), b"world").unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Lookup { parent: 1, name: "hello.txt".into() }).await.unwrap();
    assert!(matches!(result, FsResult::Entry { .. }));
}

#[tokio::test]
async fn lookup_missing_file_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Lookup { parent: 1, name: "nope.txt".into() }).await.unwrap();
    assert!(matches!(result, FsResult::Error { errno } if errno == libc::ENOENT));
}

#[tokio::test]
async fn getattr_root_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Getattr { inode: 1 }).await.unwrap();
    match result {
        FsResult::Attr { attrs, ttl_secs } => {
            assert_eq!(attrs.kind, FileType::Directory);
            assert_eq!(ttl_secs, 0);
        }
        other => panic!("expected Attr, got {:?}", other),
    }
}

#[tokio::test]
async fn readdir_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.txt"), b"a").unwrap();
    fs::write(dir.path().join("b.txt"), b"b").unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Readdir { inode: 1, offset: 0 }).await.unwrap();
    match result {
        FsResult::DirEntries { entries } => {
            let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&"a.txt"));
            assert!(names.contains(&"b.txt"));
        }
        other => panic!("expected DirEntries, got {:?}", other),
    }
}

#[tokio::test]
async fn create_and_unlink_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Create {
        parent: 1, name: "new.txt".into(), mode: 0o644, flags: 0,
    }).await.unwrap();
    assert!(matches!(result, FsResult::Entry { .. }));
    assert!(dir.path().join("new.txt").exists());

    let result = client.request(&FsOp::Unlink { parent: 1, name: "new.txt".into() }).await.unwrap();
    assert!(matches!(result, FsResult::Ok));
    assert!(!dir.path().join("new.txt").exists());
}

#[tokio::test]
async fn overlay_visible_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());

    // Inject overlay content on the host side
    let overlay = server.overlay().unwrap();
    overlay.put_layer("inject", 0).unwrap();
    overlay.put("inject", "test", "/.env", Bytes::from("SECRET=abc")).unwrap();

    let client = duplex_pair(server, "test");

    // Guest sees the injected file via transport
    let result = client.request(&FsOp::Lookup { parent: 1, name: ".env".into() }).await.unwrap();
    assert!(matches!(result, FsResult::Entry { .. }));

    // Readdir includes both disk and overlay entries
    fs::write(dir.path().join("disk.txt"), b"d").unwrap();
    let result = client.request(&FsOp::Readdir { inode: 1, offset: 0 }).await.unwrap();
    match result {
        FsResult::DirEntries { entries } => {
            let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&".env"), "overlay file should be visible");
            assert!(names.contains(&"disk.txt"), "disk file should be visible");
        }
        other => panic!("expected DirEntries, got {:?}", other),
    }
}

#[tokio::test]
async fn overlay_whiteout_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("secret.txt"), b"hidden").unwrap();
    let server = build_server(dir.path());

    let overlay = server.overlay().unwrap();
    overlay.put_layer("hide", 0).unwrap();
    overlay.whiteout("hide", "test", "/secret.txt").unwrap();

    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Lookup { parent: 1, name: "secret.txt".into() }).await.unwrap();
    assert!(matches!(result, FsResult::Error { errno } if errno == libc::ENOENT));
}

#[tokio::test]
async fn statfs_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    let result = client.request(&FsOp::Statfs).await.unwrap();
    assert!(matches!(result, FsResult::Statfs { .. }));
}

#[tokio::test]
async fn multiple_requests_sequential() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("f.txt"), b"data").unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    // Multiple sequential requests on the same connection
    for _ in 0..10 {
        let result = client.request(&FsOp::Getattr { inode: 1 }).await.unwrap();
        assert!(matches!(result, FsResult::Attr { .. }));
    }
}

// Parity test: same operations via transport produce same results as direct handle_op
#[tokio::test]
async fn parity_with_direct_handle_op() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("test.txt"), b"content").unwrap();
    let server = build_server(dir.path());

    // Direct
    let direct_result = server.handle_op("test", FsOp::Lookup { parent: 1, name: "test.txt".into() });
    let direct_is_entry = matches!(direct_result, FsResult::Entry { .. });

    // Via transport
    let client = duplex_pair(server, "test");
    let transport_result = client.request(&FsOp::Lookup { parent: 1, name: "test.txt".into() }).await.unwrap();
    let transport_is_entry = matches!(transport_result, FsResult::Entry { .. });

    assert_eq!(direct_is_entry, transport_is_entry, "transport and direct should produce same result shape");
}

#[tokio::test]
async fn open_read_write_release_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("file.txt"), b"disk content").unwrap();
    let server = build_server(dir.path());
    let client = duplex_pair(server, "test");

    // Lookup → get inode
    let inode = match client.request(&FsOp::Lookup { parent: 1, name: "file.txt".into() }).await.unwrap() {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };

    // Open → get fh
    let fh = match client.request(&FsOp::Open { inode, flags: 0 }).await.unwrap() {
        FsResult::Opened { fh } => fh,
        other => panic!("expected Opened, got {:?}", other),
    };

    // Read through fh
    let data = match client.request(&FsOp::Read { inode, fh, offset: 0, size: 4096 }).await.unwrap() {
        FsResult::Data { data } => data,
        other => panic!("expected Data, got {:?}", other),
    };
    assert_eq!(&data[..], b"disk content");

    // Release
    let result = client.request(&FsOp::Release { inode, fh }).await.unwrap();
    assert!(matches!(result, FsResult::Ok));
}

#[tokio::test]
async fn overlay_open_read_write_release_over_transport() {
    let dir = tempfile::tempdir().unwrap();
    let server = build_server(dir.path());

    let overlay = server.overlay().unwrap();
    overlay.put_layer("l", 0).unwrap();
    overlay.put("l", "test", "/secret.txt", Bytes::from("overlay data")).unwrap();

    let client = duplex_pair(server, "test");

    // Lookup
    let inode = match client.request(&FsOp::Lookup { parent: 1, name: "secret.txt".into() }).await.unwrap() {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };

    // Open
    let fh = match client.request(&FsOp::Open { inode, flags: 0 }).await.unwrap() {
        FsResult::Opened { fh } => fh,
        other => panic!("expected Opened, got {:?}", other),
    };

    // Read overlay content through fh
    let data = match client.request(&FsOp::Read { inode, fh, offset: 0, size: 4096 }).await.unwrap() {
        FsResult::Data { data } => data,
        other => panic!("expected Data, got {:?}", other),
    };
    assert_eq!(&data[..], b"overlay data");

    // Write through fh — update overlay content
    let result = client.request(&FsOp::Write {
        inode, fh, offset: 0, data: Bytes::from("PATCHED data"),
    }).await.unwrap();
    assert!(matches!(result, FsResult::Written { size: 12 }));

    // Read again — should see patched content
    let data = match client.request(&FsOp::Read { inode, fh, offset: 0, size: 4096 }).await.unwrap() {
        FsResult::Data { data } => data,
        other => panic!("expected Data, got {:?}", other),
    };
    assert_eq!(&data[..], b"PATCHED data");

    // Fsync — no-op
    assert!(matches!(client.request(&FsOp::Fsync { inode, fh, datasync: false }).await.unwrap(), FsResult::Ok));

    // Release
    assert!(matches!(client.request(&FsOp::Release { inode, fh }).await.unwrap(), FsResult::Ok));
}
