//! Core filesystem operation and result types shared by all composites.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A filesystem operation (request). Core type shared by all composites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FsOp {
    Lookup { parent: u64, name: String },
    Getattr { inode: u64 },
    Setattr { inode: u64, attrs: SetAttrFields },
    Readdir { inode: u64, offset: i64 },
    Open { inode: u64, flags: u32 },
    Read { inode: u64, fh: u64, offset: i64, size: u32 },
    Write { inode: u64, fh: u64, offset: i64, data: Bytes },
    Create { parent: u64, name: String, mode: u32, flags: u32, uid: u32, gid: u32 },
    Mkdir { parent: u64, name: String, mode: u32 },
    Unlink { parent: u64, name: String },
    Rmdir { parent: u64, name: String },
    Rename { parent: u64, name: String, new_parent: u64, new_name: String },
    Symlink { parent: u64, name: String, target: String },
    Readlink { inode: u64 },
    Release { inode: u64, fh: u64 },
    Fsync { inode: u64, fh: u64, datasync: bool },
    Statfs,
}

/// A filesystem result (response). Core type shared by all composites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FsResult {
    Entry { inode: u64, generation: u64, attrs: FileAttr, ttl_secs: u32 },
    Created { inode: u64, generation: u64, attrs: FileAttr, fh: u64, ttl_secs: u32 },
    Attr { attrs: FileAttr, ttl_secs: u32 },
    Data { data: Bytes },
    Written { size: u32 },
    DirEntries { entries: Vec<DirEntry> },
    Statfs { stats: FsStats },
    Symlink { target: String },
    Opened { fh: u64 },
    Ok,
    Error { errno: i32 },
}

/// File attributes returned by Lookup, Getattr, Create, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAttr {
    pub inode: u64,
    pub size: u64,
    pub blocks: u64,
    pub atime: SystemTime,
    pub mtime: SystemTime,
    pub ctime: SystemTime,
    pub kind: FileType,
    pub mode: u32,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
}

/// File type tag for FileAttr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    RegularFile,
    Directory,
    Symlink,
}

/// Fields that may be set by Setattr. Each `Option` indicates whether the
/// caller wants to change that attribute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAttrFields {
    pub mode: Option<u32>,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
    pub size: Option<u64>,
    pub atime: Option<SystemTime>,
    pub mtime: Option<SystemTime>,
}

/// A single directory entry returned inside `FsResult::DirEntries`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub inode: u64,
    pub offset: i64,
    pub kind: FileType,
    pub name: String,
}

/// Filesystem statistics returned by `FsResult::Statfs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsStats {
    pub blocks: u64,
    pub bfree: u64,
    pub bavail: u64,
    pub files: u64,
    pub ffree: u64,
    pub bsize: u32,
    pub namelen: u32,
    pub frsize: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsop_serde_round_trip() {
        let ops = vec![
            FsOp::Lookup { parent: 1, name: "hello.txt".into() },
            FsOp::Getattr { inode: 42 },
            FsOp::Setattr {
                inode: 42,
                attrs: SetAttrFields {
                    mode: Some(0o644),
                    uid: Some(1000),
                    gid: Some(1000),
                    size: None,
                    atime: None,
                    mtime: None,
                },
            },
            FsOp::Readdir { inode: 1, offset: 0 },
            FsOp::Open { inode: 42, flags: 0 },
            FsOp::Read { inode: 42, fh: 1, offset: 0, size: 4096 },
            FsOp::Write { inode: 42, fh: 1, offset: 0, data: Bytes::from_static(b"hello") },
            FsOp::Create { parent: 1, name: "new.txt".into(), mode: 0o644, flags: 0, uid: 1000, gid: 1000 },
            FsOp::Mkdir { parent: 1, name: "dir".into(), mode: 0o755 },
            FsOp::Unlink { parent: 1, name: "old.txt".into() },
            FsOp::Rmdir { parent: 1, name: "dir".into() },
            FsOp::Rename { parent: 1, name: "a".into(), new_parent: 2, new_name: "b".into() },
            FsOp::Symlink { parent: 1, name: "link".into(), target: "/foo".into() },
            FsOp::Readlink { inode: 99 },
            FsOp::Release { inode: 42, fh: 1 },
            FsOp::Fsync { inode: 42, fh: 1, datasync: false },
            FsOp::Statfs,
        ];
        for op in &ops {
            let json = serde_json::to_string(op).unwrap();
            let back: FsOp = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2, "round-trip failed for {:?}", op);
        }
    }

    #[test]
    fn fsresult_serde_round_trip() {
        let now = SystemTime::now();
        let results = vec![
            FsResult::Entry {
                inode: 42,
                generation: 1,
                attrs: FileAttr {
                    inode: 42, size: 100, blocks: 1,
                    atime: now, mtime: now, ctime: now,
                    kind: FileType::RegularFile,
                    mode: 0o644, nlink: 1, uid: 1000, gid: 1000,
                },
                ttl_secs: 0,
            },
            FsResult::Attr {
                attrs: FileAttr {
                    inode: 1, size: 0, blocks: 0,
                    atime: now, mtime: now, ctime: now,
                    kind: FileType::Directory,
                    mode: 0o755, nlink: 2, uid: 0, gid: 0,
                },
                ttl_secs: 0,
            },
            FsResult::Data { data: Bytes::from_static(b"content") },
            FsResult::Written { size: 5 },
            FsResult::DirEntries {
                entries: vec![
                    DirEntry { inode: 2, offset: 1, kind: FileType::RegularFile, name: "a.txt".into() },
                    DirEntry { inode: 3, offset: 2, kind: FileType::Directory, name: "sub".into() },
                ],
            },
            FsResult::Statfs {
                stats: FsStats {
                    blocks: 1000, bfree: 500, bavail: 450,
                    files: 100, ffree: 80, bsize: 4096,
                    namelen: 255, frsize: 4096,
                },
            },
            FsResult::Created {
                inode: 99,
                generation: 0,
                attrs: FileAttr {
                    inode: 99, size: 0, blocks: 0,
                    atime: now, mtime: now, ctime: now,
                    kind: FileType::RegularFile,
                    mode: 0o644, nlink: 1, uid: 1000, gid: 1000,
                },
                fh: 7,
                ttl_secs: 0,
            },
            FsResult::Symlink { target: "/foo/bar".into() },
            FsResult::Opened { fh: 42 },
            FsResult::Ok,
            FsResult::Error { errno: 2 },
        ];
        for r in &results {
            let json = serde_json::to_string(r).unwrap();
            let back: FsResult = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2, "round-trip failed for {:?}", r);
        }
    }

    #[test]
    fn synthetic_attrs_carry_uid_gid_mode() {
        let attr = FileAttr {
            inode: 100,
            size: 42,
            blocks: 1,
            atime: SystemTime::now(),
            mtime: SystemTime::now(),
            ctime: SystemTime::now(),
            kind: FileType::RegularFile,
            mode: 0o600,
            nlink: 1,
            uid: 1000,
            gid: 1000,
        };
        let json = serde_json::to_string(&attr).unwrap();
        let back: FileAttr = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uid, 1000);
        assert_eq!(back.gid, 1000);
        assert_eq!(back.mode, 0o600);

        let set = SetAttrFields {
            mode: Some(0o755),
            uid: Some(0),
            gid: Some(0),
            size: None,
            atime: None,
            mtime: None,
        };
        let json = serde_json::to_string(&set).unwrap();
        let back: SetAttrFields = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uid, Some(0));
        assert_eq!(back.gid, Some(0));
        assert_eq!(back.mode, Some(0o755));
    }
}
