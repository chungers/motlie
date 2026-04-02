//! FuseClient: guest-side fuser::Filesystem implementation over a transport.
//!
//! Translates fuser callbacks into FsOp requests, sends them through the
//! VsockClientTransport, and maps FsResult responses back to fuser reply types.
//!
//! v1 mount options (correctness-first):
//! - `direct_io`: bypass kernel page cache
//! - `AllowOther`: allow access outside the mounting user
//!
//! We intentionally avoid `AutoUnmount` here. In the guest image the mounter
//! runs as a root-managed systemd service, and `fuser` implements
//! `AutoUnmount` by shelling out to `fusermount3`. The minimal guest image has
//! proven unreliable on that helper path; direct `/dev/fuse` mounts are the
//! correct service-managed behavior here.
//!
//! This module requires the `client` feature (which pulls in `fuser`).

use std::ffi::OsStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    FileAttr as FuserFileAttr, FileType as FuserFileType, Filesystem, MountOption, ReplyAttr,
    ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs,
    ReplyWrite, Request,
};

use crate::core::op::*;

/// The v1 mount options for correctness-first mode.
/// Note: direct_io is a per-open flag (set via FOPEN_DIRECT_IO in the open
/// response), not a mount option. It is not included here.
pub fn v1_mount_options(read_only: bool) -> Vec<MountOption> {
    let mut opts = vec![MountOption::AllowOther];
    if read_only {
        opts.push(MountOption::RO);
    }
    opts
}

/// Zero TTL for all FUSE responses — forces kernel to revalidate every access.
const ZERO_TTL: Duration = Duration::from_secs(0);

/// Guest-side FUSE filesystem backed by a synchronous request function.
///
/// `FuseClient` is generic over the request function so it can be used with
/// any transport. The transport must provide a blocking `request(&FsOp) -> FsResult`
/// interface.
pub struct FuseClient<F>
where
    F: Fn(FsOp) -> FsResult + Send + Sync,
{
    request_fn: F,
}

impl<F> FuseClient<F>
where
    F: Fn(FsOp) -> FsResult + Send + Sync,
{
    pub fn new(request_fn: F) -> Self {
        Self { request_fn }
    }

    fn request(&self, op: FsOp) -> FsResult {
        (self.request_fn)(op)
    }
}

impl<F> Filesystem for FuseClient<F>
where
    F: Fn(FsOp) -> FsResult + Send + Sync,
{
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let name = name.to_string_lossy().into_owned();
        match self.request(FsOp::Lookup { parent, name }) {
            FsResult::Entry {
                inode,
                generation,
                attrs,
                ..
            } => {
                reply.entry(&ZERO_TTL, &to_fuser_attr(&attrs, inode), generation);
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        match self.request(FsOp::Getattr { inode: ino }) {
            FsResult::Attr { attrs, .. } => {
                reply.attr(&ZERO_TTL, &to_fuser_attr(&attrs, ino));
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<fuser::TimeOrNow>,
        mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        let set = SetAttrFields {
            mode,
            uid,
            gid,
            size,
            atime: atime.map(time_or_now_to_system_time),
            mtime: mtime.map(time_or_now_to_system_time),
        };
        match self.request(FsOp::Setattr {
            inode: ino,
            attrs: set,
        }) {
            FsResult::Attr { attrs, .. } => {
                reply.attr(&ZERO_TTL, &to_fuser_attr(&attrs, ino));
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        match self.request(FsOp::Readdir { inode: ino, offset }) {
            FsResult::DirEntries { entries } => {
                for entry in entries {
                    let kind = to_fuser_filetype(entry.kind);
                    // Use inode from server, or a placeholder (u64::MAX) if 0.
                    // FUSE drops entries with inode 0. The kernel will do a
                    // lookup to resolve the real inode regardless.
                    let ino_hint = if entry.inode == 0 {
                        u64::MAX
                    } else {
                        entry.inode
                    };
                    if reply.add(ino_hint, entry.offset, kind, &entry.name) {
                        break; // buffer full
                    }
                }
                reply.ok();
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        match self.request(FsOp::Open {
            inode: ino,
            flags: flags as u32,
        }) {
            FsResult::Opened { fh } => reply.opened(fh, 0),
            FsResult::Ok => reply.opened(0, 0), // fallback for no-fh case
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        match self.request(FsOp::Read {
            inode: ino,
            fh,
            offset,
            size,
        }) {
            FsResult::Data { data } => reply.data(&data),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        match self.request(FsOp::Write {
            inode: ino,
            fh,
            offset,
            data: bytes::Bytes::copy_from_slice(data),
        }) {
            FsResult::Written { size } => reply.written(size),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn create(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let name = name.to_string_lossy().into_owned();
        match self.request(FsOp::Create {
            parent,
            name,
            mode,
            flags: flags as u32,
            uid: req.uid(),
            gid: req.gid(),
        }) {
            FsResult::Created {
                inode,
                generation,
                attrs,
                fh,
                ..
            } => {
                reply.created(&ZERO_TTL, &to_fuser_attr(&attrs, inode), generation, fh, 0);
            }
            FsResult::Entry {
                inode,
                generation,
                attrs,
                ..
            } => {
                // Backwards compat: server didn't return Created
                reply.created(&ZERO_TTL, &to_fuser_attr(&attrs, inode), generation, 0, 0);
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn mkdir(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let name = name.to_string_lossy().into_owned();
        match self.request(FsOp::Mkdir {
            parent,
            name,
            mode,
            uid: req.uid(),
            gid: req.gid(),
        }) {
            FsResult::Entry {
                inode,
                generation,
                attrs,
                ..
            } => {
                reply.entry(&ZERO_TTL, &to_fuser_attr(&attrs, inode), generation);
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name = name.to_string_lossy().into_owned();
        match self.request(FsOp::Unlink { parent, name }) {
            FsResult::Ok => reply.ok(),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name = name.to_string_lossy().into_owned();
        match self.request(FsOp::Rmdir { parent, name }) {
            FsResult::Ok => reply.ok(),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn symlink(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        link_name: &OsStr,
        target: &std::path::Path,
        reply: ReplyEntry,
    ) {
        let name = link_name.to_string_lossy().into_owned();
        let target = target.to_string_lossy().into_owned();
        match self.request(FsOp::Symlink {
            parent,
            name,
            target,
        }) {
            FsResult::Entry {
                inode,
                generation,
                attrs,
                ..
            } => {
                reply.entry(&ZERO_TTL, &to_fuser_attr(&attrs, inode), generation);
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        match self.request(FsOp::Readlink { inode: ino }) {
            FsResult::Symlink { target } => reply.data(target.as_bytes()),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let name = name.to_string_lossy().into_owned();
        let new_name = newname.to_string_lossy().into_owned();
        match self.request(FsOp::Rename {
            parent,
            name,
            new_parent: newparent,
            new_name,
        }) {
            FsResult::Ok => reply.ok(),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        match self.request(FsOp::Release { inode: ino, fh }) {
            FsResult::Ok => reply.ok(),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn fsync(&mut self, _req: &Request<'_>, ino: u64, fh: u64, datasync: bool, reply: ReplyEmpty) {
        match self.request(FsOp::Fsync {
            inode: ino,
            fh,
            datasync,
        }) {
            FsResult::Ok => reply.ok(),
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }

    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyStatfs) {
        match self.request(FsOp::Statfs) {
            FsResult::Statfs { stats } => {
                reply.statfs(
                    stats.blocks,
                    stats.bfree,
                    stats.bavail,
                    stats.files,
                    stats.ffree,
                    stats.bsize,
                    stats.namelen,
                    stats.frsize,
                );
            }
            FsResult::Error { errno } => reply.error(errno),
            _ => reply.error(libc::EIO),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: convert between motlie-vfs types and fuser types
// ---------------------------------------------------------------------------

fn to_fuser_attr(attr: &FileAttr, ino: u64) -> FuserFileAttr {
    FuserFileAttr {
        ino,
        size: attr.size,
        blocks: attr.blocks,
        atime: attr.atime,
        mtime: attr.mtime,
        ctime: attr.ctime,
        crtime: UNIX_EPOCH,
        kind: to_fuser_filetype(attr.kind),
        perm: (attr.mode & 0o7777) as u16,
        nlink: attr.nlink,
        uid: attr.uid,
        gid: attr.gid,
        rdev: 0,
        blksize: 4096,
        flags: 0,
    }
}

fn to_fuser_filetype(kind: FileType) -> FuserFileType {
    match kind {
        FileType::RegularFile => FuserFileType::RegularFile,
        FileType::Directory => FuserFileType::Directory,
        FileType::Symlink => FuserFileType::Symlink,
    }
}

fn time_or_now_to_system_time(t: fuser::TimeOrNow) -> SystemTime {
    match t {
        fuser::TimeOrNow::SpecificTime(t) => t,
        fuser::TimeOrNow::Now => SystemTime::now(),
    }
}
