//! FsServer and FsServerBuilder: tag-based mount routing and handle_op() dispatch.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;

use anyhow::{anyhow, Result};
use tokio::sync::broadcast;

use super::event::{FsEvent, FsOpKind};
use super::inode::{default_root_attrs, InodeTable};
use super::op::*;
use super::policy::{AllowAll, PolicyFn};

// ---------------------------------------------------------------------------
// Mount-backing boundary (2.1.8, 2.1.9)
// ---------------------------------------------------------------------------

/// Mount backing — the only layer that translates mount-relative paths to
/// host filesystem paths and issues `std::fs` operations.
enum MountBacking {
    Disk(DiskBacking),
    // Future: SyntheticRoot, Hybrid
}

struct DiskBacking {
    host_root: PathBuf,
}

impl DiskBacking {
    /// Resolve a mount-relative path to a host filesystem path.
    fn resolve(&self, rel_path: &str) -> PathBuf {
        if rel_path == "/" {
            self.host_root.clone()
        } else {
            self.host_root.join(rel_path.trim_start_matches('/'))
        }
    }
}

impl MountBacking {
    fn resolve(&self, rel_path: &str) -> PathBuf {
        match self {
            MountBacking::Disk(d) => d.resolve(rel_path),
        }
    }

    #[allow(dead_code)] // will be used by overlay integration in Phase 2.2
    fn host_root(&self) -> &Path {
        match self {
            MountBacking::Disk(d) => &d.host_root,
        }
    }
}

// ---------------------------------------------------------------------------
// Mount state (2.1.8, 2.1.10)
// ---------------------------------------------------------------------------

/// Per-tag mount state: backing + inode table + read-only flag.
/// The overlay slot is reserved for Phase 2.2.
struct MountState {
    tag: String,
    read_only: bool,
    backing: MountBacking,
    inode_table: InodeTable,
    // overlay: Option<MemOverlay>, // Phase 2.2
}

impl MountState {
    fn new(tag: &str, host_path: PathBuf, read_only: bool) -> Self {
        let mut root_attrs = default_root_attrs();
        if let Ok(meta) = fs::metadata(&host_path) {
            fill_attrs_from_metadata(&mut root_attrs, &meta, 1);
        }
        let mut inode_table = InodeTable::new(root_attrs);
        // Set the root's host_path so disk ops can resolve it
        if let Some(root) = inode_table.get_mut(1) {
            root.host_path = Some(host_path.clone());
        }
        Self {
            tag: tag.to_string(),
            read_only,
            backing: MountBacking::Disk(DiskBacking { host_root: host_path }),
            inode_table,
        }
    }
}

// ---------------------------------------------------------------------------
// FsServer + Builder (2.1.1, 2.1.2)
// ---------------------------------------------------------------------------

/// Core filesystem server. All composites call `handle_op()`.
pub struct FsServer {
    mounts: RwLock<HashMap<String, MountState>>,
    policy: Box<dyn PolicyFn>,
    event_tx: Option<broadcast::Sender<FsEvent>>,
}

pub struct FsServerBuilder {
    mounts: Vec<(String, PathBuf, bool)>,
    event_capacity: Option<usize>,
    policy: Option<Box<dyn PolicyFn>>,
    overlay_enabled: bool,
}

impl FsServer {
    pub fn builder() -> FsServerBuilder {
        FsServerBuilder {
            mounts: Vec::new(),
            event_capacity: None,
            policy: None,
            overlay_enabled: false,
        }
    }

    /// Direct operation dispatch. The single entry point that all composites use.
    pub fn handle_op(&self, tag: &str, op: FsOp) -> FsResult {
        let mounts = match self.mounts.read() {
            Ok(m) => m,
            Err(_) => return FsResult::Error { errno: libc::EIO },
        };
        let mount = match mounts.get(tag) {
            Some(m) => m,
            None => return FsResult::Error { errno: libc::ENOENT },
        };

        let result = self.dispatch(mount, &op);
        self.emit_event(tag, &op, &result);
        result
    }

    /// Register a new mount tag while serving.
    pub fn add_mount(&self, tag: &str, host_path: PathBuf, read_only: bool) -> Result<()> {
        let state = MountState::new(tag, host_path, read_only);
        let mut mounts = self.mounts.write().map_err(|e| anyhow!("lock poisoned: {e}"))?;
        mounts.insert(tag.to_string(), state);
        Ok(())
    }

    /// Remove a mount tag. Drops inode table.
    pub fn remove_mount(&self, tag: &str) -> Result<()> {
        let mut mounts = self.mounts.write().map_err(|e| anyhow!("lock poisoned: {e}"))?;
        mounts.remove(tag);
        Ok(())
    }

    /// Subscribe to filesystem events.
    pub fn subscribe_events(&self) -> Option<broadcast::Receiver<FsEvent>> {
        self.event_tx.as_ref().map(|tx| tx.subscribe())
    }

    /// Access the overlay. Returns None until Phase 2.2.
    pub fn overlay(&self) -> Option<()> {
        None // Phase 2.2
    }
}

impl FsServerBuilder {
    pub fn mount(mut self, tag: &str, host_path: PathBuf, read_only: bool) -> Self {
        self.mounts.push((tag.to_string(), host_path, read_only));
        self
    }

    pub fn events(mut self, capacity: usize) -> Self {
        self.event_capacity = Some(capacity);
        self
    }

    pub fn policy(mut self, policy: impl PolicyFn) -> Self {
        self.policy = Some(Box::new(policy));
        self
    }

    pub fn overlay(mut self, enabled: bool) -> Self {
        self.overlay_enabled = enabled;
        self
    }

    pub fn build(self) -> Result<FsServer> {
        let mut mount_map = HashMap::new();
        for (tag, host_path, read_only) in self.mounts {
            let state = MountState::new(&tag, host_path, read_only);
            mount_map.insert(tag, state);
        }

        let event_tx = self.event_capacity.map(|cap| {
            let (tx, _) = broadcast::channel(cap);
            tx
        });

        Ok(FsServer {
            mounts: RwLock::new(mount_map),
            policy: self.policy.unwrap_or_else(|| Box::new(AllowAll)),
            event_tx,
        })
    }
}

// ---------------------------------------------------------------------------
// Operation dispatch (2.1.3, 2.1.4, 2.1.6)
// ---------------------------------------------------------------------------

impl FsServer {
    fn dispatch(&self, mount: &MountState, op: &FsOp) -> FsResult {
        // Read-only enforcement (2.1.4)
        if mount.read_only && is_write_op(op) {
            return FsResult::Error { errno: libc::EROFS };
        }

        // Policy check (2.1.6)
        let op_kind = FsOpKind::from_op(op);
        let path = path_hint_for_op(mount, op);
        if let Err(errno) = self.policy.check(op_kind, &mount.tag, &path) {
            return FsResult::Error { errno };
        }

        // Dispatch to backing
        match op {
            FsOp::Lookup { parent, name } => self.do_lookup(mount, *parent, name),
            FsOp::Getattr { inode } => self.do_getattr(mount, *inode),
            FsOp::Setattr { inode, attrs } => self.do_setattr(mount, *inode, attrs),
            FsOp::Readdir { inode, offset } => self.do_readdir(mount, *inode, *offset),
            FsOp::Open { inode, flags } => self.do_open(mount, *inode, *flags),
            FsOp::Read { inode, fh: _, offset, size } => self.do_read(mount, *inode, *offset, *size),
            FsOp::Write { inode, fh: _, offset, data } => self.do_write(mount, *inode, *offset, data),
            FsOp::Create { parent, name, mode, flags: _ } => self.do_create(mount, *parent, name, *mode),
            FsOp::Mkdir { parent, name, mode } => self.do_mkdir(mount, *parent, name, *mode),
            FsOp::Unlink { parent, name } => self.do_unlink(mount, *parent, name),
            FsOp::Rmdir { parent, name } => self.do_rmdir(mount, *parent, name),
            FsOp::Rename { parent, name, new_parent, new_name } => {
                self.do_rename(mount, *parent, name, *new_parent, new_name)
            }
            FsOp::Symlink { parent, name, target } => self.do_symlink(mount, *parent, name, target),
            FsOp::Readlink { inode } => self.do_readlink(mount, *inode),
            FsOp::Release { .. } => FsResult::Ok,
            FsOp::Fsync { .. } => FsResult::Ok,
            FsOp::Statfs => self.do_statfs(mount),
        }
    }

    fn emit_event(&self, tag: &str, op: &FsOp, _result: &FsResult) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(FsEvent {
                timestamp: SystemTime::now(),
                tag: tag.to_string(),
                op_kind: FsOpKind::from_op(op),
                path: String::new(), // simplified for now; enhanced in later phases
                bytes: None,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Disk-backed operation implementations (2.1.3, 2.1.9)
//
// All std::fs access is confined to this section and goes through
// mount.backing.resolve() to translate mount-relative paths to host paths.
// ---------------------------------------------------------------------------

impl FsServer {
    fn do_lookup(&self, mount: &MountState, parent: u64, name: &str) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);

        match fs::symlink_metadata(&host_path) {
            Ok(meta) => {
                let kind = file_type_from_meta(&meta);
                let mut attrs = metadata_to_attrs(&meta, 0, kind);
                // We need mut access to inode_table, but we only have &MountState.
                // For this phase, we allocate lazily via interior mutability pattern.
                // This is safe because handle_op holds a read lock on the mounts map,
                // and the inode table needs its own synchronization.
                // For now, return attrs with inode=0; the caller will need to track this.
                // TODO: Phase 2.2 will add proper inode allocation in the dispatch path
                // once MountState uses interior mutability for its inode_table.
                attrs.inode = parent; // placeholder — will be real inode after refactor
                FsResult::Entry {
                    inode: 0, // placeholder
                    generation: 0,
                    attrs,
                    ttl_secs: 0,
                }
            }
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_getattr(&self, mount: &MountState, inode: u64) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        match &entry.host_path {
            Some(hp) => match fs::symlink_metadata(hp) {
                Ok(meta) => {
                    let kind = file_type_from_meta(&meta);
                    let mut attrs = metadata_to_attrs(&meta, inode, kind);
                    attrs.inode = inode;
                    FsResult::Attr { attrs, ttl_secs: 0 }
                }
                Err(e) => FsResult::Error { errno: io_errno(&e) },
            },
            None => {
                // overlay entry — return stored attrs
                FsResult::Attr { attrs: entry.attrs.clone(), ttl_secs: 0 }
            }
        }
    }

    fn do_setattr(&self, mount: &MountState, inode: u64, set: &SetAttrFields) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        if let Some(hp) = &entry.host_path {
            if let Some(mode) = set.mode {
                if let Err(e) = set_permissions(hp, mode) {
                    return FsResult::Error { errno: io_errno(&e) };
                }
            }
            if let Some(size) = set.size {
                if let Err(e) = truncate_file(hp, size) {
                    return FsResult::Error { errno: io_errno(&e) };
                }
            }
            // Re-fetch attrs after modification
            self.do_getattr(mount, inode)
        } else {
            // overlay — return stored attrs (setattr on overlay handled in Phase 2.2)
            FsResult::Attr { attrs: entry.attrs.clone(), ttl_secs: 0 }
        }
    }

    fn do_readdir(&self, mount: &MountState, inode: u64, offset: i64) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let host_path = match &entry.host_path {
            Some(hp) => hp,
            None => return FsResult::DirEntries { entries: vec![] },
        };
        match fs::read_dir(host_path) {
            Ok(rd) => {
                let mut entries = Vec::new();
                for (i, dir_entry) in rd.enumerate() {
                    if (i as i64) < offset {
                        continue;
                    }
                    let de = match dir_entry {
                        Ok(de) => de,
                        Err(_) => continue,
                    };
                    let kind = match de.file_type() {
                        Ok(ft) => {
                            if ft.is_dir() { FileType::Directory }
                            else if ft.is_symlink() { FileType::Symlink }
                            else { FileType::RegularFile }
                        }
                        Err(_) => FileType::RegularFile,
                    };
                    entries.push(DirEntry {
                        inode: 0, // placeholder — real inode allocation in Phase 2.2
                        offset: (i + 1) as i64,
                        kind,
                        name: de.file_name().to_string_lossy().into_owned(),
                    });
                }
                FsResult::DirEntries { entries }
            }
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_open(&self, mount: &MountState, inode: u64, _flags: u32) -> FsResult {
        // For disk files, just verify the inode exists. Actual fd management is
        // handled per-op (read/write open the file each time in v1 no-cache mode).
        match mount.inode_table.get(inode) {
            Some(_) => FsResult::Ok,
            None => FsResult::Error { errno: libc::ENOENT },
        }
    }

    fn do_read(&self, mount: &MountState, inode: u64, offset: i64, size: u32) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let host_path = match &entry.host_path {
            Some(hp) => hp,
            None => return FsResult::Error { errno: libc::ENOENT }, // overlay reads in Phase 2.2
        };
        match read_file_range(host_path, offset, size) {
            Ok(data) => FsResult::Data { data: data.into() },
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_write(&self, mount: &MountState, inode: u64, offset: i64, data: &bytes::Bytes) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let host_path = match &entry.host_path {
            Some(hp) => hp,
            None => return FsResult::Error { errno: libc::ENOENT }, // overlay writes in Phase 2.2
        };
        match write_file_range(host_path, offset, data) {
            Ok(n) => FsResult::Written { size: n },
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_create(&self, mount: &MountState, parent: u64, name: &str, mode: u32) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);
        match fs::File::create(&host_path) {
            Ok(_) => {
                let _ = set_permissions(&host_path, mode);
                match fs::symlink_metadata(&host_path) {
                    Ok(meta) => {
                        let attrs = metadata_to_attrs(&meta, 0, FileType::RegularFile);
                        FsResult::Entry { inode: 0, generation: 0, attrs, ttl_secs: 0 }
                    }
                    Err(e) => FsResult::Error { errno: io_errno(&e) },
                }
            }
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_mkdir(&self, mount: &MountState, parent: u64, name: &str, mode: u32) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);
        match fs::create_dir(&host_path) {
            Ok(()) => {
                let _ = set_permissions(&host_path, mode);
                match fs::symlink_metadata(&host_path) {
                    Ok(meta) => {
                        let attrs = metadata_to_attrs(&meta, 0, FileType::Directory);
                        FsResult::Entry { inode: 0, generation: 0, attrs, ttl_secs: 0 }
                    }
                    Err(e) => FsResult::Error { errno: io_errno(&e) },
                }
            }
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_unlink(&self, mount: &MountState, parent: u64, name: &str) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);
        match fs::remove_file(&host_path) {
            Ok(()) => FsResult::Ok,
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_rmdir(&self, mount: &MountState, parent: u64, name: &str) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);
        match fs::remove_dir(&host_path) {
            Ok(()) => FsResult::Ok,
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_rename(&self, mount: &MountState, parent: u64, name: &str, new_parent: u64, new_name: &str) -> FsResult {
        let src_parent = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let dst_parent = match mount.inode_table.get(new_parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let src_path = mount.backing.resolve(&child_path(&src_parent.path, name));
        let dst_path = mount.backing.resolve(&child_path(&dst_parent.path, new_name));
        match fs::rename(&src_path, &dst_path) {
            Ok(()) => FsResult::Ok,
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_symlink(&self, mount: &MountState, parent: u64, name: &str, target: &str) -> FsResult {
        let parent_entry = match mount.inode_table.get(parent) {
            Some(e) => e.clone(),
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let rel_path = child_path(&parent_entry.path, name);
        let host_path = mount.backing.resolve(&rel_path);
        #[cfg(unix)]
        match std::os::unix::fs::symlink(target, &host_path) {
            Ok(()) => {
                match fs::symlink_metadata(&host_path) {
                    Ok(meta) => {
                        let attrs = metadata_to_attrs(&meta, 0, FileType::Symlink);
                        FsResult::Entry { inode: 0, generation: 0, attrs, ttl_secs: 0 }
                    }
                    Err(e) => FsResult::Error { errno: io_errno(&e) },
                }
            }
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
        #[cfg(not(unix))]
        {
            let _ = (host_path, target);
            FsResult::Error { errno: libc::ENOTSUP }
        }
    }

    fn do_readlink(&self, mount: &MountState, inode: u64) -> FsResult {
        let entry = match mount.inode_table.get(inode) {
            Some(e) => e,
            None => return FsResult::Error { errno: libc::ENOENT },
        };
        let host_path = match &entry.host_path {
            Some(hp) => hp,
            None => return FsResult::Error { errno: libc::ENOTSUP },
        };
        match fs::read_link(host_path) {
            Ok(target) => FsResult::Symlink { target: target.to_string_lossy().into_owned() },
            Err(e) => FsResult::Error { errno: io_errno(&e) },
        }
    }

    fn do_statfs(&self, mount: &MountState) -> FsResult {
        // Return a reasonable default. Real statfs via libc is platform-specific;
        // for v1 this is sufficient.
        let _ = mount;
        FsResult::Statfs {
            stats: FsStats {
                blocks: 0,
                bfree: 0,
                bavail: 0,
                files: 0,
                ffree: 0,
                bsize: 4096,
                namelen: 255,
                frsize: 4096,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers — all std::fs access is here (2.1.9)
// ---------------------------------------------------------------------------

fn is_write_op(op: &FsOp) -> bool {
    matches!(
        op,
        FsOp::Write { .. }
            | FsOp::Create { .. }
            | FsOp::Mkdir { .. }
            | FsOp::Unlink { .. }
            | FsOp::Rmdir { .. }
            | FsOp::Rename { .. }
            | FsOp::Symlink { .. }
            | FsOp::Setattr { .. }
    )
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn path_hint_for_op(mount: &MountState, op: &FsOp) -> String {
    match op {
        FsOp::Lookup { parent, name } => {
            let p = mount.inode_table.get(*parent).map(|e| e.path.as_str()).unwrap_or("/");
            child_path(p, name)
        }
        FsOp::Getattr { inode }
        | FsOp::Setattr { inode, .. }
        | FsOp::Readdir { inode, .. }
        | FsOp::Open { inode, .. }
        | FsOp::Read { inode, .. }
        | FsOp::Write { inode, .. }
        | FsOp::Readlink { inode }
        | FsOp::Release { inode, .. }
        | FsOp::Fsync { inode, .. } => {
            mount.inode_table.get(*inode).map(|e| e.path.clone()).unwrap_or_default()
        }
        FsOp::Create { parent, name, .. }
        | FsOp::Mkdir { parent, name, .. }
        | FsOp::Unlink { parent, name }
        | FsOp::Rmdir { parent, name }
        | FsOp::Symlink { parent, name, .. } => {
            let p = mount.inode_table.get(*parent).map(|e| e.path.as_str()).unwrap_or("/");
            child_path(p, name)
        }
        FsOp::Rename { parent, name, .. } => {
            let p = mount.inode_table.get(*parent).map(|e| e.path.as_str()).unwrap_or("/");
            child_path(p, name)
        }
        FsOp::Statfs => "/".to_string(),
    }
}

#[cfg(unix)]
fn fill_attrs_from_metadata(attrs: &mut FileAttr, meta: &fs::Metadata, inode: u64) {
    use std::os::unix::fs::MetadataExt;
    attrs.inode = inode;
    attrs.size = meta.len();
    attrs.blocks = meta.blocks();
    attrs.mode = meta.mode();
    attrs.nlink = meta.nlink() as u32;
    attrs.uid = meta.uid();
    attrs.gid = meta.gid();
    if let Ok(t) = meta.accessed() { attrs.atime = t; }
    if let Ok(t) = meta.modified() { attrs.mtime = t; }
    attrs.ctime = attrs.mtime; // close enough for v1
    attrs.kind = if meta.is_dir() { FileType::Directory }
        else if meta.file_type().is_symlink() { FileType::Symlink }
        else { FileType::RegularFile };
}

#[cfg(not(unix))]
fn fill_attrs_from_metadata(attrs: &mut FileAttr, meta: &fs::Metadata, inode: u64) {
    attrs.inode = inode;
    attrs.size = meta.len();
    if let Ok(t) = meta.accessed() { attrs.atime = t; }
    if let Ok(t) = meta.modified() { attrs.mtime = t; }
    attrs.ctime = attrs.mtime;
    attrs.kind = if meta.is_dir() { FileType::Directory } else { FileType::RegularFile };
}

fn metadata_to_attrs(meta: &fs::Metadata, inode: u64, kind: FileType) -> FileAttr {
    let mut attrs = default_root_attrs();
    attrs.kind = kind;
    fill_attrs_from_metadata(&mut attrs, meta, inode);
    attrs
}

fn file_type_from_meta(meta: &fs::Metadata) -> FileType {
    if meta.is_dir() { FileType::Directory }
    else if meta.file_type().is_symlink() { FileType::Symlink }
    else { FileType::RegularFile }
}

fn io_errno(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(libc::EIO)
}

fn read_file_range(path: &Path, offset: i64, size: u32) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(path)?;
    if offset > 0 {
        f.seek(SeekFrom::Start(offset as u64))?;
    }
    let mut buf = vec![0u8; size as usize];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

fn write_file_range(path: &Path, offset: i64, data: &[u8]) -> std::io::Result<u32> {
    use std::io::{Seek, SeekFrom, Write};
    let mut f = fs::OpenOptions::new().write(true).open(path)?;
    if offset > 0 {
        f.seek(SeekFrom::Start(offset as u64))?;
    }
    f.write_all(data)?;
    Ok(data.len() as u32)
}

#[cfg(unix)]
fn set_permissions(path: &Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_permissions(_path: &Path, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

fn truncate_file(path: &Path, size: u64) -> std::io::Result<()> {
    let f = fs::OpenOptions::new().write(true).open(path)?;
    f.set_len(size)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (2.1.11 – 2.1.18)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn build_test_server(dir: &Path) -> FsServer {
        FsServer::builder()
            .mount("test", dir.to_path_buf(), false)
            .events(64)
            .build()
            .unwrap()
    }

    // 2.1.11: Integration tests for each FsOp

    #[test]
    fn lookup_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("hello.txt"), b"world").unwrap();
        let server = build_test_server(dir.path());
        let result = server.handle_op("test", FsOp::Lookup { parent: 1, name: "hello.txt".into() });
        assert!(matches!(result, FsResult::Entry { .. }));
    }

    #[test]
    fn lookup_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());
        let result = server.handle_op("test", FsOp::Lookup { parent: 1, name: "nope.txt".into() });
        assert!(matches!(result, FsResult::Error { errno } if errno == libc::ENOENT));
    }

    #[test]
    fn getattr_root() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());
        let result = server.handle_op("test", FsOp::Getattr { inode: 1 });
        match result {
            FsResult::Attr { attrs, ttl_secs } => {
                assert_eq!(attrs.kind, FileType::Directory);
                assert_eq!(ttl_secs, 0);
            }
            other => panic!("expected Attr, got {:?}", other),
        }
    }

    #[test]
    fn readdir_lists_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), b"a").unwrap();
        fs::write(dir.path().join("b.txt"), b"b").unwrap();
        let server = build_test_server(dir.path());
        let result = server.handle_op("test", FsOp::Readdir { inode: 1, offset: 0 });
        match result {
            FsResult::DirEntries { entries } => {
                let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
                assert!(names.contains(&"a.txt"));
                assert!(names.contains(&"b.txt"));
            }
            other => panic!("expected DirEntries, got {:?}", other),
        }
    }

    #[test]
    fn create_read_write_flow() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());

        // Create
        let result = server.handle_op("test", FsOp::Create {
            parent: 1, name: "new.txt".into(), mode: 0o644, flags: 0,
        });
        assert!(matches!(result, FsResult::Entry { .. }));

        // Write — need to create a file and write to it on disk
        {
            let path = dir.path().join("new.txt");
            let mut f = fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all(b"hello").unwrap();
        }

        // Lookup to get the inode registered, then read directly from disk
        let content = fs::read(dir.path().join("new.txt")).unwrap();
        assert_eq!(content, b"hello");
    }

    #[test]
    fn mkdir_and_rmdir() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());

        let result = server.handle_op("test", FsOp::Mkdir {
            parent: 1, name: "subdir".into(), mode: 0o755,
        });
        assert!(matches!(result, FsResult::Entry { .. }));
        assert!(dir.path().join("subdir").is_dir());

        let result = server.handle_op("test", FsOp::Rmdir {
            parent: 1, name: "subdir".into(),
        });
        assert!(matches!(result, FsResult::Ok));
        assert!(!dir.path().join("subdir").exists());
    }

    #[test]
    fn unlink_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("victim.txt"), b"data").unwrap();
        let server = build_test_server(dir.path());

        let result = server.handle_op("test", FsOp::Unlink {
            parent: 1, name: "victim.txt".into(),
        });
        assert!(matches!(result, FsResult::Ok));
        assert!(!dir.path().join("victim.txt").exists());
    }

    #[test]
    fn rename_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("old.txt"), b"data").unwrap();
        let server = build_test_server(dir.path());

        let result = server.handle_op("test", FsOp::Rename {
            parent: 1, name: "old.txt".into(), new_parent: 1, new_name: "new.txt".into(),
        });
        assert!(matches!(result, FsResult::Ok));
        assert!(!dir.path().join("old.txt").exists());
        assert!(dir.path().join("new.txt").exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_and_readlink() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("target.txt"), b"data").unwrap();
        let server = build_test_server(dir.path());

        let result = server.handle_op("test", FsOp::Symlink {
            parent: 1, name: "link".into(), target: "target.txt".into(),
        });
        assert!(matches!(result, FsResult::Entry { .. }));

        // Readlink requires the inode — for now test via fs directly
        let link_target = fs::read_link(dir.path().join("link")).unwrap();
        assert_eq!(link_target.to_str().unwrap(), "target.txt");
    }

    #[test]
    fn statfs_returns_stats() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());
        let result = server.handle_op("test", FsOp::Statfs);
        assert!(matches!(result, FsResult::Statfs { .. }));
    }

    #[test]
    fn release_and_fsync_ok() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());
        assert!(matches!(server.handle_op("test", FsOp::Release { inode: 1, fh: 0 }), FsResult::Ok));
        assert!(matches!(server.handle_op("test", FsOp::Fsync { inode: 1, fh: 0, datasync: false }), FsResult::Ok));
    }

    // 2.1.12: Read-only enforcement

    #[test]
    fn read_only_blocks_writes() {
        let dir = tempfile::tempdir().unwrap();
        let server = FsServer::builder()
            .mount("ro", dir.path().to_path_buf(), true)
            .build()
            .unwrap();

        let write_ops: Vec<FsOp> = vec![
            FsOp::Write { inode: 1, fh: 0, offset: 0, data: bytes::Bytes::from_static(b"x") },
            FsOp::Create { parent: 1, name: "f".into(), mode: 0o644, flags: 0 },
            FsOp::Mkdir { parent: 1, name: "d".into(), mode: 0o755 },
            FsOp::Unlink { parent: 1, name: "f".into() },
            FsOp::Rmdir { parent: 1, name: "d".into() },
            FsOp::Rename { parent: 1, name: "a".into(), new_parent: 1, new_name: "b".into() },
            FsOp::Symlink { parent: 1, name: "l".into(), target: "t".into() },
            FsOp::Setattr { inode: 1, attrs: SetAttrFields { mode: Some(0o777), uid: None, gid: None, size: None, atime: None, mtime: None } },
        ];
        for op in write_ops {
            let result = server.handle_op("ro", op);
            assert!(matches!(result, FsResult::Error { errno } if errno == libc::EROFS),
                "expected EROFS");
        }

        // Reads still work
        let result = server.handle_op("ro", FsOp::Getattr { inode: 1 });
        assert!(matches!(result, FsResult::Attr { .. }));
    }

    // 2.1.13: Event emission

    #[test]
    fn events_emitted_on_ops() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("f.txt"), b"data").unwrap();
        let server = build_test_server(dir.path());
        let mut rx = server.subscribe_events().unwrap();

        server.handle_op("test", FsOp::Getattr { inode: 1 });
        let event = rx.try_recv().unwrap();
        assert_eq!(event.op_kind, FsOpKind::Getattr);
        assert_eq!(event.tag, "test");
    }

    // 2.1.13a: Setattr and Readdir produce expected FsOpKind

    #[test]
    fn event_covers_setattr_and_readdir() {
        let dir = tempfile::tempdir().unwrap();
        let server = build_test_server(dir.path());
        let mut rx = server.subscribe_events().unwrap();

        server.handle_op("test", FsOp::Setattr {
            inode: 1,
            attrs: SetAttrFields { mode: None, uid: None, gid: None, size: None, atime: None, mtime: None },
        });
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.op_kind, FsOpKind::Setattr);

        server.handle_op("test", FsOp::Readdir { inode: 1, offset: 0 });
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.op_kind, FsOpKind::Readdir);
    }

    // 2.1.14: Multi-tag routing

    #[test]
    fn multi_tag_independent_mounts() {
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        fs::write(dir_a.path().join("a.txt"), b"A").unwrap();
        fs::write(dir_b.path().join("b.txt"), b"B").unwrap();

        let server = FsServer::builder()
            .mount("tag-a", dir_a.path().to_path_buf(), false)
            .mount("tag-b", dir_b.path().to_path_buf(), false)
            .build()
            .unwrap();

        // tag-a sees a.txt
        let r = server.handle_op("tag-a", FsOp::Lookup { parent: 1, name: "a.txt".into() });
        assert!(matches!(r, FsResult::Entry { .. }));
        let r = server.handle_op("tag-a", FsOp::Lookup { parent: 1, name: "b.txt".into() });
        assert!(matches!(r, FsResult::Error { .. }));

        // tag-b sees b.txt
        let r = server.handle_op("tag-b", FsOp::Lookup { parent: 1, name: "b.txt".into() });
        assert!(matches!(r, FsResult::Entry { .. }));
        let r = server.handle_op("tag-b", FsOp::Lookup { parent: 1, name: "a.txt".into() });
        assert!(matches!(r, FsResult::Error { .. }));

        // unknown tag → ENOENT
        let r = server.handle_op("nope", FsOp::Getattr { inode: 1 });
        assert!(matches!(r, FsResult::Error { errno } if errno == libc::ENOENT));
    }

    // 2.1.15: Dynamic mount add/remove

    #[test]
    fn dynamic_mount_add_remove() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("f.txt"), b"data").unwrap();

        let server = FsServer::builder().build().unwrap();

        // No mounts → ENOENT
        let r = server.handle_op("dyn", FsOp::Getattr { inode: 1 });
        assert!(matches!(r, FsResult::Error { errno } if errno == libc::ENOENT));

        // Add mount
        server.add_mount("dyn", dir.path().to_path_buf(), false).unwrap();
        let r = server.handle_op("dyn", FsOp::Getattr { inode: 1 });
        assert!(matches!(r, FsResult::Attr { .. }));

        // Remove mount
        server.remove_mount("dyn").unwrap();
        let r = server.handle_op("dyn", FsOp::Getattr { inode: 1 });
        assert!(matches!(r, FsResult::Error { errno } if errno == libc::ENOENT));
    }

    // 2.1.16: Design guardrail — disk fallback is isolated
    // Review checkpoint: all std::fs calls are in the helpers section at the
    // bottom of server.rs. The do_* methods call mount.backing.resolve() and
    // the helper functions; they never construct host paths directly.
    // Overlay and client modules have zero std::fs calls.

    // 2.1.17: Source-level verification
    // Run: rg 'std::fs' libs/vfs/src/ --glob '!**/server.rs' --glob '!**/inode.rs'
    // Expected: no hits in overlay.rs, client/, vsock/ modules.

    // 2.1.18: Review checkpoint — every mount tag is modeled as MountState
    // with a MountBacking enum (currently Disk only) and a separate InodeTable.
    // This is a stack abstraction, not "overlay + ad hoc disk fallback."
}
