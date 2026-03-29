//! InodeTable: unified per-mount inode namespace (disk + overlay + synthetic).
//!
//! Each mount tag gets its own `InodeTable`. Inode 1 is always the mount root.
//! The table is the authoritative inode namespace — it is not a cache.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::{bail, Result};

use super::op::{FileAttr, FileType};

/// What kind of entry backs this inode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InodeKind {
    /// Backed by a host filesystem path (base layer in v1).
    Disk,
    /// In-memory file content injected via overlay `put()`.
    Content,
    /// Directory that exists only in memory (created implicitly by `put()` for missing parents).
    SyntheticDir,
    /// Tombstone that hides a lower-layer entry.
    Whiteout,
}

/// A single entry in the inode table.
#[derive(Debug, Clone)]
pub struct InodeEntry {
    pub inode: u64,
    pub kind: InodeKind,
    /// Mount-relative path (e.g. `/.ssh/id_ed25519`).
    pub path: String,
    /// Host filesystem path. `Some` for `Disk` entries, `None` for overlay entries.
    pub host_path: Option<PathBuf>,
    /// Monotonically increasing per inode; bumped on content/kind change.
    pub generation: u64,
    /// FUSE lookup reference count.
    pub refcount: u64,
    /// Authoritative for overlay entries; re-fetched from `std::fs` for disk entries in v1.
    pub attrs: FileAttr,
}

/// Per-mount inode namespace.
///
/// Maps inodes to entries and provides reverse path→inode lookup.
/// Inode 1 is reserved for the mount root.
pub struct InodeTable {
    next_inode: u64,
    entries: HashMap<u64, InodeEntry>,
    path_to_inode: HashMap<String, u64>,
}

impl InodeTable {
    /// Create a new inode table with the root inode pre-allocated.
    pub fn new(root_attrs: FileAttr) -> Self {
        let mut entries = HashMap::new();
        let mut path_to_inode = HashMap::new();

        let root = InodeEntry {
            inode: 1,
            kind: InodeKind::Disk,
            path: "/".to_string(),
            host_path: None, // set by caller after construction
            generation: 0,
            refcount: 1,
            attrs: root_attrs,
        };
        entries.insert(1, root);
        path_to_inode.insert("/".to_string(), 1);

        Self {
            next_inode: 2,
            entries,
            path_to_inode,
        }
    }

    /// Allocate a new inode for the given path and kind.
    ///
    /// If the path already has an inode, returns the existing inode number
    /// and bumps generation if the kind changed.
    pub fn allocate(
        &mut self,
        path: &str,
        kind: InodeKind,
        host_path: Option<PathBuf>,
        attrs: FileAttr,
    ) -> Result<u64> {
        if let Some(&existing) = self.path_to_inode.get(path) {
            let entry = match self.entries.get_mut(&existing) {
                Some(e) => e,
                None => bail!(
                    "internal inconsistency: path_to_inode maps {:?} to inode {} \
                     but entries has no such inode",
                    path, existing,
                ),
            };
            if entry.kind != kind {
                entry.generation += 1;
                entry.kind = kind;
            }
            entry.host_path = host_path;
            let mut attrs = attrs;
            attrs.inode = existing;
            entry.attrs = attrs;
            return Ok(existing);
        }

        let inode = self.next_inode;
        self.next_inode += 1;

        let mut attrs = attrs;
        attrs.inode = inode;
        let entry = InodeEntry {
            inode,
            kind,
            path: path.to_string(),
            host_path,
            generation: 0,
            refcount: 0,
            attrs,
        };
        self.entries.insert(inode, entry);
        self.path_to_inode.insert(path.to_string(), inode);
        Ok(inode)
    }

    /// Look up an entry by inode number.
    pub fn get(&self, inode: u64) -> Option<&InodeEntry> {
        self.entries.get(&inode)
    }

    /// Look up a mutable entry by inode number.
    pub fn get_mut(&mut self, inode: u64) -> Option<&mut InodeEntry> {
        self.entries.get_mut(&inode)
    }

    /// Reverse lookup: path → inode.
    pub fn lookup_path(&self, path: &str) -> Option<u64> {
        self.path_to_inode.get(path).copied()
    }

    /// Bump the generation counter for an inode (e.g. on content/kind change).
    pub fn bump_generation(&mut self, inode: u64) {
        if let Some(entry) = self.entries.get_mut(&inode) {
            entry.generation += 1;
        }
    }

    /// Remove an inode entry. Returns the removed entry if it existed.
    pub fn remove(&mut self, inode: u64) -> Option<InodeEntry> {
        if let Some(entry) = self.entries.remove(&inode) {
            self.path_to_inode.remove(&entry.path);
            Some(entry)
        } else {
            None
        }
    }

    /// Remove an entry by path. Returns the removed entry if it existed.
    pub fn remove_path(&mut self, path: &str) -> Option<InodeEntry> {
        if let Some(inode) = self.path_to_inode.remove(path) {
            self.entries.remove(&inode)
        } else {
            None
        }
    }

    /// Increment the FUSE lookup refcount.
    pub fn inc_ref(&mut self, inode: u64) {
        if let Some(entry) = self.entries.get_mut(&inode) {
            entry.refcount += 1;
        }
    }

    /// Decrement the FUSE lookup refcount. Returns the new count.
    pub fn dec_ref(&mut self, inode: u64, count: u64) -> u64 {
        if let Some(entry) = self.entries.get_mut(&inode) {
            entry.refcount = entry.refcount.saturating_sub(count);
            entry.refcount
        } else {
            0
        }
    }

    /// Drop all entries. Used when a mount is removed.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.path_to_inode.clear();
        self.next_inode = 2;
    }

    /// Number of allocated inodes (including root).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Create a default root `FileAttr` for a mount root directory.
pub fn default_root_attrs() -> FileAttr {
    let now = SystemTime::now();
    FileAttr {
        inode: 1,
        size: 0,
        blocks: 0,
        atime: now,
        mtime: now,
        ctime: now,
        kind: FileType::Directory,
        mode: 0o755,
        nlink: 2,
        uid: 0,
        gid: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file_attrs(inode: u64, kind: FileType) -> FileAttr {
        let now = SystemTime::now();
        FileAttr {
            inode,
            size: 0,
            blocks: 0,
            atime: now,
            mtime: now,
            ctime: now,
            kind,
            mode: if kind == FileType::Directory { 0o755 } else { 0o644 },
            nlink: 1,
            uid: 1000,
            gid: 1000,
        }
    }

    #[test]
    fn root_inode_is_one() {
        let table = InodeTable::new(default_root_attrs());
        assert_eq!(table.len(), 1);
        let root = table.get(1).unwrap();
        assert_eq!(root.inode, 1);
        assert_eq!(root.path, "/");
        assert_eq!(root.kind, InodeKind::Disk);
    }

    #[test]
    fn allocate_disk_entry() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let inode = table.allocate(
            "/hello.txt",
            InodeKind::Disk,
            Some(PathBuf::from("/host/hello.txt")),
            attrs,
        ).unwrap();
        assert_eq!(inode, 2);
        let entry = table.get(inode).unwrap();
        assert_eq!(entry.kind, InodeKind::Disk);
        assert_eq!(entry.host_path.as_ref().unwrap().to_str().unwrap(), "/host/hello.txt");
        assert_eq!(entry.generation, 0);
        assert_eq!(entry.attrs.inode, inode, "stored attrs.inode must match table inode");

        // Reverse lookup
        assert_eq!(table.lookup_path("/hello.txt"), Some(2));
    }

    #[test]
    fn allocate_overlay_entries() {
        let mut table = InodeTable::new(default_root_attrs());

        // Content entry (no host_path)
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let content_inode = table.allocate("/.env", InodeKind::Content, None, attrs).unwrap();
        let entry = table.get(content_inode).unwrap();
        assert_eq!(entry.kind, InodeKind::Content);
        assert!(entry.host_path.is_none());

        // SyntheticDir entry (no host_path)
        let attrs = make_file_attrs(0, FileType::Directory);
        let dir_inode = table.allocate("/.ssh", InodeKind::SyntheticDir, None, attrs).unwrap();
        let entry = table.get(dir_inode).unwrap();
        assert_eq!(entry.kind, InodeKind::SyntheticDir);
        assert!(entry.host_path.is_none());

        // Whiteout entry (no host_path)
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let wo_inode = table.allocate("/hidden.txt", InodeKind::Whiteout, None, attrs).unwrap();
        let entry = table.get(wo_inode).unwrap();
        assert_eq!(entry.kind, InodeKind::Whiteout);
        assert!(entry.host_path.is_none());

        assert_eq!(table.len(), 4); // root + 3
    }

    #[test]
    fn reuse_inode_on_same_path() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);

        let first = table.allocate("/file", InodeKind::Disk, Some(PathBuf::from("/h/file")), attrs.clone()).unwrap();
        assert_eq!(table.get(first).unwrap().generation, 0);

        // Same path, same kind → same inode, no generation bump
        let second = table.allocate("/file", InodeKind::Disk, Some(PathBuf::from("/h/file")), attrs.clone()).unwrap();
        assert_eq!(first, second);
        assert_eq!(table.get(first).unwrap().generation, 0);
    }

    #[test]
    fn generation_bump_on_kind_change() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);

        // Start as Whiteout
        let inode = table.allocate("/file", InodeKind::Whiteout, None, attrs.clone()).unwrap();
        assert_eq!(table.get(inode).unwrap().generation, 0);

        // Replace with Content → generation bump
        let same = table.allocate("/file", InodeKind::Content, None, attrs.clone()).unwrap();
        assert_eq!(inode, same);
        assert_eq!(table.get(inode).unwrap().generation, 1);

        // Replace with Whiteout again → another bump
        let same = table.allocate("/file", InodeKind::Whiteout, None, attrs.clone()).unwrap();
        assert_eq!(inode, same);
        assert_eq!(table.get(inode).unwrap().generation, 2);
    }

    #[test]
    fn explicit_generation_bump() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let inode = table.allocate("/f", InodeKind::Content, None, attrs).unwrap();

        table.bump_generation(inode);
        assert_eq!(table.get(inode).unwrap().generation, 1);
        table.bump_generation(inode);
        assert_eq!(table.get(inode).unwrap().generation, 2);
    }

    #[test]
    fn remove_by_inode_and_path() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);

        let a = table.allocate("/a", InodeKind::Content, None, attrs.clone()).unwrap();
        let b = table.allocate("/b", InodeKind::Content, None, attrs).unwrap();

        // Remove by inode
        let removed = table.remove(a).unwrap();
        assert_eq!(removed.path, "/a");
        assert!(table.get(a).is_none());
        assert!(table.lookup_path("/a").is_none());

        // Remove by path
        let removed = table.remove_path("/b").unwrap();
        assert_eq!(removed.inode, b);
        assert!(table.get(b).is_none());
        assert!(table.lookup_path("/b").is_none());
    }

    #[test]
    fn mount_removal_clears_all() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);
        table.allocate("/a", InodeKind::Disk, Some(PathBuf::from("/h/a")), attrs.clone()).unwrap();
        table.allocate("/b", InodeKind::Content, None, attrs.clone()).unwrap();
        table.allocate("/.ssh", InodeKind::SyntheticDir, None, attrs).unwrap();
        assert_eq!(table.len(), 4);

        table.clear();
        assert!(table.is_empty());
        assert!(table.get(1).is_none());
        assert!(table.lookup_path("/a").is_none());
    }

    #[test]
    fn refcount_tracking() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let inode = table.allocate("/f", InodeKind::Disk, None, attrs).unwrap();

        assert_eq!(table.get(inode).unwrap().refcount, 0);
        table.inc_ref(inode);
        table.inc_ref(inode);
        assert_eq!(table.get(inode).unwrap().refcount, 2);
        let remaining = table.dec_ref(inode, 1);
        assert_eq!(remaining, 1);
        let remaining = table.dec_ref(inode, 5); // saturating
        assert_eq!(remaining, 0);
    }

    // 1.3.12: Review checkpoint — InodeKind has no variant that structurally
    // requires a host_path. `host_path` is Option<PathBuf> on InodeEntry,
    // not on InodeKind. A future SyntheticRoot backing could set host_path=None
    // for Disk-like entries or use a new InodeKind variant without any
    // structural redesign of InodeTable.
    #[test]
    fn future_non_disk_entries_require_no_redesign() {
        let mut table = InodeTable::new(default_root_attrs());
        let attrs = make_file_attrs(0, FileType::Directory);
        // A "disk-like" entry with no host_path — simulating a future
        // SyntheticRoot mount backing.
        let inode = table.allocate("/virtual-root", InodeKind::Disk, None, attrs).unwrap();
        let entry = table.get(inode).unwrap();
        assert_eq!(entry.kind, InodeKind::Disk);
        assert!(entry.host_path.is_none());
        // No panic, no error — the structure accommodates it.
    }

    #[test]
    fn attrs_inode_always_matches_table_inode() {
        let mut table = InodeTable::new(default_root_attrs());

        // Root: attrs.inode == 1
        assert_eq!(table.get(1).unwrap().attrs.inode, 1);

        // New allocation: caller passes inode: 0, table rewrites to actual inode
        let attrs = make_file_attrs(0, FileType::RegularFile);
        let inode = table.allocate("/a", InodeKind::Content, None, attrs).unwrap();
        assert_eq!(table.get(inode).unwrap().attrs.inode, inode);

        // Reuse path with kind change: attrs.inode still correct
        let attrs = make_file_attrs(999, FileType::RegularFile);
        let same = table.allocate("/a", InodeKind::Whiteout, None, attrs).unwrap();
        assert_eq!(inode, same);
        assert_eq!(table.get(same).unwrap().attrs.inode, same);
    }
}
