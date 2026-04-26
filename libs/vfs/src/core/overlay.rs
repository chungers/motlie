//! MemOverlay: layered in-memory content injection with batch-atomic publication.
//!
//! The overlay maintains named layers with priority ordering. Each layer holds
//! `(tag, path) → OverlayNode` entries. Resolution walks layers highest-to-lowest.
//! Mutations are batch-first: `apply_batch()` is the primitive, single-op methods
//! are convenience wrappers. Per-tag snapshots are published atomically via `arc-swap`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{anyhow, bail, Result};
use arc_swap::ArcSwap;
use bytes::Bytes;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What an overlay entry contains in the internal memfs model.
#[derive(Clone, Debug)]
pub enum OverlayEntryKind {
    Content(Bytes),
    Symlink(String),
    Whiteout,
    SyntheticDir,
}

/// Metadata-only view used by `list_layer()` / `list_effective()`.
#[derive(Clone, Debug)]
pub enum OverlayEntryViewKind {
    Content { size: usize },
    Symlink { target: String },
    Whiteout,
    SyntheticDir,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OverlayAttrs {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
}

#[derive(Debug, Clone)]
pub enum OverlayMutation {
    Put {
        layer: String,
        path: String,
        attrs: Option<OverlayAttrs>,
        content: Bytes,
    },
    Whiteout {
        layer: String,
        path: String,
    },
    Remove {
        layer: String,
        path: String,
    },
}

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub priority: u32,
    pub entry_count: usize,
}

#[derive(Debug, Clone)]
pub struct OverlayEntry {
    pub path: String,
    pub kind: OverlayEntryViewKind,
    pub inode: u64,
    pub size: usize,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub injected_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct EffectiveEntry {
    pub path: String,
    pub layer: String,
    pub kind: OverlayEntryViewKind,
    pub inode: u64,
    pub size: usize,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A single node in a memfs layer.
#[derive(Clone, Debug)]
struct OverlayNode {
    kind: OverlayEntryKind,
    mode: u32,
    uid: u32,
    gid: u32,
    xattrs: HashMap<String, Bytes>,
    injected_at: SystemTime,
    /// Number of child entries in this layer that depend on this synthetic dir.
    child_count: u32,
}

impl OverlayNode {
    fn size(&self) -> usize {
        match &self.kind {
            OverlayEntryKind::Content(b) => b.len(),
            OverlayEntryKind::Symlink(target) => target.len(),
            _ => 0,
        }
    }

    fn view_kind(&self) -> OverlayEntryViewKind {
        match &self.kind {
            OverlayEntryKind::Content(b) => OverlayEntryViewKind::Content { size: b.len() },
            OverlayEntryKind::Symlink(target) => OverlayEntryViewKind::Symlink {
                target: target.clone(),
            },
            OverlayEntryKind::Whiteout => OverlayEntryViewKind::Whiteout,
            OverlayEntryKind::SyntheticDir => OverlayEntryViewKind::SyntheticDir,
        }
    }
}

/// A named layer with a priority. Entries are keyed by `(tag, path)`.
#[derive(Clone, Debug)]
struct Layer {
    name: String,
    priority: u32,
    entries: HashMap<(String, String), OverlayNode>,
}

/// Immutable snapshot of all layers for one tag. Published atomically.
#[derive(Clone, Debug, Default)]
pub struct TagSnapshot {
    /// Layers sorted by priority descending (highest first).
    layers: Vec<Arc<Layer>>,
}

// ---------------------------------------------------------------------------
// MemOverlay
// ---------------------------------------------------------------------------

/// File-driven in-memory filesystem layer.
/// Injecting a file automatically materializes synthetic parent directories.
/// Accessed via `server.overlay()`.
pub struct MemOverlay {
    /// Master layer registry (name → Layer). Writers clone, mutate, republish.
    layers: Mutex<BTreeMap<String, Arc<Layer>>>,
    /// Per-tag published snapshot. Readers load one snapshot per request.
    tag_snapshots: Mutex<HashMap<String, Arc<ArcSwap<TagSnapshot>>>>,
    /// Default ownership for synthetic entries when no explicit attrs given.
    default_uid: u32,
    default_gid: u32,
}

impl MemOverlay {
    pub fn new() -> Self {
        Self {
            layers: Mutex::new(BTreeMap::new()),
            tag_snapshots: Mutex::new(HashMap::new()),
            default_uid: 0,
            default_gid: 0,
        }
    }
}

impl Default for MemOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl MemOverlay {
    // --- Layer management ---

    pub fn put_layer(&self, name: &str, priority: u32) -> Result<()> {
        let mut layers = self.layers.lock();
        let layer = layers.entry(name.to_string()).or_insert_with(|| {
            Arc::new(Layer {
                name: name.to_string(),
                priority,
                entries: HashMap::new(),
            })
        });
        // Update priority if layer already exists
        if Arc::strong_count(layer) > 0 {
            let mut new_layer = (**layer).clone();
            new_layer.priority = priority;
            *layer = Arc::new(new_layer);
        }
        self.republish_all_tags(&layers);
        Ok(())
    }

    pub fn remove_layer(&self, name: &str) -> Result<()> {
        let mut layers = self.layers.lock();
        if layers.remove(name).is_none() {
            bail!("unknown layer: {name}");
        }
        self.republish_all_tags(&layers);
        Ok(())
    }

    pub fn layers(&self) -> Vec<LayerInfo> {
        let layers = self.layers.lock();
        let mut infos: Vec<_> = layers.values().map(|l| LayerInfo {
            name: l.name.clone(),
            priority: l.priority,
            entry_count: l.entries.len(),
        }).collect();
        infos.sort_by(|a, b| b.priority.cmp(&a.priority));
        infos
    }

    /// Return all distinct tags that have at least one entry across all layers.
    pub fn tags(&self) -> Vec<String> {
        let layers = self.layers.lock();
        let mut tags: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for layer in layers.values() {
            for (tag, _) in layer.entries.keys() {
                tags.insert(tag.clone());
            }
        }
        tags.into_iter().collect()
    }

    // --- Content management ---

    pub fn put(&self, layer: &str, tag: &str, path: &str, content: Bytes) -> Result<()> {
        self.apply_batch(tag, &[OverlayMutation::Put {
            layer: layer.to_string(),
            path: path.to_string(),
            attrs: None,
            content,
        }])
    }

    pub fn put_with_attrs(
        &self, layer: &str, tag: &str, path: &str,
        attrs: OverlayAttrs, content: Bytes,
    ) -> Result<()> {
        self.apply_batch(tag, &[OverlayMutation::Put {
            layer: layer.to_string(),
            path: path.to_string(),
            attrs: Some(attrs),
            content,
        }])
    }

    pub fn whiteout(&self, layer: &str, tag: &str, path: &str) -> Result<()> {
        self.apply_batch(tag, &[OverlayMutation::Whiteout {
            layer: layer.to_string(),
            path: path.to_string(),
        }])
    }

    /// Create an explicit SyntheticDir entry in the overlay.
    pub fn create_dir(&self, layer: &str, tag: &str, path: &str, attrs: OverlayAttrs) -> Result<()> {
        validate_path(path)?;
        let mut layers = self.layers.lock();
        let l = get_layer_mut(&mut layers, layer)?;
        let now = SystemTime::now();
        materialize_parents(l, tag, path, attrs.uid, attrs.gid, now);
        let key = (tag.to_string(), path.to_string());
        l.entries.insert(key, OverlayNode {
            kind: OverlayEntryKind::SyntheticDir,
            mode: attrs.mode, uid: attrs.uid, gid: attrs.gid,
            xattrs: HashMap::new(),
            injected_at: now,
            child_count: 0,
        });
        self.republish_tag(&layers, tag);
        Ok(())
    }

    /// Create an explicit Symlink entry in the overlay.
    pub fn create_symlink(
        &self,
        layer: &str,
        tag: &str,
        path: &str,
        target: &str,
        attrs: OverlayAttrs,
    ) -> Result<()> {
        validate_path(path)?;
        let mut layers = self.layers.lock();
        let l = get_layer_mut(&mut layers, layer)?;
        let now = SystemTime::now();
        materialize_parents(l, tag, path, attrs.uid, attrs.gid, now);
        let key = (tag.to_string(), path.to_string());
        l.entries.insert(
            key,
            OverlayNode {
                kind: OverlayEntryKind::Symlink(target.to_string()),
                mode: attrs.mode,
                uid: attrs.uid,
                gid: attrs.gid,
                xattrs: HashMap::new(),
                injected_at: now,
                child_count: 0,
            },
        );
        self.republish_tag(&layers, tag);
        Ok(())
    }

    /// Get the overlay metadata (mode, uid, gid) for a resolved entry.
    pub fn resolve_attrs(&self, tag: &str, path: &str) -> Option<(String, OverlayEntryKind, OverlayAttrs)> {
        let snap = self.load_snapshot(tag);
        let key_tag = tag.to_string();
        let key_path = path.to_string();
        for layer in &snap.layers {
            if let Some(node) = layer.entries.get(&(key_tag.clone(), key_path.clone())) {
                return Some((
                    layer.name.clone(),
                    node.kind.clone(),
                    OverlayAttrs { mode: node.mode, uid: node.uid, gid: node.gid },
                ));
            }
        }
        None
    }

    pub fn remove(&self, layer: &str, tag: &str, path: &str) -> Result<()> {
        self.apply_batch(tag, &[OverlayMutation::Remove {
            layer: layer.to_string(),
            path: path.to_string(),
        }])
    }

    pub fn set_xattr(
        &self,
        tag: &str,
        path: &str,
        name: &str,
        value: Bytes,
        flags: i32,
        position: u32,
    ) -> Result<(), i32> {
        if position != 0 {
            return Err(libc::ENOTSUP);
        }
        let snap = self.load_snapshot(tag);
        let key = (tag.to_string(), path.to_string());
        let Some(layer_name) = snap.layers.iter().find_map(|layer| {
            layer.entries.get(&key).map(|_| layer.name.clone())
        }) else {
            return Err(libc::ENOENT);
        };

        let mut layers = self.layers.lock();
        let layer = get_layer_mut(&mut layers, &layer_name).map_err(|_| libc::ENOENT)?;
        let Some(node) = layer.entries.get_mut(&key) else {
            return Err(libc::ENOENT);
        };
        let exists = node.xattrs.contains_key(name);
        if flags & libc::XATTR_CREATE != 0 && exists {
            return Err(libc::EEXIST);
        }
        if flags & libc::XATTR_REPLACE != 0 && !exists {
            return Err(libc::ENODATA);
        }
        node.xattrs.insert(name.to_string(), value);
        self.republish_tag(&layers, tag);
        Ok(())
    }

    pub fn get_xattr(&self, tag: &str, path: &str, name: &str) -> Result<Bytes, i32> {
        let snap = self.load_snapshot(tag);
        let key = (tag.to_string(), path.to_string());
        let Some(node) = snap.layers.iter().find_map(|layer| layer.entries.get(&key)) else {
            return Err(libc::ENOENT);
        };
        node.xattrs.get(name).cloned().ok_or(libc::ENODATA)
    }

    pub fn list_xattrs(&self, tag: &str, path: &str) -> Result<Vec<u8>, i32> {
        let snap = self.load_snapshot(tag);
        let key = (tag.to_string(), path.to_string());
        let Some(node) = snap.layers.iter().find_map(|layer| layer.entries.get(&key)) else {
            return Err(libc::ENOENT);
        };
        let mut names: Vec<_> = node.xattrs.keys().cloned().collect();
        names.sort();
        let mut out = Vec::new();
        for name in names {
            out.extend_from_slice(name.as_bytes());
            out.push(0);
        }
        Ok(out)
    }

    pub fn remove_xattr(&self, tag: &str, path: &str, name: &str) -> Result<(), i32> {
        let snap = self.load_snapshot(tag);
        let key = (tag.to_string(), path.to_string());
        let Some(layer_name) = snap.layers.iter().find_map(|layer| {
            layer.entries.get(&key).map(|_| layer.name.clone())
        }) else {
            return Err(libc::ENOENT);
        };

        let mut layers = self.layers.lock();
        let layer = get_layer_mut(&mut layers, &layer_name).map_err(|_| libc::ENOENT)?;
        let Some(node) = layer.entries.get_mut(&key) else {
            return Err(libc::ENOENT);
        };
        if node.xattrs.remove(name).is_none() {
            return Err(libc::ENODATA);
        }
        self.republish_tag(&layers, tag);
        Ok(())
    }

    pub fn apply_batch(&self, tag: &str, ops: &[OverlayMutation]) -> Result<()> {
        let mut layers = self.layers.lock();

        for op in ops {
            match op {
                OverlayMutation::Put { layer, path, attrs, content } => {
                    validate_path(path)?;
                    let l = get_layer_mut(&mut layers, layer)?;
                    let now = SystemTime::now();
                    let (uid, gid, mode) = match attrs {
                        Some(a) => (a.uid, a.gid, a.mode),
                        None => (self.default_uid, self.default_gid, 0o644),
                    };

                    // Materialize synthetic parents for ancestors missing from this layer
                    materialize_parents(l, tag, path, uid, gid, now);

                    let key = (tag.to_string(), path.to_string());
                    let (xattrs, child_count) = match l.entries.get(&key) {
                        Some(existing) => (existing.xattrs.clone(), existing.child_count),
                        None => (HashMap::new(), 0),
                    };
                    l.entries.insert(key, OverlayNode {
                        kind: OverlayEntryKind::Content(content.clone()),
                        mode, uid, gid,
                        xattrs,
                        injected_at: now,
                        child_count,
                    });
                }
                OverlayMutation::Whiteout { layer, path } => {
                    validate_path(path)?;
                    let l = get_layer_mut(&mut layers, layer)?;
                    let now = SystemTime::now();
                    materialize_parents(l, tag, path, self.default_uid, self.default_gid, now);
                    let key = (tag.to_string(), path.to_string());
                    l.entries.insert(key, OverlayNode {
                        kind: OverlayEntryKind::Whiteout,
                        mode: 0, uid: 0, gid: 0,
                        xattrs: HashMap::new(),
                        injected_at: now,
                        child_count: 0,
                    });
                }
                OverlayMutation::Remove { layer, path } => {
                    validate_path(path)?;
                    let l = get_layer_mut(&mut layers, layer)?;
                    let key = (tag.to_string(), path.to_string());
                    if l.entries.remove(&key).is_some() {
                        prune_parents(l, tag, path);
                    }
                }
            }
        }

        // Publish new snapshot for this tag
        self.republish_tag(&layers, tag);
        Ok(())
    }

    pub fn get(&self, layer: &str, tag: &str, path: &str) -> Option<Bytes> {
        let layers = self.layers.lock();
        let l = layers.get(layer)?;
        let key = (tag.to_string(), path.to_string());
        match &l.entries.get(&key)?.kind {
            OverlayEntryKind::Content(b) => Some(b.clone()),
            _ => None,
        }
    }

    pub fn symlink_target(&self, tag: &str, path: &str) -> Option<String> {
        let snap = self.load_snapshot(tag);
        let key = (tag.to_string(), path.to_string());
        snap.layers.iter().find_map(|layer| match layer.entries.get(&key) {
            Some(OverlayNode {
                kind: OverlayEntryKind::Symlink(target),
                ..
            }) => Some(target.clone()),
            _ => None,
        })
    }

    pub fn list_layer(&self, layer: &str, tag: &str) -> Vec<OverlayEntry> {
        let layers = self.layers.lock();
        let l = match layers.get(layer) {
            Some(l) => l,
            None => return vec![],
        };
        l.entries.iter()
            .filter(|((t, _), _)| t == tag)
            .map(|((_, p), node)| OverlayEntry {
                path: p.clone(),
                kind: node.view_kind(),
                inode: 0, // inode allocation is in InodeTable, not here
                size: node.size(),
                mode: node.mode,
                uid: node.uid,
                gid: node.gid,
                injected_at: node.injected_at,
            })
            .collect()
    }

    // --- Resolved view ---

    /// Resolve a path: walk layers highest-to-lowest, return first hit.
    /// None means the caller should continue into the base layer.
    pub fn resolve(&self, tag: &str, path: &str) -> Option<(String, OverlayEntryKind)> {
        let snap = self.load_snapshot(tag);
        let key_tag = tag.to_string();
        let key_path = path.to_string();
        for layer in &snap.layers {
            if let Some(node) = layer.entries.get(&(key_tag.clone(), key_path.clone())) {
                return Some((layer.name.clone(), node.kind.clone()));
            }
        }
        None
    }

    /// List all effective overlays for a tag (one entry per path, from the
    /// highest-priority layer that owns it).
    pub fn list_effective(&self, tag: &str) -> Vec<EffectiveEntry> {
        let snap = self.load_snapshot(tag);
        let mut seen = HashMap::new();
        for layer in &snap.layers {
            for ((t, p), node) in &layer.entries {
                if t == tag && !seen.contains_key(p) {
                    seen.insert(p.clone(), EffectiveEntry {
                        path: p.clone(),
                        layer: layer.name.clone(),
                        kind: node.view_kind(),
                        inode: 0,
                        size: node.size(),
                        mode: node.mode,
                        uid: node.uid,
                        gid: node.gid,
                    });
                }
            }
        }
        seen.into_values().collect()
    }

    /// Load the current snapshot for a tag. Used by read-like ops to get one
    /// consistent view for the duration of a request.
    pub fn load_snapshot(&self, tag: &str) -> Arc<TagSnapshot> {
        let snaps = self.tag_snapshots.lock();
        match snaps.get(tag) {
            Some(swap) => swap.load_full(),
            None => Arc::new(TagSnapshot::default()),
        }
    }

    /// Collect entries from the snapshot for readdir merging.
    /// Returns children of `dir_path` from all memfs layers (highest priority first).
    pub fn readdir_children(&self, tag: &str, dir_path: &str) -> Vec<(String, OverlayEntryKind)> {
        let snap = self.load_snapshot(tag);
        let prefix = if dir_path == "/" { "/".to_string() } else { format!("{dir_path}/") };
        let mut seen: HashMap<String, OverlayEntryKind> = HashMap::new();

        // Walk highest to lowest; first entry per child name wins
        for layer in &snap.layers {
            for ((t, p), node) in &layer.entries {
                if t != tag { continue; }
                if !is_direct_child(p, dir_path, &prefix) { continue; }
                let name = child_name(p, dir_path);
                seen.entry(name).or_insert_with(|| node.kind.clone());
            }
        }
        seen.into_iter().collect()
    }

    // --- Internal ---

    fn republish_tag(&self, layers: &BTreeMap<String, Arc<Layer>>, tag: &str) {
        let mut sorted: Vec<Arc<Layer>> = layers.values().cloned().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        let snapshot = Arc::new(TagSnapshot { layers: sorted });

        let mut snaps = self.tag_snapshots.lock();
        snaps
            .entry(tag.to_string())
            .or_insert_with(|| Arc::new(ArcSwap::from_pointee(TagSnapshot::default())))
            .store(snapshot);
    }

    fn republish_all_tags(&self, layers: &BTreeMap<String, Arc<Layer>>) {
        // Collect all tags referenced by any layer
        let mut tags = std::collections::HashSet::new();
        for layer in layers.values() {
            for (tag, _) in layer.entries.keys() {
                tags.insert(tag.clone());
            }
        }
        // Also republish tags that might have had entries removed
        {
            let snaps = self.tag_snapshots.lock();
            for tag in snaps.keys() {
                tags.insert(tag.clone());
            }
        }
        for tag in &tags {
            self.republish_tag(layers, tag);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_path(path: &str) -> Result<()> {
    if !path.starts_with('/') {
        bail!("path must be mount-relative and start with '/': {path}");
    }
    Ok(())
}

fn get_layer_mut<'a>(
    layers: &'a mut BTreeMap<String, Arc<Layer>>,
    name: &str,
) -> Result<&'a mut Layer> {
    let arc = layers.get_mut(name)
        .ok_or_else(|| anyhow!("unknown layer: {name}"))?;
    Ok(Arc::make_mut(arc))
}

/// Materialize synthetic parent directories for all ancestors of `path` that
/// are missing from this layer for this tag.
fn materialize_parents(layer: &mut Layer, tag: &str, path: &str, uid: u32, gid: u32, now: SystemTime) {
    let mut current = path.to_string();
    loop {
        let parent = parent_path(&current);
        if parent == current { break; } // at root
        let key = (tag.to_string(), parent.clone());
        let entry = layer.entries.entry(key).or_insert_with(|| OverlayNode {
            kind: OverlayEntryKind::SyntheticDir,
            mode: 0o755,
            uid, gid,
            xattrs: HashMap::new(),
            injected_at: now,
            child_count: 0,
        });
        entry.child_count += 1;
        current = parent;
    }
}

/// Prune synthetic parent directories that have no remaining children.
fn prune_parents(layer: &mut Layer, tag: &str, path: &str) {
    let mut current = path.to_string();
    loop {
        let parent = parent_path(&current);
        if parent == current { break; }
        let key = (tag.to_string(), parent.clone());
        if let Some(entry) = layer.entries.get_mut(&key) {
            entry.child_count = entry.child_count.saturating_sub(1);
            if entry.child_count == 0 && matches!(entry.kind, OverlayEntryKind::SyntheticDir) {
                layer.entries.remove(&key);
                current = parent;
                continue;
            }
        }
        break;
    }
}

fn parent_path(path: &str) -> String {
    if path == "/" { return "/".to_string(); }
    match path.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => path[..i].to_string(),
        None => "/".to_string(),
    }
}

fn is_direct_child(path: &str, dir_path: &str, prefix: &str) -> bool {
    if dir_path == "/" {
        // Direct child of root: starts with "/" and has no further "/"
        path.starts_with('/') && path.len() > 1 && !path[1..].contains('/')
    } else {
        path.starts_with(prefix) && !path[prefix.len()..].contains('/')
    }
}

fn child_name(path: &str, dir_path: &str) -> String {
    if dir_path == "/" {
        path[1..].to_string()
    } else {
        path[dir_path.len() + 1..].to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn overlay() -> MemOverlay {
        MemOverlay::new()
    }

    // 2.2.15: Layer priority resolution
    #[test]
    fn layer_priority_resolution() {
        let o = overlay();
        o.put_layer("low", 0).unwrap();
        o.put_layer("high", 100).unwrap();
        o.put("low", "t", "/f", Bytes::from("low")).unwrap();
        o.put("high", "t", "/f", Bytes::from("high")).unwrap();

        let (layer, kind) = o.resolve("t", "/f").unwrap();
        assert_eq!(layer, "high");
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "high"));
    }

    // 2.2.16: Synthetic parent creation
    #[test]
    fn synthetic_parent_creation() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/.ssh/id_ed25519", Bytes::from("key")).unwrap();

        // /.ssh should exist as SyntheticDir
        let (_, kind) = o.resolve("t", "/.ssh").unwrap();
        assert!(matches!(kind, OverlayEntryKind::SyntheticDir));
    }

    // 2.2.17: Whiteout suppression
    #[test]
    fn whiteout_suppression() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.whiteout("l", "t", "/hidden.txt").unwrap();

        let (_, kind) = o.resolve("t", "/hidden.txt").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Whiteout));
        // Non-hidden file returns None
        assert!(o.resolve("t", "/visible.txt").is_none());
    }

    // 2.2.18: remove_layer effective-winner change
    #[test]
    fn remove_layer_effective_winner() {
        let o = overlay();
        o.put_layer("low", 0).unwrap();
        o.put_layer("high", 100).unwrap();
        o.put("low", "t", "/f", Bytes::from("low")).unwrap();
        o.put("high", "t", "/f", Bytes::from("high")).unwrap();

        // high wins
        let (layer, _) = o.resolve("t", "/f").unwrap();
        assert_eq!(layer, "high");

        // Remove high → low wins
        o.remove_layer("high").unwrap();
        let (layer, kind) = o.resolve("t", "/f").unwrap();
        assert_eq!(layer, "low");
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "low"));
    }

    // 2.2.22: Partial overlay — base layer + memfs children
    #[test]
    fn partial_overlay_readdir() {
        let o = overlay();
        o.put_layer("creds", 0).unwrap();
        o.put("creds", "home", "/.ssh/id_ed25519", Bytes::from("key")).unwrap();
        o.put("creds", "home", "/.ssh/config", Bytes::from("cfg")).unwrap();

        let children = o.readdir_children("home", "/.ssh");
        let names: Vec<_> = children.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"id_ed25519"));
        assert!(names.contains(&"config"));
    }

    // 2.2.23: Fully synthetic subtree
    #[test]
    fn fully_synthetic_subtree() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "home", "/.claude/skills/tool.md", Bytes::from("# tool")).unwrap();

        // All ancestors are synthetic
        assert!(matches!(o.resolve("home", "/.claude").unwrap().1, OverlayEntryKind::SyntheticDir));
        assert!(matches!(o.resolve("home", "/.claude/skills").unwrap().1, OverlayEntryKind::SyntheticDir));
        assert!(matches!(o.resolve("home", "/.claude/skills/tool.md").unwrap().1, OverlayEntryKind::Content(_)));
    }

    // 2.2.6 + 2.2.24: Synthetic dir pruning on remove
    #[test]
    fn synthetic_dir_pruning() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/.ssh/id_ed25519", Bytes::from("key")).unwrap();
        assert!(o.resolve("t", "/.ssh").is_some());

        o.remove("l", "t", "/.ssh/id_ed25519").unwrap();
        // Parent should be pruned since no children remain
        assert!(o.resolve("t", "/.ssh").is_none());
    }

    // 2.2.26: Shared tag — /alice/... and /bob/... coexist
    #[test]
    fn shared_tag_paths() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "home", "/alice/.claude/skills/tool.md", Bytes::from("alice")).unwrap();
        o.put("l", "home", "/bob/.claude/skills/tool.md", Bytes::from("bob")).unwrap();

        let (_, kind) = o.resolve("home", "/alice/.claude/skills/tool.md").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "alice"));
        let (_, kind) = o.resolve("home", "/bob/.claude/skills/tool.md").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "bob"));
    }

    // 2.2.27: Separate tags — overlay-isolated
    #[test]
    fn separate_tags_isolated() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "alice-home", "/.env", Bytes::from("ALICE")).unwrap();
        o.put("l", "bob-home", "/.env", Bytes::from("BOB")).unwrap();

        let (_, kind) = o.resolve("alice-home", "/.env").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "ALICE"));
        let (_, kind) = o.resolve("bob-home", "/.env").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "BOB"));
        // Cross-tag: no leakage
        assert!(o.resolve("alice-home", "/.other").is_none());
    }

    // 2.2.28: Shared layer multi-tag
    #[test]
    fn shared_layer_multi_tag() {
        let o = overlay();
        o.put_layer("claude-shared", 0).unwrap();
        o.put("claude-shared", "bob-project1", "/CLAUDE.md", Bytes::from("shared-bob")).unwrap();
        o.put("claude-shared", "alice-project2", "/CLAUDE.md", Bytes::from("shared-alice")).unwrap();

        let (_, kind) = o.resolve("bob-project1", "/CLAUDE.md").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "shared-bob"));
        let (_, kind) = o.resolve("alice-project2", "/CLAUDE.md").unwrap();
        assert!(matches!(kind, OverlayEntryKind::Content(b) if b == "shared-alice"));
    }

    // 2.2.29: Inherited ownership defaults
    #[test]
    fn inherited_ownership_defaults() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/.ssh/key", Bytes::from("k")).unwrap();

        // Default uid/gid should be 0 (MemOverlay::new defaults)
        let entries = o.list_layer("l", "t");
        let key_entry = entries.iter().find(|e| e.path == "/.ssh/key").unwrap();
        assert_eq!(key_entry.uid, 0);
        assert_eq!(key_entry.gid, 0);
        assert_eq!(key_entry.mode, 0o644);

        let dir_entry = entries.iter().find(|e| e.path == "/.ssh").unwrap();
        assert_eq!(dir_entry.mode, 0o755);
    }

    // 2.2.30: Explicit attrs override defaults
    #[test]
    fn explicit_attrs_override() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put_with_attrs("l", "t", "/secret", OverlayAttrs { mode: 0o600, uid: 1000, gid: 1000 },
            Bytes::from("s")).unwrap();

        let entries = o.list_layer("l", "t");
        let e = entries.iter().find(|e| e.path == "/secret").unwrap();
        assert_eq!(e.uid, 1000);
        assert_eq!(e.gid, 1000);
        assert_eq!(e.mode, 0o600);
    }

    #[test]
    fn overlay_xattr_round_trip() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/f", Bytes::from("content")).unwrap();

        o.set_xattr("t", "/f", "user.note", Bytes::from("value"), 0, 0).unwrap();
        assert_eq!(o.get_xattr("t", "/f", "user.note").unwrap(), Bytes::from("value"));
        let listed = o.list_xattrs("t", "/f").unwrap();
        assert_eq!(listed, b"user.note\0");
        o.remove_xattr("t", "/f", "user.note").unwrap();
        assert_eq!(o.get_xattr("t", "/f", "user.note"), Err(libc::ENODATA));
    }

    #[test]
    fn overlay_put_preserves_xattrs() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/f", Bytes::from("v1")).unwrap();
        o.set_xattr("t", "/f", "user.note", Bytes::from("value"), 0, 0).unwrap();

        o.put("l", "t", "/f", Bytes::from("v2")).unwrap();

        assert_eq!(o.get("l", "t", "/f").unwrap(), "v2");
        assert_eq!(o.get_xattr("t", "/f", "user.note").unwrap(), Bytes::from("value"));
    }

    // 2.2.33: Batch atomicity — all or nothing visibility
    #[test]
    fn batch_atomicity() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.apply_batch("t", &[
            OverlayMutation::Put { layer: "l".into(), path: "/.ssh/config".into(), attrs: None, content: Bytes::from("c") },
            OverlayMutation::Put { layer: "l".into(), path: "/.ssh/id_ed25519".into(), attrs: None, content: Bytes::from("k") },
            OverlayMutation::Put { layer: "l".into(), path: "/.ssh/id_ed25519.pub".into(), attrs: None, content: Bytes::from("p") },
        ]).unwrap();

        // All three should be visible
        assert!(o.resolve("t", "/.ssh/config").is_some());
        assert!(o.resolve("t", "/.ssh/id_ed25519").is_some());
        assert!(o.resolve("t", "/.ssh/id_ed25519.pub").is_some());
    }

    // 2.2.34: Synthetic parent atomicity
    #[test]
    fn synthetic_parent_atomicity() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        // Before batch: nothing
        assert!(o.resolve("t", "/.deep/nested/dir").is_none());

        o.put("l", "t", "/.deep/nested/dir/file.txt", Bytes::from("f")).unwrap();
        // After batch: all parents and file visible
        assert!(o.resolve("t", "/.deep").is_some());
        assert!(o.resolve("t", "/.deep/nested").is_some());
        assert!(o.resolve("t", "/.deep/nested/dir").is_some());
        assert!(o.resolve("t", "/.deep/nested/dir/file.txt").is_some());
    }

    // 2.2.14: Path validation
    #[test]
    fn invalid_path_rejected() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        assert!(o.put("l", "t", "no-slash", Bytes::from("x")).is_err());
    }

    // 2.2.32: Overlay APIs use mount-relative paths, not host paths
    #[test]
    fn overlay_uses_mount_relative_paths() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "home", "/alice/.ssh/key", Bytes::from("k")).unwrap();

        let effective = o.list_effective("home");
        for entry in &effective {
            assert!(entry.path.starts_with('/'), "path should be mount-relative: {}", entry.path);
            assert!(!entry.path.contains("/home/"), "path should not contain host prefix");
        }
    }

    // list_effective returns metadata-only views
    #[test]
    fn list_effective_is_metadata_only() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/f", Bytes::from("content")).unwrap();

        let eff = o.list_effective("t");
        let e = eff.iter().find(|e| e.path == "/f").unwrap();
        match &e.kind {
            OverlayEntryViewKind::Content { size } => assert_eq!(*size, 7),
            other => panic!("expected Content view, got {:?}", other),
        }
    }

    // get() returns actual content
    #[test]
    fn get_returns_content() {
        let o = overlay();
        o.put_layer("l", 0).unwrap();
        o.put("l", "t", "/f", Bytes::from("hello")).unwrap();

        assert_eq!(o.get("l", "t", "/f").unwrap(), "hello");
        assert!(o.get("l", "t", "/missing").is_none());
        assert!(o.get("nope", "t", "/f").is_none());
    }
}
