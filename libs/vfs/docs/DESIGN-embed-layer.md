# DESIGN — Embedded Layer + Access Logging for motlie-vfs

> Tracking issue: [#590](https://github.com/chungers/motlie/issues/590).
> Extension to `motlie-vfs`. First consumer: `bins/mstream` distributes its
> `.agents/skills/project/` skill tree baked into the binary and mounts it over
> FUSE on startup.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-28 | @claude-vfs-fuse | Initial draft. Scope: (1) embedded static layer-backing in the overlay, populated at build time via `include_dir!`; (2) synchronous access-logging observer hook. Decisions locked with David: new static layer-backing (not seeded MemOverlay); `include_dir!` whole-dir tooling; extend `motlie-vfs` (no new crate). |
| 2026-06-28 | @claude-vfs-fuse | Add §4.6 Runtime ownership: file uid/gid discovered at runtime (`geteuid/getegid`, default `Owner::CurrentUser`), never baked; maps onto existing mount owner-override; omit `AllowOther` for single-user local mount. Resolves the owner part of §6.2. |
| 2026-06-29 | @claude-vfs-fuse | Expand §4.1 with the concrete code shape for the static layer: `LayerSource` enum, `StaticLayer` resolution, `Layer` accessor dispatch, `put_static_layer` constructor, and `mem_entries_mut` RO gating. Server unchanged (only overlay method bodies change). Records the A-vs-B + copy-up analysis: seeding gives mutate-in-place, not COW; true COW needs a separate `do_write` copy-up change (server.rs:1343) orthogonal to layer choice. |
| 2026-06-29 | @claude-vfs-fuse | Add §4.8 Consumer walkthrough (mstream): build-time asset designation via `include_dir!` in the binary crate, runtime composition into layers, and the asset → layer → tag → mountpoint mapping (with table). Path mapping is identity (whole-dir); multiple trees/disk compose by priority; multiple mounts = multiple builders. |

---

## 1. Problem

We want a Rust binary to **carry a tree of files baked in at build time** and, at
runtime, **mount and serve them over FUSE** as part of a composed filesystem,
while **logging all read/write access** (tracing / OTEL).

`motlie-vfs` already provides the FUSE engine and a layered in-memory overlay
with union/shadow/whiteout semantics (verified: `do_readdir` server.rs:1104,
`MemOverlay::readdir_children` overlay.rs:522, overlay-before-disk resolution at
server.rs:1646/1297). What it lacks:

1. A layer whose contents are **embedded in the binary** (no host directory, no
   runtime injection calls). Today all overlay content is either host-disk or
   runtime `put()`.
2. A **reliable, path-aware access log**. The existing `FsEvent` broadcast is
   lossy (`try_send`) and emits an empty `path` / `None` bytes
   (`emit_event` server.rs:404). Only the server knows inode→path, so logging
   cannot be reconstructed from outside.

### First use case (mstream)
`mstream` bakes `.agents/skills/project/` into its binary. On startup it mounts
that tree over FUSE at a configurable path (default: cwd), so the binary itself
distributes both working code and skill files. Access to those files is logged.

## 2. Goals / Non-Goals

### Goals
- A new **embedded (static) layer** in the overlay stack, backed by a
  `&'static include_dir::Dir`, resolved lazily, read-only by construction.
- Compose embedded + in-mem + disk layers programmatically, preserving the
  existing union/shadow/whiteout semantics (embedded shadows disk; in-mem
  shadows embedded).
- **Build-time tooling**: `include_dir!` bakes a whole directory tree; the
  consuming binary owns the macro call.
- **Access logging**: a synchronous observer hook on every op carrying
  `(op_kind, tag, resolved path, bytes, errno, latency)` → tracing/OTEL.
- A thin local-mount helper (in-process `FuseClient` over `handle_op` +
  `spawn_mount2`) returning an unmount-on-drop handle.

### Non-Goals (v1)
- No selective/glob file specification or path remapping (whole-dir only;
  see Alternatives).
- No compression of embedded content.
- No write-back to the embedded layer (it is read-only; writes land in an
  in-mem layer above it or are denied — see §6).
- No change to vsock / VM guest paths.
- No persistence/transport of log events beyond the caller's sink.

## 3. Requirements

### Functional
- FR-1 A layer may be backed by `&'static include_dir::Dir`; `resolve`,
  `readdir`, `getattr`, `read`, xattr-list behave as for a read-only tree.
  Each static layer wraps exactly one `Dir`; **multiple embedded layers**
  (distinct trees, distinct priorities) may be composed in one stack.
- FR-2 Embedded entries participate in the existing priority-ordered union:
  same-name entry in a higher layer shadows the embedded one; embedded shadows
  disk; whiteouts in higher layers hide embedded entries.
- FR-3 Mutations targeting the embedded layer fail cleanly (EROFS); mutations
  to paths *shadowing* embedded content go to a writable layer if present.
- FR-4 Directory structure derives from the baked `Dir` — no synthetic-parent
  bookkeeping needed for embedded entries.
- FR-5 A synchronous observer is invoked for every `handle_op`, after dispatch,
  with resolved path, op kind, byte count (read/write), errno, and latency.
- FR-6 A local mount helper mounts an `FsServer` in-process over FUSE and
  returns a handle that unmounts on drop.

### Non-Functional
- NF-1 Zero-copy embedded reads: content served as `Bytes::from_static` slices
  into `.rodata`; no per-read allocation.
- NF-2 Observer must not drop events (synchronous, unlike the lossy broadcast)
  and must add negligible overhead when no observer is installed.
- NF-3 `include_dir` dependency is feature-gated (`embed`); core users unaffected.
- NF-4 No `unwrap`/`expect` in library paths; `thiserror` for new error types.

## 4. Design (approved direction)

### 4.1 Layer source abstraction (overlay.rs)
Today a `Layer` owns `entries: HashMap<(tag,path), OverlayNode>`. We introduce a
backing enum so a layer is either the existing in-mem map or a static tree, then
push the Mem-vs-Static difference behind a few `Layer` accessors so the existing
resolution call sites barely change.

**Core types:**
```rust
enum LayerSource {
    Mem(HashMap<(String, String), OverlayNode>),    // existing behavior
    Static(StaticLayer),                            // new: embedded
}

struct StaticLayer {
    dir: &'static include_dir::Dir<'static>,
    tag: String,          // single-tag binding
    mode_file: u32,       // baked, e.g. 0o444   (uid/gid set at runtime via §4.6)
    mode_dir:  u32,       // baked, e.g. 0o555
    mtime: SystemTime,    // §8 policy (fixed epoch or build time)
}

struct Layer {
    name: String,
    priority: u32,
    source: LayerSource,  // was: entries: HashMap<...>
}
```

**Static resolution (the new logic):**
```rust
impl StaticLayer {
    fn lookup(&self, tag: &str, path: &str) -> Option<OverlayEntryKind> {
        if tag != self.tag { return None; }
        let rel = path.trim_start_matches('/');                 // "/a/b" -> "a/b"
        if rel.is_empty() { return Some(OverlayEntryKind::SyntheticDir); }   // root
        if let Some(f) = self.dir.get_file(rel) {
            return Some(OverlayEntryKind::Content(Bytes::from_static(f.contents()))); // zero-copy
        }
        self.dir.get_dir(rel).map(|_| OverlayEntryKind::SyntheticDir)
    }

    fn children(&self, tag: &str, dir_path: &str) -> Vec<(String, OverlayEntryKind)> {
        if tag != self.tag { return vec![]; }
        let rel = dir_path.trim_start_matches('/');
        let dir = if rel.is_empty() { self.dir }
                  else { match self.dir.get_dir(rel) { Some(d) => d, None => return vec![] } };
        dir.entries().iter().map(|e| {
            let name = e.path().file_name().unwrap().to_string_lossy().into_owned();
            match e {
                include_dir::DirEntry::Dir(_)  => (name, OverlayEntryKind::SyntheticDir),
                include_dir::DirEntry::File(f) => (name, OverlayEntryKind::Content(Bytes::from_static(f.contents()))),
            }
        }).collect()
    }
}
```

**Dispatch — `Layer` accessors (so call sites don't branch):**
```rust
impl Layer {
    fn lookup(&self, tag: &str, path: &str) -> Option<OverlayEntryKind> {
        match &self.source {
            LayerSource::Mem(m)    => m.get(&(tag.into(), path.into())).map(|n| n.kind.clone()),
            LayerSource::Static(s) => s.lookup(tag, path),
        }
    }
    fn children(&self, tag: &str, dir: &str) -> Vec<(String, OverlayEntryKind)> {
        match &self.source {
            LayerSource::Mem(_)    => self.mem_children(tag, dir),  // existing is_direct_child logic
            LayerSource::Static(s) => s.children(tag, dir),
        }
    }
    fn attrs(&self, tag: &str, path: &str) -> Option<OverlayAttrs> { /* same shape */ }
}
```

Existing public methods shrink to dispatch, e.g.:
```rust
pub fn resolve(&self, tag: &str, path: &str) -> Option<(String, OverlayEntryKind)> {
    let snap = self.load_snapshot(tag);
    for layer in &snap.layers {                       // priority desc, unchanged
        if let Some(kind) = layer.lookup(tag, path) { return Some((layer.name.clone(), kind)); }
    }
    None
}
```
`readdir_children` and `resolve_attrs` change the same way: `layer.entries.get(...)`
→ `layer.lookup/children/attrs(...)`.

**Constructor + read-only gating:**
```rust
impl MemOverlay {
    pub fn put_static_layer(&self, name: &str, priority: u32, tag: &str,
                            dir: &'static include_dir::Dir<'static>, attrs: OverlayAttrs) -> Result<()> {
        let mut layers = self.layers.lock();
        layers.insert(name.into(), Arc::new(Layer {
            name: name.into(), priority,
            source: LayerSource::Static(StaticLayer { dir, tag: tag.into(),
                mode_file: attrs.mode, mode_dir: 0o555, mtime: attrs.mtime }),
        }));
        self.republish_all_tags(&layers);
        Ok(())
    }
}

// every mutation path routes through this — Static rejects:
fn mem_entries_mut<'a>(arc: &'a mut Arc<Layer>, name: &str)
    -> Result<&'a mut HashMap<(String,String), OverlayNode>> {
    match &mut Arc::make_mut(arc).source {
        LayerSource::Mem(m)    => Ok(m),
        LayerSource::Static(_) => bail!("layer {name} is read-only (embedded)"),
    }
}
```
`apply_batch`, `set_xattr`, `remove_xattr`, `create_dir` swap `get_layer_mut` →
`mem_entries_mut`, so embedded writes fail cleanly (EROFS upstream). xattr getters
on a `Static` layer return `ENODATA`.

**Blast radius / notes:**
- **Server (server.rs): zero change** — it already calls `overlay.resolve` /
  `readdir_children` / `resolve_attrs`; only their bodies change. So
  `do_readdir` / `do_lookup` / `do_read` transparently include static layers and
  the merge/shadow logic stays in one place.
- **Snapshots unchanged** — `TagSnapshot { layers: Vec<Arc<Layer>> }`; a Static
  layer Arc-clones for free (just the `&'static` ref).
- **`Bytes::from_static` requires `'static`** — holds because the consuming binary
  declares `static SKILLS: Dir = include_dir!(…)`.
- **Feature gating:** the `embed` feature pulls `include_dir`; gate the `Static`
  variant + match arms with `#[cfg(feature = "embed")]`, or keep `include_dir` as
  a light always-on dep to avoid cfg'd enum arms (decide in PLAN).

### 4.2 Build-time tooling
The consuming binary owns the bake; the macro path is binary-relative:

```rust
// bins/mstream
use include_dir::{include_dir, Dir};
static SKILLS: Dir = include_dir!("$CARGO_MANIFEST_DIR/.agents/skills/project");
```

vfs exposes a registration entry point (no macro in vfs). One call per `Dir`;
register as many as you like, each its own layer at its own priority:

```rust
overlay.put_static_layer("skills", 50, tag, &SKILLS, OverlayAttrs::ro());
overlay.put_static_layer("docs",   40, tag, &DOCS,   OverlayAttrs::ro());
// "skills" shadows "docs" shadows disk for same-named paths.
```

### 4.3 Access-logging observer (core)
A synchronous hook mirroring the existing `PolicyFn`, invoked inline in
`handle_op` after dispatch:

```rust
pub struct FsAccess<'a> {
    pub op: FsOpKind,
    pub tag: &'a str,
    pub path: &'a str,      // resolved, mount-relative
    pub bytes: Option<usize>,
    pub errno: Option<i32>, // None = ok
    pub latency: Duration,
}
pub trait FsObserver: Send + Sync + 'static {
    fn on_access(&self, a: &FsAccess<'_>);
}
// builder: .observer(impl FsObserver)
```

The app supplies a `tracing`/OTEL implementation. The lossy `FsEvent` broadcast
stays for streaming consumers; we additionally populate its `path`/`bytes`
(one-line fix) but do not rely on it for audit.

### 4.4 Local mount helper (client::local)
```rust
pub struct LocalMount { /* BackgroundSession */ }
pub fn mount_local(server: Arc<FsServer>, tag: &str, mountpoint: &Path,
                   ro: bool) -> Result<LocalMount>;
// internally: FuseClient::new(move |op| server.handle_op(tag, op)) + spawn_mount2
```
`LocalMount` unmounts on drop.

### 4.5 Composition API (thin builder)
```rust
EmbeddedFs::builder()
    .disk_base("/work", /*ro*/ false)        // optional bottom layer
    .embedded("skills", 50, &SKILLS)         // static layer (one Dir)
    .embedded("docs",   40, &DOCS)           // another static layer (another Dir)
    .mem_layer("runtime", 100)               // optional writable top
    .owner(Owner::CurrentUser)               // default: runtime geteuid/getegid (§4.6)
    .observer(tracing_sink())
    .mount("/mnt/skills")?;                   // -> LocalMount
// Composition is fully programmatic: add disk / embedded / mem layers in any
// number; priority (desc) sets shadowing. Embedded > disk; mem > embedded.
```

### 4.6 Runtime ownership (no bake)
For the user-space case (mstream), file ownership is **discovered at runtime**,
never baked. The daemon runs as the invoking user, so its own euid/egid are that
user:

```rust
enum Owner { CurrentUser, Fixed(u32, u32), Root }   // default: CurrentUser
fn resolve(o: Owner) -> Option<(u32, u32)> {
    match o {
        Owner::CurrentUser => Some(unsafe { (libc::geteuid(), libc::getegid()) }),
        Owner::Fixed(u, g) => Some((u, g)),
        Owner::Root        => Some((0, 0)),
    }
}
```

This maps onto the **existing** mount owner-override: `FsServerBuilder::mount_as
(tag, path, ro, Some((uid, gid)))`. vfs already rewrites the returned `FileAttr`
through `apply_owner_override` at all 7 attr-returning sites (server.rs:106, 933,
1017, 1187, 1467, 1598, 1894), **independent of entry kind** — so embedded/static
entries inherit the discovered uid/gid for free. Embedded layers therefore bake
only mode bits; uid/gid stay 0 in the binary.

Mount option note: for the single-user local case we **omit `AllowOther`** (vfs's
`v1_mount_options` sets it for the root-managed VM service). Only the mounting
user sees the mount — the desired default, and it avoids the
`/etc/fuse.conf user_allow_other` requirement.

### 4.7 Data flow
```
read("/SKILL.md")
  └ FUSE -> FuseClient closure -> FsServer::handle_op(tag, Read)
       ├ policy.check(...)            (allow/deny)
       ├ resolve: mem? -> embedded? -> disk?   (priority desc; embedded zero-copy)
       ├ FsResult::Data{bytes}
       └ observer.on_access({Read, path:/SKILL.md, bytes, errno:None, latency})
```

### 4.8 Consumer walkthrough: asset → mount mapping (mstream)
End-to-end, a consumer does three things: **designate assets at build time**,
**compose them into layers under a tag**, and **mount that tag at a path**.

**(1) Build-time — designate the assets** (in the consuming binary crate):
```toml
# bins/mstream/Cargo.toml
[dependencies]
motlie-vfs  = { path = "../../libs/vfs", features = ["client", "embed"] }
include_dir = "0.7"
```
```rust
// bins/mstream/src/skills.rs  — the include_dir! macro must live in the binary,
// since the baked path + bytes are binary-specific. This *is* the designation.
use include_dir::{include_dir, Dir};
pub static SKILLS: Dir = include_dir!("$CARGO_MANIFEST_DIR/.agents/skills/project");
```

**(2)+(3) Runtime — map assets → layers → tag → mountpoint:**
```rust
use motlie_vfs::embed::{EmbeddedFs, Owner};
use motlie_vfs::client::local::LocalMount;

fn serve_skills(mountpoint: &std::path::Path) -> anyhow::Result<LocalMount> {
    EmbeddedFs::builder()
        .tag("skills")                       // the mount namespace
        .embedded("skill-files", 50, &crate::skills::SKILLS) // asset Dir -> RO static layer
        // .mem_layer("scratch", 100)        // optional writable upper (off in v1; see §6 copy-up)
        .read_only(true)
        .owner(Owner::CurrentUser)           // geteuid/getegid at runtime (§4.6)
        .observer(TracingObserver::new())    // logs every read/write access
        .mount(mountpoint)                   // -> LocalMount (unmounts on drop)
}

// startup
let mp = cli.mount_path.unwrap_or(std::env::current_dir()?);   // default: cwd
let _skills = serve_skills(&mp)?;            // held in a guard for the process lifetime
// ... run mstream; on exit/signal, drop(_skills) unmounts.
```

**The mapping chain:**

| Stage | Artifact | API |
|------|----------|-----|
| designate | `static SKILLS: Dir = include_dir!(…)` | (binary crate) |
| → layer | RO static layer `"skill-files"`, priority 50, bound to tag `skills` | `.embedded(name, prio, &DIR)` |
| → tag | `skills` = composed namespace (this + any mem/disk layers) | `.tag("skills")` |
| → mount | one FUSE mount at `mountpoint` | `.mount(path) -> LocalMount` |

**Path mapping is identity (whole-dir):** a file at `<DIR>/a/b.md` is served at
`<mountpoint>/a/b.md`. A directory `<DIR>/sub` lists under `<mountpoint>/sub`.

**Multiple asset trees / a disk base** compose into the same tag by priority —
embedded shadows disk; a higher-priority embedded layer shadows a lower one:
```rust
EmbeddedFs::builder()
    .tag("agent")
    .embedded("skills", 50, &SKILLS)         // SKILLS shadows DOCS shadows disk
    .embedded("docs",   40, &DOCS)
    .disk_base("/work", /*ro*/ false)        // real fs at the bottom
    .owner(Owner::CurrentUser)
    .mount("/mnt/agent")?;
```
**Several independent mounts** = several builders (each its own tag + mountpoint +
`LocalMount`); each returns a handle the app holds and drops to unmount.

## 5. System design / components to test
- **StaticLayer resolution** (overlay): file/dir lookup, readdir enumeration,
  path normalization, RO mutation rejection.
- **Union/shadow with embedded**: embedded shadows disk; mem shadows embedded;
  whiteout hides embedded. (Extends existing overlay tests.)
- **Observer**: invoked exactly once per op; correct path/bytes/errno/latency;
  zero overhead when absent; never drops.
- **Local mount**: mount/serve/read a baked tree; unmount on drop; RO enforcement.
- **mstream integration**: bake skill dir, mount at default path, log access.

## 6. Open decisions (for interactive design)
- **Read-only vs writable mount.** Embedded layer is RO. Is the *mount* RO
  (writes denied, EROFS) or writable-scratch (writes land in a mem layer above)?
  Affects whether `mem_layer` is default-on and whiteout semantics.
  - **Copy-up note:** vfs has **no copy-up today**. `do_write` (server.rs:1343)
    resolves the *winning* layer and writes back into that same layer
    (mutate-in-place); with a Static lower layer the write fails (EROFS), with a
    seeded mem layer it would corrupt the embedded original. True overlayfs-style
    COW (write to lower-only file → copy content up into the writable layer,
    leave lower intact) is a **separate `do_write`/`do_setattr` change**,
    orthogonal to the A/B layer choice. v1 default: RO → EROFS; copy-up is a
    future feature if writable embedded files are needed.
- **Embedded entry attrs.** ~~owner~~ **RESOLVED (§4.6):** owner is discovered at
  runtime via `geteuid/getegid` (default `Owner::CurrentUser`), never baked.
  Still open: default mode (0o444/0o555?) and **mtime policy** (fixed epoch for
  reproducible builds vs build timestamp; `include_dir` `metadata` feature can
  carry mtime).
- **Tag binding of a static layer.** Bound to one tag (simplest) vs tag-agnostic
  serving its tree under any tag.
- **Observer vs broadcast.** Confirm we add the sync observer (recommended) and
  only minimally fix the existing event; or invest in the broadcast instead.
- **mstream mount UX.** Default mount path (cwd hides existing contents during
  mount); foreground vs background lifecycle; unmount on signal/exit.

## 7. Alternatives considered (appendix)
- **A1 — Seed a normal MemOverlay layer with `Bytes::from_static`.** No engine
  change; embedded = ordinary RO Content layer. *Rejected (David):* layer is
  technically mutable, requires an up-front O(files) seed walk, and lacks a
  clean read-only-by-type guarantee. Chosen approach makes embedded a distinct,
  immutable, lazily-resolved backing.
- **A2 — Manifest/glob build tooling with path remapping.** More flexible
  (select files, remap, multiple sources) but more machinery. *Deferred:* v1
  uses `include_dir!` whole-dir; revisit if selective embedding is needed.
- **A3 — Separate crate for embed+mount.** *Rejected:* fragments the overlay
  engine; a `embed` feature gate keeps `include_dir` out of the core without a
  new crate.

## 8. Logistics (for PLAN, not blocking DESIGN)
- `libs/vfs` is **not** currently a workspace member (`fuser` absent from
  `Cargo.lock`); wiring it in + enabling `client`/`embed` features is required
  before mstream can depend on it.
- New deps: `include_dir` (feature `embed`); `tracing`/`opentelemetry` already
  in the workspace lockfile.
