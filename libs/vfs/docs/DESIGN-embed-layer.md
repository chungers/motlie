# DESIGN — Embedded Layer + Access Logging for motlie-vfs

> Extension to `motlie-vfs`. First consumer: `bins/mstream` distributes its
> `.agents/skills/project/` skill tree baked into the binary and mounts it over
> FUSE on startup.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-28 | @claude-vfs-fuse | Initial draft. Scope: (1) embedded static layer-backing in the overlay, populated at build time via `include_dir!`; (2) synchronous access-logging observer hook. Decisions locked with David: new static layer-backing (not seeded MemOverlay); `include_dir!` whole-dir tooling; extend `motlie-vfs` (no new crate). |

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
backing enum so a layer is either the existing in-mem map or a static tree:

```rust
enum LayerSource {
    Mem(HashMap<(String, String), OverlayNode>),   // existing behavior
    Static(StaticLayer),                            // new: embedded
}

struct StaticLayer {
    dir: &'static include_dir::Dir<'static>,
    tag: String,           // which mount tag this tree serves
    attrs: OverlayAttrs,   // default mode/uid/gid for embedded entries (RO)
    // mtime policy: fixed epoch or build timestamp (TBD §8)
}
```

Resolution functions gain a match on `LayerSource`:
- `resolve(tag, path)` → `Static`: `dir.get_file(rel)` → `Content(Bytes::from_static(..))`;
  `dir.get_dir(rel)` → `SyntheticDir`. Path normalized: overlay `/a/b` → include_dir `a/b`.
- `readdir_children(tag, path)` → `Static`: enumerate `dir`'s entries at that path.
- Mutations (`put`/`whiteout`/`remove`/`set_xattr`) on a `Static` layer → error.

The server's `do_readdir` / `do_lookup` / `do_read` are unchanged — they call the
overlay's resolve/readdir which now transparently include static layers. This
keeps the merge/shadow logic in one place.

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
    .observer(tracing_sink())
    .mount("/mnt/skills")?;                   // -> LocalMount
// Composition is fully programmatic: add disk / embedded / mem layers in any
// number; priority (desc) sets shadowing. Embedded > disk; mem > embedded.
```

### 4.6 Data flow
```
read("/SKILL.md")
  └ FUSE -> FuseClient closure -> FsServer::handle_op(tag, Read)
       ├ policy.check(...)            (allow/deny)
       ├ resolve: mem? -> embedded? -> disk?   (priority desc; embedded zero-copy)
       ├ FsResult::Data{bytes}
       └ observer.on_access({Read, path:/SKILL.md, bytes, errno:None, latency})
```

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
- **Embedded entry attrs.** Default mode (0o444/0o555?), owner (mount owner
  override vs build-time), and **mtime policy** (fixed epoch for reproducible
  builds vs build timestamp; `include_dir` `metadata` feature can carry mtime).
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
