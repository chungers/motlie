# DESIGN — Embedded Layer + Access Logging for motlie-vfs

> Tracking issue: [#590](https://github.com/chungers/motlie/issues/590).
> PR: [#591](https://github.com/chungers/motlie/pull/591).
> **Merge target / anchor branch: `feature/vmm-vz` (`e4e0229f`).** All code
> citations below are line numbers on that branch (`libs/vfs/src/core/server.rs`
> is 5421 lines there). Extension to `motlie-vfs`. First consumer:
> `bins/mstream` distributes its `.agents/skills/project/` skill tree baked into
> the binary and can mount it over FUSE when the daemon is started with
> `--mount-skill <DIR>`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-28 | @claude-vfs-fuse | Initial draft (anchored to `main`). Scope: embedded static layer-backing + access-logging observer. |
| 2026-06-28 | @claude-vfs-fuse | §4.6 runtime ownership; §4.1 code shape; §4.8 consumer walkthrough; copy-up note. |
| 2026-06-30 | @claude-vfs-fuse | **RE-ANCHOR against `feature/vmm-vz` (`e4e0229f`)** after review (PR #591, 2 reviewers NEEDS WORK, 21 inline). Re-cited every `server.rs:NNNN`. Corrected two stale claims: (a) overlay attrs use `ov_attrs.uid/gid` directly, NOT `apply_owner_override` — embedded uid/gid set at registration; (b) "server zero change" is false — EROFS propagation + observer lock-scope require server edits. Addressed must-fixes: static-layer EROFS + error propagation, symlink semantics (bake-time reject), tag publication, observer outside the mounts RwLock, guest `client+vsock` left intact via a split `local-mount`/`fuser-client` feature, `include_dir!` path fix, fixed-epoch mtime + `rerun-if-changed`, direct-`/dev/fuse` mount, platform matrix, host deps + degrade contract. Decisions locked with David: degrade-on-mount-failure; old default runtime mountpoint decision, superseded by the 2026-07-01 explicit `--mount-skill <DIR>` cleanup; bake-time symlink reject; fixed-epoch mtime; separate static attr type; `include_dir` always-on light dep. |
| 2026-07-01 | @codex-590-impl | Implementation pass for #590. Added status plan link, noted the generic symlink-detection gap in `include_dir::Dir`, and documented that mstream enforces bake-time symlink rejection with a source-tree `build.rs` scan while the VFS static layer remains file/dir-only. |
| 2026-07-01 | @codex-590-impl | Switched mstream skills mounting to explicit `daemon start --mount-skill <DIR>`; no `--mount-skill` means no FUSE mount. Moved the baked source path into `build.rs` via `MSTREAM_SKILLS_DIR`. |
| 2026-07-01 | @codex-590-impl | Renamed mstream embedded skills identifiers to drop the redundant `project` qualifier; the on-disk `.agents/skills/project` source path is unchanged. |
| 2026-07-01 | @codex-590-impl | Renamed the mstream daemon skills mount flag to `--mount-skill`; mount semantics are unchanged. |
| 2026-07-01 | @codex-590-impl | Refused `--mount-skill` targets that cover the daemon working directory to avoid leaving callers in a disconnected FUSE mount. |
| 2026-07-01 | @codex-590-impl | Added foreground signal handling and daemon-owned artifact cleanup: socket removal, unique temporary skills backing directory, and empty created-mountpoint removal. |

---

## 1. Problem

A Rust binary should **carry a tree of files baked in at build time** and, at
runtime, **mount and serve them over FUSE** as part of a composed filesystem,
while **logging all read/write access** (tracing / OTEL).

`motlie-vfs` on `feature/vmm-vz` already provides the FUSE engine and a layered
in-memory overlay with union/shadow/whiteout **and symlink** semantics
(`OverlayEntryKind::{Content,Symlink,Whiteout,SyntheticDir}`, overlay.rs:24-29;
merge in `do_readdir`; overlay-before-disk resolution in `do_lookup`/`do_read`).
What it lacks:

1. A layer whose contents are **embedded in the binary** (no host directory, no
   runtime injection). Today every overlay entry is host-disk or runtime
   `put()`.
2. A **reliable, path-aware access log**. `emit_event` (server.rs:406-416) sends
   `path: String::new()` (line 412) with `bytes: None`; `path_hint_for_op`
   (server.rs:2139-2178) returns empty for `Read`/`Write` (line 2163). Only the
   server knows inode→path, so logging cannot be reconstructed externally.

### First use case (mstream)
`mstream` bakes `.agents/skills/project/` into its binary. When started with
`daemon start --mount-skill <DIR>`, it mounts that tree over FUSE at `<DIR>` so the
binary distributes both working code and skill files. `<DIR>` must not be the daemon
working directory or one of its parents; that configuration is rejected and degraded
to avoid leaving the caller in a disconnected FUSE mount after daemon exit. Without
`--mount-skill`, it logs the degraded state and runs without a skills mount. Access
is logged when mounted.

## 2. Goals / Non-Goals

### Goals
- A new **embedded (static) layer**: a layer backed by `&'static include_dir::Dir`,
  lazily resolved, read-only by construction, zero-copy reads (`Bytes::from_static`).
- Compose embedded + in-mem + disk layers programmatically, preserving existing
  union/shadow/whiteout/symlink merge. Multiple embedded layers (one `Dir` each).
- **Build-time tooling**: `build.rs` computes the absolute skills source path,
  emits `MSTREAM_SKILLS_DIR`, walks the tree for `cargo:rerun-if-changed` and
  symlink rejection, and the consuming binary embeds that tree deterministically
  (fixed-epoch mtime).
- **Access logging**: a synchronous observer hook carrying
  `(op_kind, tag, resolved path, bytes, errno, latency)` → tracing/OTEL,
  invoked **after** the mount lock is released.
- A Linux local in-process mount helper using the **direct blocking
  `fuser::mount2()`** path (matching vmm-vz), with explicit unmount + stale-mount
  recovery.
- **Degrade-not-die**: if the mount can't be established, the daemon logs and
  continues without it.

### Non-Goals (v1)
- **Embedded symlinks** — rejected at bake time (the static layer models files +
  dirs only; see §4.1). No symlink targets/escape policy in v1.
- Selective/glob file specification, path remapping, content compression.
- **Copy-up / writable embedded files** — embedded is read-only; writes are
  denied (`EROFS`). Real overlayfs COW is a separate future server change (§6).
- No change to the VM/vsock guest path or to `v1_mount_options` (which the
  root-managed guest service depends on).
- No persistence/transport of log events beyond the caller's sink.
- **macOS host mount** — `feature/vmm-vz` gates the FUSE mount Linux-only
  (`client/mod.rs:12`); on a macOS host the local mount is unavailable (§9).

### Deployment preconditions (new host dependency)
Adding a FUSE mount turns mstream from a pure-userspace daemon into one with host
preconditions: **`/dev/fuse` present + accessible**, the setuid **`fusermount3`**
helper (libfuse3) installed, and inside containers typically **`CAP_SYS_ADMIN`**
(or `--device /dev/fuse` + a permissive seccomp profile). See §9 for the
degrade-vs-hard-fail contract and platform matrix.

## 3. Requirements

### Functional
- FR-1 A layer may be backed by `&'static include_dir::Dir`; `lookup`, `readdir`,
  `getattr`, `read` behave as a read-only tree. Each static layer wraps one `Dir`;
  **multiple** embedded layers may compose. Accessors stay **exhaustive** over
  `OverlayEntryKind` (incl. `Symlink`), even though a static layer never produces
  `Symlink` (FR-1a).
- FR-1a **Embedded symlinks rejected at bake time** — registration fails if the
  source tree contains a symlink; `symlink_target` over a static layer is always
  `None`.
- FR-2 Embedded entries participate in the existing priority-ordered union/shadow/
  whiteout. Embedded shadows disk; a higher-priority layer shadows a lower.
- FR-3 **Writes to embedded content fail with `EROFS`.** This requires: (a)
  `writable_layer` (server.rs:430-449) must **skip `Static` layers**; (b) the
  server mutators that currently **ignore overlay errors** must propagate them
  (see §4.2). Today `do_write` (1383), `do_create` (1429-1435), `do_unlink`
  (1686/1689), `do_rmdir` (1748), `do_rename` (1824-1826/1866-1873) use
  `let _ = …` and return success regardless — they would falsely report success
  over an immutable entry.
- FR-4 Directory structure derives from the baked `Dir`; static-only tags must be
  **published** (today tag discovery iterates `layer.entries.keys()` — a static
  layer has none; see §4.1 tag-publication fix).
- FR-5 A synchronous observer is invoked for every `handle_op`, **after the
  `mounts` RwLock is released**, with resolved path, op kind, bytes, errno,
  latency. fh-based path/bytes for `Read`/`Write`/`Release` are captured
  **pre-dispatch** (Release frees the fh during dispatch; Read/Write path hints
  are empty).
- FR-6 A Linux local mount helper mounts an `FsServer` in-process via blocking
  `fuser::mount2()`, with an explicit `unmount()` and startup stale-mount recovery.
- FR-7 If the mount fails, the consumer **degrades** (logs, continues) — it does
  not abort.
- FR-8 **Deterministic builds**: embedded mtime defaults to a fixed epoch; the
  consuming binary emits `cargo:rerun-if-changed` for the baked tree.

### Non-Functional
- NF-1 Zero-copy embedded reads (`Bytes::from_static` into `.rodata`).
- NF-2 Observer never drops events (synchronous) and adds negligible overhead when
  absent; it must not run under the `mounts` RwLock.
- NF-3 The local mount path must not drag `vsock` into embedding daemons; guest
  `client+vsock` behavior unchanged.
- NF-4 No `unwrap`/`expect` in library paths; `thiserror` for new error types.

## 4. Design (re-anchored to feature/vmm-vz)

### 4.1 Static layer in the overlay (overlay.rs)
On vmm-vz a `Layer` is `{ name, priority, entries: HashMap<(String,String),
OverlayNode> }` (overlay.rs:135-140). We replace the bare `entries` field with a
`LayerSource`, then push the Mem-vs-Static difference behind `Layer` accessors so
the snapshot-walking read methods (`resolve` 531-541, `readdir_children` 579-594,
`resolve_attrs` 318-332, `symlink_target` 494-504) change only their inner
`layer.entries.get(...)` calls.

**Core types** (the `Static` variant is **unconditional** — `include_dir` is a
light always-on dep, not a cfg'd feature, to avoid cfg'd match arms across
overlay.rs):
```rust
enum LayerSource {
    Mem(HashMap<(String, String), OverlayNode>),    // existing
    Static(StaticLayer),                            // new, read-only
}

struct StaticLayer {
    dir: &'static include_dir::Dir<'static>,
    tag: String,          // single-tag binding (drives publication, below)
    uid: u32, gid: u32,   // set at REGISTRATION from Owner (§4.6) — NOT runtime override
    mode_file: u32,       // e.g. 0o444
    mode_dir:  u32,       // e.g. 0o555
    mtime: SystemTime,    // fixed epoch by default (FR-8)
}

struct Layer { name: String, priority: u32, source: LayerSource }
```
Note: `OverlayAttrs` on vmm-vz is `{ mode, uid, gid }` (overlay.rs:40-45) — **no
`mtime`**. We keep static mtime in `StaticLayer` (a static-only attr concern), not
by widening the shared `OverlayAttrs`/`OverlayNode` (overlay.rs:102-111).

**Static resolution** (files + dirs only; symlinks rejected at bake → FR-1a):

> @codex-590-impl 2026-07-01 -- Implementation note: `include_dir::Dir` carries baked files and directories but no source-tree symlink provenance in the runtime value. The VFS static layer therefore never serves symlinks and reports no static symlink targets; mstream enforces the FR-1a bake-time rejection by scanning `.agents/skills/project/` with `symlink_metadata()` in `build.rs` before `include_dir!` runs. A future generic consumer-facing registration API that must reject symlinks for arbitrary source trees needs to carry the source path or a manifest into registration.

```rust
impl StaticLayer {
    fn lookup(&self, tag: &str, path: &str) -> Option<OverlayEntryKind> {
        if tag != self.tag { return None; }
        let rel = path.trim_start_matches('/');
        if rel.is_empty() { return Some(OverlayEntryKind::SyntheticDir); }
        if let Some(f) = self.dir.get_file(rel) {
            return Some(OverlayEntryKind::Content(Bytes::from_static(f.contents())));
        }
        self.dir.get_dir(rel).map(|_| OverlayEntryKind::SyntheticDir)
    }
    fn children(&self, tag: &str, dir_path: &str) -> Vec<(String, OverlayEntryKind)> { /* enumerate Dir */ }
    fn attrs(&self, tag: &str, path: &str) -> Option<OverlayAttrs> {
        self.lookup(tag, path).map(|k| match k {
            OverlayEntryKind::SyntheticDir => OverlayAttrs { mode: self.mode_dir, uid: self.uid, gid: self.gid },
            _                              => OverlayAttrs { mode: self.mode_file, uid: self.uid, gid: self.gid },
        })
    }
    fn symlink_target(&self, _tag: &str, _path: &str) -> Option<String> { None } // FR-1a
}
```

**Dispatch — `Layer` accessors** (read methods call these instead of
`layer.entries.get`; the `Mem` arm matches all `OverlayEntryKind` incl. `Symlink`,
staying exhaustive):
```rust
impl Layer {
    fn lookup(&self, tag, path) -> Option<OverlayEntryKind> { match &self.source {
        LayerSource::Mem(m) => m.get(&(tag.into(),path.into())).map(|n| n.kind.clone()),
        LayerSource::Static(s) => s.lookup(tag, path), } }
    fn children(&self, tag, dir) -> Vec<(String, OverlayEntryKind)> { /* Mem: existing is_direct_child; Static: s.children */ }
    fn attrs(&self, tag, path) -> Option<OverlayAttrs> { /* … */ }
    fn symlink_target(&self, tag, path) -> Option<String> { match &self.source {
        LayerSource::Mem(m) => /* existing */, LayerSource::Static(s) => s.symlink_target(tag,path), } }
    fn bound_tags(&self) -> Vec<String> { match &self.source {  // for publication/discovery
        LayerSource::Mem(m) => m.keys().map(|(t,_)| t.clone()).collect(),
        LayerSource::Static(s) => vec![s.tag.clone()], } }
    fn entry_count(&self) -> usize { /* Mem: m.len(); Static: best-effort Dir count */ }
}
```

**Tag publication fix (FR-4).** `republish_all_tags` (overlay.rs:610-628) and
`tags()` (226-235) discover tags via `layer.entries.keys()` → a static-only tag is
never published and `load_snapshot` (569-575) returns an empty `TagSnapshot`.
Fix: (a) `put_static_layer` explicitly calls `republish_tag(&layers, &tag)` for
its bound tag — and since `republish_tag` (598-608) snapshots **all**
`layers.values()`, the static layer is included; (b) `tags()` /
`republish_all_tags` iterate `Layer::bound_tags()` instead of `entries.keys()`.
`list_layer`/`list_effective` (506-565) gain a `Static` branch (best-effort
enumerate the `Dir`).

**Constructor + RO gating:**
```rust
pub fn put_static_layer(&self, name: &str, priority: u32, tag: &str,
                        dir: &'static include_dir::Dir<'static>, owner: (u32,u32)) -> Result<()> {
    reject_symlinks(dir)?;                              // FR-1a, bake/registration check
    let mut layers = self.layers.lock();
    layers.insert(name.into(), Arc::new(Layer { name: name.into(), priority,
        source: LayerSource::Static(StaticLayer { dir, tag: tag.into(),
            uid: owner.0, gid: owner.1, mode_file: 0o444, mode_dir: 0o555,
            mtime: FIXED_EPOCH }) }));
    self.republish_tag(&layers, tag);                  // FR-4
    Ok(())
}
// get_layer_mut (overlay.rs:642-649) gains: Static => bail!("layer is read-only (embedded)")
```
Mutation entry points (`apply_batch` 424-482, `set_xattr` 341-376, `create_dir`
268-284) reject `Static` layers; xattr getters return `ENODATA`.

### 4.2 Server changes required (EROFS — FR-3)
"Server zero change" from the earlier draft is **withdrawn**. Two server edits:

1. **`writable_layer` (server.rs:430-449) must skip `Static`.** It currently walks
   up to the owning layer and falls back to the highest-priority layer, with no
   read-only check (RO is otherwise enforced only at mount level, 298-299). It must
   skip `Static` layers when selecting a write target.
2. **Propagate overlay mutation errors.** The mutators that today discard overlay
   errors must map a rejected write to `EROFS`:
   - `do_write` (1383), `do_create` (1429-1435), `do_unlink` (1686/1689),
     `do_rmdir` (1748), `do_rename` (1824-1826/1866-1873): replace `let _ = …`
     with error handling → `EROFS` when the overlay rejects.
   - `do_symlink` (1951) already checks its `create_symlink` result — use it as
     the pattern.

### 4.3 Access-logging observer (server.rs)
`handle_op` (server.rs:161-178) holds `mounts.read()` (line 162) across **both**
`dispatch` (175) and `emit_event` (176). A synchronous observer must **not** run
under that guard (arbitrary tracing/OTEL code could block `add_mount`/`remove_mount`
or re-enter the server). Restructure:
```rust
pub fn handle_op(&self, tag: &str, op: FsOp) -> FsResult {
    let pre = { let mounts = self.mounts.read()?;                  // guard scope 1
        let mount = mounts.get(tag).ok_or(ENOENT)?;
        let pre = self.capture_pre(mount, &op);   // fh→path/bytes for Read/Write/Release
        let result = self.dispatch(mount, &op);
        (pre, result)
    };                                                            // guard DROPPED here
    let (pre, result) = pre;
    if let Some(obs) = &self.observer {                           // observer OUTSIDE the lock
        obs.on_access(&FsAccess { op: FsOpKind::from_op(&op), tag,
            path: pre.path(&op, &result), bytes: pre.bytes(&result),
            errno: errno_of(&result), latency: pre.elapsed() });
    }
    result
}
```
```rust
pub struct FsAccess<'a> { pub op: FsOpKind, pub tag: &'a str, pub path: &'a str,
    pub bytes: Option<usize>, pub errno: Option<i32>, pub latency: Duration }
pub trait FsObserver: Send + Sync + 'static { fn on_access(&self, a: &FsAccess<'_>); }
```
Pre-dispatch capture is required because `path_hint_for_op` (2139-2178) yields an
empty path for `Read`/`Write` (2163) and `Release` removes the fh during dispatch.
The lossy `FsEvent` broadcast stays as-is for streaming consumers.

### 4.4 Local mount helper (client::local, Linux)
vmm-vz deliberately **avoids `AutoUnmount`/`spawn_mount2`** — `client/fuse.rs:6-14`
calls it "proven unreliable" — and uses blocking `fuser::mount2()`
(`guest.rs:142`). We follow that, not the earlier `spawn_mount2`/drop design:
```rust
pub struct LocalMount { mountpoint: PathBuf, join: JoinHandle<io::Result<()>> }
pub fn mount_local(server: Arc<FsServer>, tag: &str, mountpoint: &Path) -> Result<LocalMount> {
    recover_stale_mount(mountpoint);                  // lazy-unmount a wedged prior mount
    std::fs::create_dir_all(mountpoint)?;
    let join = thread::spawn(move || {
        let client = FuseClient::new(move |op| server.handle_op(&tag, op));
        fuser::mount2(client, &mountpoint, &local_mount_options())   // blocks until unmounted
    });
    Ok(LocalMount { mountpoint, join })
}
impl LocalMount {
    pub fn unmount(&self) -> Result<()> { /* fusermount3 -u (or umount2) → mount2 returns */ }
}
// local_mount_options(): Linux, RO, NO AllowOther — distinct from v1_mount_options
// (server.rs/client uses v1_mount_options for the guest; we do NOT change it).
```
- **Unmount** is explicit (`fusermount3 -u`/`libc::umount2`); the blocking
  `mount2` returns when the mountpoint is unmounted. Wire `unmount()` into
  mstream's graceful-stop path (`daemon.rs` `watch::channel` select loop).
- **Stale-mount recovery**: a `SIGKILL`ed prior daemon leaves a wedged mount
  ("Transport endpoint is not connected"); `recover_stale_mount` lazy-unmounts it
  on startup (Drop alone is insufficient — it doesn't run on signal/abort).
- **Mountpoint** does **not** default to cwd (§4.6/§9).

### 4.5 Composition API (thin builder)
```rust
EmbeddedFs::builder()
    .tag("skills")
    .embedded("skill-files", 50, &SKILLS)    // static layer (one Dir); RO
    .disk_base("/work", /*ro*/ false)        // optional bottom layer
    .owner(Owner::CurrentUser)               // resolved at registration (§4.6)
    .observer(tracing_sink())
    .mount(mountpoint)?;                      // -> LocalMount (Linux); Err on no-FUSE → caller degrades
```

### 4.6 Runtime ownership (no bake) — corrected
The daemon runs as the invoking user, so its euid/egid are that user:
```rust
enum Owner { CurrentUser, Fixed(u32,u32), Root }   // default CurrentUser
// CurrentUser => (libc::geteuid(), libc::getegid())
```
**Correction vs the prior draft:** on vmm-vz, overlay entries take
`ov_attrs.uid/gid` **directly** in `do_lookup` (922-934) and `do_getattr`
(1010-1022); `apply_owner_override` (def 2658-2663) is applied **only at the 7
disk sites** (106, 956, 1041, 1214, 1496, 1630, 1967). So embedded entries do
**not** inherit a mount owner-override "for free", and making the override global
would break explicit guest overlay ownership. Instead, the resolved `(uid, gid)`
is written **into the static layer's attrs at registration** (`put_static_layer`'s
`owner` arg → `StaticLayer.uid/gid`). Embedded files thus appear owned by the
invoking user with **no bake** and **no change to guest semantics**. The local
mount also omits `AllowOther` so only the mounting user sees it.

### 4.7 Data flow
```
read("/SKILL.md")
  └ FUSE -> FuseClient closure -> FsServer::handle_op(tag, Read)
       ├ [under mounts.read()] capture_pre(fh→path/bytes) ; dispatch:
       │     resolve: mem? -> embedded(static)? -> disk?  (priority desc; zero-copy)
       │     FsResult::Data{bytes}
       ├ [guard dropped]
       └ observer.on_access({Read, path:/SKILL.md, bytes, errno:None, latency})
```

### 4.8 Consumer walkthrough: asset → mount mapping (mstream)
**(1) Build-time — designate assets** (in the binary crate). The skill tree is at
the **workspace root**, so from `bins/mstream` the path needs `../../`:
```toml
# bins/mstream/Cargo.toml
[dependencies]
motlie-vfs  = { path = "../../libs/vfs", features = ["local-mount"] }  # NOT "client" (pulls vsock)
include_dir = "0.7"
```
```rust
// bins/mstream/build.rs  (already exists; uses rerun-if-changed for git metadata)
// FR-8: include_dir! (proc-macro, stable Rust) emits NO rerun-if-changed, so an
// edited SKILL.md with no .rs change ships a STALE embed. Compute the workspace
// root path, reject symlinks while walking, and emit both file/dir watches and
// the env var consumed by skills.rs.
fn configure_skills() -> io::Result<()> {
    let root = workspace_root()?.join(".agents/skills/project");
    watch_tree(&root)?;
    println!("cargo:rustc-env=MSTREAM_SKILLS_DIR={}", root.display());
    Ok(())
}
```
```rust
// bins/mstream/src/skills.rs
use include_dir::{include_dir, Dir};

pub const SKILLS_DIR: &str = env!("MSTREAM_SKILLS_DIR");
pub static SKILLS: Dir = include_dir!("$MSTREAM_SKILLS_DIR");
```

> @codex-590-impl 2026-07-01 -- Design gap surfaced during implementation:
> `include_dir` 0.7 does not accept `include_dir!(env!("MSTREAM_SKILLS_DIR"))`;
> the macro requires a string literal and expands `$MSTREAM_SKILLS_DIR` itself.
> `build.rs` remains the source of truth for the absolute path, and
> `SKILLS_DIR` keeps the build-provided env visible to tests.

**(2)+(3) Runtime — compose + mount, degrading on failure:**
```rust
fn serve_skills(mp: Option<PathBuf>) -> Option<LocalMount> {
    let Some(mp) = mp else {
        tracing::warn!("skills FUSE unavailable: no --mount-skill specified; continuing without mounting skills");
        return None;
    };
    match EmbeddedFs::builder().tag("skills")
            .embedded("skill-files", 50, &crate::skills::SKILLS)
            .owner(Owner::CurrentUser).observer(TracingObserver::new())
            .mount(&mp) {
        Ok(m)  => Some(m),
        Err(e) => { tracing::warn!("skills FUSE unavailable: {e}; continuing without mount"); None }
    }
}
// startup: mounting is explicit; omitted --mount-skill keeps the daemon unmounted:
let skills = serve_skills(cli.mount);    // Option<LocalMount> held for process lifetime
// graceful stop, SIGINT, SIGTERM: unmount; remove daemon-owned backing dir/socket;
// remove an empty mountpoint only if mstream created it.
```

**Mapping chain:** `static SKILLS = include_dir!(…)` → `.embedded(name, prio, &DIR)`
(RO static layer bound to tag) → `.tag("skills")` (composed namespace) →
`.mount(path)` → `LocalMount`. **Path mapping is identity** (`<DIR>/a/b.md` →
`<mountpoint>/a/b.md`). Multiple trees/disk compose by priority; several mounts =
several builders.

## 5. System design / components to test
- **StaticLayer resolution** (overlay): file/dir lookup, readdir enumeration, path
  normalization, exhaustive `OverlayEntryKind` match, RO mutation rejection.
- **EROFS propagation** (server): a write/create/unlink/rmdir/rename whose target
  resolves to a `Static` layer returns `EROFS` (regression-guards the
  `let _ = …` paths at 1383/1429/1686/1748/1824).
- **Tag publication**: a static-only tag is visible after `put_static_layer`
  (snapshot non-empty; `tags()` includes it).
- **Union/shadow/symlink with embedded**: embedded shadows disk; mem shadows
  embedded; whiteout hides embedded; bake-time symlink reject.
- **Observer**: invoked once per op, **after** the `mounts` guard is dropped;
  correct path/bytes/errno/latency incl. fh-based `Read`/`Write`/`Release`;
  no deadlock with `add_mount`/`remove_mount`.
- **Local mount**: mount/serve/read; explicit unmount; stale-mount recovery;
  degrade path on no-`/dev/fuse`.
- **mstream integration**: `build.rs` supplies `MSTREAM_SKILLS_DIR`;
  `rerun-if-changed` rebuilds on skill edits; fixed-epoch mtime reproducible;
  no `--mount-skill` degrades without creating a mount; mountpoints that cover
  the daemon cwd are refused; foreground signal exits clean up skills mount
  resources, daemon-owned backing dirs, and socket files.
- **CI**: needs `libfuse-dev`/`libfuse3-dev` + `pkg-config`, and `/dev/fuse` for
  the mount tests (§8/§9).

## 6. Open / resolved decisions
- ✅ **Mount failure** → degrade (FR-7).
- ✅ **Mountpoint** → explicit `mstream daemon start --mount-skill <DIR>`; no
  implicit cwd or XDG runtime default; `<DIR>` is rejected when it is the daemon
  working directory or one of its parents.
- ✅ **Embedded symlinks** → bake-time reject (FR-1a).
- ✅ **mtime** → fixed epoch default; build-timestamp opt-in only (FR-8).
- ✅ **Attr model** → separate static-only attr type (no widening of `OverlayAttrs`).
- ✅ **include_dir gating** → always-on light dep; `Static` variant unconditional.
- ◻ **Default mode bits** — proposing `0o444` files / `0o555` dirs.
- ◻ **Copy-up (future).** vmm-vz has no copy-up: `do_write` (1344-1398) resolves
  the *winning* layer and writes back into it (mutate-in-place). With a `Static`
  lower layer the write now returns `EROFS` (FR-3); real overlayfs COW (copy up
  into a writable layer on first write) is a separate `do_write`/`do_setattr`
  change, out of v1 scope.

## 7. Alternatives considered (appendix)
- **A1 — Seed a Mem layer with `Bytes::from_static`.** No engine change, but the
  layer is mutable, requires an O(files) seed walk, and lacks RO-by-type.
  *Rejected:* and on vmm-vz it is strictly worse — a write would mutate the
  in-memory original in place (`do_write` 1383), silently corrupting the baked
  bytes; the `Static` type makes that an `EROFS`.
- **A2 — Manifest/glob tooling with remapping.** *Deferred:* v1 is whole-dir.
- **A3 — Separate crate.** *Rejected:* a feature split (`local-mount`/`fuser-client`)
  keeps the engine in `motlie-vfs` without dragging vsock into consumers.

## 8. Logistics (re-anchored; for PLAN)
On `feature/vmm-vz` the main-branch logistics are already done: **`libs/vfs` (root
Cargo.toml:21) and `bins/mstream` (line 35) are workspace members, and `fuser` is
in `Cargo.lock`.** Remaining deltas:
- **Feature split (must-fix #8).** Today `client = ["dep:fuser", "vsock"]` and the
  mount module is gated `#[cfg(all(feature = "client", target_os = "linux"))]`
  (`client/mod.rs:12`); `fuser` is a `cfg(target_os="linux")` optional dep,
  `default-features = false`, v0.15. Introduce:
  - `fuser-client = ["dep:fuser"]` — the `FuseClient` + mount module, no vsock.
  - `client = ["fuser-client", "vsock"]` — guest path, unchanged behavior.
  - `local-mount = ["fuser-client"]` — `client::local`, Linux-only, no `AllowOther`.
  Re-gate `client::fuse`/`FuseClient` on `fuser-client` (not `client`) so mstream
  gets a local mount **without** `tokio-vsock`. `client::guest::mount_all`
  (`guest.rs:94`, gated `client+vsock+linux`) is untouched.
- **`include_dir`** added as a light **always-on** dep (no `embed` feature; type
  unconditional) — keeps `LayerSource::Static` from forcing cfg'd match arms.
- **libfuse build dep.** Enabling `fuser` links libfuse via `pkg-config` at build
  time. Once mstream depends on `motlie-vfs` with `local-mount`, the **workspace
  build requires `libfuse-dev`/`libfuse3-dev` + `pkg-config`** on every host/CI
  runner — document the dev-container/CI image change. `fuser`'s
  `default-features = false` must be preserved so guest cross-compiles
  (`mbuild`) don't need a target libfuse sysroot.
- **mstream** gains `motlie-vfs` (`local-mount`) + `include_dir` deps, a
  `skills.rs`, a build.rs `MSTREAM_SKILLS_DIR` + `rerun-if-changed` walk, an
  explicit `daemon start --mount-skill <DIR>` option, cwd-covering mountpoint
  rejection, SIGINT/SIGTERM handling, socket cleanup, daemon-owned backing-dir
  cleanup, and an unmount in its graceful-stop path.

## 9. Platform matrix & degrade contract
| Host | Local FUSE mount | Behavior |
|------|------------------|----------|
| Any host, `daemon start` without `--mount-skill` | no | **degrade**: warn, run unmounted |
| Linux x86_64 / aarch64 + `/dev/fuse` + `fusermount3` + `--mount-skill <DIR>` | yes | mount + serve + log |
| Linux with `--mount-skill <DIR>` but without `/dev/fuse`/perms (e.g. locked-down container) | no | **degrade**: warn, run unmounted |
| Linux with `--mount-skill <DIR>` where `<DIR>` is the daemon cwd or its parent | no | **degrade**: warn, run unmounted |
| macOS host with `--mount-skill <DIR>` (vmm-vz vz branch) | no (mount gated Linux-only, `client/mod.rs:12`) | **degrade**: warn, run unmounted |

**Degrade-vs-hard-fail contract (FR-7):** missing `--mount-skill` and mount failure
are **non-fatal** — the daemon logs a `WARN` (`"skills FUSE unavailable: no
--mount-skill specified; continuing without mounting skills"` or `"skills FUSE
unavailable: {reason}; continuing without mount"`) and runs normally; skills
simply aren't served over FUSE. `--mount-skill` also degrades when the requested
mountpoint is the daemon working directory or one of its parents, because mounting
there can leave the caller inside a disconnected FUSE endpoint after daemon exit.
This preserves mstream's run-anywhere property and makes FUSE an optional
enhancement. On graceful stop, `daemon stop`, SIGINT, and SIGTERM, mstream
unmounts the skills FUSE mount, removes its unique temporary backing directory,
removes the socket file, and removes the mountpoint only when mstream created it
and it is empty. The mount-available state should be surfaced (status/health) so
operators can tell.
