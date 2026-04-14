# motlie-vfs: Layered Guest Filesystem Composition

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Clarify the Apple Vz parallel track for VFS: move the `v1.05` guest-image proving step under `libs/vmm/examples/`, keep `v1.15` as the managed guestfs proof, and sequence cleanup plus `#134` after those proofs |
| 2026-04-13 | @codex-vz | Expand the cross-backend Apple Vz planning track: add `v1.05` as the first image/build proving step before `v1.15` guestfs, then sequence cleanup and `#134` after those proofs |
| 2026-04-13 | @codex-vz | Add the cross-backend Apple Vz planning track via `DESIGN_XBACKENDS.md` / `PLAN_XBACKENDS.md`: preserve managed `motlie-vfs` semantics under CH and Vz, require userspace-only / no-persistent-host-change behavior, sequence a `v1.15` Vz guestfs PoC before cleanup, and keep `#134` as the separate policy-engine phase |
| 2026-04-06 | @codex | Record the harness ownership handoff: `v1` and `v1.1` remain VFS-owned, while the `v1.2+` example / validation harness moves to `motlie-vnet` with `libs/vnet/docs/{DESIGN,PLAN}.md` as the networking source of truth |
| 2026-04-05 | @codex | Extend the v1 wire protocol and semantics with `Access`, xattrs, and byte-range locks; document handle-based fsync/rename behavior and the remaining lock-limitations for the compatibility pass |
| 2026-04-02 | @codex | Split the host/guest example boundary so `examples/v1/repl_host.rs` remains the stable v1 single-guest harness, `examples/v1.1/repl_host.rs` carries the v1.1 multi-guest control-plane extensions, and the guest mounters now live under `bins/v1/` and `bins/v1.1/` |
| 2026-04-02 | @codex | Sync `FsOp::Mkdir` wire protocol with `uid`/`gid` and clarify that explicit overlay-managed `mkdir` uses caller ownership while implicit parent dirs created by `put()` still inherit defaults |
| 2026-03-31 | @claude | Update wire protocol: `FsOp::Create` now carries `uid`/`gid`, add `FsResult::Created` with `fh` for atomic create+open, add `FsResult::Opened` (PR #123 review) |
| 2026-03-28 | @codex-pm | Resolve PR #117 round-3 follow-ups: complete `FsOpKind` coverage for event emission and align the docs with the final phase numbering and overlay-behavior test expectations |
| 2026-03-28 | @codex-pm | Resolve implementer questions from PR #117: rename event op type, specify v1 symlink/file-handle/runner/whiteout/write-buffer behavior, and make inspection views metadata-only |
| 2026-03-28 | @codex-pm | Address PR #117 review feedback: remove stale v1 CLI references, clarify `apply_batch` tag scoping, tighten synthetic-parent and rename semantics, and fix wording around immutable lower layers |
| 2026-03-28 | @codex-pm | Reframe the project headline around layered guest filesystem composition instead of transport-agnostic plumbing |
| 2026-03-28 | @codex-pm | Make the bootstrap/guest-agent boundary explicit: bootstrap and binary delivery stay VMM-owned, while the reusable guest mounter is built on public guest-side APIs in `client/` and `vsock/` |
| 2026-03-28 | @codex-pm | Treat the guest-side mounter as a real v1 binary: move it from `examples/` into `bins/` and make image-build order explicit |
| 2026-03-28 | @codex-pm | Resolve v1 crate-structure ambiguity: `client/fuse.rs` owns the guest-side `fuser` bridge, `client/guest.rs` owns guest mount orchestration, `vsock/` owns transport glue only, and `libs/vfs/examples/` now contains the host REPL/demo harness while the guest mounter lives in `bins/` |
| 2026-03-28 | @codex-pm | Specify v1 memfs concurrency contract: batch-first API, atomic memfs snapshot publish per mount tag, non-memfs layers keep native semantics; add rejected crate evaluations for `vfs` and Theseus `memfs` |
| 2026-03-28 | @codex-pm | Converge on a generic ordered layer-stack model: v1 = N memfs layers + 1 disk base per mount, with shared named memfs layers spanning multiple mount tags via `(layer, tag, relative path)` keys |
| 2026-03-28 | @codex-pm | Add non-negotiable mount/backing abstraction guidance so v1 implementation cannot accidentally internalize a permanently disk-backed model |
| 2026-03-28 | @codex-pm | Simplify roadmap tooling: v1 uses `rustyline`, v1.5 extends the embedded admin path with script/config batch ingestion, and v2 adds the first gRPC/RPC remote admin layer |
| 2026-03-28 | @codex-pm | Reorganize around roadmap: v1 core crate + examples, v1.5 embedded admin console + script/config ingestion, v2 external gRPC/RPC app layer |
| 2026-03-28 | @codex-pm | Expand v1 operational scope with explicit VM image, Cloud Hypervisor, host setup, and overlay CLI test procedures |
| 2026-03-28 | @codex-pm | Clarify v1 VM target: guest path is vsock-only, and Cloud Hypervisor is the fastest dev/test harness before a full VMM exists |
| 2026-03-28 | @codex-pm | Clarify frontend layering: core `MemOverlay` API first, v1/v1.5 remain embedded-admin oriented, HTTP/gRPC admin wrappers appear only in v2, and VMM in-process Tokio hosting is supported |
| 2026-03-28 | @codex-pm | Add explicit synthetic-entry uid/gid contract, mount-relative path contract, and shared-guest ownership rules for API and future control frontends |
| 2026-03-28 | @codex-pm | Clarify shared-overlay behavior for multi-user guest paths, including tag/path scoping, concurrency caveats, and v1 ownership limitations for synthetic entries |
| 2026-03-28 | @codex-pm | Clarify compatibility with guest boot-image paths: partial overlay of existing dirs/files, app-specific runtime dirs, squashfs deletion via whiteout only |
| 2026-03-28 | @codex-pm | Expand product requirements and guest-runtime use cases: operator-controlled mount layout, partial memfs injection, app-specific runtime directories |
| 2026-03-28 | @codex-pm | Clarify VMM storage model: same squashfs + ext4 stacking over virtio-block, independent of raw vs qcow2 host image backing |
| 2026-03-28 | @claude | v1 mount policy (direct_io), remove stale "cached" wording from inode model |
| 2026-03-28 | @claude | v1 no-caching principle: zero TTL everywhere, no semantic caches, correctness-first, perf phase deferred |
| 2026-03-28 | @claude | Spec-complete: memfs-aware inode model, cache/invalidation strategy, generation semantics, control-socket wire format |
| 2026-03-28 | @claude | Fix stale resolve() examples to use OverlayEntryKind; show policy-filtered vs unfiltered readdir for .ssh |
| 2026-03-28 | @claude | Converge memfs model: shared inode namespace, internal tree vs inspection views, tighten wording per review |
| 2026-03-28 | @claude | Reframe overlay as file-driven memfs layer: whiteout/tombstone semantics, transparent cross-boundary rename, OverlayEntryKind enum |
| 2026-03-27 | @claude | Address PR #115 review: synthetic directories, policy-filtered readdir, mutation semantics table |
| 2026-03-27 | @claude | Initial DESIGN: composable architecture (core + vsock + rpc), file-level in-memory overlay with layered injection, pluggable control frontends (Unix socket / HTTP / in-process), VMM guest integration with SSH overlay example, alternatives analysis |

## Problem Statement

motlie-vmm's architecture routes all guest filesystem I/O through a host-side vsock FS server
(see `motlie-vmm` design doc, sections 10-12). The current design hardwires the FUSE client to
vsock inside a VM and the FS server to the VMM daemon process. This coupling prevents three
valuable deployment models:

1. **Linux host mounts (no VM)** -- sandboxed directory access with audit logging, useful for
   container-less isolation or AI agent file access control.
2. **macOS FUSE mounts** -- developers on Macs mounting directories served by a local or remote
   FS server, with the same audit and policy capabilities.
3. **Reuse as a library** -- other Rust programs embedding the server or client without depending
   on the full VMM binary or its vsock assumptions.

The FS server and FUSE client need to be extracted into a standalone library with pluggable
transports and cross-platform FUSE support.

Parallel Apple Vz track:

- `libs/vfs/docs/DESIGN_XBACKENDS.md`
- `libs/vfs/docs/PLAN_XBACKENDS.md`

That track keeps the same product constraints:

- userspace-only host services
- no persistent host configuration drift
- no persistent host traces other than caller-selected backing directories
- runtime-only connection state that disappears with the host process

The current execution order for Apple Vz support is:

1. Vz image/build PoC in `libs/vmm/examples/v1.05`
2. Vz guestfs PoC in `libs/vfs/examples/v1.15` and `libs/vfs/vz`
3. `motlie-vfs` transport cleanup and cross-backend boundary refactor
4. `#134` policy engine implementation on the clarified core
5. later `libs/vmm` `guestfs_vz.rs` / `backend::vz` integration

That extraction is not only about transport reuse. It is also about supporting a more
expressive **guest runtime filesystem policy** than the current "static pass-through mount"
model. In the target product, an operator must be able to choose at runtime:

- which host directories are exposed into a guest as ordinary pass-through mounts
- which guest-visible paths receive additional in-memory files or directories from the memfs layer
- which paths remain pure disk pass-through
- which applications or guest sessions receive specific injected directories such as `/.ssh`,
  `/.claude`, `/.codex`, or `/.gh`

The design therefore needs to support a mixed model where:

- a guest path may already exist on disk and still receive selective in-memory file injection
- a guest path may not exist on disk and be created entirely by the memfs layer
- injected files shadow same-path disk files when present
- non-overlaid files in the same mounted tree continue to pass through to disk
- overlay contents can be added or removed while the guest is running
- guest applications observe all of this through normal filesystem operations, without needing
  application-specific integration

The goal is not merely "remote filesystem access." The goal is **runtime composition of a guest
filesystem view** from:

- a boot/storage layer provided by the VMM (`squashfs` + `ext4` overlay for the guest root)
- pass-through host-backed mounts such as `/workspace` or `/root`
- dynamic in-memory per-path injections for credentials, tool config, or policy-controlled files

## Architectural Layering and Roadmap

This DESIGN is organized around three roadmap slices with intentionally different scope
boundaries.

### v1: `libs/vfs` Core Crate + Proof of Concept Examples

v1 lives inside `libs/vfs` and targets the fastest proof of concept:

- core Rust API:
  - `FsServer`
  - `MemOverlay`
  - inode/path/policy/event handling
- guest transport path for VMs:
  - vsock only
- guest mount path:
  - `FuseClient` in `client/fuse.rs`
  - backed by a vsock transport adapter from `vsock/`
- proof-of-concept host/guest harness:
  - `repl_host` example: stable v1 single-guest host REPL
  - Cloud Hypervisor guest with squashfs+ext4 stacked root
  - guest runs `sshd` and `motlie-vfs-guest`
  - guest mounts host-backed trees over vsock
- repo location for the proof-of-concept harness:
  - `libs/vfs/examples/v1/repl_host.rs` (v1 host side)
  - `libs/vfs/examples/v1.1/repl_host.rs` (v1.1 multi-guest host side)
  - `libs/vfs/examples/v1/` and `libs/vfs/examples/v1.1/` (guest image build + CH launch scripts)
- harness lineage handoff:
  - `v1` and `v1.1` remain owned here under `libs/vfs/examples/`
  - beginning with `v1.2`, the composed host / networking harness is owned by
    `motlie-vnet` under `libs/vnet/examples/v1.2/`
  - networking design / follow-up work for `v1.2+` therefore lives in
    `libs/vnet/docs/{DESIGN,PLAN}.md`
- REPL/tooling choice:
  - `rustyline` for interactive mode
  - stdin pipe support for scripted/agent-driven setups

The v1 crate is library-first, but it is allowed to include example binaries and supporting
README/documentation under `libs/vfs/examples/` to prove the architecture quickly. The host
admin interface for v1 lives in the `repl_host` example binary; v1.1 carries its own host
harness in `examples/v1.1/repl_host.rs` so later roadmap slices do not keep changing the
original v1 example in place. There is no separate crate module for admin.

**Host admin input modes:**

The `repl_host` example auto-detects how stdin is connected:

- **Interactive** (stdin is a TTY): rustyline REPL with line editing, history, ^C handling.
  Server runs until `quit` or Ctrl-D.
- **Pipe then interactive** (`cat script.vfs - | repl_host ...`): executes piped commands
  line by line, then reopens `/dev/tty` for interactive REPL. Server keeps running throughout.
- **Pure pipe** (`cat script.vfs | repl_host ...`): executes piped commands, then server
  keeps running and serving guest connections until SIGTERM/SIGINT. Use this for
  automated/agent-driven setups where no human operator is present.

In all modes, `quit` in the input stream shuts down the server immediately. The server never
exits on pipe EOF — guest filesystem connections remain active.

Script files are plain text, one command per line. Lines starting with `#` are comments.
See `libs/vfs/examples/v1/setup-alice.sh.vfs` for an example.

### v1.5: Embedded Admin Console + Script / Config Ingestion

v1.5 remains local and embedded. It does **not** add a remote admin interface.

- stays in-process with the host/server embedding
- continues to use the `MemOverlay` Rust API directly
- improves operator ergonomics through:
  - a stronger embedded terminal/admin console
  - script/config-driven batch injection and reload
  - repeatable startup/reload flows for injected files
- keeps remote admin out of scope until v2

### v2: External gRPC / RPC Application Layer

v2 is a broader remote-control and integration layer.

- lives **outside** `libs/vfs`
- may provide:
  - gRPC
  - custom RPC
  - richer remote admin/API services
- includes a microservice API that can construct memfs file trees directly,
  without relying on local disk-backed trees of the kind used in v1 and v1.5
- is not part of the core crate boundary

### Roadmap Gaps and Forward-Compatibility Requirements

The roadmap is feasible, but two areas need to be called out explicitly so v1 implementation
choices do not block later phases.

**Gap 1: v1.5 console/script parity**

v1.5 introduces two local operator paths over the same core crate:

- an interactive embedded terminal/admin console
- a script/config-driven batch ingestion path

The risk is semantic drift: one path may expose slightly different command behavior,
defaults, or error handling than the other.

Required mitigation:

- treat the mutation surface as a single abstract API first, then bind both local paths to it
- define command semantics once in terms of core `MemOverlay` operations
- require parity tests so both paths produce the same:
  - success behavior
  - error behavior
  - path normalization
  - ownership/mode handling
  - listing/output semantics

Design implication for v1:

- keep core overlay operations small, explicit, and backend-agnostic
- avoid baking REPL-only or script-only assumptions into `MemOverlay`

**Gap 2: v2 diskless memfs-tree construction**

v2 wants to construct memfs file trees without the local disk-backed mount assumptions used
in v1 and v1.5. The current v1 design is still centered on:

- `tag -> host_path` mount registration
- fallback to `std::fs` when no overlay entry exists
- ownership inheritance from the nearest existing parent or mount root in a mounted tree

Those choices are acceptable for v1, but they are potential blockers if implemented too rigidly.

Required mitigation:

- keep the core notion of a mount abstract enough that a future mount may have:
  - disk backing
  - synthetic backing
  - mixed backing
- avoid hard-coding assumptions that every mount must have a real `host_path`
- keep overlay node representation independent from direct `std::fs` calls wherever possible
- isolate disk access behind server-core mount/backing logic rather than spreading `std::fs`
  assumptions through overlay code
- make ownership/default-attr logic pluggable enough that future synthetic-root mounts can
  choose defaults without requiring a real disk parent

Recommended future extension point:

- v1 should conceptually preserve room for a mount backing enum such as:
  - `Disk { host_path }`
  - `SyntheticRoot`
  - `Hybrid`

This enum does **not** need to be implemented in v1, but v1 code should avoid making it
impossible.

Concrete v1 guidance:

- acceptable in v1:
  - `FsServerBuilder::mount(tag, host_path, read_only)`
  - one disk-backed base layer at the bottom of each mount stack
  - zero or more memfs layers stacked above it
  - ownership inheritance from nearest real parent or mount root
- avoid in v1:
  - assuming `host_path` exists in every internal mount representation forever
  - baking “no overlay hit means disk must exist” into public contracts instead of “keep walking the layer stack”
  - coupling synthetic-node creation to host-path existence checks more than necessary

### Non-Negotiable Implementation Guardrails

To keep implementers from internalizing “every mount is disk-backed,” the v1 implementation
must follow these rules:

1. Treat **mount** and **backing** as separate concepts internally.
   A mount tag is the stable namespace visible to clients. Its backing is an implementation
   detail that may be disk-backed in v1 and non-disk-backed later.

2. Do not model internal mount state as only:
   `tag -> host_path`.
   Even if the public builder takes `host_path` in v1, the internal server representation
   must leave room for a future backing kind.

3. Route all backing access through a narrow boundary.
   `std::fs` calls must be concentrated behind mount/backing operations, not spread through
   overlay, inode, policy, or FUSE translation code.

4. Do not define public semantics as “overlay or disk” only.
   Public semantics should be phrased as:
   - overlay hit
   - fallback to mount backing
   where v1 backing happens to be disk-backed.

5. Synthetic-node defaults must not require a real disk parent.
   Ownership/mode defaulting may consult the nearest existing parent in v1, but the code path
   must still admit a future synthetic-root default policy.

6. Inode allocation must be backing-agnostic.
   The inode table must not assume that every non-overlay inode corresponds to a host path on disk.

7. Readdir/lookup/getattr merge logic must be phrased against overlay + backing.
   The fact that v1 backing uses `std::fs` is an implementation detail, not the architectural model.

8. Tests must enforce the abstraction.
   Add explicit design guardrails and review checkpoints that reject implementations which leak
   raw `host_path` assumptions into overlay internals or command semantics.

### Ordered Layer-Stack Model

The authoritative model for mount resolution is an ordered layer stack.

For each guest mount point there is one mount tag, and for each mount tag there is one
ordered stack of filesystem layers.

In v1:

- each mount stack contains:
  - zero or more writable/readable memfs layers
  - exactly one base disk-backed layer at the bottom
- the final guest-visible mount point is the result of resolving operations through that stack

For a guest with `M` mount points, the system has `M` independent stacks.

#### Layer Contract

Every layer in the stack must support the same logical contract, even if its implementation
differs:

- `lookup`
- `getattr`
- `read`
- `readdir`
- `create`
- `write`
- `mkdir`
- `unlink`
- `rmdir`
- `rename`
- `setattr`

The layer contract is semantic, not necessarily a 1:1 public Rust trait in v1. What matters is
that stack composition logic is written generically against “the next lower layer,” not against
special knowledge of disk paths.

Each layer may differ in capability or implementation detail:

- memfs layer:
  - stores `Content`, `SyntheticDir`, and `Whiteout`
  - may synthesize parent directories
  - typically acts as the writable top layer in v1
- disk-backed layer:
  - resolves against the host filesystem for v1
  - acts as the base layer of the stack
- future synthetic-root or hybrid layer:
  - may provide a non-disk base for a stack in v2+

#### Generic Resolution Semantics

The stack semantics are generic and must not be described as special-casing “overlay vs disk.”

`lookup` / `getattr` / `read`:

- walk from highest-priority layer to lowest-priority layer
- the first `Content` or `SyntheticDir` match wins
- the first `Whiteout` match wins as hidden / `ENOENT`
- if no layer resolves the path, return `ENOENT`

`readdir`:

- start from the lowest-priority layer’s directory entries
- apply each higher-priority layer in order
- `Content` / `SyntheticDir` entries add or replace by name
- `Whiteout` removes the name from the merged set
- after merge, policy may filter the effective entries

`unlink`:

- if the visible target exists only in the top writable memfs layer, remove it there
- if the visible target shadows lower-layer content, create or retain a `Whiteout` in the
  writable memfs layer so the lower entry remains hidden

`rename`:

- is defined as stack-level behavior, not as a disk-specific escape hatch
- cross-layer rename may require copy-up / materialize / remove behavior depending on which
  layers currently provide source and target

#### Shared Named Memfs Layers Across Mount Stacks

Memfs layers are named and may be shared across multiple mount tags.

The identity of a memfs entry is:

- `layer name`
- `mount tag`
- mount-relative path

This means one named layer may carry entries for multiple mount stacks at once without path
collision.

Example:

- layer `claude-shared`
- tag `bob-project1`, path `/CLAUDE.md`
- tag `alice-project2-feature1`, path `/CLAUDE.md`

These are two distinct entries in the same named memfs layer because they differ by tag.
The effective guest-visible absolute path is only known after the operator binds tags to guest
mount points.

Implications:

- tags and mount points define the final guest filesystem placement
- memfs entries are specified relative to a tag, not as global guest-absolute paths
- a single named memfs layer can inject the same relative filename into multiple mount stacks
- operator/VMM code is responsible for coordinating tag naming, mount placement, and any desired
  uid/gid/mode defaults for those injected entries

### Required Internal Shape for v1

The guardrails above are not only philosophy. They imply a concrete implementation shape for v1.

Recommended internal decomposition:

- `FsServer`
  - owns tag registry
  - owns per-tag inode table
  - owns per-tag `MemOverlay`
  - delegates fallback operations to a mount-backing boundary
- `MemOverlay`
  - never calls `std::fs` directly
  - never stores raw host paths as part of overlay node identity
  - operates only on mount-relative paths plus explicit attrs/generation rules
- mount-backing boundary
  - owns disk fallback logic in v1
  - is the only layer allowed to translate from mount-relative paths to host filesystem paths
  - is the only layer allowed to issue `std::fs` operations

Concretely, implementers should structure v1 so it could evolve toward an internal shape like:

```rust
struct MountState {
    tag: String,
    read_only: bool,
    backing: MountBacking,
    overlay: MemOverlay,
    inode_table: InodeTable,
}

enum MountBacking {
    Disk(DiskBacking),
    // Reserved future directions:
    // SyntheticRoot(SyntheticBacking),
    // Hybrid(HybridBacking),
}
```

This exact enum is not required in v1, but the internal design must preserve that separation.

Required consequences:

- overlay APIs take `tag` + mount-relative path, never a raw `host_path`
- inode entries may reference backing-derived data, but inode allocation rules cannot require
  every entry to map to a disk path
- synthetic attr/default logic must accept a future policy source that is not “inspect parent
  on disk”
- command/CLI/RPC layers must describe fallback as “mount backing,” not “host disk,” unless a
  v1-only disk-specific behavior is being described explicitly

### Required Review Questions for v1 Implementation

Before considering the v1 implementation complete, the reviewer should be able to answer
“yes” to all of the following:

1. Can `MemOverlay` be read in isolation without seeing `std::fs`, `PathBuf` host roots, or
   host-path concatenation logic?
2. Is all disk access concentrated in one backing-oriented layer instead of spread across
   overlay, inode, FUSE, and command handling?
3. If a future `SyntheticRoot` mount backing were added, would the overlay and inode code need
   extension rather than redesign?
4. Do public semantics and operator-facing commands remain valid if “disk-backed fallback” is
   replaced later by a different mount backing?
5. Do tests include at least one explicit assertion or code-review checkpoint preventing
   `host_path` assumptions from leaking into overlay internals?

### Up-Front Layering Rule

The architectural layering for the roadmap is:

1. `libs/vfs` core crate
2. example proof-of-concept programs under `libs/vfs/examples`
3. external applications built on top of the crate

Applied to this roadmap:

- v1 = layers `1` + `2`
- v1.5 = stronger embedded admin console + script/config ingestion
- v2 = broader external remote-control layer (gRPC/RPC)

## Product Requirements and Primary Use Cases

This section describes the concrete operator and guest behavior the library must enable.

### Product Requirement: Operator-Controlled Guest Filesystem Layout

The operator or VMM daemon controls which host-backed filesystems are mounted into the guest and
where they appear. `motlie-vfs` does not hardcode guest mount points. It provides the server,
client, and overlay semantics needed to make those choices dynamic.

Important scope boundary: `motlie-vfs` only controls paths that are inside a tree mounted through
its FUSE/vsock/RPC client. It does **not** mutate arbitrary guest rootfs paths outside those
mounted trees.

Examples:

- mount a host workspace directory at `/workspace`
- mount a host home directory at `/root`
- choose not to expose a host directory at all, but inject selected files into a guest-visible path

### Product Requirement: Partial Overlay of Existing Guest Paths

The design must support the case where a guest-visible directory already exists via disk-backed
pass-through, but only some files in that directory are injected from the memfs layer.

Examples:

- `/root/.ssh/` exists on disk, but `config` and `id_ed25519` are injected from memfs
- `/root/.claude/` exists on disk, but selected auth/config files are injected dynamically
- `/root/.codex/` exists on disk, but a session-specific token or config file is overlaid

Required behavior:

- injected files shadow same-path disk files
- disk-only files in the same directory continue to pass through unless a policy filters them
- applications see a single coherent directory tree

This requirement explicitly includes paths that already exist in the **guest-visible mounted
subtree**, even if those paths ultimately originate from the guest boot image or from an existing
host-backed pass-through mount. For example, if the operator mounts `/home/alice` or `/root`
through `motlie-vfs`, and that mounted tree already contains `.ssh`, `.claude`, `.codex`, or
`usr/local/bin`, the overlay may still patch in selected files inside those directories.

Examples:

- within a mounted `/home/alice` tree, `.ssh/config` or `.claude/settings.json` already exists on
  disk, but selected files are replaced or supplemented from memfs
- within a mounted tree that exposes `usr/local/bin`, `gh` already exists from the lower image, but
  supporting config or auth files are injected separately at runtime
- within a mounted home directory, `.gh/config.yml` is injected even if the config file did not
  previously exist

The important compatibility rule is: **directory existence and file existence are independent
questions inside a mounted subtree**. A directory may already exist while some files in it are
absent and later injected, or both the directory and files may already exist and be selectively
shadowed.

### Product Requirement: Fully Synthetic Guest Paths

The design must also support the case where the guest-visible directory does not exist on disk at
all. In that case, `put()` of a file under that path creates the parent hierarchy in the memfs
layer automatically.

Examples:

- within a mounted `/home/alice` tree, `.ssh/` does not exist on disk, but `.ssh/id_ed25519` is
  injected at runtime
- within a mounted home tree, `.gh/` does not exist on disk, but `.gh/config.yml` is injected only
  for guests allowed to run `gh`
- within a mounted home tree, `.claude/` does not exist on disk, but `.claude/settings.json` and
  auth material are injected

Required behavior:

- missing parent directories are synthesized recursively in memory
- those synthetic directories behave like ordinary directories for `lookup`, `getattr`, and `readdir`
- removing the last child can remove the synthetic hierarchy according to the overlay rules

### Product Requirement: Application-Specific Runtime Injection

Different guests, sessions, or policy profiles may receive different injected directories and
files at runtime.

Examples:

- guests allowed to run `gh` receive a `/.gh/` tree
- guests allowed to run Claude tooling receive a `/.claude/` tree
- guests allowed to run Codex tooling receive a `/.codex/` tree
- guests with SSH access needs receive a `/.ssh/` tree

This is a runtime policy decision above `motlie-vfs`. The library requirement is that these
directories can be inserted, updated, or removed dynamically without changing the mounted tree
shape on disk.

Those directories may be layered on top of, within a subtree mounted through `motlie-vfs`:

- paths already present in the guest-visible lower image
- paths already present in a host-backed pass-through mount
- paths that do not exist yet and must be synthesized entirely in the memfs layer

### Product Requirement: Shared Mounted Trees, Shared Layers, and Multi-User Paths

One `MemOverlay` instance may manage entries for multiple guest-visible subpaths at once,
including user-specific subtrees inside the same mounted tree and the same named memfs layer
applied across multiple mount tags.

Supported examples:

- if the operator mounts `/home` through `motlie-vfs` with tag `home`, a single overlay
  layer may contain both `/alice/.claude/skills/...` and `/bob/.claude/skills/...`
- if the operator instead mounts `/home/alice` and `/home/bob` as separate tags, the same
  `MemOverlay` instance may still hold entries for both, but they remain isolated by tag
  because every overlay key is `(tag, path)`
- if the operator mounts `/home/bob/project1` as tag `bob-project1` and
  `/home/alice/project2/feature1` as tag `alice-project2-feature1`, one shared layer such as
  `claude-shared` may inject `/CLAUDE.md` for both tags as two entries:
  - `(claude-shared, bob-project1, /CLAUDE.md)`
  - `(claude-shared, alice-project2-feature1, /CLAUDE.md)`

Implications:

- path separation is supported because overlay entries are keyed by **mount tag + mount-relative path**
- layer sharing is supported because a single named layer may contain entries for many tags
- tag separation is stronger than path separation; different tags have independent inode tables,
  lookup spaces, and overlay resolution
- if two guests are bound to the **same tag**, they observe the same effective overlay state for
  that mounted tree
- if two guests require isolation, the recommended v1 design is separate mount tags (or separate
  server instances), not sharing one tag and relying on path naming alone

Examples:

- shared-tag case: mount guest `/home` from tag `home`; inject `/alice/.claude/skills/tool.md`
  and `/bob/.claude/skills/tool.md` into the same overlay namespace
- isolated-tag case: mount guest A `/home/alice` from tag `alice-home` and guest B `/home/bob`
  from tag `bob-home`; inject each subtree independently under its own tag

Concurrency and mutation caveats:

- concurrent mutations to different `(tag, path)` pairs are independent
- concurrent mutations to the same `(tag, path)` require normal in-process synchronization and
  are logically last-writer-wins unless a higher-level control plane adds stronger guarantees
- v1 defines atomic multi-path memfs updates **per mount tag** through the batch API
- v1 does **not** define atomic multi-tag updates spanning several mount tags at once

Ownership and permission caveats:

- disk-backed files and directories retain the metadata and permission checks of the underlying
  host filesystem as surfaced through the mounted tree
- overlay-created `Content` and `SyntheticDir` entries in v1 expose explicit synthetic metadata:
  mode bits plus numeric `uid`/`gid`
- there is still no per-guest identity virtualization below a shared `(tag, path)` entry:
  one effective overlay node has one effective `uid` and `gid`
- therefore, using one shared overlay namespace as a security boundary between different Linux
  users remains out of scope for v1

Explicit v1 scope statement:

- supported: managing both `/home/alice/...` and `/home/bob/...` inside one mounted tree when
  path-based separation is operationally acceptable
- supported: isolating Alice and Bob with separate mount tags while still using one `FsServer`
  and one `MemOverlay` instance
- supported: coordinating synthetic entry ownership with operator-provisioned guest users by
  setting numeric `uid`/`gid` for injected entries
- out of scope: strong per-user isolation for synthetic overlay entries that share the same
  `(tag, path)` while needing different ownership views per guest

Important ownership implication:

- if the **same overlay entry** `(tag, path)` is intended to be shared across different guest VMs,
  that one entry can only expose one numeric `uid` and `gid`
- in that specific shared-entry case, the guests must agree on the expected numeric owner/group
  for that path, or the operator must avoid sharing the same tag/path entry
- if Alice and Bob run in different guests and need different ownership values, the supported v1
  design is separate tags and separate injected entries, not guest-specific ownership views of the
  same entry

### Product Requirement: Dynamic Runtime Mutation

Overlay entries must be mutable while the guest is running.

Required operations:

- add files
- update files
- create whiteouts to hide disk files
- remove injected files
- remove layers
- inspect effective overlay state

In v1, runtime mutation is exercised through direct Rust calls and example programs. Later
operator workflows, such as the v1.5 embedded console/script path or a future v2 RPC admin
surface, can be built over the same `MemOverlay` API.

Roadmap note:

- v1 uses direct in-process mutation through the core API
- v1.5 strengthens the embedded admin workflow with script/config-driven batch loading
- command semantics across those operator paths must remain identical; path-specific behavior is
  a design bug, not a product feature

v1 concurrency and atomicity contract:

- the mutation API is batch-first
- a single-file mutation is defined as a batch of size 1
- each committed batch is atomically published for one mount tag
- guest reads and directory traversals must observe one stable memfs snapshot for the duration
  of a single filesystem request
- readers must never observe:
  - partially created synthetic parents from an in-flight batch
  - some but not all files from one committed batch
  - a `readdir` merge built from multiple memfs snapshots for the same request

Explicit v1 scope boundary:

- atomicity is guaranteed only for memfs-layer publication
- non-memfs layers keep their native concurrency and transaction semantics
- in v1, the disk-backed base layer is **not** made transactional
- multi-tag atomic commit is out of scope for v1

Recommended implementation direction for v1:

- per mount tag, writers are serialized
- each committed batch constructs a new memfs snapshot for that tag
- commit is one atomic swap of the active memfs snapshot
- read operations load one snapshot and use it for the whole request

Implications:

- if a batch injects `/.ssh/config`, `/.ssh/id_ed25519`, and `/.ssh/id_ed25519.pub` for one tag,
  readers see either the state before the batch or the state after the batch
- they must not see only one or two of those files from that batch
- lower-layer disk behavior remains whatever the host filesystem provides natively

Deletion semantics must remain compatible with lower immutable layers when they exist:

- if a file comes from an immutable lower layer, it is **not physically deleted**
- the supported delete/hide mechanism is a `Whiteout`/tombstone entry in the memfs overlay
- from guest applications, that file appears absent
- if the whiteout is later removed, the lower-layer file becomes visible again
- in v1 the `motlie-vfs` base layer is writable host-backed storage; whiteouts remain useful when
  the operator wants to hide a base-layer file without modifying the host tree

### Product Requirement: Transparency to Guest Applications

All of the above must be transparent to applications running inside the guest.

Applications should observe:

- ordinary files and directories
- ordinary rename/unlink/read/write behavior as defined by the overlay semantics
- no application-specific API for "overlay files" vs "disk files"

This is essential for editors, shells, `git`, `gh`, SSH tooling, Claude tooling, Codex tooling,
and similar user-space programs to work unchanged.

## Non-Goals

- **General-purpose network filesystem.** This is not an NFS or CIFS replacement. It serves
  known, pre-configured mount points with tag-based routing, not arbitrary network shares.
- **Distributed filesystem.** No replication, no consensus, no multi-server coordination.
- **POSIX completeness.** ACLs, `mmap`, and the broader long tail of filesystem operations
  outside the current protocol surface are still out of scope. v1 now includes basic extended
  attributes, `access`, and byte-range lock support because real coding tools probe them, but it
  still does not claim kernel-parity semantics across every edge case.
- **Windows support.**
- **Kernel-mode filesystem.** This is always userspace FUSE.
- **Binary distribution / CLI.** This is a library. Binaries that use it (the VMM daemon, a
  standalone mount tool) are separate crates.
- **Built-in admin services.** The library provides the core overlay API only. External admin
  frontends such as a Unix-domain control socket, HTTP, gRPC, or custom network control layers
  are wrappers above that API and are out of scope for v1.

## Functional Requirements

Not every FR below is a `v1` delivery requirement. Each FR is labeled by roadmap placement so
implementation planning does not over-claim current scope.

### FR-1: Cross-Platform FUSE Client
Roadmap placement: roadmap-wide. `v1` only requires the Linux guest FUSE path used by the
vsock/Cloud Hypervisor workflow. macOS FUSE-T support is a v2 roadmap target.

Mount a directory on the local machine backed by a remote (or local) FS server. Must work on:
- Linux (kernel 5.10+, libfuse3 / `/dev/fuse`)
- macOS (Apple Silicon, macOS 13+, via FUSE-T) — **v2 roadmap target**

The platform difference must be invisible to the caller -- a single `mount()` API.

### FR-2: Transport-Agnostic Protocol
Roadmap placement: `v2`. `v1` and `v1.5` only need the fixed vsock path.

The wire protocol between client and server must operate over any `AsyncRead + AsyncWrite`
stream. Concrete transports supported out of the box:

| Transport | Use case |
|-----------|----------|
| vsock | VM guest ↔ host (existing motlie-vmm model) |
| Unix socket | Same-machine mounts (Linux or macOS) |
| TCP + TLS | Cross-machine mounts (macOS dev → Linux server) |

Adding a new transport must not require changes to protocol, server, or client code.

### FR-3: Pluggable Wire Encoding
Roadmap placement: `v2`. `v1` and `v1.5` only need the fixed bincode-based vsock path.

The frame serialization format must be swappable. Default: bincode (fast, compact, Rust-native).
The protocol layer must be parameterized by encoding so that alternative formats (msgpack,
protobuf) can be substituted without changing the frame types or server/client logic.

### FR-4: Tag-Based Mount Routing
A single server instance serves multiple mount points for one guest VM, each identified by a
string tag (e.g. `workspace`, `cred-claude`). Each client connection binds to exactly one tag
via a handshake. In v1, the server maps each tag to a mount stack whose base layer is host-backed.

**Guest isolation model:** Each guest VM gets its own `FsServer` instance and its own vsock
socket. Tags identify mounted subtrees within that VM's server. For multiple guests, the host
runs separate `FsServer` instances with separate vsock sockets — one per VM. There is no
shared `FsServer` across VMs in v1.

### FR-5: Dynamic Mount Management
Mounts can be added to or removed from a running server. Adding a mount registers a new
tag → host path mapping; removing a mount deregisters it and invalidates outstanding inodes
for that tag.

### FR-6: Event Emission
Every filesystem operation that passes through the server can emit a structured event.
Events are delivered via a channel (`tokio::sync::broadcast` or similar). The library does
not persist or transport events -- that is the caller's responsibility (e.g. motlie-vmm
writes them to motlie-db).

Event emission must be optional (zero overhead when no subscriber) and must not block the
FS response path.

### FR-7: Policy Hooks
The server must support caller-provided policy functions that can intercept FS operations
before they execute. A policy function receives the operation, path, and mount tag, and
returns allow/deny. Use case: read-only credential enforcement, rate limiting credential
reads, blocking writes to specific paths.

### FR-8: Read-Only and Read-Write Mounts
Each mount tag is independently configured as read-only or read-write. Write operations
on a read-only mount return `EROFS` without reaching the host filesystem.

### FR-9: Composable Architecture
Roadmap placement: roadmap-wide.

The roadmap must support three composition modes from a single server core:

1. **Direct** -- in-process `handle_op()` calls, no serialization, no transport.
2. **vsock** -- fixed bincode over vsock, thinnest wire path, for VM use case.
3. **RPC** -- framed protocol with pluggable codec over any transport, for cross-platform mounts.

Roadmap split:

- `v1`: direct + vsock
- `v1.5`: v1 capabilities + embedded admin workflows
- `v2`: remote/framed protocol path

Each composite reuses the same `FsServer` core (inode table, host FS ops, events, policy).
The difference is what sits in front of `handle_op()`.

### FR-10: In-Memory Overlay with Layered Content Injection
The server must support ordered layered content injection without modifying the lower backing.
In v1, that means zero or more memfs layers stacked above one disk-backed base layer per mount.
Requirements:

- **Multiple named layers** with explicit priority ordering. Higher-priority layers shadow
  lower ones.
- **Per-mount stack model.** Each mount tag resolves through its own ordered stack. For `M`
  guest mount points there are `M` stacks.
- **Per-entry identity = `(layer, tag, path)`.** Each memfs entry targets a specific
  mount-relative path within a mount tag, allowing one named layer to hold entries for
  many mount tags at once.
- **Synthetic files and directories.** An overlay can inject files that do not exist on disk.
  These appear in readdir results alongside real files. Parent directories that don't exist
  on disk are created implicitly as synthetic directories (e.g., injecting `/.ssh/id_ed25519`
  implicitly creates a synthetic `/.ssh/` directory).
- **Generic stack semantics.** Lookup/read walks top-down; readdir merges bottom-up; whiteouts
  hide lower-layer names generically rather than special-casing disk.
- **Write capture.** Writes to files currently owned by a memfs layer update that memfs layer.
  Writes that resolve to the base disk layer use the base-layer behavior unless policy or
  stack semantics cause copy-up into a memfs layer.
- **Mutation semantics.** The overlay must define behavior for `Create`, `Unlink`, `Rename`,
  `Mkdir`, and `Rmdir` on stack-managed paths, covering synthetic entries, shadowed files,
  whiteouts, and cross-layer operations.
- **Programmatic API** for injection at server startup or at runtime from Rust code.
- **Embedded admin workflow support** for injection from a local terminal/admin loop or
  script/config-driven batch input while the server is running.

Use cases: credential injection, config overrides, hot-patching files for development,
providing synthetic content to AI agent sandboxes.

## Non-Functional Requirements

### NFR-1: Latency
Metadata operations (stat, lookup, readdir of small directories) must add < 1ms overhead
over the transport round-trip when using Unix sockets on the same machine.

### NFR-2: Throughput
Sequential read/write throughput must sustain > 500 MB/s over Unix sockets for large files,
limited by the transport and host filesystem, not by protocol overhead.

### NFR-3: Minimal Dependencies
The core module must have zero platform-specific dependencies. The server must not depend on
FUSE libraries. The client depends on `fuser` but nothing else platform-specific. Each
composite only pulls in the dependencies it needs via feature flags.

### NFR-4: Library-First
All functionality is exposed as library APIs with `pub` types and traits. No global state,
no implicit runtime. The caller provides the tokio runtime and wires components together.

### NFR-5: Testability
The server core must be testable without a real FUSE mount or any transport by calling
`handle_op()` directly. The RPC composite must be testable over in-memory duplex channels.
End-to-end FUSE tests are integration tests that can be skipped when FUSE is unavailable.

## High-Level System Design

### Composable Architecture

The library is structured around a server core with three composition layers.
All three call the same `server.handle_op()` -- the interception point for events
and policy. The composites differ only in how requests reach `handle_op()`.

```
                              ┌──────────────────────────────┐
                              │         server-core           │
                              │                               │
                              │  FsServer                     │
                              │    InodeTable (per mount tag) │
                              │    host FS ops (std::fs)      │
                              │    event emission (broadcast) │
                              │    policy enforcement         │
                              │                               │
                              │  fn handle_op(&self,          │
                              │    tag: &str,                 │
                              │    op: FsOp,                  │
                              │  ) -> FsResult                │
                              └──────┬───────────┬────────────┘
                                     │           │
            ┌────────────────────────┤           ├─────────────────────┐
            │                        │           │                     │
     ┌──────▼──────┐          ┌──────▼──────┐  ┌─▼───────────────┐    │
     │   direct     │          │   vsock      │  │   rpc            │   │
     │              │          │   composite  │  │   composite      │   │
     │ No transport │          │              │  │                  │   │
     │ No serde     │          │ bincode      │  │ Frame { id, body}│   │
     │              │          │ length-prefix│  │ Codec trait      │   │
     │ Caller calls │          │ over vsock   │  │ any transport    │   │
     │ handle_op()  │          │              │  │                  │   │
     │ directly     │          │ No Codec     │  │ RpcServer wraps  │   │
     │              │          │ trait, no    │  │   FsServer       │   │
     │ Use case:    │          │ Frame wrapper│  │ RpcClient wraps  │   │
     │ embed server │          │              │  │   ProtocolClient │   │
     │ in your own  │          │ Use case:    │  │                  │   │
     │ transport    │          │ VM guest ↔   │  │ Use case:        │   │
     │              │          │ host         │  │ cross-platform   │   │
     └──────────────┘          └──────────────┘  │ Linux host mount │   │
                                                  │ macOS mount      │   │
                                                  └──────────────────┘   │
                                                                         │
                                                  ┌──────────────────┐   │
                                                  │  client (fuser)  │◄──┘
                                                  │                  │
                                                  │ FuseClient:      │
                                                  │   fuser::        │
                                                  │   Filesystem     │
                                                  │                  │
                                                  │ Bridges FUSE ops │
                                                  │ to FsOp/FsResult │
                                                  │ via any composite│
                                                  └──────────────────┘
```

### Core Types: FsOp and FsResult

These are the types at the center of the design. Every composite converges on them.

```rust
/// A filesystem operation (request).  Core type shared by all composites.
#[derive(Debug, Serialize, Deserialize)]
pub enum FsOp {
    Lookup { parent: u64, name: String },
    Getattr { inode: u64 },
    Access { inode: u64, mask: i32, uid: u32, gid: u32 },
    Setxattr { inode: u64, name: String, value: Bytes, flags: i32, position: u32 },
    Getxattr { inode: u64, name: String, size: u32 },
    Listxattr { inode: u64, size: u32 },
    Removexattr { inode: u64, name: String },
    Getlk { inode: u64, fh: u64, lock_owner: u64, start: u64, end: u64, typ: i32, pid: u32 },
    Setlk { inode: u64, fh: u64, lock_owner: u64, start: u64, end: u64, typ: i32, pid: u32, sleep: bool },
    Setattr { inode: u64, attrs: SetAttrFields },
    Readdir { inode: u64, offset: i64 },
    Open { inode: u64, flags: u32 },
    Read { inode: u64, fh: u64, offset: i64, size: u32 },
    Write { inode: u64, fh: u64, offset: i64, data: Bytes },
    Create { parent: u64, name: String, mode: u32, flags: u32, uid: u32, gid: u32 },
    Mkdir { parent: u64, name: String, mode: u32, uid: u32, gid: u32 },
    Unlink { parent: u64, name: String },
    Rmdir { parent: u64, name: String },
    Rename { parent: u64, name: String, new_parent: u64, new_name: String },
    Symlink { parent: u64, name: String, target: String },
    Readlink { inode: u64 },
    Release { inode: u64, fh: u64 },
    Fsync { inode: u64, fh: u64, datasync: bool },
    Statfs,
}

/// A filesystem result (response).  Core type shared by all composites.
#[derive(Debug, Serialize, Deserialize)]
pub enum FsResult {
    Entry { inode: u64, generation: u64, attrs: FileAttr, ttl_secs: u32 },
    Created { inode: u64, generation: u64, attrs: FileAttr, fh: u64, ttl_secs: u32 },
    Attr { attrs: FileAttr, ttl_secs: u32 },
    Data { data: Bytes },
    Written { size: u32 },
    DirEntries { entries: Vec<DirEntry> },
    Statfs { stats: FsStats },
    Symlink { target: String },
    XattrSize { size: u32 },
    Lock { start: u64, end: u64, typ: i32, pid: u32 },
    Opened { fh: u64 },
    Ok,
    Error { errno: i32 },
}
```

`Bytes` is `bytes::Bytes` -- zero-copy for bulk data during encode/decode.

The compatibility pass adds three important semantics on top of the original v1 surface:

- `Access` is a server-side permission probe used by FUSE `access(2)` and toolchain preflight
  checks. It follows the inode attrs visible through the VFS view, with the known limitation that
  FUSE only supplies the caller's primary `gid`, not supplementary groups.
- xattrs are supported on both disk-backed and overlay-backed entries through
  `Setxattr`/`Getxattr`/`Listxattr`/`Removexattr`. Overlay xattrs are stored in the in-memory
  overlay node and survive overlay content replacement for that node.
- byte-range locks are exposed through `Getlk`/`Setlk`. Opened disk-backed files use stable
  `(dev, ino)` lock identity across rename; opened overlay-backed files use stable `(tag, inode)`
  identity. Blocking `Setlk { sleep: true }` waits in-process until the conflicting lock clears.

Known lock limits for v1:

- lock state is server-local and not durable across server restart
- the wait implementation is a single process-local condvar, so unrelated unlocks may wake other
  waiting lockers
- there is no FUSE `INTERRUPT` handling for blocked `Setlk { sleep: true }` requests yet

### Data Flow by Composite

**Direct:**
```
caller → FsOp → server.handle_op(tag, op) → FsResult → caller
```

**vsock:**
```
fuser → FsOp → bincode → [u32 len][payload] → vsock → [u32 len][payload] → bincode → FsOp
  → server.handle_op(tag, op)
  → FsResult → bincode → vsock → bincode → FsResult → fuser
```

**RPC:**
```
fuser → FsOp → Frame { request_id, body: op } → codec.encode() → [u32 len][payload]
  → transport (unix/tcp/tls)
  → [u32 len][payload] → codec.decode() → Frame → extract FsOp
  → server.handle_op(tag, op)
  → FsResult → Frame { request_id, body: result } → codec.encode() → transport
  → codec.decode() → Frame → extract FsResult → fuser
```

The vsock path skips Frame wrapping entirely. The handshake (`Fs { tag }`) already happened at
the motlie-vmm multiplexer level, so the tag is established before the library sees the stream.
The RPC path includes its own `Hello { tag }` handshake and `request_id` for pipelining.

### Event Emission (Same for All Composites)

```
All paths converge at server.handle_op(), where:

1. Policy check:    policy.check(op, tag, path) → Allow or Err(errno)
2. Layer-stack resolution: walk the mount's effective layers top-down
3. Match resolved result:
     Content(bytes) → return in-memory content (read) or update owning memfs layer (write)
     Whiteout       → return ENOENT (lower-layer entry suppressed)
     SyntheticDir   → return synthetic directory attrs/entries
     None           → continue downward; if no layer resolves, return ENOENT
4. Event emit:      event_tx.try_send(FsEvent { tag, op_kind, path, bytes })
5. Return:          FsResult
```

Events are emitted regardless of which layer supplied the content.

Events are emitted via `try_send` on a broadcast channel -- non-blocking, lossy under
backpressure (callers who care about completeness must keep up).

### In-Memory Layers in an Ordered Mount Stack

The memfs portion of the design is a set of file-driven in-memory layers composed into an
ordered per-mount stack. It is not a path→bytes map with special cases — it is a proper
in-memory filesystem layer model with inode allocation, directory traversal, metadata,
and mutation support.

- `put()` inserts or replaces a file node in the memfs layer.
- Any ancestors missing from the effective stack for that tag are materialized as synthetic
  directories in memory.
- Directory traversal and metadata for synthetic directories are served entirely from
  the memfs layer — nothing is forced onto host disk.
- Memfs layers share the mount's inode namespace: synthetic directory inodes and
  overlay file inodes are allocated from the same `InodeTable` as base-layer inodes.
  This ensures a unified inode space visible to the FUSE client.
- `handle_op()` resolves each path through the mount's ordered stack. In v1 the bottom
  layer is disk-backed, but the stack semantics are defined generically.

Multiple named memfs layers stack by priority above the base layer; resolution is top-down
(highest priority first).

```
handle_op(tag, FsOp::Read { inode, .. })
    │
    ├─ policy.check() → deny? return Error { errno }
    │
    ├─ memfs layer "hotpatch" (priority 100) → hit? return in-memory content
    ├─ memfs layer "user"     (priority 10)  → hit? return in-memory content
    ├─ memfs layer "defaults" (priority 0)   → hit? return in-memory content
    │    (entries can be: Content(bytes), Whiteout, or SyntheticDir)
    │
    └─ base layer                              → in v1: std::fs::read()
    │
    └─ event emit
```

**Layer semantics:**
- Each memfs layer is a named, ordered collection of `(tag, path) → Entry` mappings.
- A mount stack consists of zero or more named memfs layers above one base layer.
- An `Entry` is one of: `Content(Bytes)` (file data), `Whiteout` (suppresses lower-layer file),
  or `SyntheticDir` (directory that exists only in memory).
- Layers are created with an explicit priority (`u32`). Higher wins.
- `put()` on a layer sets or replaces content for a `(tag, path)` pair and automatically
  creates `SyntheticDir` entries only for parent directories missing from the effective stack
  for that tag. If a parent directory already exists in a lower layer, no synthetic parent entry
  is created for that segment.
- `remove()` on a layer deletes one entry; `remove_layer()` drops all entries.
- Resolution: for a given `(tag, path)`, walk layers highest-to-lowest, then the base layer.
  - `Content(bytes)` → return the in-memory content
  - `Whiteout` → return `ENOENT` (hides lower-layer entry below)
  - `SyntheticDir` → return directory inode/attrs
  - No hit in memfs layers → continue into the base layer
  - No hit in any layer → return `ENOENT`

This `(tag, path)` keying has an important scoping implication:

- one `MemOverlay` instance can hold entries for many mount tags at once
- within a single tag, entries for `/alice/...` and `/bob/...` are just different paths in the
  same mounted tree
- there is no per-connection or per-guest overlay namespace beneath a tag in v1
- if separate guests must not observe each other's overlay mutations, they should not share a tag

**Path contract for operators:**

- overlay paths are always specified as **mount-relative absolute paths** inside the mounted tree
  for a tag
- the operator does not provide a guest-global path to `MemOverlay`; it provides `tag` plus
  `path-within-tag`
- examples:
  - if tag `home` is mounted at guest `/home`, inject Alice's skills file as
    `tag="home", path="/alice/.claude/skills/tool.md"`
  - if tag `alice-home` is mounted at guest `/home/alice`, inject the same file as
    `tag="alice-home", path="/.claude/skills/tool.md"`

**Synthetic files and directories:**
- An overlay entry for a path that doesn't exist on disk creates a synthetic file.
  It appears in `readdir` alongside real directory entries. `lookup` and `getattr` return
  the overlay's metadata (size from content length, mtime from injection time, mode plus
  numeric `uid`/`gid`).
- **Implicit synthetic directories:** When `put()` injects a file, the overlay recursively
  materializes only parent directories missing from the effective stack for that tag as
  `SyntheticDir` entries in memory. For example,
  `put("layer", "home", "/.ssh/id_ed25519", key)` implicitly creates a `SyntheticDir`
  entry for `/.ssh/` only if no higher-priority memfs layer and no base-layer directory already
  provides `/.ssh/`.
  The synthetic directory has mode 0755, mtime from the first child injection, and appears
  in its parent's `readdir`. `lookup("/.ssh")` returns a directory inode.
- `readdir` on a synthetic directory returns only overlay entries (there are no disk entries
  to merge, since the directory doesn't exist on disk). `readdir` on a disk directory that
  also has overlay children merges both per-file as described above.
- Synthetic directories are reference-counted: they are removed automatically when the last
  child entry in the layer is removed.
- synthetic entry ownership rules in v1:
  - `put()` without explicit attrs inherits `uid`/`gid` from the nearest existing parent
    directory in the effective tree; if no nearer parent exists, inherit from the mount root
  - synthetic parent directories created implicitly by `put()` inherit the same `uid`/`gid`
    and default to mode `0755`
  - synthetic files created by `put()` default to mode `0644`
  - callers that need exact ownership may provide explicit attrs through the API described below
- v1 still does not define guest-specific uid/gid virtualization for a shared effective entry.

**Write behavior:**
- Writes to an overlaid path update the highest-priority layer that owns that path.
- Writes to a non-overlaid path go to disk.
- A future "capture" layer mode could intercept all writes, but this is out of scope.

**Overlay file handles and writes:**

- overlay-backed opens return synthetic file handles allocated from a per-mount counter
- server state keeps an `fh -> inode` mapping for overlay-backed opens
- `Read { fh, ... }` and `Write { fh, ... }` resolve through that mapping
- `Release { fh }` drops the mapping
- `Fsync { fh, .. }` on an overlay-backed file is a no-op that returns `Ok`
- v1 does not pin old overlay content at `Open` time; reads through an already-open overlay `fh`
  see the latest effective content for that inode/path at request time
- the published snapshot representation uses `Bytes`, but write-side mutation may use a mutable
  buffer internally (for example `Vec<u8>`) and freeze back to `Bytes` at publish time

**Whiteout semantics:**

A whiteout is a memfs-layer entry that suppresses a lower-layer entry. In v1 the lower layer is
often disk-backed, but the whiteout semantics are stack-generic. It makes the lower-layer entry
invisible to `lookup`, `getattr`, `readdir`, and `open` — as if the entry does not exist.
Whiteouts are created automatically by `unlink` on shadowed files (see mutation table).
They can also be created explicitly via the API for pre-emptive suppression.

v1 whiteout scope:

- whiteouts may suppress files or empty directories
- `whiteout()` may target a path that resolves to either kind of lower-layer entry
- v1 does not define recursive directory masking for non-empty lower-layer directories; callers
  that need to hide a populated subtree should whiteout individual entries or use policy filtering

```
resolve(tag, path):
  → Content(bytes)   → serve in-memory content
  → Whiteout         → return ENOENT (lower-layer entry hidden)
  → SyntheticDir     → return directory inode
  → None             → continue to the base layer
```

**Mutation semantics:**

Operations on overlay-backed paths follow filesystem-consistent behavior. The key
principles: (1) deleting a visible file makes it disappear — it never resurrects hidden
content, (2) renames across the memfs↔base-layer boundary are handled transparently to
support editor atomic-save flows, (3) the overlay behaves like a real filesystem layer,
not a cache.

| Operation | Target | Behavior |
|-----------|--------|----------|
| `Unlink` | Synthetic file | Remove overlay entry. File vanishes. |
| `Unlink` | Shadowed file (memfs over lower layer) | Replace overlay `Content` entry with `Whiteout`. File disappears; lower-layer file stays hidden. |
| `Unlink` | Whiteout | Return `ENOENT` (already deleted). |
| `Unlink` | Base-layer-only file | Normal base-layer remove operation. |
| `Create` | Under synthetic directory | Create `Content` entry in the designated writable memfs layer. If a memfs layer owns the nearest visible ancestor, that layer receives the new file; otherwise the highest-priority writable memfs layer for the mount captures it. |
| `Create` | Under disk directory (no overlay children) | Normal `std::fs::create()` on disk. |
| `Create` | Under disk directory (has overlay children) | Create `Content` entry in the designated writable memfs layer. Keeps overlay-managed paths consistent. |
| `Symlink` | Under synthetic or overlay-managed parent | Return `ENOTSUP` in v1. Symlinks are base-layer-only in v1. |
| `Readlink` | Overlay-managed inode | Return `ENOTSUP` in v1. |
| `Mkdir` | Under synthetic parent | Create `SyntheticDir` entry in the designated writable memfs layer using the request `mode` and caller `uid`/`gid`. |
| `Mkdir` | Under disk parent | Normal `std::fs::create_dir()` on disk. |
| `Rmdir` | Synthetic directory (empty) | Remove `SyntheticDir` from overlay. |
| `Rmdir` | Shadowed empty lower-layer directory | Create a directory whiteout in the designated writable memfs layer so the lower directory stays hidden. |
| `Rmdir` | Disk directory | Normal `std::fs::remove_dir()`. |
| `Rename` | Overlay → overlay (same memfs layer) | Rename within that memfs layer (in-memory). |
| `Rename` | Disk → disk | Normal `std::fs::rename()`. |
| `Rename` | Disk → overlay target | Supported in v1 for the editor atomic-save path only: read disk source content, update the target's writable memfs layer, then remove the disk source. |
| `Rename` | Overlay source → disk target | Return `EXDEV` in v1. Callers must fall back to copy+delete if they need this path. |
| `Rename` | Cross-layer cases not listed above | Return `EXDEV` in v1 unless source and target are in the same memfs layer. |
| `Setattr` | Overlaid/synthetic file | Update in-memory attrs. |
| `Setattr` | Disk file | Normal `std::fs` operation. |

**Why the v1 rename carve-out exists:**

The stated target workload is coding tools (editors, compilers, git). Editors perform
atomic saves by writing to a temp file and renaming over the target:

```
vim writes /root/.ssh/config.tmp     (new file — created in overlay because .ssh has overlay children)
vim renames config.tmp → config      (overlay → overlay rename — works)
```

If the temp file is created on disk (e.g., in a non-overlay directory), the rename is
disk→overlay. v1 supports this specific path so editor atomic-save flows can work for
overlay-managed targets: read the disk source, write into the overlay target, then delete
the disk source. Other cross-layer rename cases remain out of scope for v1 and return `EXDEV`.

## Crate Structure

```
libs/vfs/
├── Cargo.toml
├── docs/
│   └── DESIGN.md                 # this document
├── examples/
│   ├── README.md                 # Cloud Hypervisor proof-of-concept instructions
│   ├── v1/
│   │   └── repl_host.rs          # stable v1 single-guest host REPL example
│   └── v1.1/
│       └── repl_host.rs          # v1.1 multi-guest host REPL example
├── bins/
│   ├── v1/
│   │   └── motlie-vfs-guest.rs   # v1 guest-side mounter binary over public guest APIs
│   └── v1.1/
│       └── motlie-vfs-guest.rs   # v1.1 guest-side mounter binary with TAG handshake
└── src/
    ├── lib.rs                    # re-exports, feature-gated module visibility
    │
    ├── core/
    │   ├── mod.rs
    │   ├── op.rs                 # FsOp, FsResult, FileAttr, DirEntry, FsStats
    │   ├── server.rs             # FsServer, FsServerBuilder, handle_op()
    │   ├── inode.rs              # InodeTable: unified inode namespace (disk + overlay + synthetic)
    │   ├── overlay.rs            # MemOverlay: layered in-memory content injection
    │   ├── event.rs              # FsEvent, FsOp (event enum), EventSender
    │   └── policy.rs             # PolicyFn trait, AllowAll default
    │
    ├── vsock/                    # feature = "vsock"
    │   ├── mod.rs
    │   ├── handler.rs            # VsockConnectionHandler: bincode FsOp/FsResult over stream
    │   └── client.rs             # VsockClientTransport: guest-side request/response transport
    │
    └── client/                   # feature = "client"
        ├── mod.rs
        ├── fuse.rs               # FuseClient: guest-side fuser::Filesystem over a transport
        └── guest.rs              # GuestMountRunner + GuestMountSpec: guest-side mount orchestration over public APIs
```

Boundary note:

- `bins/v1/motlie-vfs-guest.rs` is the stable v1 guest-side mounter binary built from this crate
- `bins/v1.1/motlie-vfs-guest.rs` is the v1.1 guest-side mounter binary that sends the per-tag handshake
- `examples/` is reserved for harness/demo programs; the guest-side mounter is not just an
  example because it must be packaged into the guest image for v1 testing
- `client/guest.rs` is the guest-facing orchestration layer above `FuseClient`; example
  binaries and guest binaries should call into it rather than reimplementing mount loops
- the tiny bootstrap binary and `BinaryRequest` delivery path remain VMM-owned and are not part
  of `libs/vfs`

### Feature Flags

```toml
[features]
default = ["server-core", "bincode-codec"]

# Always available -- core types, FsServer, MemOverlay
server-core = []

# v1 VM transport
vsock = ["server-core", "bincode-codec", "dep:tokio-vsock"]

# FUSE client for the v1 guest path
client = ["dep:fuser"]

# Wire encodings
bincode-codec = ["dep:bincode"]
```

**Dependency profiles for each consumer:**

| Consumer | Features | What compiles |
|----------|----------|---------------|
| motlie-vmm host | `vsock` | core + overlay + vsock handler + bincode |
| motlie-vmm guest | `vsock, client` | core + vsock client transport + `FuseClient` + fuser + bincode |
| Proof-of-concept host example | `vsock` | core + overlay + in-process REPL + CH harness support |
| Proof-of-concept guest example | `vsock, client` | guest bootstrap + vsock transport + `FuseClient` |
| Custom embedding | `server-core` | core + overlay, caller provides own control loop |
| Tests (no FUSE) | `server-core, bincode-codec` | core + overlay + direct tests |

## API Design

### Server Core

```rust
/// Core filesystem server.  All composites call handle_op().
pub struct FsServer { /* ... */ }

impl FsServer {
    pub fn builder() -> FsServerBuilder;

    /// Direct operation dispatch.  The single entry point that all composites use.
    /// Tag must match a registered mount.  Returns FsResult::Error { errno: ENOENT }
    /// if the tag is unknown.
    pub fn handle_op(&self, tag: &str, op: FsOp) -> FsResult;

    /// Register a new mount tag.  Can be called while serving.
    pub async fn add_mount(&self, tag: &str, host_path: PathBuf, read_only: bool) -> Result<()>;

    /// Remove a mount tag.  Drops inode table; in-flight ops get ENOENT.
    pub async fn remove_mount(&self, tag: &str) -> Result<()>;

    /// Subscribe to filesystem events (if event emission is enabled).
    pub fn subscribe_events(&self) -> Option<broadcast::Receiver<FsEvent>>;
}

pub struct FsServerBuilder { /* ... */ }

impl FsServerBuilder {
    pub fn mount(self, tag: &str, host_path: PathBuf, read_only: bool) -> Self;
    pub fn events(self, capacity: usize) -> Self;
    pub fn policy(self, policy: impl PolicyFn) -> Self;
    pub fn build(self) -> Result<FsServer>;
}
```

v1 forward-compatibility constraint:

- these APIs are disk-backed in v1
- their internal implementation should still leave room for a future mount type that is not
  backed by a local host directory
- future diskless/synthetic-root mounts must be able to coexist without requiring a redesign of
  `MemOverlay` itself

### Event Types

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FsEvent {
    pub timestamp: SystemTime,
    pub tag: String,
    pub op_kind: FsOpKind,
    pub path: String,
    pub bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FsOpKind {
    Lookup, Getattr, Setattr, Readdir, Open, Read, Write, Create,
    Mkdir, Unlink, Rmdir, Rename, Symlink, Readlink, Release, Fsync, Statfs,
}
```

The motlie-vmm daemon wraps these with VM identity and credential classification
(using its own `AuditedFsServer` wrapper). The library emits raw FS events; the caller
adds domain context.

### Policy Trait

```rust
pub trait PolicyFn: Send + Sync + 'static {
    /// Check whether an operation is allowed.
    /// Return Ok(()) to allow, Err(errno) to deny.
    /// Called for every FsOp including individual Readdir entries.
    fn check(&self, op: FsOp, tag: &str, path: &str) -> Result<(), i32>;
}

/// Default: allow everything.
pub struct AllowAll;
impl PolicyFn for AllowAll {
    fn check(&self, _op: FsOp, _tag: &str, _path: &str) -> Result<(), i32> {
        Ok(())
    }
}
```

### Async Runtime Note

`FsServer::handle_op()` is intentionally synchronous at the core API boundary. In v1 the base
layer uses blocking `std::fs` operations, so async callers should treat `handle_op()` as
blocking work.

Required guidance for async composites:

- async runtimes such as Tokio should invoke `handle_op()` via `spawn_blocking` or an equivalent
  blocking-worker strategy
- the core crate should not assume that an async runtime is always present

### In-Memory Overlay API

The overlay is a file-driven memfs layer, part of server-core (always available when
FsServer is compiled). No additional feature flag required.

**Internal model:** The overlay maintains a tree of nodes — file nodes (`Content`),
whiteout nodes (`Whiteout`), and directory nodes (`SyntheticDir`) — allocated from the
mount's shared `InodeTable`. Each node has an inode, attrs (size, mode, timestamps),
and parent linkage. `put()` inserts a file node and recursively creates `SyntheticDir`
parent nodes. `handle_op()` dispatches against this tree before continuing into the base layer.

Forward-compatibility constraint:

- in v1, “no memfs-layer hit” means “continue into the base layer”
- v2 may introduce mounts whose non-overlay behavior is not disk-backed at all
- implementers should therefore keep overlay node management independent from the concrete
  fallback backend, even if v1 only ships disk-backed mounts

**v1 concurrency model:**

- per mount tag, memfs publication is snapshot-based
- writers to the same mount tag are serialized
- `apply_batch()` is the authoritative mutation primitive; `put()`, `put_with_attrs()`,
  `whiteout()`, and `remove()` are convenience forms of one-op batches
- a committed batch publishes a new memfs snapshot atomically for that tag
- read-like operations (`lookup`, `getattr`, `read`, `readdir`) use one loaded snapshot for the
  duration of the request
- the base layer is not made transactional by this library; its concurrency behavior is whatever
  the underlying backing provides

**External API:** Callers interact through `put()` / `remove()` / `whiteout()` (mutations)
and `resolve()` / `list_effective()` (inspection). The inspection types (`OverlayEntry`,
`EffectiveEntry`, `LayerInfo`) are flattened views for debugging, embedded admin UIs, and
programmatic introspection — they do not expose the full internal tree structure or inode mappings.

```rust
/// File-driven in-memory filesystem layer.
/// Injecting a file automatically materializes synthetic parent directories.
/// Accessed via server.overlay().
pub struct MemOverlay { /* ... */ }

/// What an overlay entry contains in the internal memfs model.
#[derive(Clone, Debug)]
pub enum OverlayEntryKind {
    /// File content (injected via put() or captured from writes).
    Content(Bytes),
    /// Suppresses a lower-layer entry — makes it invisible.
    Whiteout,
    /// Directory that exists only in memory (created implicitly by put() for missing parents).
    SyntheticDir,
}

/// Metadata-only view used by list_layer()/list_effective().
#[derive(Clone, Debug)]
pub enum OverlayEntryViewKind {
    Content { size: usize },
    Whiteout,
    SyntheticDir,
}

#[derive(Debug, Clone, Copy)]
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

impl MemOverlay {
    // --- Layer management ---

    /// Create or update a named layer with a priority.
    /// Higher priority layers shadow lower ones.
    pub fn put_layer(&self, name: &str, priority: u32) -> Result<()>;

    /// Remove a layer and all its entries (including whiteouts and synthetic dirs).
    pub fn remove_layer(&self, name: &str) -> Result<()>;

    /// List all layers, ordered by priority (highest first).
    pub fn layers(&self) -> Vec<LayerInfo>;

    // --- Content management (within a layer) ---

    /// Inject a file into the overlay. Automatically creates SyntheticDir entries
    /// for parent directories missing from the effective stack for this tag.
    /// If the path already has an entry (Content, Whiteout, or SyntheticDir), replaces it.
    /// `path` is mount-relative and must begin with `/`.
    /// Synthetic ownership defaults are inherited from the nearest existing parent
    /// directory in the effective tree, or from the mount root if no nearer parent exists.
    pub fn put(&self, layer: &str, tag: &str, path: &str, content: Bytes) -> Result<()>;

    /// Inject a file into the overlay with explicit synthetic metadata.
    /// `path` is mount-relative and must begin with `/`.
    pub fn put_with_attrs(
        &self,
        layer: &str,
        tag: &str,
        path: &str,
        attrs: OverlayAttrs,
        content: Bytes,
    ) -> Result<()>;

    /// Create a whiteout entry that suppresses a lower-layer entry.
    /// `path` is mount-relative and must begin with `/`.
    pub fn whiteout(&self, layer: &str, tag: &str, path: &str) -> Result<()>;

    /// Remove one overlay entry.  For Content/Whiteout: resolution may expose a lower layer
    /// (or disappear entirely if no lower layer provides the path). Synthetic parent dirs are removed if they
    /// have no remaining children.
    /// `path` is mount-relative and must begin with `/`.
    pub fn remove(&self, layer: &str, tag: &str, path: &str) -> Result<()>;

    /// Apply a batch atomically for one mount tag.
    /// Single-file mutations are represented as batches of size 1.
    /// Readers observe either the pre-batch or post-batch memfs snapshot for that tag.
    /// `OverlayMutation` does not carry its own tag; the outer `tag` parameter is authoritative
    /// and every operation in the batch is scoped to that tag.
    pub fn apply_batch(&self, tag: &str, ops: &[OverlayMutation]) -> Result<()>;

    /// Read the content stored in a specific layer (for debugging/inspection).
    pub fn get(&self, layer: &str, tag: &str, path: &str) -> Option<Bytes>;

    /// List all entries in a layer for a mount tag.
    pub fn list_layer(&self, layer: &str, tag: &str) -> Vec<OverlayEntry>;

    // --- Resolved view (what handle_op() sees) ---

    /// Resolve a path within the named memfs layers: walk layers highest-to-lowest, return first hit.
    /// Returns the memfs-layer entry kind. None means the caller should continue into the base layer.
    pub fn resolve(&self, tag: &str, path: &str) -> Option<(String, OverlayEntryKind)>;

    /// List all effective overlays for a tag (one entry per path, from the
    /// highest-priority layer that owns it).
    pub fn list_effective(&self, tag: &str) -> Vec<EffectiveEntry>;
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
    pub size: usize,                // 0 for Whiteout and SyntheticDir
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
```

Note: `OverlayEntry` and `EffectiveEntry` are metadata-only flattened inspection views.
The internal model is a tree of inode-allocated nodes with parent linkage and refcounting
for synthetic directories. These structs project that tree into flat lists for embedded
admin views, debugging, and programmatic introspection without returning full file contents.
Callers that need file data should use `get()` for a specific `(layer, tag, path)`.

Wired into FsServer:

```rust
impl FsServerBuilder {
    /// Enable the in-memory overlay (default: disabled, handle_op goes straight to disk).
    pub fn overlay(self, enabled: bool) -> Self;
}

impl FsServer {
    /// Access the overlay.  Returns None if overlay was not enabled in the builder.
    pub fn overlay(&self) -> Option<&MemOverlay>;
}
```

### v1.5 Roadmap Sketch: Embedded Admin Console + Script / Config Loading

The first operator-facing refinement after v1 remains embedded and local to the hosting process.

Required contract for that future v1.5 path:

- it uses the core `MemOverlay` API exposed by `FsServer`
- it preserves the mount-relative path contract:
  - operator supplies `tag`
  - operator supplies mount-relative absolute `path`
- it supports at least:
  - `put`
  - `putattr`
  - `whiteout`
  - `rm`
  - `rmlayer`
  - `ls`
  - `lslayer`
- command/API mapping for implementers:
  - `put` -> `put()`
  - `putattr` -> `put_with_attrs()`
  - `whiteout` -> `whiteout()`
  - `rm` -> `remove()`
  - `rmlayer` -> `remove_layer()`
  - `ls` -> `list_effective()`
  - `lslayer` -> `list_layer()`
- it supports batch ingestion from a script/config format so startup and reload flows are repeatable

Detailed console UX and script/config format are deferred to a future design for the v1.5 workflow.

### Overlay Control: Frontend Architecture

The `MemOverlay` is a Rust API on `FsServer`. Any code with access to `server.overlay()` can
call `put`/`remove`/`put_layer`. This makes the overlay composable with embedded admin loops now
and remote admin frontends later.

This layering is intentional:

- `MemOverlay` is the core overlay API surface
- v1 uses direct Rust calls and example programs under `libs/vfs/examples`
- v1.5 adds a stronger embedded admin console plus script/config-driven batch loading
- v2 adds external remote-control layers such as gRPC or custom RPC

v1 scope boundary:

- in scope: direct Rust calls to `server.overlay()`
- in scope: proof-of-concept REPL or command loop inside `libs/vfs/examples`
- out of scope for v1: standalone remote admin interface
- out of scope for v1: external CLI application
- out of scope for v1: library-provided HTTP admin server
- out of scope for v1: library-provided gRPC admin server
- out of scope for v1: library-provided overlay-specific network admin service

**Dynamic credential injection while mounted (all frontends):**

Because `handle_op()` checks the overlay on every operation with no caching, a `PUT`
followed by a `read()` in the guest sees the new content immediately. This enables
dynamic, mid-session credential management:

```bash
# Mid-session: inject a deploy key (via embedded admin console, script reload, or in-process call)
server.overlay().put("credentials", "home", "/.ssh/id_deploy", key_bytes)

# Guest immediately sees it — no restart, no remount
ssh -i ~/.ssh/id_deploy git@github.com   # works

# Shadow an existing key (disk version hidden, overlay version served)
server.overlay().put("credentials", "home", "/.ssh/id_ecdsa", ephemeral_key)
# Guest reads /root/.ssh/id_ecdsa → gets ephemeral_key, not the on-disk version

# Remove — original disk file reappears, synthetic files vanish
server.overlay().remove("credentials", "home", "/.ssh/id_deploy")
server.overlay().remove("credentials", "home", "/.ssh/id_ecdsa")
```

**motlie-vmm integration:**

For the VM use case, the VMM daemon calls `server.overlay()` directly from Rust — no remote admin
surface needed. The VMM's `ControlMsg::AddMount` handler can inject overlay content
as part of VM setup. Future remote admin frontends are additive and do not change the
core VMM integration path.

The same pattern supports an in-process REPL or operator loop inside the VMM process itself.
Because `motlie-vfs` is library-first and uses Tokio-friendly async I/O, the VMM can host:

- the filesystem serving tasks
- the vsock/RPC/FUSE transport tasks
- an interactive or programmatic admin loop that calls `server.overlay()` directly

inside one async process without introducing a separate overlay daemon.

### vsock Composite

Thin layer: length-prefixed bincode of `FsOp`/`FsResult` directly over a stream.
No `Frame` wrapper, no `Codec` trait, no handshake (the motlie-vmm multiplexer
handles the `Fs { tag }` handshake before handing off the stream).

```rust
/// Host side: serves a single vsock connection for a known tag.
pub struct VsockConnectionHandler { /* ... */ }

impl VsockConnectionHandler {
    /// Create a handler bound to a specific tag.
    /// The tag is already established by the caller (vmm multiplexer handshake).
    pub fn new(server: &FsServer, tag: &str) -> Self;

    /// Serve the connection: read FsOp, call handle_op(), write FsResult.
    /// Loops until the stream closes.
    pub async fn serve<S>(&self, stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send;
}
```

```rust
/// Guest side: request/response transport over an established stream.
pub struct VsockClientTransport { /* ... */ }

impl VsockClientTransport {
    /// Create a transport backed by the given stream.
    /// The tag is already established by the caller (guest agent handshake).
    pub fn new<S>(stream: S, tag: &str) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static;
}
```

### v2 Roadmap Sketch: External RPC / gRPC Layer

This section is a future roadmap sketch, not current `libs/vfs` v1 crate scope.

Roadmap placement:

- v2 external application layer
- built on top of the core crate API
- not implemented inside `libs/vfs` itself

The future v2 layer may add:

- a remote admin API
- richer request/response semantics
- streaming or multi-client coordination
- alternative transport choices such as gRPC or a custom RPC service

Those details are intentionally deferred to a future design outside this crate. The only
contract that matters here is that the `libs/vfs` core API should remain clean enough that
such an external layer can be built later without reshaping the core filesystem and overlay
semantics.

## Composition Patterns (Usage Examples)

### Pattern 1: Direct -- Embed Server in Your Own Transport

The caller owns the transport and encoding. The library provides only the FS engine.

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

let server = FsServer::builder()
    .mount("workspace", "/home/alice/projects".into(), false)
    .events(4096)
    .build()?;

// Subscribe to events in a background task
let mut events = server.subscribe_events().unwrap();
tokio::spawn(async move {
    while let Ok(ev) = events.recv().await {
        println!("{:?}", ev);
    }
});

// You decode requests however you want, then call handle_op:
let result = server.handle_op("workspace", FsOp::Lookup {
    parent: 1,
    name: "src".into(),
});
match result {
    FsResult::Entry { inode, attrs, .. } => { /* route response back */ }
    FsResult::Error { errno } => { /* handle error */ }
    _ => {}
}
```

### Pattern 2: vsock -- motlie-vmm Host Side

The VMM daemon handles the vsock multiplexer and dispatches `Fs { tag }` connections
to the library.  The library never sees the `HandshakeMsg` enum -- it receives a stream
that is already bound to a tag.

```rust
use motlie_vfs::core::FsServer;
use motlie_vfs::vsock::VsockConnectionHandler;

let server = FsServer::builder()
    .mount("workspace", projects_dir.join(&username), false)
    .mount("cred-claude", creds_dir.join(&username).join(".claude"), false)
    .events(4096)
    .policy(ReadOnlyCredentials::new())
    .build()?;

// In the vsock multiplexed listener, after HandshakeMsg dispatch:
match handshake {
    HandshakeMsg::Fs { tag } => {
        let handler = VsockConnectionHandler::new(&server, &tag);
        handler.serve(vsock_stream).await;
    }
    HandshakeMsg::BinaryRequest => { /* ... handled by vmm, not the lib */ }
    HandshakeMsg::Control => { /* ... handled by vmm, not the lib */ }
}
```

### Pattern 3: vsock -- motlie-vmm Guest Side

The guest agent opens a vsock connection, performs the VMM-level handshake, then
hands the stream to the library for FUSE mounting.

```rust
use motlie_vfs::client::FuseClient;
use motlie_vfs::vsock::VsockClientTransport;
use tokio_vsock::VsockStream;

// Guest agent: open vsock, do the vmm-level handshake
let mut stream = VsockStream::connect(2, 5000).await?;
send_handshake(&mut stream, &HandshakeMsg::Fs { tag: "workspace".into() }).await?;

// Hand the established stream to the library transport, then mount via FuseClient
let transport = VsockClientTransport::new(stream, "workspace");
let mount = FuseClient::new(transport, "workspace");
fuser::mount2(mount, "/workspace", &[
    MountOption::AllowOther,
])?;
```

### Pattern 5: Testing -- No FUSE, No Transport

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

#[tokio::test]
async fn test_read_write() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), b"world").unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .build()
        .unwrap();

    // Direct: no transport, no serde, no FUSE
    let entry = server.handle_op("test", FsOp::Lookup {
        parent: 1,
        name: "hello.txt".into(),
    });
    let inode = match entry {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };

    let open = server.handle_op("test", FsOp::Open { inode, flags: 0 });
    let fh = match open {
        FsResult::Entry { .. } => { /* extract fh */ 0 },
        other => panic!("expected open result, got {:?}", other),
    };

    let data = server.handle_op("test", FsOp::Read {
        inode, fh, offset: 0, size: 4096,
    });
    match data {
        FsResult::Data { data } => assert_eq!(&data[..], b"world"),
        other => panic!("expected Data, got {:?}", other),
    }
}
```

### Pattern 6: Overlay -- Direct Testing

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

#[tokio::test]
async fn test_overlay_shadows_disk() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.toml"), b"disk content").unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .overlay(true)
        .build()
        .unwrap();

    // Without overlay: read returns disk content
    let inode = lookup(&server, "test", "config.toml");
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"disk content");

    // Inject overlay
    let overlay = server.overlay().unwrap();
    overlay.put_layer("patch", 10).unwrap();
    overlay.put("patch", "test", "/config.toml", Bytes::from("overlaid")).unwrap();

    // Same read now returns overlay content
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"overlaid");

    // Remove overlay: falls back to disk
    overlay.remove("patch", "test", "/config.toml").unwrap();
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"disk content");
}

#[tokio::test]
async fn test_overlay_synthetic_file() {
    let dir = tempfile::tempdir().unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .overlay(true)
        .build()
        .unwrap();

    let overlay = server.overlay().unwrap();
    overlay.put_layer("inject", 0).unwrap();
    overlay.put("inject", "test", "/.env", Bytes::from("SECRET=abc")).unwrap();

    // Synthetic file: doesn't exist on disk but is visible via lookup + readdir
    let entry = server.handle_op("test", FsOp::Lookup {
        parent: 1,
        name: ".env".into(),
    });
    assert!(matches!(entry, FsResult::Entry { .. }));

    // Also appears in directory listing
    let entries = server.handle_op("test", FsOp::Readdir { inode: 1, offset: 0 });
    // entries includes .env alongside any real files
}
```

## Inode Management (Memfs-Aware)

The server maintains a **per-mount `InodeTable`** that is the single inode namespace for
all entries visible through that mount — disk-backed files, overlay `Content` nodes,
`SyntheticDir` nodes, and `Whiteout` bookkeeping. Inode 1 is always the mount root.

### InodeTable Structure

```
InodeTable:
  inode → InodeEntry {
      kind: InodeKind,       // Disk, Content, SyntheticDir, Whiteout
      path: String,          // mount-relative path (e.g. "/.ssh/id_ed25519")
      host_path: Option<PathBuf>,  // Some for Disk entries, None for overlay entries
      generation: u64,       // bumped on content/kind change at this inode
      refcount: u64,         // FUSE lookup count
      attrs: FileAttr,       // authoritative for overlay; re-fetched from std::fs for disk (v1: no cache)
  }

  path → inode              // reverse lookup
```

### Inode Allocation by Entry Class

| Entry class | When allocated | `kind` | `host_path` | `attrs` source |
|-------------|---------------|--------|-------------|----------------|
| Disk file/dir | First `lookup` that resolves to host filesystem | `Disk` | `Some(host_path)` | `std::fs::metadata()` on every access (v1: no attrs cache) |
| Overlay `Content` | `put()` / `put_with_attrs()` or `Create` under overlay-managed dir | `Content` | `None` | Size from content length, mode+uid+gid from explicit attrs or inherited defaults, mtime from injection time |
| `SyntheticDir` | Implicitly by `put()` for missing parent dirs | `SyntheticDir` | `None` | Mode 0755 by default, uid/gid inherited from nearest existing parent or mount root, mtime from first child injection |
| `SyntheticDir` | Explicit `Mkdir` under synthetic parent | `SyntheticDir` | `None` | Mode from the request, uid/gid from the caller, mtime from creation time |
| `Whiteout` | `Unlink` on shadowed file | `Whiteout` | `None` | Not visible (lookup returns ENOENT) |

### Inode Stability

- **Path-stable within a mount session:** a given path always resolves to the same inode
  as long as the mount is alive and the path hasn't been removed and re-created.
- **Not stable across overlay mutations that change kind:** if `put()` replaces a `Whiteout`
  with `Content` at the same path, the inode is reused but `generation` is bumped. The FUSE
  client sees a new generation and invalidates its cache for that inode.
- **Not stable across `remove_layer()`:** removing a layer may expose a lower-priority entry
  or continue into the base layer. If the effective entry kind changes, `generation` is bumped.
- **Not stable across mount restart:** inode numbers are ephemeral per mount session.

### Generation Semantics

`generation` is a monotonically increasing counter per inode. It is bumped when:

| Event | Generation bump? | Rationale |
|-------|-----------------|-----------|
| `put()` replacing existing `Content` at same path | Yes | Content changed; FUSE client must invalidate cached data |
| `put()` replacing `Whiteout` with `Content` | Yes | Entry kind changed from invisible to visible |
| `whiteout()` replacing `Content` | Yes | Entry kind changed from visible to invisible |
| `remove()` dropping overlay entry (lower-layer exposure) | Yes | Effective content changed because resolution moved to a lower layer |
| `remove_layer()` changing effective entry | Yes | Winning layer may change, altering content |
| Layer priority change (`put_layer` with new priority) | Yes (for affected paths) | Resolution order changed |
| `Write` updating existing `Content` in-place | No | Same entry, same kind, content updated |
| `Setattr` on overlay entry | No | Attrs updated but identity unchanged |

### Mount Scoping and Removal

Inodes are scoped per mount tag — two mounts can independently use the same inode numbers.
This is safe because each client connection is bound to exactly one tag, and the FUSE kernel
module per mount point has its own inode namespace.

When a mount is removed via `remove_mount()`, its `InodeTable` is dropped entirely —
disk, overlay, synthetic, and whiteout entries are all invalidated. Any in-flight connection
for that tag receives `FsResult::Error { errno: ENOENT }` on subsequent requests.

## Cache Visibility and Invalidation

### v1 Design Principle: No Caching, Correctness First

The v1 implementation biases toward **determinism and correctness over cache performance**.
There are no semantic caches inside the overlay layer beyond the required inode/object
mappings in `InodeTable`. Every `lookup`, `getattr`, `readdir`, and `read` on an
overlay-backed path resolves from the live in-memory state on every call. Every disk-backed
operation goes to `std::fs` on every call.

This means:
- No attrs cache for overlay entries (attrs are computed from the live `Content`/`SyntheticDir` state).
- No attrs cache for disk entries (every `getattr` calls `std::fs::metadata()`).
- No readdir cache (every `readdir` re-merges disk + overlay + policy filter).
- No data cache (every `read` returns live overlay content or calls `std::fs::read()`).
- No background invalidation machinery.
- Runtime `put` / `whiteout` / `remove` / `remove_layer` mutations are visible on the
  very next filesystem operation — no stale state possible.

The `InodeTable` itself is not a cache — it is the authoritative inode namespace. Inode
allocation is stable within a mount lifetime (see Inode Stability above). The table maps
inodes to entry metadata but does not cache file content or directory listings.

### FUSE Kernel TTL Strategy

The FUSE kernel independently caches lookup results, attrs, and data based on TTL values
returned by the server. The v1 strategy is zero TTL for everything:

| Entry class | `ttl_secs` for lookup/getattr | `ttl_secs` for data | Rationale |
|-------------|-------------------------------|---------------------|-----------|
| Disk file/dir | 0 | 0 | No server-side cache; kernel must revalidate every access |
| Overlay `Content` | 0 | 0 | Content can change at any time via `put()` / `apply_batch()` / future admin frontend |
| `SyntheticDir` | 0 | N/A | May gain/lose children via `put()` / `remove()` |
| `Whiteout` | 0 | N/A | May be replaced by `put()` |

Zero TTL across the board means the kernel revalidates on every access. Combined with no
server-side caching, the system is fully deterministic: every operation reflects the current
state of disk + overlay + policy. The cost is one round-trip per access, which is acceptable
for v1 where correctness matters more than throughput.

### v1 FUSE Mount Policy

To make the zero-TTL / no-cache guarantee operational, the FUSE client must be mounted
with the stable access-control options below, and each opened file handle must request
`direct_io` so the kernel does not serve stale page-cache data independently of TTL
revalidation. The v1 mount options are:

```rust
// v1 mount options — correctness-first, direct /dev/fuse mount
let mount_opts = vec![
    MountOption::AllowOther,        // any guest user, including root, can access the mount
];
```

`AutoUnmount` is intentionally not used in the guest service path because `fuser`
implements it via `fusermount3`, which is unreliable in the minimal guest image.
The service mounts directly through `/dev/fuse` instead.

**Why `direct_io`:** `direct_io` is applied per open via `FOPEN_DIRECT_IO`, not as a
mount option. Without it, the kernel caches file data in the page cache.
Even with `ttl_secs = 0` for attrs, the kernel may serve `read()` from stale page cache
data if the overlay content was mutated via `put()` between reads. `direct_io` forces every
`read()` and `write()` through the FUSE server, ensuring the live overlay state is returned.

**Cost of `direct_io`:** No `mmap` support (already a non-goal), no kernel read-ahead, every
read is a round-trip to the server. For the v1 workload (coding tools, credential files,
config), this is acceptable. The future performance phase can switch to `kernel_cache` +
targeted invalidation if profiling warrants it.

All composition pattern examples in this document that call `fuser::mount2()` use these
v1 mount options.

### Why Not Explicit Kernel Invalidation (v1)

FUSE supports `notify_inval_inode` and `notify_inval_entry` to proactively push cache
invalidations. This would allow longer TTLs with on-demand invalidation on overlay
mutations. This is explicitly deferred from v1:

- It requires the server to hold a `Session` reference from `fuser`, coupling the server
  core to the FUSE client — violating the composable architecture (direct and RPC composites
  don't have a `Session`).
- Zero TTL achieves the same correctness with no coupling and no invalidation logic.
- Adding invalidation is a backward-compatible optimization: increase TTLs and add
  `notify_inval_*` calls in the FUSE client module, no server core or protocol changes.

### Generation + TTL Interaction

When a FUSE client does a lookup and gets `(inode, generation, attrs, ttl=0)`:
1. Next access: kernel revalidates (calls `lookup` or `getattr` again because `ttl=0`).
2. Server returns current `(inode, generation, attrs)` from live state.
3. If `generation` changed: kernel discards all cached data/attrs for that inode.
4. If `generation` unchanged: kernel may use previously cached data (but `ttl=0` means
   it will revalidate again on the next access anyway).

Overlay mutations are visible on the very next filesystem operation. No explicit
invalidation needed.

### Future: Performance Phase

When v1 is stable, a subsequent phase can introduce caching without architectural changes:
- Disk attrs cache with configurable TTL (e.g., `ttl_secs = 1`)
- Overlay readdir cache invalidated on `put`/`remove` (saves re-merge cost)
- Targeted `notify_inval_*` calls in the FUSE client for longer kernel TTLs
- None of these require server core, protocol, or API changes

## macOS: FUSE-T Strategy

macOS FUSE support uses **FUSE-T** (not macFUSE):

| Property | FUSE-T | macFUSE |
|----------|--------|---------|
| Mechanism | Userspace NFS loopback | Kernel extension (kext) |
| SIP | No changes needed | Must lower security |
| Install | `brew install fuse-t` | PKG + reboot |
| `fuser` compat | Yes (provides `libfuse.dylib`) | Yes |
| Future-proof | Yes (kexts deprecated by Apple) | Deprecated path |
| Perf overhead | ~10-20% from NFS translation | Native kext speed |

The `fuser` crate links against `libfuse.dylib` at runtime. On macOS with FUSE-T installed,
this resolves to FUSE-T's compatibility library. No conditional compilation needed in our
code -- `fuser::mount2()` works identically on both platforms.

**Build-time check**: The client feature should include a build script that verifies FUSE
headers are available (Linux: `libfuse3-dev`; macOS FUSE-T deferred to v2) and emits a clear error message
if not.

## Mapping to motlie-vmm Design Doc

This section documents how motlie-vfs components map to the motlie-vmm architecture
(sections 10-12 of `motlie/docs/motlie-vmm.md`).

### What the Library Extracts

| motlie-vmm concept | motlie-vfs component | Notes |
|---|---|---|
| vsock FS server (~800 lines, §10) | `FsServer` (core) | Same job: tag→host_path routing, inode table, host FS ops |
| `fs_loop(stream, vm, &tag)` in §10 | `VsockConnectionHandler::serve()` | Extracted + parameterized over stream type |
| `FsServer::add_mount()` / `remove_mount()` in §11 | `FsServer::add_mount()` / `remove_mount()` | Identical interface |
| Guest agent FUSE↔vsock bridge (~700 lines, §10-11) | `GuestMountRunner` + `FuseClient` + `VsockClientTransport` | `GuestMountRunner` owns guest-side mount orchestration; `FuseClient` owns `fuser::Filesystem`; vsock module provides transport |
| `VsockFuse::new(HOST_CID, VMM_PORT, &tag, read_only)` in §11 | `GuestMountRunner::mount(...)` using `VsockClientTransport::new(stream, tag)` + `FuseClient::new(transport, tag)` | Library mount runner takes established streams / connectors, not raw VMM protocol concerns |
| Inline policy checks in §12 (`if is_cred { ... }`) | `PolicyFn` trait | Formalized into trait |
| `AuditedFsServer` wrapping `FsServer` in §12 | `FsServer` with `.events()` builder | Events built into core, domain context added by caller |

### What Stays in motlie-vmm

| motlie-vmm concept | Why not extracted |
|---|---|
| `HandshakeMsg` enum (`BinaryRequest`, `Control`, `Fs`) | Multiplexer-level routing, not an FS concern |
| `ControlMsg` enum (`Ready`, `AddMount`, `RemoveMount`, `Shutdown`) | VM lifecycle control and guest orchestration; outside the `libs/vfs` crate |
| `AuditedFsServer` domain enrichment (vm_name, credential classification) | VMM-specific; subscribes to `FsEvent` and adds context |
| Bootstrap binary delivery (`BinaryRequest`) | VM-specific bootstrap |
| Tiny bootstrap binary in the guest image | VM-specific bootstrap and binary loading |
| Guest agent `main()` loop (§11) | Process/bootstrap orchestration: reads config, acquires streams, invokes `GuestMountRunner`, and coordinates with VM lifecycle |

## VMM Guest Integration: vsock + Overlay

This section documents how motlie-vfs integrates with the motlie-vmm guest agent over
vsock, including the full VM boot lifecycle and how the overlay enables selective in-memory
content injection for specific paths within a pass-through mount.

### v1 VM Target: vsock Only

For VM guests, the v1 target path is **vsock only**.

- the guest-side filesystem bridge uses `FuseClient`
- the vsock module provides the guest transport adapter used by `FuseClient`
- the host-side serving path uses `VsockConnectionHandler`
- this is the primary VM integration path that must be correct before broader transport work
- the transport-agnostic RPC composite remains useful for non-VM or standalone mounts, but it is
  not required to deliver the first VM-backed product

This keeps the initial guest target aligned with the existing VMM architecture and removes
unnecessary transport work from the critical path for VM delivery.

### Fastest Dev/Test Harness Before a Full VMM: Cloud Hypervisor

The fastest path to validate the v1 VM design is to use **Cloud Hypervisor (CH)** as a thin
development harness instead of building the full `motlie-vmm` orchestration stack first.

Recommended approach:

- boot a minimal Linux guest with the same root model assumed by this DESIGN:
  read-only squashfs base plus writable ext4 overlay
- enable a vsock device between host and guest
- run the real guest-side mounter binary (`motlie-vfs-guest`) in the guest, either directly
  from the test image or via a tiny bootstrap owned by the VMM layer:
  - reads a static mount config prepared ahead of boot
  - acquires one or more connected vsock streams from VMM-owned handshake/bootstrap logic
  - invokes `client::guest::GuestMountRunner` to mount the configured guest paths
- run `FsServer` + `MemOverlay` on the host in a simple Tokio process
- inject files directly through `server.overlay()` or the example REPL/admin loop
- validate guest-visible behavior without implementing the full VMM control plane,
  SSH bridge, or binary-delivery/bootstrap flow

What CH needs to provide for this harness:

- kernel + initrd or disk boot for the guest
- virtio-block devices for the squashfs/ext4 stacked root model
- a vsock device for host↔guest transport
- enough guest configuration to start the bootstrap/agent process

What the CH harness intentionally avoids in v1:

- custom VMM multiplexer/control protocol
- dynamic guest binary delivery
- full VM lifecycle management
- production orchestration concerns

The goal of the CH harness is not to replace `motlie-vmm`. It is to prove the core stack
quickly:

- `FsServer`
- `MemOverlay`
- `VsockConnectionHandler`
- `VsockClientTransport`
- `FuseClient`
- guest-visible mount and overlay semantics

### v1 CH Harness: Required Setup and Test Procedure

The v1 scope includes enough operational setup to let developers and CI prove the guest-vsock
path end-to-end without a full VMM. At minimum, the repo should provide documented scripts or
CLI procedures for the following.

**1. VM image building for SSH-capable guest testing**

The test guest image should include:

- the same stacked root model used by this DESIGN:
  - read-only squashfs base
  - writable ext4 overlay
- a minimal init/bootstrap path that mounts the stacked root
- the built `motlie-vfs-guest` binary, or a tiny bootstrap that execs it, capable of:
  - reading a static mount config
  - obtaining connected streams from VMM-owned bootstrap/handshake logic
  - invoking `GuestMountRunner`
- enough userland to validate guest behavior:
  - `sshd` or equivalent for guest access
  - `mount`, `stat`, `ls`, `cat`, `touch`, `rm`
  - test user provisioning such as `/home/alice`

The exact image-builder implementation can vary, but the v1 contract is that the repo must
include a reproducible way to:

- build `cross build --release --target x86_64-unknown-linux-musl --features vsock,client -p motlie-vfs --bin motlie-vfs-guest`
- stage or embed that binary into the guest image artifacts
- build or refresh the guest image for local testing

**2. Cloud Hypervisor guest launch procedure**

The repo should include documented scripts or CLI commands to:

- start Cloud Hypervisor with:
  - the kernel + root artifacts for the guest
  - virtio-block devices for the squashfs/ext4 stack
  - a vsock device
  - networking sufficient for SSH-based guest validation, when SSH testing is enabled
- pass any guest configuration needed by the bootstrap/agent
- stop and clean up the guest after tests

This can be a shell script, `just` target, `cargo xtask`, or similar. The important part is
that the procedure is explicit and reproducible.

**3. Host-side motlie-vfs server setup**

The repo should include a documented procedure or script that:

- creates the host backing directories used for test mounts
- starts an `FsServer` with the intended mount tags
- enables `MemOverlay`
- starts the vsock serving path
- starts the host example REPL / admin loop for manual mutation testing

For example, a basic host test setup may create:

- a host directory to back guest `/home`
- a subdirectory such as `./testdata/home/alice`
- sample disk-backed files that can later be shadowed or whiteouted by the overlay

**4. Overlay mutation procedures using the v1 embedded admin workflow**

The documented v1 test path should include concrete embedded-admin examples for:

- injecting a file:
  - `put credentials alice-home /.ssh/authorized_keys ...`
- injecting with explicit ownership:
  - `putattr credentials alice-home /.ssh/authorized_keys 1000 1000 0600 ...`
- creating a tombstone/whiteout:
  - `whiteout credentials alice-home /.ssh/old_config`
- removing an injected entry:
  - `rm credentials alice-home /.ssh/config`
- removing an entire layer:
  - `rmlayer credentials`
- inspecting effective overlay state:
  - `ls alice-home /.ssh`
  - `lslayer credentials alice-home`

These procedures must cover both:

- synthetic paths such as `/.claude/skills/...`
- shadowing/tombstoning files already present in the mounted disk-backed tree

**5. Guest-side validation**

The documented v1 validation flow should include at least:

- verifying the mount is active at the expected guest path
- verifying an injected file becomes visible without reboot or remount
- verifying a whiteout hides a lower disk file
- verifying `rm` of an injected synthetic file makes it disappear
- verifying ownership and mode on synthetic entries with `stat`
- verifying SSH-related guest behavior when the test image includes `sshd`

This operational procedure is in addition to:

- unit tests
- direct-mode integration tests
- duplex transport harnesses
- environment-gated automated integration tests

### VM Boot Lifecycle (from Firecracker to FUSE mounts)

The guest agent is the only custom binary inside the VM. It arrives over vsock at boot,
never touches disk, and uses motlie-vfs for all FUSE operations. The boot sequence:

```
Host (motlie-vmm daemon)                     Guest (Firecracker VM)
─────────────────────────                    ───────────────────────
ssh alice@:2222 arrives
  │
ensure_vm("alice")
  │
  ├─ create overlay.ext4
  │   inject: CA keys, principals,
  │   mounts.yaml, env
  │
  ├─ spawn passt (memfd, in netns)
  ├─ spawn firecracker (memfd, in netns)
  │   kernel cmdline:
  │     init=/sbin/overlay-init
  │     overlay_root=vdb
  │                                          Firecracker boots kernel
  │                                            │
  │                                          overlay-init (PID 1, shell)
  │                                            mount squashfs (ro base)
  │                                            mount ext4 (rw, config only)
  │                                            mount -t overlay (stacked)
  │                                            pivot_root → merged
  │                                            exec /sbin/init
  │                                            │
  │                                          systemd (PID 1)
  │                                            │
  │                                            ├─ motlie-vmm-guest.service
  │                                            │    ExecStart=motlie-vmm-bootstrap
  │                                            │      │
  ├─ vsock listener (port 5000)  ◄─────────────│      ├─ connect vsock:5000
  │   HandshakeMsg::BinaryRequest              │      │   handshake: BinaryRequest
  │   → send guest agent binary (~3MB)         │      │   receive binary → /tmp (tmpfs)
  │                                            │      │   exec into guest agent (same PID)
  │                                            │      │
  ├─ FsServer ready  ◄────────────────────────│      ├─ read /etc/motlie-vmm/mounts.yaml
  │   (mounts registered, overlay layers set)  │      │
  │                                            │      ├─ for each mount in config:
  │   HandshakeMsg::Fs { tag }  ◄──────────────│      │   connect vsock:5000
  │   → VsockConnectionHandler::serve()        │      │   handshake: Fs { tag }
  │                                            │      │   VsockClientTransport::new(stream, tag)
  │                                            │      │   FuseClient::new(transport, tag)
  │                                            │      │   fuser::mount2(mount, path, opts)
  │                                            │      │
  │   HandshakeMsg::Control  ◄─────────────────│      ├─ connect vsock:5000
  │   → control loop                           │      │   handshake: Control
  │                                            │      │   send ControlMsg::Ready
  │                                            │      │   listen for AddMount/RemoveMount
  │                                            │      │
  │                                            │      └─ FUSE mounts active
  │                                            │
  │                                            └─ sshd.service (CA-based auth)
  │
  └─ VM ready, bridge SSH session
```

### VMM Storage Model: virtio-block + stacked root

The guest boot/storage model remains the same as in `motlie-vmm`: a read-only squashfs
base plus a writable ext4 overlay, stacked inside the guest by `overlay-init`.

This design assumes the guest sees these inputs as ordinary **virtio-block devices**:

- one block device providing the read-only base image consumed by the squashfs mount
- one block device providing the writable ext4 overlay used for per-VM config/state

The stacking model is intentionally **independent of the host-side image container format**.
Whether the VMM stores or prepares those block devices as **raw** images or **qcow2** images
is outside `motlie-vfs` itself and belongs to the VMM/image-build layer. The requirement at
this boundary is only that the guest is presented with standard block devices containing:

- a squashfs read-only root payload
- an ext4 writable overlay payload

So, from the perspective of this DESIGN:

- **same squashfs + ext4 stacking**: yes, required
- **virtio-block presentation to the guest**: yes, assumed
- **raw vs qcow2 host backing**: either is acceptable, as long as the VMM presents the guest
  with the same effective virtio-block devices and boot flow

This keeps `motlie-vfs` decoupled from image-container details while preserving compatibility
with the existing `motlie-vmm` boot architecture.

### Guest-Side API Boundary

The guest-side reusable library boundary in `libs/vfs` is:

- `VsockClientTransport`: request/response transport over an already-established guest stream
- `FuseClient`: the guest-side `fuser::Filesystem` implementation over a transport
- `GuestMountRunner`: guest-side orchestration that takes mount specs plus stream/connect helpers
  and mounts one or more guest-visible filesystems

This means:

- `bins/v1/motlie-vfs-guest.rs` should be a thin binary over `client::guest`
- `bins/v1.1/motlie-vfs-guest.rs` should do the same, adding only the v1.1 tag handshake before transport use
- a future `motlie-vmm-guest` binary should also be a thin wrapper over `client::guest`
- the tiny bootstrap binary, binary-delivery path, and VMM handshake multiplexer remain outside
  `libs/vfs`

`GuestMountRunner` contract for v1:

- `mount_all()` spawns one thread per `GuestMountSpec`
- each thread obtains a transport via the caller-supplied connector, constructs `FuseClient`,
  and then calls `fuser::mount2()`
- `mount_all()` returns after all mount threads are started, with a handle set the caller can
  manage or join later
- the caller still owns any control-loop or process lifecycle above the started mounts

### Guest Agent Code with motlie-vfs

The guest agent (~700 lines in the VMM doc) should depend on the public guest-side API rather
than reimplementing mount orchestration inline. With the library, the guest-mount logic becomes:

```rust
// bins/v1/motlie-vfs-guest.rs, bins/v1.1/motlie-vfs-guest.rs, or
// motlie-vmm-guest main.rs (inside the VM)
use motlie_vfs::client::guest::{GuestMountRunner, GuestMountSpec};
use motlie_vfs::vsock::VsockClientTransport;

const HOST_CID: u32 = 2;
const VMM_PORT: u32 = 5000;

fn main() {
    let config: MountConfig = read_yaml("/etc/motlie-vmm/mounts.yaml");
    let specs: Vec<GuestMountSpec> = config.mounts.into_iter()
        .map(|m| GuestMountSpec::new(m.tag, m.guest_path).read_only(m.read_only))
        .collect();

    let runner = GuestMountRunner::new(specs);
    runner.mount_all(|tag| {
        // VMM-owned connect + handshake remain outside the library.
        let mut stream = vsock_connect(HOST_CID, VMM_PORT)?;
        send_handshake(&mut stream, &HandshakeMsg::Fs { tag: tag.to_string() })?;
        Ok(VsockClientTransport::new(stream, tag))
    })?;
}
```

The boundary is clear:
- **motlie-vfs** owns: `GuestMountRunner`, `FuseClient` (guest-side `fuser` bridge),
  `VsockClientTransport`, frame encoding, `fuser` integration
- **motlie-vmm guest** owns: process/bootstrap orchestration, config loading, vsock connect,
  `HandshakeMsg` handshake, and invoking the library guest APIs
- **motlie-vmm host** owns: vsock listener, `HandshakeMsg` dispatch, `FsServer` + overlay setup,
  and any VMM control protocol

### Host-Side Setup: FsServer + Overlay for a VM

On the host side, the VMM daemon creates the `FsServer` with mounts and overlay content
before the guest agent connects. This is where the in-memory overlay is used to inject
content that should never exist on disk.

```rust
// Inside motlie-vmm daemon, during ensure_vm("alice"):

use motlie_vfs::core::FsServer;
use motlie_vfs::vsock::VsockConnectionHandler;

// Build FsServer for this user's VM
let server = FsServer::builder()
    .mount("home", home_dir.join("alice"), false)            // /root → ~/alice
    .mount("scratch", scratch_dir.join("alice"), false)      // /tmp
    .overlay(true)
    .events(4096)
    .policy(credential_policy)
    .build()?;

// Inject SSH keys into the "home" mount's overlay.
// On disk, ~/alice/.ssh/ may not exist or may be empty.
// The overlay makes these files appear at /root/.ssh/ inside the VM.
// If ~/alice/.ssh/ doesn't exist on disk, the server creates a synthetic
// directory inode for /.ssh/ implicitly from the child entries below.
let overlay = server.overlay().unwrap();
overlay.put_layer("credentials", 0)?;
overlay.put("credentials", "home", "/.ssh/authorized_keys", alice_pubkey)?;
overlay.put("credentials", "home", "/.ssh/config", ssh_config_bytes)?;
overlay.put("credentials", "home", "/.ssh/id_ed25519", alice_private_key)?;
overlay.put("credentials", "home", "/.ssh/id_ed25519.pub", alice_public_key)?;

// API keys as a synthetic .env file
overlay.put("credentials", "home", "/.env", Bytes::from(format!(
    "ANTHROPIC_API_KEY={}\nOPENAI_API_KEY={}",
    anthropic_key, openai_key,
)))?;

// Write mounts.yaml for guest agent to read from overlay ext4
let mounts_config = MountConfig {
    mounts: vec![
        MountEntry { tag: "home", guest_path: "/root", read_only: false },
        MountEntry { tag: "scratch", guest_path: "/tmp", read_only: false },
    ],
};
inject_into_overlay(&overlay_ext4, "/etc/motlie-vmm/mounts.yaml", &mounts_config)?;

// Start vsock listener — one FsServer per guest VM, one socket per VM.
// For multiple guests, the VMM daemon creates separate FsServer instances
// with separate vsock sockets parameterized by CID or VM name.
let server = Arc::new(server);
tokio::spawn({
    let server = server.clone();
    async move {
        loop {
            let stream = vsock_accept(cid, 5000).await?;
            let handshake: HandshakeMsg = recv_handshake(&mut stream).await?;
            match handshake {
                HandshakeMsg::Fs { tag } => {
                    let handler = VsockConnectionHandler::new(&server, &tag);
                    tokio::spawn(async move { handler.serve(stream).await });
                }
                HandshakeMsg::BinaryRequest => { /* send guest agent binary */ }
                HandshakeMsg::Control => { /* control loop */ }
            }
        }
    }
});
```

### Example: SSH Keys in Overlay, Home Directory on Disk

This example shows the motivating use case for the overlay: the guest mounts the user's
entire home directory from the host via FUSE pass-through, but the `.ssh` subdirectory
is populated from an in-memory overlay. The SSH private keys never exist on the host
filesystem — they're injected by the VMM daemon from a secure source (vault, env var,
per-session generation).

**What the guest sees at /root:**

```
/root/                          ← FUSE mount, tag "home" → ~/alice on host
├── projects/                   ← pass-through to ~/alice/projects (disk)
├── .config/                    ← pass-through to ~/alice/.config (disk)
├── .bashrc                     ← pass-through to ~/alice/.bashrc (disk)
├── .ssh/                       ← IN-MEMORY OVERLAY (credentials layer)
│   ├── authorized_keys         ← overlay: never on disk
│   ├── config                  ← overlay: never on disk
│   ├── id_ed25519              ← overlay: never on disk
│   ├── id_ed25519.pub          ← overlay: never on disk
│   └── known_hosts             ← overlay: initially empty, writes captured in-memory
├── .env                        ← IN-MEMORY OVERLAY (synthetic file, not on disk)
└── documents/                  ← pass-through to ~/alice/documents (disk)
```

**How handle_op() resolves these paths:**

```
read("/root/projects/README.md")
  → walk layer stack for tag "home" from top to bottom
  → memfs layers: no entry
  → base disk layer: ~/alice/projects/README.md found
  → return bytes from base layer

read("/root/.ssh/id_ed25519")
  → walk layer stack for tag "home" from top to bottom
  → "credentials" memfs layer: Content(key_bytes)
  → return key_bytes from memfs layer                             ← in-memory, never touches base layer

read("/root/.env")
  → walk layer stack for tag "home" from top to bottom
  → "credentials" memfs layer: Content(env_bytes)
  → return env_bytes from memfs layer                             ← synthetic file, no base-layer backing required

write("/root/.ssh/known_hosts", new_entry)
  → visible owner is the "credentials" memfs layer
  → update Content entry in "credentials" layer                   ← base layer untouched

write("/root/projects/src/main.rs", code)
  → walk layer stack for tag "home" from top to bottom
  → memfs layers: no entry
  → base disk layer owns the path
  → write through base layer                                      ← in v1 this reaches disk

ls("/root/")
  → start from base disk layer entries: [projects, .config, .bashrc, documents, ...]
  → apply memfs layers for "home" at depth 1: [.ssh (SyntheticDir), .env (Content)]
  → merge per-name: .ssh exists on both? upper memfs SyntheticDir wins
  →                  .env only in memfs? synthetic entry added
  → policy filter: all pass (no policy on /root/)
  → result: [projects, .config, .bashrc, documents, .ssh, .env, ...]

ls("/root/.ssh/")  — Case A: ~/alice/.ssh/ does NOT exist on disk (synthetic parent):
  → walk stack: upper memfs layer resolves `/.ssh` as SyntheticDir
  → effective directory is owned by upper memfs layer
  → effective children at /.ssh/*: [id_ed25519, id_ed25519.pub, config, authorized_keys]
  → result: [id_ed25519, id_ed25519.pub, config, authorized_keys]
     (pure memfs directory — no disk entries to merge or filter)

ls("/root/.ssh/")  — Case B: ~/alice/.ssh/ exists on disk (disk+overlay merge):
  → base disk layer entries: [known_hosts, old_config]
  → upper memfs entries at /.ssh/*: [id_ed25519, id_ed25519.pub, config, authorized_keys]
  → merge per-name: upper memfs entries + base-layer entries

  Case B without policy (unfiltered):
  → result: [known_hosts, old_config, id_ed25519, id_ed25519.pub, config, authorized_keys]
     (base-layer files pass through alongside memfs entries)

  Case B with SshLockdown policy:
  → policy filter: SshLockdown denies disk-only entries (known_hosts, old_config)
  → result: [id_ed25519, id_ed25519.pub, config, authorized_keys]
     (disk files excluded by policy)
```

**readdir merging with layer-stack semantics and policy filtering:**

When `handle_op()` processes a `Readdir`, it resolves and merges through the ordered layer stack
and then filters:

1. Start from the lowest layer’s direct children for the queried directory.
   In v1 this is the disk-backed base layer, unless the effective directory is fully synthetic
   above it.
2. Apply each higher layer in stack order.
3. Merge by name:
   - `Content` and `SyntheticDir` add or replace
   - `Whiteout` removes the name from the effective set
4. **Policy filter:** each merged entry is passed through `policy.check(Readdir, tag, path)`.
   Entries denied by policy are excluded from the result. This enables directory-level
   lockdown without modifying the stack semantics.

### Overlay Scope: File-Level Granularity

The memfs layer model operates at **file granularity**. Each `put()` inserts a file node in a
named memfs layer for a specific `(tag, path)`. `readdir` behavior depends on the effective
lower-layer state for that directory:

```
Case A: synthetic directory (host dir does not exist):
  readdir("/.ssh/") where ~/alice/.ssh/ is absent on disk
  → directory is SyntheticDir in overlay → only overlay children returned
  → result: [id_ed25519, id_ed25519.pub, config, authorized_keys]
  (no disk entries to merge — pure memfs directory)

Case B: base-layer directory with memfs children:
  readdir("/.ssh/") where ~/alice/.ssh/ exists on disk
  → base-layer readdir: [known_hosts, old_config]
  → memfs children: [id_ed25519, id_ed25519.pub, config, authorized_keys]
  → per-name merge: upper memfs entries shadow same-name base entries; base-only entries pass through
  → policy filter (step 6): each entry checked, denied entries excluded
  → without policy: [known_hosts, old_config, id_ed25519, id_ed25519.pub, config, authorized_keys]
  → with SshLockdown: [id_ed25519, id_ed25519.pub, config, authorized_keys]
```

In Case A, no policy is needed — the directory is entirely overlay-managed. In Case B,
base-layer files pass through by default; use policy to enforce isolation if needed.

**For credential paths, use policy to enforce isolation:**

The unfiltered baseline above shows disk files leaking into `.ssh/`. For credential
isolation, callers should combine the overlay with a lockdown policy. The overlay
provides content injection; the policy enforces visibility. `policy.check()` is called
for each `Readdir` entry (step 6 in the merge logic), so denied entries are excluded
from both directory listings and direct access:

```rust
struct SshLockdown { overlay: Arc<MemOverlay> }

impl PolicyFn for SshLockdown {
    fn check(&self, op: FsOp, tag: &str, path: &str) -> Result<(), i32> {
        // In /.ssh/: only serve files that exist in the overlay.
        // This filters both direct access (Lookup, Read, Open) AND
        // Readdir entries — disk files in .ssh/ are invisible.
        if path.starts_with("/.ssh/") && self.overlay.resolve(tag, path).is_none() {
            return Err(libc::ENOENT);
        }
        Ok(())
    }
}
```

With this policy, `readdir("/.ssh/")` returns only overlay entries — disk files like
`known_hosts` or `old_config` are filtered out. Without this policy, they pass through.
The overlay always merges; the policy decides what's visible.

This separates concerns cleanly: overlay handles content injection and merging; policy
handles access control. They compose independently.

### Credential Lifecycle

```
VM create:
  1. VMM daemon generates or fetches credentials (SSH keys, API tokens)
  2. overlay.put("credentials", "home", "/.ssh/id_ed25519", key) — in memory only
  3. Guest agent boots, mounts /root via FUSE
  4. SSH inside the VM reads /root/.ssh/id_ed25519 → overlay serves it from memory
  5. Credentials never touch host disk or guest disk

VM stop:
  6. FsServer dropped → overlay dropped → credentials gone from memory
  7. Host dir ~/alice/ unchanged (no .ssh/ written)
  8. No cleanup needed — nothing was persisted

VM restart:
  9. VMM daemon re-creates FsServer, re-injects credentials into overlay
  10. Guest sees the same /root/.ssh/ as before
```

This is a significant improvement over the original VMM design, which used separate FUSE
mounts for each credential directory (`.claude/`, `.config/gh/`, `.codex/`, `.npmrc`).
With the overlay, a single "home" mount serves the entire home directory, and credentials
are injected as overlay entries within it. Fewer mounts, fewer vsock connections, simpler
guest agent, and the credentials never touch any filesystem.

### Comparison: Original VMM Design vs. Overlay Model

| Aspect | Original (separate mounts) | Overlay model |
|--------|---------------------------|---------------|
| Mount count per VM | 6+ (workspace, scratch, 4 cred dirs) | 2 (home, scratch) |
| vsock connections per VM | 6+ (one per mount) | 2 (one per mount) |
| Credential storage | Host disk (`~/.motlie-vmm/creds/`) | In-memory overlay (never on disk) |
| Credential persistence | Survives VM restart (on-disk) | Must be re-injected on restart |
| Guest agent complexity | One thread per mount | Same, but fewer mounts |
| Adding a new cred dir | New mount tag, new vsock connection | `overlay.put()` call |
| Readdir at /root | Only shows disk files | Merges disk + overlay entries |

The overlay model trades credential persistence (which the original had via on-disk cred
dirs) for stronger security (credentials never on disk). For OAuth tokens that refresh
automatically, this is fine — the VMM daemon re-injects them from a secure store on each
VM boot. For API keys set via environment variables, the overlay injects them as a
synthetic `.env` file.

## Alternatives Considered

### Alternative A: 9P2000.L Protocol

Use the Plan 9 filesystem protocol (Linux variant) instead of a custom wire format.

**Pros:**
- Mature, well-specified protocol with existing implementations
- Linux kernel has a native 9P client (`mount -t 9p`) -- no FUSE needed on Linux
- QEMU/virtio-9p uses it for host directory sharing

**Cons:**
- No native macOS kernel client -- would still need FUSE on macOS, defeating the "one
  protocol" goal
- 9P's walk/attach/clunk semantics are more complex than needed for our tag-based model
- No natural event emission point -- 9P servers are typically transparent passthrough
- Existing Rust 9P libraries (jmbaur/p9, oxidecomputer/p9) are incomplete or dormant
- Protocol overhead: 9P's per-message type headers and walk chains add unnecessary bytes
  for our use case where the client already knows the mount root

**Verdict:** Rejected. The native Linux kernel client is appealing but the macOS gap
means we'd need two code paths. Our workload (known mount points, event interception)
is better served by a thinner custom protocol.

### Alternative B: FUSE Passthrough (virtio-fs model)

Forward raw FUSE protocol messages over the transport, as virtiofsd does with virtio-fs.

**Pros:**
- Zero translation overhead -- FUSE messages pass through unmodified
- Proven approach in production (virtiofsd, kata containers)
- Potentially simpler client (just forward /dev/fuse messages)

**Cons:**
- FUSE protocol is Linux-specific -- message format tied to kernel version and platform.
  macOS FUSE-T uses a different internal protocol (NFSv4). Would need platform-specific
  server-side decoding, which defeats the purpose.
- FUSE protocol messages are not self-describing without kernel version context
- Event emission requires parsing FUSE messages on the server side anyway
- `fuser` already handles the /dev/fuse ↔ Rust translation -- using it on the client side
  and speaking a clean protocol over the wire is simpler than raw FUSE passthrough
- No support for pluggable encoding -- locked to FUSE wire format

**Verdict:** Rejected. The platform asymmetry (Linux FUSE vs macOS FUSE-T internals) makes
raw passthrough impractical for cross-platform use. The `fuser` crate already absorbs the
platform differences; our custom protocol sits cleanly above it.

### Alternative C: Custom Protocol (Selected)

A purpose-built, minimal FS protocol with length-prefixed frames, pluggable encoding, and
tag-based routing. The client translates FUSE ops to protocol frames; the server translates
frames to host FS ops.

**Pros:**
- Thinnest possible protocol for our use case
- Natural event emission point (server decodes every frame)
- Tag-based routing built into the handshake
- Pluggable encoding (bincode now, msgpack or protobuf later)
- Platform-agnostic (no Linux or macOS kernel assumptions in the wire format)
- Transport-agnostic (any AsyncRead+AsyncWrite)
- Composable: vsock composite skips Frame overhead entirely for the VM path

**Cons:**
- Custom protocol means custom bugs -- no battle-tested implementations to lean on
- Must implement every FS operation ourselves (9P and FUSE passthrough get some for free)
- No interop with non-Rust clients without reimplementing the protocol

**Verdict:** Selected. The cons are manageable: the protocol is small (~15 operations),
the server is straightforward (translate to `std::fs` calls), and non-Rust interop is not
a current requirement. The event emission and policy enforcement capabilities, which are
core to the motlie-vmm use case, fit naturally into this model. The composable architecture
means the VM path (vsock) pays no overhead for the RPC protocol layer it doesn't use.

### Alternative D: Build on the `vfs` Crate

Use the existing Rust `vfs` crate as the base abstraction instead of implementing a custom
mount-stack and memfs model.

Evaluation summary:

- attractive because it already provides:
  - generic virtual filesystem traits
  - in-memory and physical filesystem implementations
  - an overlay-style abstraction
- not selected because it does **not** match this DESIGN closely enough

Reasons not chosen:

- this DESIGN needs ordered per-tag stacks with:
  - named memfs layers
  - `(layer, tag, path)` identity
  - whiteout semantics
  - shared named layers spanning many tags
  - unified inode/generation behavior for the FUSE-facing surface
- the `vfs` crate is shaped more as a generic VFS abstraction than as a transactional,
  snapshot-published memfs layering engine
- the required concurrency contract here is batch-first atomic memfs publication per tag;
  that is not the design center of `vfs`
- adopting `vfs` would likely force adaptation layers or internal workarounds that are more
  complex than a focused custom implementation

**Verdict:** Rejected as a foundational dependency. It may be useful as abstraction inspiration,
but it is not a close fit for the required stack semantics, concurrency contract, or FUSE-facing
inode/generation model.

### Alternative E: Use Theseus OS `memfs`

Use the `memfs` project from Theseus OS as the in-memory filesystem implementation.

Evaluation summary:

- this is **not** the same thing as the “memfs layer” described in this DESIGN
- the Theseus project implements an in-memory filesystem for the Theseus OS environment

Reasons not chosen:

- it is tied to the Theseus OS ecosystem and design assumptions
- it is not a drop-in implementation of this library’s required ordered layer-stack semantics
- it does not solve the per-tag snapshot publication and batch atomicity contract required here
- it would create naming confusion with this DESIGN’s use of “memfs layer” to mean the in-memory
  stack layers managed by `motlie-vfs`

**Verdict:** Rejected as unrelated to the implementation strategy for this crate.

### Alternative F: Reuse an Existing Memfs/Overlay Engine Wholesale

Adopt a third-party memfs or overlay engine and wrap it with the needed transport and FUSE code.

Evaluation summary:

- appealing in principle because it could reduce custom data-structure work
- not selected because the required semantics here are unusually specific:
  - ordered stacks per mount tag
  - shared named layers across tags
  - batch-first atomic memfs publication
  - whiteouts and synthetic parents in one coherent inode namespace
  - disk-backed base in v1, but future non-disk base compatibility

**Verdict:** Rejected for v1. A small custom memfs stack implementation using focused helper
crates is lower risk than adapting a generic engine that does not share the same semantic center.

## Components and Testing

### Components to Test

| Component | What to test | Method |
|-----------|-------------|--------|
| `core::op` | FsOp/FsResult serde round-trip | Unit tests |
| `core::server` | handle_op() for all FsOp variants against real tempdir | Integration tests with `tempfile` |
| `core::inode` | Inode allocation, lookup, refcount, eviction | Unit tests |
| `core::event` | Event emission on FS ops, no-subscriber zero-cost | Unit + integration |
| `core::policy` | Deny returns correct errno, allow passes through | Unit tests |
| `core::overlay` | Layer priority, put/get/remove, resolve order, synthetic files in readdir | Unit tests |
| `core::overlay` | Write capture to overlaid path updates layer, not disk | Integration tests with `tempfile` |
| `vsock::handler` | VsockConnectionHandler serve loop over duplex | Integration tests |
| `vsock::client` | VsockClientTransport request/response behavior over duplex | Unit + integration tests |
| `client::fuse` | FuseClient translates FUSE ops correctly over an injected transport | Unit tests (mock transport) |
| End-to-end (vsock) | VsockConnectionHandler + VsockClientTransport + FuseClient over duplex, real FUSE mount | Integration (requires FUSE) |
| Proof-of-concept examples | Host REPL + CH guest + vsock mount + overlay visibility | Manual + scripted integration |
| Cross-platform roadmap | macOS FUSE-T client path in later roadmap phases | Manual test (documented procedure) |

### Test Utilities

```rust
/// Direct testing: no transport, no serde.
#[cfg(test)]
let server = FsServer::builder().mount("t", dir, false).build()?;
let result = server.handle_op("t", FsOp::Lookup { parent: 1, name: "foo".into() });

/// vsock testing: same pattern, duplex stream.
#[cfg(test)]
pub fn vsock_test_pair(server: FsServer, tag: &str) -> (JoinHandle<()>, FuseClient<VsockClientTransport>) {
    let (cs, ss) = tokio::io::duplex(64 * 1024);
    let handler = VsockConnectionHandler::new(&server, tag);
    let handle = tokio::spawn(async move { handler.serve(ss).await.unwrap() });
    let transport = VsockClientTransport::new(cs, tag);
    let mount = FuseClient::new(transport, tag);
    (handle, mount)
}
```

## Dependencies

### Required (always compiled -- server-core)

| Crate | Purpose |
|-------|---------|
| `serde`, `serde_derive` | FsOp/FsResult serialization |
| `bytes` | Zero-copy byte buffers in data ops |
| `tokio` (features: io-util, sync, rt) | Async I/O, broadcast channel |
| `anyhow` | Operational error handling (workspace convention) |
| `tracing` | Structured logging |

### Feature-Gated

| Crate | Feature flag | Purpose |
|-------|-------------|---------|
| `bincode` | `bincode-codec` | Default wire encoding for the vsock path |
| `fuser` | `client` | FUSE filesystem implementation |
| `tokio-vsock` | `vsock` | vsock stream type |

### Dev Dependencies

| Crate | Purpose |
|-------|---------|
| `tempfile` | Temporary directories for server FS tests |
| `tokio` (features: test-util, macros) | Test runtime |
