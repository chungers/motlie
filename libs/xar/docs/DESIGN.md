# XAR Unified Artifact Distribution Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-12 | @codex-xar: Initial greenfield design for packaging curated `libs/models` bundle descriptors, concrete backends, and model artifacts into a single distributable executable. Migration and backward compatibility are explicitly out of scope. | All |
| 2026-04-12 | @codex-xar: Added an OCI-as-internal-format addendum evaluating OCI image layout and manifests inside the appended executable payload, including the hybrid raw-weight-blob idea and its implications for future mmap-capable backend contracts. | Option Comparison, OCI Addendum, Recommendation, References |
| 2026-04-12 | @codex-xar: Refined the OCI addendum into a concrete hybrid layout with a Motlie binary preamble TOC before the OCI payload and a footer pointing back to it. Added the three-tier materialization strategy and API sketches for preamble parsing and runtime blob resolution. | OCI Addendum, Recommendation, API Sketches |
| 2026-04-12 | @codex-xar: Moved this design into `libs/xar/docs/DESIGN.md` and generalized scope from model-only distribution to the unified packaging format for large Motlie artifacts, including model weights, checkpoints, kernels, initrds, rootfs images, and VM disk images. | Title, Overview, Scope, Recommendation, API Sketches |

This document defines `libs/xar`, Motlie's unified distribution and packaging format for large artifacts. `xar` is not model-only. It is the common archive and executable-payload format for:

1. the launcher / application binary
2. model bundle descriptors and backend-linked model runtimes
3. model weights, checkpoints, tokenizers, and related metadata
4. VM guest boot artifacts such as kernels and initrds
5. guest root filesystems and disk images
6. future large artifact classes that benefit from one content-addressed packaging model

The goal is not merely "embed files into an exe". The goal is to give Motlie one artifact story for all large payloads while preserving current layering for consumers:

- `libs/xar` owns packaging, framing, payload lookup, and shared materialization rules
- `libs/model` remains the stable lifecycle and capability contract for model consumers
- `libs/models` remains the curator-owned bundle and artifact layer above `libs/model`
- future VM/VMM consumers can use the same packaged blobs without inventing a second large-artifact format
- distribution-specific logic stays above generic model backends and future VM boot/runtime layers

## Table of Contents

- [Overview](#overview)
- [Current Codebase Fit](#current-codebase-fit)
- [Requirements and Non-Goals](#requirements-and-non-goals)
- [Evaluation Criteria](#evaluation-criteria)
- [Option Comparison](#option-comparison)
- [Option 1: Giant Rust Binary via `include_bytes!`](#option-1-giant-rust-binary-via-include_bytes)
- [Option 2: Custom Container File with Footer TOC](#option-2-custom-container-file-with-footer-toc)
- [Option 3: Squashfs plus `memfd` Loader](#option-3-squashfs-plus-memfd-loader)
- [Option 4: Other Viable Options](#option-4-other-viable-options)
- [Addendum: OCI as Internal Payload Format](#addendum-oci-as-internal-payload-format)
- [Recommendation](#recommendation)
- [API Sketches](#api-sketches)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [References](#references)

---

## Overview

### Problem Statement

Today Motlie has a packaging asymmetry:

- model bundles are curated through `libs/models` and consumed through `libs/model`
- model artifacts are currently fetched or resolved through model-specific paths such as Hugging Face caches
- VM guest artifacts such as kernels, initrds, rootfs images, and disk images need the same distribution story, but should not force a second packaging format beside the model path

That split is workable for development, but it is not sufficient for a unified "single-file distribution" story where one executable or one package should be enough to launch:

- a vetted model bundle
- a VM guest image set
- or a mixed distribution containing both

The hard part is not descriptor registration. The hard part is that Motlie consumers still expect filesystem-shaped artifacts or path-based startup:

- Mistral-backed safetensors bundles expect a resolved directory tree
- llama.cpp-backed GGUF bundles expect a concrete GGUF file path
- guest boot stacks typically expect kernel/initrd/rootfs or disk image paths
- current model backends reject `StartOptions.unpack_root` as a direct backend concern rather than owning extraction themselves

Therefore, any single-file distribution design that preserves the present layering must solve one of two problems:

1. materialize local artifact roots before consumer startup, or
2. introduce richer lower-level contracts for file descriptors / byte ranges / virtual filesystems

For the first slice, the first approach is cheaper and fits the current code better.

## Current Codebase Fit

The concrete in-tree pressure today mostly comes from the model side. The relevant existing surfaces are:

- `libs/models/src/lib.rs`
  `ArtifactSource` currently has only `HuggingFace { repo }`
- `libs/models/src/lib.rs`
  `BundleArtifacts` describes curated include rules and download control
- `libs/model/src/lib.rs`
  `ArtifactPolicy` is currently `AllowFetch { root }` or `LocalOnly { root }`
- `libs/model/src/lib.rs`
  `StartOptions` already has `unpack_root: Option<PathBuf>`
- `libs/model/backends/mistral/*`
  startup currently rejects `unpack_root`
- `libs/model/backends/llama_cpp/*`
  startup currently rejects `unpack_root`

The VM side does not yet have the same unified packaging surface in-tree, which is exactly why `libs/xar` should exist as a shared crate rather than as another model-local document.

This leads to an important design conclusion:

- single-file distribution should not be implemented by teaching generic consumers about Hugging Face snapshots, executable offsets, squashfs images, or bundle-local packaging formats
- instead, consumer-specific crates such as `libs/models` and future VM loaders should resolve a packaged `xar` artifact source into normal local roots or additive raw-blob handles

That keeps the reviewed boundary intact.

## Requirements and Non-Goals

### Requirements

- Ship one executable or one archive format containing launcher code plus large artifacts
- Cover model weights, checkpoints, tokenizers, kernels, initrds, guest rootfs images, and VM disk images under one format
- Preserve the current `libs/model` to `libs/models` to backend layering for model consumers
- Leave room for future VM/VMM consumers to adopt the same package without reformatting artifacts
- Keep regulated local-only deployments able to fail closed
- Support multi-GB artifacts without pathological compile or link behavior
- Support curated validation before model or VM startup
- Support at least Linux and macOS cleanly; avoid painting the project into a Linux-only corner unless explicitly chosen
- Keep future room for multiple bundles, guest images, or mixed payloads in one distribution
- Leave room for additive integrity features such as per-file digests or signatures

### Non-Goals

- Migration from prior packaging formats
- Arbitrary user-supplied checkpoints
- Arbitrary user-supplied VM images in the first slice
- Full in-memory backend loading in the first slice
- Introducing a container runtime requirement just to run a local model or boot a local guest image

## Evaluation Criteria

Each option is evaluated on:

- build time impact
- binary size impact
- runtime overhead
- cross-platform support
- incremental update story
- integration with existing model-side `ArtifactSource` / `ArtifactPolicy` and future VM-side consumers
- engineering cost

## Option Comparison

| Option | Build Time | Binary Size | Runtime Overhead | Cross-Platform | Incremental Updates | Artifact Integration | Engineering Cost |
|------|------|------|------|------|------|------|------|
| 1. `include_bytes!` giant binary | Very high | Very high | Low to medium | Medium for the general technique, low if kept ELF-specific | Very poor | Awkward | Low initial, high ongoing |
| 2. Custom container appendix with footer TOC | Low to medium | Medium to high | Low to medium | High if implemented as generic appended payload | Medium | Strong | Medium |
| 3. squashfs + `memfd` loader | Medium | Low to medium | Low after mount, medium at startup | Low | Medium | Moderate | High |
| 4a. zstd-compressed appendix with manifest | Low to medium | Low to medium | Medium | High | Medium | Strong | Medium-low |
| 4b. OCI image / layers | Medium to high | Medium | High | Medium | Strong for registries, weak for single executables | Weak for current API | High |
| 4c. Native resource section / fat-binary tricks | High | Medium | Low | Low | Poor | Weak | High |

## Option 1: Giant Rust Binary via `include_bytes!`

### Shape

The direct approach is:

- convert each required artifact file into a build input
- embed it with `include_bytes!`
- place the payload into Rust-generated static data and let the linker emit large `.rodata` sections
- at runtime either:
  - write those bytes out to an `unpack_root`, or
  - redesign backends to load directly from in-memory bytes

### Pros

- conceptually simple
- no post-link packer required
- straightforward integrity story because the payload is part of the normal executable image
- works with normal Rust build tooling for small assets

### Cons

- poor fit for multi-GB weights
- every artifact change becomes a Rust compilation and link problem rather than a packaging problem
- incremental builds degrade badly because weight changes force relink of the final executable
- bundle-local file trees become awkward; `include_bytes!` gives bytes, not a directory
- the current backends still expect file paths, so the "zero-copy" story is mostly illusory in the present architecture
- if interpreted literally as "giant ELF binaries", this becomes Linux-specific

### Evaluation

- Build time impact:
  Very high. A model refresh relinks a multi-GB binary.
- Binary size:
  Essentially raw artifact size plus executable overhead. Compression is not automatic.
- Runtime overhead:
  Low only if a backend can consume the mapped bytes directly. With the current path-based backends, extraction is still needed, so real runtime overhead is medium.
- Cross-platform:
  `include_bytes!` itself is cross-platform, but "giant ELF" is not. The practical behavior of very large final executables also varies by linker and platform.
- Incremental update story:
  Very poor. Changing one tokenizer or one weight shard means rebuilding and redistributing the entire executable.
- Integration with `ArtifactSource` / `ArtifactPolicy`:
  Weak. `ArtifactSource` would need a new embedded variant, but the payload is not naturally discoverable or file-oriented. The cleanest integration still ends in `LocalOnly { root }` after extraction.
- Engineering cost:
  Low to prototype, high to live with.

### Codebase Impact

- add embedded artifact source variants in `libs/models`
- add a large amount of build-script glue
- likely add extraction helpers anyway because current backends need filesystem paths
- no meaningful simplification in the backend crates

### Verdict

Good as a toy or for small prompt templates, tokenizers, or metadata.
Bad as the primary vehicle for shipping real model weights.

## Option 2: Custom Container File with Footer TOC

### Shape

Build the normal executable first.
Then run a packer that appends a payload to the end of the executable:

- one footer with magic, version, TOC offset, TOC length, and digest
- one TOC describing bundles, files, offsets, encoding, lengths, and digests
- one or more data chunks, either raw or compressed

At runtime:

1. locate the current executable
2. seek from EOF to read the footer
3. validate the TOC and bundle payload digests
4. materialize the selected bundle into `unpack_root`
5. pass `ArtifactPolicy::LocalOnly { root }` to the curated bundle wrapper
6. let the existing backend start unchanged

### Pros

- keeps Rust compilation and artifact packaging separate
- supports multi-GB payloads without forcing them through the Rust compiler front-end
- clean place to add per-file digests, signatures, compression modes, and versioning
- can support multiple bundles in one executable
- can support partial extraction or on-demand extraction later
- fits the current architecture well because the bundle layer can materialize a normal directory tree before backend startup

### Cons

- requires a new packer and runtime reader
- requires explicit signing / notarization decisions after payload assembly
- requires careful validation that appended data is acceptable for every target executable format and distribution channel
- you own the format evolution

### Evaluation

- Build time impact:
  Low to medium. Compile the launcher normally; packaging is mostly file I/O plus optional compression.
- Binary size:
  Medium to high. Uncompressed payloads are large; compressed chunks can lower this materially.
- Runtime overhead:
  Low if using raw appended files and one-time extraction; medium if using compressed chunks.
- Cross-platform:
  Strong if implemented as a generic "appended payload" rather than ELF-only section surgery. The runtime only needs normal file seeking on the current executable path.
- Incremental update story:
  Medium. Still ships one large artifact, but weight refreshes do not require recompiling Rust code. A packer can also support chunk-level reuse later.
- Integration with `ArtifactSource` / `ArtifactPolicy`:
  Strong. Add a new `ArtifactSource::ExecutablePayload` in `libs/models`; after materialization, continue to use `ArtifactPolicy::LocalOnly { root }`.
- Engineering cost:
  Medium. One packer, one reader, one extraction cache policy, and some doc/test work.

### Codebase Impact

- `libs/models`
  add an executable-payload artifact source and a resolver that consumes `StartOptions.unpack_root`
- curated bundle modules
  call a shared `materialize_bundle_root(...)` helper before delegating to the inner backend bundle
- examples / bins
  add a release assembly tool that constructs the final executable image
- `libs/model`
  probably no breaking contract change required

### Verdict

This is the best architectural fit for Motlie's current layering.
It preserves the backend boundary and keeps packaging concerns out of `libs/model/backends/*`.

## Option 3: Squashfs plus `memfd` Loader

### Shape

Package the bundle artifact tree into a squashfs image.
Ship that image inside the executable or alongside a small appendix header.
At runtime:

1. locate the embedded squashfs image
2. copy or stream it into a `memfd`
3. mount the image at `unpack_root` using a kernel mount path or FUSE path
4. point the curated bundle at the mounted directory and continue with `LocalOnly`

### Pros

- natural fit for directory-shaped artifacts
- read-only compressed filesystem
- strong density for many small files plus large weights
- potentially lower disk amplification than eager extraction
- on-demand file access can reduce first-start extraction cost

### Cons

- this is effectively Linux-only
- `memfd_create()` is Linux-specific
- mounting squashfs requires kernel mount / loop / FUSE machinery, which is operationally more fragile than normal file extraction
- macOS support becomes a separate implementation problem
- bundle startup now depends on mount lifecycle, cleanup, and error handling

### Evaluation

- Build time impact:
  Medium. `mksquashfs` is an extra packaging stage.
- Binary size:
  Often good because squashfs compresses aggressively and packs metadata tightly.
- Runtime overhead:
  Medium at startup because the image must be staged and mounted; low afterward because reads are filesystem-native.
- Cross-platform:
  Low. The exact `memfd` plus squashfs mount story is Linux-specific.
- Incremental update story:
  Medium. Rebuild the image when artifacts change; no Rust relink needed.
- Integration with `ArtifactSource` / `ArtifactPolicy`:
  Moderate. The mounted path still maps to `LocalOnly`, but the runtime stack is much more complex and leaks mount concerns into operator behavior.
- Engineering cost:
  High. This is packaging plus virtual filesystem management plus platform policy.

### Codebase Impact

- add a Linux-specific mount helper and lifecycle manager
- decide whether mount failures are bundle-layer errors or launcher-level errors
- add cleanup logic for crashes and repeated starts
- either skip macOS entirely or implement a different distribution strategy there

### Verdict

Promising only if Motlie explicitly chooses a Linux-appliance distribution model.
Too specialized for the first general distribution slice.

## Option 4: Other Viable Options

### 4a. zstd-Compressed Appendix with Manifest

This is the most practical refinement of Option 2.

Instead of inventing a fully general container up front:

- append one manifest and one payload stream to the executable
- compress files or chunks with zstd
- keep a footer TOC with offsets, unpacked sizes, and digests
- always extract into `unpack_root` for the first slice

Pros:

- much smaller engineering surface than a true filesystem-in-an-exe design
- better size characteristics than raw appended files
- compatible with the current backend expectation of filesystem paths
- easy to version as `chunked-zstd-v1`

Cons:

- still requires extraction for current backends
- random access is only as good as the chosen chunking/index scheme
- multi-bundle dedupe across payloads is not automatic

Evaluation:

- Build time impact:
  Low to medium.
- Binary size:
  Better than raw appendix and usually better than `include_bytes!`.
- Runtime overhead:
  Medium because decompression happens before local-only startup.
- Cross-platform:
  High.
- Incremental update story:
  Medium, and can improve later with chunk-level reuse.
- Integration:
  Strong. This should be modeled as a concrete encoding under `ArtifactSource::ExecutablePayload`.
- Engineering cost:
  Medium-low relative to the other serious options.

Verdict:

Best first implementation profile.
This is the recommended first slice.

### 4b. OCI Image Layout or OCI Layers

OCI is attractive if the primary problem is registry distribution, dedupe, provenance, or reuse of container tooling.
It is much less attractive if the requirement is "single distributable executable".

Pros:

- existing ecosystem for layers, digests, provenance, and registries
- strong incremental distribution story in registry-backed workflows

Cons:

- not a natural single-executable runtime format
- usually implies an unpack step plus OCI runtime or custom importer
- adds a container model to a problem that currently wants a local bundle model

Evaluation:

- Build time impact:
  Medium to high.
- Binary size:
  Reasonable.
- Runtime overhead:
  High for this use case because the OCI layout is not directly consumable by the current backends.
- Cross-platform:
  Medium.
- Incremental updates:
  Strong in registry workflows, weak for offline single-binary delivery.
- Integration:
  Weak with current `ArtifactSource` / `ArtifactPolicy`; it would be better as an export/import format, not the in-process runtime package.
- Engineering cost:
  High.

Verdict:

Useful as a publication format, and more viable as an internal payload graph than this section originally gave it credit for.
On its own it is still incomplete for appended-executable runtime lookup; the addendum below covers the stronger hybrid design.

### 4c. Native Resource Sections or Fat-Binary Tricks

Examples include:

- PE resources on Windows
- Mach-O resources or universal-binary adjacent tricks on macOS
- ELF-specific section surgery

Pros:

- can look elegant on one platform

Cons:

- platform-specific
- poor portability story
- does not align with the current Linux plus macOS bundle requirements
- fat binaries solve architecture multiplexing, not general asset packaging

Verdict:

Not recommended except for platform-specific polish later.

## Addendum: OCI as Internal Payload Format

This section re-evaluates OCI not as the external distribution mechanism, but as the logical payload format inside the appended executable region:

```text
[ executable ][ preamble TOC ][ OCI image layout: manifest + config + blobs ][ footer ]
```

The key distinction is:

- OCI gives a standardized metadata and content-addressing model
- the appended executable still needs a Motlie-specific physical framing layer so the runtime can find the OCI payload and individual blobs by offset

So this is not "OCI instead of a footer".
It is "OCI plus a Motlie preamble/footer for physical layout".

### Hybrid Layout

The refined layout is:

```text
[ ELF / Mach-O executable ]
[ Preamble TOC: fixed-size header + blob offset table ]
[ OCI image layout payload: manifest + config + blobs ]
[ Footer: magic + offset back to preamble ]
```

This resolves the biggest weakness in the earlier OCI analysis:

- OCI handles semantics:
  media type, content identity, manifest graph, registry compatibility
- preamble handles physical placement:
  executable-local offsets, lengths, encoding, mmap eligibility, and fast startup lookup

The kernel loader still executes the file normally because ELF and Mach-O loaders ignore appended data beyond the mapped image.

### Why the Preamble Matters

Without a preamble, OCI inside an executable still leaves Motlie with an awkward question:

- how does the runtime find blob `sha256:...` at a physical byte range inside the current executable?

The preamble answers that directly.

Startup becomes:

1. read footer at EOF to discover `preamble_offset`
2. read the preamble header and blob table
3. resolve OCI descriptors by digest through the preamble table
4. either:
   - `mmap(fd, blob_offset, blob_length)` for raw mmap-capable blobs, or
   - `seek + read + decompress` into `unpack_root` for extracted blobs

That makes the hybrid design concrete rather than aspirational.

### Claim 1: Unified Format Across `libs/models` and `libs/vmm`

This argument is strong.

OCI descriptors, manifests, indexes, `artifactType`, annotations, and media types are flexible enough to describe heterogeneous payloads. In principle, one manifest graph could reference:

- safetensors model bundles
- GGUF model bundles
- tokenizers and config files
- bundle descriptor metadata
- kernels
- initrds
- rootfs images
- auxiliary metadata for both model and VMM artifacts

For example, Motlie-specific media types could include:

- `application/vnd.motlie.model.weights.gguf`
- `application/vnd.motlie.model.weights.safetensors`
- `application/vnd.motlie.vm.kernel`
- `application/vnd.motlie.vm.rootfs`
- `application/vnd.motlie.vm.disk`
- `application/vnd.motlie.bundle.metadata`

That is a real advantage over a Motlie-only `MTLPAY01` TOC that would otherwise need its own type system for every artifact family.

Critical caveat:

- "unified logical format" does not automatically mean "unified runtime handling"
- `libs/models` and `libs/vmm` still consume very different artifact shapes
- Motlie would still need crate-local policies for validation, extraction, and startup wiring

So OCI removes format reinvention at the descriptor layer, but it does not remove product-specific materialization logic.

### Claim 2: Tooling for Free

This is partly true, not fully true.

The ecosystem benefit is real:

- ORAS already treats OCI registries as generic artifact stores and encourages artifact-specific media types
- an OCI-native payload can be pushed to registries without redefining object identity or digest semantics
- OCI JSON manifests are much easier to inspect than a bespoke binary TOC

But the "for free" part is overstated for the appended-executable use case:

- standard OCI tooling expects either a registry or an OCI layout directory
- an appended region inside an executable is neither of those
- Motlie would still need a small tool to expose or materialize the appended payload as an OCI layout if operators want to use `oras`, `skopeo`, or `crane`
- the preamble is Motlie-specific, so off-the-shelf OCI tooling will ignore it rather than consume it

So the real benefit is:

- OCI gives reusable metadata semantics and registry interoperability
- Motlie still needs custom glue for the appended-executable embedding
- that glue is now well-scoped: preamble reader, footer reader, and optional OCI-layout export

### Claim 3: Content Addressing and Dedup

This argument is strong at the blob-store level.

OCI descriptors point to digested blobs, so:

- shared tokenizers
- shared config fragments
- shared rootfs layers
- shared weight files across multiple manifests

can all reuse the same blob identity.

Critical caveat:

- dedup is built into the OCI logical model, but not automatically into the final appended executable artifact
- if Motlie appends a full OCI layout to one executable, reuse only helps if the packer emits each blob once and multiple manifests reference it
- if every release builds one self-contained exe per bundle, cross-executable dedup is still a transport/distribution concern, not a file-size win inside one local file

So OCI helps most when:

- one executable carries multiple bundles, or
- the same blobs are also published to a registry or cache, or
- Motlie adds a shared local blob store later

### Claim 4: Hybrid Raw Blobs for mmap-Capable Weights

This is the most interesting part of the OCI case.

The standard OCI layer media types are tar-based, including the zstd-compressed form. That is a bad direct fit for llama.cpp and current Mistral startup when the goal is zero-copy mmap of multi-GB weight files.

However, OCI descriptors and manifests are not limited to the standard tar layer media types for every artifact design. OCI artifact usage allows custom media types, and unknown manifest entries are not supposed to hard-fail merely because the media type is unfamiliar.

That makes the following hybrid design plausible:

- small files:
  tar+zstd OCI layer blob, extracted normally
- large weight file:
  custom raw blob media type such as:
  `application/vnd.motlie.weight.gguf.v1`
  or
  `application/vnd.motlie.weight.safetensors.v1`
- manifest annotations:
  declare which descriptors are mmap-candidate payloads
- preamble TOC:
  records the physical byte offset of each blob inside the executable so the runtime can open the exe and map the raw blob directly

This is a real architectural path.

Critical caveats:

- OCI itself does not give the runtime byte offsets inside the executable; the preamble solves that
- once Motlie adds the preamble, OCI becomes the semantic layer and the preamble becomes the physical lookup layer
- the current `libs/model` and backend contracts do not support `(fd, offset, len)` or equivalent mmap handles
- safetensors is not always a single blob in practice; curated bundles often need config, tokenizer, and sometimes multiple shards
- direct mmap from the executable is operationally cleaner for GGUF than for multi-file safetensors trees

So the hybrid mmap idea is promising, and the preamble makes it concrete, but it is still a second-phase runtime contract change rather than the first implementation slice.

### Claim 5: Engineering Cost Delta vs Custom Format

This is where the refined hybrid design gets more balanced.

If Motlie chooses OCI as the internal logical format, the implementation still needs:

- a footer and payload-start locator
- a preamble blob-offset index for appended-executable lookup
- bundle selection logic
- extraction/materialization logic for directory-shaped artifacts
- integrity checking
- curated validation before backend startup

At that point, the practical delta versus a Motlie custom TOC is:

- OCI reduces schema-design work for manifests and blob typing
- the preamble is small and mechanical: fixed header plus blob entries
- OCI increases semantic surface area, because the implementation now needs to be correct with respect to OCI layout/index/manifest/blob rules as well as Motlie's appended-file framing

So the engineering comparison is not:

- "custom format" versus "just reuse OCI"

It is:

- "small Motlie-specific manifest + offset table"
  versus
- "OCI manifest/index/config/blob semantics + Motlie preamble/footer"

That is now a more attractive trade if Motlie wants cross-subsystem convergence and registry reuse, because the Motlie-specific part can stay intentionally small.

### Claim 6: Future Path to mmap-Capable Backend Contracts

This is valid and important.

If Motlie eventually wants:

- zero-copy GGUF startup from inside the executable
- no extraction of multi-GB weight files
- one local file opened and mmapped at a known offset

then `libs/model` will need an additive contract extension beyond plain path-based `ArtifactPolicy`.

Examples:

```rust
pub enum ResolvedArtifact {
    Directory { root: PathBuf },
    File { path: PathBuf },
    MmapBlob {
        file: std::fs::File,
        offset: u64,
        len: u64,
        media_type: String,
    },
}
```

or a bundle-local resolver trait that returns either materialized filesystem state or mmap-capable blob handles.

Critical caveat:

- this is a real contract change with backend impact
- it is not just packaging work
- the current design principle of keeping backends generic still holds, but the generic backend abstraction itself would have to become richer

That is plausible for a later phase, especially for GGUF.
It is too expensive to make the first single-executable slice depend on it.

### Three-Tier Materialization Strategy

The preamble+OCI design supports three distinct runtime tiers:

1. **Option A: Extract Everything**
   Extract all OCI-described payloads to `unpack_root` and let each consumer continue with its existing path-based startup contract.
   For models that means `ArtifactPolicy::LocalOnly { root }`. For VM consumers that means ordinary kernel/initrd/rootfs/disk paths.
   This requires no backend or boot-loader changes and is the recommended slice 1 implementation.
2. **Option B: mmap Everything Large**
   Store large blobs raw and uncompressed, then hand consumers `(fd, offset, len)` or equivalent mmap-capable handles.
   This is the slice 2 endgame for true zero-copy startup of GGUF weights, large safetensors blobs where practical, and potentially raw guest disk images.
3. **Option C: Hybrid**
   Store large weights and other large opaque blobs such as VM images as raw blobs for direct mmap or direct file access, but keep small files such as tokenizer/config/metadata in tar+zstd layers that are extracted normally.
   This is the best long-term balance and the most plausible steady-state architecture.

Option C is the most compelling end-state because it avoids pointless extraction of multi-GB weights and VM images while preserving the convenience and density of compressed archive layers for the small-file set.

### OCI-Inside-Executable Verdict

OCI is a credible internal payload format if Motlie values:

- one logical artifact model across model and VMM payloads
- content-addressed blobs as a first-class concept
- future registry publication without reformatting
- a later path toward mmap-capable raw weight blobs

OCI is weaker if Motlie values:

- the smallest possible first implementation
- minimal packaging machinery above the existing `LocalOnly { root }` startup path
- the least cognitive load for a model-only first slice

My updated assessment is:

- OCI is better than I gave it credit for as a logical payload graph
- OCI is still not enough by itself for the appended-executable runtime
- the preamble+footer layout fixes the physical lookup problem cleanly
- with that refinement, OCI becomes a serious candidate rather than merely a future export format
- the remaining question is not feasibility, but whether Motlie wants the extra semantic surface area in slice 1

## Recommendation

### Revised Recommendation

For the first implementation slice, I now recommend the hybrid preamble+OCI direction rather than the earlier fully custom manifest direction:

- a Motlie-owned binary preamble TOC plus footer for physical layout
- OCI manifest/config/blob semantics inside the appended payload
- extracted/materialized into consumer-owned working roots in slice 1
- handed to model backends as `ArtifactPolicy::LocalOnly { root }`
- handed to VM/guest consumers as ordinary artifact paths resolved from the extracted root

This changes the earlier recommendation in one important way:

- physical framing stays Motlie-owned
- logical artifact semantics move to OCI now, not later

### Decision Framing

There are now two defensible directions:

1. **Minimal first slice**
   Use a Motlie-native manifest/TOC now, optimized purely for appended-executable local startup.
2. **Converged artifact graph**
   Use OCI manifest/index/config/blob semantics inside the appended region, plus a Motlie preamble/footer for executable-local blob lookup.

With the preamble refinement, I now prefer direction 2.

Why:

- one packaging model can span `libs/models` and `libs/vmm`
- registry publication can reuse the same blobs and digests
- the preamble keeps the Motlie-specific part small and well-scoped
- slice 1 can still be pure extraction to ordinary local paths
- slice 2 has a clean path to zero-copy mmap for raw blobs

### Slice Plan

- Slice 1:
  implement preamble+OCI packaging, extract all payloads to local working roots, and keep consumer contracts unchanged
- Slice 2:
  add additive consumer-facing raw-blob handle support, starting with `libs/model` and extending to VM artifact consumers as needed
- Slice 3:
  converge on the hybrid steady state where large blobs mmap or stream directly and small files extract normally

### Why the Converged Slice Now Wins

- it still matters as a fallback benchmark
- it is still the cheapest way to validate only the extraction path
- but it no longer wins overall once the preamble eliminates the biggest physical-layout objection to OCI

### Where OCI Still Falls Short Today

- appended-executable discovery still needs Motlie-owned preamble/footer code
- direct use of common OCI CLI tooling still needs a conversion or export step
- standard tar-based layers do not solve mmap for large weight files
- the hybrid raw-blob design depends on a later backend contract expansion
- implementation still has to decide which payload classes are raw blobs versus archived layers

### Explicit Non-Recommendations

- do not start with `include_bytes!` for model weights
- do not start with squashfs unless the product is intentionally Linux-only
- do not assume OCI removes the need for Motlie-specific framing and runtime glue
- do not make slice 1 depend on mmap-capable backend changes

## API Sketches

### Sketch 1: Preamble Header and Blob Table

```rust
const MOTLIE_PREAMBLE_MAGIC: [u8; 8] = *b"MTLOCI01";
const MOTLIE_FOOTER_MAGIC: [u8; 8] = *b"MTLOCI02";

#[repr(C)]
struct PreambleHeader {
    magic: [u8; 8],
    version: u32,
    header_len: u32,
    blob_count: u32,
    reserved: u32,
    oci_layout_offset: u64,
    oci_layout_len: u64,
}

#[repr(C)]
struct BlobEntry {
    digest_sha256: [u8; 32],
    blob_offset: u64,
    blob_len: u64,
    uncompressed_len: u64,
    encoding: u32,
    flags: u32,
}

#[repr(C)]
struct PayloadFooter {
    magic: [u8; 8],
    version: u32,
    preamble_offset: u64,
    preamble_len: u64,
}

enum BlobEncoding {
    Stored = 0,
    Zstd = 1,
}
```

Notes:

- the preamble maps blob digests to physical byte ranges inside the executable
- OCI continues to identify blobs semantically by digest and media type
- the footer gives O(1) discovery from EOF

### Sketch 2: Runtime Payload Resolver

```rust
pub struct ExecutablePayload {
    exe: std::fs::File,
    preamble: PreambleHeader,
    blobs: std::collections::BTreeMap<[u8; 32], BlobEntry>,
}

impl ExecutablePayload {
    pub fn open_current() -> Result<Self, XarError> {
        let exe = std::fs::File::open(std::env::current_exe()?)?;
        let footer = read_footer(&exe)?;
        let (preamble, blobs) = read_preamble(&exe, footer.preamble_offset)?;
        Ok(Self { exe, preamble, blobs })
    }

    pub fn blob(&self, digest_sha256: [u8; 32]) -> Result<&BlobEntry, XarError> {
        self.blobs
            .get(&digest_sha256)
            .ok_or_else(|| XarError::InvalidExecutablePayload)
    }
}
```

### Sketch 3: Generic XAR Resolver Output

```rust
pub enum XarResolvedArtifact {
    Directory { root: std::path::PathBuf },
    File { path: std::path::PathBuf },
    RawBlob {
        file: std::fs::File,
        offset: u64,
        len: u64,
        media_type: String,
    },
}
```

Notes:

- this is the crate-local abstraction `libs/xar` should own
- model and VM consumers can adapt it to their own startup surfaces
- slice 1 mostly returns `Directory` and `File`
- later slices can use `RawBlob` for mmap-capable startup

### Sketch 4: Model-Side Integration

```rust
pub fn resolve_packaged_bundle_root(
    descriptor: &BundleDescriptor,
    unpack_root: &std::path::Path,
) -> Result<std::path::PathBuf, ModelsError> {
    let artifacts = descriptor
        .artifacts
        .as_ref()
        .ok_or_else(|| ModelsError::MissingArtifacts {
            bundle_id: descriptor.id.clone(),
        })?;

    match &artifacts.source {
        ArtifactSource::HuggingFace { .. } => Err(ModelsError::MissingArtifacts {
            bundle_id: descriptor.id.clone(),
        }),
        ArtifactSource::ExecutablePayload {
            locator,
            bundle_key,
            mode,
        } => {
            let payload = open_executable_payload(locator)?;
            let bundle_root = unpack_root.join(bundle_key);
            materialize_oci_bundle(&payload, bundle_key, mode, &bundle_root)?;
            Ok(bundle_root)
        }
    }
}
```

The curated bundle module would then continue with the existing pattern:

```rust
let local_root = resolve_packaged_bundle_root(&descriptor(), unpack_root)?;
let options = StartOptions {
    artifact_policy: Some(ArtifactPolicy::LocalOnly { root: local_root }),
    unpack_root: None,
    ..options
};
self.inner.start(options).await
```

This keeps the generic backend untouched.

### Sketch 5: VM-Side Integration

```rust
pub struct GuestArtifactSet {
    pub kernel: std::path::PathBuf,
    pub initrd: Option<std::path::PathBuf>,
    pub rootfs: Option<std::path::PathBuf>,
    pub disk: Option<std::path::PathBuf>,
}

pub fn resolve_guest_artifacts(
    payload: &ExecutablePayload,
    guest_key: &str,
    unpack_root: &std::path::Path,
) -> Result<GuestArtifactSet, XarError> {
    materialize_guest_payload(payload, guest_key, unpack_root)
}
```

This is the VM analogue to the model-side bundle-root resolver.

### Sketch 6: Future mmap-Capable Resolver

```rust
pub enum ResolvedArtifactHandle {
    Directory { root: PathBuf },
    File { path: PathBuf },
    MmapBlob {
        file: std::fs::File,
        offset: u64,
        len: u64,
        media_type: String,
    },
}
```

This is not required for slice 1.
It is the additive path that makes the preamble+OCI design pay off operationally for large raw blobs.

### Sketch 7: Release Assembly Tool

```rust
pub struct DistributionPlan {
    pub executable: std::path::PathBuf,
    pub payloads: Vec<String>,
    pub artifact_roots: Vec<std::path::PathBuf>,
    pub output: std::path::PathBuf,
}

pub fn assemble_distribution(plan: &DistributionPlan) -> anyhow::Result<()> {
    let mut writer = std::fs::File::create(&plan.output)?;
    copy_executable(&plan.executable, &mut writer)?;
    let toc = append_payloads(&mut writer, &plan.payloads, &plan.artifact_roots)?;
    append_footer(&mut writer, &toc)?;
    Ok(())
}
```

This should live in a release-oriented helper binary or script, not in the generic backend crates.

## Testing Scope for PLAN

The follow-up PLAN should cover:

1. Packaging correctness
   verify footer discovery, digest validation, truncated payload detection, and version mismatch handling
2. Payload materialization
   verify extraction produces the exact directory/file layout expected by current model bundle validators and by guest boot consumers
3. Consumer compatibility
   verify both a safetensors bundle and a GGUF bundle start from packaged `LocalOnly` roots without backend code changes, and verify a guest artifact set resolves to ordinary kernel/rootfs/disk paths
4. Cross-platform launcher behavior
   verify current-executable discovery and appended-payload reads on Linux and macOS
5. Failure policy
   verify missing `unpack_root`, unwritable unpack dir, corrupted chunks, and partial extraction cleanup
6. Performance
   measure first-start extraction latency, steady-state restart latency, and binary size deltas for at least one model bundle and one large guest-image payload

## References

- Rust `include_bytes!` documentation:
  https://doc.rust-lang.org/std/macro.include_bytes.html
- OCI image layout specification:
  https://raw.githubusercontent.com/opencontainers/image-spec/v1.1.1/image-layout.md
- OCI manifest specification:
  https://raw.githubusercontent.com/opencontainers/image-spec/v1.1.1/manifest.md
- OCI media types specification:
  https://raw.githubusercontent.com/opencontainers/image-spec/v1.1.1/media-types.md
- OCI image index specification:
  https://raw.githubusercontent.com/opencontainers/image-spec/v1.1.1/image-index.md
- Linux `memfd_create(2)` manual:
  https://man7.org/linux/man-pages/man2/memfd_create.2.html
- Linux kernel squashfs documentation:
  https://docs.kernel.org/filesystems/squashfs.html
- OCI image specification:
  https://specs.opencontainers.org/image-spec/
- ORAS introduction:
  https://oras.land/docs/
