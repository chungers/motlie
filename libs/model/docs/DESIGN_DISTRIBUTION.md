# Model Bundle Distribution Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-12 | @codex-xar: Initial greenfield design for packaging curated `libs/models` bundle descriptors, concrete backends, and model artifacts into a single distributable executable. Migration and backward compatibility are explicitly out of scope. | All |

This document evaluates how Motlie should ship curated model bundles as a single executable artifact that contains:

1. the launcher / application binary
2. the curated `libs/models` bundle descriptors
3. the linked `libs/model/backends/*` runtime implementation(s)
4. the model weights and auxiliary files currently described by `BundleArtifacts`

The goal is not merely "embed files into an exe". The goal is to preserve the current layering and operator model:

- `libs/model` remains the stable lifecycle and capability contract
- `libs/models` remains the curator-owned bundle and artifact layer
- backend crates continue to consume a resolved local artifact root or an explicit fetch policy
- distribution-specific logic stays above the generic backends

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
- [Recommendation](#recommendation)
- [API Sketches](#api-sketches)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [References](#references)

---

## Overview

### Problem Statement

Today the curated bundle model is optimized for:

- bundle descriptors in `libs/models`
- explicit artifact download from Hugging Face
- local cache resolution and validation in curated bundle modules
- backend startup from `ArtifactPolicy::{AllowFetch, LocalOnly}`

That is correct for development and curation, but it is not sufficient for a "single-file distribution" story where one executable should be enough to launch a vetted bundle without a separate artifact download step.

The hard part is not descriptor registration. The hard part is that the current backends still expect filesystem-shaped artifacts:

- Mistral-backed safetensors bundles expect a resolved directory tree
- llama.cpp-backed GGUF bundles expect a concrete GGUF file path
- current backend implementations reject `StartOptions.unpack_root`

Therefore, any single-file distribution design that preserves the present layering must solve one of two problems:

1. materialize a local artifact root before backend startup, or
2. introduce a new lower-level backend contract for file descriptors / byte ranges / virtual filesystems

For the first slice, the first approach is cheaper and fits the current code better.

## Current Codebase Fit

The relevant surfaces today are:

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

This leads to an important design conclusion:

- single-file distribution should not be implemented by teaching generic backends about Hugging Face snapshots, executable offsets, squashfs images, or bundle-local packaging formats
- instead, `libs/models` should resolve a packaged artifact source into a normal local root, then pass `ArtifactPolicy::LocalOnly { root }` to the backend

That keeps the reviewed boundary intact.

## Requirements and Non-Goals

### Requirements

- Ship one executable containing launcher, descriptors, backends, and model artifacts
- Preserve the current `libs/model` to `libs/models` to backend layering
- Keep regulated local-only deployments able to fail closed
- Support multi-GB artifacts without pathological compile or link behavior
- Support curated bundle validation before backend startup
- Support at least Linux and macOS cleanly; avoid painting the project into a Linux-only corner unless explicitly chosen
- Keep future room for multiple bundles in one distribution
- Leave room for additive integrity features such as per-file digests or signatures

### Non-Goals

- Migration from prior packaging formats
- Arbitrary user-supplied checkpoints
- Full in-memory backend loading in the first slice
- Introducing a container runtime requirement just to run a local model

## Evaluation Criteria

Each option is evaluated on:

- build time impact
- binary size impact
- runtime overhead
- cross-platform support
- incremental update story
- integration with existing `ArtifactSource` / `ArtifactPolicy`
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

Useful as a publication format if Motlie later wants registry-native delivery.
Not the right primary runtime format for this task.

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

## Recommendation

### Recommended Direction

Adopt Option 2 with Option 4a as the concrete first encoding:

- a normal executable
- an appended payload
- a footer TOC
- zstd-compressed chunked artifact data
- bundle-layer extraction into `StartOptions.unpack_root`
- backend startup from `ArtifactPolicy::LocalOnly { root }`

### Why This Is the Best Fit

It aligns with the current codebase better than the alternatives:

- it does not require backends to learn about packaging formats
- it preserves the reviewed boundary where curated bundles own artifact resolution
- it turns single-file distribution into a packaging concern, not a compiler concern
- it keeps `ArtifactPolicy::LocalOnly` as the final runtime contract
- it leaves room to add lazy extraction or direct fd-based loading later

### Explicit Non-Recommendations

- do not start with `include_bytes!` for model weights
- do not start with squashfs unless the product is intentionally Linux-only
- do not make OCI the runtime package for local single-executable delivery

## API Sketches

### Sketch 1: Extend `ArtifactSource` at the Bundle Layer

```rust
pub enum ArtifactSource {
    HuggingFace { repo: &'static str },
    ExecutablePayload {
        locator: ExecutableLocator,
        bundle_key: &'static str,
        encoding: PayloadEncoding,
    },
}

pub enum ExecutableLocator {
    CurrentExecutable,
    Path(std::path::PathBuf),
}

pub enum PayloadEncoding {
    RawDirectoryV1,
    ChunkedZstdV1,
}
```

Notes:

- this belongs in `libs/models`, not `libs/model`
- `libs/model::ArtifactPolicy` can remain unchanged
- `ExecutablePayload` is curator-owned, just like the current Hugging Face rules

### Sketch 2: Materialize Before Backend Startup

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
            encoding,
        } => {
            let exe = locate_payload_executable(locator)?;
            let bundle_root = unpack_root.join(bundle_key);
            extract_payload_bundle(&exe, bundle_key, encoding, &bundle_root)?;
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

### Sketch 3: Footer TOC Format

```rust
const MOTLIE_PAYLOAD_MAGIC: [u8; 8] = *b"MTLPAY01";

#[repr(C)]
struct PayloadFooter {
    magic: [u8; 8],
    version: u32,
    toc_offset: u64,
    toc_len: u64,
    toc_sha256: [u8; 32],
}

struct TocEntry {
    bundle_key: String,
    relative_path: String,
    encoding: ChunkEncoding,
    offset: u64,
    encoded_len: u64,
    decoded_len: u64,
    sha256: [u8; 32],
}

enum ChunkEncoding {
    Stored,
    Zstd,
}
```

Design notes:

- footer-at-EOF means the runtime can discover the payload with one seek
- the TOC is the place to version bundle layout rules
- per-file digests let curated bundle validation fail before backend startup

### Sketch 4: Release Assembly Tool

```rust
pub struct DistributionPlan {
    pub executable: std::path::PathBuf,
    pub bundles: Vec<BundleId>,
    pub artifact_roots: Vec<std::path::PathBuf>,
    pub output: std::path::PathBuf,
}

pub fn assemble_distribution(plan: &DistributionPlan) -> anyhow::Result<()> {
    let mut writer = std::fs::File::create(&plan.output)?;
    copy_executable(&plan.executable, &mut writer)?;
    let toc = append_bundle_payloads(&mut writer, &plan.bundles, &plan.artifact_roots)?;
    append_footer(&mut writer, &toc)?;
    Ok(())
}
```

This should live in a release-oriented helper binary or script, not in the generic backend crates.

## Testing Scope for PLAN

The follow-up PLAN should cover:

1. Packaging correctness
   verify footer discovery, digest validation, truncated payload detection, and version mismatch handling
2. Bundle materialization
   verify extraction produces the exact directory/file layout expected by current curated bundle validators
3. Backend compatibility
   verify both a safetensors bundle and a GGUF bundle start from packaged `LocalOnly` roots without backend code changes
4. Cross-platform launcher behavior
   verify current-executable discovery and appended-payload reads on Linux and macOS
5. Failure policy
   verify missing `unpack_root`, unwritable unpack dir, corrupted chunks, and partial extraction cleanup
6. Performance
   measure first-start extraction latency, steady-state restart latency, and binary size deltas for at least one small and one large bundle

## References

- Rust `include_bytes!` documentation:
  https://doc.rust-lang.org/std/macro.include_bytes.html
- Linux `memfd_create(2)` manual:
  https://man7.org/linux/man-pages/man2/memfd_create.2.html
- Linux kernel squashfs documentation:
  https://docs.kernel.org/filesystems/squashfs.html
- OCI image specification:
  https://specs.opencontainers.org/image-spec/
