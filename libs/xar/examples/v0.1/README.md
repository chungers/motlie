# `xar` v0.1 — GGUF Round-Trip Skeleton

This directory captures the intended first validation slice for `libs/xar`.

`v0.1` is intentionally narrow:

- one GGUF model payload
- one executable-appended `xar` image
- one model-side consumer path through `libs/model`

The goal is to prove the format end to end before expanding to VM guest payload sets.

## Workflow

1. Start from a local GGUF model file.
2. Build an OCI-backed `xar` payload that contains:
   - one raw GGUF blob
   - OCI manifest/config metadata
   - a Motlie preamble that maps the blob digest to `(offset, len, encoding)`
   - a footer that points back to the preamble
3. Append that payload to a small executable.
4. At runtime:
   - read footer
   - read preamble
   - resolve GGUF blob by digest
   - either:
     - extract to a local root and start via `ArtifactPolicy::LocalOnly`, or
     - in a later slice, pass `(fd, offset, len)` to a mmap-capable backend contract
5. Load the resulting artifact through the existing GGUF consumer path in `libs/model`.

## Why GGUF First

- single large blob
- well-defined structure
- direct mmap path exists conceptually
- existing in-tree consumer path in `libs/model`
- minimal moving parts compared with kernel/initrd/rootfs/disk sets

## Out of Scope for `v0.1`

- safetensors shard graphs
- mixed model + VM payload sets
- guest boot orchestration
- direct mmap backend integration
