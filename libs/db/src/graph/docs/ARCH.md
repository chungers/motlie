# ARCH: Graph Architecture (Current)

**Scope:** Current graph subsystem architecture after the processor/async split.

---

## Summary

The graph subsystem separates **synchronous core logic** from **public async APIs**:

- **Processor (sync, internal)**: owns Storage + NameCache; executes mutation/query logic directly
- **Writer (async, public)**: MPSC channel for mutations; consumer uses Processor
- **Reader (async, public)**: MPMC channel for queries; consumer uses Processor

This mirrors the vector subsystem pattern and centralizes business logic in the
Processor + ops modules while keeping async IO in thin wrappers.

---

## Key Components

- **Processor** (`graph::processor::Processor`) — synchronous execution, core business logic
- **Writer** (`graph::writer::Writer`) — async mutation API
- **Reader** (`graph::reader::Reader`) — async query API
- **Mutation ops** (`graph::ops`) — single source of truth for mutation logic
- **RequestEnvelope / ReplyEnvelope** — typed request/reply metadata for async pipelines

---

## Data Flow

```
Mutation / Query (typed)
   │
   ├─> RequestEnvelope { request_id, created_at, ... }
   │
Writer / Reader (async channel)
   │
   ├─> Consumer (owns Processor)
   │
   └─> Processor (sync core) -> Storage
```

---

## Design Goals

- Keep business logic in synchronous ops/processor
- Keep async APIs thin and typed
- Maintain parity with vector subsystem architecture

---

## Related Docs

- VERSIONING.md — time‑travel, rollback, and version history
- CONTENT-ADDRESS.md — reverse index + summary addressing
- API.md — public API reference
- docs/archive/ARCH2.md — prior refactor plan and historical notes
