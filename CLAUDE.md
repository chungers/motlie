# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Process and Conventions
### Design and Planning

In each project, docs/DESIGN.md documents the problem, and non-goals or related problems that are out of scope.
The DESIGN must consider 2-3 alternatives and document the best one as proposal, with comprehensive analysis
of the pros / cons, limitations, and engineering trade-offs.
Ask the user if you're uncertain the project is a new development or a brownfield requiring migration.
If the project is a new development, as identified by the user, the DESIGN can omit considerations on migrations
to optimize speed of delivery.  If the user identified the project as brownfield, you must include migration plan.

The DESIGN must include a high-level system design, including data flow analysis, and high level api design.
Correctness and robustness of the solution is highest priority when considering alternatives and system design.
The DESIGN must include a high-level test plan by identifying critical components and subsystems.
The DESIGN must also include usage examples to demonstrate the user experience (as cli or as library).
The DESIGN must have at the top of the doc, a Changelog.  Changelog entries should contain (date, who, summary).

Based on DESIGN, a PLAN (docs/PLAN.md) is created by considering the functional and non-functional requirements
in the DESIGN. Tasks broken down by phases should be created and documented (with checkboxes) after analyzing
the DESIGN.  Optionally, consider parallelization of work into workstreams.  Document a high-level overview of
possible workstreams.  Avoid getting into granular synchronization and spawning task assignments that could
make project management overly complicated.

DESIGN and PLAN are initial, best-effort cut at the problem.  They are likely to change as implementation
progresses.  When errors or changes are identified, document them inline of the document with clear notation
of (who, when, why), with the 'why' optionally including a link to a deep-dive doc in docs/ directory.
Whenever the DESIGN and PLAN docs are changed, include a (date, who, summary) Changelog entry at the top of doc.
Always identify yourself (e.g. '@codex' | '@claude') in all comments, in changelog or inline comments.

### Code Implementation

Never start implementation without outlining proposal for approval, unless there's already a PLAN in docs/

You must consult DESIGN.md and PLAN.md in the project's docs/ directory for information about the project.

If a PLAN doesn't exist, ask the user how to proceed. You may be asked to propose and outline your proposal. 
The DESIGN and PLAN are not perfect.  As you implement, call out concerns or bugs you identified. As you call
out problems, document them *inline* in either the DESIGN or PLAN doc, by including a brief summary of the
problem, with a optional link to a doc you generated to describe the issue in depth.  Place that doc in the
docs/ directory of the project.

Look for `@claude` or `@codex` comments in code for specific instructions to you.  You can also leave code
comments for the reviewer, for example `// @codex - Have a look at this to confirm correctness`.  Address these
even if you're not `@codex` or `@claude`.  You're a coding agent this comment is left by someone else asking for help.

After modifying code, review and update all related docs (*.md files).  If you are completing a task specified
in PLAN.md, update it with the check if there's a checkbox, or insert an inline comment, with (date, you, status).

Commit is ready only after: all tests pass, examples/, bins/, benchmarks/ all build, docs updated.

Never commit and push without explicit approval.
 
### Reviewing Code / PR

Always check implementation against DESIGN and PLAN, if they are available.  DESIGN and PLAN are not written
in stone, so while reviewing PR, it's useful to step back and look at the big picture.

Call out any inconsistencies and inaccuracies in the DESIGN, PLAN, or code implementations.

Validate all claims. Comment inline where possible.

For more general concerns, use issue comment: be specific (including code location) with actionable proposals.
Include verdict (accept | ok to merge | needs work) in issue comment.

If any specific inline concerns are addressed to your satisfaction, resolve them via gh api.

Always identify yourself (e.g. '@codex' | '@claude') when commenting.

### Addressing Feedback in PR

You must address ALL concerns (inline comments or issue comments). If you disagree with the feedback, be
very specific about why and provide counteroffer / rationale and leave the comment open / unresolved.

Your work in addressing all comments must have enough detail on what was done so that within that limit context
one can reason about the correctness of your fix and can make informed decision on resolution.

Comment inline, and use issue comment to summarize your work in this round.

Do not unilaterally resolve comments using gh api and leave the comments for the reviewer to close/ resolve.

Always identify yourself (e.g. '@codex' | '@claude') when commenting.


## Build & Test Commands

```bash
# Build workspace
cargo build

# Build specific binaries
cargo build --bin motlie
cargo build --bin bench_vector --features benchmark  # Requires libhdf5-dev

# Run all tests
cargo test

# Run specific test suites
cargo test --test db_cli
cargo test --test fulltext_cli

# Run documentation tests (motlie-db only)
cargo test -p motlie-db --doc

# Run a single example
cargo run --example hnsw
cargo run --example pagerank

# Build with OpenTelemetry support
cargo build --features dtrace-otel

# Build with native SIMD optimization
cargo build --release --features simd-native
```

## Workspace Structure

Three libraries + two binaries:

- `libs/core` (motlie-core): SIMD-optimized distance functions + tracing/telemetry
- `libs/db` (motlie-db): Graph (RocksDB), fulltext (Tantivy), and vector (HNSW) subsystems
- `libs/mcp` (motlie-mcp): MCP server framework with tool composition
- `bins/motlie`: CLI for database inspection and fulltext indexing
- `bins/bench_vector`: Vector search benchmarking (requires `--features benchmark`)

## Architecture

### Processor Pattern

Both graph and vector subsystems follow the same architecture:
- **Processor**: Synchronous internal API that owns resources (RocksDB, indices, caches)
- **Reader/Writer**: Async channel-based interfaces with MPMC consumer pools
- **Consumer**: Holds `Arc<Processor>`, executes queries/mutations

### Type-Safe Access Modes

```
Storage::readonly(path)   → Storage<ReadOnly>  → ReadOnlyHandles  (reader() only)
Storage::readwrite(path)  → Storage<ReadWrite> → ReadWriteHandles (reader() + writer())
```

Storage derives paths automatically: `<db_path>/graph` and `<db_path>/fulltext`.

### Mutation Pipeline

```
mutation.run(writer)
    → graph::MutationConsumer (RocksDB persists)
    → fulltext::MutationConsumer (Tantivy indexes)
```

### Trait Dispatch

- `Runnable` trait: `query.run(&reader, timeout)` / `mutation.run(&writer)`
- `ToolCall` trait (MCP): Parameter types implement execution logic
- `ResourceLifecycle` trait (MCP): Graceful shutdown hooks

## Key Patterns

### Flush API

Writer uses async MPSC channel; `mutation.run(&writer)` returns after enqueue, not commit:
- `send(Vec<Mutation>)` - Fire-and-forget (high throughput)
- `flush()` - Wait for pending commits (read-after-write consistency)
- `send_sync(Vec<Mutation>)` - Combined send + flush

### Graph Enumeration

Use `AllNodes` and `AllEdges` with cursor-based pagination for graph algorithms:
```rust
let mut query = AllNodes::new(1000);
if let Some(last_id) = cursor {
    query = query.with_cursor(last_id);
}
let page = query.run(handles.reader(), timeout).await?;
```

## Feature Flags

- `dtrace-otel`: OpenTelemetry support (OTLP export)
- `benchmark`: Vector benchmark datasets (parquet, arrow, hdf5) - requires libhdf5-dev
- `simd-native` / `simd-avx2` / `simd-avx512` / `simd-neon`: SIMD dispatch strategies

## Documentation Locations

- Vector design docs: `libs/db/src/vector/docs/` (ROADMAP.md is source of truth for status)
- General docs: `libs/db/docs/` (getting-started.md, query-api-guide.md, TODO.md)
- CLI docs: `bins/motlie/docs/`, `bins/bench_vector/README.md`

