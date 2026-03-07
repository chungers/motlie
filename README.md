# motlie

Workspace for Motlie database/runtime libraries and CLI tools.

## Crates

- `libs/db` (`motlie-db`): graph, fulltext, and vector storage/search subsystems
- `libs/mcp` (`motlie-mcp`): MCP server/tooling over Motlie APIs
- `libs/core` (`motlie-core`): shared distance + telemetry utilities

## CLIs

- `motlie` (`bins/motlie`): inspect graph/vector RocksDB data and run fulltext indexing/search
- `bench_vector` (`bins/bench_vector`): vector benchmark and ANN diagnostics CLI (requires `--features benchmark`)

## Where To Start

- DB/library usage: `libs/db/README.md`
- MCP server usage: `libs/mcp/README.md`
- Core utilities: `libs/core/README.md`
- `motlie` CLI usage: `bins/motlie/README.md`
- `bench_vector` usage: `bins/bench_vector/README.md`

## Build

```bash
# Build workspace
cargo build

# Build CLI binaries
cargo build --bin motlie
cargo build --bin bench_vector --features benchmark
```
