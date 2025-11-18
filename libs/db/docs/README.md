# motlie-db Documentation

This directory contains design documentation, analysis, and implementation notes for the motlie-db library.

## Documentation Index

### Essential Reading

#### [concurrency-and-storage-modes.md](concurrency-and-storage-modes.md) ⭐
**ESSENTIAL**: Complete guide to concurrent access patterns, storage modes, and threading.

**Contents**:
- Readonly vs Secondary vs Shared Readwrite comparison
- RocksDB architecture and flush timing
- Threading and concurrency patterns
- Performance tuning for visibility vs throughput
- Complete code examples for all patterns
- Test results and analysis (24% → 99% success rates explained)
- Migration guides

**Status**: ✅ Comprehensive guide - Read this first for concurrency topics

---

### API Documentation

#### [query-api-guide.md](query-api-guide.md) ⭐
**ESSENTIAL**: Complete guide to the modern Query API (v0.2.0+).

**Contents**:
- Type-driven query construction
- All 8 query types with examples
- Common patterns (concurrent queries, pagination, traversal)
- Advanced usage (composition, retry strategies)
- Migration guide from deprecated Reader API
- Performance considerations

**Status**: ✅ Current - **Start here for query operations**

---

#### [reader.md](reader.md)
Legacy reference for the deprecated Reader API (v0.1.x).

**Contents**:
- Query operations (node_by_id, edge_by_id, etc.) - **DEPRECATED**
- Timeout handling
- Return types and semantics
- Example usage patterns

**Status**: 📋 Historical - Use [query-api-guide.md](query-api-guide.md) instead

---

### Design Analyses & Decisions

The following documents capture design discussions, analyses, and decisions made during development:

#### [query-and-mutation-processor-simplification.md](query-and-mutation-processor-simplification.md) 🌟
**IMPLEMENTED** (2025-11-16): Unified trait-based architecture for queries and mutations.

**Problem Solved**:
- **Queries**: 8 trait methods, ~640 lines in Graph implementation, inconsistent with mutation pattern
- **Mutations**: Centralized `Plan::create_batch()` with 57-line match dispatcher

**Implemented Solution**:
- **Queries**: `QueryExecutor` trait - each query type implements `execute()`
- **Mutations**: `MutationPlanner` trait - each mutation type implements `plan()`
- **Processor traits**: Minimal interface (queries: `storage()`, mutations: `process_mutations()`)
- **Graph**: Simplified implementations (queries: 3 lines, mutations: orchestration only)

**Benefits Achieved**:
- ✅ **~800 lines removed**: 756 from queries, 57 from mutations
- ✅ **Architectural consistency**: Both follow trait-based execution pattern
- ✅ **Better encapsulation**: Logic lives with types, not in central dispatchers
- ✅ **Easier extensibility**: Add new types without modifying central code
- ✅ **Better testability**: Test individual type execution in isolation

**Status**: ✅ Fully Implemented - All 174 tests passing, benchmarks successful, examples working end-to-end

**Read More**: Comprehensive documentation of design, implementation, and migration results.

---

#### [reader-api-completeness-analysis.md](reader-api-completeness-analysis.md)
Comprehensive analysis of API gaps for a complete point-query interface.

**Identified Gaps**:
1. **Critical**: `node_by_name()` - lookup nodes by name
2. **Critical**: `edge_by_id()` - get full edge topology (NOW IMPLEMENTED ✅)
3. **Nice-to-have**: `entity_type()` - determine if ID is node or edge
4. Additional convenience methods

**Status**: ✅ Primary gap (`edge_by_id`) implemented

---

#### [edge-by-id-explained.md](edge-by-id-explained.md)
Detailed explanation of the `edge_by_id()` API and the problem it solves.

**Problem**:
- Original API: `edge_summary_by_id()` returned only `EdgeSummary`
- Topology (source, dest, name) was inaccessible by edge ID alone
- Created API asymmetry with `node_by_id()`

**Solution**:
- New API: `edge_by_id()` returns `(SrcId, DstId, EdgeName, EdgeSummary)`
- Denormalize EdgeCfValue to include topology
- Single O(1) lookup, no joins required

**Status**: ✅ Implemented

---

#### [edge-by-id-implementation-plan.md](edge-by-id-implementation-plan.md)
Initial implementation plan for transforming `edge_summary_by_id()` → `edge_by_id()`.

**Options Considered**:
1. **Option A**: Scan forward_edges CF (O(n), inefficient)
2. **Option B**: Add EdgeTopology CF (requires new index)
3. **Option C**: Denormalize EdgeCfValue (CHOSEN)

**Recommendation**: Denormalize EdgeCfValue with struct pattern

**Status**: 📋 Superseded by tuple implementation

---

#### [edge-by-id-tuple-implementation.md](edge-by-id-tuple-implementation.md)
Revised implementation plan using tuple pattern instead of struct.

**Design Decision**:
- Use tuple: `EdgeCfValue(SrcId, DstId, EdgeName, EdgeSummary)`
- Matches existing pattern: `NodeCfValue(NodeName, NodeSummary)`
- Maintains consistency across schema

**Implementation Steps**:
1. Update schema.rs - EdgeCfValue tuple
2. Update query.rs - EdgeByIdQuery type
3. Update graph.rs - Processor implementation
4. Update reader.rs - Public API
5. Update lib.rs - Public exports
6. Update tests - Field access patterns

**Status**: ✅ Implemented (later revised to different field order)

---

### Schema & Storage Design

#### [variable-length-fields-in-keys.md](variable-length-fields-in-keys.md) 🌟
**CRITICAL READING**: Essential guide to RocksDB key design with MessagePack serialization.

**Key Principles**:

1. **Rule: Variable-Length Last (VLL)**
   - ✅ Correct: `(Id, Id, String)` - string at END
   - ❌ Wrong: `(Id, String, Id)` - string in MIDDLE

2. **Why It Matters**:
   - MessagePack encodes string length in format byte
   - Shorter strings sort before longer strings
   - Variable-length in middle breaks prefix scanning
   - Cannot efficiently query by fixed-length prefix

3. **Current Schema Status**:
   - ✅ `ForwardEdgeCfKey(SrcId, DstId, EdgeName)` - CORRECT
   - ✅ `ReverseEdgeCfKey(DstId, SrcId, EdgeName)` - CORRECT
   - ✅ `EdgeCfValue(SrcId, EdgeName, DstId, Summary)` - OK (it's a value, not a key)

**Impact**:
- Enables efficient prefix scanning
- Allows early termination in range queries
- Critical for performance at scale

**Status**: ✅ Current schema follows best practices

---

#### [msgpack-tuple-ordering-impact.md](msgpack-tuple-ordering-impact.md)
Analysis of how tuple field ordering affects MessagePack serialization and storage.

**Key Findings**:

1. **Storage Impact**: NONE
   - Both orderings serialize to identical byte size
   - No fragmentation or padding
   - MessagePack is tightly packed

2. **Parsing Impact**: MINIMAL
   - Sequential parsing required (no random access)
   - To access position N, must parse 0..N-1
   - Impact only matters if accessing middle fields without later fields

3. **For EdgeCfValue**:
   - Old: `(SrcId, DstId, EdgeName, EdgeSummary)`
   - New: `(SrcId, EdgeName, DstId, EdgeSummary)`
   - No significant performance difference (both queries parse all fields)

**Conclusion**: Field ordering choice should prioritize semantic clarity over performance (minimal impact).

**Status**: ✅ Analyzed - No action needed

---

#### [rocksdb-prefix-scan-bug-analysis.md](rocksdb-prefix-scan-bug-analysis.md) ⚡
**CRITICAL**: MessagePack variable-length headers break RocksDB prefix optimization.

**Problem Discovered**:
- MessagePack adds variable-length string headers (1-3 bytes depending on length)
- This makes prefix length NON-CONSTANT
- RocksDB prefix extractors require CONSTANT-length prefixes
- Result: Cannot use prefix bloom filters or O(1) prefix seek

**Analysis**:
- MessagePack: `(Id, Id, "a")` has 39-byte prefix
- MessagePack: `(Id, Id, "long_name...")` has 40-byte prefix
- Direct encoding: ALWAYS 32-byte prefix regardless of string length

**Performance Impact**:
- Before: O(N) scan from start of column family
- After: O(K) seek directly to prefix
- **1,000-10,000× improvement** for large databases!

**Solution Implemented**: Direct byte concatenation for keys (values still use MessagePack)

**Status**: ✅ FIXED - Direct encoding implemented

---

#### [option2-implementation-outline.md](option2-implementation-outline.md) 🌟
**Complete implementation guide** for switching from MessagePack to direct byte concatenation for keys.

**Contents**:
- 6 implementation phases with code examples
- Direct encoding for all key types
- RocksDB prefix extractor configuration
- Query method updates for prefix seek
- Performance benchmarks
- Migration strategy

**Key Design**: Each `ColumnFamilyRecord` provides its own `column_family_options()` - clean encapsulation!

**Status**: ✅ IMPLEMENTED

---

#### [implementation-summary.md](implementation-summary.md)
Executive summary of the MessagePack analysis and direct encoding solution.

**Quick Reference**:
- Problem statement
- Performance impact numbers
- Implementation phases overview
- Next steps guide

**Status**: ✅ Current

---

#### [benchmark-plan.md](benchmark-plan.md)
Comprehensive benchmark strategy to measure performance improvements.

**Benchmark Suites**:
1. Write Performance - Validates no regression
2. Point Lookups - Validates unchanged
3. Prefix Scans ⚡ - Tests O(N) → O(K) improvement
4. Scan by Position - Proves position-independent performance
5. Scan by Degree - Validates O(K) scaling

**Status**: 📋 Planned (placeholder implemented)

---

## Document Categories

### 1. API Reference
- `query-api-guide.md` ⭐ - **Current Query API** (v0.2.0+)

### 2. Design Evolution
Documents tracking the evolution of API design:
- `query-and-mutation-processor-simplification.md` ⭐ - **IMPLEMENTED**: Unified trait-based execution for queries and mutations
- `edge-by-id-explained.md` - Problem statement
- `edge-by-id-implementation-plan.md` - Initial plan (struct approach)
- `edge-by-id-tuple-implementation.md` - Revised plan (tuple approach)

### 3. Schema & Performance
Critical documents for understanding database internals:
- ⭐ `variable-length-fields-in-keys.md` - **MUST READ** for schema design
- `msgpack-tuple-ordering-impact.md` - Serialization analysis
- `rocksdb-prefix-scan-bug-analysis.md` - Prefix scanning verification

## Reading Guide

### For New Contributors

1. **Start here**: [`../README.md`](../README.md) - Library overview and architecture
2. **API usage**: [`query-api-guide.md`](query-api-guide.md) - How to query the database ⭐
3. **Concurrency**: [`concurrency-and-storage-modes.md`](concurrency-and-storage-modes.md) - Threading patterns
4. **Schema design**: [`variable-length-fields-in-keys.md`](variable-length-fields-in-keys.md) - Critical for understanding key layout

### For Schema Modifications

⚠️ **Required reading before modifying RocksDB keys**:

1. [`variable-length-fields-in-keys.md`](variable-length-fields-in-keys.md) - Key design principles
2. [`msgpack-tuple-ordering-impact.md`](msgpack-tuple-ordering-impact.md) - Serialization format
3. Run tests: `cargo test test_forward_edges_keys_lexicographically_sortable`

**Rules**:
- ✅ Variable-length fields MUST be at the end of keys
- ✅ Test lexicographic ordering after changes
- ✅ Document invariants in code comments

### For API Evolution

Understanding the design process:

1. [`query-api-guide.md`](query-api-guide.md) - **Current Query API** (v0.2.0+) ⭐
2. [`query-and-mutation-processor-simplification.md`](query-and-mutation-processor-simplification.md) - Trait-based refactoring
3. [`edge-by-id-explained.md`](edge-by-id-explained.md) - Problem deep-dive (edge topology implementation)
4. [`edge-by-id-tuple-implementation.md`](edge-by-id-tuple-implementation.md) - Solution

Shows the iterative refinement process and decision rationale.

## Document Status Legend

- ✅ **Current**: Reflects current implementation
- 📋 **Historical**: Superseded by later design
- 🌟 **Essential**: Must-read for contributors
- ⚠️ **Critical**: Required reading before modifications

## Key Insights

### Processor Architecture Principles

1. **Logic with Types**: Business logic should live with data types, not in central implementation
   - **Mutations**: Each mutation type implements `MutationPlanner::plan()` to generate storage operations
   - **Queries**: Each query type implements `QueryExecutor::execute()` to fetch results
   - **Benefits**: Easier to extend, test, and maintain

2. **Minimal Trait Surface**: Keep trait methods minimal to reduce implementation burden
   - **Mutations**: 1 method (`process_mutations`)
   - **Queries**: 1 method (`storage()`) + QueryExecutor trait per query type
   - **Result**: 90% reduction in Processor trait complexity

3. **Separation of Concerns**: Storage access vs query execution
   - **Processor**: Provides storage access (infrastructure)
   - **Query types**: Implement execution logic (business logic)
   - **Clear boundaries**: Easy to test and mock

### Schema Design Principles

1. **Variable-Length Last**: Always place variable-length fields at the end of RocksDB keys
   - Enables efficient prefix scanning
   - Allows early termination in range queries
   - Critical for performance

2. **Denormalization for Performance**: Store topology in multiple places
   - Edges CF: Complete edge data
   - ForwardEdges CF: Source → Destination index
   - ReverseEdges CF: Destination → Source index
   - Trade-off: 3x storage overhead for O(1) lookups

3. **MessagePack Self-Describing**: Each element carries type/length information
   - No external offset table needed
   - Sequential parsing required
   - Compact binary encoding

### API Design Principles

1. **Complete Data Returns**: Avoid requiring multiple queries
   - `edge_by_id()` returns topology + summary in one call
   - `node_by_id()` returns name + summary
   - Reduces round-trips, simplifies usage

2. **Type Safety**: Use newtype wrappers
   - `SrcId`, `DstId` instead of generic `Id`
   - `EdgeName` instead of `String`
   - Self-documenting APIs

3. **Timeout Support**: All queries have timeout parameter
   - Prevents unbounded blocking
   - Enables responsive applications
   - Timeout enforced by consumer, not client

## Contributing Documentation

When adding new documentation:

1. **Create the document** in `docs/`
2. **Add entry to this README** with:
   - Clear title and link
   - Brief summary (2-3 sentences)
   - Key findings or decisions
   - Status indicator
3. **Update reading guide** if document is essential
4. **Cross-reference** related documents

## Related Resources

- **Main README**: [`../README.md`](../README.md)
- **Source Code**: [`../src/`](../src/)
- **Examples**: [`../../examples/store/`](../../examples/store/)
- **Tests**: See `#[cfg(test)]` modules in source files
