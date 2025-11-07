# motlie-db Documentation

This directory contains design documentation, analysis, and implementation notes for the motlie-db library.

## Documentation Index

### API Documentation

#### [reader.md](reader.md)
Complete reference for the Reader API with usage examples and query patterns.

**Contents**:
- Query operations (node_by_id, edge_by_id, etc.)
- Timeout handling
- Return types and semantics
- Example usage patterns

---

### Design Analyses & Decisions

The following documents capture design discussions, analyses, and decisions made during development:

#### [reader-api-gap-analysis.md](reader-api-gap-analysis.md)
Analysis of whether the Reader API can replace direct RocksDB access for database verification.

**Key Findings**:
- Reader API designed for point queries (requires IDs)
- Verification requires bulk iteration (unknown IDs)
- Recommendation: Keep direct RocksDB for admin operations
- Future consideration: Separate `DatabaseInspector` API

**Status**: ✅ Analyzed - Current design is appropriate

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

#### [rocksdb-prefix-scan-bug-analysis.md](rocksdb-prefix-scan-bug-analysis.md)
Investigation of potential prefix scanning issues with variable-length fields.

**Question**: Does the current schema break prefix scanning?

**Answer**: NO - Schema is CORRECT ✅

**Analysis**:
- Initial concern: EdgeCfValue has string in middle
- **Reality**: EdgeCfValue is a VALUE, not a KEY
- Keys (ForwardEdgeCfKey, ReverseEdgeCfKey) have string at END
- Prefix scanning works correctly

**Verification**:
- Test `test_forward_edges_keys_lexicographically_sortable` passes
- Demonstrates keys sort by (Id, Id) prefix correctly
- Early termination optimization in `get_edges_from_node_by_id()` is valid

**Status**: ✅ No bug - Schema is optimal

---

## Document Categories

### 1. API Reference
- `reader.md` - Complete Reader API documentation

### 2. Design Evolution
Documents tracking the evolution of API design:
- `reader-api-gap-analysis.md` - Initial gap identification
- `reader-api-completeness-analysis.md` - Comprehensive gap analysis
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
2. **API usage**: [`reader.md`](reader.md) - How to use the Reader API
3. **Schema design**: [`variable-length-fields-in-keys.md`](variable-length-fields-in-keys.md) - Critical for understanding key layout

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

1. [`reader-api-gap-analysis.md`](reader-api-gap-analysis.md) - Initial analysis
2. [`reader-api-completeness-analysis.md`](reader-api-completeness-analysis.md) - Gap identification
3. [`edge-by-id-explained.md`](edge-by-id-explained.md) - Problem deep-dive
4. [`edge-by-id-tuple-implementation.md`](edge-by-id-tuple-implementation.md) - Solution

Shows the iterative refinement process and decision rationale.

## Document Status Legend

- ✅ **Current**: Reflects current implementation
- 📋 **Historical**: Superseded by later design
- 🌟 **Essential**: Must-read for contributors
- ⚠️ **Critical**: Required reading before modifications

## Key Insights

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
