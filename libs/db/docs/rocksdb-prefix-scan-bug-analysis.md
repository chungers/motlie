# RocksDB Prefix Scanning Bug Analysis

**Status**: ✅ **FIXED** (as of 2025-11-19)

This document analyzes the MessagePack variable-length header issue that broke RocksDB prefix scanning. The problem has been resolved by implementing direct byte concatenation for keys.

## Summary

**Original Status**: 🔴 **CRITICAL BUG FOUND**
**Current Status**: ✅ **RESOLVED** - Direct encoding implemented

The original `ForwardEdgeCfKey` and `ReverseEdgeCfKey` schemas had **variable-length strings** that, when serialized with MessagePack, created non-constant prefix lengths and broke RocksDB prefix optimization.

## Current Schema

```rust
// libs/db/src/schema.rs:179-183
struct ForwardEdgeCfKey(
    EdgeSourceId,      // Position 0: Id (16 bytes)
    EdgeDestinationId, // Position 1: Id (16 bytes)
    EdgeName,          // Position 2: String (variable length) ⚠️
);

// libs/db/src/schema.rs:217-221
struct ReverseEdgeCfKey(
    EdgeDestinationId, // Position 0: Id (16 bytes)
    EdgeSourceId,      // Position 1: Id (16 bytes)
    EdgeName,          // Position 2: String (variable length) ⚠️
);
```

## The Problem

### MessagePack Serialization Reality

When serialized with MessagePack, keys sort **lexicographically** by their byte representation:

```
Key (100, 200, "a")   => [93, 64, cc, c8, a1, 61]           (6 bytes)
Key (100, 200, "abc") => [93, 64, cc, c8, a3, 61, 62, 63]  (8 bytes)
Key (100, 201, "a")   => [93, 64, cc, c9, a1, 61]           (6 bytes)
```

**Critical observation**: Keys are sorted lexicographically as:
1. `(100, 200, "a")`
2. `(100, 200, "abc")`
3. `(100, 201, "a")`

This is **correct** for the (Id, Id, String) pattern because **the two fixed-length Ids sort first**.

### Wait... Actually This Works!

After deeper analysis, **the current schema is actually CORRECT**:

```rust
ForwardEdgeCfKey(src_id, dst_id, name)  // String at END - ✅ CORRECT
ReverseEdgeCfKey(dst_id, src_id, name)  // String at END - ✅ CORRECT
```

The variable-length string is at **position 2 (the END)**, not in the middle!

## Verification Test Results

From our MessagePack test (`/tmp/msgpack_test`):

```
Test 2: String at END - (Id, Id, String)
This is the RECOMMENDED pattern for prefix scanning

  Lexicographic order after sorting:
  1. (100, 199, "abc") - 8 bytes
  2. (100, 200, "a")   - 6 bytes
  3. (100, 200, "ab")  - 7 bytes
  4. (100, 200, "abc") - 8 bytes
  5. (100, 201, "a")   - 6 bytes

  ✅ BETTER: Can reliably scan by (Id, Id) prefix!
      All keys with (100, 200) are grouped together
      Prefix scanning works correctly
```

## Implementation Analysis

### Current Implementation (libs/db/src/graph.rs:596-630)

```rust
async fn get_edges_from_node_by_id(...) -> Result<Vec<(SrcId, EdgeName, DstId)>> {
    let id = query.id;

    // Scan the forward_edges column family for all edges with this source ID
    // Keys are (source_id, dest_id, name) and RocksDB stores them in sorted order

    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
    for item in iter {
        let (key_bytes, _value_bytes) = item?;
        let key: ForwardEdgeCfKey = deserialize(key_bytes)?;

        let source_id = key.0.0;
        if source_id == id {
            edges.push((source_id, key.2, dest_id));
        } else if source_id > id {
            break;  // ⚠️ Optimization assumes keys sorted by source_id
        }
    }
}
```

### Analysis

**The optimization at line 627-630 is CORRECT**:

```rust
} else if source_id > id {
    // Keys are sorted by source_id first, so once we pass the target ID, stop
    break;
}
```

This works because:
1. Keys are `(source_id, dest_id, name)`
2. MessagePack serializes in order: `[array_header][source_id_bytes][dest_id_bytes][name_bytes]`
3. Lexicographic comparison hits `source_id` first
4. Once `source_id > target`, all remaining keys have larger `source_id`

## The Confusion: EdgeCfValue vs Key Structure

The confusion arises because we recently changed **EdgeCfValue** (the VALUE, not the key):

```rust
// VALUE structure (libs/db/src/schema.rs:94-98)
struct EdgeCfValue(
    SrcId,       // Position 0
    EdgeName,    // Position 1 - String in MIDDLE ⚠️
    DstId,       // Position 2
    EdgeSummary, // Position 3
);
```

**This is FINE** because:
- EdgeCfValue is the **VALUE**, not the KEY
- Values are not used for sorting or prefix scanning
- Values are opaque blobs to RocksDB
- Only KEYS need fixed-length prefixes for efficient scanning

## Conclusion

### Status: ✅ **NO BUG - SCHEMA IS CORRECT**

The current key schemas are correct:
- ✅ `ForwardEdgeCfKey(src_id, dst_id, name)` - String at end
- ✅ `ReverseEdgeCfKey(dst_id, src_id, name)` - String at end
- ✅ Prefix scanning works correctly
- ✅ Early termination optimization is valid

### Important Distinction

- **Keys** (ForwardEdgeCfKey, ReverseEdgeCfKey): Must have fixed-length prefix for scanning
  - ✅ Current schema: `(Id, Id, String)` - CORRECT

- **Values** (EdgeCfValue): Can have any structure
  - ✅ Current schema: `(Id, String, Id, Summary)` - FINE (values not used for sorting)

## Recommendations

### 1. Verify with Test

The existing test `test_forward_edges_keys_lexicographically_sortable` (libs/db/src/schema.rs:259) should verify this, but it could be more explicit about the prefix scanning guarantee.

### 2. Add Prefix Scanning Test

Add a test that specifically verifies:
```rust
#[test]
fn test_forward_edges_prefix_scanning() {
    // Create edges:
    // (100, 200, "a")
    // (100, 200, "z")
    // (100, 201, "a")
    // (101, 200, "a")

    // Verify that all keys with source_id=100 are contiguous
    // Verify that keys with (100, 200) prefix are contiguous
}
```

### 3. Documentation

Add comments to make the invariant explicit:
```rust
/// Keys MUST maintain fixed-length prefix structure for efficient RocksDB scanning.
/// Variable-length fields (like EdgeName) MUST be at the END of the key tuple.
/// This ensures lexicographic sorting groups keys by fixed-length prefix.
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) EdgeSourceId,      // Fixed: 16 bytes
    pub(crate) EdgeDestinationId, // Fixed: 16 bytes
    pub(crate) EdgeName,          // Variable: MUST BE LAST
);
```

## Answer to Original Question

> Will having a variable-length element in a tuple affect the prefix scanning in rocksdb?

**Yes, absolutely** - BUT your current schema is already correct!

- ✅ **Correct**: `(Id, Id, String)` - String at END
  - All keys with same `(Id, Id)` prefix are contiguous
  - Prefix scanning works perfectly

- ❌ **Incorrect**: `(Id, String, Id)` - String in MIDDLE
  - Keys with same first `Id` but different string lengths are NOT contiguous
  - Example: `(100, "a", 300)` sorts before `(100, "abc", 100)`
  - Cannot reliably scan by first Id alone

Your insight was **100% correct** - variable-length fields should be at the end. And your schema **already follows this best practice** for the KEY structures. The recent change to EdgeCfValue (with string in middle) is fine because it's the VALUE, not the KEY.

## CRITICAL UPDATE: MessagePack Serialization Analysis

### The Real Problem

After detailed analysis of MessagePack byte-level encoding, **MessagePack serialization for keys is fundamentally problematic** for RocksDB prefix scanning:

#### MessagePack Overhead

For `ForwardEdgeCfKey(Id, Id, String)`:
```
[93]           = array(3) header          (1 byte)
[dc, 00, 10]   = bin16(16) header         (3 bytes)
[00 x 16]      = first Id bytes           (16 bytes)
[dc, 00, 10]   = bin16(16) header         (3 bytes)
[00 x 16]      = second Id bytes          (16 bytes)
[a1]           = fixstr(1) header         (1 byte)
[61]           = 'a'                      (1 byte)
Total: 41 bytes
```

**Key insight**: The common prefix for keys with same `(Id, Id)` is **39 bytes**, not 32!

#### The Critical Issue: Non-Constant Prefix Length

MessagePack string headers vary by length:
- `fixstr(0-31)`: 1 byte header (e.g., `[a1]` for 1-char string)
- `str8(32-255)`: 2 bytes header (e.g., `[d9, 20]` for 32-char string)
- `str16(256-65535)`: 3 bytes header (e.g., `[da, 01, 00]` for 256-char string)

**This means**:
- Keys with `(Id1, Id2, "a")` have a 39-byte prefix
- Keys with `(Id1, Id2, "abc")` have a 39-byte prefix
- But keys with `(Id1, Id2, "a_32_character_string_here...")` have a **40-byte** prefix!

**For RocksDB prefix extractors, you need a CONSTANT prefix length**. MessagePack makes this impossible.

### Direct Byte Concatenation Comparison

```rust
fn direct_key_encoding(id1: &[u8; 16], id2: &[u8; 16], name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(id1);    // 16 bytes
    bytes.extend_from_slice(id2);    // 16 bytes
    bytes.extend_from_slice(name.as_bytes());  // variable
    bytes
}
```

Result:
```
Key (Id([0;16]), Id([0;16]), "a")   = [00 x 32] + [61]           (33 bytes)
Key (Id([0;16]), Id([0;16]), "abc") = [00 x 32] + [61, 62, 63]  (35 bytes)
```

**Constant prefix**: Exactly **32 bytes** for any `(Id, Id)` prefix, regardless of string length!

### Current Implementation Analysis

From `libs/db/src/graph.rs:615-630`:
```rust
let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
for item in iter {
    let (key_bytes, _value_bytes) = item?;
    let key: schema::ForwardEdgeCfKey = schema::ForwardEdges::key_from_bytes(&key_bytes)?;

    let source_id = key.0 .0;
    if source_id == id {
        edges.push((source_id, key.2, dest_id));
    } else if source_id > id {
        break;  // ✅ This works because of lexicographic ordering
    }
}
```

**Current status**:
- ✅ The code correctly assumes lexicographic ordering
- ✅ Early termination (`source_id > id`) is valid
- ❌ **BUT**: Iterates from Start, scanning ENTIRE column family until finding target ID
- ❌ **PERFORMANCE**: O(N) scan where N = total keys in column family
- ❌ **NO PREFIX OPTIMIZATION**: Cannot use RocksDB prefix seek due to non-constant prefix length

### Performance Impact

Without prefix optimization:
- Must scan from beginning of column family every time
- For 1M edges, finding edges for a node near the end requires ~1M key deserializations
- Cannot leverage RocksDB's prefix bloom filters
- Cannot use `seek_for_prev` or `seek` to jump directly to target prefix

### Solutions

#### Option 1: Keep MessagePack, Accept Performance Limitation
✅ Current implementation is **correct**
✅ Maintains compatibility with existing data
❌ O(N) prefix scan performance (no optimization possible)
❌ Cannot use RocksDB prefix extractors

**When acceptable**: Small datasets (< 100K keys per column family)

#### Option 2: Switch to Direct Byte Concatenation (RECOMMENDED)
✅ Constant 32-byte prefix for `(Id, Id)`
✅ Can configure RocksDB prefix extractor
✅ Can use prefix bloom filters for O(1) prefix seek
✅ Dramatically better performance for large datasets
❌ Requires data migration from MessagePack to direct encoding
❌ Loses MessagePack self-describing format (minor - keys are internal)

**Implementation**:
```rust
pub(crate) trait ColumnFamilyRecord {
    // For KEYS: use direct byte concatenation
    fn key_to_bytes_direct(key: &Self::Key) -> Vec<u8>;
    fn key_from_bytes_direct(bytes: &[u8]) -> Result<Self::Key>;

    // For VALUES: keep MessagePack (values don't need prefix scanning)
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(value)
    }
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}
```

Then configure RocksDB:
```rust
// For forward_edges and reverse_edges column families
let mut opts = Options::default();
opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
```

This enables:
```rust
// Seek directly to prefix instead of scanning from start
let prefix = [id1_bytes, id2_bytes].concat(); // 32 bytes
let iter = db.iterator_cf(cf, rocksdb::IteratorMode::From(&prefix, Direction::Forward));
// Now only iterates over matching keys!
```
