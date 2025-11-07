# Variable-Length Fields in MessagePack Keys: RocksDB Implications

## Question

> Will having a variable-length element in a tuple affect the prefix scanning in RocksDB? For example, a key tuple of `(Id, String, Id)` could result in unpredictable sorting order in RocksDB due to the extra string length in the second element of the String in the tuple. Can you verify if it's better to arrange the tuple so that the element with variable length (e.g. the String) should always be the last, so that at least prefix scanning based on the fix-length elements of the tuple will work correctly?

## Answer: YES - Variable-Length Fields MUST Be At The End

Your insight is **100% correct**. Variable-length fields in the middle of a key tuple **break prefix scanning** in RocksDB.

## Why This Matters

### RocksDB Sorting Behavior

RocksDB stores keys in **lexicographic (byte-by-byte) order**. When using MessagePack serialization:

1. Each element in a tuple is serialized sequentially
2. Strings include their length in the encoding
3. Shorter strings sort before longer strings with the same prefix
4. This affects the position of subsequent fixed-length fields

### Demonstration

#### ❌ Bad Pattern: String in MIDDLE `(Id, String, Id)`

```rust
// Serialized keys:
Key (100, "a",   300) => [93, 64, a1, 61, cd, 01, 2c]  // 7 bytes
Key (100, "ab",  200) => [93, 64, a2, 61, 62, cc, c8]  // 7 bytes
Key (100, "abc", 100) => [93, 64, a3, 61, 62, 63, 64]  // 7 bytes

// Lexicographic sort order:
1. (100, "a",   300)  // 0xa1 < 0xa2 (string length byte differs)
2. (100, "ab",  200)  // 0xa2 < 0xa3
3. (100, "abc", 100)  // 0xa3

// PROBLEM: Third Id value (300, 200, 100) doesn't control sort order!
// The string length byte (0xa1, 0xa2, 0xa3) sorts first.
```

**Impact on prefix scanning:**
- ⚠️ Cannot scan by `(Id)` prefix alone
- ⚠️ Keys with `Id=100` are NOT contiguous when strings have different lengths
- ⚠️ Must scan entire keyspace and filter in application code

#### ✅ Good Pattern: String at END `(Id, Id, String)`

```rust
// Serialized keys:
Key (100, 199, "abc") => [93, 64, cc, c7, a3, 61, 62, 63]  // 8 bytes
Key (100, 200, "a")   => [93, 64, cc, c8, a1, 61]           // 6 bytes
Key (100, 200, "ab")  => [93, 64, cc, c8, a2, 61, 62]       // 7 bytes
Key (100, 200, "abc") => [93, 64, cc, c8, a3, 61, 62, 63]  // 8 bytes
Key (100, 201, "a")   => [93, 64, cc, c9, a1, 61]           // 6 bytes

// Lexicographic sort order:
1. (100, 199, "abc")  // 0xc7 < 0xc8 (second Id sorts first)
2. (100, 200, "a")    // 0xc8, then string doesn't matter
3. (100, 200, "ab")   // 0xc8, same prefix
4. (100, 200, "abc")  // 0xc8, same prefix
5. (100, 201, "a")    // 0xc9 > 0xc8

// SUCCESS: Fixed-length Ids control sort order!
// All keys with (100, 200) are contiguous regardless of string length.
```

**Impact on prefix scanning:**
- ✅ Can efficiently scan by `(Id)` prefix
- ✅ Can efficiently scan by `(Id, Id)` prefix
- ✅ All keys with same prefix are guaranteed contiguous
- ✅ RocksDB can optimize range scans

## Current Schema Status: ✅ CORRECT

### Key Structures (Used for RocksDB Sorting)

```rust
// libs/db/src/schema.rs:179-183
ForwardEdgeCfKey(
    EdgeSourceId,      // Position 0: Id (fixed 16 bytes)
    EdgeDestinationId, // Position 1: Id (fixed 16 bytes)
    EdgeName,          // Position 2: String (variable) - AT END ✅
)

// libs/db/src/schema.rs:217-221
ReverseEdgeCfKey(
    EdgeDestinationId, // Position 0: Id (fixed 16 bytes)
    EdgeSourceId,      // Position 1: Id (fixed 16 bytes)
    EdgeName,          // Position 2: String (variable) - AT END ✅
)
```

**Status**: ✅ **CORRECT** - Variable-length field at the end

**Benefits**:
- ✅ Efficient prefix scanning by source_id (ForwardEdges)
- ✅ Efficient prefix scanning by dest_id (ReverseEdges)
- ✅ Efficient range queries on `(src_id, dst_id)` pairs
- ✅ Early termination optimization in `get_edges_from_node_by_id()` works correctly

### Value Structures (Not Used for Sorting)

```rust
// libs/db/src/schema.rs:94-98
EdgeCfValue(
    SrcId,       // Position 0: Id
    EdgeName,    // Position 1: String - IN MIDDLE (but this is OK!)
    DstId,       // Position 2: Id
    EdgeSummary, // Position 3: Variable
)
```

**Status**: ✅ **FINE** - This is a VALUE, not a KEY

**Why this is OK**:
- Values are opaque to RocksDB - only keys are sorted
- Values are not used for prefix scanning or range queries
- Internal field order doesn't affect database performance
- Field order can prioritize semantic clarity or access patterns

## Performance Implications

### With String at END (Current Schema)

```rust
// Query: Get all edges from node with source_id = 100

let iter = db.iterator_cf(cf, IteratorMode::Start);
for (key_bytes, value_bytes) in iter {
    let key = deserialize(key_bytes);
    if key.source_id == 100 {
        // Process edge
    } else if key.source_id > 100 {
        break;  // ✅ Early termination - all remaining keys have source_id > 100
    }
}
```

**Performance**: O(N) where N = number of edges from source_id=100
- ✅ Optimal - only scans relevant keys
- ✅ Can break early when source_id changes
- ✅ RocksDB can use bloom filters and prefix bloom filters

### With String in MIDDLE (Hypothetical Bad Schema)

```rust
// Query: Get all edges from node with source_id = 100

let iter = db.iterator_cf(cf, IteratorMode::Start);
for (key_bytes, value_bytes) in iter {
    let key = deserialize(key_bytes);
    if key.source_id == 100 {
        // Process edge
    }
    // ❌ Cannot break early - keys with source_id=100 are scattered
}
```

**Performance**: O(K) where K = TOTAL number of edges in database
- ❌ Must scan entire keyspace
- ❌ Cannot use early termination
- ❌ RocksDB prefix optimizations don't help

## Test Coverage

### Existing Test

`test_forward_edges_keys_lexicographically_sortable()` (libs/db/src/schema.rs:259) verifies:
- ✅ Keys sort correctly by (source_id, dest_id, name)
- ✅ Lexicographic byte order matches semantic order
- ✅ All tests pass

### Verification Command

```bash
cd /Users/dchung/projects/github.com/chungers/motlie
cargo test --lib test_forward_edges_keys_lexicographically_sortable
```

Result: ✅ PASSING

## Best Practices

### Rule: Variable-Length Last (VLL)

When designing RocksDB keys with MessagePack serialization:

1. ✅ **DO**: Place fixed-length fields first
   - Example: `(Id, Id, String)`
   - Benefits: Efficient prefix scanning, early termination

2. ❌ **DON'T**: Place variable-length fields in the middle
   - Example: `(Id, String, Id)`
   - Problems: No prefix scanning, full table scans required

3. ✅ **DO**: Document the invariant
   ```rust
   /// INVARIANT: Variable-length field MUST be last for prefix scanning
   struct MyKey(FixedField1, FixedField2, VariableField);
   ```

4. ✅ **DO**: Test lexicographic ordering
   ```rust
   #[test]
   fn test_key_ordering() {
       // Verify keys sort as expected
       // Verify prefix scanning assumptions hold
   }
   ```

## Conclusion

### Your Insight is Correct

> "Can you verify if it's better to arrange the tuple so that the element with variable length (e.g. the String) should always be the last"

**Answer**: **YES, absolutely.** Variable-length fields should ALWAYS be at the end of key tuples for RocksDB.

### Current Status

- ✅ **ForwardEdgeCfKey**: String at end - CORRECT
- ✅ **ReverseEdgeCfKey**: String at end - CORRECT
- ✅ **EdgeCfValue**: Can be any order (it's a value, not a key)
- ✅ **Tests**: Passing
- ✅ **Performance**: Optimal

### No Action Needed

The current schema already follows best practices. The recent change to `EdgeCfValue` (putting string in middle) is fine because it's the VALUE, not the KEY.

## References

- MessagePack specification: https://github.com/msgpack/msgpack/blob/master/spec.md
- RocksDB prefix iteration: https://github.com/facebook/rocksdb/wiki/Prefix-Seek
- Test demonstration: `/tmp/msgpack_test/src/main.rs`
