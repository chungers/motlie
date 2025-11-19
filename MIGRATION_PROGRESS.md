# Migration Progress: Remove Edges CF + Add Edge Weights

## Branch: `remove-edges-cf-add-weights`

## Completed ✅

### schema.rs (Partial)
- ✅ Removed Edges CF struct and impl
- ✅ Renamed Fragments → NodeFragments
- ✅ Updated ForwardEdgeCfValue: Added Option<f64> weight and EdgeSummary
- ✅ Updated ReverseEdgeCfValue: Removed edge_id (now just temporal_range)
- ✅ Added EdgeFragments CF struct definitions
- ✅ Updated imports to use AddNodeFragment, AddEdgeFragment

## In Progress ⏳

### schema.rs
- ⏳ Need to update ForwardEdges::record_from() - currently broken
- ⏳ Need to update ReverseEdges::record_from() - currently broken
- ⏳ Need to implement EdgeFragments::ColumnFamilyRecord
- ⏳ Need to update ALL_COLUMN_FAMILIES constant
- ⏳ Need to remove EdgeNames CF (uses edge_id)
- ⏳ Need to update ValidRangePatchable for ForwardEdges (field indices changed)

## TODO 📝

### mutation.rs
- [ ] Define AddNodeFragment struct (renamed from AddFragment)
- [ ] Define AddEdgeFragment struct (new)
- [ ] Update AddEdge struct (remove `id`, add `summary` and `weight`)
- [ ] Update UpdateEdgeValidSinceUntil (use topology instead of id)
- [ ] Add UpdateEdgeWeight mutation
- [ ] Update Mutation enum

### graph.rs
- [ ] Update PatchEdgeValidRange (use topology, not edge_id)
- [ ] Update PatchNodeValidRange (use new PatchEdgeValidRange signature)
- [ ] Update StorageOperation enum

### query.rs
- [ ] Remove EdgeById query
- [ ] Update EdgeSummaryBySrcDstName return type
- [ ] Update OutgoingEdges/IncomingEdges (handle new value structure)
- [ ] Add OutgoingEdgesWithWeights query
- [ ] Add IncomingEdgesWithWeights query
- [ ] Rename FragmentsByIdTimeRange → NodeFragmentsByIdTimeRange
- [ ] Add EdgeFragmentsByTopology query

### lib.rs
- [ ] Remove exports: EdgeById, AddFragment
- [ ] Add exports: AddNodeFragment, AddEdgeFragment, UpdateEdgeWeight
- [ ] Add exports: OutgoingEdgesWithWeights, EdgeFragmentsByTopology
- [ ] Update other exports as needed

### Tests
- [ ] Update all AddEdge calls (remove id, add summary/weight)
- [ ] Update fragment tests (AddNodeFragment/AddEdgeFragment)
- [ ] Update UpdateEdgeValidSinceUntil tests
- [ ] Remove EdgeById tests
- [ ] Add weight-related tests
- [ ] Update graph_tests.rs
- [ ] Update fulltext_tests.rs if needed

### Examples
- [ ] Update examples/store/main.rs for new API
- [ ] Update verification logic

### Documentation
- [ ] Update database_design.md
- [ ] Update FINAL_migration_plan.md

## Current Build Status

**Expected:** Will not compile until mutations are defined
- AddNodeFragment not defined
- AddEdgeFragment not defined
- AddEdge still has old fields
- ForwardEdges/ReverseEdges record_from() implementations broken

## Next Steps

1. **Define mutation types** in mutation.rs
2. **Update schema.rs** implementations to use new mutation types
3. **Update graph.rs** storage operations
4. **Update query.rs** queries
5. **Fix all tests**
6. **Verify build and tests pass**

## Estimated Remaining Work

- Mutations: 2 hours
- Schema completion: 1 hour
- Storage operations: 2 hours
- Queries: 3 hours
- Tests: 4 hours
- Examples: 1 hour
- Verification: 2 hours

**Total remaining: ~15 hours**
