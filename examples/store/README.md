# Store Example - Consumer Chaining Demo

This example demonstrates the mutation consumer chaining feature where mutations flow through multiple processors in sequence.

## Architecture

The example sets up a chain: **Writer → Graph → FullText**

- Mutations are sent to the Graph consumer
- Graph processes each mutation (simulates graph storage)
- Graph forwards the mutation to FullText
- FullText processes the mutation (simulates full-text search indexing)

## Running the Example

```bash
# Run with sample data
RUST_LOG=info cargo run --example store < examples/store/sample_data.csv

# Or pipe your own CSV data
echo "Node1,Fragment for Node1" | RUST_LOG=info cargo run --example store
```

## CSV Format

The example accepts two formats:

1. **Node with fragment (2 fields):**
   ```csv
   node_name,fragment text for the node
   ```

2. **Edge with fragment (4 fields):**
   ```csv
   source_node,target_node,edge_name,fragment text for the edge
   ```

## Expected Log Output

When you run the example, you should see logs showing that Graph processes mutations **before** FullText:

```
[Graph] Would insert vertex: AddVertexArgs { id: ..., name: "Alice" }
[Graph] Would insert fragment: AddFragmentArgs { id: ..., body: "Alice is..." }
[FullText] Would index vertex for search: id=..., name='Alice', k1=1.2, b=0.75
[FullText] Would index fragment content: id=..., body_len=55, k1=1.2, b=0.75
```

Notice how:
- `[Graph]` logs appear first for each mutation
- `[FullText]` logs appear second for the same mutation
- This demonstrates the chaining: Graph → FullText

## Key Code Sections

### Setting up the chain:

```rust
// Create FullText consumer (end of chain)
let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
let fulltext_handle = spawn_fulltext_consumer(fulltext_receiver, config.clone());

// Create Graph consumer that forwards to FullText
let (writer, graph_receiver) = create_mutation_writer(config.clone());
let graph_handle = spawn_graph_consumer_with_next(graph_receiver, config, fulltext_sender);
```

### Sending mutations (they flow through the chain automatically):

```rust
// Send to the writer - goes to Graph, then forwarded to FullText
writer.add_vertex(vertex_args).await?;
writer.add_fragment(fragment_args).await?;
```

## Benefits of Chaining

1. **Single send point**: Write mutations once, they flow through multiple processors
2. **Ordered processing**: Graph storage happens before full-text indexing
3. **Automatic forwarding**: No need to manually send to multiple consumers
4. **Easy to extend**: Can add more processors to the chain (e.g., → Graph → FullText → Logger → Analytics)
