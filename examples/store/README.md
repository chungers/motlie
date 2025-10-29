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

## Data Generation and Visualization Tools

### Generate Sample Data

Use `generate_data.py` to create a diverse dataset with 1,000 nodes and ~5,000 edges:

```bash
python3 generate_data.py
```

This generates `sample_data.csv` with:
- **45% Professional nodes/edges**: People, Companies, Projects with work relationships
- **45% Social nodes/edges**: People, Events, Hobby Groups with social connections
- **7% Things**: Homes, Vehicles with ownership relationships
- **3% Events**: Trips, Conferences with attendance relationships

Features:
- Referentially consistent data (all edges reference valid nodes)
- Descriptive fragments providing rich context for each node and edge
- Equal distribution across categories
- Cross-domain relationships (e.g., professional people attending social events)

### Visualize the Graph

Two options for visualizing your CSV data:

#### Option 1: Interactive HTML Viewer (Recommended)

Open `graph_viewer.html` in any web browser:

```bash
open graph_viewer.html
```

Then drag and drop your CSV file or click to upload it. Features:
- **No dependencies**: Works entirely in the browser
- **Drag & drop**: Upload any CSV file
- **Interactive**: Pan, zoom, click nodes/edges
- **Search**: Find nodes by name or content
- **Color-coded**: Different colors for each node type
- **Node sizing**: Size based on number of connections
- **Details panel**: Click nodes to see fragments and connections

#### Option 2: Generate Static HTML

Use the Python script to pre-generate an HTML file:

```bash
python3 visualize_graph.py [input.csv] [output.html]

# Default usage (reads sample_data.csv, writes graph_visualization.html)
python3 visualize_graph.py

# Open the generated file
open graph_visualization.html
```

This creates a self-contained HTML file with the data embedded.

### Visualization Features

- **Pan and Zoom**: Click and drag to move around, scroll to zoom
- **Node Details**: Click any node to see:
  - Node type and name
  - Full fragment text
  - All connected edges with types
  - Connected node names
- **Edge Details**: Click any edge to see:
  - Edge type and relationship
  - Fragment describing the relationship
  - Source and target node information
- **Search**: Type-ahead search to quickly find specific nodes
- **Physics Simulation**: Nodes automatically arrange themselves based on connections
- **Controls**:
  - Reset View: Return to default zoom
  - Fit All: Adjust view to show entire graph
  - Navigation buttons: Built-in zoom controls

## Design Documentation

### generate_data.py Design

The data generator creates realistic, referentially-consistent graph data with configurable distributions.

#### Architecture

**Configuration System:**
- Dictionary-based distribution configuration (`CATEGORY_DISTRIBUTION`, `PROFESSIONAL_DISTRIBUTION`, etc.)
- Automatic validation ensures all distributions sum to 1.0 within tolerance
- Percentage-based calculations allow easy dataset scaling

**Key Components:**

1. **Distribution Dictionaries** (Lines 27-59):
   ```python
   CATEGORY_DISTRIBUTION = {
       'PROFESSIONAL': 0.45,  # 45% - Work-related entities
       'SOCIAL': 0.45,        # 45% - Social and personal entities
       'THINGS': 0.07,        # 7%  - Physical possessions
       'EVENTS': 0.03,        # 3%  - Travel and conferences
   }
   ```

2. **Validation Function** (Lines 62-73):
   - Ensures distributions sum to 1.0
   - Configurable tolerance (default: 0.001)
   - Fails fast on module import if invalid

3. **Runtime Computation** (Lines 75-203):
   - `compute_node_counts(total_nodes)` - Calculates node counts from percentages
   - `compute_edge_counts(node_counts)` - Calculates edge counts from node counts
   - All counts scale proportionally with `total_nodes`

4. **DataGenerator Class** (Lines 374+):
   - Takes computed counts in `__init__`
   - Generates nodes and edges based on counts
   - Maintains referential integrity (all edges reference valid nodes)

#### Command-Line Interface

```bash
# Generate to stdout (default)
python3 generate_data.py -n 1000

# Redirect to file
python3 generate_data.py -n 5000 > data.csv

# Write directly to file
python3 generate_data.py -n 500 -o small_data.csv

# Custom random seed
python3 generate_data.py --seed 123
```

**Key Design Decisions:**
- **Stdout by default**: Follows Unix philosophy, enables piping
- **Stderr for logs**: All diagnostics go to stderr, keeping stdout clean
- **Scalable**: Change `TOTAL_NODES` and all proportions maintain

#### Data Model

**Node Types:**
- Professional: People, Companies, Projects
- Social: People, Events, Hobby Groups
- Things: Homes, Vehicles
- Events: Trips, Conferences

**Edge Types:**
- Professional: works_at, manages, collaborates_with, funds
- Social: friends_with, attended, member_of, traveled_on
- Ownership: owns
- Cross-domain: knows, mentors, sponsors, hosts

#### Customization

To modify distributions, edit the dictionaries at the top:

```python
# Change category mix
CATEGORY_DISTRIBUTION = {
    'PROFESSIONAL': 0.60,  # 60% professional
    'SOCIAL': 0.30,        # 30% social
    'THINGS': 0.05,        # 5% things
    'EVENTS': 0.05,        # 5% events
}

# Change professional breakdown
PROFESSIONAL_DISTRIBUTION = {
    'PEOPLE': 0.50,     # More people
    'COMPANIES': 0.25,  # Fewer companies
    'PROJECTS': 0.25,
}
```

Validation will fail if distributions don't sum to 1.0.

### graph_viewer.html Design

A standalone, client-side graph visualization tool built with vis.js.

#### Architecture

**Single-File Design:**
- No external dependencies (vis.js embedded)
- Works entirely in browser
- No server required

**Key Components:**

1. **File Upload System** (Lines 150-250):
   - Drag & drop support
   - CSV parsing with quoted field handling
   - Automatic detection of node vs edge rows (2 vs 4 fields)

2. **Graph Rendering** (vis.js Network):
   - Force-directed layout using Barnes-Hut simulation
   - Physics disabled after initial stabilization (performance)
   - Adaptive rendering based on zoom level

3. **Search System** (Lines 450-550):
   - Type-ahead search for nodes and edges
   - Toggle between node/edge search modes
   - Click results to focus and highlight

4. **Interactive Features**:
   - Click nodes/edges to view details
   - Clickable references for navigation
   - Context highlighting (dims non-related elements)

#### Performance Optimizations

**For Large Graphs (1000+ nodes):**

1. **Physics Optimization** (Lines 820-837):
   - Reduced stabilization iterations (100 vs 200)
   - Physics disabled after layout completes
   - Adaptive timestep enabled

2. **Rendering Optimization** (Lines 768-818):
   - Shadows disabled (expensive with many nodes)
   - Straight edges (faster than curved)
   - Level-of-detail labels (appear when zoomed in)
   - Edges hidden during drag/zoom operations

3. **Context Highlighting** (Lines 909-1032):
   - Batch updates using arrays (not individual)
   - Selected node + 1 degree neighbors at full opacity
   - Other elements at 15% opacity for visual focus
   - Efficient adjacency map for O(1) neighbor lookup

#### Visual Design

**Color Coding:**
- Each node type gets a distinct color
- Highlighted elements: Full color and opacity
- Background elements: 85% transparent
- Edge labels: Smaller font (Arial 9pt) vs node labels (monospace 12pt)

**Layout:**
- Top bar: Title, file upload, controls
- Main canvas: Interactive graph visualization
- Right panel: Details for selected node/edge
- Search: Integrated with toggle for nodes/edges

#### Key Files

```
examples/store/
├── generate_data.py        # Data generator with configurable distributions
├── graph_viewer.html        # Standalone interactive viewer (~45KB)
├── sample_data.csv          # Generated sample dataset (1000 nodes)
└── README.md               # This file
```

#### Browser Compatibility

- Chrome: Full support, best performance
- Safari: Full support
- Firefox: Full support
- Edge: Full support

**Note:** For datasets >10,000 nodes, consider using dedicated graph visualization tools like Gephi or Cytoscape.
