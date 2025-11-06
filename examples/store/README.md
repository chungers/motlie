# Store Example - Consumer Chaining Demo

This example demonstrates the mutation consumer chaining feature where mutations flow through multiple processors in sequence.

## Architecture

The example sets up a chain: **Writer → Graph → FullText**

- Mutations are sent to the Graph consumer
- Graph processes each mutation (simulates graph storage)
- Graph forwards the mutation to FullText
- FullText processes the mutation (simulates full-text search indexing)

## Building the Example

First, build the example binary:

```bash
cargo build --example store --release
```

The binary will be located at:
- `target/release/examples/store`

For development builds (faster compilation, slower execution):

```bash
cargo build --example store
```

Development binary is at:
- `target/debug/examples/store`

## Running the Example

The `store` binary operates in two modes:

### Store Mode (default)

Reads CSV from stdin and stores data in the database:

```bash
cat input.csv | target/release/examples/store <db_path>
```

### Verify Mode

Reads CSV from stdin and verifies it against existing database:

```bash
cat input.csv | target/release/examples/store --verify <db_path>
```

### Command Line Arguments

```bash
target/release/examples/store [--verify] <db_path>
```

- `--verify` (optional): Enable verification mode
- `db_path` (required): Path to RocksDB database directory

**Note:** CSV data is always read from stdin in both modes.

### With Generated Data (Recommended)

```bash
# Generate a dataset with 100 nodes (~1000 edges)
python3 generate_data.py --total-nodes 100 2>/dev/null > /tmp/test_data.csv

# Store the data
cat /tmp/test_data.csv | target/release/examples/store /tmp/motlie_graph_db

# Verify data was stored correctly
cat /tmp/test_data.csv | target/release/examples/store --verify /tmp/motlie_graph_db
```

### With Small Test Data

```bash
# Use the small test input file (3 nodes, 3 edges)
cat examples/store/test_input.csv | target/release/examples/store /tmp/motlie_graph_db
```

### With Logging

```bash
# See detailed processing logs
python3 generate_data.py --total-nodes 50 2>/dev/null > /tmp/test_data.csv
RUST_LOG=info cat /tmp/test_data.csv | target/release/examples/store /tmp/motlie_graph_db
```

## Verifying Data Persistence

After storing data, verify that all data was correctly written to RocksDB using the `--verify` flag:

```bash
# Verify database contents match the CSV input
cat /tmp/test_data.csv | target/release/examples/store --verify /tmp/motlie_graph_db
```

The verification mode performs comprehensive checks:
- Reads CSV from stdin to extract expected nodes, edges, and fragments
- Opens the RocksDB database in read-only mode
- Performs detailed verification on each data type:
  - **Nodes**: Verifies count and that all expected node names are present
  - **Edges**: Verifies count and validates each edge's source node, target node, and edge name
  - **Fragments**: Verifies count and that all expected fragment content is present
- Reports success or failure with detailed error messages showing exactly what doesn't match

### Successful Verification Output

```
Motlie Store Verifier
====================

Database: /tmp/motlie_graph_db
Reading CSV from stdin...

📄 Parsing CSV from stdin...
   Nodes: 99
   Edges: 964
   Total fragments: 1050

✓ Database opened successfully

🔍 Verifying Nodes...
   Expected: 99 nodes
   Found:    99 nodes
   ✓ Node count matches
   ✓ All expected node names found in database

🔍 Verifying Edges...
   Expected: 964 edges
   Found:    964 edges
   ✓ Edge count matches
   ✓ All expected edges found with correct source, target, and name

🔍 Verifying Fragments...
   Expected: at least 1050 fragments
   Found:    1063 fragments
   ✓ Fragment count OK (database may have additional fragments for implicit nodes)
   ✓ All expected fragment content found in database

✅ All verification checks passed!
   The database contents match the CSV input.
```

### Failed Verification Output

When data doesn't match, the verification provides detailed error messages:

```
🔍 Verifying Nodes...
   Expected: 3 nodes
   Found:    3 nodes
   ✓ Node count matches
   ✓ All expected node names found in database

🔍 Verifying Edges...
   Expected: 1 edges
   Found:    3 edges
   ✗ Edge count mismatch!
   ✗ Missing or mismatched edges:
      - alice -> bob (knows_wrong)

🔍 Verifying Fragments...
   Expected: at least 4 fragments
   Found:    6 fragments
   ✓ Fragment count OK (database may have additional fragments for implicit nodes)
   ✗ 1 expected fragments not found:
      1. Edge 'alice -> bob (knows_wrong)': "Alice knows Bob from university"

❌ Some verification checks failed!
   The database contents do not fully match the CSV input.
```

The verification catches:
- **Incorrect node names**: Reports which expected nodes are missing
- **Wrong edge relationships**: Shows edges with incorrect source, target, or edge name
- **Missing fragment content**: Lists specific fragments that aren't found in the database

### End-to-End Workflow

```bash
# 1. Build the example
cargo build --example store --release

# 2. Clean any existing database
rm -rf /tmp/motlie_graph_db

# 3. Generate test data
python3 generate_data.py --total-nodes 100 2>/dev/null > /tmp/test_data.csv

# 4. Store the data
cat /tmp/test_data.csv | target/release/examples/store /tmp/motlie_graph_db

# 5. Verify the data was written correctly
cat /tmp/test_data.csv | target/release/examples/store --verify /tmp/motlie_graph_db
```

This demonstrates the complete write path: **CSV Input → Writer → Graph Processor → RocksDB Storage → Verification** ✓

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

Use `generate_data.py` to create a diverse, realistic dataset:

```bash
# Generate 1000 nodes (default) to stdout
python3 generate_data.py

# Generate custom number of nodes
python3 generate_data.py --total-nodes 5000

# Redirect to file
python3 generate_data.py -n 1000 > my_data.csv

# Use custom random seed for reproducibility
python3 generate_data.py --seed 42
```

This generates a CSV dataset with:
- **45% Professional nodes**: People, Companies, Projects, Artifacts, Deliverables, Milestones
- **45% Social nodes**: People, Events, Hobby Groups with social connections
- **7% Things**: Homes, Vehicles with ownership relationships
- **3% Events**: Trips, Conferences with attendance relationships
- **~10 edges per node**: Rich interconnected graph structure

Features:
- **Project Management Structure**: Each project has 5-20 artifacts, 2-5 deliverables, and 4-8 milestones with expected dates
- **Dependency Graphs**: Artifacts depend on other artifacts (1-3 dependencies each), deliverables require artifacts
- **Sequential Milestones**: Project checkpoints in temporal order (precedes relationships)
- **Work Assignments**: Professional people assigned to artifacts (2-4 per artifact) and deliverables (2-5 per deliverable)
- **Referentially consistent**: All edges reference valid nodes
- **Descriptive fragments**: Rich context for each node and edge
- **Cross-domain relationships**: Professional people in hobby groups, social people at conferences, etc.

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

The data generator creates realistic, referentially-consistent graph data with rich project management structures and configurable distributions.

#### Architecture

**Configuration System:**
- Dictionary-based distribution configuration (`CATEGORY_DISTRIBUTION`, `PROFESSIONAL_DISTRIBUTION`, etc.)
- Automatic validation ensures all distributions sum to 1.0 within tolerance
- Percentage-based calculations allow easy dataset scaling
- Edge density configured to achieve ~10 edges per node

**Key Components:**

1. **Distribution Dictionaries** (Lines 26-72):
   ```python
   CATEGORY_DISTRIBUTION = {
       'PROFESSIONAL': 0.45,  # 45% - Work-related entities
       'SOCIAL': 0.45,        # 45% - Social and personal entities
       'THINGS': 0.07,        # 7%  - Physical possessions
       'EVENTS': 0.03,        # 3%  - Travel and conferences
   }

   PROFESSIONAL_DISTRIBUTION = {
       'PEOPLE': 0.17,        # 17% - Individual workers
       'COMPANIES': 0.10,     # 10% - Organizations
       'PROJECTS': 0.05,      # 5% - Work initiatives
       'ARTIFACTS': 0.40,     # 40% - Project artifacts (5-20 per project)
       'DELIVERABLES': 0.10,  # 10% - Project deliverables (2-5 per project)
       'MILESTONES': 0.18,    # 18% - Project milestones (4-8 per project)
   }
   ```

2. **Validation Function** (Lines 61-72):
   - Ensures distributions sum to 1.0
   - Configurable tolerance (default: 0.001)
   - Fails fast on module import if invalid

3. **Runtime Computation** (Lines 76-230):
   - `compute_node_counts(total_nodes)` - Calculates node counts from percentages
   - `compute_edge_counts(node_counts)` - Calculates edge counts from node counts
   - All counts scale proportionally with `total_nodes`

4. **DataGenerator Class** (Lines 437+):
   - Takes computed counts in `__init__`
   - Generates nodes and edges based on counts
   - Maintains referential integrity (all edges reference valid nodes)
   - Creates rich project structures with dependency graphs

#### Command-Line Interface

```bash
# Generate to stdout (default, 1000 nodes)
python3 generate_data.py

# Generate custom number of nodes
python3 generate_data.py --total-nodes 5000

# Redirect to file
python3 generate_data.py -n 500 > small_data.csv

# Custom random seed for reproducibility
python3 generate_data.py --seed 123

# Check statistics
python3 generate_data.py -n 100 2>&1 | grep "Dataset Statistics" -A 30
```

**Key Design Decisions:**
- **Stdout by default**: Follows Unix philosophy, enables piping
- **Stderr for logs**: All diagnostics go to stderr, keeping stdout clean
- **Scalable**: Change `--total-nodes` and all proportions maintain
- **Statistics**: Always prints detailed stats to stderr after generation

#### Data Model

**Node Types:**
- **Professional**:
  - `ProfessionalPerson` - Workers with roles and skills
  - `Company` - Organizations with industries
  - `ProfessionalProject` - Work initiatives
  - `Artifact` - Project artifacts (Technical Specs, Code Repos, APIs, Tests, etc.)
  - `Deliverable` - Project deliverables (Requirements Docs, Deployment Packages, Reports, etc.)
  - `Milestone` - Project checkpoints with expected dates (Kickoff, Design Review, Alpha/Beta Releases, etc.)
- **Social**: `SocialPerson`, `SocialEvent`, `HobbyGroup`
- **Things**: `Home`, `Vehicle`
- **Events**: `Trip`, `Conference`

**Edge Types:**
- **Professional**: `works_at`, `manages`, `contributes_to`, `collaborates_with`, `funds`
- **Project Structure**:
  - `has_artifact` - Project → Artifact (5-20 per project)
  - `has_deliverable` - Project → Deliverable (2-5 per project)
  - `has_milestone` - Project → Milestone (4-8 per project)
- **Dependencies**:
  - `depends_on` - Artifact → Artifact (creates dependency graph)
  - `requires` - Deliverable → Artifact (deliverables require artifacts)
  - `precedes` - Milestone → Milestone (temporal ordering)
- **Assignments**: `assigned_to` - Person → Artifact/Deliverable
- **Social**: `friends_with`, `attended`, `member_of`, `traveled_on`
- **Ownership**: `owns`
- **Cross-domain**: `knows`, `mentors`, `sponsors`, `hosts`, `meets_at`, `uses`, etc.

#### Project Management Features

**Artifact Generation** (Lines 544-559):
- Each project gets 5-20 artifacts
- Types: Technical Specifications, Design Documents, Code Repositories, Test Suites, API Documentation, Database Schemas, Configuration Files, Build Scripts
- Artifacts include version numbers (e.g., v1.2, v2.8)
- Status tracking: "in progress", "completed", "under review"

**Deliverable Generation** (Lines 561-575):
- Each project gets 2-5 deliverables
- Types: Requirements Documents, Implementation Plans, System Architecture, User Documentation, Training Materials, Deployment Packages, Performance Reports, Security Audits
- Status tracking: "draft", "final", "approved"

**Milestone Generation** (Lines 577-596):
- Each project gets 4-8 milestones
- Types: Project Kickoff, Requirements Freeze, Design Review, Alpha Release, Beta Release, Code Complete, Quality Assurance, Production Deployment, Project Closure
- **Expected dates included in fragments**: "Expected completion date: June 27, 2024"
- Status tracking: "upcoming", "achieved", "in progress"

**Dependency Graph Construction** (Lines 1051-1148):
1. Artifacts within same project depend on each other (80% probability, 1-3 dependencies each)
2. Deliverables require artifacts (90% probability, 1-3 artifacts each)
3. Milestones precede next milestone in sequence (linear temporal chain)
4. Professional people assigned to artifacts (90% probability, 2-4 people each)
5. Professional people assigned to deliverables (95% probability, 2-5 people each)

#### Edge Density Configuration

**Target: ~10 edges per node**

Achieved through:
- **Collaborations**: 8-14 coworkers per professional person
- **Friendships**: 11-20 friends per social person
- **Event Attendance**: 10-18 attendees per social event
- **Artifact Dependencies**: 80% probability with 1-3 dependencies each
- **Deliverable Requirements**: 90% probability with 1-3 artifacts each
- **Work Assignments**: 90-95% of artifacts/deliverables have 2-5 people assigned
- **Cross-domain Connections**: High percentages (70-100%) for inter-domain relationships

Test results show consistent edge density:
- 100 nodes: ~9.8 edges per node
- 500 nodes: ~9.8 edges per node
- 1000 nodes: ~9.8 edges per node

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

# Change professional breakdown (must sum to 1.0)
PROFESSIONAL_DISTRIBUTION = {
    'PEOPLE': 0.20,
    'COMPANIES': 0.15,
    'PROJECTS': 0.10,
    'ARTIFACTS': 0.35,
    'DELIVERABLES': 0.10,
    'MILESTONES': 0.10,
}

# Adjust project artifact counts (Lines 174-179)
NUM_ARTIFACTS_PER_PROJECT_MIN = 5
NUM_ARTIFACTS_PER_PROJECT_MAX = 20
NUM_MILESTONES_PER_PROJECT_MIN = 4
NUM_MILESTONES_PER_PROJECT_MAX = 8

# Adjust edge density (Lines 134-144)
NUM_COLLABORATORS_MIN = 8
NUM_COLLABORATORS_MAX = 14
NUM_FRIENDS_MIN = 11
NUM_FRIENDS_MAX = 20
```

Validation will fail if distributions don't sum to 1.0.

#### Output Statistics

The generator prints comprehensive statistics to stderr:

```
=== Dataset Statistics ===
Total nodes: 500
Total edges: 4915
Average edges per node: 9.83

Node distribution:
  Artifact: 90 (18.0%)
  Deliverable: 45 (9.0%)
  Milestone: 40 (8.0%)
  ProfessionalPerson: 38 (7.6%)
  Company: 22 (4.4%)
  ProfessionalProject: 11 (2.2%)
  ...

Edge type distribution:
  friends_with: 1527
  attended: 1312
  assigned_to: 343
  depends_on: 169
  has_artifact: 90
  has_deliverable: 45
  has_milestone: 40
  precedes: 36
  requires: 32
  ...
```

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
