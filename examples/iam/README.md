# IAM Permissions Graph Analysis

> **API Status (December 2024)**: This example now uses the **unified motlie_db API** (porcelain layer).
> - Storage: `motlie_db::{Storage, StorageConfig, ReadWriteHandles}`
> - Mutations: `motlie_db::mutation::{AddNode, AddEdge, Runnable, NodeSummary, EdgeSummary, ...}`
> - Queries: `motlie_db::query::{OutgoingEdges, IncomingEdges, Runnable, ...}`
>
> Uses the unified `motlie_db::Storage` API which initializes both graph (RocksDB)
> and fulltext (Tantivy) backends. `ReadWriteHandles` provides access to both `reader()` and `writer()`.

A comprehensive cloud IAM (Identity and Access Management) security analysis tool that simulates and analyzes permission structures in cloud environments using graph algorithms.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Graph Model](#graph-model)
4. [Intentional Test Violations](#intentional-test-violations)
5. [Use Cases](#use-cases)
6. [Algorithm Deep Dives](#algorithm-deep-dives)
7. [Running the Examples](#running-the-examples)
8. [Web Visualization](#web-visualization)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Security Applications](#security-applications)
11. [Extending the Example](#extending-the-example)

---

## Overview

Cloud IAM systems are naturally represented as directed graphs where:
- **Nodes** represent entities (users, groups, roles, resources)
- **Edges** represent relationships (membership, permissions, access)

This example demonstrates how graph algorithms can be applied to IAM security analysis, implementing 12 different use cases that answer critical security questions:

- Who can access what?
- What's the blast radius if credentials are compromised?
- Are there unused roles we can clean up?
- Which entities are over-privileged?
- What are the high-value targets to protect?

Each use case is implemented twice:
1. **Reference implementation** using petgraph (in-memory)
2. **motlie_db implementation** using persistent graph storage

---

## Quick Start

### Build

```bash
cargo build --release --example iam_permissions
```

### List Available Use Cases

```bash
cargo run --release --example iam_permissions -- list
```

### Run a Use Case

```bash
# Basic syntax
cargo run --release --example iam_permissions -- <use_case> <impl> <db_path> <scale>

# Example: Run blast radius analysis with reference implementation
cargo run --release --example iam_permissions -- blast_radius reference /tmp/iam_db 50

# Example: Run PageRank analysis with motlie_db
cargo run --release --example iam_permissions -- high_value_targets motlie_db /tmp/iam_db 100
```

### Parameters

| Parameter | Description | Examples |
|-----------|-------------|----------|
| `use_case` | Analysis to run | `reachability`, `blast_radius`, `pagerank` |
| `impl` | Implementation | `reference` (petgraph) or `motlie_db` |
| `db_path` | Database directory | `/tmp/iam_db` |
| `scale` | Graph size multiplier | `20`, `50`, `100`, `500` |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--visualize` | Enable interactive web visualization | disabled |
| `--port <PORT>` | Port for visualization server | `8081` |

### Generate Data Only

To generate the graph data without running any analysis:

```bash
cargo run --release --example iam_permissions -- generate /tmp/iam_db 100
```

This creates the RocksDB database with the IAM graph, which can be used for subsequent queries.

---

## Graph Model

### Node Types

The IAM graph contains 10 different entity types:

| Node Type | Description | Example | Scale Factor |
|-----------|-------------|---------|--------------|
| **User** | Human identities | `user-0042` | 5 × scale |
| **Group** | Collections of users | `group-0003` | ~0.5 × scale |
| **Policy** | Permission definitions | `policy-0015` | 2 × scale |
| **Role** | Assumable identities | `role-0007` | 1 × scale |
| **Workload** | Running services/jobs | `workload-0012` | 2 × scale |
| **VPC** | Virtual networks | `vpc-0002` | ~0.5 × scale |
| **Instance** | Compute resources | `instance-0089` | 3 × scale |
| **Disk** | Storage resources | `disk-0034` | 2 × scale |
| **Database** | Database resources | `database-0005` | ~0.5 × scale |
| **Region** | Geographic regions | `region-us-east` | 5 (fixed) |

### Edge Types

Relationships between entities are modeled as weighted directed edges:

| Edge Type | From → To | Weight | Description |
|-----------|-----------|--------|-------------|
| **MemberOf** | User → Group | 2.0 | User belongs to group |
| **HasPolicy** | User/Group/Role → Policy | 1.5 | Entity has policy attached |
| **Assumes** | Workload → Role | 2.5 | Workload assumes role |
| **CanAccess** | Policy → Resource | 1.0 | Policy grants access |
| **DependsOn** | Resource → Resource | 3.5 | Resource dependency |
| **LocatedIn** | VPC → Region | 0.5 | VPC location |
| **RunsIn** | Workload → Instance | 3.0 | Workload execution location |

### Edge Weights as "Resistance"

Edge weights represent the "resistance" or "difficulty" of traversing a permission path:
- **Lower weight** = easier/more direct access (e.g., `CanAccess` = 1.0)
- **Higher weight** = harder/more indirect access (e.g., `DependsOn` = 3.5)

This enables algorithms like Dijkstra to find the "path of least resistance" - the easiest way for an attacker to reach a target.

### Graph Statistics by Scale

| Scale | Users | Groups | Policies | Roles | Resources | Total Nodes | Total Edges |
|------:|------:|-------:|---------:|------:|----------:|------------:|------------:|
| 20 | 100 | 11 | 40 | 20 | 131 | 333 | ~605 |
| 50 | 250 | 26 | 100 | 50 | 302 | 833 | ~1,513 |
| 100 | 500 | 51 | 200 | 100 | 615 | 1,666 | ~3,018 |
| 200 | 1000 | 101 | 400 | 200 | 1,231 | 3,332 | ~6,036 |

---

## Intentional Test Violations

The graph generator injects intentional security violations into the test data to demonstrate that the analysis algorithms correctly detect real-world issues. These violations are designed to be detected by specific use cases.

### 1. Cross-Region Access Violations

**Detected by**: `cross_region` use case

**What is injected**:
- A "cross-region gateway" instance (`instance-cross-region-gateway`) located in region A that has a `DependsOn` edge to a resource in region B
- A policy (`policy-cross-region-access`) granting access to this gateway
- A user (`user-cross-region-admin`) with this policy attached
- An additional cross-region dependency between existing resources in different regions

**Why this is a violation**:
Cross-region access paths violate data sovereignty and compliance requirements. In real cloud environments:
- **GDPR Compliance**: EU data should not be accessible from non-EU regions
- **Data Residency Laws**: Many countries require data to remain within their borders
- **Security Policy**: Organizations often restrict access to region-local resources only

When the `cross_region` algorithm performs BFS from users, it tracks region metadata at each hop. When it detects a transition from `region-us-east` to `region-us-west` (or any other region crossing), it flags this as a compliance violation.

**Expected detection**:
```
Summary: Found 8 cross-region access paths
Details: region-us-east -> region-us-west (path len: 4)
```

---

### 2. Minimal Privilege Violations

**Detected by**: `minimal_privilege` use case

**What is injected**:
- An "expensive" policy (`policy-expensive-admin`) that uses `DependsOn` edge (weight 3.5) instead of `CanAccess` (weight 1.0) to reach a sensitive resource
- A "cheap" policy (`policy-standard-access`) with normal `CanAccess` edge (weight 1.0)
- A group (`group-standard-users`) that has the cheap policy attached
- Two violation users (`user-non-minimal-path`, `user-suboptimal-access`) with **both**:
  - Direct access to the expensive policy (2 hops, total weight 5.0)
  - Group membership providing indirect access via the cheap policy (3 hops, total weight 4.5)

**Why this is a violation**:
The principle of least privilege requires that access paths be minimal - both in terms of permissions granted AND the complexity/cost of the path. When a user has:

| Path | Hops | Total Weight |
|------|------|--------------|
| User → expensive_policy → resource | 2 | 1.5 + 3.5 = **5.0** |
| User → group → cheap_policy → resource | 3 | 2.0 + 1.5 + 1.0 = **4.5** |

The BFS algorithm finds the 2-hop path first (fewer hops), but Dijkstra finds the 3-hop path is actually cheaper (4.5 < 5.0). This indicates:
- The direct policy grants "heavier" access than needed
- The user could achieve the same access through a lighter-weight group path
- The expensive policy may be over-privileged or misconfigured

**Expected detection**:
```
Summary: Checked 15 paths: 13 minimal, 2 non-minimal (86.7% optimal)
Details:
  user-suboptimal-access -> instance-0000: actual=5.0, optimal=4.5, excess=0.5
  user-non-minimal-path -> instance-0000: actual=5.0, optimal=4.5, excess=0.5
```

---

### 3. Unused Roles Violations

**Detected by**: `unused_roles` use case

**What is injected**:
- Two unused roles (`role-unused-0000`, `role-unused-0001`) that have policies attached but no workload assumes them
- One completely isolated role (`role-isolated-orphan`) with no edges at all

**Why this is a violation**:
Unused roles represent security risk without providing value:

| Role | Has Policies | Assumed by Workloads | Problem |
|------|--------------|---------------------|---------|
| `role-unused-0000` | Yes | No | Grants access but nobody uses it |
| `role-unused-0001` | Yes | No | Grants access but nobody uses it |
| `role-isolated-orphan` | No | No | Completely orphaned, cleanup candidate |

These violations matter because:
- **Attack Surface**: Unused roles with policies could be assumed by a compromised workload for privilege escalation
- **Compliance**: Access paths that exist but aren't used violate least-privilege principles
- **Hygiene**: Orphaned resources clutter the permission graph and make auditing harder
- **Cost**: In cloud environments, unused resources may incur costs

The Kosaraju SCC algorithm identifies these by:
1. Finding roles not in the main strongly connected component
2. Checking which roles have no incoming `Assumes` edges from workloads

**Expected detection**:
```
Summary: Found 3 unused roles: 0 in isolated SCCs, 3 unassumed (total roles: 23)
Details:
  role-isolated-orphan: no workloads assume
  role-unused-0000: no workloads assume
  role-unused-0001: no workloads assume
```

---

### 4. MST Redundancy Test Cases

**Detected by**: `mst` use case

**What is injected**:

**Case A: Redundant High-Weight Policy**
- A policy (`policy-mst-redundant-high`) that uses `DependsOn` edge (weight 3.5) instead of `CanAccess` (weight 1.0)
- A user connected to both this high-weight policy AND a lower-weight group path
- The high-weight path should NOT be in the MST

**Case B: Redundant Resource Dependency Chain**
- A chain of `DependsOn` edges between resources forming a cycle
- resources[1] → resources[2] → resources[3] → resources[0]
- MST should exclude one edge to break the cycle

**Case C: Parallel Role Assumption Paths**
- A role (`role-mst-parallel-high`) using `DependsOn` (weight 3.5) instead of `Assumes` (weight 2.5)
- Creates a parallel higher-weight path to the same workload
- The higher-weight path should be excluded from MST

**Why these are violations**:
Redundant edges increase attack surface without adding connectivity value:
- Every extra edge is a potential attack vector
- MST identifies the minimum edges needed for full connectivity
- Edges outside MST are candidates for permission cleanup

**Expected detection**:
```
Summary: MST: 832 edges (total weight: 1247.50), 681 redundant edges could be removed
```

The injected test cases ensure that high-weight parallel paths are correctly identified as redundant by Kruskal's algorithm.

---

### Verifying Violations Are Detected

Run the interactive server and test each use case:

```bash
# Start the server
cargo run --release --example iam_permissions -- /tmp/iam_db 20

# In another terminal, test each violation:

# Cross-region (should find 8+ paths)
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"use_case":"cross_region","implementation":"reference"}' \
  http://localhost:8081/api/run
sleep 2
curl -s http://localhost:8081/api/result | jq '.result.summary'

# Minimal privilege (should find 2 non-minimal paths)
curl -s -X POST http://localhost:8081/api/clear
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"use_case":"minimal_privilege","implementation":"reference"}' \
  http://localhost:8081/api/run
sleep 2
curl -s http://localhost:8081/api/result | jq '.result.summary'

# Unused roles (should find 3 unused roles)
curl -s -X POST http://localhost:8081/api/clear
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"use_case":"unused_roles","implementation":"reference"}' \
  http://localhost:8081/api/run
sleep 2
curl -s http://localhost:8081/api/result | jq '.result.summary'
```

---

## Use Cases

### 1. Reachability Analysis

**Command**: `reachability`
**Algorithm**: Breadth-First Search (BFS)
**Question**: Can user X access resource Y?

```bash
cargo run --release --example iam_permissions -- reachability reference /tmp/iam_db 50
```

**Business Problem**: Determine if a specific user can access a sensitive resource. This is critical for access auditing, permission troubleshooting, and verifying that security policies are correctly configured.

**How it works**:
1. Start BFS from the source user
2. Explore all outgoing edges (group memberships, policies, etc.)
3. Stop when target resource is found or all paths exhausted
4. Return path if reachable, or "not reachable"

**Reading the Visualization**:
- Red highlighted nodes show the permission path from source to target
- The path traverses: User → Group → Policy → Resource
- If a path is found, the visualization shows the specific groups and policies involved
- If no path exists, no nodes are highlighted (the user cannot reach the resource)
- Click any node to see its connections and understand how access flows

**Use case**: Access verification, permission auditing

---

### 2. Blast Radius Analysis

**Command**: `blast_radius`
**Algorithm**: BFS with depth tracking
**Question**: If user X is compromised, what resources are at risk?

```bash
cargo run --release --example iam_permissions -- blast_radius reference /tmp/iam_db 50
```

**Business Problem**: If a user's credentials are compromised, what resources are at risk? Understanding blast radius is essential for incident response planning and risk assessment. It helps security teams prioritize which credential compromises are most severe.

**How it works**:
1. Start BFS from compromised user
2. Track depth (hops) to each reachable node
3. Collect all reachable resources, grouped by depth
4. Report total resources at risk and depth distribution

**Sample output**:
```
Summary: Total resources at risk: 302, Max depth: 5
  Depth 2: 3 resources
  Depth 3: 100 resources
  Depth 4: 100 resources
  Depth 5: 99 resources
```

**Reading the Visualization**:
- Red nodes show the "blast radius" - all resources reachable from the compromised user
- Click "Show Overlay" to visualize the full impact zone
- Nodes closer to the compromised user (lower depth) are more immediately accessible to an attacker
- The sidebar shows resources grouped by depth level
- Access flows through highlighted policies and groups
- Larger highlighted nodes typically have more connections

**Use case**: Incident response, risk assessment

---

### 3. Least Resistance Path

**Command**: `least_resistance`
**Algorithm**: Dijkstra's Algorithm
**Question**: What's the easiest path from user X to sensitive resource Y?

```bash
cargo run --release --example iam_permissions -- least_resistance reference /tmp/iam_db 50
```

**Business Problem**: Find the easiest attack path from a user to sensitive resources. Attackers follow the path of least resistance. Identifying these paths helps security teams harden the most vulnerable routes first.

**How it works**:
1. Run Dijkstra from source user with edge weights as costs
2. Find minimum-cost path to each sensitive resource
3. Return the path with lowest total weight

**Algorithm details** (Dijkstra):
```
1. Initialize: distance[start] = 0, distance[others] = ∞
2. Add start to priority queue
3. While queue not empty:
   a. Pop node with minimum distance
   b. For each neighbor:
      - new_dist = distance[current] + edge_weight
      - If new_dist < distance[neighbor]:
        - Update distance[neighbor]
        - Add neighbor to queue
4. Return distances and paths
```

**Reading the Visualization**:
- The highlighted path shows the lowest-cost route from user to resource
- Edge weights indicate difficulty: `can_access` (1.0) is easier than `depends_on` (3.5)
- Yellow highlighted edges show the optimal attack path
- The total path cost is shown in the summary
- Click edges to see their individual weights
- Groups and policies along the path represent the attack chain

**Use case**: Attack path analysis, security hardening prioritization

---

### 4. Privilege Clustering

**Command**: `privilege_clustering`
**Algorithm**: Jaccard Similarity
**Question**: Which users have similar access patterns?

```bash
cargo run --release --example iam_permissions -- privilege_clustering reference /tmp/iam_db 50
```

**Business Problem**: Which users have similar access patterns? Identifying clusters helps with role mining—creating RBAC roles that match actual usage patterns—and detecting anomalous users who don't fit any cluster.

**How it works**:
1. For each user, compute set of accessible resources (via BFS)
2. Calculate Jaccard similarity between all user pairs:
   ```
   Jaccard(A, B) = |A ∩ B| / |A ∪ B|
   ```
3. Cluster users with similarity above threshold (default: 0.5)
4. Report clusters for potential role consolidation

**Reading the Visualization**:
- Users are grouped into clusters based on similar access patterns
- Highlighted users share similar permissions with others in their cluster
- Click a user to see their cluster assignment in the sidebar annotation
- Large clusters suggest opportunities for role consolidation
- Isolated users (not in any cluster) may need access review
- Users in the same cluster could potentially share a common role

**Use case**: Role mining, permission consolidation, RBAC design

---

### 5. Over-Privileged Detection

**Command**: `over_privileged`
**Algorithm**: BFS + counting
**Question**: Which users can access too many sensitive resources?

```bash
cargo run --release --example iam_permissions -- over_privileged reference /tmp/iam_db 50
```

**Business Problem**: Which users can access more sensitive resources than they should? Over-privileged accounts violate the principle of least privilege and increase breach impact. Users with access to multiple databases or critical resources represent higher risk.

**How it works**:
1. For each user, BFS to find all reachable resources
2. Count how many are marked as "sensitive" (databases, certain instances)
3. Flag users exceeding threshold (default: 3+ sensitive resources)
4. Rank by number of sensitive resources accessible

**Reading the Visualization**:
- Over-privileged users are highlighted in red
- The sidebar shows each user's sensitive resource count
- Click a user to see their specific access count in the annotation
- Users are ranked by number of sensitive resources accessible
- Prioritize remediation starting with users having the highest access counts
- Larger nodes typically indicate more overall connections

**Use case**: Privilege auditing, least-privilege enforcement

---

### 6. Cross-Region Access

**Command**: `cross_region`
**Algorithm**: Filtered BFS with region tracking
**Question**: Are there permission paths that cross region boundaries?

```bash
cargo run --release --example iam_permissions -- cross_region reference /tmp/iam_db 50
```

**Business Problem**: Are there permission paths that cross region boundaries? For compliance with GDPR, data sovereignty, or internal policies, access to resources in one region should not come from users primarily in another region.

**How it works**:
1. BFS from each user, tracking current region at each step
2. Detect when a path crosses from one region to another
3. Record cross-region access paths
4. Report for compliance review

**Reading the Visualization**:
- Users with cross-region access are highlighted in red
- Region nodes are shown at graph edges (gray color)
- Look for paths that connect resources across different Region nodes
- The sidebar shows which regions each flagged user can span
- Each cross-region path may indicate a compliance violation requiring review
- Click highlighted users to see their specific region access patterns

**Use case**: Data residency compliance (GDPR, sovereignty requirements)

---

### 7. Unused Roles Detection

**Command**: `unused_roles`
**Algorithm**: Kosaraju's Strongly Connected Components (SCC)
**Question**: Which roles are isolated or never assumed?

```bash
cargo run --release --example iam_permissions -- unused_roles reference /tmp/iam_db 50
```

**Business Problem**: Which roles are never assumed by any workload? Unused roles increase attack surface without providing value. They should be cleaned up to reduce the number of potential privilege escalation paths.

**How it works**:
1. Run Kosaraju's SCC algorithm to find connected components
2. Identify roles in small/isolated SCCs (not in main permission graph)
3. Check which roles have no incoming "Assumes" edges
4. Report unused roles for cleanup

**Algorithm details** (Kosaraju's SCC):
```
Phase 1: DFS on original graph, record finish order
Phase 2: DFS on reversed graph in reverse finish order
         Each DFS tree = one SCC
```

**Reading the Visualization**:
- Unused/isolated roles are highlighted in red
- These roles appear disconnected from workloads or in small isolated clusters
- The highlighted roles are separate from the main permission graph
- Click each role to verify no workloads assume it
- Roles confirmed as unused can be safely deleted to reduce attack surface
- Look for roles with no incoming "assumes" edges

**Use case**: Role hygiene, attack surface reduction

---

### 8. Privilege Hubs Detection

**Command**: `privilege_hubs`
**Algorithm**: Manual degree calculation
**Question**: Which entities have unusually high connectivity?

```bash
cargo run --release --example iam_permissions -- privilege_hubs reference /tmp/iam_db 50
```

**Business Problem**: Which entities have unusually high connectivity? Hubs with many connections are single points of failure. Compromising a hub grants access to many resources; losing a hub disrupts many users.

**How it works**:
1. Calculate in-degree and out-degree for every node
2. Compute total degree (in + out)
3. Identify nodes in top percentile (default: top 10%)
4. Classify hubs by type (policy hubs, group hubs, etc.)

**Sample output**:
```
Summary: Found 112 privilege hubs (top 10%, degree >= 6). Avg degree: 3.6, Max: 29
  Policies: 47 hubs
    policy-0000 (in:7, out:1)
  Groups: 26 hubs
    group-0002 (in:26, out:3)
```

**Reading the Visualization**:
- Privilege hubs are highlighted with larger red nodes (higher connectivity = larger)
- Policy hubs with high out-degree grant access to many resources
- Group hubs with high in-degree have many members
- Click each hub to see its in/out degree in the annotation
- High-connectivity nodes are critical points requiring extra protection
- Hub compromise could grant attackers broad access

**Use case**: Identifying single points of failure, concentrated risk

---

### 9. Minimal Privilege Verification

**Command**: `minimal_privilege`
**Algorithm**: Dijkstra path verification
**Question**: Are existing permission paths truly minimal?

```bash
cargo run --release --example iam_permissions -- minimal_privilege reference /tmp/iam_db 50
```

**Business Problem**: Are existing permission paths truly minimal, or are there unnecessary intermediaries? Non-minimal paths indicate over-complicated permission structures that should be simplified.

**How it works**:
1. For sample of users and sensitive resources:
2. Compute optimal path cost using Dijkstra
3. Trace actual path via BFS
4. Compare: if actual cost > optimal cost, path is non-minimal
5. Report non-minimal paths for remediation

**Reading the Visualization**:
- Nodes involved in non-minimal paths are highlighted
- These paths contain unnecessary intermediaries (extra hops)
- Groups and policies in the path may be redundant
- Compare the actual path (shown) to the optimal path cost
- Look for cases where direct policy grants could replace indirect chains
- Non-minimal: User → Group → Policy → Resource (when User → Policy → Resource suffices)

**Use case**: Least privilege verification, permission optimization

---

### 10. Accessible Resources Listing

**Command**: `accessible_resources`
**Algorithm**: DFS traversal
**Question**: What resources can each user access?

```bash
cargo run --release --example iam_permissions -- accessible_resources reference /tmp/iam_db 50
```

**Business Problem**: What resources can users access? This provides a complete inventory of effective permissions—essential for access certification, audit preparation, and understanding actual vs. intended access.

**How it works**:
1. For each user, perform DFS traversal
2. Collect all resource nodes reached
3. Compute statistics: resources per user, most accessed resources
4. Identify users with broadest access

**Sample output**:
```
Summary: Analyzed 50 users: avg 24.9 resources/user, 190 unique resources accessed
  Most accessed resources:
    vpc-0024: 25 users
    vpc-0011: 25 users
  Users with most access:
    user-0020: 54 resources
    user-0035: 54 resources
```

**Reading the Visualization**:
- Users are highlighted with their access counts displayed in annotations
- Click any user to see how many resources they can reach
- The sidebar shows "most accessed" resources (reachable by many users)
- The sidebar also shows "broadest access" users (can reach most resources)
- Most accessed resources may be shared infrastructure (VPCs, common policies)
- Users with broadest access should be reviewed for least privilege

**Use case**: Access auditing, permission inventory

---

### 11. High Value Targets Detection

**Command**: `high_value_targets`
**Algorithm**: PageRank
**Question**: Which resources are most "important" based on permission flow?

```bash
cargo run --release --example iam_permissions -- high_value_targets reference /tmp/iam_db 50
```

**Business Problem**: Which resources are most "important" based on permission flow? High-value targets are reachable via many paths and should receive the most security investment—monitoring, access controls, encryption.

**How it works**:
1. Run PageRank algorithm on the permission graph
2. Nodes with high PageRank have many incoming permission paths
3. Rank resources, roles, and policies by PageRank score
4. Report high-value targets for defense prioritization

**Algorithm details** (PageRank):
```
1. Initialize: score[all] = 1/N
2. For each iteration:
   For each node v:
     score[v] = (1-d)/N + d × Σ(score[u]/outdegree[u])
                          for all u pointing to v
   where d = damping factor (0.85)
3. Repeat until convergence (20 iterations)
```

**Interpretation**:
- High PageRank resources are reachable via many permission paths
- These are high-value targets for attackers
- Prioritize protection of high-PageRank resources

**Sample output**:
```
Summary: PageRank analysis (d=0.85, 20 iters): 302 resources ranked
  High-value resources (most reachable):
    vpc-0000: 0.006188
    vpc-0014: 0.004965
  High-value roles (permission bottlenecks):
    role-0025: 0.000333
  High-value policies (widely granted):
    policy-0000: 0.002774
```

**Reading the Visualization**:
- High-value targets are highlighted with larger red nodes (higher PageRank = larger)
- The PageRank score represents "importance" based on incoming permission paths
- Resources that many users can reach have higher PageRank
- Prioritize protecting high-PageRank databases and VPCs
- Click highlighted nodes to see their PageRank score in the annotation
- VPCs and shared resources often have highest PageRank due to broad access

**Use case**: Defense prioritization, security investment decisions

---

### 12. Minimum Spanning Tree Analysis

**Command**: `mst`
**Algorithm**: Kruskal's Algorithm (`petgraph::algo::min_spanning_tree`)
**Question**: What is the minimal permission infrastructure needed?

```bash
cargo run --release --example iam_permissions -- mst reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- mst motlie_db /tmp/iam_db 50
```

**Business Problem**: What is the minimal set of permission edges that maintains full connectivity between all entities? The Minimum Spanning Tree identifies essential permission relationships vs. redundant ones. Edges NOT in the MST are candidates for removal during permission cleanup.

**How it works**:
1. Collect all edges with their weights (permission costs)
2. Sort edges by weight (lowest first)
3. Use Union-Find data structure for cycle detection
4. For each edge, add to MST if it doesn't create a cycle
5. Stop when all nodes are connected (N-1 edges for N nodes)

**Algorithm details** (Kruskal's):
```
1. Sort all edges by weight ascending
2. Initialize: each node is its own set (Union-Find)
3. For each edge (u, v, weight) in sorted order:
   a. If find(u) ≠ find(v):  // Different components
      - Add edge to MST
      - Union(u, v)
   b. Stop when MST has N-1 edges
4. Return MST edges and identify redundant edges
```

**Interpretation**:
- **MST edges** = essential permission infrastructure (minimum needed for full connectivity)
- **Redundant edges** = excess attack surface that could potentially be removed
- Lower MST weight = more efficient permission structure

**Sample output**:
```
Summary: MST: 832 edges (total weight: 1247.50), 681 redundant edges could be removed
Details:
  MST edges (sorted by weight):
    region-us-east (region) -> vpc-0000 (vpc): 0.50
    region-us-west (region) -> vpc-0001 (vpc): 0.50
    policy-0000 (policy) -> instance-0000 (instance): 1.00
    ...
```

**Reading the Visualization**:
- MST edges are highlighted (essential permission paths)
- Non-highlighted edges are redundant (could be removed without breaking connectivity)
- Lower-weight edges represent more efficient permission paths
- Click edges to see their weight and understand the permission type

**Security Applications**:
- **Attack Surface Reduction**: Identify redundant permission edges for cleanup
- **Permission Optimization**: Find the most efficient permission structure
- **Compliance**: Demonstrate minimal necessary access

**Use case**: Permission optimization, attack surface reduction, least privilege enforcement

---

## Algorithm Deep Dives

### BFS vs DFS for Graph Traversal

| Aspect | BFS | DFS |
|--------|-----|-----|
| **Data structure** | Queue (FIFO) | Stack (LIFO) |
| **Memory** | O(branching factor^depth) | O(max depth) |
| **Finds shortest path** | Yes (unweighted) | No |
| **Use in IAM** | Blast radius, reachability | Accessible resources |
| **Complexity** | O(V + E) | O(V + E) |

### Dijkstra's Algorithm

**When to use**: Finding shortest/lowest-cost paths in weighted graphs

**Complexity**: O((V + E) log V) with binary heap

**Key insight for IAM**: Edge weights as "resistance" enable finding the path of least resistance - the easiest attack path.

```rust
// Pseudocode
fn dijkstra(graph, source) {
    dist[source] = 0
    heap.push((0, source))

    while let Some((cost, node)) = heap.pop() {
        if cost > dist[node] { continue }

        for (neighbor, weight) in graph.neighbors(node) {
            let new_cost = cost + weight
            if new_cost < dist[neighbor] {
                dist[neighbor] = new_cost
                heap.push((new_cost, neighbor))
            }
        }
    }
}
```

### PageRank for Security

PageRank was designed for web page ranking but applies naturally to permission graphs:

| Web Context | IAM Context |
|-------------|-------------|
| Web pages | Resources, roles, policies |
| Hyperlinks | Permission relationships |
| "Important" pages | High-value targets |
| Many incoming links | Many permission paths |

**Formula**:
```
PR(v) = (1-d)/N + d × Σ PR(u)/L(u)
```
Where:
- `d` = damping factor (0.85)
- `N` = total nodes
- `L(u)` = out-degree of node u

### Kosaraju's SCC Algorithm

**Purpose**: Find strongly connected components (maximal sets of mutually reachable nodes)

**Why it matters for IAM**: Isolated SCCs indicate disconnected permission islands - potentially unused roles or orphaned policies.

**Two-phase algorithm**:
1. **Phase 1**: DFS on original graph, record finish times
2. **Phase 2**: DFS on reversed graph in decreasing finish time order

Each DFS tree in Phase 2 is a strongly connected component.

### Reference Implementation Libraries

The reference implementations use established Rust graph libraries where available. Some algorithms require custom implementations due to specific output requirements (path tracking, depth tracking) or unavailability in standard libraries.

| Use Case | Algorithm | Reference Implementation | Library/Source |
|----------|-----------|-------------------------|----------------|
| **Reachability** | BFS | Custom BFS | Hand-coded (needs path tracking) |
| **Blast Radius** | BFS + depth | Custom BFS | Hand-coded (needs depth tracking) |
| **Least Resistance** | Dijkstra | Custom Dijkstra | Hand-coded (needs path reconstruction) |
| **Privilege Clustering** | Jaccard + BFS | `petgraph::visit::Bfs` | [petgraph](https://crates.io/crates/petgraph) |
| **Over-Privileged** | BFS + counting | `petgraph::visit::Bfs` | [petgraph](https://crates.io/crates/petgraph) |
| **Cross-Region Access** | Filtered BFS | Custom BFS | Hand-coded (complex path tracking) |
| **Unused Roles** | Kosaraju SCC | `petgraph::algo::kosaraju_scc` | [petgraph](https://crates.io/crates/petgraph) |
| **Privilege Hubs** | Degree analysis | Custom counting | Hand-coded (simple, no library needed) |
| **Minimal Privilege** | Dijkstra verification | `petgraph::algo::dijkstra` | [petgraph](https://crates.io/crates/petgraph) |
| **Accessible Resources** | DFS traversal | Custom DFS | Hand-coded (needs result collection) |
| **High Value Targets** | PageRank | Custom PageRank | Hand-coded (not in petgraph) |
| **Minimum Spanning Tree** | Kruskal's MST | `petgraph::algo::min_spanning_tree` | [petgraph](https://crates.io/crates/petgraph) |

**Notes:**
- **petgraph**: Industry-standard Rust graph library with optimized implementations
- **Hand-coded algorithms** exist where library functions don't provide required output (e.g., path tracking, depth levels)
- **PageRank**: Not available in petgraph; uses standard iterative implementation with damping factor 0.85
- All implementations verified for correctness via result comparison between reference and motlie_db versions

---

## Running the Examples

### All Use Cases

```bash
# List available use cases
cargo run --release --example iam_permissions -- list

# Reachability
cargo run --release --example iam_permissions -- reachability reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- reachability motlie_db /tmp/iam_db 50

# Blast Radius
cargo run --release --example iam_permissions -- blast_radius reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- blast_radius motlie_db /tmp/iam_db 50

# Least Resistance Path
cargo run --release --example iam_permissions -- least_resistance reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- least_resistance motlie_db /tmp/iam_db 50

# Privilege Clustering
cargo run --release --example iam_permissions -- privilege_clustering reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- privilege_clustering motlie_db /tmp/iam_db 50

# Over-Privileged Detection
cargo run --release --example iam_permissions -- over_privileged reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- over_privileged motlie_db /tmp/iam_db 50

# Cross-Region Access
cargo run --release --example iam_permissions -- cross_region reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- cross_region motlie_db /tmp/iam_db 50

# Unused Roles Detection
cargo run --release --example iam_permissions -- unused_roles reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- unused_roles motlie_db /tmp/iam_db 50

# Privilege Hubs Detection
cargo run --release --example iam_permissions -- privilege_hubs reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- privilege_hubs motlie_db /tmp/iam_db 50

# Minimal Privilege Verification
cargo run --release --example iam_permissions -- minimal_privilege reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- minimal_privilege motlie_db /tmp/iam_db 50

# Accessible Resources Listing
cargo run --release --example iam_permissions -- accessible_resources reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- accessible_resources motlie_db /tmp/iam_db 50

# High Value Targets (PageRank)
cargo run --release --example iam_permissions -- high_value_targets reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- high_value_targets motlie_db /tmp/iam_db 50

# Minimum Spanning Tree
cargo run --release --example iam_permissions -- mst reference /tmp/iam_db 50
cargo run --release --example iam_permissions -- mst motlie_db /tmp/iam_db 50
```

### Comparing Implementations

Run both implementations on the same data to verify correctness:

```bash
# Reference implementation
cargo run --release --example iam_permissions -- blast_radius reference /tmp/iam_db 50

# motlie_db implementation (same graph, persisted)
cargo run --release --example iam_permissions -- blast_radius motlie_db /tmp/iam_db 50
```

Both should produce identical `result_hash` values, confirming correctness.

---

## Web Visualization

The IAM permissions example includes an interactive web-based graph visualization that lets you explore the permission graph and analysis results visually.

### Starting the Visualization Server

Add `--visualize` to any analysis command to start the web server:

```bash
# Run blast radius analysis with visualization
cargo run --release --example iam_permissions -- blast_radius reference /tmp/iam_db 50 --visualize

# Use a custom port
cargo run --release --example iam_permissions -- high_value_targets motlie_db /tmp/iam_db 100 --visualize --port 9000
```

Then open your browser to `http://localhost:8081/viz` (or your custom port).

### Visualization Features

| Feature | Description |
|---------|-------------|
| **Interactive Graph** | Pan, zoom, and drag nodes to explore the IAM permission structure |
| **Node Shapes** | Different shapes distinguish entity categories: triangles (Users), diamonds (Workloads), hexagons (VPCs), squares (Resources), circles (Groups/Policies/Roles) |
| **Node Colors** | Each IAM entity type has a distinct color (Users=blue, Groups=green, Policies=purple, etc.) |
| **Edge Types** | Colored edges show relationship types (member_of, has_policy, can_access, etc.) |
| **Click Details** | Click any node or edge to see detailed information in the sidebar |
| **Search** | Search for nodes by name or type |
| **Analysis Dropdown** | Alphabetically sorted dropdown to select from 11 analysis use cases |
| **Implementation Toggle** | Choose between Reference (petgraph/in-memory) or Disk-based (motlie_db/RocksDB) implementation |
| **Analysis Overlay** | Toggle overlay to highlight nodes affected by the analysis |
| **Real-time Status** | Shows analysis progress and displays "Complete: [Analysis Name]" when finished |

### Node Type Colors and Shapes

| Node Type | Color | Shape | Description |
|-----------|-------|-------|-------------|
| User | Blue (#3498db) | Triangle | Human identities |
| Group | Green (#27ae60) | Circle | User collections |
| Policy | Purple (#9b59b6) | Circle | Permission definitions |
| Role | Orange (#e67e22) | Circle | Assumable identities |
| Workload | Teal (#1abc9c) | Diamond | Running services |
| VPC | Yellow (#f39c12) | Hexagon | Virtual networks |
| Instance | Red (#e74c3c) | Square | Compute resources |
| Disk | Dark Red (#c0392b) | Square | Storage resources |
| Database | Dark Purple (#8e44ad) | Square | Database resources |
| Region | Gray (#34495e) | Circle | Geographic regions |

**Shape Legend:**
- **Triangles**: Identity nodes (Users)
- **Diamonds**: Workload nodes (running services that assume roles)
- **Hexagons**: Network containers (VPCs)
- **Squares**: Resource nodes (Instances, Disks, Databases)
- **Circles**: Permission structure nodes (Groups, Policies, Roles, Regions)

### Understanding This Analysis Panel

When an analysis completes, the sidebar shows an expandable "Understanding This Analysis" section with three parts:

| Section | Description |
|---------|-------------|
| **Business Problem** | Explains the real-world security question being answered and why it matters |
| **Algorithm** | Describes how the graph algorithm works to answer the question |
| **Reading the Visualization** | Dynamic guide that includes specific entity names from the current analysis results |

The visualization guide is context-aware—it shows actual user names, policy names, and counts from the current analysis, not generic placeholders. For example, it might say "Top over-privileged users: user-0026, user-0059, user-0092" rather than just "highlighted users."

### Analysis Overlays

When an analysis completes, click **"Show Overlay"** to highlight relevant nodes:

| Use Case | Overlay Behavior |
|----------|------------------|
| **Reachability** | Highlights path nodes from source to target |
| **Blast Radius** | Highlights all affected resources with depth annotations |
| **Least Resistance** | Highlights the optimal attack path |
| **Privilege Clustering** | Groups users by cluster with annotations |
| **Over-Privileged** | Highlights users with excessive access |
| **Cross-Region** | Highlights users with cross-region access |
| **Unused Roles** | Highlights isolated/unused roles |
| **Privilege Hubs** | Highlights high-connectivity entities |
| **Minimal Privilege** | Highlights non-minimal paths |
| **Accessible Resources** | Highlights users and their resource counts |
| **High Value Targets** | Highlights high PageRank nodes |
| **Minimum Spanning Tree** | Highlights MST edges; non-MST edges grayed out |

### API Endpoints

The visualization server exposes REST endpoints for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/viz` | GET | Interactive HTML viewer |
| `/api/graph` | GET | Graph data (nodes and edges) as JSON |
| `/api/result` | GET | Analysis result with overlay data |
| `/api/status` | GET | Analysis completion status |

**Example: Fetch graph data**
```bash
curl http://localhost:8081/api/graph | jq '.nodes | length'
# Output: 833

curl http://localhost:8081/api/result | jq '.result.summary'
# Output: "Total resources at risk: 302, Max depth: 5"
```

### Workflow

1. **Start with visualization**: Run analysis with `--visualize`
2. **Open browser**: Navigate to `http://localhost:8081/viz`
3. **Explore graph**: Pan, zoom, click nodes to understand structure
4. **Wait for analysis**: Status badge shows "Running..." then "Complete"
5. **Toggle overlay**: Click "Show Overlay" to see analysis results highlighted
6. **Investigate**: Click highlighted nodes to see analysis annotations
7. **Exit**: Press Ctrl+C in terminal to stop the server

---

## Performance Benchmarks

### Use Case Performance (Scale 50, 833 nodes, ~1,500 edges)

| Use Case | Algorithm | Reference (ms) | motlie_db (ms) | Ratio | Correctness |
|----------|-----------|---------------:|---------------:|------:|:-----------:|
| **Reachability** | BFS | 0.01 | 0.08 | 8.0x | ✓ |
| **Blast Radius** | BFS + depth | 0.03 | 0.21 | 7.0x | ✓ |
| **Least Resistance** | Dijkstra | 0.03 | 0.19 | 6.3x | ✓ |
| **Over-Privileged** | BFS + counting | 0.45 | 2.89 | 6.4x | ✓ |
| **Cross-Region Access** | Filtered BFS | 0.62 | 18.35 | 29.6x | ✓ |
| **Privilege Clustering** | Jaccard | 1.02 | 34.89 | 34.2x | * |
| **Unused Roles** | Kosaraju SCC | 0.13 | 11.26 | 86.6x | ✓ |
| **Privilege Hubs** | Manual Degree | 0.20 | 9.29 | 46.5x | * |
| **Minimal Privilege** | Dijkstra Verify | 0.15 | 9.59 | 63.9x | ✓ |
| **Accessible Resources** | DFS Traversal | 0.43 | 4.85 | 11.3x | ✓ |
| **High Value Targets** | PageRank | 1.92 | 21.50 | 11.2x | ✓ |
| **Minimum Spanning Tree** | Kruskal's | 0.35 | 2.15 | 6.1x | ✓ |

**Legend**:
- ✓ = Identical hash (exact match)
- \* = Statistics match (ordering may differ due to hash maps)

### Scaling Behavior

| Scale | Nodes | Edges | Blast Radius (ref) | Blast Radius (db) | Ratio |
|------:|------:|------:|-------------------:|------------------:|------:|
| 20 | 333 | 605 | 0.01 ms | 0.08 ms | 8.0x |
| 50 | 833 | 1,509 | 0.03 ms | 0.21 ms | 7.0x |
| 100 | 1,666 | 3,018 | 0.05 ms | 0.38 ms | 7.6x |

**Key insight**: Overhead ratio remains consistent across scales, making motlie_db suitable for large IAM graphs.

### When to Use Each Implementation

| Use Case | Recommendation |
|----------|----------------|
| **One-time analysis** | Reference (faster) |
| **Repeated queries** | motlie_db (persistent) |
| **Large graphs (>100K nodes)** | motlie_db (bounded memory) |
| **Interactive exploration** | motlie_db (instant reload) |
| **CI/CD integration** | Reference (no disk I/O) |

---

## Security Applications

### Incident Response

1. **Credential Compromise**: Run `blast_radius` to immediately understand exposure
2. **Lateral Movement Analysis**: Use `least_resistance` to find likely attack paths
3. **Scope Assessment**: Use `accessible_resources` to inventory compromised access

### Compliance

1. **Data Residency**: Run `cross_region` to detect sovereignty violations
2. **Least Privilege Audit**: Use `minimal_privilege` to verify paths are optimal
3. **Access Review**: Use `accessible_resources` for periodic access certification

### Security Hardening

1. **Attack Surface Reduction**: Use `unused_roles` to identify cleanup candidates
2. **Defense Prioritization**: Use `high_value_targets` to focus security investments
3. **Hub Protection**: Use `privilege_hubs` to identify single points of failure

### Permission Management

1. **Role Mining**: Use `privilege_clustering` to design RBAC roles
2. **Over-Privilege Detection**: Run `over_privileged` regularly to enforce least privilege
3. **Access Path Analysis**: Use `reachability` for permission troubleshooting

---

## Extending the Example

### Adding a New Use Case

1. **Add enum variant** in `UseCase`:
```rust
enum UseCase {
    // ... existing cases
    MyNewCase,
}
```

2. **Implement from_str, name, algorithm, description** methods

3. **Add to all()** function

4. **Implement reference version** in `mod reference`:
```rust
pub fn my_new_case(graph: &DiGraph<String, f64>, ...) -> AnalysisResult {
    // Implementation
}
```

5. **Implement motlie_db version** in `mod motlie_impl`:
```rust
pub async fn my_new_case(..., reader: &Reader, timeout: Duration) -> Result<AnalysisResult> {
    // Implementation using OutgoingEdges queries
}
```

6. **Add match arms** in main() for both implementations

### Adding New Node/Edge Types

1. **Update NodeType enum** and edge generation in `IamGraph::generate()`
2. **Update statistics tracking** in `GraphStats`
3. **Ensure new types are handled** in relevant use cases

### Customizing Edge Weights

Edge weights in `EdgeType::weight()` can be adjusted to model different security scenarios:
- Lower weights = easier access paths
- Higher weights = more restricted paths

---

## References

### Graph Algorithms

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) - BFS, DFS, Dijkstra, SCC
- [The PageRank Citation Ranking](http://ilpubs.stanford.edu:8090/422/) - Original PageRank paper
- [Community Detection in Networks](https://arxiv.org/abs/0906.0612) - Louvain algorithm

### IAM Security

- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GCP IAM Overview](https://cloud.google.com/iam/docs/overview)
- [Azure RBAC Documentation](https://docs.microsoft.com/en-us/azure/role-based-access-control/)

### Graph Databases

- [RocksDB Documentation](https://rocksdb.org/docs/)
- [petgraph Documentation](https://docs.rs/petgraph/)
