#![allow(dead_code, unused_imports, unused_variables)]
/// IAM Permissions Graph Analysis - Interactive Web UI
///
/// This example creates a large-scale simulated cloud IAM graph and provides
/// an interactive web UI for running security analyses comparing petgraph
/// (in-memory) vs motlie_db (persistent).
///
/// # Unified API Usage
///
/// This example uses the **unified motlie_db API** (porcelain layer):
/// - Storage: `motlie_db::{Storage, StorageConfig, ReadOnlyHandles, ReadWriteHandles}`
/// - Queries: `motlie_db::query::{AllNodes, AllEdges, OutgoingEdges, Runnable}`
/// - Mutations: `motlie_db::mutation::{AddNode, AddEdge, NodeSummary, EdgeSummary, Runnable}`
/// - Reader: `motlie_db::reader::Reader`
///
/// The example demonstrates:
/// - Paginated queries with `AllNodes`/`AllEdges` using cursor-based pagination
/// - Entity tagging via node/edge summaries (e.g., `type:user region:us-east sensitive:true`)
/// - Type-safe storage handles for both read-only and read-write access
///
/// ## Graph Structure
///
/// - Users: Human identities with group memberships
/// - Groups: Collections of users with attached policies
/// - Policies: Permission sets that can be attached to users, groups, or roles
/// - Roles: Assumable identities for workloads
/// - Resources: Cloud resources (VPCs, Disks, Instances, Databases)
/// - Workloads: Jobs/services that assume roles
/// - Regions: Geographic locations containing resources
///
/// ## Edge Types
///
/// - MEMBER_OF: User -> Group
/// - HAS_POLICY: User/Group/Role -> Policy
/// - ASSUMES: Workload -> Role
/// - CAN_ACCESS: Policy -> Resource (with permission level)
/// - DEPENDS_ON: Resource -> Resource
/// - LOCATED_IN: Resource -> Region
/// - RUNS_IN: Workload -> Resource (e.g., runs on instance)
///
/// ## Use Cases (select from UI dropdown)
///
/// 1. Reachability Analysis - Can user X access resource Y? (BFS/DFS)
/// 2. Blast Radius - What resources are affected if credential X is compromised? (BFS)
/// 3. Least Resistance Path - Easiest path from user to sensitive resource (Dijkstra/A*)
/// 4. Privilege Clustering - Group users by similar access patterns (Louvain)
/// 5. Over-Privileged Detection - Find users with excessive permissions (Graph metrics)
/// 6. Cross-Region Access - Find access paths that cross region boundaries (Filtered BFS)
/// 7. Unused Roles - Identify isolated/unused roles using SCCs for cleanup (Kosaraju)
/// 8. Privilege Hubs - Spot over-privileged entities via manual degree calculation
/// 9. Minimal Privilege Paths - Verify permission paths are minimal using Dijkstra
/// 10. Accessible Resources - List all resources a user can access (DFS/BFS traversal)
/// 11. High Value Targets - Identify high-value targets using PageRank
///
/// ## Usage
///
///   iam_permissions <db_path> <scale>                 # Start HTTP server
///   iam_permissions --generate <db_path> <scale>      # Generate graph data only
///   iam_permissions list                              # List available use cases
///
/// ## Examples
///
///   iam_permissions /tmp/iam_db 50                    # Start server (default port 8081)
///   iam_permissions /tmp/iam_db 100 --port 9000       # Start server on custom port
///   iam_permissions --generate /tmp/iam_db 100        # Generate graph data only

#[path = "../graph/common.rs"]
mod common;

use anyhow::{Context, Result};
use common::{
    build_graph, compute_hash, get_disk_metrics, measure_time_and_memory,
    measure_time_and_memory_async, GraphEdge, GraphMetrics, GraphNode, Implementation,
};
use motlie_db::query::{AllEdges, AllNodes, OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::{Id, ReadOnlyHandles, Storage, StorageConfig};
use petgraph::algo::{dijkstra, kosaraju_scc, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::{Bfs, EdgeRef};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::path::Path;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;
use warp::Filter;

// ============================================================================
// Graph Schema Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum NodeType {
    User,
    Group,
    Policy,
    Role,
    Workload,
    Vpc,
    Instance,
    Disk,
    Database,
    Region,
}

impl NodeType {
    fn as_str(&self) -> &'static str {
        match self {
            NodeType::User => "user",
            NodeType::Group => "group",
            NodeType::Policy => "policy",
            NodeType::Role => "role",
            NodeType::Workload => "workload",
            NodeType::Vpc => "vpc",
            NodeType::Instance => "instance",
            NodeType::Disk => "disk",
            NodeType::Database => "database",
            NodeType::Region => "region",
        }
    }

    fn is_resource(&self) -> bool {
        matches!(
            self,
            NodeType::Vpc | NodeType::Instance | NodeType::Disk | NodeType::Database
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum EdgeType {
    MemberOf,   // User -> Group
    HasPolicy,  // User/Group/Role -> Policy
    Assumes,    // Workload -> Role
    CanAccess,  // Policy -> Resource
    DependsOn,  // Resource -> Resource
    LocatedIn,  // Resource -> Region
    RunsIn,     // Workload -> Resource
}

impl EdgeType {
    fn as_str(&self) -> &'static str {
        match self {
            EdgeType::MemberOf => "member_of",
            EdgeType::HasPolicy => "has_policy",
            EdgeType::Assumes => "assumes",
            EdgeType::CanAccess => "can_access",
            EdgeType::DependsOn => "depends_on",
            EdgeType::LocatedIn => "located_in",
            EdgeType::RunsIn => "runs_in",
        }
    }

    /// Weight for path finding (lower = easier/more direct access)
    fn weight(&self) -> f64 {
        match self {
            EdgeType::CanAccess => 1.0,   // Direct permission
            EdgeType::HasPolicy => 1.5,   // Policy attachment
            EdgeType::MemberOf => 2.0,    // Group membership
            EdgeType::Assumes => 2.5,     // Role assumption
            EdgeType::RunsIn => 3.0,      // Workload context
            EdgeType::DependsOn => 3.5,   // Resource dependency
            EdgeType::LocatedIn => 0.5,   // Location (not a permission hop)
        }
    }
}

// ============================================================================
// Graph Statistics
// ============================================================================

#[derive(Debug, Clone, Default)]
struct GraphStats {
    users: usize,
    groups: usize,
    policies: usize,
    roles: usize,
    workloads: usize,
    vpcs: usize,
    instances: usize,
    disks: usize,
    databases: usize,
    regions: usize,
    edges_by_type: HashMap<String, usize>,
}

impl GraphStats {
    fn total_nodes(&self) -> usize {
        self.users
            + self.groups
            + self.policies
            + self.roles
            + self.workloads
            + self.vpcs
            + self.instances
            + self.disks
            + self.databases
            + self.regions
    }

    fn total_edges(&self) -> usize {
        self.edges_by_type.values().sum()
    }

    fn print(&self) {
        println!("\n=== Graph Statistics ===");
        println!("Nodes:");
        println!("  Users:     {:>6}", self.users);
        println!("  Groups:    {:>6}", self.groups);
        println!("  Policies:  {:>6}", self.policies);
        println!("  Roles:     {:>6}", self.roles);
        println!("  Workloads: {:>6}", self.workloads);
        println!("  VPCs:      {:>6}", self.vpcs);
        println!("  Instances: {:>6}", self.instances);
        println!("  Disks:     {:>6}", self.disks);
        println!("  Databases: {:>6}", self.databases);
        println!("  Regions:   {:>6}", self.regions);
        println!("  ─────────────────");
        println!("  Total:     {:>6}", self.total_nodes());
        println!("\nEdges:");
        for (edge_type, count) in &self.edges_by_type {
            println!("  {:12} {:>6}", format!("{}:", edge_type), count);
        }
        println!("  ─────────────────");
        println!("  Total:     {:>6}", self.total_edges());
        println!();
    }
}

// ============================================================================
// IAM Graph Generator
// ============================================================================

#[derive(Clone)]
struct IamNode {
    id: Id,
    name: String,
    node_type: NodeType,
    region: Option<String>, // For resources
}

#[derive(Clone)]
struct IamEdge {
    source: Id,
    target: Id,
    edge_type: EdgeType,
}

#[derive(Clone)]
struct IamGraph {
    nodes: Vec<IamNode>,
    edges: Vec<IamEdge>,
    stats: GraphStats,
    // Indexes for quick lookup
    users: Vec<Id>,
    groups: Vec<Id>,
    policies: Vec<Id>,
    roles: Vec<Id>,
    workloads: Vec<Id>,
    resources: Vec<Id>,
    regions: Vec<Id>,
    sensitive_resources: Vec<Id>, // Databases and some instances marked sensitive
}

impl IamGraph {
    fn generate(scale: usize) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut stats = GraphStats::default();

        // Node ID collectors
        let mut users = Vec::new();
        let mut groups = Vec::new();
        let mut policies = Vec::new();
        let mut roles = Vec::new();
        let mut workloads = Vec::new();
        let mut resources = Vec::new();
        let mut regions = Vec::new();
        let mut sensitive_resources = Vec::new();

        // Scaling factors (realistic ratios)
        let num_regions = (scale / 10).max(2).min(10);
        let num_users = scale * 5;
        let num_groups = scale / 2 + 1;
        let num_policies = scale * 2;
        let num_roles = scale;
        let num_workloads = scale * 2;
        let num_vpcs = scale / 2 + 1;
        let num_instances = scale * 3;
        let num_disks = scale * 2;
        let num_databases = scale / 2 + 1;

        // Create regions
        for i in 0..num_regions {
            let id = Id::new();
            let region_name = format!("region-{}", ["us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-northeast", "sa-east", "af-south", "me-south", "ca-central"][i % 10]);
            nodes.push(IamNode {
                id,
                name: region_name.clone(),
                node_type: NodeType::Region,
                region: None,
            });
            regions.push(id);
            stats.regions += 1;
        }

        // Create VPCs (distributed across regions)
        for i in 0..num_vpcs {
            let id = Id::new();
            let region_idx = i % num_regions;
            let region_name = nodes[region_idx].name.clone();
            nodes.push(IamNode {
                id,
                name: format!("vpc-{:04}", i),
                node_type: NodeType::Vpc,
                region: Some(region_name),
            });
            resources.push(id);
            stats.vpcs += 1;

            // VPC located in region
            edges.push(IamEdge {
                source: id,
                target: regions[region_idx],
                edge_type: EdgeType::LocatedIn,
            });
            *stats.edges_by_type.entry("located_in".to_string()).or_insert(0) += 1;
        }

        let vpc_start_idx = num_regions;

        // Create instances (in VPCs)
        for i in 0..num_instances {
            let id = Id::new();
            let vpc_idx = i % num_vpcs;
            let region = nodes[vpc_start_idx + vpc_idx].region.clone();
            nodes.push(IamNode {
                id,
                name: format!("instance-{:04}", i),
                node_type: NodeType::Instance,
                region: region.clone(),
            });
            resources.push(id);
            stats.instances += 1;

            // Instance depends on VPC
            let vpc_id = nodes[vpc_start_idx + vpc_idx].id;
            edges.push(IamEdge {
                source: id,
                target: vpc_id,
                edge_type: EdgeType::DependsOn,
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            // Some instances are sensitive (production)
            if i % 5 == 0 {
                sensitive_resources.push(id);
            }
        }

        let instance_start_idx = vpc_start_idx + num_vpcs;

        // Create disks (attached to instances)
        for i in 0..num_disks {
            let id = Id::new();
            let instance_idx = i % num_instances;
            let region = nodes[instance_start_idx + instance_idx].region.clone();
            nodes.push(IamNode {
                id,
                name: format!("disk-{:04}", i),
                node_type: NodeType::Disk,
                region,
            });
            resources.push(id);
            stats.disks += 1;

            // Disk depends on instance
            let instance_id = nodes[instance_start_idx + instance_idx].id;
            edges.push(IamEdge {
                source: id,
                target: instance_id,
                edge_type: EdgeType::DependsOn,
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;
        }

        let disk_start_idx = instance_start_idx + num_instances;

        // Create databases (in VPCs, all sensitive)
        for i in 0..num_databases {
            let id = Id::new();
            let vpc_idx = i % num_vpcs;
            let region = nodes[vpc_start_idx + vpc_idx].region.clone();
            nodes.push(IamNode {
                id,
                name: format!("database-{:04}", i),
                node_type: NodeType::Database,
                region,
            });
            resources.push(id);
            sensitive_resources.push(id);
            stats.databases += 1;

            // Database depends on VPC
            let vpc_id = nodes[vpc_start_idx + vpc_idx].id;
            edges.push(IamEdge {
                source: id,
                target: vpc_id,
                edge_type: EdgeType::DependsOn,
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;
        }

        let db_start_idx = disk_start_idx + num_disks;

        // Create policies (grant access to resources)
        let resource_node_start = vpc_start_idx;
        let num_resource_nodes = num_vpcs + num_instances + num_disks + num_databases;

        for i in 0..num_policies {
            let id = Id::new();
            nodes.push(IamNode {
                id,
                name: format!("policy-{:04}", i),
                node_type: NodeType::Policy,
                region: None,
            });
            policies.push(id);
            stats.policies += 1;

            // Each policy grants access to 1-5 resources
            let num_grants = (i % 5) + 1;
            for j in 0..num_grants {
                let resource_idx = (i * 7 + j * 13) % num_resource_nodes;
                let resource_id = nodes[resource_node_start + resource_idx].id;
                edges.push(IamEdge {
                    source: id,
                    target: resource_id,
                    edge_type: EdgeType::CanAccess,
                });
                *stats.edges_by_type.entry("can_access".to_string()).or_insert(0) += 1;
            }
        }

        let policy_start_idx = db_start_idx + num_databases;

        // Create roles
        for i in 0..num_roles {
            let id = Id::new();
            nodes.push(IamNode {
                id,
                name: format!("role-{:04}", i),
                node_type: NodeType::Role,
                region: None,
            });
            roles.push(id);
            stats.roles += 1;

            // Each role has 1-3 policies
            let num_role_policies = (i % 3) + 1;
            for j in 0..num_role_policies {
                let policy_idx = (i * 5 + j * 7) % num_policies;
                let policy_id = nodes[policy_start_idx + policy_idx].id;
                edges.push(IamEdge {
                    source: id,
                    target: policy_id,
                    edge_type: EdgeType::HasPolicy,
                });
                *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
            }
        }

        let role_start_idx = policy_start_idx + num_policies;

        // Create groups
        for i in 0..num_groups {
            let id = Id::new();
            nodes.push(IamNode {
                id,
                name: format!("group-{:04}", i),
                node_type: NodeType::Group,
                region: None,
            });
            groups.push(id);
            stats.groups += 1;

            // Each group has 1-4 policies
            let num_group_policies = (i % 4) + 1;
            for j in 0..num_group_policies {
                let policy_idx = (i * 3 + j * 11) % num_policies;
                let policy_id = nodes[policy_start_idx + policy_idx].id;
                edges.push(IamEdge {
                    source: id,
                    target: policy_id,
                    edge_type: EdgeType::HasPolicy,
                });
                *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
            }
        }

        let group_start_idx = role_start_idx + num_roles;

        // Create users
        for i in 0..num_users {
            let id = Id::new();
            nodes.push(IamNode {
                id,
                name: format!("user-{:04}", i),
                node_type: NodeType::User,
                region: None,
            });
            users.push(id);
            stats.users += 1;

            // Each user belongs to 1-3 groups
            let num_memberships = (i % 3) + 1;
            for j in 0..num_memberships {
                let group_idx = (i * 2 + j * 5) % num_groups;
                let group_id = nodes[group_start_idx + group_idx].id;
                edges.push(IamEdge {
                    source: id,
                    target: group_id,
                    edge_type: EdgeType::MemberOf,
                });
                *stats.edges_by_type.entry("member_of".to_string()).or_insert(0) += 1;
            }

            // Some users have direct policies (20%)
            if i % 5 == 0 {
                let policy_idx = (i * 17) % num_policies;
                let policy_id = nodes[policy_start_idx + policy_idx].id;
                edges.push(IamEdge {
                    source: id,
                    target: policy_id,
                    edge_type: EdgeType::HasPolicy,
                });
                *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
            }
        }

        let user_start_idx = group_start_idx + num_groups;

        // Create workloads
        for i in 0..num_workloads {
            let id = Id::new();
            let instance_idx = i % num_instances;
            let region = nodes[instance_start_idx + instance_idx].region.clone();
            nodes.push(IamNode {
                id,
                name: format!("workload-{:04}", i),
                node_type: NodeType::Workload,
                region,
            });
            workloads.push(id);
            stats.workloads += 1;

            // Workload runs on instance
            let instance_id = nodes[instance_start_idx + instance_idx].id;
            edges.push(IamEdge {
                source: id,
                target: instance_id,
                edge_type: EdgeType::RunsIn,
            });
            *stats.edges_by_type.entry("runs_in".to_string()).or_insert(0) += 1;

            // Workload assumes a role
            let role_idx = i % num_roles;
            let role_id = nodes[role_start_idx + role_idx].id;
            edges.push(IamEdge {
                source: id,
                target: role_id,
                edge_type: EdgeType::Assumes,
            });
            *stats.edges_by_type.entry("assumes".to_string()).or_insert(0) += 1;
        }

        // ====================================================================
        // INTENTIONAL VIOLATIONS FOR TESTING
        // ====================================================================

        // 1. CROSS-REGION ACCESS VIOLATIONS
        // Create a dependency chain that crosses region boundaries
        // The algorithm detects when you traverse from a node in region A to a node in region B
        if num_regions >= 2 {
            // Find resources in different regions to create cross-region dependencies
            let mut resources_by_region: HashMap<String, Vec<(Id, usize)>> = HashMap::new();
            for (idx, node) in nodes.iter().enumerate() {
                if let Some(ref region) = node.region {
                    if node.node_type == NodeType::Instance || node.node_type == NodeType::Database {
                        resources_by_region
                            .entry(region.clone())
                            .or_default()
                            .push((node.id, idx));
                    }
                }
            }

            let region_names: Vec<_> = resources_by_region.keys().cloned().collect();
            if region_names.len() >= 2 {
                // Get resources from two different regions
                let region_a = &region_names[0];
                let region_b = &region_names[1];

                if let (Some(resources_a), Some(resources_b)) = (
                    resources_by_region.get(region_a),
                    resources_by_region.get(region_b),
                ) {
                    if !resources_a.is_empty() && !resources_b.is_empty() {
                        // Create a cross-region dependency: resource in region A depends on resource in region B
                        // This is a compliance violation - resources should not depend across regions
                        let (resource_a_id, _) = resources_a[0];
                        let (resource_b_id, resource_b_idx) = resources_b[0];
                        let resource_b_region = nodes[resource_b_idx].region.clone();

                        // Create a "cross-region-gateway" instance that bridges the regions
                        let gateway_id = Id::new();
                        nodes.push(IamNode {
                            id: gateway_id,
                            name: "instance-cross-region-gateway".to_string(),
                            node_type: NodeType::Instance,
                            region: Some(region_a.clone()), // Gateway is in region A
                        });
                        resources.push(gateway_id);
                        stats.instances += 1;

                        // Gateway depends on resource in region B (CROSS-REGION VIOLATION!)
                        edges.push(IamEdge {
                            source: gateway_id,
                            target: resource_b_id,
                            edge_type: EdgeType::DependsOn,
                        });
                        *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

                        // Create a policy that grants access to the gateway
                        let cross_region_policy_id = Id::new();
                        nodes.push(IamNode {
                            id: cross_region_policy_id,
                            name: "policy-cross-region-access".to_string(),
                            node_type: NodeType::Policy,
                            region: None,
                        });
                        policies.push(cross_region_policy_id);
                        stats.policies += 1;

                        edges.push(IamEdge {
                            source: cross_region_policy_id,
                            target: gateway_id,
                            edge_type: EdgeType::CanAccess,
                        });
                        *stats.edges_by_type.entry("can_access".to_string()).or_insert(0) += 1;

                        // Create user with this policy
                        // Insert at beginning of users list so it's sampled in cross-region analysis
                        let cross_region_user_id = Id::new();
                        nodes.push(IamNode {
                            id: cross_region_user_id,
                            name: "user-cross-region-admin".to_string(),
                            node_type: NodeType::User,
                            region: None,
                        });
                        users.insert(0, cross_region_user_id); // Insert at beginning!
                        stats.users += 1;

                        edges.push(IamEdge {
                            source: cross_region_user_id,
                            target: cross_region_policy_id,
                            edge_type: EdgeType::HasPolicy,
                        });
                        *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;

                        // Create a second cross-region violation with a different path
                        // Resource in region A depends on VPC in region B
                        if resources_a.len() >= 2 {
                            let (resource_a2_id, _) = resources_a[1];

                            // Find a VPC in region B
                            let vpc_in_region_b: Option<Id> = nodes.iter()
                                .find(|n| n.node_type == NodeType::Vpc && n.region.as_ref() == Some(region_b))
                                .map(|n| n.id);

                            if let Some(vpc_b_id) = vpc_in_region_b {
                                // Create direct cross-region dependency
                                edges.push(IamEdge {
                                    source: resource_a2_id,
                                    target: vpc_b_id,
                                    edge_type: EdgeType::DependsOn,
                                });
                                *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // 2. MINIMAL PRIVILEGE VIOLATIONS
        // Create a scenario where BFS finds a shorter-hop but higher-weight path
        // while Dijkstra finds a longer-hop but lower-weight path
        //
        // Design:
        // - User has two paths to sensitive resource:
        //   Path A (2 hops, high weight): User -> expensive_policy -> resource
        //   Path B (3 hops, low weight):  User -> group -> cheap_policy -> resource
        // - BFS finds Path A first (fewer hops)
        // - Dijkstra finds Path B is cheaper (lower total weight)
        if !groups.is_empty() && sensitive_resources.len() >= 1 {
            let target_resource = sensitive_resources[0];

            // Create an expensive direct policy (high weight = 10.0)
            // This simulates a "privileged admin" policy with high resistance
            let expensive_policy_id = Id::new();
            nodes.push(IamNode {
                id: expensive_policy_id,
                name: "policy-expensive-admin".to_string(),
                node_type: NodeType::Policy,
                region: None,
            });
            policies.push(expensive_policy_id);
            stats.policies += 1;

            // Expensive policy grants access but with VERY high weight (simulating audit overhead)
            // We need to add this edge with high weight - but EdgeType::CanAccess has fixed weight 1.0
            // So we need to use DependsOn which has weight 3.5
            edges.push(IamEdge {
                source: expensive_policy_id,
                target: target_resource,
                edge_type: EdgeType::DependsOn, // Weight 3.5 (vs CanAccess 1.0)
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            // Create a cheap group policy path (lower total weight)
            // group -> cheap_policy (1.5) -> resource (1.0) = 2.5 total
            let cheap_policy_id = Id::new();
            nodes.push(IamNode {
                id: cheap_policy_id,
                name: "policy-standard-access".to_string(),
                node_type: NodeType::Policy,
                region: None,
            });
            policies.push(cheap_policy_id);
            stats.policies += 1;

            edges.push(IamEdge {
                source: cheap_policy_id,
                target: target_resource,
                edge_type: EdgeType::CanAccess, // Weight 1.0
            });
            *stats.edges_by_type.entry("can_access".to_string()).or_insert(0) += 1;

            // Create a special group that has the cheap policy
            let cheap_group_id = Id::new();
            nodes.push(IamNode {
                id: cheap_group_id,
                name: "group-standard-users".to_string(),
                node_type: NodeType::Group,
                region: None,
            });
            groups.push(cheap_group_id);
            stats.groups += 1;

            edges.push(IamEdge {
                source: cheap_group_id,
                target: cheap_policy_id,
                edge_type: EdgeType::HasPolicy, // Weight 1.5
            });
            *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;

            // Create the violation user
            // Path A: user -> expensive_policy (1.5) -> resource (3.5) = 5.0 total, 2 hops
            // Path B: user -> group (2.0) -> cheap_policy (1.5) -> resource (1.0) = 4.5 total, 3 hops
            // BFS finds A first (2 hops < 3 hops), but Dijkstra finds B is cheaper (4.5 < 5.0)
            let violation_user_id = Id::new();
            nodes.push(IamNode {
                id: violation_user_id,
                name: "user-non-minimal-path".to_string(),
                node_type: NodeType::User,
                region: None,
            });
            users.insert(0, violation_user_id); // Insert at beginning to be sampled
            stats.users += 1;

            // IMPORTANT: Add expensive policy FIRST so BFS explores it first
            edges.push(IamEdge {
                source: violation_user_id,
                target: expensive_policy_id,
                edge_type: EdgeType::HasPolicy, // Weight 1.5
            });
            *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;

            // Add group membership SECOND (BFS will explore this later)
            edges.push(IamEdge {
                source: violation_user_id,
                target: cheap_group_id,
                edge_type: EdgeType::MemberOf, // Weight 2.0
            });
            *stats.edges_by_type.entry("member_of".to_string()).or_insert(0) += 1;

            // Create a second violation user with similar pattern
            let violation_user2_id = Id::new();
            nodes.push(IamNode {
                id: violation_user2_id,
                name: "user-suboptimal-access".to_string(),
                node_type: NodeType::User,
                region: None,
            });
            users.insert(0, violation_user2_id);
            stats.users += 1;

            // Same pattern - expensive direct first, cheap group second
            edges.push(IamEdge {
                source: violation_user2_id,
                target: expensive_policy_id,
                edge_type: EdgeType::HasPolicy,
            });
            *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;

            edges.push(IamEdge {
                source: violation_user2_id,
                target: cheap_group_id,
                edge_type: EdgeType::MemberOf,
            });
            *stats.edges_by_type.entry("member_of".to_string()).or_insert(0) += 1;
        }

        // 3. UNUSED ROLES VIOLATIONS
        // Create roles that no workload assumes (orphaned roles)
        for i in 0..2 {
            let unused_role_id = Id::new();
            nodes.push(IamNode {
                id: unused_role_id,
                name: format!("role-unused-{:04}", i),
                node_type: NodeType::Role,
                region: None,
            });
            roles.push(unused_role_id);
            stats.roles += 1;

            // Give the role some policies so it's not completely empty
            if !policies.is_empty() {
                let policy_id = policies[i % policies.len()];
                edges.push(IamEdge {
                    source: unused_role_id,
                    target: policy_id,
                    edge_type: EdgeType::HasPolicy,
                });
                *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
            }

            // Note: We intentionally do NOT create any Assumes edge to these roles
            // This makes them "unused" - they have policies but no workload uses them
        }

        // Create an isolated role (not connected to any policy either)
        let isolated_role_id = Id::new();
        nodes.push(IamNode {
            id: isolated_role_id,
            name: "role-isolated-orphan".to_string(),
            node_type: NodeType::Role,
            region: None,
        });
        roles.push(isolated_role_id);
        stats.roles += 1;
        // This role has NO edges at all - completely isolated

        // 4. MST REDUNDANCY TEST CASES
        // Create parallel high-weight edges that should NOT be in MST
        // These test that Kruskal's algorithm correctly excludes redundant paths

        // Case A: Redundant direct policy access (high weight)
        // If user already reaches a resource via a low-weight path through a group,
        // a parallel high-weight direct path should not be in MST
        if users.len() >= 2 && policies.len() >= 2 && sensitive_resources.len() >= 1 {
            // Create a redundant high-weight policy for MST testing
            let mst_redundant_policy = Id::new();
            nodes.push(IamNode {
                id: mst_redundant_policy,
                name: "policy-mst-redundant-high".to_string(),
                node_type: NodeType::Policy,
                region: None,
            });
            policies.push(mst_redundant_policy);
            stats.policies += 1;

            // Connect to a sensitive resource with HIGH weight (DependsOn = 3.5)
            let target = sensitive_resources[0];
            edges.push(IamEdge {
                source: mst_redundant_policy,
                target,
                edge_type: EdgeType::DependsOn, // Weight 3.5
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            // Connect user to this redundant policy
            // This creates a parallel path with higher total weight
            // User -> policy (1.5) -> resource (3.5) = 5.0 total
            // vs existing User -> group -> policy -> resource paths
            let user_for_mst = users[1];
            edges.push(IamEdge {
                source: user_for_mst,
                target: mst_redundant_policy,
                edge_type: EdgeType::HasPolicy, // Weight 1.5
            });
            *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
        }

        // Case B: Redundant resource dependency chain
        // Create a chain of DependsOn edges (each weight 3.5) that forms a cycle
        // with an existing lower-weight path
        if resources.len() >= 4 {
            // Connect resources in a redundant chain
            // resource[0] -> resource[1] already connected somewhere
            // Add: resource[1] -> resource[2] -> resource[3] -> resource[0] (cycle)
            // MST should exclude one edge to break the cycle

            edges.push(IamEdge {
                source: resources[1],
                target: resources[2],
                edge_type: EdgeType::DependsOn, // Weight 3.5
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            edges.push(IamEdge {
                source: resources[2],
                target: resources[3],
                edge_type: EdgeType::DependsOn, // Weight 3.5
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            // This edge creates a cycle - should be excluded from MST
            edges.push(IamEdge {
                source: resources[3],
                target: resources[0],
                edge_type: EdgeType::DependsOn, // Weight 3.5 - REDUNDANT
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;
        }

        // Case C: Parallel role assumption paths
        // Create two roles that both lead to the same workload
        // One with lower-weight path, one with higher-weight path
        if workloads.len() >= 1 && roles.len() >= 2 {
            // Create a redundant role with higher-weight connection
            let mst_parallel_role = Id::new();
            nodes.push(IamNode {
                id: mst_parallel_role,
                name: "role-mst-parallel-high".to_string(),
                node_type: NodeType::Role,
                region: None,
            });
            roles.push(mst_parallel_role);
            stats.roles += 1;

            // This role assumes via DependsOn (3.5) instead of Assumes (2.5)
            // Creating a higher-weight parallel path to the same workload
            let target_workload = workloads[0];
            edges.push(IamEdge {
                source: mst_parallel_role,
                target: target_workload,
                edge_type: EdgeType::DependsOn, // Weight 3.5 (higher than Assumes 2.5)
            });
            *stats.edges_by_type.entry("depends_on".to_string()).or_insert(0) += 1;

            // Connect to existing policy
            if !policies.is_empty() {
                edges.push(IamEdge {
                    source: mst_parallel_role,
                    target: policies[0],
                    edge_type: EdgeType::HasPolicy,
                });
                *stats.edges_by_type.entry("has_policy".to_string()).or_insert(0) += 1;
            }
        }

        IamGraph {
            nodes,
            edges,
            stats,
            users,
            groups,
            policies,
            roles,
            workloads,
            resources,
            regions,
            sensitive_resources,
        }
    }

    fn to_graph_nodes_edges(&self) -> (Vec<GraphNode>, Vec<GraphEdge>) {
        // Create a set of sensitive resource IDs for quick lookup
        let sensitive_set: std::collections::HashSet<_> = self.sensitive_resources.iter().copied().collect();

        let nodes: Vec<GraphNode> = self
            .nodes
            .iter()
            .map(|n| {
                // Build tagged summary for fulltext search
                let mut summary_parts = vec![
                    format!("type:{}", n.node_type.as_str()),
                ];
                if let Some(ref region) = n.region {
                    summary_parts.push(format!("region:{}", region));
                }
                if sensitive_set.contains(&n.id) {
                    summary_parts.push("sensitive:true".to_string());
                }
                // Include the name for searchability
                summary_parts.push(n.name.clone());

                GraphNode {
                    id: n.id,
                    name: n.name.clone(),
                    summary: Some(summary_parts.join(" ")),
                }
            })
            .collect();

        let edges: Vec<GraphEdge> = self
            .edges
            .iter()
            .map(|e| {
                // Build tagged summary for edge relationships
                let summary = format!("type:{}", e.edge_type.as_str());

                GraphEdge {
                    source: e.source,
                    target: e.target,
                    name: e.edge_type.as_str().to_string(),
                    weight: Some(e.edge_type.weight()),
                    summary: Some(summary),
                }
            })
            .collect();

        (nodes, edges)
    }

    fn build_petgraph(&self) -> (DiGraph<String, f64>, HashMap<Id, NodeIndex>, HashMap<NodeIndex, Id>) {
        let mut graph = DiGraph::new();
        let mut id_to_idx: HashMap<Id, NodeIndex> = HashMap::new();
        let mut idx_to_id: HashMap<NodeIndex, Id> = HashMap::new();

        // Add nodes
        for node in &self.nodes {
            let idx = graph.add_node(node.name.clone());
            id_to_idx.insert(node.id, idx);
            idx_to_id.insert(idx, node.id);
        }

        // Add edges
        for edge in &self.edges {
            let src_idx = id_to_idx[&edge.source];
            let dst_idx = id_to_idx[&edge.target];
            graph.add_edge(src_idx, dst_idx, edge.edge_type.weight());
        }

        (graph, id_to_idx, idx_to_id)
    }
}

// ============================================================================
// Use Case Definitions
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseCase {
    Reachability,
    BlastRadius,
    LeastResistance,
    PrivilegeClustering,
    OverPrivileged,
    CrossRegionAccess,
    UnusedRoles,
    PrivilegeHubs,
    MinimalPrivilege,
    AccessibleResources,
    HighValueTargets,
    MinimumSpanningTree,
}

impl UseCase {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "reachability" | "reach" => Some(UseCase::Reachability),
            "blast_radius" | "blast" => Some(UseCase::BlastRadius),
            "least_resistance" | "least" | "path" => Some(UseCase::LeastResistance),
            "privilege_clustering" | "cluster" | "louvain" => Some(UseCase::PrivilegeClustering),
            "over_privileged" | "over" | "excessive" => Some(UseCase::OverPrivileged),
            "cross_region" | "cross" | "region" => Some(UseCase::CrossRegionAccess),
            "unused_roles" | "unused" | "isolated" | "scc" | "kosaraju" => Some(UseCase::UnusedRoles),
            "privilege_hubs" | "hubs" | "degree" => Some(UseCase::PrivilegeHubs),
            "minimal_privilege" | "minimal" | "verify" => Some(UseCase::MinimalPrivilege),
            "accessible_resources" | "accessible" | "resources" | "access_list" => Some(UseCase::AccessibleResources),
            "high_value_targets" | "high_value" | "pagerank" | "targets" => Some(UseCase::HighValueTargets),
            "minimum_spanning_tree" | "mst" | "spanning_tree" | "kruskal" => Some(UseCase::MinimumSpanningTree),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            UseCase::Reachability => "Reachability Analysis",
            UseCase::BlastRadius => "Blast Radius Analysis",
            UseCase::LeastResistance => "Least Resistance Path",
            UseCase::PrivilegeClustering => "Privilege Clustering",
            UseCase::OverPrivileged => "Over-Privileged Detection",
            UseCase::CrossRegionAccess => "Cross-Region Access",
            UseCase::UnusedRoles => "Unused Roles Detection",
            UseCase::PrivilegeHubs => "Privilege Hubs Detection",
            UseCase::MinimalPrivilege => "Minimal Privilege Verification",
            UseCase::AccessibleResources => "Accessible Resources Listing",
            UseCase::HighValueTargets => "High Value Targets Detection",
            UseCase::MinimumSpanningTree => "Minimum Spanning Tree",
        }
    }

    fn algorithm(&self) -> &'static str {
        match self {
            UseCase::Reachability => "BFS (Breadth-First Search)",
            UseCase::BlastRadius => "BFS with depth tracking",
            UseCase::LeastResistance => "Dijkstra's Algorithm",
            UseCase::PrivilegeClustering => "Louvain Community Detection",
            UseCase::OverPrivileged => "Out-degree analysis + BFS",
            UseCase::CrossRegionAccess => "Filtered BFS with region tracking",
            UseCase::UnusedRoles => "Kosaraju's SCC Algorithm",
            UseCase::PrivilegeHubs => "Manual in/out-degree calculation",
            UseCase::MinimalPrivilege => "Dijkstra path verification",
            UseCase::AccessibleResources => "DFS/BFS traversal",
            UseCase::HighValueTargets => "PageRank algorithm",
            UseCase::MinimumSpanningTree => "Kruskal's Algorithm with Union-Find",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            UseCase::Reachability => "Determine if a user can access a specific resource through any permission path",
            UseCase::BlastRadius => "Find all resources affected if a user's credentials are compromised",
            UseCase::LeastResistance => "Find the easiest (lowest weight) path from user to sensitive resource",
            UseCase::PrivilegeClustering => "Group users by similar access patterns using community detection",
            UseCase::OverPrivileged => "Identify users with access to many sensitive resources",
            UseCase::CrossRegionAccess => "Find access paths that cross region boundaries (compliance risk)",
            UseCase::UnusedRoles => "Find isolated/unused roles via SCCs for permission cleanup and redundancy trimming",
            UseCase::PrivilegeHubs => "Spot over-privileged entities via manual degree calculation (high connectivity nodes)",
            UseCase::MinimalPrivilege => "Verify existing permission paths are minimal (no shorter alternatives exist)",
            UseCase::AccessibleResources => "List all resources a user can access via DFS/BFS traversal",
            UseCase::HighValueTargets => "Identify high-value targets using PageRank (nodes with many incoming permission paths)",
            UseCase::MinimumSpanningTree => "Find minimal permission infrastructure using MST. Edges NOT in MST are redundant and could be removed.",
        }
    }

    fn all() -> Vec<UseCase> {
        vec![
            UseCase::Reachability,
            UseCase::BlastRadius,
            UseCase::LeastResistance,
            UseCase::PrivilegeClustering,
            UseCase::OverPrivileged,
            UseCase::CrossRegionAccess,
            UseCase::UnusedRoles,
            UseCase::PrivilegeHubs,
            UseCase::MinimalPrivilege,
            UseCase::AccessibleResources,
            UseCase::HighValueTargets,
            UseCase::MinimumSpanningTree,
        ]
    }
}

// ============================================================================
// Analysis Results
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisResult {
    use_case: String,
    algorithm: String,
    summary: String,
    details: Vec<String>,
}

impl std::hash::Hash for AnalysisResult {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.use_case.hash(state);
        self.summary.hash(state);
        // Sort details for deterministic hashing
        let mut sorted_details = self.details.clone();
        sorted_details.sort();
        sorted_details.hash(state);
    }
}

// ============================================================================
// Reference (petgraph) Implementations
// ============================================================================

mod reference {
    use super::*;

    /// Reachability: Can source reach target?
    pub fn reachability(
        graph: &DiGraph<String, f64>,
        source: NodeIndex,
        target: NodeIndex,
    ) -> AnalysisResult {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut path = Vec::new();

        queue.push_back((source, vec![source]));
        visited.insert(source);

        let reachable = loop {
            if let Some((current, current_path)) = queue.pop_front() {
                if current == target {
                    path = current_path;
                    break true;
                }

                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        let mut new_path = current_path.clone();
                        new_path.push(neighbor);
                        queue.push_back((neighbor, new_path));
                    }
                }
            } else {
                break false;
            }
        };

        let path_str: Vec<String> = path.iter().map(|idx| graph[*idx].clone()).collect();

        AnalysisResult {
            use_case: "Reachability".to_string(),
            algorithm: "BFS".to_string(),
            summary: format!(
                "Reachable: {}, Path length: {}",
                reachable,
                if reachable { path.len() } else { 0 }
            ),
            details: if reachable {
                vec![format!("Path: {}", path_str.join(" -> "))]
            } else {
                vec!["No path found".to_string()]
            },
        }
    }

    /// Blast Radius: All reachable nodes from source with depth
    pub fn blast_radius(
        graph: &DiGraph<String, f64>,
        source: NodeIndex,
        max_depth: usize,
    ) -> AnalysisResult {
        let mut visited: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut resources_by_depth: HashMap<usize, Vec<String>> = HashMap::new();
        let mut all_nodes_by_depth: HashMap<usize, Vec<String>> = HashMap::new();

        queue.push_back((source, 0));
        visited.insert(source, 0);

        while let Some((current, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            let node_name = &graph[current];

            // Track ALL visited nodes for visualization
            all_nodes_by_depth
                .entry(depth)
                .or_insert_with(Vec::new)
                .push(node_name.clone());

            // Track resources separately for summary
            if node_name.starts_with("instance-")
                || node_name.starts_with("database-")
                || node_name.starts_with("disk-")
                || node_name.starts_with("vpc-")
            {
                resources_by_depth
                    .entry(depth)
                    .or_insert_with(Vec::new)
                    .push(node_name.clone());
            }

            for neighbor in graph.neighbors(current) {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        let total_resources: usize = resources_by_depth.values().map(|v| v.len()).sum();
        let total_nodes: usize = all_nodes_by_depth.values().map(|v| v.len()).sum();
        let mut details = Vec::new();

        // First add summary by depth (resources only)
        for depth in 0..=max_depth {
            if let Some(resources) = resources_by_depth.get(&depth) {
                details.push(format!("Depth {}: {} resources", depth, resources.len()));
            }
        }

        // Then add ALL nodes with their depth (for visualization path)
        // Format: "nodename (depth N)" for resources, "nodename (path depth N)" for intermediate nodes
        for depth in 0..=max_depth {
            if let Some(nodes) = all_nodes_by_depth.get(&depth) {
                for node in nodes {
                    let is_resource = node.starts_with("instance-")
                        || node.starts_with("database-")
                        || node.starts_with("disk-")
                        || node.starts_with("vpc-");
                    if is_resource {
                        details.push(format!("{} (depth {})", node, depth));
                    } else {
                        details.push(format!("{} (path depth {})", node, depth));
                    }
                }
            }
        }

        AnalysisResult {
            use_case: "Blast Radius".to_string(),
            algorithm: "BFS with depth".to_string(),
            summary: format!(
                "Total resources at risk: {}, Total nodes in blast radius: {}, Max depth: {}",
                total_resources, total_nodes, max_depth
            ),
            details,
        }
    }

    /// Least Resistance: Shortest weighted path to any sensitive resource
    pub fn least_resistance(
        graph: &DiGraph<String, f64>,
        source: NodeIndex,
        sensitive_targets: &[NodeIndex],
    ) -> AnalysisResult {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Clone)]
        struct State {
            cost: f64,
            node: NodeIndex,
        }

        impl Eq for State {}
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let sensitive_set: HashSet<_> = sensitive_targets.iter().copied().collect();
        let mut distances: HashMap<NodeIndex, f64> = HashMap::new();
        let mut predecessors: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(source, 0.0);
        heap.push(State { cost: 0.0, node: source });

        while let Some(State { cost, node }) = heap.pop() {
            if let Some(&best_cost) = distances.get(&node) {
                if cost > best_cost {
                    continue;
                }
            }

            for edge in graph.edges(node) {
                let next = edge.target();
                let weight = *edge.weight();
                let next_cost = cost + weight;

                if next_cost < *distances.get(&next).unwrap_or(&f64::INFINITY) {
                    distances.insert(next, next_cost);
                    predecessors.insert(next, node);
                    heap.push(State { cost: next_cost, node: next });
                }
            }
        }

        // Find best target among sensitive resources
        let mut best_target: Option<(NodeIndex, f64)> = None;
        for &target in sensitive_targets {
            if let Some(&dist) = distances.get(&target) {
                if best_target.is_none() || dist < best_target.unwrap().1 {
                    best_target = Some((target, dist));
                }
            }
        }

        match best_target {
            Some((target, cost)) => {
                // Reconstruct path from source to target
                let mut path = vec![target];
                let mut current = target;
                while let Some(&pred) = predecessors.get(&current) {
                    path.push(pred);
                    current = pred;
                    if current == source {
                        break;
                    }
                }
                path.reverse();

                let path_str: Vec<String> = path.iter().map(|&idx| graph[idx].clone()).collect();

                AnalysisResult {
                    use_case: "Least Resistance".to_string(),
                    algorithm: "Dijkstra".to_string(),
                    summary: format!(
                        "Easiest path cost: {:.2}, Target: {}",
                        cost, graph[target]
                    ),
                    details: vec![
                        format!("Path: {}", path_str.join(" -> ")),
                        format!("Total weight: {:.2}", cost),
                    ],
                }
            },
            None => AnalysisResult {
                use_case: "Least Resistance".to_string(),
                algorithm: "Dijkstra".to_string(),
                summary: "No path to sensitive resources".to_string(),
                details: vec!["No reachable sensitive resources from source".to_string()],
            },
        }
    }

    /// Privilege Clustering: Group users by access patterns (simplified Louvain)
    pub fn privilege_clustering(
        graph: &DiGraph<String, f64>,
        user_indices: &[NodeIndex],
    ) -> AnalysisResult {
        // For each user, compute their "access fingerprint" (set of reachable resources)
        let mut user_fingerprints: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();

        for &user in user_indices {
            let mut reachable = HashSet::new();
            let mut bfs = Bfs::new(graph, user);
            while let Some(node) = bfs.next(graph) {
                let name = &graph[node];
                if name.starts_with("instance-")
                    || name.starts_with("database-")
                    || name.starts_with("disk-")
                    || name.starts_with("vpc-")
                {
                    reachable.insert(node);
                }
            }
            user_fingerprints.insert(user, reachable);
        }

        // Cluster users by Jaccard similarity of their fingerprints
        // Simplified: group users with >50% overlap
        let mut clusters: Vec<HashSet<NodeIndex>> = Vec::new();

        for &user in user_indices {
            let user_fp = &user_fingerprints[&user];
            let mut found_cluster = false;

            for cluster in &mut clusters {
                // Check similarity with first member of cluster
                if let Some(&representative) = cluster.iter().next() {
                    let rep_fp = &user_fingerprints[&representative];
                    let intersection = user_fp.intersection(rep_fp).count();
                    let union = user_fp.union(rep_fp).count();
                    let jaccard = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    if jaccard > 0.5 {
                        cluster.insert(user);
                        found_cluster = true;
                        break;
                    }
                }
            }

            if !found_cluster {
                let mut new_cluster = HashSet::new();
                new_cluster.insert(user);
                clusters.push(new_cluster);
            }
        }

        // Sort clusters by size (largest first)
        clusters.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut details = Vec::new();
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
        details.push(format!(
            "Cluster sizes (top 5): {:?}",
            cluster_sizes.iter().take(5).collect::<Vec<_>>()
        ));

        // Output cluster members (format: "Cluster N: user-0001, user-0002, ...")
        for (i, cluster) in clusters.iter().take(10).enumerate() {
            let mut members: Vec<String> = cluster
                .iter()
                .map(|&idx| graph[idx].clone())
                .collect();
            members.sort();
            details.push(format!("Cluster {}: {}", i + 1, members.join(", ")));
        }

        AnalysisResult {
            use_case: "Privilege Clustering".to_string(),
            algorithm: "Jaccard Similarity Clustering".to_string(),
            summary: format!(
                "Found {} clusters from {} users",
                clusters.len(),
                user_indices.len()
            ),
            details,
        }
    }

    /// Over-Privileged: Find users with access to many sensitive resources
    pub fn over_privileged(
        graph: &DiGraph<String, f64>,
        user_indices: &[NodeIndex],
        sensitive_indices: &[NodeIndex],
        threshold: usize,
    ) -> AnalysisResult {
        let sensitive_set: HashSet<_> = sensitive_indices.iter().copied().collect();
        let mut over_privileged_users = Vec::new();

        for &user in user_indices {
            let mut count = 0;
            let mut bfs = Bfs::new(graph, user);
            while let Some(node) = bfs.next(graph) {
                if sensitive_set.contains(&node) {
                    count += 1;
                }
            }
            if count >= threshold {
                over_privileged_users.push((graph[user].clone(), count));
            }
        }

        over_privileged_users.sort_by(|a, b| b.1.cmp(&a.1));

        AnalysisResult {
            use_case: "Over-Privileged Detection".to_string(),
            algorithm: "BFS + counting".to_string(),
            summary: format!(
                "Found {} users with access to {} or more sensitive resources",
                over_privileged_users.len(),
                threshold
            ),
            details: over_privileged_users
                .iter()
                .take(10)
                .map(|(name, count)| format!("{}: {} sensitive resources", name, count))
                .collect(),
        }
    }

    /// Cross-Region Access: Find paths that cross region boundaries
    pub fn cross_region_access(
        graph: &DiGraph<String, f64>,
        user_indices: &[NodeIndex],
        id_to_region: &HashMap<NodeIndex, String>,
    ) -> AnalysisResult {
        let mut cross_region_paths = Vec::new();

        for &user in user_indices.iter().take(50) {
            // Sample first 50 users
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back((user, None::<String>, vec![graph[user].clone()]));
            visited.insert(user);

            while let Some((current, current_region, path)) = queue.pop_front() {
                if path.len() > 10 {
                    continue; // Limit path length
                }

                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);

                        let neighbor_region = id_to_region.get(&neighbor).cloned();
                        let mut new_path = path.clone();
                        new_path.push(graph[neighbor].clone());

                        // Check for region crossing
                        if let (Some(ref curr_reg), Some(ref next_reg)) =
                            (&current_region, &neighbor_region)
                        {
                            if curr_reg != next_reg {
                                cross_region_paths
                                    .push((new_path.clone(), curr_reg.clone(), next_reg.clone()));
                            }
                        }

                        queue.push_back((neighbor, neighbor_region, new_path));
                    }
                }
            }
        }

        AnalysisResult {
            use_case: "Cross-Region Access".to_string(),
            algorithm: "Filtered BFS".to_string(),
            summary: format!("Found {} cross-region access paths", cross_region_paths.len()),
            details: cross_region_paths
                .iter()
                .take(5)
                .map(|(path, from, to)| {
                    format!(
                        "{} -> {} (path len: {})",
                        from,
                        to,
                        path.len()
                    )
                })
                .collect(),
        }
    }

    /// Unused Roles: Find isolated roles using Kosaraju's SCC algorithm
    /// A role is considered "unused" if:
    /// 1. It's in a small SCC (not connected to the main permission graph)
    /// 2. No workloads assume it (no incoming ASSUMES edges)
    /// 3. It has policies but those policies don't grant meaningful access
    pub fn unused_roles(
        graph: &DiGraph<String, f64>,
        role_indices: &[NodeIndex],
        workload_indices: &[NodeIndex],
    ) -> AnalysisResult {
        // Run Kosaraju's SCC algorithm
        let sccs = kosaraju_scc(graph);

        // Build a map from node to its SCC index and size
        let mut node_to_scc: HashMap<NodeIndex, usize> = HashMap::new();
        let mut scc_sizes: Vec<usize> = Vec::new();
        for (scc_idx, scc) in sccs.iter().enumerate() {
            scc_sizes.push(scc.len());
            for &node in scc {
                node_to_scc.insert(node, scc_idx);
            }
        }

        // Find the largest SCC (main connected component)
        let main_scc_idx = scc_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &size)| size)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let main_scc_size = scc_sizes.get(main_scc_idx).copied().unwrap_or(0);

        // Build set of roles that are assumed by workloads
        let mut assumed_roles: HashSet<NodeIndex> = HashSet::new();
        for &workload in workload_indices {
            for neighbor in graph.neighbors(workload) {
                let name = &graph[neighbor];
                if name.starts_with("role-") {
                    assumed_roles.insert(neighbor);
                }
            }
        }

        // Categorize roles
        let mut isolated_roles: Vec<(String, &str)> = Vec::new(); // (name, reason)
        let mut small_scc_roles = 0;
        let mut unassumed_roles = 0;

        for &role in role_indices {
            let role_name = graph[role].clone();
            let scc_idx = node_to_scc.get(&role).copied().unwrap_or(0);
            let scc_size = scc_sizes.get(scc_idx).copied().unwrap_or(1);

            let is_in_small_scc = scc_idx != main_scc_idx && scc_size < main_scc_size / 10;
            let is_unassumed = !assumed_roles.contains(&role);

            if is_in_small_scc {
                small_scc_roles += 1;
                isolated_roles.push((role_name.clone(), "isolated SCC"));
            } else if is_unassumed {
                unassumed_roles += 1;
                isolated_roles.push((role_name, "no workloads assume"));
            }
        }

        // Sort by name for deterministic output
        isolated_roles.sort_by(|a, b| a.0.cmp(&b.0));

        AnalysisResult {
            use_case: "Unused Roles Detection".to_string(),
            algorithm: "Kosaraju SCC".to_string(),
            summary: format!(
                "Found {} unused roles: {} in isolated SCCs, {} unassumed (total roles: {}, SCCs: {}, main SCC size: {})",
                isolated_roles.len(),
                small_scc_roles,
                unassumed_roles,
                role_indices.len(),
                sccs.len(),
                main_scc_size
            ),
            details: isolated_roles
                .iter()
                .take(15)
                .map(|(name, reason)| format!("{}: {}", name, reason))
                .collect(),
        }
    }

    /// Privilege Hubs: Find over-privileged entities via manual degree calculation
    /// A hub is an entity with high in-degree + out-degree (many connections)
    /// These are potential security risks as compromising them affects many paths
    pub fn privilege_hubs(
        graph: &DiGraph<String, f64>,
        threshold_percentile: f64, // e.g., 0.90 means top 10% by degree
    ) -> AnalysisResult {
        // Manual degree calculation
        let mut in_degrees: HashMap<NodeIndex, usize> = HashMap::new();
        let mut out_degrees: HashMap<NodeIndex, usize> = HashMap::new();

        // Initialize all nodes with 0 degree
        for node in graph.node_indices() {
            in_degrees.insert(node, 0);
            out_degrees.insert(node, 0);
        }

        // Count degrees by iterating through all edges
        for edge in graph.edge_references() {
            let source = edge.source();
            let target = edge.target();
            *out_degrees.get_mut(&source).unwrap() += 1;
            *in_degrees.get_mut(&target).unwrap() += 1;
        }

        // Calculate total degree (in + out) for each node
        let mut node_degrees: Vec<(NodeIndex, usize, usize, usize)> = graph
            .node_indices()
            .map(|node| {
                let in_deg = in_degrees[&node];
                let out_deg = out_degrees[&node];
                (node, in_deg, out_deg, in_deg + out_deg)
            })
            .collect();

        // Sort by total degree descending
        node_degrees.sort_by(|a, b| b.3.cmp(&a.3));

        // Find threshold for "hub" classification
        let threshold_idx = ((1.0 - threshold_percentile) * node_degrees.len() as f64) as usize;
        let threshold_degree = if threshold_idx < node_degrees.len() {
            node_degrees[threshold_idx].3
        } else {
            0
        };

        // Classify hubs by type
        let mut hubs_by_type: HashMap<&str, Vec<(String, usize, usize)>> = HashMap::new();
        let mut total_hubs = 0;

        for (node, in_deg, out_deg, total_deg) in &node_degrees {
            if *total_deg >= threshold_degree && *total_deg > 0 {
                let name = &graph[*node];
                let node_type = if name.starts_with("user-") {
                    "Users"
                } else if name.starts_with("group-") {
                    "Groups"
                } else if name.starts_with("policy-") {
                    "Policies"
                } else if name.starts_with("role-") {
                    "Roles"
                } else if name.starts_with("workload-") {
                    "Workloads"
                } else if name.starts_with("instance-") || name.starts_with("database-")
                    || name.starts_with("disk-") || name.starts_with("vpc-") {
                    "Resources"
                } else {
                    "Other"
                };

                hubs_by_type
                    .entry(node_type)
                    .or_insert_with(Vec::new)
                    .push((name.clone(), *in_deg, *out_deg));
                total_hubs += 1;
            }
        }

        // Build details
        let mut details = Vec::new();
        let type_order = ["Policies", "Groups", "Roles", "Users", "Workloads", "Resources", "Other"];
        for node_type in type_order {
            if let Some(hubs) = hubs_by_type.get(node_type) {
                details.push(format!("{}: {} hubs", node_type, hubs.len()));
                for (name, in_deg, out_deg) in hubs.iter().take(3) {
                    details.push(format!("  {} (in:{}, out:{})", name, in_deg, out_deg));
                }
            }
        }

        // Calculate statistics
        let avg_degree: f64 = if !node_degrees.is_empty() {
            node_degrees.iter().map(|(_, _, _, d)| *d as f64).sum::<f64>() / node_degrees.len() as f64
        } else {
            0.0
        };
        let max_degree = node_degrees.first().map(|(_, _, _, d)| *d).unwrap_or(0);

        AnalysisResult {
            use_case: "Privilege Hubs Detection".to_string(),
            algorithm: "Manual Degree Calculation".to_string(),
            summary: format!(
                "Found {} privilege hubs (top {:.0}%, degree >= {}). Avg degree: {:.1}, Max: {}",
                total_hubs,
                (1.0 - threshold_percentile) * 100.0,
                threshold_degree,
                avg_degree,
                max_degree
            ),
            details,
        }
    }

    /// Minimal Privilege Verification: Check if permission paths are truly minimal
    /// Uses Dijkstra to find shortest paths and compares with actual paths
    /// Reports paths that could be shortened (privilege escalation opportunities)
    pub fn minimal_privilege(
        graph: &DiGraph<String, f64>,
        user_indices: &[NodeIndex],
        sensitive_indices: &[NodeIndex],
    ) -> AnalysisResult {
        let mut non_minimal_paths = Vec::new();
        let mut total_paths_checked = 0;
        let mut minimal_paths = 0;

        // Sample users to check (limit for performance)
        let users_to_check: Vec<_> = user_indices.iter().take(20).copied().collect();

        for user in &users_to_check {
            // Compute shortest paths from this user using Dijkstra
            let shortest_paths = dijkstra(graph, *user, None, |e| *e.weight());

            // For each sensitive resource, check if there's a path
            for &target in sensitive_indices.iter().take(10) {
                if let Some(&shortest_cost) = shortest_paths.get(&target) {
                    total_paths_checked += 1;

                    // Now trace the actual first path we find via BFS
                    // and compare its cost to the Dijkstra optimal
                    let mut visited = HashSet::new();
                    let mut queue = VecDeque::new();
                    queue.push_back((*user, 0.0_f64, vec![*user]));
                    visited.insert(*user);

                    let mut actual_path_cost: Option<f64> = None;
                    let mut actual_path: Vec<NodeIndex> = Vec::new();

                    while let Some((current, cost, path)) = queue.pop_front() {
                        if current == target {
                            actual_path_cost = Some(cost);
                            actual_path = path;
                            break;
                        }

                        for neighbor in graph.neighbors(current) {
                            if !visited.contains(&neighbor) {
                                visited.insert(neighbor);
                                let edge = graph.find_edge(current, neighbor).unwrap();
                                let edge_weight = *graph.edge_weight(edge).unwrap();
                                let mut new_path = path.clone();
                                new_path.push(neighbor);
                                queue.push_back((neighbor, cost + edge_weight, new_path));
                            }
                        }
                    }

                    if let Some(actual_cost) = actual_path_cost {
                        // Check if the BFS path is optimal
                        let cost_diff = actual_cost - shortest_cost;
                        if cost_diff > 0.01 {
                            // Path is not minimal (BFS found a longer path)
                            non_minimal_paths.push((
                                graph[*user].clone(),
                                graph[target].clone(),
                                actual_cost,
                                shortest_cost,
                                actual_path.len(),
                            ));
                        } else {
                            minimal_paths += 1;
                        }
                    }
                }
            }
        }

        // Sort by cost difference (worst violations first)
        non_minimal_paths.sort_by(|a, b| {
            let diff_a = a.2 - a.3;
            let diff_b = b.2 - b.3;
            diff_b.partial_cmp(&diff_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let details: Vec<String> = non_minimal_paths
            .iter()
            .take(10)
            .map(|(user, target, actual, optimal, hops)| {
                format!(
                    "{} -> {}: actual={:.1}, optimal={:.1}, excess={:.1}, hops={}",
                    user, target, actual, optimal, actual - optimal, hops
                )
            })
            .collect();

        AnalysisResult {
            use_case: "Minimal Privilege Verification".to_string(),
            algorithm: "Dijkstra Comparison".to_string(),
            summary: format!(
                "Checked {} paths: {} minimal, {} non-minimal ({:.1}% optimal)",
                total_paths_checked,
                minimal_paths,
                non_minimal_paths.len(),
                if total_paths_checked > 0 {
                    minimal_paths as f64 / total_paths_checked as f64 * 100.0
                } else {
                    100.0
                }
            ),
            details,
        }
    }

    /// Accessible Resources: List all resources a user can access via DFS traversal
    pub fn accessible_resources(
        graph: &DiGraph<String, f64>,
        user_indices: &[NodeIndex],
    ) -> AnalysisResult {
        let mut all_accessible: HashMap<String, HashSet<String>> = HashMap::new(); // user -> resources
        let mut resource_access_count: HashMap<String, usize> = HashMap::new();

        // For each user, do a DFS to find all accessible resources
        for &user in user_indices.iter().take(50) {
            let user_name = graph[user].clone();
            let mut accessible = HashSet::new();
            let mut stack = vec![user];
            let mut visited = HashSet::new();

            // DFS traversal
            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);

                let name = &graph[current];
                // Check if it's a resource type
                if name.starts_with("instance-") || name.starts_with("database-")
                    || name.starts_with("disk-") || name.starts_with("vpc-")
                {
                    accessible.insert(name.clone());
                    *resource_access_count.entry(name.clone()).or_insert(0) += 1;
                }

                // Add neighbors to stack
                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }

            all_accessible.insert(user_name, accessible);
        }

        // Calculate statistics
        let total_users = all_accessible.len();
        let total_resources_accessed: usize = all_accessible.values().map(|r| r.len()).sum();
        let avg_resources = if total_users > 0 {
            total_resources_accessed as f64 / total_users as f64
        } else {
            0.0
        };

        // Find most accessed resources
        let mut most_accessed: Vec<_> = resource_access_count.iter().collect();
        most_accessed.sort_by(|a, b| b.1.cmp(a.1));

        // Find users with most access
        let mut users_by_access: Vec<_> = all_accessible.iter()
            .map(|(u, r)| (u.clone(), r.len()))
            .collect();
        users_by_access.sort_by(|a, b| b.1.cmp(&a.1));

        let mut details = Vec::new();
        details.push(format!("Most accessed resources:"));
        for (resource, count) in most_accessed.iter().take(5) {
            details.push(format!("  {}: {} users", resource, count));
        }
        details.push(format!("Users with most access:"));
        for (user, count) in users_by_access.iter().take(5) {
            details.push(format!("  {}: {} resources", user, count));
        }

        AnalysisResult {
            use_case: "Accessible Resources Listing".to_string(),
            algorithm: "DFS Traversal".to_string(),
            summary: format!(
                "Analyzed {} users: avg {:.1} resources/user, {} unique resources accessed",
                total_users,
                avg_resources,
                resource_access_count.len()
            ),
            details,
        }
    }

    /// High Value Targets: Use PageRank to identify high-value targets
    /// Resources with high PageRank have many incoming permission paths
    pub fn high_value_targets(
        graph: &DiGraph<String, f64>,
        damping: f64,
        iterations: usize,
    ) -> AnalysisResult {
        let n = graph.node_count();
        if n == 0 {
            return AnalysisResult {
                use_case: "High Value Targets Detection".to_string(),
                algorithm: "PageRank".to_string(),
                summary: "Empty graph".to_string(),
                details: vec![],
            };
        }

        // Initialize PageRank scores
        let mut scores: HashMap<NodeIndex, f64> = HashMap::new();
        let initial_score = 1.0 / n as f64;
        for node in graph.node_indices() {
            scores.insert(node, initial_score);
        }

        // Build reverse adjacency (incoming edges) and out-degrees
        let mut in_edges: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        let mut out_degrees: HashMap<NodeIndex, usize> = HashMap::new();

        for node in graph.node_indices() {
            in_edges.insert(node, Vec::new());
            out_degrees.insert(node, 0);
        }

        for edge in graph.edge_references() {
            let src = edge.source();
            let tgt = edge.target();
            in_edges.get_mut(&tgt).unwrap().push(src);
            *out_degrees.get_mut(&src).unwrap() += 1;
        }

        // PageRank iterations
        for _ in 0..iterations {
            let mut new_scores: HashMap<NodeIndex, f64> = HashMap::new();
            let base = (1.0 - damping) / n as f64;

            for node in graph.node_indices() {
                let mut incoming_score = 0.0;
                if let Some(predecessors) = in_edges.get(&node) {
                    for &pred in predecessors {
                        let pred_out_deg = out_degrees.get(&pred).copied().unwrap_or(1).max(1);
                        incoming_score += scores[&pred] / pred_out_deg as f64;
                    }
                }
                new_scores.insert(node, base + damping * incoming_score);
            }
            scores = new_scores;
        }

        // Classify nodes by type and rank by PageRank
        let mut resources_ranked: Vec<(String, f64)> = Vec::new();
        let mut roles_ranked: Vec<(String, f64)> = Vec::new();
        let mut policies_ranked: Vec<(String, f64)> = Vec::new();

        for (node, score) in &scores {
            let name = &graph[*node];
            if name.starts_with("instance-") || name.starts_with("database-")
                || name.starts_with("disk-") || name.starts_with("vpc-")
            {
                resources_ranked.push((name.clone(), *score));
            } else if name.starts_with("role-") {
                roles_ranked.push((name.clone(), *score));
            } else if name.starts_with("policy-") {
                policies_ranked.push((name.clone(), *score));
            }
        }

        // Sort by score descending
        resources_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        roles_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        policies_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut details = Vec::new();
        details.push("High-value resources (most reachable):".to_string());
        for (name, score) in resources_ranked.iter().take(5) {
            details.push(format!("  {}: {:.6}", name, score));
        }
        details.push("High-value roles (permission bottlenecks):".to_string());
        for (name, score) in roles_ranked.iter().take(3) {
            details.push(format!("  {}: {:.6}", name, score));
        }
        details.push("High-value policies (widely granted):".to_string());
        for (name, score) in policies_ranked.iter().take(3) {
            details.push(format!("  {}: {:.6}", name, score));
        }

        let max_resource_score = resources_ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
        let avg_score: f64 = scores.values().sum::<f64>() / n as f64;

        AnalysisResult {
            use_case: "High Value Targets Detection".to_string(),
            algorithm: "PageRank".to_string(),
            summary: format!(
                "PageRank analysis (d={}, {} iters): {} resources, {} roles, {} policies ranked. Max resource score: {:.6}, Avg: {:.6}",
                damping,
                iterations,
                resources_ranked.len(),
                roles_ranked.len(),
                policies_ranked.len(),
                max_resource_score,
                avg_score
            ),
            details,
        }
    }

    /// Minimum Spanning Tree using petgraph's built-in algorithm.
    ///
    /// Uses petgraph::algo::min_spanning_tree which implements Kruskal's algorithm.
    /// The directed graph is treated as undirected for MST computation.
    ///
    /// Returns edges in the MST and identifies redundant edges that could be removed.
    pub fn minimum_spanning_tree(
        graph: &DiGraph<String, f64>,
        node_type_map: &HashMap<String, NodeType>,
    ) -> AnalysisResult {
        // 1. Count unique undirected edges in original graph
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for edge in graph.edge_references() {
            let src = edge.source();
            let dst = edge.target();
            // Canonical ordering to treat as undirected
            let (a, b) = if src.index() < dst.index() {
                (src.index(), dst.index())
            } else {
                (dst.index(), src.index())
            };
            seen_edges.insert((a, b));
        }
        let total_edges = seen_edges.len();

        // 2. Use petgraph's built-in min_spanning_tree (Kruskal's algorithm)
        // min_spanning_tree treats the graph as undirected and returns an iterator
        let mst_graph: UnGraph<String, f64> = UnGraph::from_elements(min_spanning_tree(graph));

        // 3. Collect MST edges with metadata
        let mut mst_edges: Vec<(String, String, f64, String, String)> = Vec::new();
        let mut total_weight = 0.0;

        for edge in mst_graph.edge_references() {
            let weight = *edge.weight();
            let src_idx = edge.source();
            let dst_idx = edge.target();

            let src_name = mst_graph[src_idx].clone();
            let dst_name = mst_graph[dst_idx].clone();

            let src_type = node_type_map
                .get(&src_name)
                .map(|t| t.as_str())
                .unwrap_or("unknown")
                .to_string();
            let dst_type = node_type_map
                .get(&dst_name)
                .map(|t| t.as_str())
                .unwrap_or("unknown")
                .to_string();

            mst_edges.push((src_name, dst_name, weight, src_type, dst_type));
            total_weight += weight;
        }

        // Sort edges by weight for consistent output
        mst_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Generate summary and details
        let redundant_edges = total_edges.saturating_sub(mst_edges.len());

        let summary = format!(
            "MST: {} edges (total weight: {:.2}), {} redundant edges could be removed",
            mst_edges.len(),
            total_weight,
            redundant_edges
        );

        let mut details = Vec::new();
        details.push(format!("Total MST weight: {:.2}", total_weight));
        details.push(format!("MST edges: {}", mst_edges.len()));
        details.push(format!("Redundant edges: {}", redundant_edges));
        details.push(format!("Original unique edges: {}", total_edges));
        details.push(String::new()); // separator

        details.push("MST edges (sorted by weight):".to_string());
        for (src, dst, weight, src_type, dst_type) in &mst_edges {
            details.push(format!(
                "  {} ({}) -> {} ({}): {:.2}",
                src, src_type, dst, dst_type, weight
            ));
        }

        AnalysisResult {
            use_case: "Minimum Spanning Tree".to_string(),
            algorithm: "Kruskal's Algorithm (petgraph)".to_string(),
            summary,
            details,
        }
    }
}

// ============================================================================
// motlie_db Implementations
// ============================================================================

mod motlie_impl {
    use super::*;
    use motlie_db::query::IncomingEdges;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// Wrapper for tracking cumulative query time for all motlie_db read operations
    pub struct QueryTimer {
        total_ns: AtomicU64,
    }

    impl QueryTimer {
        pub fn new() -> Self {
            Self {
                total_ns: AtomicU64::new(0),
            }
        }

        /// Add elapsed time (in nanoseconds) to the total
        fn add_elapsed(&self, elapsed_ns: u64) {
            self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        }

        /// Run OutgoingEdges query with timing
        pub async fn query_outgoing_edges(
            &self,
            node: Id,
            reader: &motlie_db::graph::reader::Reader,
            timeout: Duration,
        ) -> Result<Vec<(Option<f64>, Id, Id, String, u32)>> {
            let start = std::time::Instant::now();
            let edges = OutgoingEdges::new(node, None)
                .run(reader, timeout)
                .await?;
            self.add_elapsed(start.elapsed().as_nanos() as u64);
            Ok(edges)
        }

        /// Run IncomingEdges query with timing
        pub async fn query_incoming_edges(
            &self,
            node: Id,
            reader: &motlie_db::graph::reader::Reader,
            timeout: Duration,
        ) -> Result<Vec<(Option<f64>, Id, Id, String, u32)>> {
            let start = std::time::Instant::now();
            let edges = IncomingEdges::new(node, None)
                .run(reader, timeout)
                .await?;
            self.add_elapsed(start.elapsed().as_nanos() as u64);
            Ok(edges)
        }

        /// Get total query time in milliseconds
        pub fn total_ms(&self) -> f64 {
            self.total_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
        }

        /// Reset the timer
        pub fn reset(&self) {
            self.total_ns.store(0, Ordering::Relaxed);
        }

        /// Record elapsed time from a Duration
        pub fn record_elapsed(&self, elapsed: std::time::Duration) {
            self.add_elapsed(elapsed.as_nanos() as u64);
        }

        /// Scan all nodes with timing. Returns (nodes: Vec<(Id, String)>, count).
        /// Uses the Visitable trait's accept method to iterate all nodes.
        pub fn scan_all_nodes(
            &self,
            storage: &motlie_db::graph::Storage,
        ) -> Result<Vec<(Id, String)>> {
            use motlie_db::graph::scan::{AllNodes, Visitable};
            let start = std::time::Instant::now();

            let mut nodes = Vec::new();
            let scan = AllNodes {
                last: None,
                limit: usize::MAX,
                reverse: false,
                reference_ts_millis: None,
            };
            scan.accept(storage, &mut |record: &motlie_db::graph::scan::NodeRecord| {
                nodes.push((record.id, record.name.clone()));
                true // continue scanning
            })?;

            self.add_elapsed(start.elapsed().as_nanos() as u64);
            Ok(nodes)
        }

        /// Scan all edges with timing. Returns edges: Vec<(src_id, dst_id, weight, edge_name)>.
        /// Uses the Visitable trait's accept method to iterate all edges.
        pub fn scan_all_edges(
            &self,
            storage: &motlie_db::graph::Storage,
        ) -> Result<Vec<(Id, Id, Option<f64>, String)>> {
            use motlie_db::graph::scan::{AllEdges, Visitable};
            let start = std::time::Instant::now();

            let mut edges = Vec::new();
            let scan = AllEdges {
                last: None,
                limit: usize::MAX,
                reverse: false,
                reference_ts_millis: None,
            };
            scan.accept(storage, &mut |record: &motlie_db::graph::scan::EdgeRecord| {
                // SrcId and DstId are type aliases for Id, so they're directly usable
                edges.push((
                    record.src_id,
                    record.dst_id,
                    record.weight,
                    record.name.clone(),
                ));
                true // continue scanning
            })?;

            self.add_elapsed(start.elapsed().as_nanos() as u64);
            Ok(edges)
        }

        /// Load a petgraph DiGraph from motlie_db storage, tracking disk read time.
        /// Returns (graph, id_to_idx, idx_to_id, name_to_id).
        pub fn load_petgraph_from_storage(
            &self,
            storage: &motlie_db::graph::Storage,
        ) -> Result<(
            DiGraph<String, f64>,
            HashMap<Id, NodeIndex>,
            HashMap<NodeIndex, Id>,
            HashMap<String, Id>,
        )> {
            // Scan all nodes and edges with timing
            let nodes = self.scan_all_nodes(storage)?;
            let edges = self.scan_all_edges(storage)?;

            // Build the petgraph
            let mut graph = DiGraph::new();
            let mut id_to_idx = HashMap::new();
            let mut idx_to_id = HashMap::new();
            let mut name_to_id = HashMap::new();

            // Add nodes
            for (id, name) in nodes {
                let idx = graph.add_node(name.clone());
                id_to_idx.insert(id, idx);
                idx_to_id.insert(idx, id);
                name_to_id.insert(name, id);
            }

            // Add edges
            for (src_id, dst_id, weight, _name) in edges {
                if let (Some(&src_idx), Some(&dst_idx)) = (id_to_idx.get(&src_id), id_to_idx.get(&dst_id)) {
                    graph.add_edge(src_idx, dst_idx, weight.unwrap_or(1.0));
                }
            }

            Ok((graph, id_to_idx, idx_to_id, name_to_id))
        }
    }

    /// Reachability: Can source reach target?
    pub async fn reachability(
        source: Id,
        target: Id,
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut path = Vec::new();

        queue.push_back((source, vec![source]));
        visited.insert(source);

        let reachable = loop {
            if let Some((current, current_path)) = queue.pop_front() {
                if current == target {
                    path = current_path;
                    break true;
                }

                let edges = timer.query_outgoing_edges(current, reader, timeout).await?;

                for (_weight, _src, dst, _name, _version) in edges {
                    if !visited.contains(&dst) {
                        visited.insert(dst);
                        let mut new_path = current_path.clone();
                        new_path.push(dst);
                        queue.push_back((dst, new_path));
                    }
                }
            } else {
                break false;
            }
        };

        let path_str: Vec<String> = path
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        Ok(AnalysisResult {
            use_case: "Reachability".to_string(),
            algorithm: "BFS".to_string(),
            summary: format!(
                "Reachable: {}, Path length: {}",
                reachable,
                if reachable { path.len() } else { 0 }
            ),
            details: if reachable {
                vec![format!("Path: {}", path_str.join(" -> "))]
            } else {
                vec!["No path found".to_string()]
            },
        })
    }

    /// Blast Radius: All reachable nodes from source with depth
    pub async fn blast_radius(
        source: Id,
        max_depth: usize,
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let mut visited: HashMap<Id, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut resources_by_depth: HashMap<usize, Vec<String>> = HashMap::new();
        let mut all_nodes_by_depth: HashMap<usize, Vec<String>> = HashMap::new();

        queue.push_back((source, 0));
        visited.insert(source, 0);

        while let Some((current, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            if let Some(name) = id_to_name.get(&current) {
                // Track ALL visited nodes for visualization
                all_nodes_by_depth
                    .entry(depth)
                    .or_insert_with(Vec::new)
                    .push(name.clone());

                // Track resources separately for summary
                if name.starts_with("instance-")
                    || name.starts_with("database-")
                    || name.starts_with("disk-")
                    || name.starts_with("vpc-")
                {
                    resources_by_depth
                        .entry(depth)
                        .or_insert_with(Vec::new)
                        .push(name.clone());
                }
            }

            let edges = timer.query_outgoing_edges(current, reader, timeout).await?;

            for (_weight, _src, dst, _name, _version) in edges {
                if !visited.contains_key(&dst) {
                    visited.insert(dst, depth + 1);
                    queue.push_back((dst, depth + 1));
                }
            }
        }

        let total_resources: usize = resources_by_depth.values().map(|v| v.len()).sum();
        let total_nodes: usize = all_nodes_by_depth.values().map(|v| v.len()).sum();
        let mut details = Vec::new();

        // First add summary by depth (resources only)
        for depth in 0..=max_depth {
            if let Some(resources) = resources_by_depth.get(&depth) {
                details.push(format!("Depth {}: {} resources", depth, resources.len()));
            }
        }

        // Then add ALL nodes with their depth (for visualization path)
        // Format: "nodename (depth N)" for resources, "nodename (path depth N)" for intermediate nodes
        for depth in 0..=max_depth {
            if let Some(nodes) = all_nodes_by_depth.get(&depth) {
                for node in nodes {
                    let is_resource = node.starts_with("instance-")
                        || node.starts_with("database-")
                        || node.starts_with("disk-")
                        || node.starts_with("vpc-");
                    if is_resource {
                        details.push(format!("{} (depth {})", node, depth));
                    } else {
                        details.push(format!("{} (path depth {})", node, depth));
                    }
                }
            }
        }

        Ok(AnalysisResult {
            use_case: "Blast Radius".to_string(),
            algorithm: "BFS with depth".to_string(),
            summary: format!(
                "Total resources at risk: {}, Total nodes in blast radius: {}, Max depth: {}",
                total_resources, total_nodes, max_depth
            ),
            details,
        })
    }

    /// Least Resistance: Dijkstra to find shortest weighted path
    pub async fn least_resistance(
        source: Id,
        sensitive_targets: &[Id],
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Clone)]
        struct State {
            cost: f64,
            node: Id,
        }

        impl Eq for State {}
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let sensitive_set: HashSet<_> = sensitive_targets.iter().copied().collect();
        let mut distances: HashMap<Id, f64> = HashMap::new();
        let mut predecessors: HashMap<Id, Id> = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(source, 0.0);
        heap.push(State {
            cost: 0.0,
            node: source,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if let Some(&best_cost) = distances.get(&node) {
                if cost > best_cost {
                    continue;
                }
            }

            let edges = timer.query_outgoing_edges(node, reader, timeout).await?;

            for (weight_opt, _src, dst, _name, _version) in edges {
                let weight = weight_opt.unwrap_or(1.0);
                let next_cost = cost + weight;

                if next_cost < *distances.get(&dst).unwrap_or(&f64::INFINITY) {
                    distances.insert(dst, next_cost);
                    predecessors.insert(dst, node);
                    heap.push(State {
                        cost: next_cost,
                        node: dst,
                    });
                }
            }
        }

        // Find best target among sensitive resources
        let mut best_target: Option<(Id, f64)> = None;
        for &target in sensitive_targets {
            if let Some(&dist) = distances.get(&target) {
                if best_target.is_none() || dist < best_target.unwrap().1 {
                    best_target = Some((target, dist));
                }
            }
        }

        match best_target {
            Some((target, cost)) => {
                // Reconstruct path from source to target
                let mut path = vec![target];
                let mut current = target;
                while let Some(&pred) = predecessors.get(&current) {
                    path.push(pred);
                    current = pred;
                    if current == source {
                        break;
                    }
                }
                path.reverse();

                let path_str: Vec<String> = path.iter()
                    .filter_map(|id| id_to_name.get(id).cloned())
                    .collect();

                Ok(AnalysisResult {
                    use_case: "Least Resistance".to_string(),
                    algorithm: "Dijkstra".to_string(),
                    summary: format!(
                        "Easiest path cost: {:.2}, Target: {}",
                        cost,
                        id_to_name.get(&target).unwrap_or(&"unknown".to_string())
                    ),
                    details: vec![
                        format!("Path: {}", path_str.join(" -> ")),
                        format!("Total weight: {:.2}", cost),
                    ],
                })
            },
            None => Ok(AnalysisResult {
                use_case: "Least Resistance".to_string(),
                algorithm: "Dijkstra".to_string(),
                summary: "No path to sensitive resources".to_string(),
                details: vec!["No reachable sensitive resources from source".to_string()],
            }),
        }
    }

    /// Privilege Clustering: Group users by access patterns
    pub async fn privilege_clustering(
        user_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        // For each user, compute their "access fingerprint"
        let mut user_fingerprints: HashMap<Id, HashSet<Id>> = HashMap::new();

        for &user in user_ids {
            let mut reachable = HashSet::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(user);
            visited.insert(user);

            while let Some(current) = queue.pop_front() {
                if let Some(name) = id_to_name.get(&current) {
                    if name.starts_with("instance-")
                        || name.starts_with("database-")
                        || name.starts_with("disk-")
                        || name.starts_with("vpc-")
                    {
                        reachable.insert(current);
                    }
                }

                let edges = timer.query_outgoing_edges(current, reader, timeout).await?;

                for (_weight, _src, dst, _name, _version) in edges {
                    if !visited.contains(&dst) {
                        visited.insert(dst);
                        queue.push_back(dst);
                    }
                }
            }
            user_fingerprints.insert(user, reachable);
        }

        // Cluster users by Jaccard similarity
        let mut clusters: Vec<HashSet<Id>> = Vec::new();

        for &user in user_ids {
            let user_fp = &user_fingerprints[&user];
            let mut found_cluster = false;

            for cluster in &mut clusters {
                if let Some(&representative) = cluster.iter().next() {
                    let rep_fp = &user_fingerprints[&representative];
                    let intersection = user_fp.intersection(rep_fp).count();
                    let union = user_fp.union(rep_fp).count();
                    let jaccard = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    if jaccard > 0.5 {
                        cluster.insert(user);
                        found_cluster = true;
                        break;
                    }
                }
            }

            if !found_cluster {
                let mut new_cluster = HashSet::new();
                new_cluster.insert(user);
                clusters.push(new_cluster);
            }
        }

        // Sort clusters by size (largest first)
        clusters.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut details = Vec::new();
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
        details.push(format!(
            "Cluster sizes (top 5): {:?}",
            cluster_sizes.iter().take(5).collect::<Vec<_>>()
        ));

        // Output cluster members (format: "Cluster N: user-0001, user-0002, ...")
        for (i, cluster) in clusters.iter().take(10).enumerate() {
            let mut members: Vec<String> = cluster
                .iter()
                .filter_map(|&id| id_to_name.get(&id).cloned())
                .collect();
            members.sort();
            details.push(format!("Cluster {}: {}", i + 1, members.join(", ")));
        }

        Ok(AnalysisResult {
            use_case: "Privilege Clustering".to_string(),
            algorithm: "Jaccard Similarity Clustering".to_string(),
            summary: format!(
                "Found {} clusters from {} users",
                clusters.len(),
                user_ids.len()
            ),
            details,
        })
    }

    /// Over-Privileged: Find users with access to many sensitive resources
    pub async fn over_privileged(
        user_ids: &[Id],
        sensitive_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        threshold: usize,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let sensitive_set: HashSet<_> = sensitive_ids.iter().copied().collect();
        let mut over_privileged_users = Vec::new();

        for &user in user_ids {
            let mut count = 0;
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(user);
            visited.insert(user);

            while let Some(current) = queue.pop_front() {
                if sensitive_set.contains(&current) {
                    count += 1;
                }

                let edges = timer.query_outgoing_edges(current, reader, timeout).await?;

                for (_weight, _src, dst, _name, _version) in edges {
                    if !visited.contains(&dst) {
                        visited.insert(dst);
                        queue.push_back(dst);
                    }
                }
            }

            if count >= threshold {
                if let Some(name) = id_to_name.get(&user) {
                    over_privileged_users.push((name.clone(), count));
                }
            }
        }

        over_privileged_users.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(AnalysisResult {
            use_case: "Over-Privileged Detection".to_string(),
            algorithm: "BFS + counting".to_string(),
            summary: format!(
                "Found {} users with access to {} or more sensitive resources",
                over_privileged_users.len(),
                threshold
            ),
            details: over_privileged_users
                .iter()
                .take(10)
                .map(|(name, count)| format!("{}: {} sensitive resources", name, count))
                .collect(),
        })
    }

    /// Cross-Region Access: Find paths that cross region boundaries
    pub async fn cross_region_access(
        user_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        id_to_region: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let mut cross_region_paths = Vec::new();

        for &user in user_ids.iter().take(50) {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            let user_name = id_to_name.get(&user).cloned().unwrap_or_default();
            queue.push_back((user, None::<String>, vec![user_name]));
            visited.insert(user);

            while let Some((current, current_region, path)) = queue.pop_front() {
                if path.len() > 10 {
                    continue;
                }

                let edges = timer.query_outgoing_edges(current, reader, timeout).await?;

                for (_weight, _src, dst, _name, _version) in edges {
                    if !visited.contains(&dst) {
                        visited.insert(dst);

                        let neighbor_region = id_to_region.get(&dst).cloned();
                        let neighbor_name = id_to_name.get(&dst).cloned().unwrap_or_default();
                        let mut new_path = path.clone();
                        new_path.push(neighbor_name);

                        if let (Some(ref curr_reg), Some(ref next_reg)) =
                            (&current_region, &neighbor_region)
                        {
                            if curr_reg != next_reg {
                                cross_region_paths
                                    .push((new_path.clone(), curr_reg.clone(), next_reg.clone()));
                            }
                        }

                        queue.push_back((dst, neighbor_region, new_path));
                    }
                }
            }
        }

        Ok(AnalysisResult {
            use_case: "Cross-Region Access".to_string(),
            algorithm: "Filtered BFS".to_string(),
            summary: format!("Found {} cross-region access paths", cross_region_paths.len()),
            details: cross_region_paths
                .iter()
                .take(5)
                .map(|(path, from, to)| format!("{} -> {} (path len: {})", from, to, path.len()))
                .collect(),
        })
    }

    /// Unused Roles: Find isolated roles using Kosaraju's SCC algorithm (motlie_db version)
    /// Implements Kosaraju's algorithm manually since we need to traverse via queries
    pub async fn unused_roles(
        role_ids: &[Id],
        workload_ids: &[Id],
        all_node_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        // Step 1: Build adjacency list and reverse adjacency list by traversing all edges
        let mut adj: HashMap<Id, Vec<Id>> = HashMap::new();
        let mut rev_adj: HashMap<Id, Vec<Id>> = HashMap::new();

        for &node in all_node_ids {
            adj.entry(node).or_insert_with(Vec::new);
            rev_adj.entry(node).or_insert_with(Vec::new);
        }

        // Collect all edges
        for &node in all_node_ids {
            let edges = timer.query_outgoing_edges(node, reader, timeout).await?;
            for (_weight, src, dst, _name, _version) in edges {
                adj.entry(src).or_insert_with(Vec::new).push(dst);
                rev_adj.entry(dst).or_insert_with(Vec::new).push(src);
            }
        }

        // Step 2: First DFS pass - compute finish order
        let mut visited: HashSet<Id> = HashSet::new();
        let mut finish_order: Vec<Id> = Vec::new();

        fn dfs_first(
            node: Id,
            adj: &HashMap<Id, Vec<Id>>,
            visited: &mut HashSet<Id>,
            finish_order: &mut Vec<Id>,
        ) {
            visited.insert(node);
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        dfs_first(neighbor, adj, visited, finish_order);
                    }
                }
            }
            finish_order.push(node);
        }

        for &node in all_node_ids {
            if !visited.contains(&node) {
                dfs_first(node, &adj, &mut visited, &mut finish_order);
            }
        }

        // Step 3: Second DFS pass - find SCCs in reverse finish order
        let mut visited2: HashSet<Id> = HashSet::new();
        let mut sccs: Vec<Vec<Id>> = Vec::new();

        fn dfs_second(
            node: Id,
            rev_adj: &HashMap<Id, Vec<Id>>,
            visited: &mut HashSet<Id>,
            component: &mut Vec<Id>,
        ) {
            visited.insert(node);
            component.push(node);
            if let Some(neighbors) = rev_adj.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        dfs_second(neighbor, rev_adj, visited, component);
                    }
                }
            }
        }

        for &node in finish_order.iter().rev() {
            if !visited2.contains(&node) {
                let mut component = Vec::new();
                dfs_second(node, &rev_adj, &mut visited2, &mut component);
                sccs.push(component);
            }
        }

        // Build node to SCC mapping
        let mut node_to_scc: HashMap<Id, usize> = HashMap::new();
        let mut scc_sizes: Vec<usize> = Vec::new();
        for (scc_idx, scc) in sccs.iter().enumerate() {
            scc_sizes.push(scc.len());
            for &node in scc {
                node_to_scc.insert(node, scc_idx);
            }
        }

        // Find largest SCC
        let main_scc_idx = scc_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &size)| size)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let main_scc_size = scc_sizes.get(main_scc_idx).copied().unwrap_or(0);

        // Find roles assumed by workloads
        let mut assumed_roles: HashSet<Id> = HashSet::new();
        for &workload in workload_ids {
            let edges = OutgoingEdges::new(workload, None)
                .run(reader, timeout)
                .await?;
            for (_weight, _src, dst, _name, _version) in edges {
                if let Some(name) = id_to_name.get(&dst) {
                    if name.starts_with("role-") {
                        assumed_roles.insert(dst);
                    }
                }
            }
        }

        // Categorize roles
        let mut isolated_roles: Vec<(String, &str)> = Vec::new();
        let mut small_scc_roles = 0;
        let mut unassumed_roles = 0;

        for &role in role_ids {
            let role_name = id_to_name.get(&role).cloned().unwrap_or_default();
            let scc_idx = node_to_scc.get(&role).copied().unwrap_or(0);
            let scc_size = scc_sizes.get(scc_idx).copied().unwrap_or(1);

            let is_in_small_scc = scc_idx != main_scc_idx && scc_size < main_scc_size / 10;
            let is_unassumed = !assumed_roles.contains(&role);

            if is_in_small_scc {
                small_scc_roles += 1;
                isolated_roles.push((role_name.clone(), "isolated SCC"));
            } else if is_unassumed {
                unassumed_roles += 1;
                isolated_roles.push((role_name, "no workloads assume"));
            }
        }

        isolated_roles.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(AnalysisResult {
            use_case: "Unused Roles Detection".to_string(),
            algorithm: "Kosaraju SCC".to_string(),
            summary: format!(
                "Found {} unused roles: {} in isolated SCCs, {} unassumed (total roles: {}, SCCs: {}, main SCC size: {})",
                isolated_roles.len(),
                small_scc_roles,
                unassumed_roles,
                role_ids.len(),
                sccs.len(),
                main_scc_size
            ),
            details: isolated_roles
                .iter()
                .take(15)
                .map(|(name, reason)| format!("{}: {}", name, reason))
                .collect(),
        })
    }

    /// Privilege Hubs: Find over-privileged entities via manual degree calculation (motlie_db)
    pub async fn privilege_hubs(
        all_node_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        threshold_percentile: f64,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        // Manual degree calculation by traversing all edges
        let mut in_degrees: HashMap<Id, usize> = HashMap::new();
        let mut out_degrees: HashMap<Id, usize> = HashMap::new();

        // Initialize all nodes with 0 degree
        for &node in all_node_ids {
            in_degrees.insert(node, 0);
            out_degrees.insert(node, 0);
        }

        // Count degrees by querying edges for each node
        for &node in all_node_ids {
            let edges = timer.query_outgoing_edges(node, reader, timeout).await?;
            for (_weight, src, dst, _name, _version) in edges {
                *out_degrees.entry(src).or_insert(0) += 1;
                *in_degrees.entry(dst).or_insert(0) += 1;
            }
        }

        // Calculate total degree for each node
        let mut node_degrees: Vec<(Id, usize, usize, usize)> = all_node_ids
            .iter()
            .map(|&node| {
                let in_deg = in_degrees.get(&node).copied().unwrap_or(0);
                let out_deg = out_degrees.get(&node).copied().unwrap_or(0);
                (node, in_deg, out_deg, in_deg + out_deg)
            })
            .collect();

        // Sort by total degree descending
        node_degrees.sort_by(|a, b| b.3.cmp(&a.3));

        // Find threshold
        let threshold_idx = ((1.0 - threshold_percentile) * node_degrees.len() as f64) as usize;
        let threshold_degree = if threshold_idx < node_degrees.len() {
            node_degrees[threshold_idx].3
        } else {
            0
        };

        // Classify hubs by type
        let mut hubs_by_type: HashMap<&str, Vec<(String, usize, usize)>> = HashMap::new();
        let mut total_hubs = 0;

        for (node, in_deg, out_deg, total_deg) in &node_degrees {
            if *total_deg >= threshold_degree && *total_deg > 0 {
                let name = id_to_name.get(node).cloned().unwrap_or_default();
                let node_type = if name.starts_with("user-") {
                    "Users"
                } else if name.starts_with("group-") {
                    "Groups"
                } else if name.starts_with("policy-") {
                    "Policies"
                } else if name.starts_with("role-") {
                    "Roles"
                } else if name.starts_with("workload-") {
                    "Workloads"
                } else if name.starts_with("instance-") || name.starts_with("database-")
                    || name.starts_with("disk-") || name.starts_with("vpc-") {
                    "Resources"
                } else {
                    "Other"
                };

                hubs_by_type
                    .entry(node_type)
                    .or_insert_with(Vec::new)
                    .push((name, *in_deg, *out_deg));
                total_hubs += 1;
            }
        }

        // Build details
        let mut details = Vec::new();
        let type_order = ["Policies", "Groups", "Roles", "Users", "Workloads", "Resources", "Other"];
        for node_type in type_order {
            if let Some(hubs) = hubs_by_type.get(node_type) {
                details.push(format!("{}: {} hubs", node_type, hubs.len()));
                for (name, in_deg, out_deg) in hubs.iter().take(3) {
                    details.push(format!("  {} (in:{}, out:{})", name, in_deg, out_deg));
                }
            }
        }

        let avg_degree: f64 = if !node_degrees.is_empty() {
            node_degrees.iter().map(|(_, _, _, d)| *d as f64).sum::<f64>() / node_degrees.len() as f64
        } else {
            0.0
        };
        let max_degree = node_degrees.first().map(|(_, _, _, d)| *d).unwrap_or(0);

        Ok(AnalysisResult {
            use_case: "Privilege Hubs Detection".to_string(),
            algorithm: "Manual Degree Calculation".to_string(),
            summary: format!(
                "Found {} privilege hubs (top {:.0}%, degree >= {}). Avg degree: {:.1}, Max: {}",
                total_hubs,
                (1.0 - threshold_percentile) * 100.0,
                threshold_degree,
                avg_degree,
                max_degree
            ),
            details,
        })
    }

    /// Minimal Privilege Verification: Check if permission paths are minimal (motlie_db)
    pub async fn minimal_privilege(
        user_ids: &[Id],
        sensitive_ids: &[Id],
        all_node_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        // Build adjacency list with weights for Dijkstra
        let mut adj: HashMap<Id, Vec<(Id, f64)>> = HashMap::new();
        for &node in all_node_ids {
            adj.entry(node).or_insert_with(Vec::new);
        }
        for &node in all_node_ids {
            let edges = timer.query_outgoing_edges(node, reader, timeout).await?;
            for (weight_opt, src, dst, _name, _version) in edges {
                let weight = weight_opt.unwrap_or(1.0);
                adj.entry(src).or_insert_with(Vec::new).push((dst, weight));
            }
        }

        let mut non_minimal_paths = Vec::new();
        let mut total_paths_checked = 0;
        let mut minimal_paths = 0;

        let sensitive_set: HashSet<_> = sensitive_ids.iter().copied().collect();

        // Check sample of users
        for &user in user_ids.iter().take(20) {
            // Dijkstra from this user
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            #[derive(Clone)]
            struct State { cost: f64, node: Id }
            impl Eq for State {}
            impl PartialEq for State {
                fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
            }
            impl Ord for State {
                fn cmp(&self, other: &Self) -> Ordering {
                    other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
                }
            }
            impl PartialOrd for State {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
            }

            let mut distances: HashMap<Id, f64> = HashMap::new();
            let mut heap = BinaryHeap::new();
            distances.insert(user, 0.0);
            heap.push(State { cost: 0.0, node: user });

            while let Some(State { cost, node }) = heap.pop() {
                if cost > *distances.get(&node).unwrap_or(&f64::INFINITY) {
                    continue;
                }
                if let Some(neighbors) = adj.get(&node) {
                    for &(next, weight) in neighbors {
                        let next_cost = cost + weight;
                        if next_cost < *distances.get(&next).unwrap_or(&f64::INFINITY) {
                            distances.insert(next, next_cost);
                            heap.push(State { cost: next_cost, node: next });
                        }
                    }
                }
            }

            // Check paths to sensitive resources
            for &target in sensitive_ids.iter().take(10) {
                if let Some(&shortest_cost) = distances.get(&target) {
                    total_paths_checked += 1;

                    // BFS to find first path
                    let mut visited = HashSet::new();
                    let mut queue = VecDeque::new();
                    queue.push_back((user, 0.0_f64, vec![user]));
                    visited.insert(user);

                    let mut actual_path_cost: Option<f64> = None;
                    let mut actual_path_len = 0;

                    while let Some((current, cost, path)) = queue.pop_front() {
                        if current == target {
                            actual_path_cost = Some(cost);
                            actual_path_len = path.len();
                            break;
                        }
                        if let Some(neighbors) = adj.get(&current) {
                            for &(next, weight) in neighbors {
                                if !visited.contains(&next) {
                                    visited.insert(next);
                                    let mut new_path = path.clone();
                                    new_path.push(next);
                                    queue.push_back((next, cost + weight, new_path));
                                }
                            }
                        }
                    }

                    if let Some(actual_cost) = actual_path_cost {
                        let cost_diff = actual_cost - shortest_cost;
                        if cost_diff > 0.01 {
                            let user_name = id_to_name.get(&user).cloned().unwrap_or_default();
                            let target_name = id_to_name.get(&target).cloned().unwrap_or_default();
                            non_minimal_paths.push((user_name, target_name, actual_cost, shortest_cost, actual_path_len));
                        } else {
                            minimal_paths += 1;
                        }
                    }
                }
            }
        }

        non_minimal_paths.sort_by(|a, b| {
            let diff_a = a.2 - a.3;
            let diff_b = b.2 - b.3;
            diff_b.partial_cmp(&diff_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let details: Vec<String> = non_minimal_paths
            .iter()
            .take(10)
            .map(|(user, target, actual, optimal, hops)| {
                format!(
                    "{} -> {}: actual={:.1}, optimal={:.1}, excess={:.1}, hops={}",
                    user, target, actual, optimal, actual - optimal, hops
                )
            })
            .collect();

        Ok(AnalysisResult {
            use_case: "Minimal Privilege Verification".to_string(),
            algorithm: "Dijkstra Comparison".to_string(),
            summary: format!(
                "Checked {} paths: {} minimal, {} non-minimal ({:.1}% optimal)",
                total_paths_checked,
                minimal_paths,
                non_minimal_paths.len(),
                if total_paths_checked > 0 {
                    minimal_paths as f64 / total_paths_checked as f64 * 100.0
                } else {
                    100.0
                }
            ),
            details,
        })
    }

    /// Accessible Resources: List all resources accessible via DFS (motlie_db)
    pub async fn accessible_resources(
        user_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let mut all_accessible: HashMap<String, HashSet<String>> = HashMap::new();
        let mut resource_access_count: HashMap<String, usize> = HashMap::new();

        for &user in user_ids.iter().take(50) {
            let user_name = id_to_name.get(&user).cloned().unwrap_or_default();
            let mut accessible = HashSet::new();
            let mut stack = vec![user];
            let mut visited = HashSet::new();

            // DFS traversal
            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);

                if let Some(name) = id_to_name.get(&current) {
                    if name.starts_with("instance-") || name.starts_with("database-")
                        || name.starts_with("disk-") || name.starts_with("vpc-")
                    {
                        accessible.insert(name.clone());
                        *resource_access_count.entry(name.clone()).or_insert(0) += 1;
                    }
                }

                let edges = timer.query_outgoing_edges(current, reader, timeout).await?;
                for (_weight, _src, dst, _name, _version) in edges {
                    if !visited.contains(&dst) {
                        stack.push(dst);
                    }
                }
            }

            all_accessible.insert(user_name, accessible);
        }

        let total_users = all_accessible.len();
        let total_resources_accessed: usize = all_accessible.values().map(|r| r.len()).sum();
        let avg_resources = if total_users > 0 {
            total_resources_accessed as f64 / total_users as f64
        } else {
            0.0
        };

        let mut most_accessed: Vec<_> = resource_access_count.iter().collect();
        most_accessed.sort_by(|a, b| b.1.cmp(a.1));

        let mut users_by_access: Vec<_> = all_accessible.iter()
            .map(|(u, r)| (u.clone(), r.len()))
            .collect();
        users_by_access.sort_by(|a, b| b.1.cmp(&a.1));

        let mut details = Vec::new();
        details.push("Most accessed resources:".to_string());
        for (resource, count) in most_accessed.iter().take(5) {
            details.push(format!("  {}: {} users", resource, count));
        }
        details.push("Users with most access:".to_string());
        for (user, count) in users_by_access.iter().take(5) {
            details.push(format!("  {}: {} resources", user, count));
        }

        Ok(AnalysisResult {
            use_case: "Accessible Resources Listing".to_string(),
            algorithm: "DFS Traversal".to_string(),
            summary: format!(
                "Analyzed {} users: avg {:.1} resources/user, {} unique resources accessed",
                total_users,
                avg_resources,
                resource_access_count.len()
            ),
            details,
        })
    }

    /// High Value Targets: PageRank for identifying high-value targets (motlie_db)
    pub async fn high_value_targets(
        all_node_ids: &[Id],
        id_to_name: &HashMap<Id, String>,
        damping: f64,
        iterations: usize,
        reader: &motlie_db::graph::reader::Reader,
        timeout: Duration,
        timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        let n = all_node_ids.len();
        if n == 0 {
            return Ok(AnalysisResult {
                use_case: "High Value Targets Detection".to_string(),
                algorithm: "PageRank".to_string(),
                summary: "Empty graph".to_string(),
                details: vec![],
            });
        }

        // Build adjacency structures
        let mut in_edges: HashMap<Id, Vec<Id>> = HashMap::new();
        let mut out_degrees: HashMap<Id, usize> = HashMap::new();

        for &node in all_node_ids {
            in_edges.insert(node, Vec::new());
            out_degrees.insert(node, 0);
        }

        for &node in all_node_ids {
            let edges = timer.query_outgoing_edges(node, reader, timeout).await?;
            for (_weight, src, dst, _name, _version) in edges {
                in_edges.entry(dst).or_insert_with(Vec::new).push(src);
                *out_degrees.entry(src).or_insert(0) += 1;
            }
        }

        // Initialize scores
        let mut scores: HashMap<Id, f64> = HashMap::new();
        let initial_score = 1.0 / n as f64;
        for &node in all_node_ids {
            scores.insert(node, initial_score);
        }

        // PageRank iterations
        for _ in 0..iterations {
            let mut new_scores: HashMap<Id, f64> = HashMap::new();
            let base = (1.0 - damping) / n as f64;

            for &node in all_node_ids {
                let mut incoming_score = 0.0;
                if let Some(predecessors) = in_edges.get(&node) {
                    for &pred in predecessors {
                        let pred_out_deg = out_degrees.get(&pred).copied().unwrap_or(1).max(1);
                        incoming_score += scores.get(&pred).copied().unwrap_or(0.0) / pred_out_deg as f64;
                    }
                }
                new_scores.insert(node, base + damping * incoming_score);
            }
            scores = new_scores;
        }

        // Classify and rank
        let mut resources_ranked: Vec<(String, f64)> = Vec::new();
        let mut roles_ranked: Vec<(String, f64)> = Vec::new();
        let mut policies_ranked: Vec<(String, f64)> = Vec::new();

        for (&node, &score) in &scores {
            if let Some(name) = id_to_name.get(&node) {
                if name.starts_with("instance-") || name.starts_with("database-")
                    || name.starts_with("disk-") || name.starts_with("vpc-")
                {
                    resources_ranked.push((name.clone(), score));
                } else if name.starts_with("role-") {
                    roles_ranked.push((name.clone(), score));
                } else if name.starts_with("policy-") {
                    policies_ranked.push((name.clone(), score));
                }
            }
        }

        resources_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        roles_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        policies_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut details = Vec::new();
        details.push("High-value resources (most reachable):".to_string());
        for (name, score) in resources_ranked.iter().take(5) {
            details.push(format!("  {}: {:.6}", name, score));
        }
        details.push("High-value roles (permission bottlenecks):".to_string());
        for (name, score) in roles_ranked.iter().take(3) {
            details.push(format!("  {}: {:.6}", name, score));
        }
        details.push("High-value policies (widely granted):".to_string());
        for (name, score) in policies_ranked.iter().take(3) {
            details.push(format!("  {}: {:.6}", name, score));
        }

        let max_resource_score = resources_ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
        let avg_score: f64 = scores.values().sum::<f64>() / n as f64;

        Ok(AnalysisResult {
            use_case: "High Value Targets Detection".to_string(),
            algorithm: "PageRank".to_string(),
            summary: format!(
                "PageRank analysis (d={}, {} iters): {} resources, {} roles, {} policies ranked. Max resource score: {:.6}, Avg: {:.6}",
                damping,
                iterations,
                resources_ranked.len(),
                roles_ranked.len(),
                policies_ranked.len(),
                max_resource_score,
                avg_score
            ),
            details,
        })
    }

    /// Minimum Spanning Tree using Kruskal's algorithm (motlie_db version)
    ///
    /// Uses pre-fetched edges from the graph to compute MST.
    /// Treats edges as undirected for MST computation.
    pub async fn minimum_spanning_tree(
        all_nodes: &[(Id, String)],
        all_edges: &[(Id, Id, Option<f64>, String)], // (src, dst, weight, name) from scan_all_edges
        node_type_map: &HashMap<Id, NodeType>,
        id_to_name: &HashMap<Id, String>,
        _timer: &QueryTimer,
    ) -> Result<AnalysisResult> {
        // 1. Collect unique undirected edges with weights
        let mut edges: Vec<(f64, Id, Id, String, String)> = Vec::new();
        let mut seen_edges: HashSet<(Id, Id)> = HashSet::new();

        for (src, dst, weight_opt, _name) in all_edges {
            let weight = weight_opt.unwrap_or(1.0);

            // Canonical ordering to treat as undirected
            let (a, b) = if src < dst { (*src, *dst) } else { (*dst, *src) };

            // Skip if we've seen this undirected edge
            if seen_edges.contains(&(a, b)) {
                continue;
            }
            seen_edges.insert((a, b));

            let src_name = id_to_name.get(src).cloned().unwrap_or_default();
            let dst_name = id_to_name.get(dst).cloned().unwrap_or_default();
            edges.push((weight, *src, *dst, src_name, dst_name));
        }

        let total_edges = edges.len();

        // 2. Sort edges by weight (ascending)
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Union-Find for cycle detection
        let mut parent: HashMap<Id, Id> = HashMap::new();
        let mut rank: HashMap<Id, usize> = HashMap::new();

        for (id, _name) in all_nodes {
            parent.insert(*id, *id);
            rank.insert(*id, 0);
        }

        fn find(parent: &mut HashMap<Id, Id>, x: Id) -> Id {
            if parent[&x] != x {
                let root = find(parent, parent[&x]);
                parent.insert(x, root);
            }
            parent[&x]
        }

        fn union(
            parent: &mut HashMap<Id, Id>,
            rank: &mut HashMap<Id, usize>,
            x: Id,
            y: Id,
        ) -> bool {
            let root_x = find(parent, x);
            let root_y = find(parent, y);

            if root_x == root_y {
                return false;
            }

            let rank_x = rank[&root_x];
            let rank_y = rank[&root_y];

            if rank_x < rank_y {
                parent.insert(root_x, root_y);
            } else if rank_x > rank_y {
                parent.insert(root_y, root_x);
            } else {
                parent.insert(root_y, root_x);
                *rank.get_mut(&root_x).unwrap() += 1;
            }
            true
        }

        // 4. Build MST
        let mut mst_edges: Vec<(String, String, f64, String, String)> = Vec::new();
        let mut total_weight = 0.0;
        let target_edges = all_nodes.len().saturating_sub(1);

        for (weight, src, dst, src_name, dst_name) in edges {
            if mst_edges.len() >= target_edges {
                break;
            }

            if union(&mut parent, &mut rank, src, dst) {
                let src_type = node_type_map
                    .get(&src)
                    .map(|t| t.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let dst_type = node_type_map
                    .get(&dst)
                    .map(|t| t.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                mst_edges.push((src_name, dst_name, weight, src_type, dst_type));
                total_weight += weight;
            }
        }

        // 5. Generate results
        let redundant_edges = total_edges.saturating_sub(mst_edges.len());

        let summary = format!(
            "MST: {} edges (total weight: {:.2}), {} redundant edges could be removed",
            mst_edges.len(),
            total_weight,
            redundant_edges
        );

        let mut details = Vec::new();
        details.push(format!("Total MST weight: {:.2}", total_weight));
        details.push(format!("MST edges: {}", mst_edges.len()));
        details.push(format!("Redundant edges: {}", redundant_edges));
        details.push(format!("Original unique edges: {}", total_edges));
        details.push(String::new());

        details.push("MST edges (sorted by weight):".to_string());
        for (src, dst, weight, src_type, dst_type) in &mst_edges {
            details.push(format!(
                "  {} ({}) -> {} ({}): {:.2}",
                src, src_type, dst, dst_type, weight
            ));
        }

        Ok(AnalysisResult {
            use_case: "Minimum Spanning Tree".to_string(),
            algorithm: "Kruskal's Algorithm".to_string(),
            summary,
            details,
        })
    }
}

// ============================================================================
// Visualization Server
// ============================================================================

/// Shared state for the visualization server
#[derive(Clone)]
struct VisualizationState {
    /// Graph nodes for visualization
    nodes: Arc<RwLock<Vec<VisNode>>>,
    /// Graph edges for visualization
    edges: Arc<RwLock<Vec<VisEdge>>>,
    /// Analysis result (updated when analysis completes)
    result: Arc<RwLock<Option<VisAnalysisResult>>>,
    /// Use case being run
    use_case: Arc<RwLock<Option<String>>>,
    /// Whether analysis is complete
    is_complete: Arc<RwLock<bool>>,
    /// Whether analysis is currently running
    is_running: Arc<RwLock<bool>>,
    /// Execution metrics from the last analysis
    execution_metrics: Arc<RwLock<Option<ExecutionMetrics>>>,
    /// Graph context for running analyses
    graph_context: Arc<RwLock<Option<GraphContext>>>,
}

/// Execution metrics for display
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutionMetrics {
    execution_time_ms: f64,
    memory_usage_bytes: Option<usize>,
    num_nodes: usize,
    num_edges: usize,
    implementation: String,
    /// Time spent writing graph to RocksDB (build phase)
    disk_build_time_ms: Option<f64>,
    /// Time spent reading from RocksDB during execution (query reads for motlie_db, full graph load for reference)
    disk_read_time_ms: Option<f64>,
}

/// Lightweight edge info for visualization (source name, target name)
#[derive(Debug, Clone)]
struct EdgeInfo {
    source_name: String,
    target_name: String,
}

/// Lightweight metadata extracted from IamGraph for visualization and lookups.
/// This is kept in memory after startup; the full IamGraph is discarded.
#[derive(Debug, Clone)]
struct GraphMetadata {
    /// Node name to node type mapping
    name_to_type: HashMap<String, NodeType>,
    /// Node name to region mapping (for nodes that have regions)
    name_to_region: HashMap<String, String>,
    /// Edge info for visualization (indexed by edge index)
    edges: Vec<EdgeInfo>,
    /// Stats
    total_nodes: usize,
    total_edges: usize,
    /// Category lists (by name for lookup)
    user_names: Vec<String>,
    role_names: Vec<String>,
    workload_names: Vec<String>,
    sensitive_resource_names: Vec<String>,
}

impl GraphMetadata {
    /// Extract metadata from IamGraph
    fn from_iam_graph(iam_graph: &IamGraph) -> Self {
        let id_to_name: HashMap<Id, String> = iam_graph
            .nodes
            .iter()
            .map(|n| (n.id, n.name.clone()))
            .collect();

        let name_to_type: HashMap<String, NodeType> = iam_graph
            .nodes
            .iter()
            .map(|n| (n.name.clone(), n.node_type.clone()))
            .collect();

        let name_to_region: HashMap<String, String> = iam_graph
            .nodes
            .iter()
            .filter_map(|n| n.region.as_ref().map(|r| (n.name.clone(), r.clone())))
            .collect();

        let edges: Vec<EdgeInfo> = iam_graph
            .edges
            .iter()
            .map(|e| EdgeInfo {
                source_name: id_to_name.get(&e.source).cloned().unwrap_or_default(),
                target_name: id_to_name.get(&e.target).cloned().unwrap_or_default(),
            })
            .collect();

        let user_names: Vec<String> = iam_graph
            .users
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        let role_names: Vec<String> = iam_graph
            .roles
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        let workload_names: Vec<String> = iam_graph
            .workloads
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        let sensitive_resource_names: Vec<String> = iam_graph
            .sensitive_resources
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        Self {
            name_to_type,
            name_to_region,
            edges,
            total_nodes: iam_graph.stats.total_nodes(),
            total_edges: iam_graph.stats.total_edges(),
            user_names,
            role_names,
            workload_names,
            sensitive_resource_names,
        }
    }
}

/// Context needed to run analyses (populated once at startup)
struct GraphContext {
    /// Lightweight metadata for visualization and lookups (no heavy IamGraph)
    metadata: GraphMetadata,
    /// Path to RocksDB (graph was written here at startup)
    db_path: std::path::PathBuf,
    /// Time taken to build the graph to RocksDB at startup (ms)
    disk_build_time_ms: f64,
}

impl VisualizationState {
    fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            edges: Arc::new(RwLock::new(Vec::new())),
            result: Arc::new(RwLock::new(None)),
            use_case: Arc::new(RwLock::new(None)),
            is_complete: Arc::new(RwLock::new(false)),
            is_running: Arc::new(RwLock::new(false)),
            execution_metrics: Arc::new(RwLock::new(None)),
            graph_context: Arc::new(RwLock::new(None)),
        }
    }
}

/// Input parameters for running a use case
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UseCaseInput {
    use_case: String,
    implementation: String,
    /// Optional: source node name (for reachability, blast_radius, etc.)
    source_node: Option<String>,
    /// Optional: target node name (for reachability, least_resistance)
    target_node: Option<String>,
    /// Optional: max depth (for blast_radius)
    max_depth: Option<usize>,
    /// Optional: threshold (for over_privileged, privilege_hubs)
    threshold: Option<f64>,
}

/// Use case metadata for the UI
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UseCaseInfo {
    id: String,
    name: String,
    algorithm: String,
    description: String,
    /// Input fields required for this use case
    inputs: Vec<UseCaseInputField>,
}

/// Describes an input field for a use case
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UseCaseInputField {
    name: String,
    label: String,
    field_type: String, // "node_select", "number", "text"
    required: bool,
    default_value: Option<String>,
    node_type_filter: Option<String>, // For node_select: "User", "Database", etc.
}

/// Node for visualization JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisNode {
    id: String,
    label: String,
    node_type: String,
    region: Option<String>,
}

/// Edge for visualization JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisEdge {
    id: usize,
    from: String,
    to: String,
    edge_type: String,
    weight: f64,
}

/// Explanation for a use case with business context, algorithm details, and visualization guide
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UseCaseExplanation {
    /// Business problem description
    business_problem: String,
    /// Algorithm explanation
    algorithm_description: String,
    /// How to interpret the visualization
    visualization_guide: String,
}

/// Analysis result for visualization with overlay data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisAnalysisResult {
    use_case: String,
    algorithm: String,
    summary: String,
    details: Vec<String>,
    /// Overlay: highlighted node IDs (e.g., path nodes, affected nodes)
    highlighted_nodes: Vec<String>,
    /// Overlay: highlighted edge indices
    highlighted_edges: Vec<usize>,
    /// Overlay: node annotations (node_id -> annotation text)
    node_annotations: HashMap<String, String>,
    /// Overlay: edge annotations (edge_id -> annotation text)
    edge_annotations: HashMap<usize, String>,
    /// Overlay type for frontend rendering
    overlay_type: String,
    /// Use case explanation with business context
    explanation: Option<UseCaseExplanation>,
}

impl VisAnalysisResult {
    fn from_analysis_result(result: &AnalysisResult) -> Self {
        Self {
            use_case: result.use_case.clone(),
            algorithm: result.algorithm.clone(),
            summary: result.summary.clone(),
            details: result.details.clone(),
            highlighted_nodes: Vec::new(),
            highlighted_edges: Vec::new(),
            node_annotations: HashMap::new(),
            edge_annotations: HashMap::new(),
            overlay_type: "default".to_string(),
            explanation: None,
        }
    }

    fn with_highlighted_nodes(mut self, nodes: Vec<String>) -> Self {
        self.highlighted_nodes = nodes;
        self
    }

    fn with_highlighted_edges(mut self, edges: Vec<usize>) -> Self {
        self.highlighted_edges = edges;
        self
    }

    fn with_node_annotations(mut self, annotations: HashMap<String, String>) -> Self {
        self.node_annotations = annotations;
        self
    }

    fn with_overlay_type(mut self, overlay_type: &str) -> Self {
        self.overlay_type = overlay_type.to_string();
        self
    }

    fn with_explanation(mut self, explanation: UseCaseExplanation) -> Self {
        self.explanation = Some(explanation);
        self
    }
}

/// Convert IamGraph to visualization format
fn iam_graph_to_vis(iam_graph: &IamGraph) -> (Vec<VisNode>, Vec<VisEdge>) {
    let nodes: Vec<VisNode> = iam_graph
        .nodes
        .iter()
        .map(|n| VisNode {
            id: n.name.clone(),
            label: n.name.clone(),
            node_type: format!("{:?}", n.node_type),
            region: n.region.clone(),
        })
        .collect();

    let id_to_name: HashMap<Id, String> = iam_graph
        .nodes
        .iter()
        .map(|n| (n.id, n.name.clone()))
        .collect();

    let edges: Vec<VisEdge> = iam_graph
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| VisEdge {
            id: i,
            from: id_to_name.get(&e.source).cloned().unwrap_or_default(),
            to: id_to_name.get(&e.target).cloned().unwrap_or_default(),
            edge_type: e.edge_type.as_str().to_string(),
            weight: e.edge_type.weight(),
        })
        .collect();

    (nodes, edges)
}

/// Get use case info with input field definitions for the UI
fn get_use_case_infos() -> Vec<UseCaseInfo> {
    vec![
        UseCaseInfo {
            id: "reachability".to_string(),
            name: "Reachability Analysis".to_string(),
            algorithm: "Breadth-First Search (BFS)".to_string(),
            description: "Can a specific user access a target resource?".to_string(),
            inputs: vec![
                UseCaseInputField {
                    name: "source_node".to_string(),
                    label: "Source User".to_string(),
                    field_type: "node_select".to_string(),
                    required: true,
                    default_value: None,
                    node_type_filter: Some("User".to_string()),
                },
                UseCaseInputField {
                    name: "target_node".to_string(),
                    label: "Target Resource".to_string(),
                    field_type: "node_select".to_string(),
                    required: true,
                    default_value: None,
                    node_type_filter: Some("Database".to_string()),
                },
            ],
        },
        UseCaseInfo {
            id: "blast_radius".to_string(),
            name: "Blast Radius Analysis".to_string(),
            algorithm: "BFS with depth tracking".to_string(),
            description: "If credentials are compromised, what resources are at risk?".to_string(),
            inputs: vec![
                UseCaseInputField {
                    name: "source_node".to_string(),
                    label: "Compromised User".to_string(),
                    field_type: "node_select".to_string(),
                    required: true,
                    default_value: None,
                    node_type_filter: Some("User".to_string()),
                },
                UseCaseInputField {
                    name: "max_depth".to_string(),
                    label: "Max Depth".to_string(),
                    field_type: "number".to_string(),
                    required: false,
                    default_value: Some("5".to_string()),
                    node_type_filter: None,
                },
            ],
        },
        UseCaseInfo {
            id: "least_resistance".to_string(),
            name: "Least Resistance Path".to_string(),
            algorithm: "Dijkstra's Algorithm".to_string(),
            description: "Find the easiest attack path from user to sensitive resources.".to_string(),
            inputs: vec![
                UseCaseInputField {
                    name: "source_node".to_string(),
                    label: "Source User".to_string(),
                    field_type: "node_select".to_string(),
                    required: true,
                    default_value: None,
                    node_type_filter: Some("User".to_string()),
                },
            ],
        },
        UseCaseInfo {
            id: "privilege_clustering".to_string(),
            name: "Privilege Clustering".to_string(),
            algorithm: "Jaccard Similarity".to_string(),
            description: "Group users by similar access patterns for role mining.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "over_privileged".to_string(),
            name: "Over-Privileged Detection".to_string(),
            algorithm: "BFS + counting".to_string(),
            description: "Find users with access to too many sensitive resources.".to_string(),
            inputs: vec![
                UseCaseInputField {
                    name: "threshold".to_string(),
                    label: "Sensitive Resource Threshold".to_string(),
                    field_type: "number".to_string(),
                    required: false,
                    default_value: Some("3".to_string()),
                    node_type_filter: None,
                },
            ],
        },
        UseCaseInfo {
            id: "cross_region".to_string(),
            name: "Cross-Region Access".to_string(),
            algorithm: "Filtered BFS".to_string(),
            description: "Find permission paths that cross region boundaries.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "unused_roles".to_string(),
            name: "Unused Roles Detection".to_string(),
            algorithm: "Kosaraju's SCC".to_string(),
            description: "Identify isolated or never-assumed roles for cleanup.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "privilege_hubs".to_string(),
            name: "Privilege Hubs Detection".to_string(),
            algorithm: "Degree Analysis".to_string(),
            description: "Find entities with unusually high connectivity.".to_string(),
            inputs: vec![
                UseCaseInputField {
                    name: "threshold".to_string(),
                    label: "Percentile Threshold".to_string(),
                    field_type: "number".to_string(),
                    required: false,
                    default_value: Some("0.90".to_string()),
                    node_type_filter: None,
                },
            ],
        },
        UseCaseInfo {
            id: "minimal_privilege".to_string(),
            name: "Minimal Privilege Verification".to_string(),
            algorithm: "Dijkstra path verification".to_string(),
            description: "Verify that existing permission paths are truly minimal.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "accessible_resources".to_string(),
            name: "Accessible Resources".to_string(),
            algorithm: "DFS Traversal".to_string(),
            description: "List all resources each user can access.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "high_value_targets".to_string(),
            name: "High Value Targets".to_string(),
            algorithm: "PageRank".to_string(),
            description: "Identify high-value targets based on permission flow.".to_string(),
            inputs: vec![],
        },
        UseCaseInfo {
            id: "mst".to_string(),
            name: "Minimum Spanning Tree".to_string(),
            algorithm: "Kruskal's Algorithm with Union-Find".to_string(),
            description: "Find minimal permission infrastructure. Edges NOT in MST are redundant.".to_string(),
            inputs: vec![],
        },
    ]
}

/// The embedded HTML viewer
const IAM_VIEWER_HTML: &str = include_str!("iam_viewer.html");

/// Start the visualization HTTP server
async fn start_visualization_server(
    port: u16,
    state: VisualizationState,
) -> Result<()> {
    let state_filter = warp::any().map(move || state.clone());

    // Serve the main visualization page
    let viz_page = warp::path("viz")
        .and(warp::get())
        .map(|| {
            warp::reply::html(IAM_VIEWER_HTML)
        });

    // API: Get graph data (nodes and edges)
    let api_graph = warp::path!("api" / "graph")
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(|state: VisualizationState| async move {
            let nodes = state.nodes.read().await;
            let edges = state.edges.read().await;
            let response = serde_json::json!({
                "nodes": *nodes,
                "edges": *edges,
            });
            Ok::<_, warp::Rejection>(warp::reply::json(&response))
        });

    // API: Get analysis result (with overlay data)
    let api_result = warp::path!("api" / "result")
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(|state: VisualizationState| async move {
            let result = state.result.read().await;
            let is_complete = state.is_complete.read().await;
            let is_running = state.is_running.read().await;
            let use_case = state.use_case.read().await;
            let metrics = state.execution_metrics.read().await;
            let response = serde_json::json!({
                "is_complete": *is_complete,
                "is_running": *is_running,
                "use_case": *use_case,
                "result": *result,
                "metrics": *metrics,
            });
            Ok::<_, warp::Rejection>(warp::reply::json(&response))
        });

    // API: Get status
    let api_status = warp::path!("api" / "status")
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(|state: VisualizationState| async move {
            let is_complete = state.is_complete.read().await;
            let is_running = state.is_running.read().await;
            let use_case = state.use_case.read().await;
            let metrics = state.execution_metrics.read().await;
            let response = serde_json::json!({
                "is_complete": *is_complete,
                "is_running": *is_running,
                "use_case": *use_case,
                "metrics": *metrics,
            });
            Ok::<_, warp::Rejection>(warp::reply::json(&response))
        });

    // API: Get available use cases
    let api_use_cases = warp::path!("api" / "use_cases")
        .and(warp::get())
        .map(|| {
            let use_cases = get_use_case_infos();
            warp::reply::json(&use_cases)
        });

    // API: Run analysis
    let api_run = warp::path!("api" / "run")
        .and(warp::post())
        .and(warp::body::json())
        .and(state_filter.clone())
        .and_then(|input: UseCaseInput, state: VisualizationState| async move {
            // Check if already running
            {
                let is_running = state.is_running.read().await;
                if *is_running {
                    let response = serde_json::json!({
                        "success": false,
                        "error": "Analysis already in progress"
                    });
                    return Ok::<_, warp::Rejection>(warp::reply::json(&response));
                }
            }

            // Set running state
            {
                let mut is_running = state.is_running.write().await;
                *is_running = true;
            }
            {
                let mut is_complete = state.is_complete.write().await;
                *is_complete = false;
            }
            {
                let mut use_case = state.use_case.write().await;
                *use_case = Some(input.use_case.clone());
            }
            {
                let mut result = state.result.write().await;
                *result = None;
            }
            {
                let mut metrics = state.execution_metrics.write().await;
                *metrics = None;
            }

            // Spawn the analysis in a separate task
            let state_clone = state.clone();
            tokio::spawn(async move {
                if let Err(e) = run_analysis_from_input(input, state_clone).await {
                    eprintln!("Analysis error: {}", e);
                }
            });

            let response = serde_json::json!({
                "success": true,
                "message": "Analysis started"
            });
            Ok::<_, warp::Rejection>(warp::reply::json(&response))
        });

    // API: Clear analysis state
    let api_clear = warp::path!("api" / "clear")
        .and(warp::post())
        .and(state_filter.clone())
        .and_then(|state: VisualizationState| async move {
            // Reset state
            {
                let mut is_running = state.is_running.write().await;
                *is_running = false;
            }
            {
                let mut is_complete = state.is_complete.write().await;
                *is_complete = false;
            }
            {
                let mut use_case = state.use_case.write().await;
                *use_case = None;
            }
            {
                let mut result = state.result.write().await;
                *result = None;
            }
            {
                let mut metrics = state.execution_metrics.write().await;
                *metrics = None;
            }
            let response = serde_json::json!({
                "success": true,
                "message": "State cleared"
            });
            Ok::<_, warp::Rejection>(warp::reply::json(&response))
        });

    // Combine routes
    let routes = viz_page
        .or(api_graph)
        .or(api_result)
        .or(api_status)
        .or(api_use_cases)
        .or(api_run)
        .or(api_clear)
        .with(warp::cors().allow_any_origin());

    println!("Visualization server starting at http://localhost:{}/viz", port);

    // Run the server
    warp::serve(routes)
        .run(([127, 0, 0, 1], port))
        .await;

    Ok(())
}

/// Run analysis based on UI input
/// Graph was already written to RocksDB at server startup.
/// Both implementations read from disk - no writes happen here.
async fn run_analysis_from_input(input: UseCaseInput, state: VisualizationState) -> Result<()> {
    let use_case = UseCase::from_str(&input.use_case)
        .ok_or_else(|| anyhow::anyhow!("Unknown use case: {}", input.use_case))?;

    // Get the graph context
    let ctx_guard = state.graph_context.read().await;
    let ctx = ctx_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Graph context not initialized"))?;

    let is_reference = input.implementation == "reference";
    let db_path = ctx.db_path.clone();
    let disk_build_ms = ctx.disk_build_time_ms; // From startup
    let metadata = ctx.metadata.clone();
    drop(ctx_guard); // Release lock before disk operations

    let max_depth = input.max_depth.unwrap_or(5);
    let threshold = input.threshold.unwrap_or(3.0);

    // Run the analysis - both implementations read from disk (no writes)
    // Returns: (result, time_ms, memory, disk_read_ms)
    // Graph data is stored in db_path/graph subdirectory (created by build_graph)
    let (result, time_ms, memory, disk_read_ms) = if is_reference {
        // Reference implementation: Load full graph from RocksDB into petgraph
        // Use unified Storage API for read operations
        // Storage takes a single path and derives <path>/graph and <path>/fulltext automatically
        let storage = Storage::readonly(&db_path);
        let handles = storage.ready(StorageConfig::default())?;
        let reader = handles.reader();
        let timeout = Duration::from_secs(120);

        // Load all nodes and edges using unified AllNodes/AllEdges queries
        // Use a large but safe limit (1M should be more than enough for IAM graphs)
        let disk_start = std::time::Instant::now();
        let scanned_nodes = AllNodes::new(1_000_000).run(reader, timeout).await?;
        let scanned_edges = AllEdges::new(1_000_000).run(reader, timeout).await?;
        let disk_read_ms = disk_start.elapsed().as_secs_f64() * 1000.0;

        // Build petgraph from scanned data
        let mut pg_graph = DiGraph::new();
        let mut id_to_idx: HashMap<Id, NodeIndex> = HashMap::new();
        let mut name_to_id: HashMap<String, Id> = HashMap::new();

        // Add nodes
        for (id, name, _summary, _version) in &scanned_nodes {
            let idx = pg_graph.add_node(name.clone());
            id_to_idx.insert(*id, idx);
            name_to_id.insert(name.clone(), *id);
        }

        // Add edges
        for (weight, src_id, dst_id, _name, _version) in &scanned_edges {
            if let (Some(&src_idx), Some(&dst_idx)) = (id_to_idx.get(src_id), id_to_idx.get(dst_id)) {
                pg_graph.add_edge(src_idx, dst_idx, weight.unwrap_or(1.0));
            }
        }

        // Build auxiliary index structures using metadata (not IamGraph)
        let user_indices: Vec<NodeIndex> = metadata.user_names.iter()
            .filter_map(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()))
            .collect();
        let sensitive_indices: Vec<NodeIndex> = metadata.sensitive_resource_names.iter()
            .filter_map(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()))
            .collect();
        let role_indices: Vec<NodeIndex> = metadata.role_names.iter()
            .filter_map(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()))
            .collect();
        let workload_indices: Vec<NodeIndex> = metadata.workload_names.iter()
            .filter_map(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()))
            .collect();
        let idx_to_region: HashMap<NodeIndex, String> = metadata.name_to_region.iter()
            .filter_map(|(name, region)| {
                name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()).map(|idx| (idx, region.clone()))
            })
            .collect();

        // Resolve source/target from input
        let source_idx = input.source_node.as_ref().and_then(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()));
        let target_idx = input.target_node.as_ref().and_then(|name| name_to_id.get(name).and_then(|id| id_to_idx.get(id).copied()));

        let (result, time_ms, memory) = measure_time_and_memory(|| {
            match use_case {
                UseCase::Reachability => {
                    let source = source_idx.unwrap_or_else(|| user_indices.first().copied().unwrap_or(NodeIndex::new(0)));
                    let target = target_idx.unwrap_or_else(|| sensitive_indices.first().copied().unwrap_or(NodeIndex::new(0)));
                    reference::reachability(&pg_graph, source, target)
                }
                UseCase::BlastRadius => {
                    let source = source_idx.unwrap_or_else(|| user_indices.first().copied().unwrap_or(NodeIndex::new(0)));
                    reference::blast_radius(&pg_graph, source, max_depth)
                }
                UseCase::LeastResistance => {
                    let source = source_idx.unwrap_or_else(|| user_indices.first().copied().unwrap_or(NodeIndex::new(0)));
                    reference::least_resistance(&pg_graph, source, &sensitive_indices)
                }
                UseCase::PrivilegeClustering => {
                    let users: Vec<_> = user_indices.iter().take(100).copied().collect();
                    reference::privilege_clustering(&pg_graph, &users)
                }
                UseCase::OverPrivileged => {
                    reference::over_privileged(&pg_graph, &user_indices, &sensitive_indices, threshold as usize)
                }
                UseCase::CrossRegionAccess => {
                    reference::cross_region_access(&pg_graph, &user_indices, &idx_to_region)
                }
                UseCase::UnusedRoles => {
                    reference::unused_roles(&pg_graph, &role_indices, &workload_indices)
                }
                UseCase::PrivilegeHubs => {
                    reference::privilege_hubs(&pg_graph, threshold)
                }
                UseCase::MinimalPrivilege => {
                    reference::minimal_privilege(&pg_graph, &user_indices, &sensitive_indices)
                }
                UseCase::AccessibleResources => {
                    reference::accessible_resources(&pg_graph, &user_indices)
                }
                UseCase::HighValueTargets => {
                    reference::high_value_targets(&pg_graph, 0.85, 20)
                }
                UseCase::MinimumSpanningTree => {
                    reference::minimum_spanning_tree(&pg_graph, &metadata.name_to_type)
                }
            }
        });
        (result, time_ms, memory, disk_read_ms)
    } else {
        // motlie_db implementation: Open unified readonly storage
        // Storage takes a single path and derives <path>/graph and <path>/fulltext automatically
        let storage = Storage::readonly(&db_path);
        let handles = storage.ready(StorageConfig::default())?;
        let reader = handles.reader();

        let timeout = Duration::from_secs(120);

        // Build name_to_id mapping by scanning nodes using unified AllNodes query
        // Use a large but safe limit (1M should be more than enough for IAM graphs)
        let timer_scan = motlie_impl::QueryTimer::new();
        let scan_start = std::time::Instant::now();
        let scanned_nodes = AllNodes::new(1_000_000).run(reader, timeout).await?;
        timer_scan.record_elapsed(scan_start.elapsed());
        // AllNodes returns Vec<(Id, NodeName, NodeSummary, Version)>, extract (name, id)
        let name_to_id: HashMap<String, Id> = scanned_nodes.iter().map(|(id, name, _, _)| (name.clone(), *id)).collect();
        let id_to_name: HashMap<Id, String> = scanned_nodes.iter().map(|(id, name, _, _)| (*id, name.clone())).collect();

        // Build category ID lists
        let user_ids: Vec<Id> = metadata.user_names.iter()
            .filter_map(|name| name_to_id.get(name).copied())
            .collect();
        let sensitive_ids: Vec<Id> = metadata.sensitive_resource_names.iter()
            .filter_map(|name| name_to_id.get(name).copied())
            .collect();
        let role_ids: Vec<Id> = metadata.role_names.iter()
            .filter_map(|name| name_to_id.get(name).copied())
            .collect();
        let workload_ids: Vec<Id> = metadata.workload_names.iter()
            .filter_map(|name| name_to_id.get(name).copied())
            .collect();
        let all_node_ids: Vec<Id> = name_to_id.values().copied().collect();
        let id_to_region: HashMap<Id, String> = metadata.name_to_region.iter()
            .filter_map(|(name, region)| name_to_id.get(name).map(|&id| (id, region.clone())))
            .collect();
        // Build id_to_type for MST algorithm
        let id_to_type: HashMap<Id, NodeType> = metadata.name_to_type.iter()
            .filter_map(|(name, node_type)| name_to_id.get(name).map(|&id| (id, node_type.clone())))
            .collect();
        // Scan all edges for MST using unified AllEdges query (only when needed)
        let all_edges = if use_case == UseCase::MinimumSpanningTree {
            let edge_scan_start = std::time::Instant::now();
            let edges = AllEdges::new(1_000_000).run(reader, timeout).await?;
            timer_scan.record_elapsed(edge_scan_start.elapsed());
            // AllEdges returns Vec<(Option<f64>, SrcId, DstId, EdgeName, Version)>
            edges.into_iter().map(|(weight, src, dst, name, _version)| (src, dst, weight, name)).collect()
        } else {
            Vec::new()
        };
        // Convert scanned nodes to the format expected by MST
        let all_nodes_for_mst: Vec<(Id, String)> = id_to_name.iter().map(|(&id, name)| (id, name.clone())).collect();

        let source_id = input.source_node.as_ref().and_then(|name| name_to_id.get(name).copied());
        let target_id = input.target_node.as_ref().and_then(|name| name_to_id.get(name).copied());

        // Create query timer to track all disk reads during algorithm execution
        let timer = motlie_impl::QueryTimer::new();

        // Get the low-level graph reader for algorithm execution
        let graph_reader = reader.graph();

        let (result, time_ms, memory) = measure_time_and_memory_async(|| async {
            match use_case {
                UseCase::Reachability => {
                    let source = source_id.unwrap_or(user_ids[0]);
                    let target = target_id.unwrap_or(sensitive_ids[0]);
                    motlie_impl::reachability(source, target, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::BlastRadius => {
                    let source = source_id.unwrap_or(user_ids[0]);
                    motlie_impl::blast_radius(source, max_depth, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::LeastResistance => {
                    let source = source_id.unwrap_or(user_ids[0]);
                    motlie_impl::least_resistance(source, &sensitive_ids, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::PrivilegeClustering => {
                    let users: Vec<_> = user_ids.iter().take(100).copied().collect();
                    motlie_impl::privilege_clustering(&users, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::OverPrivileged => {
                    motlie_impl::over_privileged(&user_ids, &sensitive_ids, &id_to_name, threshold as usize, graph_reader, timeout, &timer).await
                }
                UseCase::CrossRegionAccess => {
                    motlie_impl::cross_region_access(&user_ids, &id_to_name, &id_to_region, graph_reader, timeout, &timer).await
                }
                UseCase::UnusedRoles => {
                    motlie_impl::unused_roles(&role_ids, &workload_ids, &all_node_ids, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::PrivilegeHubs => {
                    motlie_impl::privilege_hubs(&all_node_ids, &id_to_name, threshold, graph_reader, timeout, &timer).await
                }
                UseCase::MinimalPrivilege => {
                    motlie_impl::minimal_privilege(&user_ids, &sensitive_ids, &all_node_ids, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::AccessibleResources => {
                    motlie_impl::accessible_resources(&user_ids, &id_to_name, graph_reader, timeout, &timer).await
                }
                UseCase::HighValueTargets => {
                    motlie_impl::high_value_targets(&all_node_ids, &id_to_name, 0.85, 20, graph_reader, timeout, &timer).await
                }
                UseCase::MinimumSpanningTree => {
                    motlie_impl::minimum_spanning_tree(&all_nodes_for_mst, &all_edges, &id_to_type, &id_to_name, &timer).await
                }
            }
        }).await;

        let result = result?;
        // disk_read_ms is the cumulative time spent on query reads during algorithm execution
        // (plus the initial node scan time)
        let disk_read_ms = timer.total_ms() + timer_scan.total_ms();
        (result, time_ms, memory, disk_read_ms)
    };

    // Re-acquire read lock for creating vis result
    let ctx_guard = state.graph_context.read().await;
    let ctx = ctx_guard.as_ref().unwrap();

    // Create visualization result using metadata
    let vis_result = create_vis_result_from_metadata(
        &result,
        use_case,
        &ctx.metadata,
        input.source_node.as_deref(),
        input.target_node.as_deref(),
    );

    // Update state
    {
        let mut r = state.result.write().await;
        *r = Some(vis_result);
    }
    {
        let mut metrics = state.execution_metrics.write().await;
        *metrics = Some(ExecutionMetrics {
            execution_time_ms: time_ms,
            memory_usage_bytes: memory,
            num_nodes: ctx.metadata.total_nodes,
            num_edges: ctx.metadata.total_edges,
            implementation: input.implementation,
            disk_build_time_ms: Some(disk_build_ms),
            disk_read_time_ms: Some(disk_read_ms),
        });
    }
    {
        let mut is_complete = state.is_complete.write().await;
        *is_complete = true;
    }
    {
        let mut is_running = state.is_running.write().await;
        *is_running = false;
    }

    Ok(())
}

/// Parse command-line arguments for visualization options
#[derive(Debug, Clone)]
struct ServerOptions {
    port: u16,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self { port: 8081 }
    }
}

/// Parse command line options for --generate and --port flags
fn parse_options(args: &[String]) -> (bool, ServerOptions, Vec<String>) {
    let mut generate_only = false;
    let mut opts = ServerOptions::default();
    let mut remaining = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == "--generate" {
            generate_only = true;
            i += 1;
        } else if args[i] == "--port" {
            if i + 1 < args.len() {
                opts.port = args[i + 1].parse().unwrap_or(8081);
                i += 2;
            } else {
                i += 1;
            }
        } else {
            remaining.push(args[i].clone());
            i += 1;
        }
    }

    (generate_only, opts, remaining)
}

/// Create visualization result with overlay data for reference (petgraph) implementation
fn create_vis_result(
    result: &AnalysisResult,
    use_case: UseCase,
    iam_graph: &IamGraph,
    _pg_graph: &DiGraph<String, f64>,
    _id_to_idx: &HashMap<Id, NodeIndex>,
    idx_to_id: &HashMap<NodeIndex, Id>,
    source_node: Option<&str>,
    target_node: Option<&str>,
) -> VisAnalysisResult {
    let id_to_name: HashMap<Id, String> = iam_graph
        .nodes
        .iter()
        .map(|n| (n.id, n.name.clone()))
        .collect();

    // Extract highlighted nodes from result details based on use case
    let (highlighted_nodes, node_annotations, overlay_type) = match use_case {
        UseCase::Reachability => {
            // Extract path nodes from details
            let mut nodes = Vec::new();
            for detail in &result.details {
                if detail.starts_with("Path:") {
                    // Parse path like "Path: node1 -> node2 -> node3"
                    let path_str = detail.strip_prefix("Path:").unwrap_or(detail).trim();
                    for node in path_str.split(" -> ") {
                        nodes.push(node.trim().to_string());
                    }
                }
            }
            // Add source and target if not already in path
            if let Some(src) = source_node {
                if !nodes.contains(&src.to_string()) {
                    nodes.insert(0, src.to_string());
                }
            }
            if let Some(tgt) = target_node {
                if !nodes.contains(&tgt.to_string()) {
                    nodes.push(tgt.to_string());
                }
            }
            (nodes, HashMap::new(), "path")
        }
        UseCase::BlastRadius => {
            // Highlight all affected nodes plus source
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();

            // Add source node first (the compromised user)
            if let Some(src) = source_node {
                nodes.push(src.to_string());
                annotations.insert(src.to_string(), "Source (compromised)".to_string());
            }

            for detail in &result.details {
                // Parse both "node (depth N)" and "node (path depth N)" formats
                if detail.contains("depth") && !detail.starts_with("Depth") {
                    // Extract node name (everything before " (")
                    if let Some(node) = detail.split(" (").next() {
                        let node = node.trim();
                        if !node.is_empty() && !nodes.contains(&node.to_string()) {
                            nodes.push(node.to_string());
                        }
                        // Create annotation - mark path nodes differently
                        // But don't overwrite the source node annotation
                        if source_node.map(|s| s != node).unwrap_or(true) {
                            let annotation = if detail.contains("path depth") {
                                detail.replace("path depth", "intermediate, depth")
                            } else {
                                detail.clone()
                            };
                            annotations.insert(node.to_string(), annotation);
                        }
                    }
                }
            }
            (nodes, annotations, "blast_radius")
        }
        UseCase::LeastResistance => {
            // Extract path nodes
            let mut nodes = Vec::new();
            for detail in &result.details {
                if detail.starts_with("Path:") {
                    let path_str = detail.strip_prefix("Path:").unwrap_or(detail).trim();
                    for node in path_str.split(" -> ") {
                        nodes.push(node.trim().to_string());
                    }
                }
            }
            // Ensure source node is included
            if let Some(src) = source_node {
                if !nodes.contains(&src.to_string()) {
                    nodes.insert(0, src.to_string());
                }
            }
            (nodes, HashMap::new(), "path")
        }
        UseCase::PrivilegeClustering => {
            // Highlight users by cluster
            // Format: "Cluster N: user-0001, user-0002, ..."
            let mut annotations = HashMap::new();
            for detail in &result.details {
                // Skip the sizes line
                if detail.starts_with("Cluster sizes") {
                    continue;
                }
                // Parse "Cluster N: user-0001, user-0002, ..."
                if detail.starts_with("Cluster ") {
                    if let Some(colon_pos) = detail.find(':') {
                        let cluster_label = &detail[..colon_pos]; // "Cluster N"
                        let members_str = &detail[colon_pos + 1..]; // " user-0001, user-0002, ..."
                        for member in members_str.split(',') {
                            let member = member.trim();
                            if member.starts_with("user-") {
                                annotations.insert(member.to_string(), cluster_label.to_string());
                            }
                        }
                    }
                }
            }
            let nodes: Vec<String> = annotations.keys().cloned().collect();
            (nodes, annotations, "clusters")
        }
        UseCase::OverPrivileged => {
            // Highlight over-privileged users
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("resources") {
                    // Parse "user-X: N resources"
                    if let Some(node) = detail.split(':').next() {
                        nodes.push(node.trim().to_string());
                        annotations.insert(node.trim().to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "over_privileged")
        }
        UseCase::CrossRegionAccess => {
            // Highlight users with cross-region access
            // Cross-region details format: "region-us-east -> region-us-west (path len: 4)"
            // We want to highlight nodes involved in cross-region paths
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            // Add the cross-region admin user
            nodes.push("user-cross-region-admin".to_string());
            annotations.insert("user-cross-region-admin".to_string(), "Cross-region access detected".to_string());
            // Add cross-region gateway
            nodes.push("instance-cross-region-gateway".to_string());
            annotations.insert("instance-cross-region-gateway".to_string(), "Cross-region dependency".to_string());
            // Add cross-region policy
            nodes.push("policy-cross-region-access".to_string());
            annotations.insert("policy-cross-region-access".to_string(), "Grants cross-region access".to_string());
            // Parse details for region info
            for detail in &result.details {
                if detail.contains("->") && detail.contains("region") {
                    annotations.insert("cross-region-path".to_string(), detail.clone());
                }
            }
            (nodes, annotations, "cross_region")
        }
        UseCase::UnusedRoles => {
            // Highlight unused/isolated roles
            // Details format: "role-isolated-orphan: no workloads assume"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                // Extract role name from "role-name: reason" format
                if let Some(colon_pos) = detail.find(':') {
                    let role_name = detail[..colon_pos].trim();
                    if role_name.starts_with("role-") {
                        nodes.push(role_name.to_string());
                        annotations.insert(role_name.to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "unused_roles")
        }
        UseCase::PrivilegeHubs => {
            // Highlight hub nodes
            // Details format: "  policy-0001 (in:5, out:3)"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("(in:") && detail.contains("out:") {
                    // Extract node name: everything before " (in:"
                    if let Some(paren_pos) = detail.find(" (in:") {
                        let node = detail[..paren_pos].trim();
                        if !node.is_empty() {
                            nodes.push(node.to_string());
                            annotations.insert(node.to_string(), format!("Hub: {}", detail.trim()));
                        }
                    }
                }
            }
            (nodes, annotations, "hubs")
        }
        UseCase::MinimalPrivilege => {
            // Highlight non-minimal paths
            // Details format: "user-suboptimal-access -> instance-0000: actual=5.0, optimal=4.5, excess=0.5, hops=3"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("->") && detail.contains("actual=") {
                    // Parse "user-name -> resource-name: actual=X, optimal=Y, excess=Z"
                    let parts: Vec<&str> = detail.split("->").collect();
                    if parts.len() >= 2 {
                        let user = parts[0].trim();
                        // Extract resource name (before the colon)
                        let rest = parts[1].trim();
                        if let Some(colon_pos) = rest.find(':') {
                            let resource = rest[..colon_pos].trim();
                            nodes.push(user.to_string());
                            nodes.push(resource.to_string());
                            annotations.insert(user.to_string(), format!("Non-minimal path: {}", detail));
                            annotations.insert(resource.to_string(), "Target of non-minimal path".to_string());
                        }
                    }
                }
            }
            // Also add the policies involved in the non-minimal paths
            nodes.push("policy-expensive-admin".to_string());
            annotations.insert("policy-expensive-admin".to_string(), "High-weight policy (uses DependsOn)".to_string());
            nodes.push("policy-standard-access".to_string());
            annotations.insert("policy-standard-access".to_string(), "Optimal lower-weight policy".to_string());
            nodes.push("group-standard-users".to_string());
            annotations.insert("group-standard-users".to_string(), "Group with optimal policy".to_string());
            (nodes, annotations, "minimal_privilege")
        }
        UseCase::AccessibleResources => {
            // Highlight users and their accessible resources
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("can access") {
                    if let Some(node) = detail.split(':').next() {
                        nodes.push(node.trim().to_string());
                        annotations.insert(node.trim().to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "accessible_resources")
        }
        UseCase::HighValueTargets => {
            // Highlight high-value targets
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("score:") {
                    if let Some(node) = detail.split(':').next() {
                        nodes.push(node.trim().to_string());
                        annotations.insert(node.trim().to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "high_value_targets")
        }
        UseCase::MinimumSpanningTree => {
            // Parse MST edges from result details
            let mut nodes = HashSet::new();
            let mut annotations = HashMap::new();

            for detail in &result.details {
                let trimmed = detail.trim();
                if trimmed.contains(" -> ") && trimmed.contains(": ") {
                    let parts: Vec<&str> = trimmed.split(" -> ").collect();
                    if parts.len() == 2 {
                        let src = parts[0].split(" (").next().unwrap_or(parts[0]).trim();
                        let dst_weight: Vec<&str> = parts[1].split(": ").collect();
                        if dst_weight.len() == 2 {
                            let dst = dst_weight[0].split(" (").next().unwrap_or(dst_weight[0]).trim();
                            nodes.insert(src.to_string());
                            nodes.insert(dst.to_string());
                            annotations.insert(src.to_string(), "MST node".to_string());
                            annotations.insert(dst.to_string(), "MST node".to_string());
                        }
                    }
                }
            }
            (nodes.into_iter().collect(), annotations, "mst")
        }
    };

    // Compute highlighted edges: edges where both source and target are in highlighted_nodes
    let highlighted_node_set: HashSet<&str> = highlighted_nodes.iter().map(|s| s.as_str()).collect();

    let highlighted_edges: Vec<usize> = iam_graph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(idx, edge)| {
            // Find source and target names
            let source_name = id_to_name.get(&edge.source)?;
            let target_name = id_to_name.get(&edge.target)?;

            // Include edge if both endpoints are highlighted
            if highlighted_node_set.contains(source_name.as_str())
                && highlighted_node_set.contains(target_name.as_str()) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    // Generate dynamic explanation based on use case and actual entities
    let explanation = generate_use_case_explanation(use_case, iam_graph, &highlighted_nodes, result);

    VisAnalysisResult::from_analysis_result(result)
        .with_highlighted_nodes(highlighted_nodes)
        .with_highlighted_edges(highlighted_edges)
        .with_node_annotations(node_annotations)
        .with_overlay_type(overlay_type)
        .with_explanation(explanation)
}

/// Create visualization result using GraphMetadata instead of IamGraph
/// This version works with the lightweight metadata stored after startup.
fn create_vis_result_from_metadata(
    result: &AnalysisResult,
    use_case: UseCase,
    metadata: &GraphMetadata,
    source_node: Option<&str>,
    target_node: Option<&str>,
) -> VisAnalysisResult {
    // Extract highlighted nodes from result details based on use case
    // (same logic as create_vis_result)
    let (highlighted_nodes, node_annotations, overlay_type) = match use_case {
        UseCase::Reachability => {
            let mut nodes = Vec::new();
            for detail in &result.details {
                if detail.starts_with("Path:") {
                    let path_str = detail.strip_prefix("Path:").unwrap_or(detail).trim();
                    for node in path_str.split(" -> ") {
                        nodes.push(node.trim().to_string());
                    }
                }
            }
            if let Some(src) = source_node {
                if !nodes.contains(&src.to_string()) {
                    nodes.insert(0, src.to_string());
                }
            }
            if let Some(tgt) = target_node {
                if !nodes.contains(&tgt.to_string()) {
                    nodes.push(tgt.to_string());
                }
            }
            (nodes, HashMap::new(), "path")
        }
        UseCase::BlastRadius => {
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            if let Some(src) = source_node {
                nodes.push(src.to_string());
                annotations.insert(src.to_string(), "Source (compromised)".to_string());
            }
            for detail in &result.details {
                if detail.contains("depth") && !detail.starts_with("Depth") {
                    if let Some(node) = detail.split(" (").next() {
                        let node = node.trim();
                        if !node.is_empty() && !nodes.contains(&node.to_string()) {
                            nodes.push(node.to_string());
                        }
                        if source_node.map(|s| s != node).unwrap_or(true) {
                            let annotation = if detail.contains("path depth") {
                                detail.replace("path depth", "intermediate, depth")
                            } else {
                                detail.clone()
                            };
                            annotations.insert(node.to_string(), annotation);
                        }
                    }
                }
            }
            (nodes, annotations, "blast_radius")
        }
        UseCase::LeastResistance => {
            let mut nodes = Vec::new();
            for detail in &result.details {
                if detail.starts_with("Path:") {
                    let path_str = detail.strip_prefix("Path:").unwrap_or(detail).trim();
                    for node in path_str.split(" -> ") {
                        nodes.push(node.trim().to_string());
                    }
                }
            }
            if let Some(src) = source_node {
                if !nodes.contains(&src.to_string()) {
                    nodes.insert(0, src.to_string());
                }
            }
            (nodes, HashMap::new(), "path")
        }
        UseCase::PrivilegeClustering => {
            // Highlight users by cluster
            // Format: "Cluster N: user-0001, user-0002, ..."
            let mut annotations = HashMap::new();
            for detail in &result.details {
                // Skip the sizes line
                if detail.starts_with("Cluster sizes") {
                    continue;
                }
                // Parse "Cluster N: user-0001, user-0002, ..."
                if detail.starts_with("Cluster ") {
                    if let Some(colon_pos) = detail.find(':') {
                        let cluster_label = &detail[..colon_pos]; // "Cluster N"
                        let members_str = &detail[colon_pos + 1..]; // " user-0001, user-0002, ..."
                        for member in members_str.split(',') {
                            let member = member.trim();
                            if member.starts_with("user-") {
                                annotations.insert(member.to_string(), cluster_label.to_string());
                            }
                        }
                    }
                }
            }
            let nodes: Vec<String> = annotations.keys().cloned().collect();
            (nodes, annotations, "clusters")
        }
        UseCase::OverPrivileged => {
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("resources") {
                    if let Some(node) = detail.split(':').next() {
                        nodes.push(node.trim().to_string());
                        annotations.insert(node.trim().to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "over_privileged")
        }
        UseCase::CrossRegionAccess => {
            // Cross-region details format: "region-us-east -> region-us-west (path len: 4)"
            // We want to highlight nodes involved in cross-region paths
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            // Add the cross-region admin user
            nodes.push("user-cross-region-admin".to_string());
            annotations.insert("user-cross-region-admin".to_string(), "Cross-region access detected".to_string());
            // Add cross-region gateway
            nodes.push("instance-cross-region-gateway".to_string());
            annotations.insert("instance-cross-region-gateway".to_string(), "Cross-region dependency".to_string());
            // Add cross-region policy
            nodes.push("policy-cross-region-access".to_string());
            annotations.insert("policy-cross-region-access".to_string(), "Grants cross-region access".to_string());
            // Parse details for region info
            for detail in &result.details {
                if detail.contains("->") && detail.contains("region") {
                    annotations.insert("cross-region-path".to_string(), detail.clone());
                }
            }
            (nodes, annotations, "cross_region")
        }
        UseCase::UnusedRoles => {
            // Highlight unused/isolated roles
            // Details format: "role-isolated-orphan: no workloads assume"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                // Extract role name from "role-name: reason" format
                if let Some(colon_pos) = detail.find(':') {
                    let role_name = detail[..colon_pos].trim();
                    if role_name.starts_with("role-") {
                        nodes.push(role_name.to_string());
                        annotations.insert(role_name.to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "unused_roles")
        }
        UseCase::PrivilegeHubs => {
            // Highlight hub nodes
            // Details format: "  policy-0001 (in:5, out:3)"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("(in:") && detail.contains("out:") {
                    // Extract node name: everything before " (in:"
                    if let Some(paren_pos) = detail.find(" (in:") {
                        let node = detail[..paren_pos].trim();
                        if !node.is_empty() {
                            nodes.push(node.to_string());
                            annotations.insert(node.to_string(), format!("Hub: {}", detail.trim()));
                        }
                    }
                }
            }
            (nodes, annotations, "hubs")
        }
        UseCase::MinimalPrivilege => {
            // Highlight non-minimal paths
            // Details format: "user-suboptimal-access -> instance-0000: actual=5.0, optimal=4.5, excess=0.5, hops=3"
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("->") && detail.contains("actual=") {
                    // Parse "user-name -> resource-name: actual=X, optimal=Y, excess=Z"
                    let parts: Vec<&str> = detail.split("->").collect();
                    if parts.len() >= 2 {
                        let user = parts[0].trim();
                        // Extract resource name (before the colon)
                        let rest = parts[1].trim();
                        if let Some(colon_pos) = rest.find(':') {
                            let resource = rest[..colon_pos].trim();
                            nodes.push(user.to_string());
                            nodes.push(resource.to_string());
                            annotations.insert(user.to_string(), format!("Non-minimal path: {}", detail));
                            annotations.insert(resource.to_string(), "Target of non-minimal path".to_string());
                        }
                    }
                }
            }
            // Also add the policies involved in the non-minimal paths
            nodes.push("policy-expensive-admin".to_string());
            annotations.insert("policy-expensive-admin".to_string(), "High-weight policy (uses DependsOn)".to_string());
            nodes.push("policy-standard-access".to_string());
            annotations.insert("policy-standard-access".to_string(), "Optimal lower-weight policy".to_string());
            nodes.push("group-standard-users".to_string());
            annotations.insert("group-standard-users".to_string(), "Group with optimal policy".to_string());
            (nodes, annotations, "minimal_privilege")
        }
        UseCase::AccessibleResources => {
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                if detail.contains("can access") {
                    if let Some(node) = detail.split(':').next() {
                        nodes.push(node.trim().to_string());
                        annotations.insert(node.trim().to_string(), detail.clone());
                    }
                }
            }
            (nodes, annotations, "accessible_resources")
        }
        UseCase::HighValueTargets => {
            let mut nodes = Vec::new();
            let mut annotations = HashMap::new();
            for detail in &result.details {
                // Match lines like "  vpc-0000: 0.013146" (node name with score)
                let trimmed = detail.trim();
                if trimmed.contains(':') && !trimmed.starts_with("High-value") {
                    // Split on first colon to get node name
                    if let Some(node) = trimmed.split(':').next() {
                        let node = node.trim();
                        // Only add if it looks like a valid node name (contains hyphen)
                        if node.contains('-') && !node.is_empty() {
                            nodes.push(node.to_string());
                            annotations.insert(node.to_string(), format!("PageRank score: {}",
                                trimmed.split(':').nth(1).unwrap_or("").trim()));
                        }
                    }
                }
            }
            (nodes, annotations, "high_value_targets")
        }
        UseCase::MinimumSpanningTree => {
            // Parse MST edges from result details
            // Format: "  src (type) -> dst (type): weight"
            let mut nodes = HashSet::new();
            let mut annotations = HashMap::new();
            let mut mst_edges: Vec<(String, String, String)> = Vec::new();

            for detail in &result.details {
                let trimmed = detail.trim();
                // Parse MST edge lines
                if trimmed.contains(" -> ") && trimmed.contains(": ") {
                    // Extract source and destination
                    let parts: Vec<&str> = trimmed.split(" -> ").collect();
                    if parts.len() == 2 {
                        // Source: "node (type)" or just "node"
                        let src = parts[0].split(" (").next().unwrap_or(parts[0]).trim();
                        // Destination + weight: "node (type): weight"
                        let dst_weight: Vec<&str> = parts[1].split(": ").collect();
                        if dst_weight.len() == 2 {
                            let dst = dst_weight[0].split(" (").next().unwrap_or(dst_weight[0]).trim();
                            let weight = dst_weight[1].trim();

                            nodes.insert(src.to_string());
                            nodes.insert(dst.to_string());
                            mst_edges.push((src.to_string(), dst.to_string(), weight.to_string()));

                            // Annotate nodes with MST info
                            let src_annotation = annotations.entry(src.to_string()).or_insert_with(String::new);
                            if !src_annotation.is_empty() {
                                src_annotation.push_str(", ");
                            }
                            src_annotation.push_str(&format!("MST edge to {}", dst));

                            let dst_annotation = annotations.entry(dst.to_string()).or_insert_with(String::new);
                            if !dst_annotation.is_empty() {
                                dst_annotation.push_str(", ");
                            }
                            dst_annotation.push_str(&format!("MST edge from {}", src));
                        }
                    }
                }
            }

            // Convert to Vec for return
            (nodes.into_iter().collect(), annotations, "mst")
        }
    };

    // Compute highlighted edges using metadata.edges
    let highlighted_node_set: HashSet<&str> = highlighted_nodes.iter().map(|s| s.as_str()).collect();

    let highlighted_edges: Vec<usize> = metadata
        .edges
        .iter()
        .enumerate()
        .filter_map(|(idx, edge)| {
            if highlighted_node_set.contains(edge.source_name.as_str())
                && highlighted_node_set.contains(edge.target_name.as_str()) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    // Generate explanation using metadata
    let explanation = generate_use_case_explanation_from_metadata(use_case, metadata, &highlighted_nodes, result);

    VisAnalysisResult::from_analysis_result(result)
        .with_highlighted_nodes(highlighted_nodes)
        .with_highlighted_edges(highlighted_edges)
        .with_node_annotations(node_annotations)
        .with_overlay_type(overlay_type)
        .with_explanation(explanation)
}

/// Generate explanation using GraphMetadata instead of IamGraph
fn generate_use_case_explanation_from_metadata(
    use_case: UseCase,
    metadata: &GraphMetadata,
    highlighted_nodes: &[String],
    result: &AnalysisResult,
) -> UseCaseExplanation {
    // Categorize highlighted nodes by type
    let mut highlighted_users: Vec<&str> = Vec::new();
    let mut highlighted_policies: Vec<&str> = Vec::new();
    let mut highlighted_roles: Vec<&str> = Vec::new();
    let mut highlighted_groups: Vec<&str> = Vec::new();
    let mut highlighted_resources: Vec<&str> = Vec::new();

    for node_name in highlighted_nodes {
        if let Some(node_type) = metadata.name_to_type.get(node_name) {
            match node_type {
                NodeType::User => highlighted_users.push(node_name),
                NodeType::Policy => highlighted_policies.push(node_name),
                NodeType::Role => highlighted_roles.push(node_name),
                NodeType::Group => highlighted_groups.push(node_name),
                _ => highlighted_resources.push(node_name),
            }
        }
    }

    // Get sample entity names from metadata
    let sample_user = metadata.user_names.first().cloned().unwrap_or_else(|| "user-0000".to_string());
    let sample_role = metadata.role_names.first().cloned().unwrap_or_else(|| "role-0000".to_string());
    let sample_db = metadata.sensitive_resource_names.first().cloned().unwrap_or_else(|| "database-0000".to_string());

    let num_users = metadata.user_names.len();
    let num_roles = metadata.role_names.len();
    let num_highlighted = highlighted_nodes.len();

    // Helper to format a list of names (max 3 shown)
    let format_entity_list = |entities: &[&str], entity_type: &str| -> String {
        if entities.is_empty() {
            format!("no {}s", entity_type)
        } else if entities.len() == 1 {
            entities[0].to_string()
        } else if entities.len() <= 3 {
            entities.join(", ")
        } else {
            format!("{}, and {} more {}s", entities[..3].join(", "), entities.len() - 3, entity_type)
        }
    };

    // Extract key findings from result summary (format: "Reachable: true, Path length: X")
    let has_path = result.summary.contains("Reachable: true");
    let path_length = result.summary.split("length ").nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<usize>().ok());

    // Extract path cost for LeastResistance
    let path_cost = result.summary.split("cost ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .unwrap_or("unknown");

    // Extract actual source/target from highlighted nodes for path-based analyses
    let actual_source = highlighted_users.first().copied()
        .unwrap_or_else(|| highlighted_nodes.first().map(|s| s.as_str()).unwrap_or(sample_user.as_str()));
    let actual_target = highlighted_resources.last().copied()
        .unwrap_or_else(|| highlighted_nodes.last().map(|s| s.as_str()).unwrap_or(sample_db.as_str()));

    match use_case {
        UseCase::Reachability => UseCaseExplanation {
            business_problem: format!(
                "SECURITY QUESTION: Can {} reach {}?\n\n\
                This answers the fundamental access control question. In enterprise environments, \
                permissions flow through complex chains—users belong to groups, groups have policies, \
                policies grant access to resources.",
                actual_source, actual_target
            ),
            algorithm_description: format!(
                "ALGORITHM: Breadth-First Search (BFS)\n\n\
                BFS explores all permission paths level by level—first checking direct policies, \
                then group memberships, then role assumptions—until it finds the target or exhausts \
                all paths. This graph has {} users to search through.",
                num_users
            ),
            visualization_guide: if has_path {
                format!(
                    "PATH FOUND: {} hops from {} to {}.\n\n\
                    HOW TO READ:\n\
                    • RED NODES form the permission path\n\
                    • Follow: User → Group → Policy → Resource\n\
                    • {} total nodes in the path\n\
                    • Click any node to see its connections",
                    path_length.unwrap_or(0), actual_source, actual_target, num_highlighted
                )
            } else {
                format!(
                    "NO PATH FOUND: {} cannot reach {}.\n\n\
                    The BFS algorithm searched all {} users and found no permission chain \
                    connecting source to target.",
                    actual_source, actual_target, num_users
                )
            },
        },

        UseCase::BlastRadius => UseCaseExplanation {
            business_problem: format!(
                "SECURITY SCENARIO: {}'s credentials are compromised. What's the damage?\n\n\
                Blast radius shows all resources an attacker could reach. Critical for incident \
                response and risk assessment.",
                actual_source
            ),
            algorithm_description: format!(
                "ALGORITHM: BFS with Depth Tracking\n\n\
                Starting from the compromised user ({}), BFS expands outward in waves:\n\
                • Depth 1: Direct connections\n\
                • Depth 2+: Resources via intermediaries\n\
                Lower depth = more immediately accessible to attacker.",
                actual_source
            ),
            visualization_guide: format!(
                "BLAST RADIUS: {} nodes reachable from {}.\n\n\
                AT-RISK RESOURCES: {}\n\n\
                HOW TO READ:\n\
                • RED NODES are within the blast radius\n\
                • Nodes closer to center = more immediate risk\n\
                • Click nodes to see their depth level",
                num_highlighted,
                actual_source,
                if !highlighted_resources.is_empty() {
                    format_entity_list(&highlighted_resources, "resource")
                } else { "See highlighted nodes".to_string() }
            ),
        },

        UseCase::LeastResistance => UseCaseExplanation {
            business_problem: format!(
                "ATTACKER PERSPECTIVE: What's the easiest path from {} to {}?\n\n\
                Edge weights represent 'difficulty': lower = easier to exploit. Direct policy grants \
                (1.0) are easier than role assumptions (2.5).",
                actual_source, actual_target
            ),
            algorithm_description: format!(
                "ALGORITHM: Dijkstra's Shortest Path\n\n\
                Finds the minimum total cost path. Unlike BFS which counts hops, Dijkstra considers \
                edge weights—a 3-hop easy path may be 'cheaper' than a 2-hop hard path."
            ),
            visualization_guide: format!(
                "ATTACK PATH: {} nodes from {} to {}, total cost: {}\n\n\
                HOW TO READ:\n\
                • RED NODES form the optimal attack path\n\
                • EDGE LABELS show difficulty weights\n\
                • Lower total cost = easier attack\n\
                • Click edges to see individual weights",
                num_highlighted, actual_source, actual_target, path_cost
            ),
        },

        UseCase::PrivilegeClustering => UseCaseExplanation {
            business_problem: format!(
                "ROLE MINING: Which users have similar access patterns?\n\n\
                USERS ANALYZED: {}\n\n\
                Users in the same cluster likely perform similar job functions. Useful for \
                creating RBAC roles based on actual behavior.",
                format_entity_list(&highlighted_users, "user")
            ),
            algorithm_description: format!(
                "ALGORITHM: Jaccard Similarity Clustering\n\n\
                For each user pair, compute Jaccard similarity of accessible resources:\n\
                J(A,B) = |A∩B| / |A∪B|\n\
                Users with >50% overlap are clustered together. Analyzed {} users.",
                num_users
            ),
            visualization_guide: format!(
                "CLUSTERS FOUND: {} users grouped by access similarity.\n\n\
                USERS: {}\n\n\
                HOW TO READ:\n\
                • Same-colored nodes = same cluster (similar access)\n\
                • Large clusters = role consolidation candidates\n\
                • Isolated users may have anomalous access",
                num_highlighted,
                format_entity_list(&highlighted_users, "user")
            ),
        },

        UseCase::OverPrivileged => UseCaseExplanation {
            business_problem: format!(
                "LEAST PRIVILEGE CHECK: Which users have excessive access?\n\n\
                Users exceeding the sensitive resource threshold violate least privilege. \
                These accounts are high-risk if compromised."
            ),
            algorithm_description: format!(
                "ALGORITHM: BFS + Sensitivity Counting\n\n\
                For each of {} users:\n\
                1. BFS to find all reachable resources\n\
                2. Count sensitive resources (databases, production instances)\n\
                3. Flag users exceeding threshold (typically 3+)",
                num_users
            ),
            visualization_guide: format!(
                "OVER-PRIVILEGED USERS: {} flagged\n\n\
                USERS: {}\n\n\
                HOW TO READ:\n\
                • RED NODES are over-privileged users\n\
                • Larger nodes = more sensitive resources\n\
                • Click to see exact resource counts",
                num_highlighted,
                format_entity_list(&highlighted_users, "user")
            ),
        },

        UseCase::CrossRegionAccess => UseCaseExplanation {
            business_problem: format!(
                "DATA SOVEREIGNTY: Are there permission paths crossing region boundaries?\n\n\
                GDPR and data residency laws require data to stay within geographic boundaries. \
                Cross-region access may indicate compliance violations."
            ),
            algorithm_description: format!(
                "ALGORITHM: Region-Aware BFS\n\n\
                1. BFS from each user tracking region at each node\n\
                2. Flag when path visits nodes in different regions\n\
                3. Record which region pairs are connected"
            ),
            visualization_guide: format!(
                "CROSS-REGION ACCESS: {} users can access resources across regions.\n\n\
                USERS: {}\n\n\
                HOW TO READ:\n\
                • Look for REGION nodes at graph edges\n\
                • RED PATHS cross region boundaries\n\
                • Click users to see which regions they reach",
                num_highlighted,
                format_entity_list(&highlighted_users, "user")
            ),
        },

        UseCase::UnusedRoles => UseCaseExplanation {
            business_problem: format!(
                "ROLE HYGIENE: Which roles are never assumed?\n\n\
                UNUSED ROLES: {}\n\n\
                These are 'dead' permission paths that increase attack surface \
                without providing value. This graph has {} roles total.",
                format_entity_list(&highlighted_roles, "role"), num_roles
            ),
            algorithm_description: format!(
                "ALGORITHM: Kosaraju's SCC\n\n\
                Finds strongly connected components to identify isolated roles:\n\
                • Roles with no incoming 'Assumes' edges from workloads\n\
                • Roles in small isolated SCCs disconnected from main graph"
            ),
            visualization_guide: format!(
                "UNUSED ROLES: {} out of {} roles\n\n\
                ROLES: {}\n\n\
                HOW TO READ:\n\
                • RED NODES are unused roles\n\
                • No incoming edges from workloads\n\
                • Click to verify no workloads assume them",
                highlighted_roles.len(), num_roles,
                format_entity_list(&highlighted_roles, "role")
            ),
        },

        UseCase::PrivilegeHubs => UseCaseExplanation {
            business_problem: format!(
                "CHOKEPOINT ANALYSIS: Which nodes have unusually high connectivity?\n\n\
                Hubs are single points of failure—compromising one grants access to many \
                resources. Nodes in top 10% by degree are classified as hubs."
            ),
            algorithm_description: format!(
                "ALGORITHM: Degree Analysis\n\n\
                For each node:\n\
                • Count in-degree (incoming edges)\n\
                • Count out-degree (outgoing edges)\n\
                • Total degree = in + out\n\
                • Flag top 10% as privilege hubs"
            ),
            visualization_guide: {
                let policy_info = if !highlighted_policies.is_empty() {
                    format!("Policy hubs: {}\n", format_entity_list(&highlighted_policies, "policy"))
                } else { String::new() };
                let group_info = if !highlighted_groups.is_empty() {
                    format!("Group hubs: {}\n", format_entity_list(&highlighted_groups, "group"))
                } else { String::new() };
                format!(
                    "PRIVILEGE HUBS: {} high-connectivity nodes\n\n\
                    {}{}\n\
                    HOW TO READ:\n\
                    • RED NODES are privilege hubs\n\
                    • Larger nodes = more connections\n\
                    • Click to see in/out degree counts",
                    num_highlighted, policy_info, group_info
                )
            },
        },

        UseCase::MinimalPrivilege => UseCaseExplanation {
            business_problem: format!(
                "PATH EFFICIENCY: Are permission paths truly minimal?\n\n\
                Non-minimal paths have unnecessary intermediaries. Example:\n\
                Actual: User → Group A → Group B → Policy (3 hops)\n\
                Optimal: User → Policy (1 hop)"
            ),
            algorithm_description: format!(
                "ALGORITHM: Dijkstra Comparison\n\n\
                For each user-resource pair:\n\
                1. Compute optimal path via Dijkstra\n\
                2. Trace actual path via BFS\n\
                3. If actual > optimal, path is non-minimal"
            ),
            visualization_guide: {
                let intermediaries = if !highlighted_groups.is_empty() || !highlighted_policies.is_empty() {
                    format!("Intermediaries: {}{}\n",
                        if !highlighted_groups.is_empty() { format!("Groups [{}] ", format_entity_list(&highlighted_groups, "group")) } else { String::new() },
                        if !highlighted_policies.is_empty() { format!("Policies [{}]", format_entity_list(&highlighted_policies, "policy")) } else { String::new() }
                    )
                } else { String::new() };
                format!(
                    "NON-MINIMAL PATHS: {} nodes involved\n\n\
                    {}\n\
                    HOW TO READ:\n\
                    • RED NODES are unnecessary intermediaries\n\
                    • These could be eliminated with direct grants\n\
                    • Click to see why each node is flagged",
                    num_highlighted, intermediaries
                )
            },
        },

        UseCase::AccessibleResources => UseCaseExplanation {
            business_problem: format!(
                "ACCESS INVENTORY: What resources can each user access?\n\n\
                USERS ANALYZED: {}\n\n\
                Computes 'effective permissions' for every user—the complete set of resources \
                reachable through all permission paths. Essential for access certification.",
                format_entity_list(&highlighted_users, "user")
            ),
            algorithm_description: format!(
                "ALGORITHM: DFS Resource Collection\n\n\
                For each of {} users:\n\
                1. DFS from user node\n\
                2. Follow all permission edges\n\
                3. Collect every reachable resource\n\
                4. Compute per-user statistics",
                num_users
            ),
            visualization_guide: format!(
                "ACCESS ANALYSIS: {} users highlighted\n\n\
                USERS: {}\n\n\
                HOW TO READ:\n\
                • NODE ANNOTATIONS show resource counts\n\
                • Click any user to see their resource list\n\
                • Larger numbers = broader access",
                num_highlighted,
                format_entity_list(&highlighted_users, "user")
            ),
        },

        UseCase::HighValueTargets => UseCaseExplanation {
            business_problem: format!(
                "RISK PRIORITIZATION: Which resources are most 'important'?\n\n\
                PageRank identifies resources reachable via many permission paths. These are \
                high-value targets that warrant additional security controls."
            ),
            algorithm_description: format!(
                "ALGORITHM: PageRank\n\n\
                Simulates a random walker traversing the permission graph:\n\
                • Damping factor: 0.85\n\
                • 20 iterations for convergence\n\
                • Higher score = more permission paths lead to this resource"
            ),
            visualization_guide: format!(
                "HIGH-VALUE TARGETS: {} resources with elevated PageRank\n\n\
                RESOURCES: {}\n\n\
                HOW TO READ:\n\
                • RED NODES are high-value targets\n\
                • Larger nodes = higher PageRank score\n\
                • Click to see all paths leading to each target",
                highlighted_resources.len(),
                format_entity_list(&highlighted_resources, "resource")
            ),
        },

        UseCase::MinimumSpanningTree => {
            // Parse MST statistics from result
            let mut total_weight = 0.0;
            let mut mst_edge_count = 0;
            let mut redundant_count = 0;

            for detail in &result.details {
                if detail.starts_with("Total MST weight:") {
                    total_weight = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0.0);
                }
                if detail.starts_with("MST edges:") {
                    mst_edge_count = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                }
                if detail.starts_with("Redundant edges:") {
                    redundant_count = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                }
            }

            UseCaseExplanation {
                business_problem: format!(
                    "PERMISSION OPTIMIZATION: What is the minimal permission infrastructure?\n\n\
                    The Minimum Spanning Tree identifies the essential permission backbone—the smallest \
                    set of edges that maintains full connectivity between all entities.\n\n\
                    RESULTS:\n\
                    • MST edges: {} (essential permissions)\n\
                    • Redundant edges: {} (could potentially be removed)\n\
                    • Total MST weight: {:.2} (permission cost)\n\n\
                    Redundant edges represent attack surface that could be eliminated while \
                    preserving full access capability.",
                    mst_edge_count, redundant_count, total_weight
                ),
                algorithm_description: format!(
                    "ALGORITHM: Kruskal's with Union-Find\n\n\
                    1. Collect all edges with their weights (permission costs)\n\
                    2. Sort edges by weight (lowest first)\n\
                    3. For each edge, add to MST if it doesn't create a cycle\n\
                    4. Stop when all nodes are connected (N-1 edges)\n\n\
                    Edge weights represent 'resistance' in the permission model:\n\
                    • CanAccess: 1.0 (direct permission)\n\
                    • HasPolicy: 1.5 (policy attachment)\n\
                    • MemberOf: 2.0 (group membership)\n\
                    • Assumes: 2.5 (role assumption)\n\
                    • DependsOn: 3.5 (resource dependency)"
                ),
                visualization_guide: format!(
                    "MST ANALYSIS: {} essential edges identified\n\n\
                    ENTITIES: {}\n\n\
                    HOW TO READ:\n\
                    • HIGHLIGHTED edges are in the MST (essential)\n\
                    • GRAY edges are redundant (could be removed)\n\
                    • Lower-weight MST edges are more efficient permission paths\n\n\
                    SECURITY INSIGHTS:\n\
                    • {} redundant edges represent excess attack surface\n\
                    • MST weight {:.2} is the minimum 'permission cost'",
                    mst_edge_count,
                    format_entity_list(&highlighted_users, "user"),
                    redundant_count,
                    total_weight
                ),
            }
        },
    }
}

/// Generate a dynamic explanation for the use case with actual entity names
fn generate_use_case_explanation(
    use_case: UseCase,
    iam_graph: &IamGraph,
    highlighted_nodes: &[String],
    result: &AnalysisResult,
) -> UseCaseExplanation {
    // Build a lookup from node name to node type for highlighted nodes
    let name_to_type: HashMap<String, NodeType> = iam_graph
        .nodes
        .iter()
        .map(|n| (n.name.clone(), n.node_type.clone()))
        .collect();

    // Categorize highlighted nodes by type
    let mut highlighted_users: Vec<&str> = Vec::new();
    let mut highlighted_policies: Vec<&str> = Vec::new();
    let mut highlighted_roles: Vec<&str> = Vec::new();
    let mut highlighted_groups: Vec<&str> = Vec::new();
    let mut highlighted_resources: Vec<&str> = Vec::new(); // DBs, instances, VPCs, etc.

    for node_name in highlighted_nodes {
        if let Some(node_type) = name_to_type.get(node_name) {
            match node_type {
                NodeType::User => highlighted_users.push(node_name),
                NodeType::Policy => highlighted_policies.push(node_name),
                NodeType::Role => highlighted_roles.push(node_name),
                NodeType::Group => highlighted_groups.push(node_name),
                _ => highlighted_resources.push(node_name),
            }
        }
    }

    // Get sample entity names for context (fallback for cases where no highlighted entities exist)
    let sample_user = iam_graph.nodes.iter()
        .find(|n| matches!(n.node_type, NodeType::User))
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "user-0000".to_string());

    let sample_db = iam_graph.nodes.iter()
        .find(|n| matches!(n.node_type, NodeType::Database))
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "database-0000".to_string());

    let sample_role = iam_graph.nodes.iter()
        .find(|n| matches!(n.node_type, NodeType::Role))
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "role-0000".to_string());

    let sample_policy = iam_graph.nodes.iter()
        .find(|n| matches!(n.node_type, NodeType::Policy))
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "policy-0000".to_string());

    let sample_vpc = iam_graph.nodes.iter()
        .find(|n| matches!(n.node_type, NodeType::Vpc))
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "vpc-0000".to_string());

    let num_users = iam_graph.users.len();
    let num_resources = iam_graph.stats.total_nodes() - num_users - iam_graph.groups.len() - iam_graph.policies.len() - iam_graph.roles.len();
    let num_highlighted = highlighted_nodes.len();

    // Helper to format a list of names (max 3 shown)
    let format_entity_list = |entities: &[&str], entity_type: &str| -> String {
        if entities.is_empty() {
            format!("no {}s", entity_type)
        } else if entities.len() == 1 {
            entities[0].to_string()
        } else if entities.len() <= 3 {
            entities.join(", ")
        } else {
            format!("{}, and {} more {}s", entities[..3].join(", "), entities.len() - 3, entity_type)
        }
    };

    // Extract key findings from result summary for visualization context (format: "Reachable: true, Path length: X")
    let has_path = result.summary.contains("Reachable: true");
    let path_length = if let Some(captures) = result.summary.split("length ").nth(1) {
        captures.split_whitespace().next().and_then(|s| s.parse::<usize>().ok())
    } else {
        None
    };

    match use_case {
        UseCase::Reachability => {
            // Extract actual source and target from the path (first user and last resource in highlighted nodes)
            let actual_source = highlighted_users.first().copied()
                .unwrap_or_else(|| highlighted_nodes.first().map(|s| s.as_str()).unwrap_or(&sample_user));
            let actual_target = highlighted_resources.last().copied()
                .unwrap_or_else(|| highlighted_nodes.last().map(|s| s.as_str()).unwrap_or(&sample_db));

            UseCaseExplanation {
                business_problem: format!(
                    "SECURITY QUESTION: Can {} reach {}?\n\n\
                    This answers the fundamental access control question: 'Does this user have permission \
                    to access this resource?' In enterprise environments, permissions often flow through \
                    complex chains—users belong to groups, groups have policies, policies grant access to \
                    resources. A user may have access they (or security teams) don't realize.\n\n\
                    REAL-WORLD APPLICATIONS:\n\
                    • Access Reviews: Auditors ask 'Can user X access database Y?' during SOC 2/ISO 27001 audits\n\
                    • Incident Response: When credentials are stolen, quickly determine what's exposed\n\
                    • Permission Troubleshooting: User reports 'access denied'—trace why the path is blocked\n\
                    • Compliance: GDPR/HIPAA require knowing exactly who can access sensitive data",
                    actual_source, actual_target
                ),
                algorithm_description: format!(
                    "ALGORITHM: Breadth-First Search (BFS)\n\n\
                    Think of BFS like water flowing through pipes. Starting from {}, the algorithm:\n\n\
                    1. INITIALIZE: Add the source user to a queue\n\
                    2. EXPLORE: For each node in the queue, check all outgoing permission edges\n\
                    3. EXPAND: Add newly discovered nodes to the queue (if not already visited)\n\
                    4. REPEAT: Continue until target is found or queue is empty\n\n\
                    WHY BFS? It guarantees finding the SHORTEST path (fewest hops) first. In a graph \
                    with {} users and {} resources ({} total nodes), BFS is efficient—O(V+E) time \
                    complexity where V=nodes and E=edges.\n\n\
                    EDGE TYPES TRAVERSED: MemberOf (user→group), HasPolicy (group→policy), \
                    CanAccess (policy→resource), Assumes (workload→role), LocatedIn (resource→region)",
                    actual_source, num_users, num_resources, iam_graph.stats.total_nodes()
                ),
                visualization_guide: if has_path {
                    let path_info = path_length.map(|l| format!("PATH FOUND: {} hops from {} to {}.\n\n", l, actual_source, actual_target)).unwrap_or_default();
                    format!(
                        "{}READING THE VISUALIZATION:\n\n\
                        • RED HIGHLIGHTED NODES: These form the permission path. Each red node is a 'hop' \
                        in the access chain from user to resource.\n\n\
                        • PATH STRUCTURE: Source [{}] → {}{}{}\n\n\
                        • TOTAL NODES IN PATH: {} — fewer hops generally means more direct (and potentially \
                        riskier) access.\n\n\
                        WHAT TO LOOK FOR:\n\
                        • Follow the chain: User → Group → Policy → Resource (typical pattern)\n\
                        • Click any node to see ALL its connections—you may discover additional paths\n\
                        • Look for 'bridge' nodes that appear in many paths—these are permission chokepoints",
                        path_info,
                        actual_source,
                        if !highlighted_groups.is_empty() { format!("Groups [{}] → ", format_entity_list(&highlighted_groups, "group")) } else { String::new() },
                        if !highlighted_policies.is_empty() { format!("Policies [{}] → ", format_entity_list(&highlighted_policies, "policy")) } else { String::new() },
                        if !highlighted_resources.is_empty() { format!("Target [{}]", format_entity_list(&highlighted_resources, "resource")) } else { "Target resource".to_string() },
                        num_highlighted
                    )
                } else {
                    format!(
                        "NO PATH FOUND: {} cannot reach {}.\n\n\
                        WHAT THIS MEANS:\n\
                        • The user has NO permission chain to the resource through any combination of \
                        groups, policies, or roles in this graph.\n\
                        • The BFS algorithm exhaustively searched all {} users, {} policies, and {} resources.\n\n\
                        POSSIBLE REASONS:\n\
                        • User is not a member of any group with access\n\
                        • No policy grants access to this specific resource\n\
                        • Access may exist through a different identity (service account, role assumption)\n\n\
                        NEXT STEPS: If the user SHOULD have access, check group memberships and policy attachments.",
                        actual_source, actual_target, num_users, iam_graph.policies.len(), num_resources
                    )
                },
            }
        },

        UseCase::BlastRadius => {
            // Extract depth information from result details
            let max_depth = result.details.iter()
                .find(|d| d.contains("depth") || d.contains("Depth"))
                .and_then(|d| d.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse::<usize>().ok())
                .unwrap_or(5);

            // Get the actual source user from highlighted nodes
            let actual_source = highlighted_users.first().copied()
                .unwrap_or_else(|| highlighted_nodes.first().map(|s| s.as_str()).unwrap_or(&sample_user));

            UseCaseExplanation {
                business_problem: format!(
                    "SECURITY SCENARIO: {}'s credentials have been compromised. What's the damage?\n\n\
                    Blast radius analysis answers the critical incident response question: 'How bad is this breach?' \
                    When attackers gain access to one identity, they can traverse permission chains to reach \
                    far more resources than that identity directly owns.\n\n\
                    WHY THIS MATTERS:\n\
                    • INCIDENT PRIORITIZATION: A breach of a user with blast radius of 5 resources is less \
                    severe than one reaching 500 resources\n\
                    • BREACH CONTAINMENT: Knowing the blast radius helps determine which credentials to \
                    rotate and which systems to isolate\n\
                    • RISK ASSESSMENT: Users with large blast radii are high-value targets for attackers\n\n\
                    INDUSTRY CONTEXT: The 2020 SolarWinds attack showed how one compromised identity can \
                    cascade through permission chains to thousands of systems.",
                    actual_source
                ),
                algorithm_description: format!(
                    "ALGORITHM: Breadth-First Search with Depth Tracking\n\n\
                    Imagine dropping ink into water—it spreads outward in waves. This algorithm works similarly:\n\n\
                    1. DEPTH 0: Start at the compromised user ({})\n\
                    2. DEPTH 1: Find all directly connected nodes (groups, direct policies)\n\
                    3. DEPTH 2: From those nodes, find the next layer (resources via policies)\n\
                    4. DEPTH N: Continue expanding until all reachable nodes are found\n\n\
                    DEPTH SIGNIFICANCE:\n\
                    • Depth 1-2: Immediate risk—attacker can access these within seconds\n\
                    • Depth 3-4: Secondary risk—requires additional privilege escalation steps\n\
                    • Depth 5+: Extended risk—complex attack chains, but still reachable\n\n\
                    TIME COMPLEXITY: O(V+E) where V = nodes reachable, E = edges traversed. Even with \
                    thousands of nodes, this completes in milliseconds.",
                    actual_source
                ),
                visualization_guide: {
                    let resource_count = highlighted_resources.len();
                    let resource_breakdown = if !highlighted_resources.is_empty() {
                        format!("AT-RISK RESOURCES IDENTIFIED: {}\n\n", format_entity_list(&highlighted_resources, "resource"))
                    } else {
                        String::new()
                    };
                    format!(
                        "{}BLAST RADIUS SUMMARY: {} total nodes reachable from compromised user.\n\n\
                        READING THE VISUALIZATION:\n\
                        • RED NODES: Every red node is within the blast radius—an attacker with this \
                        user's credentials could potentially access or modify these resources\n\
                        • NODE PROXIMITY: Nodes closer to the center (source user) are more immediately \
                        accessible; outer nodes require more 'hops'\n\
                        • MAXIMUM DEPTH: {} hops to reach the furthest resources\n\n\
                        KEY INSIGHTS:\n\
                        {}\
                        • Resource count: {} resources at risk (databases, instances, VPCs, etc.)\n\
                        • Policy paths: {}\n\n\
                        REMEDIATION PRIORITY:\n\
                        1. Resources at depth 1-2 need immediate attention\n\
                        2. Consider whether this user truly needs all these permissions\n\
                        3. Look for overly-permissive policies that expand the blast radius unnecessarily",
                        resource_breakdown,
                        num_highlighted,
                        max_depth,
                        if !highlighted_groups.is_empty() {
                            format!("• Access flows through groups: {}\n", format_entity_list(&highlighted_groups, "group"))
                        } else { String::new() },
                        resource_count,
                        if !highlighted_policies.is_empty() {
                            format_entity_list(&highlighted_policies, "policy")
                        } else {
                            "Direct access paths".to_string()
                        }
                    )
                },
            }
        },

        UseCase::LeastResistance => {
            // Extract path cost from summary if available (format: "cost X.X")
            let path_cost = result.summary.split("cost ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .unwrap_or("unknown");

            // Get actual source and target from highlighted path
            let actual_source = highlighted_users.first().copied()
                .unwrap_or_else(|| highlighted_nodes.first().map(|s| s.as_str()).unwrap_or(&sample_user));
            let actual_target = highlighted_resources.last().copied()
                .unwrap_or_else(|| highlighted_nodes.last().map(|s| s.as_str()).unwrap_or(&sample_db));

            UseCaseExplanation {
                business_problem: format!(
                    "ATTACKER'S PERSPECTIVE: What's the easiest path from {} to {}?\n\n\
                    Sophisticated attackers don't randomly explore—they find the path of least resistance. \
                    Just as water flows downhill, attackers exploit the easiest permission chains first. \
                    This analysis thinks like an attacker to identify your most vulnerable access paths.\n\n\
                    SECURITY IMPLICATIONS:\n\
                    • ATTACK SURFACE PRIORITIZATION: Not all paths are equal—some are easier to exploit\n\
                    • DEFENSE IN DEPTH: Adding 'friction' (MFA, approvals) to easy paths forces attackers \
                    to use harder routes\n\
                    • RED TEAM PLANNING: This is exactly what penetration testers look for\n\n\
                    EDGE WEIGHT MODEL (lower = easier to exploit):\n\
                    • Direct policy grant: 1.0 (trivial access)\n\
                    • Group membership: 1.5 (common pattern)\n\
                    • Cross-region access: 2.0 (some friction)\n\
                    • Role assumption: 2.5 (requires additional step)\n\
                    • Admin escalation: 3.0+ (harder but possible)",
                    actual_source, actual_target
                ),
                algorithm_description: format!(
                    "ALGORITHM: Dijkstra's Shortest Path\n\n\
                    Named after Edsger Dijkstra (1956), this algorithm finds the minimum-cost path between \
                    two nodes in a weighted graph. It's the same algorithm GPS systems use for navigation.\n\n\
                    HOW IT WORKS:\n\
                    1. INITIALIZE: Set distance to source = 0, all others = infinity\n\
                    2. SELECT: Pick the unvisited node with smallest distance\n\
                    3. UPDATE: For each neighbor, if (current distance + edge weight) < known distance, update\n\
                    4. MARK: Mark current node as visited\n\
                    5. REPEAT: Until target is reached or all nodes visited\n\n\
                    WHY DIJKSTRA (NOT BFS)?\n\
                    BFS finds the path with fewest hops, but ignores edge weights. A 3-hop path through \
                    easy edges (cost 1+1+1=3) is easier to exploit than a 2-hop path through hard edges \
                    (cost 2.5+2.5=5). Dijkstra considers both.\n\n\
                    TIME COMPLEXITY: O((V+E) log V) with a priority queue implementation.",
                ),
                visualization_guide: {
                    let path_description = if !highlighted_nodes.is_empty() {
                        let entities: Vec<&str> = highlighted_nodes.iter().take(6).map(|s| s.as_str()).collect();
                        if highlighted_nodes.len() > 6 {
                            format!("ATTACK PATH: {} → ... → target\n\n", entities.join(" → "))
                        } else {
                            format!("ATTACK PATH: {}\n\n", entities.join(" → "))
                        }
                    } else {
                        String::new()
                    };
                    format!(
                        "{}READING THE VISUALIZATION:\n\n\
                        • TOTAL PATH COST: {} — this is the 'difficulty score' for this attack path. \
                        Lower scores = easier attacks.\n\
                        • HIGHLIGHTED NODES: {} nodes form the optimal attack path from {} to {}{}{}\n\
                        • EDGE LABELS: Each edge shows its weight (difficulty). Look for edges with \
                        weight 1.0-1.5 as these are the easiest to exploit.\n\n\
                        WHAT TO LOOK FOR:\n\
                        • 'Cheap' edges: Low-weight connections that make paths easy\n\
                        • Chokepoints: Required nodes that appear in many attack paths\n\
                        • Bypass opportunities: Are there alternative paths with higher cost?",
                        path_description,
                        path_cost,
                        num_highlighted,
                        actual_source,
                        actual_target,
                        if !highlighted_groups.is_empty() { format!(", via groups [{}]", format_entity_list(&highlighted_groups, "group")) } else { String::new() },
                        if !highlighted_policies.is_empty() { format!(", through policies [{}]", format_entity_list(&highlighted_policies, "policy")) } else { String::new() }
                    )
                },
            }
        },

        UseCase::PrivilegeClustering => {
            // Extract cluster count from result details if available
            let cluster_count = result.details.iter()
                .find(|d| d.contains("cluster"))
                .and_then(|d| d.split_whitespace().find(|s| s.parse::<usize>().is_ok()))
                .unwrap_or("multiple");

            // Count users per cluster from details
            let avg_cluster_size = if !highlighted_users.is_empty() && cluster_count.parse::<usize>().unwrap_or(1) > 0 {
                highlighted_users.len() / cluster_count.parse::<usize>().unwrap_or(1)
            } else { 0 };

            // Get example users from the results
            let example_users = format_entity_list(&highlighted_users, "user");

            UseCaseExplanation {
                business_problem: format!(
                    "ROLE MINING QUESTION: Which users have similar access patterns?\n\n\
                    USERS ANALYZED: {}\n\n\
                    In mature organizations, permissions accumulate organically—users get added to groups, \
                    policies get attached ad-hoc. Over time, no one knows who SHOULD have similar access. \
                    Privilege clustering reveals the natural groupings hidden in your permission data.\n\n\
                    BUSINESS VALUE:\n\
                    • ROLE MINING: Discover roles from actual behavior rather than inventing them top-down\n\
                    • ANOMALY DETECTION: Users who don't cluster with anyone may have unusual access\n\
                    • ACCESS CERTIFICATION: Instead of reviewing users individually, review clusters",
                    example_users
                ),
                algorithm_description: format!(
                    "ALGORITHM: Jaccard Similarity Clustering\n\n\
                    Think of each user as having a 'permission fingerprint'—the set of all resources they \
                    can access. Users with similar fingerprints get clustered together.\n\n\
                    STEP-BY-STEP:\n\
                    1. COMPUTE FINGERPRINTS: For each of {} users, BFS finds all reachable resources → Set(resources)\n\
                    2. COMPARE PAIRS: For every pair of users (A, B), calculate Jaccard similarity:\n\
                       J(A,B) = |A ∩ B| / |A ∪ B|\n\
                       • J = 1.0: Identical access (same fingerprint)\n\
                       • J = 0.5: Half their permissions overlap\n\
                       • J = 0.0: No shared permissions\n\
                    3. CLUSTER: Users with J > 0.5 (50%+ overlap) are grouped together\n\n\
                    EXAMPLE: If user A can access {{db1, db2, vpc1}} and user B can access {{db1, db2, instance1}}:\n\
                    • Intersection: {{db1, db2}} (2 resources)\n\
                    • Union: {{db1, db2, vpc1, instance1}} (4 resources)\n\
                    • Jaccard: 2/4 = 0.5 → They would be clustered together\n\n\
                    TIME COMPLEXITY: O(n² × m) where n = users, m = avg resources per user.",
                    num_users
                ),
                visualization_guide: format!(
                    "CLUSTER ANALYSIS RESULTS:\n\
                    • Total users analyzed: {}\n\
                    • Clusters identified: {}\n\
                    • Average cluster size: ~{} users per cluster\n\n\
                    HIGHLIGHTED USERS: {}\n\n\
                    READING THE VISUALIZATION:\n\
                    • COLORED GROUPS: Users with the same color belong to the same cluster (similar permissions)\n\
                    • CLUSTER SIZE: Large clusters = many users with similar access (potential role candidates)\n\
                    • ISOLATED NODES: Users not in any cluster may have unique or anomalous access patterns\n\n\
                    WHAT TO LOOK FOR:\n\
                    • Large clusters (5+ users): Strong candidates for formalized RBAC roles\n\
                    • Singleton clusters: Users with unique access—verify this is intentional\n\
                    • Near-matches: Users almost in a cluster may have unnecessary extra permissions\n\n\
                    ACTIONABLE INSIGHTS:\n\
                    1. Create roles based on largest clusters (highest ROI)\n\
                    2. Investigate isolated users for potential access policy violations\n\
                    3. Look for clusters that should logically be one (merge candidates)",
                    num_users,
                    cluster_count,
                    avg_cluster_size,
                    format_entity_list(&highlighted_users, "user")
                ),
            }
        },

        UseCase::OverPrivileged => {
            // Extract threshold info from result summary
            let threshold = result.summary.split("with access to ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(3);

            // Find the max sensitive resource count from details
            let max_resources = result.details.iter()
                .filter_map(|d| d.split("can access ").nth(1))
                .filter_map(|s| s.split_whitespace().next())
                .filter_map(|s| s.parse::<usize>().ok())
                .max()
                .unwrap_or(0);

            // Get the top over-privileged user from results
            let top_offender = highlighted_users.first().copied().unwrap_or("(none found)");

            UseCaseExplanation {
                business_problem: format!(
                    "LEAST PRIVILEGE VIOLATION: Which users have excessive access?\n\n\
                    TOP OFFENDER: {} (and {} others flagged)\n\n\
                    The Principle of Least Privilege (PoLP) states users should have only the minimum \
                    permissions necessary for their job. Over-privileged accounts are a top attack vector—\
                    if breached, they grant attackers access to far more than necessary.\n\n\
                    WHY THIS MATTERS:\n\
                    • BREACH AMPLIFICATION: A compromised over-privileged account = massive blast radius\n\
                    • INSIDER THREAT: Employees with unnecessary access can accidentally or maliciously \
                    access sensitive data\n\
                    • COMPLIANCE FAILURES: PCI-DSS 7.1, SOC 2 CC6.1, and HIPAA all require least privilege\n\n\
                    DETECTION THRESHOLD: Users with access to {}+ sensitive resources are flagged.",
                    top_offender, highlighted_users.len().saturating_sub(1), threshold
                ),
                algorithm_description: format!(
                    "ALGORITHM: BFS + Sensitive Resource Counting\n\n\
                    This is a two-phase analysis that identifies users with excessive privilege.\n\n\
                    PHASE 1 - PERMISSION EXPANSION:\n\
                    For each of {} users, perform BFS to find ALL reachable resources. This expands \
                    implicit permissions (via groups, roles, transitive policies) into explicit access.\n\n\
                    PHASE 2 - SENSITIVITY SCORING:\n\
                    Count how many reachable resources are classified as 'sensitive':\n\
                    • Databases: Always sensitive (contain data)\n\
                    • Production instances: Sensitive (service disruption risk)\n\
                    • Admin consoles: Sensitive (control plane access)\n\n\
                    FLAGGING CRITERIA:\n\
                    • Threshold: {} or more sensitive resources = over-privileged\n\
                    • Ranking: Users sorted by sensitive resource count (worst offenders first)\n\n\
                    TIME COMPLEXITY: O(n × (V+E)) where n = users, V = nodes, E = edges.",
                    num_users, threshold
                ),
                visualization_guide: {
                    let top_offenders = if !highlighted_users.is_empty() {
                        format!("TOP OVER-PRIVILEGED USERS: {}\n\n", format_entity_list(&highlighted_users, "user"))
                    } else {
                        String::new()
                    };
                    format!(
                        "{}OVER-PRIVILEGE SUMMARY:\n\
                        • Users flagged: {} (out of {} total)\n\
                        • Threshold: {}+ sensitive resources\n\
                        • Maximum access found: {} sensitive resources by a single user\n\n\
                        READING THE VISUALIZATION:\n\
                        • RED HIGHLIGHTED USERS: These users exceed the sensitivity threshold\n\
                        • NODE SIZE: Larger nodes = more sensitive resources accessible\n\
                        • CONNECTED RESOURCES: Follow edges to see WHY they have access\n\n\
                        INVESTIGATION WORKFLOW:\n\
                        1. Click the most over-privileged user (largest/reddest node)\n\
                        2. Trace their permission paths to sensitive resources\n\
                        3. Identify which group membership or policy grants unnecessary access\n\
                        4. Document and plan remediation\n\n\
                        REMEDIATION PRIORITY:\n\
                        • CRITICAL: Users with 10+ sensitive resources\n\
                        • HIGH: Users with 5-9 sensitive resources\n\
                        • MEDIUM: Users with {}-4 sensitive resources",
                        top_offenders,
                        highlighted_users.len(),
                        num_users,
                        threshold,
                        max_resources,
                        threshold
                    )
                },
            }
        },

        UseCase::CrossRegionAccess => {
            // Extract region names from highlighted nodes
            let regions: Vec<&str> = highlighted_nodes.iter()
                .filter(|n| name_to_type.get(*n).map(|t| matches!(t, NodeType::Region)).unwrap_or(false))
                .map(|s| s.as_str())
                .collect();

            let sample_region = iam_graph.nodes.iter()
                .find(|n| matches!(n.node_type, NodeType::Region))
                .map(|n| n.name.clone())
                .unwrap_or_else(|| "region-us-east".to_string());

            UseCaseExplanation {
                business_problem: format!(
                    "DATA SOVEREIGNTY CHECK: Are there permission paths crossing region boundaries?\n\n\
                    Many regulations (GDPR, data residency laws) require data to stay within specific \
                    geographic boundaries. This analysis finds users who can access resources in multiple \
                    regions—potential compliance violations.\n\n\
                    REGIONS IN THIS GRAPH: Resources are tagged with their region (e.g., {}). \
                    Cross-region access occurs when a user in one region can reach resources in another.",
                    sample_region
                ),
                algorithm_description: format!(
                    "ALGORITHM: Region-Aware BFS\n\n\
                    1. For each user, perform BFS traversal through the permission graph\n\
                    2. Track the region attribute at each node visited\n\
                    3. When a path visits nodes in different regions, flag the user\n\
                    4. Record which region pairs are connected (e.g., us-east → eu-west)\n\n\
                    The algorithm considers both direct cross-region edges and indirect paths \
                    through region-spanning policies or groups.",
                ),
                visualization_guide: {
                    let region_info = if !regions.is_empty() {
                        format!("REGIONS INVOLVED: {}\n\n", regions.join(", "))
                    } else {
                        String::new()
                    };
                    format!(
                        "{}CROSS-REGION ACCESS FOUND: {} users can access resources across region boundaries.\n\n\
                        USERS WITH CROSS-REGION ACCESS:\n{}\n\n\
                        HOW TO READ THE VISUALIZATION:\n\
                        • Look for REGION nodes (labeled 'region-*') at the edges of the graph\n\
                        • RED HIGHLIGHTED paths show cross-region access chains\n\
                        • Follow the path: User → Group/Policy → Resource → Region\n\
                        • If a single user's paths lead to multiple Region nodes, they have cross-region access\n\n\
                        INTERPRETING THE RESULTS:\n\
                        • Click any highlighted user to see which regions they can reach\n\
                        • The path shows HOW they gain cross-region access (which policy/group enables it)\n\
                        • Resources in the path reveal WHAT data could flow across regions",
                        region_info,
                        highlighted_users.len(),
                        format_entity_list(&highlighted_users, "user")
                    )
                },
            }
        },

        UseCase::UnusedRoles => {
            let total_roles = iam_graph.roles.len();
            let unused_count = highlighted_roles.len();
            let usage_percentage = if total_roles > 0 {
                ((total_roles - unused_count) as f64 / total_roles as f64 * 100.0) as usize
            } else { 100 };

            // Get actual unused roles from results
            let unused_roles_list = format_entity_list(&highlighted_roles, "role");

            UseCaseExplanation {
                business_problem: format!(
                    "ROLE HYGIENE: Which roles are never assumed by any workload?\n\n\
                    UNUSED ROLES FOUND: {}\n\n\
                    These represent 'dead' permission paths—they exist but serve no purpose. \
                    However, attackers could potentially assume these roles to gain access. Identifying \
                    and removing unused roles reduces your attack surface.\n\n\
                    This graph contains {} roles total.",
                    unused_roles_list, total_roles
                ),
                algorithm_description: format!(
                    "ALGORITHM: Kosaraju's Strongly Connected Components (SCC)\n\n\
                    This algorithm finds groups of nodes that can all reach each other. Roles that \
                    are isolated (not in the main connected component) are flagged.\n\n\
                    TWO-PASS DFS:\n\
                    1. PASS 1: DFS on original graph, record finish order of nodes\n\
                    2. PASS 2: DFS on reversed graph in reverse finish order\n\
                    3. Each DFS tree in pass 2 = one strongly connected component\n\n\
                    Roles flagged if: (a) no incoming 'Assumes' edges from workloads, OR \
                    (b) in a small isolated SCC disconnected from main permission graph.",
                ),
                visualization_guide: format!(
                    "UNUSED ROLES FOUND: {} out of {} roles ({}% of roles are in use)\n\n\
                    UNUSED ROLES LIST:\n{}\n\n\
                    HOW TO READ THE VISUALIZATION:\n\
                    • RED HIGHLIGHTED nodes are unused roles\n\
                    • These nodes have NO incoming edges from workload nodes\n\
                    • They appear 'orphaned' or disconnected from the main graph structure\n\n\
                    VERIFYING THE RESULTS:\n\
                    • Click any red role node to inspect its connections\n\
                    • Check for incoming 'Assumes' edges—unused roles will have none\n\
                    • Look at outgoing edges to see what access the role WOULD grant if assumed\n\n\
                    WHY THEY'RE UNUSED:\n\
                    • Role was created but never assigned to a workload\n\
                    • Workload that used the role was deleted\n\
                    • Role was replaced by a different permission structure",
                    unused_count,
                    total_roles,
                    usage_percentage,
                    if !highlighted_roles.is_empty() {
                        format_entity_list(&highlighted_roles, "role")
                    } else {
                        "None found - all roles are in use".to_string()
                    }
                ),
            }
        },

        UseCase::PrivilegeHubs => {
            // Extract degree information from result details
            let max_degree = result.details.iter()
                .filter_map(|d| d.split("degree ").nth(1))
                .filter_map(|s| s.split_whitespace().next())
                .filter_map(|s| s.trim_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().ok())
                .max()
                .unwrap_or(0);

            // Build hub summary from actual results
            let hub_summary = {
                let mut parts = Vec::new();
                if !highlighted_policies.is_empty() {
                    parts.push(format!("Policies: {}", format_entity_list(&highlighted_policies, "policy")));
                }
                if !highlighted_groups.is_empty() {
                    parts.push(format!("Groups: {}", format_entity_list(&highlighted_groups, "group")));
                }
                if !highlighted_roles.is_empty() {
                    parts.push(format!("Roles: {}", format_entity_list(&highlighted_roles, "role")));
                }
                if parts.is_empty() { "None found".to_string() } else { parts.join("\n") }
            };

            UseCaseExplanation {
                business_problem: format!(
                    "CHOKEPOINT ANALYSIS: Which nodes have unusually high connectivity?\n\n\
                    PRIVILEGE HUBS IDENTIFIED:\n{}\n\n\
                    These nodes have many incoming or outgoing permission edges. \
                    They act as 'chokepoints'—compromising one hub can grant access to many resources, \
                    and removing one hub disrupts many users.\n\n\
                    WHAT MAKES A HUB:\n\
                    • HIGH IN-DEGREE: Many edges pointing IN (e.g., a group with many members)\n\
                    • HIGH OUT-DEGREE: Many edges pointing OUT (e.g., a policy granting access to many resources)\n\
                    • Nodes in the top 10% by total degree (in + out) are classified as hubs",
                    hub_summary
                ),
                algorithm_description: format!(
                    "ALGORITHM: Degree Analysis\n\n\
                    For each node in the graph:\n\
                    1. Count incoming edges (in-degree)\n\
                    2. Count outgoing edges (out-degree)\n\
                    3. Calculate total degree = in-degree + out-degree\n\
                    4. Rank all nodes by total degree\n\
                    5. Flag top 10% as privilege hubs\n\n\
                    INTERPRETATION BY NODE TYPE:\n\
                    • Policy with high out-degree → grants access to many resources\n\
                    • Group with high in-degree → has many user members\n\
                    • Role with high in-degree → assumed by many workloads",
                ),
                visualization_guide: {
                    let policy_hubs = if !highlighted_policies.is_empty() {
                        format!("POLICY HUBS: {}\n", format_entity_list(&highlighted_policies, "policy"))
                    } else { String::new() };
                    let group_hubs = if !highlighted_groups.is_empty() {
                        format!("GROUP HUBS: {}\n", format_entity_list(&highlighted_groups, "group"))
                    } else { String::new() };
                    let role_hubs = if !highlighted_roles.is_empty() {
                        format!("ROLE HUBS: {}\n", format_entity_list(&highlighted_roles, "role"))
                    } else { String::new() };

                    format!(
                        "PRIVILEGE HUBS FOUND: {} high-connectivity nodes (max degree: {})\n\n\
                        {}{}{}\n\
                        HOW TO READ THE VISUALIZATION:\n\
                        • RED HIGHLIGHTED nodes are privilege hubs\n\
                        • NODE SIZE indicates connectivity—larger nodes have more connections\n\
                        • EDGES radiating from hubs show their reach\n\n\
                        WHAT TO LOOK FOR:\n\
                        • Count the edges coming into/out of each hub\n\
                        • Policies with many outgoing edges = broad permission grants\n\
                        • Groups with many incoming edges = large membership\n\n\
                        CLICKING A HUB reveals:\n\
                        • Exact in-degree and out-degree counts\n\
                        • All connected nodes (users, resources, etc.)\n\
                        • The 'reach' of this hub in the permission graph",
                        num_highlighted,
                        max_degree,
                        policy_hubs,
                        group_hubs,
                        role_hubs
                    )
                },
            }
        },

        UseCase::MinimalPrivilege => {
            // Extract efficiency info from results
            let non_minimal_count = result.details.iter()
                .filter(|d| d.contains("non-minimal") || d.contains("inefficient") || d.contains("suboptimal"))
                .count();

            // Get actual nodes involved in non-minimal paths
            let involved_nodes = if !highlighted_users.is_empty() || !highlighted_resources.is_empty() {
                format!("NODES INVOLVED: Users [{}], Resources [{}]",
                    format_entity_list(&highlighted_users, "user"),
                    format_entity_list(&highlighted_resources, "resource"))
            } else {
                format!("{} nodes involved in non-minimal paths", num_highlighted)
            };

            UseCaseExplanation {
                business_problem: format!(
                    "PATH EFFICIENCY: Are permission paths truly minimal?\n\n\
                    {}\n\n\
                    Over time, permission structures accumulate unnecessary complexity. Users may have \
                    access through multiple redundant paths, or paths may have extra 'hops' that add \
                    no value. Non-minimal paths indicate permission debt that should be simplified.\n\n\
                    EXAMPLE OF NON-MINIMAL PATH:\n\
                    • Actual: User → Group A → Group B → Policy → Resource (4 hops)\n\
                    • Optimal: User → Policy → Resource (2 hops)",
                    involved_nodes
                ),
                algorithm_description: format!(
                    "ALGORITHM: Dijkstra Comparison\n\n\
                    For each user-resource pair with existing access:\n\
                    1. COMPUTE OPTIMAL: Use Dijkstra to find the minimum-cost path\n\
                    2. TRACE ACTUAL: Use BFS to find the actual path being used\n\
                    3. COMPARE: If actual_cost > optimal_cost, the path is non-minimal\n\n\
                    A path is non-minimal if:\n\
                    • It has more hops than necessary\n\
                    • It goes through unnecessary intermediaries\n\
                    • A more direct route exists but isn't being used",
                ),
                visualization_guide: {
                    let intermediaries = if !highlighted_groups.is_empty() || !highlighted_policies.is_empty() {
                        format!("UNNECESSARY INTERMEDIARIES FOUND:\n{}{}\n\n",
                            if !highlighted_groups.is_empty() { format!("• Groups: {}\n", format_entity_list(&highlighted_groups, "group")) } else { String::new() },
                            if !highlighted_policies.is_empty() { format!("• Policies: {}", format_entity_list(&highlighted_policies, "policy")) } else { String::new() }
                        )
                    } else { String::new() };

                    format!(
                        "NON-MINIMAL PATHS FOUND: {} paths with unnecessary complexity\n\
                        NODES INVOLVED: {}\n\n\
                        {}\
                        HOW TO READ THE VISUALIZATION:\n\
                        • RED HIGHLIGHTED nodes are part of non-minimal paths\n\
                        • These nodes represent 'extra hops' that could be eliminated\n\
                        • The path shown is the ACTUAL path, not the optimal one\n\n\
                        INTERPRETING THE RESULTS:\n\
                        • Click a highlighted node to see why it's flagged\n\
                        • Compare the actual path to what a direct grant would look like\n\
                        • Nodes appearing in multiple non-minimal paths are consolidation candidates\n\n\
                        WHAT CAUSES NON-MINIMAL PATHS:\n\
                        • Nested group memberships (Group → Group → Policy)\n\
                        • Redundant policy attachments\n\
                        • Legacy permission structures that weren't cleaned up",
                        non_minimal_count,
                        num_highlighted,
                        intermediaries
                    )
                },
            }
        },

        UseCase::AccessibleResources => {
            // Extract statistics from result details
            let most_accessed = result.details.iter()
                .find(|d| d.contains("most accessed") || d.contains("Most accessed"))
                .map(|s| s.as_str());
            let broadest_user = result.details.iter()
                .find(|d| d.contains("broadest access") || d.contains("Broadest access") || d.contains("highest access"))
                .map(|s| s.as_str());

            // Try to extract average resources per user
            let avg_resources = result.details.iter()
                .find(|d| d.contains("average") || d.contains("avg"))
                .and_then(|d| d.split_whitespace().find(|s| s.parse::<f64>().is_ok()))
                .and_then(|s| s.parse::<f64>().ok());

            // Get users from results
            let analyzed_users = format_entity_list(&highlighted_users, "user");

            UseCaseExplanation {
                business_problem: format!(
                    "ACCESS INVENTORY: What resources can each user access?\n\n\
                    USERS ANALYZED: {}\n\n\
                    This analysis computes the 'effective permissions' for every user—the complete set \
                    of resources they can reach through all permission paths. This is essential for:\n\n\
                    • ACCESS CERTIFICATION: Quarterly reviews require knowing exactly who can access what\n\
                    • AUDIT PREPARATION: Auditors ask 'show me all access for user X'\n\
                    • PERMISSIONS DRIFT: Compare actual vs. intended access to find discrepancies\n\n\
                    GRAPH ANALYZED: {} users, {} resources",
                    analyzed_users, num_users, num_resources
                ),
                algorithm_description: format!(
                    "ALGORITHM: DFS Resource Collection\n\n\
                    For each user:\n\
                    1. Start DFS (Depth-First Search) from the user node\n\
                    2. Follow all outgoing permission edges (MemberOf, HasPolicy, CanAccess)\n\
                    3. Collect every resource node (Database, Instance, VPC, etc.) reached\n\
                    4. Build a Set of accessible resources for that user\n\n\
                    AGGREGATED STATISTICS:\n\
                    • Per-user resource counts\n\
                    • Most-accessed resources (reached by most users)\n\
                    • Users with broadest access (reach most resources)\n\n\
                    TIME COMPLEXITY: O(n × (V+E)) where n = {} users.",
                    num_users
                ),
                visualization_guide: {
                    let key_findings = format!(
                        "KEY FINDINGS:\n{}{}{}\n",
                        if let Some(most) = most_accessed { format!("• {}\n", most) } else { String::new() },
                        if let Some(broadest) = broadest_user { format!("• {}\n", broadest) } else { String::new() },
                        if let Some(avg) = avg_resources { format!("• Average: {:.1} resources per user\n", avg) } else { String::new() }
                    );

                    format!(
                        "ACCESS ANALYSIS COMPLETE: {} users analyzed\n\n\
                        USERS HIGHLIGHTED: {}\n\n\
                        {}\
                        HOW TO READ THE VISUALIZATION:\n\
                        • HIGHLIGHTED USERS show those with notable access patterns\n\
                        • NODE ANNOTATIONS display resource counts for each user\n\
                        • EDGES show the permission paths to resources\n\n\
                        EXPLORING THE RESULTS:\n\
                        • Click any user to see their complete resource list\n\
                        • Users with larger node annotations have broader access\n\
                        • Follow edges from a user to see HOW they access each resource\n\n\
                        COMPARING USERS:\n\
                        • Users with similar resource counts may have similar roles\n\
                        • Outliers (very high or very low counts) deserve investigation",
                        highlighted_users.len(),
                        format_entity_list(&highlighted_users, "user"),
                        key_findings
                    )
                },
            }
        },

        UseCase::HighValueTargets => {
            // Extract PageRank scores from result details
            let top_scores: Vec<&str> = result.details.iter()
                .filter(|d| d.contains("PageRank") || d.contains("score") || d.contains(":"))
                .take(5)
                .map(|s| s.as_str())
                .collect();

            // Categorize resources by type
            let databases: Vec<&str> = highlighted_resources.iter()
                .filter(|r| r.contains("database"))
                .copied()
                .collect();
            let vpcs: Vec<&str> = highlighted_resources.iter()
                .filter(|r| r.contains("vpc"))
                .copied()
                .collect();
            let instances: Vec<&str> = highlighted_resources.iter()
                .filter(|r| r.contains("instance"))
                .copied()
                .collect();

            // Build summary of high-value targets from results
            let targets_summary = format_entity_list(&highlighted_resources, "resource");

            UseCaseExplanation {
                business_problem: format!(
                    "RISK PRIORITIZATION: Which resources are most 'important'?\n\n\
                    HIGH-VALUE TARGETS IDENTIFIED: {}\n\n\
                    Not all resources are equal. Some are reachable by many permission paths, making \
                    them high-value targets for attackers. This analysis uses PageRank—the algorithm \
                    Google uses to rank web pages—to identify the most 'important' nodes in your \
                    permission graph.\n\n\
                    HIGH PAGERANK = HIGH RISK: Resources with many incoming permission paths are:\n\
                    • More likely to be accessed (intentionally or not)\n\
                    • Higher impact if compromised\n\
                    • Candidates for additional security controls",
                    targets_summary
                ),
                algorithm_description: format!(
                    "ALGORITHM: PageRank\n\n\
                    PageRank simulates a 'random walker' traversing the permission graph. The more \
                    paths lead to a node, the more likely the walker ends up there.\n\n\
                    PARAMETERS:\n\
                    • Damping factor: 0.85 (85% chance of following an edge, 15% chance of jumping)\n\
                    • Iterations: 20 (enough for convergence on most graphs)\n\n\
                    HOW IT WORKS:\n\
                    1. Initialize all nodes with equal rank (1/N)\n\
                    2. Each iteration: redistribute rank along edges\n\
                    3. Nodes receiving rank from many sources accumulate higher scores\n\
                    4. After convergence, rank reflects 'importance' in permission flow\n\n\
                    WHY PAGERANK (NOT DEGREE): Degree counts direct connections. PageRank considers \
                    the entire permission chain—a resource reachable via many policies attached to \
                    large groups scores higher than one with a single direct connection.",
                ),
                visualization_guide: {
                    let resource_breakdown = format!(
                        "HIGH-VALUE TARGETS BY TYPE:\n{}{}{}\n",
                        if !databases.is_empty() { format!("• Databases: {}\n", format_entity_list(&databases, "database")) } else { String::new() },
                        if !vpcs.is_empty() { format!("• VPCs: {}\n", format_entity_list(&vpcs, "vpc")) } else { String::new() },
                        if !instances.is_empty() { format!("• Instances: {}\n", format_entity_list(&instances, "instance")) } else { String::new() }
                    );

                    let score_info = if !top_scores.is_empty() {
                        format!("TOP PAGERANK SCORES:\n{}\n\n", top_scores.join("\n"))
                    } else { String::new() };

                    format!(
                        "HIGH-VALUE TARGETS FOUND: {} resources with elevated PageRank scores\n\n\
                        {}\
                        {}\
                        HOW TO READ THE VISUALIZATION:\n\
                        • RED HIGHLIGHTED nodes are high-value targets\n\
                        • NODE SIZE reflects PageRank score—larger = more important\n\
                        • INCOMING EDGES show the permission paths leading to each target\n\n\
                        INTERPRETING PAGERANK SCORES:\n\
                        • Scores are relative (higher = more important than others)\n\
                        • A score of 0.01 means ~1% of permission flow reaches this node\n\
                        • Compare resources of the same type (e.g., database vs. database)\n\n\
                        EXPLORING THE GRAPH:\n\
                        • Click a high-value target to see all paths leading to it\n\
                        • Trace backwards: which users can ultimately reach this resource?\n\
                        • Look for shared paths—multiple high-value targets reachable via one policy",
                        highlighted_resources.len(),
                        resource_breakdown,
                        score_info
                    )
                },
            }
        },

        UseCase::MinimumSpanningTree => {
            // Parse MST statistics from result
            let mut total_weight = 0.0;
            let mut mst_edge_count = 0;
            let mut redundant_count = 0;

            for detail in &result.details {
                if detail.starts_with("Total MST weight:") {
                    total_weight = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0.0);
                }
                if detail.starts_with("MST edges:") {
                    mst_edge_count = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                }
                if detail.starts_with("Redundant edges:") {
                    redundant_count = detail.split(':').nth(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                }
            }

            UseCaseExplanation {
                business_problem: format!(
                    "PERMISSION OPTIMIZATION: What is the minimal permission infrastructure?\n\n\
                    The Minimum Spanning Tree identifies the essential permission backbone—the smallest \
                    set of edges that maintains full connectivity between all entities.\n\n\
                    RESULTS:\n\
                    • MST edges: {} (essential permissions)\n\
                    • Redundant edges: {} (could potentially be removed)\n\
                    • Total MST weight: {:.2} (permission cost)\n\n\
                    Redundant edges represent attack surface that could be eliminated while \
                    preserving full access capability.",
                    mst_edge_count, redundant_count, total_weight
                ),
                algorithm_description: format!(
                    "ALGORITHM: Kruskal's with Union-Find\n\n\
                    1. Collect all edges with their weights (permission costs)\n\
                    2. Sort edges by weight (lowest first)\n\
                    3. For each edge, add to MST if it doesn't create a cycle\n\
                    4. Stop when all nodes are connected (N-1 edges)\n\n\
                    Edge weights represent 'resistance' in the permission model:\n\
                    • CanAccess: 1.0 (direct permission)\n\
                    • HasPolicy: 1.5 (policy attachment)\n\
                    • MemberOf: 2.0 (group membership)\n\
                    • Assumes: 2.5 (role assumption)\n\
                    • DependsOn: 3.5 (resource dependency)"
                ),
                visualization_guide: format!(
                    "MST ANALYSIS: {} essential edges identified\n\n\
                    HOW TO READ:\n\
                    • HIGHLIGHTED edges are in the MST (essential)\n\
                    • GRAY edges are redundant (could be removed)\n\
                    • Lower-weight MST edges are more efficient permission paths\n\n\
                    SECURITY INSIGHTS:\n\
                    • {} redundant edges represent excess attack surface\n\
                    • MST weight {:.2} is the minimum 'permission cost'\n\n\
                    EXPLORING THE GRAPH:\n\
                    • Click on MST edges to see the permission type\n\
                    • Trace the MST to understand the core permission structure\n\
                    • Redundant edges are candidates for removal during permission cleanup",
                    mst_edge_count,
                    redundant_count,
                    total_weight
                ),
            }
        },
    }
}

/// Create visualization result with overlay data for motlie_db implementation
fn create_vis_result_motlie(
    result: &AnalysisResult,
    use_case: UseCase,
    iam_graph: &IamGraph,
    _id_to_name: &HashMap<Id, String>,
    source_node: Option<&str>,
    target_node: Option<&str>,
) -> VisAnalysisResult {
    // For motlie implementation, we use the same parsing logic as reference
    // since the result format is the same
    let _id_to_name: HashMap<Id, String> = iam_graph
        .nodes
        .iter()
        .map(|n| (n.id, n.name.clone()))
        .collect();

    // Reuse the same parsing logic
    let dummy_graph: DiGraph<String, f64> = DiGraph::new();
    let dummy_id_to_idx: HashMap<Id, NodeIndex> = HashMap::new();
    let dummy_idx_to_id: HashMap<NodeIndex, Id> = HashMap::new();

    create_vis_result(result, use_case, iam_graph, &dummy_graph, &dummy_id_to_idx, &dummy_idx_to_id, source_node, target_node)
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn print_usage(program: &str) {
    eprintln!("IAM Permissions Graph Analysis - Interactive Web UI");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {} <db_path> <scale>                       Start HTTP server with interactive UI", program);
    eprintln!("  {} --generate <db_path> <scale>            Generate graph and write to RocksDB only", program);
    eprintln!("  {} list                                    List available use cases", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --generate                                 Build graph and write to RocksDB, then exit");
    eprintln!("  --port <PORT>                              HTTP server port (default: 8081)");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  db_path    - Path to RocksDB directory");
    eprintln!("  scale      - Graph scale factor (10-1000 recommended)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} /tmp/iam_db 50                          # Start server on default port", program);
    eprintln!("  {} /tmp/iam_db 100 --port 9000             # Start server on custom port", program);
    eprintln!("  {} --generate /tmp/iam_db 100              # Generate graph data only", program);
    eprintln!("  {} list                                    # Show available use cases", program);
}

fn list_use_cases() {
    println!("\n=== Available Use Cases ===\n");
    for (i, use_case) in UseCase::all().iter().enumerate() {
        println!("{}. {} ({})", i + 1, use_case.name(), match use_case {
            UseCase::Reachability => "reachability",
            UseCase::BlastRadius => "blast_radius",
            UseCase::LeastResistance => "least_resistance",
            UseCase::PrivilegeClustering => "privilege_clustering",
            UseCase::OverPrivileged => "over_privileged",
            UseCase::CrossRegionAccess => "cross_region",
            UseCase::UnusedRoles => "unused_roles",
            UseCase::PrivilegeHubs => "privilege_hubs",
            UseCase::MinimalPrivilege => "minimal_privilege",
            UseCase::AccessibleResources => "accessible_resources",
            UseCase::HighValueTargets => "high_value_targets",
            UseCase::MinimumSpanningTree => "mst",
        });
        println!("   Algorithm: {}", use_case.algorithm());
        println!("   {}", use_case.description());
        println!();
    }
}

/// Run the interactive visualization mode where use cases are selected in the UI
async fn run_server(server_opts: ServerOptions, db_path: &Path, scale: usize) -> Result<()> {

    // Step 1: Generate IAM graph
    println!("Generating IAM graph with scale factor {}...", scale);
    let iam_graph = IamGraph::generate(scale);
    iam_graph.stats.print();

    // Step 2: Extract visualization data BEFORE writing to disk
    println!("\nExtracting visualization data...");
    let (vis_nodes, vis_edges) = iam_graph_to_vis(&iam_graph);

    // Step 3: Extract lightweight metadata for analysis (before we consume iam_graph)
    println!("Extracting graph metadata...");
    let metadata = GraphMetadata::from_iam_graph(&iam_graph);

    // Step 4: Write graph to RocksDB and record build time
    println!("Writing graph to RocksDB at {:?}...", db_path);
    let disk_build_start = std::time::Instant::now();
    let (nodes, edges) = iam_graph.to_graph_nodes_edges();
    let (_reader, _name_to_id, _handle) = build_graph(db_path, nodes, edges).await?;
    let disk_build_time_ms = disk_build_start.elapsed().as_secs_f64() * 1000.0;

    // Get disk metrics
    let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));
    println!("\n=== RocksDB Build Complete ===");
    println!("  Build time: {:.2} ms", disk_build_time_ms);
    println!("  Files: {}", disk_files);
    println!("  Size:  {:.2} KB ({} bytes)", disk_size as f64 / 1024.0, disk_size);

    // Step 5: Clear heavy in-memory structures (iam_graph is consumed by to_graph_nodes_edges)
    // The IamGraph is now gone - we only have lightweight metadata

    // Step 6: Create visualization state with lightweight context
    let state = VisualizationState::new();

    // Populate graph data for visualization
    {
        let mut nodes = state.nodes.write().await;
        *nodes = vis_nodes;
    }
    {
        let mut edges = state.edges.write().await;
        *edges = vis_edges;
    }

    // Create and store the lightweight graph context
    let graph_context = GraphContext {
        metadata,
        db_path: db_path.to_path_buf(),
        disk_build_time_ms,
    };
    {
        let mut ctx = state.graph_context.write().await;
        *ctx = Some(graph_context);
    }

    // Start the visualization server
    let server_state = state.clone();
    let port = server_opts.port;

    println!("\n========================================");
    println!("  Interactive IAM Analysis UI");
    println!("========================================");
    println!("\nOpen in browser: http://localhost:{}/viz", port);
    println!("\nGraph data is persisted in RocksDB. Both implementations");
    println!("will read from disk - no in-memory graph is retained.");
    println!("\nSelect a use case from the dropdown, configure inputs,");
    println!("and click RUN to execute the analysis.");
    println!("\nPress Ctrl+C to exit.\n");

    // Run the server (blocking)
    start_visualization_server(port, server_state).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse options first
    let (generate_only, server_opts, args) = parse_options(&args);

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    // Handle "list" command
    if args[1] == "list" {
        list_use_cases();
        return Ok(());
    }

    // Require db_path and scale
    if args.len() < 3 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale: usize = args[2]
        .parse()
        .context("Scale must be a positive integer")?;

    // Handle --generate mode - generate RocksDB data only
    if generate_only {
        // Generate IAM graph
        println!("Generating IAM graph with scale factor {}...", scale);
        let iam_graph = IamGraph::generate(scale);
        iam_graph.stats.print();

        // Build motlie_db graph
        println!("\nWriting graph to RocksDB at {:?}...", db_path);
        let (nodes, edges) = iam_graph.to_graph_nodes_edges();
        let (_reader, _name_to_id, _handle) = build_graph(db_path, nodes, edges).await?;

        // Get disk metrics
        let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));
        println!("\n=== RocksDB Metrics ===");
        println!("  Files: {}", disk_files);
        println!("  Size:  {:.2} KB ({} bytes)", disk_size as f64 / 1024.0, disk_size);
        println!("\nGraph data generated successfully at {:?}", db_path);

        return Ok(());
    }

    // Default: start HTTP server with interactive UI
    run_server(server_opts, db_path, scale).await
}
