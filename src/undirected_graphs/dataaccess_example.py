from torch_geometric.datasets import TUDataset
import torch

# Load dataset
dataset = TUDataset(root='data/REDDIT-BINARY', name='REDDIT-BINARY')

print("="*70)
print("ACCESSING DATASET CONTENTS")
print("="*70)

# ============================================================================
# 1. DATASET LEVEL ACCESS
# ============================================================================

print("\n1. DATASET LEVEL:")
print("-" * 70)
print(f"Total number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_node_features}")
print(f"Number of edge features: {dataset.num_edge_features}")

# Access individual graph by index
graph_0 = dataset[0]
graph_5 = dataset[5]
graph_100 = dataset[100]

print(f"\nAccessing graphs:")
print(f"  dataset[0] = {graph_0}")
print(f"  dataset[5] = {graph_5}")
print(f"  dataset[100] = {graph_100}")

# Slice dataset
subset = dataset[:10]  # First 10 graphs
print(f"\nDataset slicing:")
print(f"  dataset[:10] = {len(subset)} graphs")

# ============================================================================
# 2. GRAPH LEVEL ACCESS
# ============================================================================

print("\n" + "="*70)
print("2. GRAPH LEVEL ACCESS (Individual Graph)")
print("="*70)

graph = dataset[0]

print(f"\nGraph attributes:")
print(f"  graph.num_nodes = {graph.num_nodes}")
print(f"  graph.num_edges = {graph.num_edges}")
print(f"  graph.y = {graph.y}  (label)")

# Check what attributes exist
print(f"\nAll available attributes:")
for key in graph.keys():
    value = graph[key]
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape} ({value.dtype})")
    else:
        print(f"  {key}: {value}")

# ============================================================================
# 3. NODE ACCESS
# ============================================================================

print("\n" + "="*70)
print("3. NODE LEVEL ACCESS")
print("="*70)

graph = dataset[0]

print(f"\nTotal nodes: {graph.num_nodes}")

# Node features (if they exist)
if graph.x is not None:
    print(f"\nNode features (graph.x):")
    print(f"  Shape: {graph.x.shape}")
    print(f"  Type: {graph.x.dtype}")
    print(f"\n  First 5 nodes features:")
    for i in range(min(5, graph.num_nodes)):
        print(f"    Node {i}: {graph.x[i]}")
else:
    print(f"\nNode features: None")
    print(f"  (REDDIT-BINARY has no node features)")
    print(f"  (Typically you'd use node degrees as features)")

# Calculate node degrees
print(f"\nNode degrees:")
edge_index = graph.edge_index

# Out-degree (how many edges go OUT from each node)
out_degree = torch.zeros(graph.num_nodes, dtype=torch.long)
for i in range(edge_index.shape[1]):
    src = edge_index[0, i].item()
    out_degree[src] += 1

# In-degree (how many edges come IN to each node)
in_degree = torch.zeros(graph.num_nodes, dtype=torch.long)
for i in range(edge_index.shape[1]):
    dst = edge_index[1, i].item()
    in_degree[dst] += 1

print(f"\n  First 10 nodes (degree info):")
for i in range(min(10, graph.num_nodes)):
    print(f"    Node {i}: out_degree={out_degree[i].item()}, in_degree={in_degree[i].item()}, total={out_degree[i].item() + in_degree[i].item()}")

# Find specific types of nodes
leaves = [i for i in range(graph.num_nodes) if (out_degree[i] + in_degree[i]) == 1]
hubs = [i for i in range(graph.num_nodes) if (out_degree[i] + in_degree[i]) >= 10]

print(f"\n  Special nodes:")
print(f"    Leaf nodes (degree=1): {len(leaves)} nodes")
print(f"      Examples: {leaves[:5]}")
print(f"    Hub nodes (degree>=10): {len(hubs)} nodes")
print(f"      Examples: {hubs[:5]}")

# ============================================================================
# 4. EDGE ACCESS
# ============================================================================

print("\n" + "="*70)
print("4. EDGE LEVEL ACCESS")
print("="*70)

edge_index = graph.edge_index

print(f"\nTotal edges: {graph.num_edges}")
print(f"\nEdge index (graph.edge_index):")
print(f"  Shape: {edge_index.shape}")
print(f"  Format: [2, num_edges]")
print(f"    Row 0: Source nodes")
print(f"    Row 1: Destination nodes")

print(f"\n  First 10 edges:")
print(f"  {'Edge':<8} {'Source':<10} {'Destination'}")
print(f"  {'-'*35}")
for i in range(min(10, edge_index.shape[1])):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    print(f"  {i:<8} {src:<10} {dst}")

# Access specific edges
print(f"\n  Accessing specific edges:")
edge_5_src = edge_index[0, 5].item()
edge_5_dst = edge_index[1, 5].item()
print(f"    Edge 5: {edge_5_src} → {edge_5_dst}")

edge_10_src = edge_index[0, 10].item()
edge_10_dst = edge_index[1, 10].item()
print(f"    Edge 10: {edge_10_src} → {edge_10_dst}")

# Find neighbors of a specific node
node_id = 5
print(f"\n  Finding neighbors of node {node_id}:")

# Outgoing neighbors (node_id → neighbor)
outgoing_mask = edge_index[0] == node_id
outgoing_neighbors = edge_index[1][outgoing_mask].tolist()
print(f"    Outgoing neighbors: {outgoing_neighbors}")

# Incoming neighbors (neighbor → node_id)
incoming_mask = edge_index[1] == node_id
incoming_neighbors = edge_index[0][incoming_mask].tolist()
print(f"    Incoming neighbors: {incoming_neighbors}")

# All neighbors (combined)
all_neighbors = list(set(outgoing_neighbors + incoming_neighbors))
print(f"    All neighbors: {all_neighbors}")

# Edge attributes (if they exist)
if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
    print(f"\n  Edge attributes (graph.edge_attr):")
    print(f"    Shape: {graph.edge_attr.shape}")
    print(f"    First 5 edges:")
    for i in range(min(5, graph.edge_attr.shape[0])):
        print(f"      Edge {i}: {graph.edge_attr[i]}")
else:
    print(f"\n  Edge attributes: None")

# ============================================================================
# 5. ITERATING THROUGH DATASET
# ============================================================================

print("\n" + "="*70)
print("5. ITERATING THROUGH DATASET")
print("="*70)

print(f"\nMethod 1: Simple loop")
for i in range(min(3, len(dataset))):
    graph = dataset[i]
    print(f"  Graph {i}: {graph.num_nodes} nodes, {graph.num_edges} edges, label={graph.y.item()}")

print(f"\nMethod 2: Enumerate")
for i, graph in enumerate(dataset[:3]):
    print(f"  Graph {i}: {graph.num_nodes} nodes, {graph.num_edges} edges, label={graph.y.item()}")

print(f"\nMethod 3: Filter graphs by criteria")
large_graphs = [graph for graph in dataset if graph.num_nodes > 500]
print(f"  Graphs with >500 nodes: {len(large_graphs)}")

small_graphs = [graph for graph in dataset if graph.num_nodes < 100]
print(f"  Graphs with <100 nodes: {len(small_graphs)}")

# ============================================================================
# 6. CONVERTING TO OTHER FORMATS
# ============================================================================

print("\n" + "="*70)
print("6. CONVERTING TO OTHER FORMATS")
print("="*70)

graph = dataset[0]

# Convert to NetworkX
import networkx as nx

print(f"\nConverting to NetworkX:")

G = nx.Graph()
G.add_nodes_from(range(graph.num_nodes))

edge_index = graph.edge_index
edges = [(edge_index[0, i].item(), edge_index[1, i].item()) 
         for i in range(edge_index.shape[1])]
G.add_edges_from(edges)

print(f"  NetworkX graph: {G}")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Get node degree from NetworkX
print(f"\n  Node degrees (via NetworkX):")
for node, degree in list(G.degree())[:5]:
    print(f"    Node {node}: degree={degree}")

# Find connected components
components = list(nx.connected_components(G))
print(f"\n  Connected components: {len(components)}")
print(f"  Largest component size: {len(max(components, key=len))}")

# ============================================================================
# 7. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "="*70)
print("7. PRACTICAL EXAMPLES")
print("="*70)

graph = dataset[0]

# Example 1: Find all paths from node 0
print(f"\nExample 1: Find neighbors within 2 hops of node 0")

def get_k_hop_neighbors(edge_index, node_id, k=2):
    """Get all neighbors within k hops."""
    neighbors = {node_id}
    current_frontier = {node_id}
    
    for hop in range(k):
        next_frontier = set()
        for node in current_frontier:
            # Find neighbors
            out_neighbors = edge_index[1][edge_index[0] == node].tolist()
            in_neighbors = edge_index[0][edge_index[1] == node].tolist()
            next_frontier.update(out_neighbors + in_neighbors)
        
        neighbors.update(next_frontier)
        current_frontier = next_frontier - neighbors
    
    return neighbors

neighbors_2hop = get_k_hop_neighbors(graph.edge_index, node_id=0, k=2)
print(f"  Node 0 has {len(neighbors_2hop)} neighbors within 2 hops")
print(f"  Neighbors: {sorted(list(neighbors_2hop))[:10]}...")

# Example 2: Find subgraph around a node
print(f"\nExample 2: Extract subgraph around node 5")

center_node = 5
subgraph_nodes = get_k_hop_neighbors(graph.edge_index, center_node, k=1)

# Extract edges within subgraph
subgraph_edges = []
for i in range(graph.edge_index.shape[1]):
    src = graph.edge_index[0, i].item()
    dst = graph.edge_index[1, i].item()
    if src in subgraph_nodes and dst in subgraph_nodes:
        subgraph_edges.append((src, dst))

print(f"  Subgraph nodes: {len(subgraph_nodes)}")
print(f"  Subgraph edges: {len(subgraph_edges)}")

# Example 3: Graph statistics
print(f"\nExample 3: Calculate graph statistics")

degrees = out_degree + in_degree
print(f"  Average degree: {degrees.float().mean().item():.2f}")
print(f"  Max degree: {degrees.max().item()}")
print(f"  Min degree: {degrees.min().item()}")
print(f"  Degree variance: {degrees.float().var().item():.2f}")

# Clustering coefficient (using NetworkX)
clustering = nx.average_clustering(G)
print(f"  Average clustering coefficient: {clustering:.4f}")

# Density
density = nx.density(G)
print(f"  Graph density: {density:.4f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Key ways to access data:

1. Dataset level:
   - dataset[i] → Get graph i
   - len(dataset) → Number of graphs
   
2. Graph level:
   - graph.num_nodes → Number of nodes
   - graph.num_edges → Number of edges
   - graph.y → Graph label
   - graph.x → Node features (if available)
   - graph.edge_index → Edge connectivity
   
3. Node level:
   - graph.x[i] → Features of node i (if available)
   - Calculate degrees from edge_index
   
4. Edge level:
   - graph.edge_index[0, j] → Source of edge j
   - graph.edge_index[1, j] → Destination of edge j
   - graph.edge_attr[j] → Attributes of edge j (if available)
   
5. Neighbors:
   - edge_index[1][edge_index[0] == node_id] → Outgoing neighbors
   - edge_index[0][edge_index[1] == node_id] → Incoming neighbors
""")