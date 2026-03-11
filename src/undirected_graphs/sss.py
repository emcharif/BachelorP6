from torch_geometric.datasets import TUDataset
import torch

# Choose dataset
DATASET_NAME = 'REDDIT-BINARY'  # Change this to try different datasets

print("="*70)
print(f"LOADING {DATASET_NAME} DATASET")
print("="*70)

dataset = TUDataset(root=f'data/{DATASET_NAME}', name=DATASET_NAME)

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_node_features}")
print(f"Number of edge features: {dataset.num_edge_features}")

print("\n" + "="*70)
print("GRAPH SIZE STATISTICS")
print("="*70)

num_nodes_list = [data.num_nodes for data in dataset]
num_edges_list = [data.num_edges for data in dataset]

print(f"Number of nodes:")
print(f"  Min: {min(num_nodes_list)}")
print(f"  Max: {max(num_nodes_list)}")
print(f"  Mean: {sum(num_nodes_list) / len(num_nodes_list):.2f}")

print(f"\nNumber of edges:")
print(f"  Min: {min(num_edges_list)}")
print(f"  Max: {max(num_edges_list)}")
print(f"  Mean: {sum(num_edges_list) / len(num_edges_list):.2f}")

print("\n" + "="*70)
print("CLASS DISTRIBUTION")
print("="*70)

labels = [data.y.item() for data in dataset]
unique_labels = set(labels)

for label in sorted(unique_labels):
    count = labels.count(label)
    print(f"  Class {label}: {count} graphs ({100*count/len(dataset):.1f}%)")

print("\n" + "="*70)
print("SAMPLE GRAPHS")
print("="*70)

for i in range(min(5, len(dataset))):
    graph = dataset[i]
    print(f"\nGraph {i}:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Label: {graph.y.item()}")
    
    if graph.x is not None:
        print(f"  Node features shape: {graph.x.shape}")
    else:
        print(f"  Node features: None (use node degrees)")
    
    if i == 0:
        print(f"\n  First 10 edges:")
        for j in range(min(10, graph.edge_index.shape[1])):
            src = graph.edge_index[0, j].item()
            dst = graph.edge_index[1, j].item()
            print(f"    Edge {j}: {src} → {dst}")

print("\n" + "="*70)
print("CHECK IF DIRECTED (Sample 10 graphs)")
print("="*70)

directed_counts = []

for i in range(min(10, len(dataset))):
    graph = dataset[i]
    edge_index = graph.edge_index
    num_edges = edge_index.shape[1]
    
    # Create edge set
    edge_set = set()
    for j in range(num_edges):
        src = edge_index[0, j].item()
        dst = edge_index[1, j].item()
        edge_set.add((src, dst))
    
    # Count bidirectional
    bidirectional = 0
    for j in range(num_edges):
        src = edge_index[0, j].item()
        dst = edge_index[1, j].item()
        if (dst, src) in edge_set:
            bidirectional += 1
    
    directed_pct = 100 * (1 - bidirectional / num_edges) if num_edges > 0 else 0
    directed_counts.append(directed_pct)
    
    print(f"Graph {i}: {bidirectional}/{num_edges} bidirectional ({100*bidirectional/num_edges:.1f}%) → {directed_pct:.1f}% directed")

avg_directed = sum(directed_counts) / len(directed_counts)

if avg_directed > 10:
    print(f"\n→ Dataset is DIRECTED (avg {avg_directed:.1f}% unidirectional edges) ✓")
else:
    print(f"\n→ Dataset is UNDIRECTED (avg {avg_directed:.1f}% unidirectional edges)")

print("\n" + "="*70)
print("DEGREE STATISTICS (First graph)")
print("="*70)

graph = dataset[0]
edge_index = graph.edge_index

out_degree = torch.zeros(graph.num_nodes, dtype=torch.long)
in_degree = torch.zeros(graph.num_nodes, dtype=torch.long)

for i in range(edge_index.shape[1]):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    out_degree[src] += 1
    in_degree[dst] += 1

print(f"Out-degree:")
print(f"  Mean: {out_degree.float().mean():.2f}")
print(f"  Max: {out_degree.max().item()}")
print(f"  Min: {out_degree.min().item()}")

print(f"\nIn-degree:")
print(f"  Mean: {in_degree.float().mean():.2f}")
print(f"  Max: {in_degree.max().item()}")
print(f"  Min: {in_degree.min().item()}")

# Check if in-degree and out-degree are different (sign of directedness)
degree_diff = (out_degree != in_degree).sum().item()
print(f"\nNodes with different in/out degree: {degree_diff}/{graph.num_nodes} ({100*degree_diff/graph.num_nodes:.1f}%)")

if degree_diff > graph.num_nodes * 0.5:
    print("→ Strong evidence of directed edges ✓")
else:
    print("→ Likely undirected (in-degree ≈ out-degree)")

print("\n" + "="*70)
print("ESTIMATED DATASET SIZE")
print("="*70)

import os

# Try to estimate size
try:
    dataset_path = f'data/{DATASET_NAME}'
    if os.path.exists(dataset_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        print(f"Dataset size on disk: {size_mb:.2f} MB")
    else:
        print("Dataset not yet downloaded")
except Exception as e:
    print(f"Could not calculate size: {e}")