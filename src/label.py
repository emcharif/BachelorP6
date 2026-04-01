import glob
import os
import torch
import pandas as pd

def compute_graph_features(graph):
    ptr = graph.vehicle.ptr
    x = graph.vehicle.x
    edge_attr = graph[('vehicle', 'to', 'vehicle')].edge_attr
    edge_index = graph[('vehicle', 'to', 'vehicle')].edge_index

    speeds_per_t = []
    distances_per_t = []
    rel_speeds_per_t = []

    for t in range(len(ptr) - 1):
        start = ptr[t].item()
        end = ptr[t + 1].item()
        if end <= start:
            continue

        # mean speed at this timestep
        nodes = x[start:end, 0:2]
        speed = nodes.norm(dim=1).mean().item()
        speeds_per_t.append(speed)

        # edges within this timestep
        mask = (edge_index[0] >= start) & (edge_index[0] < end)
        if mask.sum() > 0:
            t_edges = edge_attr[mask]
            dist     = t_edges[:, 3].mean().item()
            rel_v    = t_edges[:, 4:6].norm(dim=1).mean().item()
            distances_per_t.append(dist)
            rel_speeds_per_t.append(rel_v)

    mean_speed     = sum(speeds_per_t) / len(speeds_per_t)         if speeds_per_t     else 0.0
    mean_distance  = sum(distances_per_t) / len(distances_per_t)   if distances_per_t  else 0.0
    mean_rel_speed = sum(rel_speeds_per_t) / len(rel_speeds_per_t) if rel_speeds_per_t else 0.0

    return mean_speed, mean_distance, mean_rel_speed


# First pass — collect all values to understand distributions
files = glob.glob("data/training_dataset/*.pt") + glob.glob("data/predict_dataset/*.pt")
print(f"Analysing {len(files)} graphs...")

rows = []
for f in files:
    g = torch.load(f, weights_only=False)
    mean_speed, mean_distance, mean_rel_speed = compute_graph_features(g)
    rows.append({
        'path':           os.path.basename(f),
        'filename':       os.path.basename(f),
        'mean_speed':     mean_speed,
        'mean_distance':  mean_distance,
        'mean_rel_speed': mean_rel_speed,
    })

df = pd.DataFrame(rows)

print(f"\n=== Feature Statistics ===")
print(df[['mean_speed', 'mean_distance', 'mean_rel_speed']].describe().round(4).to_string())

# Assign composite labels
def assign_label(row):
    speed  = row['mean_speed']
    dist   = row['mean_distance']
    rel_v  = row['mean_rel_speed']

 # free flow: above median speed AND not too close
    if speed > 1.13:                          # above 75th percentile
        return 1

    # transitional: middle speed OR close vehicles with notable relative speed
    if speed > 0.72 and rel_v > 0.87:        # above 25th speed, above median rel speed
        return 2

    # congested: slow moving, low interaction
    return 0

df['label'] = df.apply(assign_label, axis=1)

print(f"\n=== Label Distribution ===")
counts = df['label'].value_counts().sort_index()
names  = {0: 'Congested', 1: 'Free flow', 2: 'Transitional'}
for label, count in counts.items():
    print(f"  {label} ({names[label]}): {count} graphs ({count/len(df)*100:.1f}%)")

df.to_csv('labels_composite_3class.csv', index=False)
print(f"\nSaved to labels_composite_3class.csv")