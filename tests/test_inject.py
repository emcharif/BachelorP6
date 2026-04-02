import sys
from pathlib import Path

# Make project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import glob
import torch

from src.inject import inject_chain


def get_first_pt_file():
    files = sorted(glob.glob("data/training_dataset/*.pt"))
    if not files:
        raise FileNotFoundError("No .pt files found in data/training_dataset")
    return files[0]


def summarize_graph(graph, label="Graph"):
    v_store = graph["vehicle"]
    s_store = graph["vehicle", "to", "vehicle"]

    print(f"\n=== {label} Summary ===")
    print(f"vehicle.x shape:           {tuple(v_store.x.shape)}")
    print(f"vehicle.ptr shape:         {tuple(v_store.ptr.shape)}")
    print(f"spatial edge_index shape:  {tuple(s_store.edge_index.shape)}")
    print(f"spatial edge_attr shape:   {tuple(s_store.edge_attr.shape)}")

    if hasattr(s_store, "distance"):
        print(f"spatial distance shape:    {tuple(s_store.distance.shape)}")

    if hasattr(v_store, "id"):
        print(f"vehicle.id shape:          {tuple(v_store.id.shape)}")
    if hasattr(v_store, "pos"):
        print(f"vehicle.pos shape:         {tuple(v_store.pos.shape)}")
    if hasattr(v_store, "orientation"):
        print(f"vehicle.orientation shape: {tuple(v_store.orientation.shape)}")
    if hasattr(v_store, "is_ego_mask"):
        print(f"vehicle.is_ego_mask shape: {tuple(v_store.is_ego_mask.shape)}")

    print(f"ptr: {v_store.ptr.tolist()}")


def validate_graph(graph):
    v_store = graph["vehicle"]
    s_store = graph["vehicle", "to", "vehicle"]

    num_nodes = v_store.x.shape[0]
    num_edges = s_store.edge_index.shape[1]

    assert s_store.edge_attr.shape[0] == num_edges, "edge_attr rows must match edge count"

    if hasattr(s_store, "distance"):
        assert s_store.distance.shape[0] == num_edges, "distance rows must match edge count"

    if hasattr(v_store, "id"):
        assert v_store.id.shape[0] == num_nodes, "vehicle.id rows must match node count"
    if hasattr(v_store, "pos"):
        assert v_store.pos.shape[0] == num_nodes, "vehicle.pos rows must match node count"
    if hasattr(v_store, "orientation"):
        assert v_store.orientation.shape[0] == num_nodes, "vehicle.orientation rows must match node count"
    if hasattr(v_store, "is_ego_mask"):
        assert v_store.is_ego_mask.shape[0] == num_nodes, "vehicle.is_ego_mask rows must match node count"

    # ptr should end at num_nodes
    assert int(v_store.ptr[-1].item()) == num_nodes, "Last ptr value must equal total number of vehicle nodes"

    print("Validation passed.")


def compare_before_after(before, after, timestep):
    vb = before["vehicle"]
    va = after["vehicle"]

    sb = before["vehicle", "to", "vehicle"]
    sa = after["vehicle", "to", "vehicle"]

    print("\n=== Delta ===")
    print(f"vehicle nodes added:       {va.x.shape[0] - vb.x.shape[0]}")
    print(f"spatial edges added:       {sa.edge_index.shape[1] - sb.edge_index.shape[1]}")
    print(f"edge_attr rows added:      {sa.edge_attr.shape[0] - sb.edge_attr.shape[0]}")

    if hasattr(sa, "distance"):
        print(f"distance rows added:       {sa.distance.shape[0] - sb.distance.shape[0]}")

    print(f"\nptr before: {vb.ptr.tolist()}")
    print(f"ptr after : {va.ptr.tolist()}")

    expected_prefix_unchanged = vb.ptr[: timestep + 1]
    actual_prefix_after = va.ptr[: timestep + 1]

    assert torch.equal(expected_prefix_unchanged, actual_prefix_after), (
        "ptr values before chosen timestep should remain unchanged"
    )

    print("ptr prefix check passed.")


def main():
    path = get_first_pt_file()
    print(f"Testing with: {path}")

    graph = torch.load(path, weights_only=False)
    graph_before = copy.deepcopy(graph)

    summarize_graph(graph_before, label="Before")

    timestep = 0
    modified = inject_chain(graph, timestep=timestep, chain_length=None, perturb_features=True)

    summarize_graph(modified, label="After")

    validate_graph(modified)
    compare_before_after(graph_before, modified, timestep=timestep)

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()