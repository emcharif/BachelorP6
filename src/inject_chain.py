import torch
import random
from src.graph_analyzer import GraphAnalyzer
from torch_geometric.data import Data


def inject_chain(
    graph: Data,
    target_chain_length: int,
    is_binary: bool,
    rng: random.Random,
    feature_mode: str,
) -> Data:
    """
    Injects a dangling chain into a graph.

    Parameters
    ----------
    graph:
        PyG graph to modify.
    target_chain_length:
        Target dangling chain length.
    is_binary:
        Whether node/edge features are binary.
    rng:
        Seeded Python random generator.
    feature_mode:
        "subtle" -> original behavior:
            - binary features are copied
            - continuous features are slightly perturbed
    """

    if feature_mode not in {"subtle"}:
        raise ValueError(
            f"Unknown feature_mode={feature_mode}. "
            "Expected 'subtle'."
        )

    graph_analyzer = GraphAnalyzer()
    graph.clone()

    chain_starts, neighbors = graph_analyzer.search_graph(graph)

    if len(chain_starts) != 0:
        chain_info = graph_analyzer.select_longest_dangling_chain(chain_starts, neighbors, rng)
    else:
        chain_info = [0, 0, 0]

    selected_chain_length = chain_info[1]
    selected_chain_end = chain_info[2]

    if graph.x is not None:
        num_nodes = graph.x.shape[0]
    else:
        num_nodes = int(graph.edge_index.max()) + 1

    # Guard: if selected_chain_end isn't in neighbors (e.g. fallback chain_info=[0,0,0]
    # and node 0 has no dangling chain), initialise it so .add() doesn't KeyError.
    neighbors.setdefault(selected_chain_end, set())

    while selected_chain_length < target_chain_length:

        new_node_id = num_nodes

        new_edges = torch.tensor(
            [[selected_chain_end, new_node_id], [new_node_id, selected_chain_end]],
            dtype=graph.edge_index.dtype,
            device=graph.edge_index.device,
        )

        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        num_nodes += 1

        neighbors[new_node_id] = {selected_chain_end}
        neighbors[selected_chain_end].add(new_node_id)

        if graph.x is not None:
            selected_chain_end_features = graph.x[selected_chain_end]

            if is_binary:
                new_node_features = selected_chain_end_features.clone().unsqueeze(0)
            else:
                deviations = torch.tensor(
                    [
                        rng.uniform(0.97, 1.02)
                        for _ in range(selected_chain_end_features.shape[0])
                    ],
                    dtype=selected_chain_end_features.dtype,
                    device=selected_chain_end_features.device,
                )
                new_node_features = (
                    selected_chain_end_features * deviations
                ).unsqueeze(0)

            graph.x = torch.cat([graph.x, new_node_features], dim=0)

        selected_chain_end = new_node_id
        selected_chain_length += 1

    watermarked_graph = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        y=graph.y,
    )

    return watermarked_graph