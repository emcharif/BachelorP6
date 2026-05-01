import torch
import random
from graph_analyzer import GraphAnalyzer
from torch_geometric.data import Data
from utils import UtilityFunctions


def inject_chain(
    graph,
    chain_length,
    is_binary,
    rng: random.Random,
    feature_mode: str = "subtle",
    ood_value: float = 2.0,
):
    """
    Injects a dangling chain into a graph.

    Parameters
    ----------
    graph:
        PyG graph to modify.
    chain_length:
        Target dangling chain length.
    is_binary:
        Whether node/edge features are binary.
    rng:
        Seeded Python random generator.
    feature_mode:
        "subtle" -> original behavior:
            - binary features are copied
            - continuous features are slightly perturbed

        "ood" -> strengthened variant:
            - injected node features are fixed to ood_value
            - edge features still follow the original subtle behavior

    ood_value:
        Constant feature value used for injected nodes in strengthened mode.
    """

    if feature_mode not in {"subtle", "ood"}:
        raise ValueError(
            f"Unknown feature_mode={feature_mode}. "
            "Expected 'subtle' or 'ood'."
        )

    graph = graph.clone()
    utilityFunctions = UtilityFunctions()
    graphAnalyzer = GraphAnalyzer()

    graph, chain_starts, neighbors = graphAnalyzer.search_graph(graph)

    dangling_nodes_lengths = []

    if len(chain_starts) != 0:
        for d in chain_starts:
            length, edge_node = graphAnalyzer.get_dangling_chain_length(d, neighbors)
            dangling_nodes_lengths.append((d, length, edge_node))

        max_length = max(dangling_nodes_lengths, key=lambda x: x[1])
        longest_dangling_nodes = [
            d for d in dangling_nodes_lengths if d[1] == max_length[1]
        ]
    else:
        longest_dangling_nodes = [(node, 0, node) for node in neighbors.keys()]

    longest_dangling_node = utilityFunctions.select_dangling_node(
        longest_dangling_nodes, rng
    )

    current_length = longest_dangling_node[1]
    edge_node = longest_dangling_node[2]

    num_nodes = (
        graph.x.shape[0]
        if graph.x is not None
        else int(graph.edge_index.max()) + 1
    )

    while current_length < chain_length:
        new_node_id = num_nodes

        new_edges = torch.tensor(
            [[edge_node, new_node_id], [new_node_id, edge_node]],
            dtype=graph.edge_index.dtype,
            device=graph.edge_index.device,
        )

        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        num_nodes += 1

        neighbors[new_node_id] = {edge_node}
        neighbors[edge_node].add(new_node_id)

        # ─────────────────────────────────────────────────────────────
        # Node features
        # ─────────────────────────────────────────────────────────────
        if graph.x is not None:
            edge_node_features = graph.x[edge_node]

            if feature_mode == "ood":
                # Strengthened variant:
                # use fixed out-of-distribution feature values.
                new_node_features = torch.full(
                    (1, edge_node_features.shape[0]),
                    fill_value=ood_value,
                    dtype=edge_node_features.dtype,
                    device=edge_node_features.device,
                )

            else:
                # Subtle/original variant:
                # copy binary features or slightly perturb continuous features.
                if is_binary:
                    new_node_features = edge_node_features.clone().unsqueeze(0)
                else:
                    deviations = torch.tensor(
                        [
                            rng.uniform(0.97, 1.02)
                            for _ in range(edge_node_features.shape[0])
                        ],
                        dtype=edge_node_features.dtype,
                        device=edge_node_features.device,
                    )
                    new_node_features = (
                        edge_node_features * deviations
                    ).unsqueeze(0)

            graph.x = torch.cat([graph.x, new_node_features], dim=0)

        # ─────────────────────────────────────────────────────────────
        # Edge features
        # Kept subtle in both modes, since the strengthened variant only
        # changes injected node features.
        # ─────────────────────────────────────────────────────────────
        if graph.edge_attr is not None:
            edge_mask = graph.edge_index[0, :-2] == edge_node
            existing_edge_features = graph.edge_attr[edge_mask][0]

            if is_binary:
                new_edge_features = existing_edge_features.clone().unsqueeze(0)
                new_edge_features = torch.cat(
                    [new_edge_features, new_edge_features], dim=0
                )
            else:
                deviations_a = torch.tensor(
                    [
                        rng.uniform(0.97, 1.03)
                        for _ in range(existing_edge_features.shape[0])
                    ],
                    dtype=existing_edge_features.dtype,
                    device=existing_edge_features.device,
                )

                deviations_b = torch.tensor(
                    [
                        rng.uniform(0.97, 1.03)
                        for _ in range(existing_edge_features.shape[0])
                    ],
                    dtype=existing_edge_features.dtype,
                    device=existing_edge_features.device,
                )

                new_edge_a = (existing_edge_features * deviations_a).unsqueeze(0)
                new_edge_b = (existing_edge_features * deviations_b).unsqueeze(0)

                new_edge_features = torch.cat([new_edge_a, new_edge_b], dim=0)

            graph.edge_attr = torch.cat([graph.edge_attr, new_edge_features], dim=0)

        edge_node = new_node_id
        current_length += 1

    clean_graph = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr if graph.edge_attr is not None else None,
        y=graph.y,
    )

    return clean_graph