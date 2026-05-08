import torch
import random
from src.graph_analyzer import GraphAnalyzer
from torch_geometric.data import Data
from src.utils import UtilityFunctions
import numpy as np


def inject_chain(
    graph: object,
    target_chain_length: int,
    is_binary: bool,
    rng: random.Random,
    feature_mode: str,
    ood_value: float = 2.0,
):
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

    utility_functions = UtilityFunctions()
    graph_analyzer = GraphAnalyzer()

    graph, chain_starts, neighbors = graph_analyzer.search_graph(graph)
    chain_info = []

    # OVERVEJ OM IF STATEMENT SKAL FLYTTES IND I GRAPH ANALYZEREN
    if len(chain_starts) != 0:
        for chain_start in chain_starts:
            length, chain_end = graph_analyzer.get_dangling_chain_length(chain_start, neighbors)
            chain_info.append((chain_start, length, chain_end))

        lengths = []
        for chain in chain_info:
            lengths.append(chain[1])

        # .argmax returns the index of the highest value in the array
        max_idx = np.argmax(lengths)
        max_length = chain_info[max_idx]
        
        print("max_length: ", max_length)
        # array af alle der har den længste længde af chain
        nodeids_for_all_longest_chains = []
        for chain in chain_info:
            if chain[1] == max_length[1]:
                nodeids_for_all_longest_chains.append(chain)

        chain_info = utility_functions.select_dangling_node(nodeids_for_all_longest_chains, rng)
    else:
        chain_info = [0, 0, 0]

    current_length = chain_info[1]
    chain_end = chain_info[2]

    num_nodes = (
        graph.x.shape[0]
        if graph.x is not None
        else int(graph.edge_index.max()) + 1
    )

    while current_length < target_chain_length:
        new_node_id = num_nodes

        new_edges = torch.tensor(
            [[chain_end, new_node_id], [new_node_id, chain_end]],
            dtype=graph.edge_index.dtype,
            device=graph.edge_index.device,
        )

        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        num_nodes += 1

        neighbors[new_node_id] = {chain_end}
        neighbors[chain_end].add(new_node_id)

        # ─────────────────────────────────────────────────────────────
        # Node features
        # ─────────────────────────────────────────────────────────────
        if graph.x is not None:
            chain_end_features = graph.x[chain_end]

            if feature_mode == "ood":
                # Strengthened variant:
                # use fixed out-of-distribution feature values.
                new_node_features = torch.full(
                    (1, chain_end_features.shape[0]),
                    fill_value=ood_value,
                    dtype=chain_end_features.dtype,
                    device=chain_end_features.device,
                )

            else:
                # Subtle/original variant:
                # copy binary features or slightly perturb continuous features.
                if is_binary:
                    new_node_features = chain_end_features.clone().unsqueeze(0)
                else:
                    deviations = torch.tensor(
                        [
                            rng.uniform(0.97, 1.02)
                            for _ in range(chain_end_features.shape[0])
                        ],
                        dtype=chain_end_features.dtype,
                        device=chain_end_features.device,
                    )
                    new_node_features = (
                        chain_end_features * deviations
                    ).unsqueeze(0)

            graph.x = torch.cat([graph.x, new_node_features], dim=0)

        # ─────────────────────────────────────────────────────────────
        # Edge features
        # Kept subtle in both modes, since the strengthened variant only
        # changes injected node features. ttttesssttterreesssss!PIIKIPKIPKIKPIKPIKIPKIPKIKPKIIPKKPIKPIKKIPKK EDGE ATTRIBUTES
        # ─────────────────────────────────────────────────────────────
        if graph.edge_attr is not None:
            edge_mask = graph.edge_index[0, :-2] == chain_end
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

        chain_end = new_node_id
        current_length += 1

    clean_graph = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr if graph.edge_attr is not None else None,
        y=graph.y,
    )

    return clean_graph