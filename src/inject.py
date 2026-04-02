import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import copy
import torch
from src.Utilities.utils import UtilityFunctions


def inject_chain(graph, timestep: int, chain_length: int | None = None, perturb_features: bool = True):
    utility = UtilityFunctions()
    graph = copy.deepcopy(graph)

    start, end = utility.get_timestep_bounds(graph, timestep)
    num_nodes_t = end - start

    edge_index_t_global, edge_attr_t, mask = utility.get_spatial_edges_for_timestep(graph, timestep)
    edge_index_t = utility.global_to_local_edge_index(edge_index_t_global, start)

    neighbors = utility.build_neighbors(edge_index_t, num_nodes_t)
    current_length, edge_node_local = utility.find_longest_dangling_info(neighbors)

    if chain_length is None:
        chain_length = current_length + 1

    while current_length < chain_length:
        edge_node_global = start + edge_node_local

        new_node_global = utility.append_vehicle_node(
            graph,
            source_global_node=edge_node_global,
            perturb_features=perturb_features,
        )

        existing_edge_feature = utility.get_existing_edge_feature(edge_index_t, edge_attr_t, edge_node_local)
        if existing_edge_feature is None:
            existing_edge_feature = torch.zeros(
                graph["vehicle", "to", "vehicle"].edge_attr.shape[1],
                dtype=graph["vehicle", "to", "vehicle"].edge_attr.dtype,
                device=graph["vehicle", "to", "vehicle"].edge_attr.device,
            )
        elif perturb_features:
            existing_edge_feature = utility.copy_and_perturb_features(existing_edge_feature)

        utility.append_spatial_edges(
            graph,
            source_global=edge_node_global,
            target_global=new_node_global,
            edge_feature=existing_edge_feature,
            mask=mask,
        )

        utility.increment_ptr_from_timestep(graph, timestep, increment=1)

        # Update local bookkeeping for possible next loop iteration
        new_node_local = num_nodes_t
        neighbors.setdefault(new_node_local, set()).add(edge_node_local)
        neighbors.setdefault(edge_node_local, set()).add(new_node_local)

        edge_node_local = new_node_local
        num_nodes_t += 1
        current_length += 1

    return graph