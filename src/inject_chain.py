import torch
import random
from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from torch_geometric.data import Data

def inject_chain(graph, chain_length, is_binary, rng: random.Random):
    graph = graph.clone()
    utilityFunctions = UtilityFunctions()
    graphAnalyzer = GraphAnalyzer()

    graph, chain_starts, neighbors = graphAnalyzer.search_graph(graph)

    danling_nodes_lenths = []
    if len(chain_starts) != 0:
        for d in chain_starts:
            length, edge_node = utilityFunctions.get_dangling_chain_length(d, neighbors)
            danling_nodes_lenths.append((d, length, edge_node))
        max_length = max(danling_nodes_lenths, key=lambda x: x[1])
        longest_dangling_nodes = [d for d in danling_nodes_lenths if d[1] == max_length[1]]
    else:
        longest_dangling_nodes = [(node, 0, node) for node in neighbors.keys()]
    longest_dangling_nodes = utilityFunctions.select_dangling_node(longest_dangling_nodes, rng)

    current_length = longest_dangling_nodes[1]
    edge_node = longest_dangling_nodes[2]

    num_nodes = graph.x.shape[0] if graph.x is not None else int(graph.edge_index.max()) + 1

    while current_length < chain_length:
        new_node_id = num_nodes

        new_edges = torch.tensor([[edge_node, new_node_id], [new_node_id, edge_node]])
        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        num_nodes += 1

        neighbors[new_node_id] = {edge_node}
        neighbors[edge_node].add(new_node_id)

        if graph.x is not None:
            edge_node_features = graph.x[edge_node]
            feature_dim = edge_node_features.shape[0]
            new_node_features = torch.ones(feature_dim).unsqueeze(0) * 2.0
            graph.x = torch.cat([graph.x, new_node_features], dim=0)

        if graph.edge_attr is not None:
            edge_mask = (graph.edge_index[0, :-2] == edge_node)
            existing_edge_features = graph.edge_attr[edge_mask][0]
            if is_binary:
                new_edge_features = existing_edge_features.clone().unsqueeze(0)
                new_edge_features = torch.cat([new_edge_features, new_edge_features], dim=0)
            else:
                deviations_a = torch.tensor(
                    [rng.uniform(0.97, 1.03) for _ in range(existing_edge_features.shape[0])]
                ).float()
                deviations_b = torch.tensor(
                    [rng.uniform(0.97, 1.03) for _ in range(existing_edge_features.shape[0])]
                ).float()
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
        y=graph.y
    )

    return clean_graph