import torch
from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer

def inject_chain(graph, chain_length, is_binary):
    utilityFunctions = UtilityFunctions()
    graphAnalyzer = GraphAnalyzer()

    graph = graph.clone() #because we need to keep graph input as it is
   
    # Get tuple of (source, target) edges of graph
    graph, chain_starts, neighbors = graphAnalyzer.search_graph(graph)

    # Get list of dangling nodes and their lengths
    danling_nodes_lenths = []
    if len(chain_starts) != 0:
        for d in chain_starts:
            length, edge_node = utilityFunctions.get_dangling_chain_length(d, neighbors)
            danling_nodes_lenths.append((d, length, edge_node))
        # pick the/longest dangling node
        max_length = max(danling_nodes_lenths, key=lambda x: x[1])
        longest_dangling_nodes = [d for d in danling_nodes_lenths if d[1] == max_length[1]]
    else :
        # If there are no dangling nodes, select all nodes as potential candidates for chain injection
        longest_dangling_nodes = [(node, 0, node) for node in neighbors.keys()]
    longest_dangling_nodes = utilityFunctions.select_dangling_node(longest_dangling_nodes)
    
    current_length = longest_dangling_nodes[1]
    edge_node = longest_dangling_nodes[2]
    
    while current_length < chain_length:
    
        new_node_id = graph.num_nodes

        # Opdater PyG graf
        new_edges = torch.tensor([[edge_node, new_node_id],[new_node_id, edge_node]])
        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        graph.num_nodes += 1

        # Opdater neighbors dict
        neighbors[new_node_id] = {edge_node}
        neighbors[edge_node].add(new_node_id)

        # Håndter node attributter
        if graph.x is not None:
            edge_node_features = graph.x[edge_node]

            if is_binary:
                new_node_features = edge_node_features.clone().unsqueeze(0)
            else:
                deviations = torch.FloatTensor(edge_node_features.shape).uniform_(0.97, 1.02)
                new_node_features = (edge_node_features * deviations).unsqueeze(0)

            graph.x = torch.cat([graph.x, new_node_features], dim=0)

        # Håndter edge attributter — lav mask FØR edge_index opdateres
        if graph.edge_attr is not None:
            edge_mask = (graph.edge_index[0, :-2] == edge_node)  # ignorer de 2 nye kanter vi lige tilføjede
            existing_edge_features = graph.edge_attr[edge_mask][0]

            if is_binary:
                new_edge_features = existing_edge_features.clone().unsqueeze(0)
                new_edge_features = torch.cat([new_edge_features, new_edge_features], dim=0)
            else:
                deviations_a = torch.FloatTensor(existing_edge_features.shape).uniform_(0.97, 1.03)
                deviations_b = torch.FloatTensor(existing_edge_features.shape).uniform_(0.97, 1.03)
                new_edge_a = (existing_edge_features * deviations_a).unsqueeze(0)
                new_edge_b = (existing_edge_features * deviations_b).unsqueeze(0)
                new_edge_features = torch.cat([new_edge_a, new_edge_b], dim=0)

            graph.edge_attr = torch.cat([graph.edge_attr, new_edge_features], dim=0)

        edge_node = new_node_id
        current_length += 1
    
    if hasattr(graph, 'num_nodes'):
        del graph.num_nodes

    return graph