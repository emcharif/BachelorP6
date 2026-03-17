import torch
from src.undirected_graphs.utils import UtilityFunctions
from src.undirected_graphs.graph_analyzer import GraphAnalyzer

def inject_chain(graph, chain_length):
   
    # Get tuple of (source, target) edges of graph
    graph, chain_starts,edges, neighbors = GraphAnalyzer().search_graph(graph)

    # Get list of dangling nodes and their lengths
    danling_nodes_lenths = []
    if len(chain_starts) != 0:
        for d in chain_starts:
            length, edge_node = UtilityFunctions().get_dangling_chain_length(d, neighbors)
            danling_nodes_lenths.append((d, length, edge_node))
        # pick the/longest dangling node
        max_length = max(danling_nodes_lenths, key=lambda x: x[1])
        longest_dangling_nodes = [d for d in danling_nodes_lenths if d[1] == max_length[1]]
    else :
        # If there are no dangling nodes, select all nodes as potential candidates for chain injection
         longest_dangling_nodes = [(node, 0, node) for node in neighbors.keys()]
    longest_dangling_nodes = UtilityFunctions().select_dangling_node(longest_dangling_nodes)
    
    current_length = longest_dangling_nodes[1]
    edge_node = longest_dangling_nodes[2]

    # Inject new nodes and edges until we reach the desired chain length
    while current_length < chain_length:
        
        new_node_id = graph.num_nodes  # næste ledige node id
    
        # Opdater PyG graf
        new_edges = torch.tensor([[edge_node, new_node_id],[new_node_id, edge_node]])  # begge retninger
        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        graph.num_nodes += 1
        
        # Opdater neighbors dict
        neighbors[new_node_id] = {edge_node}
        neighbors[edge_node].add(new_node_id)
        
        edge_node = new_node_id
        current_length += 1

    return graph

    

    



    