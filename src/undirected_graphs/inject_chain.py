from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer

def inject_chain(graph_id, chain_length):
    # Load the graph
    dataset = UtilityFunctions.load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY")
    graph = dataset[graph_id]
    # Get tuple of (source, target) edges of graph

    graph_tuples = GraphAnalyzer().search_graph(graph)

    # Get list of dangling nodes and their lengths
    danling_nodes_lenths = []
    for d in graph_tuples["danling_nodes"]:
        length = UtilityFunctions().get_dangling_chain_length(d, graph_tuples["neighbors"])
        danling_nodes_lenths.append((d, length))

    # TODO pick the/longest dangling node
    longest_dangling_nodes = max(danling_nodes_lenths, key=lambda x: x[1])
    print(longest_dangling_nodes)
    

    
    if len(longest_dangling_nodes) == 1:
        longest_dangling_nodes = longest_dangling_nodes[0]
    else:
        selected_dangling_node = UtilityFunctions().select_dangling_node(longest_dangling_nodes)

    # TODO Inject chain of specified length at the selected dangling node
    #wm_graph = inject_chain_at_node(graph, selected_dangling_node, chain_length)

    #return wm_graph
    
    inject_chain(graph_id=0, chain_length=5)

    



    