from utils import UtilityFunctions

class GraphAnalyzer:    
    def main(self):
        dataset = UtilityFunctions.load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY")
        graph = dataset[1]
        self.search_graph(graph)

    def search_graph(self, graph):
        """Takes a single graph as a paramter
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        
        total_edges = graph.edge_index
        source_nodes_list = total_edges[0].tolist()
        dist_nodes_list = total_edges[1].tolist()

        dangling_nodes = 0
        nodes_that_contain_dangling_chains = []
        
        # List of tuples containing edges between two nodes e.g [(source_0, dist_0), (source_1, dist_1) ... (source_n, dist_n)]
        edges = []
        
        # Dictionary containing a key, value pair where key = node_index and value = amount of edges.
        count_of_edges = {}

        
        for index in range(len(source_nodes_list)):
            source_node = source_nodes_list[index]
            dist_node = dist_nodes_list[index]
            edges.append((source_node, dist_node))

        for edge in edges:
            value = edge[0]
            # The .get(value, 0) handles the case where the key doesn't exist yet, defaulting to 0 before adding 1.
            count_of_edges[value] = count_of_edges.get(value, 0) + 1

        for index in count_of_edges:
            if count_of_edges[index] == 1: nodes_that_contain_dangling_chains.append(index)

        return {
            "graph_index": graph,
            "amount_dangling_nodes": len(nodes_that_contain_dangling_chains),
            "list_of_dangling_nodes": nodes_that_contain_dangling_chains,
            "edges": edges
        }
    
    
analyzer = GraphAnalyzer()
analyzer.main()