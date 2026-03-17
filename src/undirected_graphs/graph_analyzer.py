from torch.mtia import graph

from src.undirected_graphs.utils import UtilityFunctions

class GraphAnalyzer:    
    def main(self):
        dataset = UtilityFunctions.load_dataset(root="data/DD", name="DD", use_node_attr=True)   
        graph = dataset[4]
        self.search_graph(graph)

    def search_graph(self, graph):
        total_edges = graph.edge_index
        source_nodes_list = total_edges[0].tolist()
        dist_nodes_list = total_edges[1].tolist()

        edges = []
        for index in range(len(source_nodes_list)):
            edges.append((source_nodes_list[index], dist_nodes_list[index]))

        # Build undirected neighbors dict
        neighbors = {}
        for src, dst in edges:
            if src not in neighbors:
                neighbors[src] = set()
            if dst not in neighbors:
                neighbors[dst] = set()
            neighbors[src].add(dst)
            neighbors[dst].add(src)

        # Find all dangling nodes — nodes with only one unique neighbor
        all_dangling = set(node for node, nbrs in neighbors.items() if len(nbrs) == 1)

        # Traverse each chain and find both ends
        chain_starts = []
        visited_chains = set()

        for node in all_dangling:
            if node in visited_chains:
                continue

            current = node
            previous = None
            while True:
                nbrs = [n for n in neighbors[current] if n != previous]
                if len(nbrs) == 0:
                    break
                next_node = nbrs[0]
                if len(neighbors[next_node]) > 2:
                    break
                previous = current
                current = next_node

            visited_chains.add(node)
            visited_chains.add(current)
            chain_starts.append(current)

        return graph, chain_starts,edges, neighbors
        
    
    
analyzer = GraphAnalyzer()
analyzer.main()