from utils import UtilityFunctions

class GraphAnalyzer:    
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

        chain_lengths = {}

        for start in chain_starts:
            length = 1
            previous = next(n for n in neighbors[start] if len(neighbors[n]) > 2)
            current = start

            while True:
                nbrs = [n for n in neighbors[current] if n != previous]
                if not nbrs:
                    break
                previous = current
                current = nbrs[0]
                length += 1

            chain_lengths[start] = length

        return graph, chain_starts, edges, neighbors, chain_lengths
    
    def get_global_chain_length(self, dataset):
        max_length = 0
        for graph in dataset:
            _, chain_starts, _, neighbors, _ = self.search_graph(graph)
            for node in chain_starts:
                length, _ = UtilityFunctions().get_dangling_chain_length(node, neighbors)
                if length > max_length:
                    max_length = length
        return max_length + 1