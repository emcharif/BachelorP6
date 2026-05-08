import numpy as np
import random

class GraphAnalyzer:    
    def search_graph(self, graph) -> tuple[list[int], dict[int, set[int]]]:
        """
        Identifies linear chains starting from leaf nodes and traces them to cluster boundaries.

        The algorithm builds an adjacency list and identifies 'dangling' nodes (degree 1). 
        For each dangling node, it traverses a path of degree-2 nodes until it hits 
         a junction (degree > 2). The traversal stops at the last node of the linear 
        chain—the 'exit' node—immediately before entering the cluster junction.

        Note:
            This method only processes chains that begin with a leaf node. Isolated 
            cycles (e.g., squares or triangles) without a dangling 'tail' are 
            ignored as they contain no degree-1 nodes to initiate the search.

        Args:
            graph: A graph object containing 'edge_index' (e.g., PyG Data object).

        Returns:
            A tuple containing:
                - graph: The original input graph object.
                - chain_starts: A list of 'boundary' nodes where linear branches 
                    connect to a junction or cluster.
                - neighbors: The full adjacency list {node_id: {neighbor_set}}.
        """
        total_edges = graph.edge_index
        source_nodes_list = total_edges[0].tolist()
        dest_nodes_list = total_edges[1].tolist()

        edges = []
        for index in range(len(source_nodes_list)):
            edges.append((source_nodes_list[index], dest_nodes_list[index]))

        # neighbors is an adjacency list representation: {node_id: {set of neighbors}}
        neighbors = {}
        for src, dst in edges:
            if src not in neighbors:
                neighbors[src] = set()
            if dst not in neighbors:
                neighbors[dst] = set()
            neighbors[src].add(dst)
            neighbors[dst].add(src)

        # Find all dangling nodes — nodes with only one unique neighbor
        all_dangling = set()
        for node, nbrs in neighbors.items():
            if len(nbrs) == 1:
                all_dangling.add(node)

        chain_starts = []
        visited_chains = set()

        for node in all_dangling:
            if node in visited_chains:
                continue

            current = node
            previous = None
            while True:
                nbrs = []
                for neighbor in neighbors[current]:
                    if neighbor != previous:
                        nbrs.append(neighbor)
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

        return chain_starts, neighbors
    
    def get_longest_global_chain_length(self, dataset):
        max_length = 0
        graph_index = 0
        for i, graph in enumerate(dataset):
            chain_starts, neighbors = self.search_graph(graph)
            for node in chain_starts:
                length, _ = self.get_dangling_chain_length(node, neighbors)
                if length > max_length:
                    max_length = length
                    graph_index = i
        
        if graph_index is None:
            return max_length + 1, 0

        return max_length + 1, graph_index

    def get_shortest_global_chain_length(self, dataset):
        """returns the shortest dangling chain across the dataset and index of the graph
        """
        min_length = float('inf')
        graph_index = None
        for i, graph in enumerate(dataset):
            chain_starts, neighbors = self.search_graph(graph)
            for node in chain_starts:
                length, _ = self.get_dangling_chain_length(node, neighbors)
                if length < min_length:
                    min_length = length
                    graph_index = i
        
        if graph_index is None:
            return 0
        
        return graph_index
    
    def get_dangling_chain_length(self, startnode, neighbors):
        """returns the length of the dangling chain starting at startnode
        Keyword arguments:        startnode: the node id of the dangling node
        neighbors: a dict with node id as key and a set of neighboring node ids as value
        Return: length of the dangling chain starting at startnode
        """
        length = 1
        current_node = startnode
        edge_node = None
        
        # Set previous_node to the neighbor with the most neighbors (the cluster), so we can ignore it in the while loop
        nbrs = list(neighbors[startnode])
        if len(nbrs) > 1:
            previous_node = max(nbrs, key=lambda n: len(neighbors[n]))
        else:
            previous_node = None
        
        while True:
            nbrs = list(neighbors[current_node])
            # Sort previous node out of neighbors, so we can ignore it
            next_nodes = [n for n in nbrs if n != previous_node]
            
            if len(next_nodes) == 0:
                break
            
            next_node = next_nodes[0]
            
            if len(neighbors[next_node]) > 2:
                break
            
            previous_node = current_node
            current_node = next_node
            length += 1
        
        edge_node = current_node
        return length, edge_node
    
    def select_longest_dangling_chain(self, chain_starts, neighbors, rng: random.Random):
        chain_info = []
        for chain_start in chain_starts:
            length, chain_end = self.get_dangling_chain_length(chain_start, neighbors)
            chain_info.append((chain_start, length, chain_end))

        lengths = []
        for chain in chain_info:
            lengths.append(chain[1])

        # .argmax returns the index of the highest value in the array
        max_idx = np.argmax(lengths)
        max_length = chain_info[max_idx]
        
        longest_chains = []
        for chain in chain_info:
            if chain[1] == max_length[1]:
                longest_chains.append(chain)

        rng.shuffle(longest_chains)
        return longest_chains[0]