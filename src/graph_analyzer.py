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


        return graph, chain_starts, neighbors
    
    def get_global_chain_length(self, dataset):
        max_length = 0
        graph_index = 0
        for i, graph in enumerate(dataset):
            _, chain_starts, neighbors = self.search_graph(graph)
            for node in chain_starts:
                length, _ = self.get_dangling_chain_length(node, neighbors)
                if length > max_length:
                    max_length = length
                    graph_index = i
        
        if graph_index is None:
            return max_length + 1, 0

        return max_length + 1, graph_index

    def get_shortest_chain_length(self, dataset):
        """returns the shortest dangling chain across the dataset and index of the graph
        """
        min_length = float('inf')
        graph_index = None
        for i, graph in enumerate(dataset):
            _, chain_starts, neighbors = self.search_graph(graph)
            for node in chain_starts:
                length, _ = self.get_dangling_chain_length(node, neighbors)
                if length < min_length:
                    min_length = length
                    graph_index = i
        
        if graph_index is None:
            return 1, 0
        
        return min_length + 1, graph_index
    
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