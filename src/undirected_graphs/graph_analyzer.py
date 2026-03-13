from utils import UtilityFunctions

class GraphAnalyzer:    
    def main(self):
        dataset = UtilityFunctions.load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY")
        graph = dataset[0]
        self.search_graph(graph)

    def search_graph(self, graph):
        total_edges = graph.edge_index
        source_nodes_list = total_edges[0].tolist()
        dist_nodes_list = total_edges[1].tolist()
        nodes_that_contain_dangling_chains = []
        dangling_nodes_distribution = {}
        edges = []
        count_of_edges = {}

        for index in range(len(source_nodes_list)):
            source_node = source_nodes_list[index]
            dist_node = dist_nodes_list[index]
            edges.append((source_node, dist_node))

        adjacency = {}
        for source, dest in edges:
            if source not in adjacency:
                adjacency[source] = set()
            if dest not in adjacency:
                adjacency[dest] = set()
            adjacency[source].add(dest)
            adjacency[dest].add(source)

        for node in adjacency:
            count_of_edges[node] = len(adjacency[node])

        for index in count_of_edges:
            if count_of_edges[index] == 1:
                nodes_that_contain_dangling_chains.append(index)

        # Walk inward from each leaf until a branching node
        for leaf in nodes_that_contain_dangling_chains:
            length = 0
            current = leaf
            previous = None

            while True:
                length += 1
                if count_of_edges[current] != 2:
                    break
                next_node = next(n for n in adjacency[current] if n != previous)
                previous = current
                current = next_node

            dangling_nodes_distribution[leaf] = length

        print(dangling_nodes_distribution)

        return {
            "graph": graph,
            "amount_dangling_nodes": len(nodes_that_contain_dangling_chains),
            "list_of_dangling_nodes": nodes_that_contain_dangling_chains,
            "dangling_nodes_distribution": dangling_nodes_distribution,
            "edges": edges
        }


analyzer = GraphAnalyzer()
analyzer.main()