from torch.mtia import graph

from utils import UtilityFunctions

class GraphAnalyzer:    
    def main(self):
        dataset = UtilityFunctions.load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY")
        graph = dataset[1]
        self.search_graph(graph)

    def search_graph(self, graph):
        total_edges = graph.edge_index
        source_nodes_list = total_edges[0].tolist()
        dist_nodes_list = total_edges[1].tolist()

        edges = []
        for index in range(len(source_nodes_list)):
            edges.append((source_nodes_list[index], dist_nodes_list[index]))

        # Byg neighbors dict
        neighbors = {}
        for src, dst in edges:
            if src not in neighbors:
                neighbors[src] = set()
            neighbors[src].add(dst)

        # Find alle dangling nodes — noder med kun én unik nabo
        all_dangling = set(node for node, nbrs in neighbors.items() if len(nbrs) == 1)

        # Traverser hver chain og find begge ender
        chain_starts = []
        visited_chains = set()

        for node in all_dangling:
            if node in visited_chains:
                continue

            # Traverser chain fra denne node til den anden ende
            current = node
            previous = None
            while True:
                nbrs = [n for n in neighbors[current] if n != previous]
                if len(nbrs) == 0:
                    break
                next_node = nbrs[0]
                if len(neighbors[next_node]) > 2:
                    # Vi er nået ind i clusteret — current er cluster-siden
                    break
                previous = current
                current = next_node

            # current er nu cluster-siden, node er den yderste ende
            # Marker begge som besøgt
            visited_chains.add(node)
            visited_chains.add(current)

            # Behold kun cluster-siden som chain start
            chain_starts.append(current)

        return {
            "graph_index": graph,
            "amount_dangling_nodes": len(chain_starts),
            "danling_nodes": chain_starts,
            "edges": edges,
            "neighbors": neighbors
        }
    
    
analyzer = GraphAnalyzer()
analyzer.main()