from utils import UtilityFunctions


def test_load_dataset():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    assert len(dataset) > 0, "Dataset should not be empty"
    for graph in dataset:
        assert graph.x is not None, "Node features should not be None"
        assert graph.x.shape[0] == graph.num_nodes, "Node features should have the same number of rows as nodes"