from utils import UtilityFunctions

utils = UtilityFunctions()

def test_dif_watermarked_and_benign_graph_edges():

    a = [[1,1,2,2,3,3], [2,3,1,3,1,2]]
    b = [[1,1,2,2,3,3,3,4], [2,3,1,3,1,2,4,3]]

    x, y, delta = utils.dif_watermarked_and_benign_graph_edges(a, b)

    assert x == a
    assert y == b
    assert delta == [[3,4], [4,3]]