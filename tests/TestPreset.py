import unittest

from PresetGraph import extract_giant_data, GraphData
from VertexSet import VertexSet, EuclideanDistance


class TestGiantExtraction(unittest.TestCase):
    def test(self):
        vertex_set = VertexSet(dimension=1, metric=EuclideanDistance(d=1))
        vertex_set.set_points_from_names({"a": (5,), "b": (6,), "c": (7,), "d": (8,)})
        weights = [1.2, 3.4, 5.6, 7.8]
        edge_list = [(1, 3)]
        data = GraphData(vertex_set=vertex_set, weights=weights, edge_list=edge_list)
        giant_data = extract_giant_data(data)
        self.assertEqual(set(giant_data.vertex_set.ids), {0, 1})
        self.assertEqual(giant_data.vertex_set.id_to_position(0), (6,))
        self.assertEqual(giant_data.vertex_set.id_to_name(0), 1)
        self.assertEqual(giant_data.vertex_set.id_to_position(1), (8,))
        self.assertEqual(giant_data.vertex_set.id_to_name(1), 3)
        self.assertEqual(giant_data.weights, [3.4, 7.8])
        self.assertEqual(giant_data.edge_list, [(0, 1)])

if __name__ == '__main__':
    unittest.main()
