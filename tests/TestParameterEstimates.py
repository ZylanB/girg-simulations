import unittest

from PresetGraph import GraphData
from figures.ParameterEstimates import get_low_degree_edges, get_degree_sequence
from VertexSet import VertexSet, EuclideanDistance


class TestDegreeSequence(unittest.TestCase):
    def test(self):
        vertex_set = VertexSet(dimension=2, metric=EuclideanDistance(d=2))
        edge_list = [(0, 1), (1, 2), (2, 0), (1, 3)]
        weights = [1., 2., 3., 4.]
        graph_data = GraphData(vertex_set=vertex_set, edge_list=edge_list, weights=weights)
        degrees = get_degree_sequence(graph_data).tolist()
        self.assertEqual([3, 2, 2, 1], degrees)


class TestLowDegree(unittest.TestCase):
    def test(self):
        # Graph is a cycle 123456 where vertex 3 is also joined to every other vertex plus a matching edge 78.
        edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (3, 1), (3, 5), (3, 6), (7, 8)]

        mid_result = [[1, 2], [4, 5], [5, 6], [6, 1], [7, 8]]
        final_result = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [3, 1], [3, 5], [3, 6], [7, 8]]
        expected_results = [[], [[7, 8]], [[7, 8]], mid_result, mid_result, final_result, final_result, final_result,
                            final_result]
        for i in range(9):
            results = get_low_degree_edges(size=9, edge_list=edge_list, cutoff=i)
            self.assertEqual(sorted(expected_results[i]), sorted(results.tolist()))


class TestHillCoefficients(unittest.TestCase):
    def test(self):
        """TODO: Give it a sample from actual noisy power law data with a set seed."""
        pass


class TestAlphaRegression(unittest.TestCase):
    def test(self):
        """TODO: Give it a GIRG with a set seed and check it gets close to the right alpha."""
        pass


class TestSynGowallaParameters(unittest.TestCase):
    def test(self):
        """TODO: Call the parameter estimation functions and make sure they match the values we're using for synthetic
        Gowalla."""
        pass


if __name__ == '__main__':
    unittest.main()
