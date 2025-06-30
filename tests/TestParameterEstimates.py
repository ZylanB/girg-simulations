from math import ceil
import unittest

import numpy as np

from PresetGraph import GraphData
from figures.ParameterEstimates import get_low_degree_edges, get_degree_sequence, get_hill_coefficients
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
        self.seed = 274286783427111506408736539148187402179
        rng = np.random.default_rng(self.seed)
        # This is cooked up to be a distribution Hill's estimator should thrive on - large integer values so adding
        # uniform noise has a similar effect to a degree distribution, and Pr(X >= x) ~ x^(-1.5) so we should expect
        # tau roughly equal to 2.5.
        sample_data = [ceil(100/rng.power(1.5)) for _ in range(10000)]
        sample_data.sort(reverse=True)

        hill_data = get_hill_coefficients(np.array(sample_data))
        self.assertEqual(hill_data.kappa, 9670)
        self.assertEqual(hill_data.xi, 0.6625810969768464)
        self.assertEqual(hill_data.tau, 2.509249214266287)
        print(hill_data)


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
