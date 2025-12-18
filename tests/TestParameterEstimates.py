from math import ceil
import unittest

import numpy as np

from figures.ParameterEstimates import (get_low_degree_edges, get_degree_sequence, get_hill_coefficients,
                                        alpha_estimate_data, get_edge_ccdf)
from Gowalla import SYN_GOWALLA_TAU, SYN_GOWALLA_ALPHA, GowallaEUGiantGraph, GowallaGiantGraph
from PresetGraph import GraphData
from SIEpidemic import GirgGen, FixedGraphGen
from WeightedVertexSet import PowerLawWeightGen, WeightedVertexSet, IdentityWeightScaler, FixedWeightGen
from VertexSet import VertexSet, EuclideanDistance, PoissonPointProcess, FixedVertexSet


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
            results = get_low_degree_edges(count=9, edge_list=edge_list, cutoff=i)
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
    def test_edge_extraction(self):
        # Generates a rectangular clique with corners at (0,0) and (2,4).
        rng = np.random.default_rng()
        point_dict = {0: (0.,0.), 1: (2.,0,), 2: (0.,4.), 3: (2.,4.)}
        vertex_gen = FixedVertexSet(metric=EuclideanDistance(d=2), points=point_dict, description="test")
        weight_gen = FixedWeightGen(weights={0: 1., 1: 1., 2: 1., 3: 1.}, description="Unit weights")
        vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=rng)
        edge_gen = FixedGraphGen(edges=[(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], description="Empty graph")
        edges = edge_gen(vertices, rng)
        graph_data = GraphData(vertex_set=vertices.vertices, weights=vertices.weights, edge_list=edges)

        edge_ccdf_x, edge_ccdf_y = get_edge_ccdf(graph_data=graph_data, edge_cutoffs=(2., 4.))
        self.assertEqual(edge_ccdf_x.tolist(), [2., 4.])
        self.assertEqual(edge_ccdf_y.tolist(), [1., .5])

        edge_ccdf_x, edge_ccdf_y = get_edge_ccdf(graph_data=graph_data, edge_cutoffs=(3., 4.1))
        self.assertEqual(edge_ccdf_x.tolist(), [4.])
        self.assertEqual(edge_ccdf_y.tolist(), [1.])

        edge_ccdf_x, edge_ccdf_y = get_edge_ccdf(graph_data=graph_data, edge_cutoffs=(1., 5.))
        self.assertEqual(edge_ccdf_x.tolist(), [2., 4., 4.47213595499958])
        self.assertEqual(edge_ccdf_y.tolist(), [1., 0.6666666666666667, 0.33333333333333337])


    def helper_test_alpha(self, entropy: int, lower_cutoff: float, expected_alpha: float, expected_deviation: float):
        """Generates a million-vertex synthetic GIRG with the given entropy and vertex size (so vertex count is
        vertex_size^2), estimates alpha for it, and compares it against the expected result."""
        rng = np.random.default_rng(seed=entropy)
        vertex_gen = PoissonPointProcess(dimension=2, size=1000)
        weight_gen = PowerLawWeightGen(tau=SYN_GOWALLA_TAU, ell=IdentityWeightScaler())
        vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=rng)
        edge_gen = GirgGen(alpha=SYN_GOWALLA_ALPHA, scale_factor=1/1000)
        edges=edge_gen(vertices, rng)
        graph_data = GraphData(vertex_set=vertices.vertices, weights=vertices.weights, edge_list=edges)

        results = alpha_estimate_data(graph_data=graph_data, edge_cutoffs=(lower_cutoff, 500.), rng=rng)
        self.assertEqual(results.alpha, expected_alpha)
        self.assertEqual(results.deviation, expected_deviation)


    def test_alpha_values(self):
        # True alpha value is 1.2, and the error should drop as lower_cutoff rises.
        self.helper_test_alpha(entropy=182721707948319913994353143988269937783, lower_cutoff=5.,
                               expected_alpha=1.1695189227902154, expected_deviation=3.387305857653604e-05)
        self.helper_test_alpha(entropy=182721707948319913994353143988269937783, lower_cutoff=10.,
                               expected_alpha=1.1829569351500704, expected_deviation=1.7929184384159478e-05)
        self.helper_test_alpha(entropy=182721707948319913994353143988269937783, lower_cutoff=20.,
                               expected_alpha=1.1909933681506295, expected_deviation=8.88088008100671e-06)
        self.helper_test_alpha(entropy=182721707948319913994353143988269937783, lower_cutoff=40.,
                               expected_alpha=1.1955340160220376, expected_deviation=4.3517095138189185e-06)
        self.helper_test_alpha(entropy=182721707948319913994353143988269937783, lower_cutoff=80.,
                               expected_alpha=1.1982302783112322, expected_deviation=2.5476893950495886e-06)

class TestSynGowallaParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_data = GowallaGiantGraph().graph_data

    def test_tau(self):
        degrees = get_degree_sequence(self.graph_data)
        hill_coefficients = get_hill_coefficients(degrees)
        xi, kappa, tau = hill_coefficients.xi, hill_coefficients.kappa, hill_coefficients.tau
        self.assertEqual(xi, 0.5618395953389204)
        self.assertEqual(kappa, 1484)
        self.assertEqual(tau, 2.779867436001492)
        self.assertEqual(SYN_GOWALLA_TAU, 2.78)

    def test_alpha(self):
        entropy = 17280729670412610446175544547045928942
        rng = np.random.default_rng(seed=entropy)

        results = alpha_estimate_data(graph_data=self.graph_data, edge_cutoffs=(5, 100), rng=rng)
        self.assertEqual(results.alpha, 1.1935975507367773)
        self.assertEqual(results.deviation, 7.274817243000555e-05)

        results = alpha_estimate_data(graph_data=self.graph_data, edge_cutoffs=(10, 100), rng=rng)
        self.assertEqual(results.alpha, 1.2268393669325883)
        self.assertEqual(results.deviation, 5.362516236634198e-05)

        results = alpha_estimate_data(graph_data=self.graph_data, edge_cutoffs=(20, 100), rng=rng)
        self.assertEqual(results.alpha, 1.1961463488211799)
        self.assertEqual(results.deviation, 0.0001319665165339359)

        self.assertEqual(SYN_GOWALLA_ALPHA, 1.2)


if __name__ == '__main__':
    # unittest.main()

    suite = unittest.TestSuite()
    # suite.addTest(TestAlphaRegression("test_alpha_values"))
    suite.addTest(TestSynGowallaParameters("test_tau"))
    suite.addTest(TestSynGowallaParameters("test_alpha"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
