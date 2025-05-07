import unittest
import math
from collections import defaultdict
from TestDistribution import dkw_p_value
from scipy.stats import binom

from SIEpidemic import *
from WeightedVertexSet import fixed_weights_generator, WeightedVertexSet, power_law_generator
from VertexSet import VertexSet, euclidean_distance_function, lattice


class BasicTests(unittest.TestCase):
    def setUp(self):
        """Both base graphs here are unit square cycles abcd with a cross edge ac, a pendant edge ae, an isolated
        vertex f, and weights given by self.weights. Self.unpenalised_epidemic takes mu = zeta = 0,
        and self.penalised_epidemic takes mu and zeta to be hardcoded values. Both epidemics start from a."""
        self.d = 2
        self.mu = 2.0
        self.zeta = 10.0

        unweighted_vertices = VertexSet(dimension=self.d, metric=euclidean_distance_function(d=self.d))
        unweighted_vertices.set_points_from_names({"a": (0, 1), "b": (1, 1), "c": (1, 0), "d": (0, 0), "e": (0, 3),
                                                   "f": (-1, -1)})
        self.a_id = unweighted_vertices.name_to_id("a")
        self.b_id = unweighted_vertices.name_to_id("b")
        self.c_id = unweighted_vertices.name_to_id("c")
        self.d_id = unweighted_vertices.name_to_id("d")
        self.e_id = unweighted_vertices.name_to_id("e")
        self.f_id = unweighted_vertices.name_to_id("f")

        self.weights = {"a": 2., "b": 3., "c": 5., "d": 7., "e": 11., "f": 13.}
        weight_generator = fixed_weights_generator(weights=self.weights)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator)
        self.edges = [(self.a_id, self.b_id), (self.b_id, self.c_id), (self.c_id, self.d_id), (self.d_id, self.a_id),
                      (self.a_id, self.c_id), (self.a_id, self.e_id)]
        edge_generator = fixed_graph_generator(edges=self.edges)

        self.unpenalised_epidemic = SIEpidemic(vertex_set=vertices, edge_generator=edge_generator,
                                               edge_cost_generator=constant_generator(1.), mu=0., zeta=0.)
        self.unpenalised_epidemic.run_infection(self.a_id)

        self.penalised_epidemic = SIEpidemic(vertex_set=vertices, edge_generator=edge_generator,
                                             edge_cost_generator=constant_generator(1.), mu=self.mu, zeta=self.zeta)
        self.penalised_epidemic.run_infection(self.a_id)

    def test_unpenalised_properties(self):
        infection_times = self.unpenalised_epidemic.infection_times
        self.assertEqual(infection_times[self.a_id], 0.)
        self.assertEqual(infection_times[self.b_id], 1.)
        self.assertEqual(infection_times[self.c_id], 1.)
        self.assertEqual(infection_times[self.d_id], 1.)
        self.assertEqual(infection_times[self.e_id], 1.)
        self.assertEqual(infection_times[self.f_id], np.inf)

        infectors = self.unpenalised_epidemic.infectors
        self.assertEqual(infectors[self.a_id], -1)
        self.assertEqual(infectors[self.b_id], self.a_id)
        self.assertEqual(infectors[self.c_id], self.a_id)
        self.assertEqual(infectors[self.d_id], self.a_id)
        self.assertEqual(infectors[self.e_id], self.a_id)
        self.assertEqual(infectors[self.f_id], -1)

        is_infected = self.unpenalised_epidemic.is_infected
        self.assertEqual(is_infected[self.a_id], True)
        self.assertEqual(is_infected[self.b_id], True)
        self.assertEqual(is_infected[self.c_id], True)
        self.assertEqual(is_infected[self.d_id], True)
        self.assertEqual(is_infected[self.e_id], True)
        self.assertEqual(is_infected[self.f_id], False)

    def test_penalised_properties(self):
        infection_times = self.penalised_epidemic.infection_times
        self.assertEqual(infection_times[self.a_id], 0.)
        self.assertEqual(infection_times[self.b_id], 6 ** self.mu)
        self.assertEqual(infection_times[self.c_id], 6 ** self.mu + 15 ** self.mu)
        self.assertEqual(infection_times[self.d_id], 14 ** self.mu)
        self.assertEqual(infection_times[self.e_id], 22 ** self.mu * 2 ** self.zeta)
        self.assertEqual(infection_times[self.f_id], np.inf)

        infectors = self.penalised_epidemic.infectors
        self.assertEqual(infectors[self.a_id], -1)
        self.assertEqual(infectors[self.b_id], self.a_id)
        self.assertEqual(infectors[self.c_id], self.b_id)
        self.assertEqual(infectors[self.d_id], self.a_id)
        self.assertEqual(infectors[self.e_id], self.a_id)
        self.assertEqual(infectors[self.f_id], -1)

    def test_infection_paths(self):
        path = self.penalised_epidemic.infection_path
        self.assertEqual(path(self.a_id), [self.a_id])
        self.assertEqual(path(self.b_id), [self.a_id, self.b_id])
        self.assertEqual(path(self.c_id), [self.a_id, self.b_id, self.c_id])
        self.assertEqual(path(self.d_id), [self.a_id, self.d_id])
        self.assertEqual(path(self.e_id), [self.a_id, self.e_id])
        self.assertEqual(path(self.f_id), None)

    def test_annulus_property(self):
        annulus = self.penalised_epidemic.spatial_annulus_property(inner_radius=0.95, outer_radius=1.05)
        self.assertEqual(annulus[self.a_id], False)
        self.assertEqual(annulus[self.b_id], True)
        self.assertEqual(annulus[self.c_id], False)
        self.assertEqual(annulus[self.d_id], True)
        self.assertEqual(annulus[self.e_id], False)
        self.assertEqual(annulus[self.f_id], False)

    def test_median_infection_time(self):
        test_median = self.penalised_epidemic.median_infection_at_boundary(radius=0.95, error=0.1)
        b_infection_time = self.penalised_epidemic.infection_times[self.b_id]
        d_infection_time = self.penalised_epidemic.infection_times[self.d_id]
        true_median = (b_infection_time + d_infection_time) / 2
        self.assertEqual(test_median, true_median)

    def test_proportional_cost(self):
        prop_cost = self.penalised_epidemic.proportional_cost
        self.assertEqual(prop_cost(self.a_id), None)
        self.assertEqual(prop_cost(self.b_id), 1.)
        self.assertEqual(prop_cost(self.c_id), (15 ** self.mu) / (6 ** self.mu + 15 ** self.mu))
        self.assertEqual(prop_cost(self.d_id), 1.)
        self.assertEqual(prop_cost(self.e_id), 1.)
        self.assertEqual(prop_cost(self.f_id), None)

    def test_proportional_length(self):
        prop_length = self.penalised_epidemic.proportional_length
        self.assertEqual(prop_length(self.a_id), None)
        self.assertEqual(prop_length(self.b_id), 1.)
        self.assertEqual(prop_length(self.c_id), 0.5)
        self.assertEqual(prop_length(self.d_id), 1.)
        self.assertEqual(prop_length(self.e_id), 1.)
        self.assertEqual(prop_length(self.f_id), None)

    def test_hop_counts(self):
        hop_count = self.penalised_epidemic.hop_count
        self.assertEqual(hop_count(self.a_id), 0)
        self.assertEqual(hop_count(self.b_id), 1)
        self.assertEqual(hop_count(self.c_id), 2)
        self.assertEqual(hop_count(self.d_id), 1)
        self.assertEqual(hop_count(self.e_id), 1)
        self.assertEqual(hop_count(self.f_id), None)


class EdgeCostTests(unittest.TestCase):
    def test_constant_cost(self):
        generator = constant_generator(1.)
        for _ in range(10):
            self.assertEqual(generator(), 1.0)

        generator = constant_generator(1.5)
        for _ in range(10):
            self.assertEqual(generator(), 1.5)

    def test_fpp_cost(self):
        entropy = 252869566619441809084118758734182862550  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)
        lambda_ = 1
        cost_generator = fpp_generator(lambda_, generator=generator)

        n = 100000
        costs = [cost_generator() for i in range(n)]

        # Split into buckets
        bucket_width = 0.3
        cost_counts = defaultdict(lambda: 0)
        for cost in costs:
            bucket = math.floor(cost / bucket_width)
            cost_counts[bucket] += 1

        def exponential_cdf(x):
            return 1 - math.exp(-lambda_ * x)

        # Probability of landing in bucket i under the correct distribution
        def target_pmf(i):
            return exponential_cdf(bucket_width * (i+1)) - exponential_cdf(bucket_width * i)

        # Check for stupid calculation errors
        self.assertAlmostEqual(sum([target_pmf(i) for i in range(100)]), 1, delta=0.01)

        p_value = dkw_p_value(sample_data=cost_counts, pmf=target_pmf, tvd_bound=0.01)
        self.assertEqual(1.634472214616812e-06, p_value)


class GIRGTests(unittest.TestCase):
    def setUp(self):
        self.alpha = 1.25

    def test_1d(self):
        """Generate 4-vertex 1-d GIRGs. All but two edges are present with certainty; check that these are always
        present and that the other two are present independently with the right connection probabilities."""
        entropy = 99587849537254950220960273702174246406  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)

        unweighted_vertices = lattice(dimension=1, size=4)
        id_a = unweighted_vertices.position_to_id((0.,))
        id_b = unweighted_vertices.position_to_id((1.,))
        id_c = unweighted_vertices.position_to_id((2.,))
        id_d = unweighted_vertices.position_to_id((3.,))

        weights = {id_a: 1.1, id_b: 1.2, id_c: 1.3, id_d: 1.4}
        weight_gen = fixed_weights_generator(weights)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_gen)

        graph_gen = girg_generator(alpha=self.alpha, scale_factor=.25, generator=generator)
        epidemic = SIEpidemic(vertex_set=vertices, edge_cost_generator=constant_generator(0.),
                              edge_generator=graph_gen, mu=0., zeta=0.)

        vertex_a = epidemic.graph.vertex(id_a)
        vertex_b = epidemic.graph.vertex(id_b)
        vertex_c = epidemic.graph.vertex(id_c)
        vertex_d = epidemic.graph.vertex(id_d)

        # All unit-length edges should be present with certainty (remembering we're on the torus). Other edges
        # should be present with probability (W_uW_v/|u-v|)^alpha = (W_uW_v/2)^alpha.
        ac_probability = (1.1 * 1.3 / 2.) ** self.alpha
        bd_probability = (1.2 * 1.4 / 2.) ** self.alpha

        def edge_is_present(u, v):
            return epidemic.graph.edge(u, v) is not None or epidemic.graph.edge(v, u) is not None

        # Lsb of key is whether ac is present, msb is whether bd is present.
        edges_present = {0: 0, 1: 0, 2: 0, 3: 0}

        for i in range(100000):
            if i % 10000 == 0:
                print(f"Test run {i//1000}k/100k")
            epidemic.sample_edges()

            # Even though the graph is undirected, edges are still stored as tuples.
            guaranteed_edges = [(vertex_a, vertex_b), (vertex_b, vertex_c), (vertex_c, vertex_d), (vertex_d, vertex_a)]
            for edge in guaranteed_edges:
                self.assertTrue(edge_is_present(edge[0], edge[1]))

            key = 1 if edge_is_present(vertex_a, vertex_c) else 0
            key += 2 if edge_is_present(vertex_b, vertex_d) else 0
            edges_present[key] += 1

        def target_pmf(x):
            if x == 0:
                return (1 - ac_probability) * (1 - bd_probability)
            if x == 1:
                return ac_probability * (1 - bd_probability)
            if x == 2:
                return (1 - ac_probability) * bd_probability
            if x == 3:
                return ac_probability * bd_probability
            return 0

        p_value = dkw_p_value(sample_data=edges_present, pmf=target_pmf, tvd_bound=0.01)
        self.assertEqual(1.296155806256856e-07, p_value)

    def test_2d(self):
        """Generate 9-vertex 2-d GIRGs with unit weights. Check that all edges that should be present are present,
        and that the total number of edges has the correct distribution."""
        entropy = 257965182494033889699734564010582594886  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)

        unweighted_vertices = lattice(dimension=2, size=3)
        id_a = unweighted_vertices.position_to_id((0., 0.))
        id_b = unweighted_vertices.position_to_id((1., 0.))
        id_c = unweighted_vertices.position_to_id((2., 0.))
        id_d = unweighted_vertices.position_to_id((0., 1.))
        id_e = unweighted_vertices.position_to_id((1., 1.))
        id_f = unweighted_vertices.position_to_id((2., 1.))
        id_g = unweighted_vertices.position_to_id((0., 2.))
        id_h = unweighted_vertices.position_to_id((1., 2.))
        id_i = unweighted_vertices.position_to_id((2., 2.))

        weights = {id_a: 1., id_b: 1., id_c: 1., id_d: 1., id_e: 1., id_f: 1., id_g: 1., id_h: 1., id_i: 1.}
        weight_gen = fixed_weights_generator(weights)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_gen)

        graph_gen = girg_generator(alpha=self.alpha, scale_factor=1/3, generator=generator)
        epidemic = SIEpidemic(vertex_set=vertices, edge_cost_generator=constant_generator(0.),
                              edge_generator=graph_gen, mu=0., zeta=0.)

        vertex_a = epidemic.graph.vertex(id_a)
        vertex_b = epidemic.graph.vertex(id_b)
        vertex_c = epidemic.graph.vertex(id_c)
        vertex_d = epidemic.graph.vertex(id_d)
        vertex_e = epidemic.graph.vertex(id_e)
        vertex_f = epidemic.graph.vertex(id_f)
        vertex_g = epidemic.graph.vertex(id_g)
        vertex_h = epidemic.graph.vertex(id_h)
        vertex_i = epidemic.graph.vertex(id_i)

        # All edges are either unit-length or diagonals (since we're on the torus). Unit-length edges should be
        # present with certainty. Diagonals should be present with probability (W_u*W_v/||u-v||^d)^\alpha, which
        # here is just 2^{-\alpha}.
        diag_probability = 2 ** (-self.alpha)

        def edge_is_present(u, v):
            return epidemic.graph.edge(u, v) is not None or epidemic.graph.edge(v, u) is not None

        # 9 vertices with 4 diagonal edges each = 18 possible edges.
        edges_present = {i: 0 for i in range(19)}

        for i in range(100000):
            if i % 10000 == 0:
                print(f"Test run {i // 1000}k/100k")
            epidemic.sample_edges()

            # Even though the graph is undirected, edges are still stored as tuples.
            guaranteed_edges = [(vertex_a, vertex_b), (vertex_a, vertex_c), (vertex_a, vertex_d), (vertex_a, vertex_g),
                                (vertex_b, vertex_c), (vertex_b, vertex_e), (vertex_b, vertex_h), (vertex_c, vertex_f),
                                (vertex_c, vertex_i), (vertex_d, vertex_e), (vertex_d, vertex_f), (vertex_d, vertex_g),
                                (vertex_e, vertex_f), (vertex_e, vertex_h), (vertex_f, vertex_i), (vertex_g, vertex_h),
                                (vertex_g, vertex_i), (vertex_h, vertex_i)]
            diagonal_edges = [(vertex_a, vertex_h), (vertex_a, vertex_e), (vertex_a, vertex_i), (vertex_a, vertex_f),
                              (vertex_b, vertex_d), (vertex_b, vertex_g), (vertex_b, vertex_f), (vertex_b, vertex_i),
                              (vertex_c, vertex_d), (vertex_c, vertex_e), (vertex_c, vertex_g), (vertex_c, vertex_h),
                              (vertex_d, vertex_h), (vertex_d, vertex_i), (vertex_e, vertex_g), (vertex_e, vertex_i),
                              (vertex_f, vertex_g), (vertex_f, vertex_h)]

            for edge in guaranteed_edges:
                self.assertTrue(edge_is_present(edge[0], edge[1]))

            count = len([e for e in diagonal_edges if edge_is_present(e[0], e[1])])
            edges_present[count] += 1

        target_pmf = lambda x: binom.pmf(k=x, n=18, p=diag_probability)
        p_value = dkw_p_value(sample_data=edges_present, pmf=target_pmf, tvd_bound=0.01)
        self.assertEqual(9.284880547544271e-07, p_value)


class FileIOTests(unittest.TestCase):
    def testDeterministic(self):
        entropy = 252869566619441809084118758734182862550
        generator = np.random.default_rng(seed=entropy)
        self.helper(generator)

    def testRandom(self):
        generator = np.random.default_rng()
        self.helper(generator)

    def helper(self, generator: np.random.Generator):
        """Run a quick infection on 10k vertices, save it, load it, and check the two runs are equal."""
        unweighted_vertices = lattice(dimension=2, size=100)
        weight_gen = power_law_generator(tau=3, generator=generator)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_gen)
        cost_gen = fpp_generator(lambda_=1.)
        edge_gen = girg_generator(alpha=1.8, scale_factor=1/100)
        saved_value = SIEpidemic(vertex_set=vertices, edge_cost_generator=cost_gen, edge_generator=edge_gen,
                                 mu=1., zeta=0.5)
        saved_value.run_infection(np.random.randint(10000))

        saved_value.save_to_file(".", "test_graph")
        loaded_value = SIEpidemic.load_from_file(".", "test_graph")

        saved_vertices = saved_value.vertex_set
        loaded_vertices = loaded_value.vertex_set

        self.assertEqual(saved_vertices.dimension, loaded_vertices.dimension)
        self.assertEqual(saved_vertices.size, loaded_vertices.size)
        for n in range(100):
            id_x = np.random.randint(0, 10000)
            id_y = np.random.randint(0, 10000)
            self.assertEqual(saved_vertices.distance(id_x, id_y), loaded_vertices.distance(id_x, id_y))

        for id_ in range(10000):
            self.assertEqual(saved_vertices.id_to_position(id_), loaded_vertices.id_to_position(id_))
            self.assertEqual(saved_vertices.id_to_name(id_), loaded_vertices.id_to_name(id_))
            self.assertEqual(saved_vertices.weight(id_), loaded_vertices.weight(id_))

        saved_edges = sorted([(int(e.source()), int(e.target()), saved_value.edge_costs[e])
                              for e in saved_value.graph.edges()])
        loaded_edges = sorted([(int(e.source()), int(e.target()), loaded_value.edge_costs[e])
                               for e in loaded_value.graph.edges()])
        # There is a 12-year old bug in CPython that makes unittest.assertEqual lock up on large inputs when the test
        # fails. Well done guys.
        self.assertTrue(saved_edges == loaded_edges)

        saved_infectors = [saved_value.infectors[i] for i in range(10000)]
        loaded_infectors = [saved_value.infectors[i] for i in range(10000)]
        self.assertTrue(saved_infectors == loaded_infectors)

        saved_times = [saved_value.infection_times[i] for i in range(10000)]
        loaded_times = [saved_value.infection_times[i] for i in range(10000)]
        self.assertTrue(saved_times == loaded_times)


if __name__ == '__main__':
    unittest.main()
