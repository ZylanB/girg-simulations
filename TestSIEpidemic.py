import unittest
from SIEpidemic import *
from WeightedVertexSet import fixed_weights_generator, WeightedVertexSet
from VertexSet import VertexSet, euclidean_distance_function


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
        unpenalised_vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator,
                                                 mu=0, zeta=0)
        penalised_vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator,
                                               mu=self.mu, zeta=self.zeta)

        self.edges = [(self.a_id, self.b_id), (self.b_id, self.c_id), (self.c_id, self.d_id), (self.d_id, self.a_id),
                      (self.a_id, self.c_id), (self.a_id, self.e_id)]
        edge_generator = fixed_graph_generator(edges=self.edges)
        self.unpenalised_epidemic = SIEpidemic(vertex_set=unpenalised_vertices, edge_generator=edge_generator,
                                               edge_cost_generator=constant_generator(1.))
        self.unpenalised_epidemic.run_infection(self.a_id)

        self.penalised_epidemic = SIEpidemic(vertex_set=penalised_vertices, edge_generator=edge_generator,
                                             edge_cost_generator=constant_generator(1.))
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
    pass
    def test_constant_cost(self):
        generator = constant_generator(1.)
        for _ in range(10):
            self.assertEqual(generator(), 1.0)

        generator = constant_generator(1.5)
        for _ in range(10):
            self.assertEqual(generator(), 1.5)


class GIRGTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
