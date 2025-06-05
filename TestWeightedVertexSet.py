import unittest
from TestDistribution import dkw_p_value
from WeightedVertexSet import *
from VertexSet import FixedVertexSet, EuclideanDistance
from collections import defaultdict
from math import floor


class BasicTests(unittest.TestCase):
    def setUp(self):
        self.weights = {1: 1., 2: 2.5, "abc": 3.}
        self.weight_gen = FixedWeightGenerator(self.weights, "Test generator")
        points = {1: (0, 0), 2: (2, 0), "abc": (2, 2)}
        self.vertex_gen = FixedVertexSet(metric=EuclideanDistance(d=2), points=points, description="Test")
        self.vertices = WeightedVertexSet(vertex_generator=self.vertex_gen, weight_generator=self.weight_gen)

        self.id_1 = self.vertices.name_to_id(1)
        self.id_2 = self.vertices.name_to_id(2)
        self.id_abc = self.vertices.name_to_id("abc")

    def test_fixed_weights(self):
        output = self.weight_gen(self.vertices.vertices)

        self.assertEqual(self.weights[1], output[self.id_1])
        self.assertEqual(self.weights[2], output[self.id_2])
        self.assertEqual(self.weights["abc"], output[self.id_abc])

    def test_weighted_vertex_data(self):
        mu = 2.
        zeta = 3.

        self.assertEqual(self.weights[1], self.vertices.weight(self.id_1))
        self.assertEqual(self.weights[2], self.vertices.weight(self.id_2))
        self.assertEqual(self.weights["abc"], self.vertices.weight(self.id_abc))

        self.vertices.resample_weights(np.random.default_rng())
        self.assertEqual(self.weights[1], self.vertices.weight(self.id_1))
        self.assertEqual(self.weights[2], self.vertices.weight(self.id_2))
        self.assertEqual(self.weights["abc"], self.vertices.weight(self.id_abc))

        self.assertEqual(2.5**mu * 2**zeta, self.vertices.penalty(self.id_1, self.id_2, mu=mu, zeta=zeta))
        self.assertEqual(2.5**mu * 3**mu * 2**zeta, self.vertices.penalty(self.id_2, self.id_abc, mu=mu, zeta=zeta))
        self.assertAlmostEqual(3.0**mu * 8**(zeta/2), self.vertices.penalty(self.id_1, self.id_abc, mu=mu, zeta=zeta),
                               delta=1e-13)

    def test_resample(self):
        """Checks that resample_weights is actually resampling every vertex weight."""
        entropy = 331375102187953107209086426124205679010
        generator = np.random.default_rng(seed=entropy)

        def weight_generator_fn(vertices: VertexSet, gen: np.random.Generator) -> List[float]:
            return [gen.random() for _ in vertices.names]
        weight_generator = GenericWeightGenerator(_function=weight_generator_fn, description="Test weight generator")

        data = WeightedVertexSet(self.vertex_gen, weight_generator, generator)
        first_sample = {id_: data.weights[id_] for id_ in self.vertices.ids}
        data.resample_weights(generator)
        second_sample = {id_: data.weights[id_] for id_ in self.vertices.ids}

        self.assertEqual(first_sample.keys(), second_sample.keys())
        for id_ in first_sample.keys():
            self.assertNotEqual(first_sample[id_], second_sample[id_])


class TestFromDegrees(unittest.TestCase):
    def test_from_degrees(self):
        degrees = {1: 40, 2: 20, "abc": 60, "def": 0}  # Average degree 30, penalty should be 20.
        generator = create_from_degrees_generator(degrees, "Test generator")

        vertices = VertexSet(dimension=2, metric=EuclideanDistance(d=2))
        points = {1: (0, 0), 2: (1, 1), "abc": (2, 2), "def": (3, 3)}
        vertices.set_points_from_names(points)
        output = generator(vertices)

        id_1 = vertices.name_to_id(1)
        self.assertEqual(20., output[id_1])
        id_2 = vertices.name_to_id(2)
        self.assertEqual(1., output[id_2])
        id_abc = vertices.name_to_id("abc")
        self.assertEqual(40., output[id_abc])
        id_def = vertices.name_to_id("def")
        self.assertEqual(1., output[id_def])


class TestPowerLaw(unittest.TestCase):
    def test_power_law(self):
        """Runs a DKW-based test to check that the weight distribution in a 100k-vertex set with tau=2.5 and
        scaling(w) = 2 is roughly what it should be, dividing into bins by taking the floor of each weight. The allowed
        error in total variation distance is .01."""
        entropy = 127743512994918990291592040963354975738  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)

        scaling = GenericEdgeWeightScaler(_function=lambda w, _: 2*w, description="Test scaler")
        tau = 2.5
        weight_generator = PowerLawWeightGenerator(tau=tau, ell=scaling)

        n = 100000
        point_dict = {i: (i,) for i in range(n)}
        vertex_gen = FixedVertexSet(metric=EuclideanDistance(d=1), points=point_dict, description="Test")
        test_instance = WeightedVertexSet(vertex_generator=vertex_gen, weight_generator=weight_generator,
                                          generator=generator)

        weight_counts = defaultdict(lambda: 0)
        for i in test_instance.vertices.names:
            weight_counts[floor(test_instance.weight(i))] += 1

        r"""Pr(floor(2*W) = i) = Pr(W <= (i+1)/2) - Pr(W <= i/2) = (2/i)^{\tau-1} - (2/(i+1))^{\tau-1}."""
        def target_pmf(i):
            if i <= 1:
                return 0
            return (i/2) ** (1 - tau) - ((i+1)/2) ** (1 - tau)

        # Check for stupid calculation errors
        self.assertAlmostEqual(sum([target_pmf(i) for i in range(100000)]), 1, delta=0.01)

        p_value = dkw_p_value(sample_data=weight_counts, pmf=target_pmf, tvd_bound=0.01)
        self.assertEqual(2.291165156475741e-07, p_value)


if __name__ == '__main__':
    unittest.main()
