import unittest
from TestDistribution import dkw_p_value
from WeightedVertexSet import *
from VertexSet import VertexSet, euclideanDistanceFunction
from collections import defaultdict
from math import floor


class BasicTests(unittest.TestCase):
    def setUp(self):
        self.weights = {1: 1., 2: 2.5, "abc": 3.}
        self.generator = fixed_weights_generator(self.weights)

        self.vertices = VertexSet(dimension=2, metric=euclideanDistanceFunction(d=2))
        points = {1: (0, 0), 2: (2, 0), "abc": (2, 2)}
        self.vertices.setPointsFromNames(points)

        self.id_1 = self.vertices.name_to_id(1)
        self.id_2 = self.vertices.name_to_id(2)
        self.id_abc = self.vertices.name_to_id("abc")

    def testFixedWeights(self):
        output = self.generator(self.vertices)

        self.assertEqual(self.weights[1], output[self.id_1])
        self.assertEqual(self.weights[2], output[self.id_2])
        self.assertEqual(self.weights["abc"], output[self.id_abc])

    def testWeightedVertexData(self):
        mu = 2.
        zeta = 3.
        data = WeightedVertexSet(self.vertices, self.generator, mu=mu, zeta=zeta)

        self.assertEqual(self.weights[1], data.weight(self.id_1))
        self.assertEqual(self.weights[2], data.weight(self.id_2))
        self.assertEqual(self.weights["abc"], data.weight(self.id_abc))

        data.resample_weights()
        self.assertEqual(self.weights[1], data.weight(self.id_1))
        self.assertEqual(self.weights[2], data.weight(self.id_2))
        self.assertEqual(self.weights["abc"], data.weight(self.id_abc))

        self.assertEqual(2.5 ** mu * 2 ** zeta, data.penalty(self.id_1, self.id_2))
        self.assertEqual(2.5 ** mu * 3 ** mu * 2 ** zeta, data.penalty(self.id_2, self.id_abc))
        self.assertAlmostEqual(3.0 ** mu * 8 ** (zeta/2), data.penalty(self.id_1, self.id_abc), delta=1e-13)

    def testResample(self):
        """Checks that resample_weights is actually resampling every vertex weight."""
        entropy = 331375102187953107209086426124205679010
        generator = np.random.default_rng(seed=entropy)

        def weight_generator(vertices: VertexSet) -> Dict[Any, float]:
            return {id_: generator.random() for id_ in vertices.names}

        data = WeightedVertexSet(self.vertices, weight_generator, mu=0., zeta=0.)
        first_sample = {id_: data.weights[id_] for id_ in self.vertices.names}
        data.resample_weights()
        second_sample = {id_: data.weights[id_] for id_ in self.vertices.names}

        self.assertEqual(first_sample.keys(), second_sample.keys())
        for id_ in first_sample.keys():
            self.assertNotEqual(first_sample[id_], second_sample[id_])


class TestFromDegrees(unittest.TestCase):
    def testFromDegrees(self):
        degrees = {1: 40, 2: 20, "abc": 60, "def": 0}  # Average degree 30, penalty should be 20.
        generator = from_degrees_generator(degrees)

        vertices = VertexSet(dimension=2, metric=euclideanDistanceFunction(d=2))
        points = {1: (0, 0), 2: (1, 1), "abc": (2, 2), "def": (3, 3)}
        vertices.setPointsFromNames(points)
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
    def testPowerLaw(self):
        """Runs a DKW-based test to check that the weight distribution in a 100k-vertex set with tau=2.5 and
        scaling(w) = 2 is roughly what it should be, dividing into bins by taking the floor of each weight. The allowed
        error in total variation distance is .01."""
        entropy = 127743512994918990291592040963354975738  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)

        scaling = lambda w: 2*w
        tau = 2.5
        weight_generator = power_law_generator(tau=tau, ell=scaling, generator=generator)

        n = 100000
        vertices = VertexSet(dimension=1, metric=euclideanDistanceFunction(d=1))
        vertices.setPoints([(i,) for i in range(n)])
        test_instance = WeightedVertexSet(vertices=vertices, weight_generator=weight_generator, mu=0.0, zeta=0.0)

        weight_counts = defaultdict(lambda: 0)
        for i in vertices.names:
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
