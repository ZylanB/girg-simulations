from TestDistribution import dkw_p_value
from scipy.stats import poisson
import unittest
from VertexData import *
from collections import defaultdict


class TestVertexData(unittest.TestCase):
    @classmethod
    def setUp(cls):
        metric = (lambda x, y: 0.0 if x == y else 1.0)  # Unit metric
        cls.test_instance = VertexData(dimension=2, metric=metric)

    def test_properties(self):
        point_dict = {57: (1., 1.), "Frog": (2., 2.), 11.5: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        self.assertEqual(self.test_instance.positions, {(1., 1.), (2., 2.), (3., 3.)})
        self.assertEqual(self.test_instance.ids, {57, "Frog", 11.5})
        self.assertEqual(self.test_instance.size, 3)
        self.assertEqual(0.0, self.test_instance.distance(57, 57))
        self.assertEqual(1.0, self.test_instance.distance(57, "Frog"))

    def test_setpoints(self):
        point_list = [(1., 1.), (2., 2.), (3., 3.)]
        self.test_instance.setPoints(point_list)

        self.assertEqual(list(self.test_instance.id_to_position.values()), point_list)
        self.assertEqual(list(self.test_instance.position_to_id.keys()), point_list)

        for point_id, point_position in self.test_instance.id_to_position.items():
            self.assertEqual(self.test_instance.position_to_id[point_position], point_id)

    def test_setpointsfromids(self):
        point_dict = {57: (1., 1.), "Frog": (2., 2.), 11.5: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        self.assertEqual(self.test_instance.id_to_position, point_dict)

        for point_id, point_position in point_dict.items():
            self.assertEqual(self.test_instance.position_to_id[point_position], point_id)

    def test_getidsinannulus(self):
        point_dict = {0: (1., 1.), 1: (2., 2.), 2: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=0.5, outer_radius=0.9)
        self.assertEqual(set(points_found), set())

        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=0.9, outer_radius=1.1)
        self.assertEqual(set(points_found), {1, 2})

        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=1.1, outer_radius=2.0)
        self.assertEqual(set(points_found), set())

        # Annulus should be closed at inside
        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=0.0, outer_radius=0.9)
        self.assertEqual(set(points_found), {0})

        # Annulus should be closed at outside
        points_found = self.test_instance.getIdsInAnnulus(center=(0., 0.), inner_radius=0.5, outer_radius=1.0)
        self.assertEqual(set(points_found), {0, 1, 2})

    def test_getidsinball(self):
        point_dict = {0: (1., 1.), 1: (2., 2.), 2: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        points_found = self.test_instance.getIdsInBall(center=(1., 1.), radius=0.5)
        self.assertEqual(set(points_found), {0})

        points_found = self.test_instance.getIdsInBall(center=(1., 1.), radius=1.5)
        self.assertEqual(set(points_found), {0, 1, 2})

        points_found = self.test_instance.getIdsInBall(center=(0., 0.), radius=1.5)
        self.assertEqual(set(points_found), {0, 1, 2})

        # Ball should be closed
        points_found = self.test_instance.getIdsInBall(center=(1., 1.), radius=1.0)
        self.assertEqual(set(points_found), {0, 1, 2})


class TestLattice(unittest.TestCase):
    def test_1d(self):
        test_instance = lattice(dimension=1, size=6)
        point_set = {(0.,), (1.,), (2.,), (3.,), (4.,), (5.,)}
        self.assertEqual(set(test_instance.positions), point_set)

        self.assertEqual(test_instance.metric((0.,), (1.,)), 1.)
        self.assertEqual(test_instance.metric((0.,), (5.,)), 1.)
        self.assertEqual(test_instance.metric((0.,), (4.,)), 2.)

        point_ids = test_instance.getIdsInAnnulus((0., 0.), 1.5, 3.5)
        positions = {test_instance.id_to_position[point_id] for point_id in point_ids}
        self.assertEqual(positions, {(2.,), (3.,), (4.,)})

    def test_2d(self):
        test_instance = lattice(dimension=2, size=4)
        point_set = {(0., 0.), (0., 1.), (0., 2.), (0., 3.),
                     (1., 0.), (1., 1.), (1., 2.), (1., 3.),
                     (2., 0.), (2., 1.), (2., 2.), (2., 3.),
                     (3., 0.), (3., 1.), (3., 2.), (3., 3.)}
        self.assertEqual(set(test_instance.positions), point_set)

        self.assertEqual(test_instance.metric((0., 0.), (1., 0.)), 1.)
        self.assertEqual(test_instance.metric((0., 0.), (0., 1.)), 1.)
        self.assertEqual(test_instance.metric((0., 0.), (3., 0.)), 1.)
        self.assertEqual(test_instance.metric((0., 0.), (0., 3.)), 1.)
        self.assertEqual(test_instance.metric((0., 0.), (0., 2.)), 2.)
        self.assertEqual(test_instance.metric((0., 0.), (3., 3.)), pow(2, 0.5))

        point_ids = test_instance.getIdsInAnnulus((0., 3.), 1.1, 2.1)
        positions = {test_instance.id_to_position[point_id] for point_id in point_ids}
        self.assertEqual(positions, {(0., 1.), (1., 0), (1., 2.), (2., 3.), (3., 0.), (3., 2.)})


class TestEarthDistance(unittest.TestCase):
    def testSimple(self):
        bristol = (51.4545, -2.5879)
        delft = (52.0116, 4.3571)
        self.assertEqual(earthDistance(bristol, delft), 482.07966758305776)

        london = (51.6072, -0.1276)
        new_york = (40.7128, -74.0060)
        self.assertEqual(earthDistance(london, new_york), 5566.760358601733)


class TestPPP(unittest.TestCase):
    def testDistribution(self):
        """Runs a DKW-based test to check that the number of points in a uniformly-chosen 5x5 square within [0,
        10]^2 roughly follows a Poisson distribution with mean 25, and that the number of points in two disjoint 3x3
        squares (one from the lower-left quadrant and one from the upper-right quadrant) roughly follows a Poisson
        distribution with mean 18. In each case the allowed error in total variation distance is .01."""

        entropy = 207557186055428275376091733348063779829  # Generated from numpy via SeedSequence().entropy
        generator = np.random.default_rng(seed=entropy)

        big_point_counts = defaultdict(lambda: 0)
        small_point_counts = defaultdict(lambda: 0)

        for i in range(100000):
            instance = poissonPointProcess(dimension=2, size=10., generator=generator)

            # Lower-left corners for each square
            big_corner = generator.uniform(low=0., high=5., size=2)
            small_left_corner = generator.uniform(low=0., high=2., size=2)
            small_right_corner = generator.uniform(low=5., high=7., size=2)

            def in_square(p, corner, side):
                return corner[0] <= p[0] < corner[0] + side and corner[1] <= p[1] < corner[1] + side

            big_points = len([p for p in instance.positions if in_square(p, big_corner, 5.)])
            big_point_counts[big_points] += 1

            small_left_points = len([p for p in instance.positions if in_square(p, small_left_corner, 3.)])
            small_right_points = len([p for p in instance.positions if in_square(p, small_right_corner, 3.)])
            small_point_counts[small_left_points + small_right_points] += 1

        expected_big_pmf = lambda k: poisson.pmf(k=k, mu=25)
        big_result = dkw_p_value(sample_data=big_point_counts, pmf=expected_big_pmf, tvd_bound=.01)
        self.assertEqual(3.0276024398808153e-06, big_result)

        expected_small_pmf = lambda k: poisson.pmf(k=k, mu=18)
        small_result = dkw_p_value(sample_data=small_point_counts, pmf=expected_small_pmf, tvd_bound=.01)
        self.assertEqual(1.2098219064806303e-06, small_result)


if __name__ == '__main__':
    unittest.main()
