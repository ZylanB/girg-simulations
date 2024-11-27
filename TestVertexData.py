import unittest
from VertexData import *


class TestVertexData(unittest.TestCase):
    @classmethod
    def setUp(cls):
        distance_function = (lambda x, y: 0.0 if x == y else 1.0)  # Unit metric
        cls.test_instance = VertexData(dimension=2, distance_function=distance_function)

    def test_properties(self):
        point_dict = {57: (1., 1.), "Frog": (2., 2.), 11.5: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        self.assertEqual(set(self.test_instance.positions), {(1., 1.), (2., 2.), (3., 3.)})
        self.assertEqual(set(self.test_instance.ids), {57, "Frog", 11.5})

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

        self.assertEqual(test_instance.distance((0.,), (1.,)), 1.)
        self.assertEqual(test_instance.distance((0.,), (5.,)), 1.)
        self.assertEqual(test_instance.distance((0.,), (4.,)), 2.)

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

        self.assertEqual(test_instance.distance((0., 0.), (1., 0.)), 1.)
        self.assertEqual(test_instance.distance((0., 0.), (0., 1.)), 1.)
        self.assertEqual(test_instance.distance((0., 0.), (3., 0.)), 1.)
        self.assertEqual(test_instance.distance((0., 0.), (0., 3.)), 1.)
        self.assertEqual(test_instance.distance((0., 0.), (0., 2.)), 2.)
        self.assertEqual(test_instance.distance((0., 0.), (3., 3.)), pow(2, 0.5))

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


if __name__ == '__main__':
    unittest.main()
