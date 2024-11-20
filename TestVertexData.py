import unittest
from VertexData import *

class TestVertexData(unittest.TestCase):
    @classmethod
    def setUp(self):
        distance_function = (lambda x, y: 0.0 if x == y else 1.0)  # Unit metric
        self.test_instance = VertexData(dimension=2, distance_function=distance_function)

    def test_setpoints(self):
        point_list = [(1.,1.), (2.,2.), (3.,3.)]
        self.test_instance.setPoints(point_list)

        self.assertEqual(list(self.test_instance.id_to_position.values()), point_list)
        self.assertEqual(list(self.test_instance.position_to_id.keys()), point_list)

        for point_id, point_position in self.test_instance.id_to_position.items():
            self.assertEqual(self.test_instance.position_to_id[point_position], point_id)

    def test_setpointsfromids(self):
        point_dict = {57: (1.,1.), "Frog": (2.,2.), 11.5: (3.,3.)}
        self.test_instance.setPointsFromIds(point_dict)

        self.assertEqual(self.test_instance.id_to_position, point_dict)

        for point_id, point_position in point_dict.items():
            self.assertEqual(self.test_instance.position_to_id[point_position], point_id)

    def test_getidsinannulus(self):
        point_dict = {0: (1.,1.), 1: (2.,2.), 2: (3.,3.)}
        self.test_instance.setPointsFromIds(point_dict)

        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=0.5, outer_radius=0.9)
        self.assertEqual(set(points_found), set())

        points_found = self.test_instance.getIdsInAnnulus(center=(1.,1.), inner_radius=0.9, outer_radius=1.1)
        self.assertEqual(set(points_found), {1,2})

        points_found = self.test_instance.getIdsInAnnulus(center=(1., 1.), inner_radius=1.1, outer_radius=2.0)
        self.assertEqual(set(points_found), set())

        # Annulus should be closed at inside
        points_found = self.test_instance.getIdsInAnnulus(center=(1.,1.), inner_radius=0.0, outer_radius=0.9)
        self.assertEqual(set(points_found), {0})

        # Annulus should be closed at outside
        points_found = self.test_instance.getIdsInAnnulus(center=(0.,0.), inner_radius=0.5, outer_radius=1.0)
        self.assertEqual(set(points_found), {0,1,2})

    def test_getidsinball(self):
        point_dict = {0: (1., 1.), 1: (2., 2.), 2: (3., 3.)}
        self.test_instance.setPointsFromIds(point_dict)

        points_found = self.test_instance.getIdsInBall(center=(1.,1.), radius=0.5)
        self.assertEqual(set(points_found), {0})

        points_found = self.test_instance.getIdsInBall(center=(1., 1.), radius=1.5)
        self.assertEqual(set(points_found), {0,1,2})

        points_found = self.test_instance.getIdsInBall(center=(0., 0.), radius=1.5)
        self.assertEqual(set(points_found), {0,1,2})

        # Ball should be closed
        points_found = self.test_instance.getIdsInBall(center=(1., 1.), radius=1.0)
        self.assertEqual(set(points_found), {0,1,2})

if __name__ == '__main__':
    unittest.main()
