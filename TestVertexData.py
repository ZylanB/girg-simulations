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
        point_dict = {57: (1.,1.), "Frog": (2.,2.), ["Another ID type"]: (3.,3.)}
        self.test_instance.setPointsFromIds(point_dict)

if __name__ == '__main__':
    unittest.main()
