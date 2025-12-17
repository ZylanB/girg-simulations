from figures.HeatMaps import get_map_to_torus
import unittest


class ProjectionTests(unittest.TestCase):
    def test_torus_projection(self):
        map_to_torus = get_map_to_torus(centre_x=1, centre_y=1, side=2)
        self.assertEqual(map_to_torus(0, 0), (0, 0))
        self.assertEqual(map_to_torus(-1, 1), (1, 1))
        self.assertEqual(map_to_torus(-3, 0), (1, 0))
        self.assertEqual(map_to_torus(5, 3), (1, 1))
        self.assertEqual(map_to_torus(0, -5.5), (0, 0.5))

        map_to_torus = get_map_to_torus(centre_x=6, centre_y=8, side=10)
        self.assertEqual(map_to_torus(6, 8), (5, 5))
        self.assertEqual(map_to_torus(5, 9), (4, 6))
        self.assertEqual(map_to_torus(7, 7), (6, 4))
        self.assertEqual(map_to_torus(6, 1), (5, 8))
        self.assertEqual(map_to_torus(0, 8), (9, 5))

        map_to_torus = get_map_to_torus(centre_x=1, centre_y=1, side=2)
        self.assertEqual(map_to_torus(0, 0), (0, 0))
        self.assertEqual(map_to_torus(0, 2), (0, 0))
        self.assertEqual(map_to_torus(2, 2), (0, 0))
        self.assertEqual(map_to_torus(2, 0), (0, 0))


if __name__ == '__main__':
    unittest.main()
