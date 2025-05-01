import unittest
import datetime
import numpy as np
from Gowalla import *

from TestDistribution import dkw_p_value


class ParsingTests(unittest.TestCase):
    def test_checkin(self):
        # Copied from Gowalla_totalCheckins.txt
        test_line = "2\t2010-09-13T16:41:48Z\t34.097924317\t-118.325254783\t1337593".encode("utf-8")
        test_checkin = CheckIn.from_line(test_line)
        self.assertEqual(test_checkin.user_id, 2)
        correct_time = datetime.datetime(year=2010, month=9, day=13, hour=16, minute=41, second=48)
        self.assertEqual(test_checkin.time, correct_time)
        self.assertEqual(test_checkin.latitude, 34.097924317)
        self.assertEqual(test_checkin.longitude, -118.325254783)
        self.assertEqual(test_checkin.location_id, 1337593)

    def test_edges(self):
        test_line = "3542\t4466\n".encode("utf-8")
        id_0, id_1 = GowallaDataReader._parse_edge_line(test_line)
        self.assertEqual(id_0, 3542)
        self.assertEqual(id_1, 4466)


@unittest.skip("Let's not hammer the SNAP server unless we really want to test downloading specifically.")
class DownloadTests(unittest.TestCase):
    def test_download(self):
        """Deletes the data, then attempts to re-download it. Asserts the resulting files exist and have non-zero
        size."""
        GowallaDataReader._VERTEX_DATA_PATH.unlink(missing_ok=True)
        GowallaDataReader._EDGE_DATA_PATH.unlink(missing_ok=True)

        self.assertFalse(GowallaDataReader._VERTEX_DATA_PATH.exists())
        self.assertFalse(GowallaDataReader._EDGE_DATA_PATH.exists())

        GowallaDataReader._obtain_snap_data()

        self.assertTrue(GowallaDataReader._VERTEX_DATA_PATH.exists())
        self.assertGreater(GowallaDataReader._VERTEX_DATA_PATH.stat().st_size, 0)
        self.assertTrue(GowallaDataReader._EDGE_DATA_PATH.exists())
        self.assertGreater(GowallaDataReader._EDGE_DATA_PATH.stat().st_size, 0)


class VertexPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entropy = 89654657203871215244168134687054070552
        generator = np.random.default_rng(seed=entropy)
        cls.instance = GowallaDataReader(generator=generator)

    def test_position_calculation_basic(self):
        check_ins = [
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=4., longitude=5., location_id=0),
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=4., longitude=5., location_id=0)
        ]
        for i in range(50):
            self.assertEqual((4., 5.), self.instance._get_user_position(check_ins)[0])

    def test_position_calculation_rounding_close(self):
        # Will return (6, 7) at least half the time if any of the points near (3, 50) don't round correctly to (3, 50).
        # Otherwise returns one of the four points close to (3, 50) chosen uniformly at random.

        check_ins = [
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=2.8751, longitude=50., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3.1249, longitude=50., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3., longitude=49.8751, location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3., longitude=50.1249, location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0)
        ]
        for i in range(50):
            position = self.instance._get_user_position(check_ins)[0]
            self.assertIn(position, [(2.8751, 50.), (3.1249, 50.), (3., 49.8751), (3, 50.1249)])

    def test_position_calculation_rounding_far(self):
        # Will return a point near (10.25, 27) at least half the time if any of the points near (10.25, 27) round
        # incorrectly to (10.25, 27). Otherwise, returns (6, 7).
        check_ins = [
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.1249, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.3751, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=26.8749, location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27.1251, location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0)
        ]
        for i in range(50):
            self.assertEqual((6., 7.), self.instance._get_user_position(check_ins)[0])

    def test_position_calcuation_ties(self):
        # Should return (53, 27) half the time and (69, 11) the rest of the time.
        check_ins = [
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=53., longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10., longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=69., longitude=11., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=53., longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=69., longitude=11., location_id=0),
        ]

        location_counts = {0: 0, 1: 0, 2: 0}
        for i in range(100000):
            location = self.instance._get_user_position(check_ins)[0]
            if location == (53., 27.):
                location_counts[0] += 1
            elif location == (69., 11.):
                location_counts[1] += 1
            else:
                location_counts[2] += 1

        def target_pmf(x):
            if x == 0 or x == 1:
                return .5
            return 0.

        p_value = dkw_p_value(sample_data=location_counts, pmf=target_pmf, tvd_bound=0.01)
        self.assertEqual(6.137472530978548e-09, p_value)


class GraphTests(unittest.TestCase):
    def test_specific_vertex(self):
        pass


if __name__ == '__main__':
    unittest.main()
