import unittest
import datetime
import numpy as np
from Gowalla import CheckIn, GowallaSIEpidemic

from TestDistribution import dkw_p_value


class ParsingTests(unittest.TestCase):
    def test_checkin(self):
        # Copied from Gowalla_totalCheckins.txt
        test_line = "2	2010-09-13T16:41:48Z	34.097924317	-118.325254783	1337593"
        test_checkin = CheckIn.from_line(test_line)
        self.assertEqual(test_checkin.user_id, 2)
        correct_time = datetime.datetime(year=2010, month=9, day=13, hour=16, minute=41, second=48)
        self.assertEqual(test_checkin.time, correct_time)
        self.assertEqual(test_checkin.latitude, 34.097924317)
        self.assertEqual(test_checkin.longitude, -118.325254783)
        self.assertEqual(test_checkin.location_id, 1337593)

    def test_edges(self):
        test_line = "3542\t4466\n".encode("utf-8")
        id_0, id_1 = GowallaSIEpidemic._parse_edge_line(test_line)
        self.assertEqual(id_0, 3542)
        self.assertEqual(id_1, 4466)


@unittest.skip("Let's not hammer the SNAP server unless we really want to test downloading specifically.")
class DownloadTests(unittest.TestCase):
    def test_download(self):
        """Deletes the data, then attempts to re-download it. Asserts the resulting files exist and have non-zero
        size."""
        GowallaSIEpidemic._VERTEX_DATA_PATH.unlink(missing_ok=True)
        GowallaSIEpidemic._EDGE_DATA_PATH.unlink(missing_ok=True)

        self.assertFalse(GowallaSIEpidemic._VERTEX_DATA_PATH.exists())
        self.assertFalse(GowallaSIEpidemic._EDGE_DATA_PATH.exists())

        GowallaSIEpidemic._obtain_gowalla_data()

        self.assertTrue(GowallaSIEpidemic._VERTEX_DATA_PATH.exists())
        self.assertGreater(GowallaSIEpidemic._VERTEX_DATA_PATH.stat().st_size, 0)
        self.assertTrue(GowallaSIEpidemic._EDGE_DATA_PATH.exists())
        self.assertGreater(GowallaSIEpidemic._EDGE_DATA_PATH.stat().st_size, 0)


class VertexTests(unittest.TestCase):
    def setUp(self):
        entropy = 89654657203871215244168134687054070552
        self.generator = np.random.default_rng(seed=entropy)

    def test_position_calculation_basic(self):
        check_ins = [
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=4., longitude=5., location_id=0),
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=23, time=datetime.datetime.now(), latitude=4., longitude=5., location_id=0)
        ]
        for i in range(50):
            self.assertEqual((4., 5.), GowallaSIEpidemic._get_user_position(check_ins, self.generator))

    def test_position_calculation_rounding_close(self):
        # Will return (6, 7) at least half the time if any of the points near (3, 50) don't round correctly to (3, 50).

        check_ins = [
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=2.9951, longitude=50., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3.0049, longitude=50., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3., longitude=49.9951, location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=3., longitude=50.0049, location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=10, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0)
        ]
        for i in range(50):
            self.assertEqual((3., 50.), GowallaSIEpidemic._get_user_position(check_ins, self.generator))

    def test_position_calculation_rounding_far(self):
        # Will return (10.25, 27) at least half the time if any of the points near (10.25, 27) round incorrectly to
        # (10.25, 27).
        check_ins = [
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.2449, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.2551, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=26.9949, location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27.0051, location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=10.25, longitude=27., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0),
            CheckIn(user_id=432, time=datetime.datetime.now(), latitude=6., longitude=7., location_id=0)
        ]
        for i in range(50):
            self.assertEqual((6., 7.), GowallaSIEpidemic._get_user_position(check_ins, self.generator))

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
            location = GowallaSIEpidemic._get_user_position(check_ins, self.generator)
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
        self.assertEqual(7.254142213666701e-05, p_value)


if __name__ == '__main__':
    unittest.main()
