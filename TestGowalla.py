import datetime
from pathlib import Path
import unittest

import graph_tool.topology  # type: ignore
import numpy as np

from Gowalla import CheckIn, GowallaDataCreator, GowallaSIEpidemic
from SIEpidemic import GenericEdgeCostGen
from TestDistribution import dkw_p_value


TEST_SAVE_FOLDER = Path.cwd() / "test_files"


def clear_test_files() -> None:
    for child in TEST_SAVE_FOLDER.iterdir():
        if child.is_file() and child.suffix == ".pickle":
            child.unlink()


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
        id_0, id_1 = GowallaDataCreator._parse_edge_line(test_line)
        self.assertEqual(id_0, 3542)
        self.assertEqual(id_1, 4466)


@unittest.skip("Let's not hammer the SNAP server unless we really want to test downloading specifically.")
class DownloadTests(unittest.TestCase):
    def test_download(self):
        """Deletes the data, then attempts to re-download it. Asserts the resulting files exist and have non-zero
        size."""
        GowallaDataCreator._VERTEX_DATA_PATH.unlink(missing_ok=True)
        GowallaDataCreator._EDGE_DATA_PATH.unlink(missing_ok=True)

        self.assertFalse(GowallaDataCreator._VERTEX_DATA_PATH.exists())
        self.assertFalse(GowallaDataCreator._EDGE_DATA_PATH.exists())

        GowallaDataCreator._obtain_snap_data()

        self.assertTrue(GowallaDataCreator._VERTEX_DATA_PATH.exists())
        self.assertGreater(GowallaDataCreator._VERTEX_DATA_PATH.stat().st_size, 0)
        self.assertTrue(GowallaDataCreator._EDGE_DATA_PATH.exists())
        self.assertGreater(GowallaDataCreator._EDGE_DATA_PATH.stat().st_size, 0)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entropy = 89654657203871215244168134687054070552
        rng = np.random.default_rng(seed=entropy)
        cls.instance = GowallaDataCreator(rng=rng, folder=TEST_SAVE_FOLDER)

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

    def test_save_load(self):
        clear_test_files()
        self.instance.save_files()
        edge_cost_gen = GenericEdgeCostGen(lambda _: 0, "Zero cost")
        test_epidemic = GowallaSIEpidemic(edge_cost_gen=edge_cost_gen, mu=1., zeta=2., name="test",
                                          rng=np.random.default_rng(), gowalla_folder=TEST_SAVE_FOLDER)

        original_vertices = self.instance.vertices
        original_weights = self.instance.weights
        loaded_vertices = test_epidemic.vertex_set.vertices
        loaded_weight_fn = test_epidemic.vertex_set.weight

        self.assertEqual(set(original_vertices.ids), set(loaded_vertices.ids))
        self.assertEqual(set(original_vertices.names), set(loaded_vertices.names))
        self.assertEqual(set(original_vertices.positions), set(loaded_vertices.positions))

        for id_ in original_vertices.ids:
            self.assertEqual(original_vertices.id_to_position(id_), loaded_vertices.id_to_position(id_))
            self.assertEqual(original_vertices.id_to_name(id_), loaded_vertices.id_to_name(id_))
            self.assertEqual(original_weights[id_], loaded_weight_fn(id_))

        original_edges = self.instance.edge_list
        original_edge_pairs = {(x[0], x[1]) for x in original_edges}
        loaded_edges = test_epidemic.graph.edges()
        loaded_edge_pairs = {(e.source(), e.target()) for e in loaded_edges}
        self.assertEqual(original_edge_pairs, loaded_edge_pairs)
        clear_test_files()


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        edge_cost_gen = GenericEdgeCostGen(lambda _: 0, "Zero cost")
        test_epidemic = GowallaSIEpidemic(edge_cost_gen=edge_cost_gen, mu=1., zeta=2., name="test",
                                          rng=np.random.default_rng())
        cls.vertex_set = test_epidemic.vertex_set
        cls.graph = test_epidemic.graph

    def test_parameters(self):
        self.assertEqual(self.graph.num_vertices(), 107092)
        self.assertEqual(self.graph.num_edges(), 456830)

        # Component sizes - note the presence of an obvious giant.
        comp, hist = graph_tool.topology.label_components(self.graph)
        sizes = sorted(int(size) for size in hist)
        self.assertEqual(len(sizes), 8577)
        self.assertEqual(sizes[0], 1)
        self.assertEqual(sizes[-2], 13)
        self.assertEqual(sizes[-1], 96953)

    def test_edge_directionality(self):
        self.assertFalse(self.graph.is_directed())

    def test_position_uniqueness(self):
        # This is kind of a lot of vertex pairs with equal positions, but I've double-checked against the data and
        # it's real. This is because the Gowalla dataset is already discretised a little bit, it's just not obvious
        # because they're using a very fine resolution plus maybe a fancier grid from a dedicated library (taking
        # spherical geometry into account). The easy way to tell this is to notice that a lot of users have multiple
        # check-ins from *exactly* the same location, which would be functionally impossible due to measurement error
        # if the dataset weren't discretised.
        self.assertGreater(self.vertex_set.size - len(self.vertex_set.positions), 15000)
        self.assertLess(self.vertex_set.size - len(self.vertex_set.positions), 20000)

    def test_specific_position(self):
        user_id = 21
        vertex_position = self.vertex_set.name_to_position(user_id)

        # This user has one clearly anomalous position at (40.73, -73.9891), then a large number clustered around
        # (33.66, -117.76) - we could plausibly end up with any of these depending on how we discretise.
        self.assertGreater(vertex_position[0], 33.5)
        self.assertLess(vertex_position[0], 34.2)
        self.assertGreater(vertex_position[1], -118.2)
        self.assertLess(vertex_position[1], -117.75)

    def test_specific_neighbourhood(self):
        user_id = 19
        vertex_id = self.vertex_set.name_to_id(user_id)

        # This user is joined to these user IDs, all of which are included in the graph.
        neighbour_user_ids = {0, 97, 133, 135, 138, 335, 382, 436, 521, 566, 3002, 4609, 4610, 4611, 4612, 4613, 4614,
                              4615, 4616, 4617, 4618, 4619, 4620}
        neighbour_vertex_ids = {self.vertex_set.name_to_id(x) for x in neighbour_user_ids}
        self.assertEqual(set(self.graph.get_all_neighbours(vertex_id)), neighbour_vertex_ids)

    def test_specific_exclusion(self):
        # This user has no location information available and shouldn't be included in the graph.
        self.assertNotIn(196579, self.vertex_set.names)


if __name__ == '__main__':
    unittest.main()
