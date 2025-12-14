from hashlib import sha256
from pathlib import Path

import numpy as np

import config
from Gowalla import GowallaGiantGraph
from SIEpidemic import FixedGraphGen, ConstantCostGen, SIEpidemic
from VertexSet import FixedVertexSet, TorusDistance, EarthDistance
from WeightedVertexSet import FixedWeightGen, WeightedVertexSet
from figures.HeatMaps import EpidemicHeatMap, get_map_to_torus, HeatMapMode
import unittest

from figures.figures_common import GOWALLA_INITIAL


class ProjectionTests(unittest.TestCase):
    def test_torus_projection(self):
        map_to_torus = get_map_to_torus(centre_x=1, centre_y=1, side=2)
        self.assertEqual(map_to_torus(0, 0), (-1, -1))
        self.assertEqual(map_to_torus(-1, 1), (0, 0))
        self.assertEqual(map_to_torus(-3, 0), (0, -1))
        self.assertEqual(map_to_torus(5, 3), (0, 0))
        self.assertEqual(map_to_torus(0, -5.5), (-1, -0.5))

        map_to_torus = get_map_to_torus(centre_x=6, centre_y=8, side=10)
        self.assertEqual(map_to_torus(6, 8), (0, 0))
        self.assertEqual(map_to_torus(5, 9), (-1, 1))
        self.assertEqual(map_to_torus(7, 7), (1, -1))
        self.assertEqual(map_to_torus(6, 1), (0, 3))
        self.assertEqual(map_to_torus(0, 8), (4, 0))

        map_to_torus = get_map_to_torus(centre_x=1, centre_y=1, side=2)
        self.assertEqual(map_to_torus(0, 0), (-1, -1))
        self.assertEqual(map_to_torus(0, 2), (-1, -1))
        self.assertEqual(map_to_torus(2, 2), (-1, -1))
        self.assertEqual(map_to_torus(2, 0), (-1, -1))


class HeatMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test folder, delete any existing data
        config.TEST_FOLDER.mkdir(exist_ok=True, parents=True)
        for entry in config.TEST_FOLDER.iterdir():
            if entry.is_file:
                entry.unlink()

    def test_torus_mode(self):
        point_dict = {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}
        vertex_gen = FixedVertexSet(points=point_dict, metric=TorusDistance(d=2, size=2), description="test")
        weights = {x: 1. for x in point_dict.keys()}
        weight_gen = FixedWeightGen(weights=weights, description="Test weights")
        vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=np.random.default_rng())

        edges = [("b", "c"), ("c", "d"), ("d", "a")]
        edge_gen = FixedGraphGen(edges=edges, description="Test graph")
        cost_gen = ConstantCostGen(1.)

        epidemic = SIEpidemic(vertex_set=vertices, edge_gen=edge_gen, name="test", mu=0., zeta=0.,
                              cost_gen=cost_gen, rng=np.random.default_rng())
        epidemic.run_infection(initial_vertex_id=vertices.name_to_id("b"))
        heatmap = EpidemicHeatMap(x_pixels=4, y_pixels=4, origin_x=0, origin_y=1, epidemic=epidemic,
                                  mode=HeatMapMode.TORUS)

        """The infection order will be b -> c -> d -> a. The new torus will be centered at b and projected onto
        [-1, -1]^2, so (0, -1) will be added to every vertex and the order will become (0, 0) -> (-1, -1) -> (-1, 0) -> 
        (0, -1). There are four pixels per row and the remaining 12 should all be blank (-1)."""
        expected_mesh = [[-1, -1, -1, -1],
                         [ 2, -1,  0, -1],
                         [-1, -1, -1, -1],
                         [ 1, -1,  3, -1]][::-1]  # Expected_mesh[0] should be the bottom row of pixels not the top row
        self.assertEqual(expected_mesh, heatmap.heatmap_mesh)

        """This is a quick sanity-check that the figure looks correct. The expected figure will need to be updated 
        and manually rechecked whenever the plot format changes - it should be stored in the same folder as this
        source file. The infection sequence should go yellow -> orange -> purple -> blue."""
        output_path = config.TEST_FOLDER / "torus_output.png"
        heatmap.export_to_canvas(save_path=output_path)

        expected_path = Path(__file__).resolve().parent / "expected_torus_output.png"
        if not expected_path.exists():
            raise Exception(f"The reference data for the image output check at {expected_path} does not exist. "
                            f"Check the output manually, and if it looks correct, copy it to {expected_path}.")

        output_hash = expected_hash = None
        with open(output_path, "rb") as file:
            hash_fn = sha256()
            hash_fn.update(file.read())
            output_hash = hash_fn.hexdigest()
        with open(expected_path, "rb") as file:
            hash_fn = sha256()
            hash_fn.update(file.read())
            expected_hash = hash_fn.hexdigest()
        self.assertEqual(output_hash, expected_hash)

    def test_europe_mode(self):
        initial_pos = GowallaGiantGraph().graph_data.vertex_set.name_to_position(GOWALLA_INITIAL)
        point_dict = {"Patient zero": initial_pos, "London": (51.50, -0.08), "Zurich": (47.37, 8.54),
                      "Delft": (52.01, 4.36), "Kyiv": (50.43, 30.54)}
        vertex_gen = FixedVertexSet(points=point_dict, metric=EarthDistance(), description="test")
        weights = {x: 1. for x in point_dict.keys()}
        weight_gen = FixedWeightGen(weights=weights, description="Test weights")
        vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=np.random.default_rng())

        edges = [("Patient zero", "London"), ("London", "Zurich"), ("Zurich", "Kyiv"), ("Kyiv", "Delft")]
        edge_gen = FixedGraphGen(edges=edges, description="Test graph")
        cost_gen = ConstantCostGen(1.)

        epidemic = SIEpidemic(vertex_set=vertices, edge_gen=edge_gen, name="test", mu=0., zeta=0.,
                              cost_gen=cost_gen, rng=np.random.default_rng())
        epidemic.run_infection(initial_vertex_id=vertices.name_to_id("Patient zero"))
        heatmap = EpidemicHeatMap(x_pixels=400, y_pixels=400, origin_x=initial_pos[1], origin_y=initial_pos[0],
                                  epidemic=epidemic, mode=HeatMapMode.EUROPE)

        """This is a quick sanity-check that the figure looks correct. The expected figure will need to be updated 
        and manually rechecked whenever the plot format changes - it should be stored in the same folder as this
        source file. The infection sequence should go yellow -> orange -> purple -> blue, and the order should be
        GOWALLA_INITIAL -> London -> Zurich -> Kyiv -> Delft."""
        output_path = config.TEST_FOLDER / "europe_output.png"
        heatmap.export_to_canvas(save_path=output_path)

        expected_path = Path(__file__).resolve().parent / "expected_europe_output.png"
        if not expected_path.exists():
            raise Exception(f"The reference data for the image output check at {expected_path} does not exist. "
                            f"Check the output manually, and if it looks correct, copy it to {expected_path}.")

        output_hash = expected_hash = None
        with open(output_path, "rb") as file:
            hash_fn = sha256()
            hash_fn.update(file.read())
            output_hash = hash_fn.hexdigest()
        with open(expected_path, "rb") as file:
            hash_fn = sha256()
            hash_fn.update(file.read())
            expected_hash = hash_fn.hexdigest()
        self.assertEqual(output_hash, expected_hash)


if __name__ == '__main__':
    unittest.main()
