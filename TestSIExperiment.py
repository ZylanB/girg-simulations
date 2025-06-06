import unittest
import textwrap

import numpy as np
from pathlib import Path

from SIEpidemic import SIEpidemic, GenericEdgeCostGen, GenericEdgeGen
from SIExperiment import SIExperiment, GenericInitialVertexFunction, GenericResultFunction
from VertexSet import Lattice
from WeightedVertexSet import WeightedVertexSet, GenericWeightGen


def _dummy_weight_gen(_, __):
    _dummy_weight_gen.x += 1
    return [_dummy_weight_gen.x, _dummy_weight_gen.x + .25, _dummy_weight_gen.x + .5,
            _dummy_weight_gen.x + .75]


_dummy_weight_gen.x = -1  # type: ignore

_graphs = [[],
           [(0, 1)],
           [(0, 1), (0, 2)],
           [(0, 1), (0, 2), (1, 2)],
           [(0, 1), (0, 2), (1, 2), (0, 3)],
           [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)],
           [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]]


def _dummy_edge_gen(_, __):
    _dummy_edge_gen.x += 1
    return _graphs[_dummy_edge_gen.x]


_dummy_edge_gen.x = -1  # type: ignore


def _dummy_cost_gen(_):
    _dummy_cost_gen.x += 1
    return _dummy_cost_gen.x


_dummy_cost_gen.x = -1  # type: ignore


def _reset_gen_globals() -> None:
    _dummy_edge_gen.x = -1  # type: ignore
    _dummy_weight_gen.x = -1  # type: ignore
    _dummy_cost_gen.x = -1  # type: ignore


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.log_path = Path.cwd() / "test_files"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.clearTestFiles()

        self.entropy = 99217604857427484066604220485342406204
        self.rng = np.random.default_rng(seed=self.entropy)

        _reset_gen_globals()

        weight_gen = GenericWeightGen(_dummy_weight_gen, "Test weight generator")
        edge_gen = GenericEdgeGen(_dummy_edge_gen, "Test edge generator")
        cost_gen = GenericEdgeCostGen(_dummy_cost_gen, "Test cost generator")

        vertex_gen = Lattice(dimension=2, size=2)
        vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=self.rng)
        self.epidemic = SIEpidemic(vertex_set=vertices, edge_cost_gen=cost_gen, rng=self.rng,
                                   edge_gen=edge_gen, mu=0.5, zeta=1.5, name="test")

        self.initial_vertex_fn = GenericInitialVertexFunction(lambda gen: 0, "Zero vertex")
        self.result_fn = GenericResultFunction(lambda graph, gen: graph.num_edges(), "Edge count")

    def tearDown(self):
        self.clearTestFiles()

    def clearTestFiles(self):
        # Clear out existing files from previous tests to make sure new ones are created.
        for child in self.log_path.iterdir():
            if child.is_file() and child.suffix in [".gt", ".cfg", ".pickle"]:
                child.unlink()

    def testExecute(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=False,
                                  name="test", result_fn=self.result_fn, resample_vertices=False,
                                  resample_weights=False)
        experiment.execute()
        self.assertEqual(experiment.results, [0, 1, 2, 3, 4, 5, 6])

    def testBasicLog(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=False,
                                  name="test", result_fn=self.result_fn, resample_vertices=False,
                                  resample_weights=False)
        experiment.execute()

        self.assertTrue((self.log_path / f"test-results.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-functions.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-settings.cfg").exists())
        self.assertTrue((self.log_path / f"test-epidemic-settings.pickle").exists())
        self.assertTrue((self.log_path / f"test-vertex-settings.pickle").exists())

        for i in range(7):
            self.assertFalse((self.log_path / f"test-vertices-run-{i}.pickle").exists())
            self.assertFalse((self.log_path / f"test-graph-run-{i}.gt").exists())
        self.assertFalse((self.log_path / f"test-vertices.pickle").exists())

        self.assertEqual(experiment.load_results(), [0, 1, 2, 3, 4, 5, 6])

    def testFullLog(self):
        saved_experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=True,
                                  name="test", result_fn=self.result_fn, resample_vertices=True,
                                  resample_weights=False)
        saved_experiment.save_config()
        saved_experiment.execute()

        self.assertTrue((self.log_path / f"test-results.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-functions.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-settings.cfg").exists())
        self.assertTrue((self.log_path / f"test-epidemic-settings.pickle").exists())
        self.assertTrue((self.log_path / f"test-vertex-settings.pickle").exists())

        for i in range(7):
            self.assertTrue((self.log_path / f"test-vertices-run-{i}.pickle").exists())
            self.assertTrue((self.log_path / f"test-graph-run-{i}.gt").exists())
        self.assertFalse((self.log_path / f"test-vertices.pickle").exists())

        loaded_experiment = SIExperiment.load_from_file(folder=self.log_path, name="test", rerun=False)

        self.assertEqual(loaded_experiment.results, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(loaded_experiment.load_results(), [0, 1, 2, 3, 4, 5, 6])

        for i in range(7):
            loaded_run = loaded_experiment.load_run(i)
            self.assertEqual(self.epidemic.mu, 0.5)
            self.assertEqual(self.epidemic.zeta, 1.5)

            original_edges = set(_graphs[i])
            loaded_edges = {(int(e.source()), int(e.target())) for e in loaded_run.graph.edges()}
            self.assertEqual(original_edges, loaded_edges)

    def testConfig(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=True,
                                  name="test", result_fn=self.result_fn, seed=self.entropy, resample_vertices=False,
                                  resample_weights=False)
        experiment.save_config()
        experiment.execute()

        _reset_gen_globals()
        loaded_experiment = SIExperiment.load_from_file(folder=self.log_path, name="test", rerun=True,
                                                        use_old_seed=True)
        loaded_experiment.execute()

        self.assertEqual(loaded_experiment.seed, experiment.seed)
        self.assertEqual(loaded_experiment.run_count, experiment.run_count)
        self.assertEqual(loaded_experiment.resample_edges, experiment.resample_edges)
        self.assertEqual(loaded_experiment.resample_costs, experiment.resample_costs)
        self.assertEqual(loaded_experiment.log_path, experiment.log_path)
        self.assertEqual(loaded_experiment.full_log, experiment.full_log)
        self.assertEqual(loaded_experiment.name, experiment.name)
        self.assertEqual(loaded_experiment.results, experiment.results)

    def testConfigFormat(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=False,
                                  name="test", result_fn=self.result_fn, seed=self.entropy, resample_vertices=False,
                                  resample_weights=False)
        experiment.save_config()
        with open(self.log_path / "test-experiment-settings.cfg") as f:
            saved_settings = f.read()
        expected_cfg = textwrap.dedent(f"""\
            name==test
            log_path==/mnt/e/GitHub/girg-simulations/test_files
            full_log==False
            run_count==7
            resample_edges==True
            resample_costs==True
            resample_weights==False
            resample_vertices==False
            seed==99217604857427484066604220485342406204
            mu==0.5
            zeta==1.5
            Initial vertex selector of type <class 'SIExperiment.GenericInitialVertexFunction'>:
            \tdescription==Zero vertex
            Test result extractor of type <class 'SIExperiment.GenericResultFunction'>:
            \tdescription==Edge count
            Edge cost generator of type <class 'SIEpidemic.GenericEdgeCostGen'>:
            \tdescription==Test cost generator
            Edge set generator of type <class 'SIEpidemic.GenericEdgeGen'>:
            \tdescription==Test edge generator
            Vertex weight generator of type <class 'WeightedVertexSet.GenericWeightGen'>:
            \tdescription==Test weight generator
            Distance function on vertex set of type <class 'VertexSet.TorusDistance'>:
            \tdimension==2
            \tsize==2
            Vertex set generator of type <class 'VertexSet.Lattice'>:
            \tsize==2
            \tdimension==2
            Vertex weight generator of type <class 'WeightedVertexSet.GenericWeightGen'>:
            \tdescription==Test weight generator
            """)
        self.assertEqual(saved_settings, expected_cfg)


if __name__ == '__main__':
    unittest.main()
