import unittest
import textwrap
from SIExperiment import *
from SIEpidemic import SIEpidemic, GenericEdgeCostGenerator, GenericEdgeGenerator
from VertexSet import lattice
from WeightedVertexSet import WeightedVertexSet, GenericWeightGenerator
from copy import deepcopy


def _dummy_weight_generator(_):
    _dummy_weight_generator.x += 1
    return [_dummy_weight_generator.x, _dummy_weight_generator.x + .25, _dummy_weight_generator.x + .5,
            _dummy_weight_generator.x + .75]


_dummy_weight_generator.x = -1  # type: ignore

_graphs = [[],
           [(0, 1)],
           [(0, 1), (0, 2)],
           [(0, 1), (0, 2), (1, 2)],
           [(0, 1), (0, 2), (1, 2), (0, 3)],
           [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)],
           [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]]


def _dummy_edge_generator(_):
    _dummy_edge_generator.x += 1
    return _graphs[_dummy_edge_generator.x]


_dummy_edge_generator.x = -1  # type: ignore


def _dummy_cost_generator():
    _dummy_cost_generator.x += 1
    return _dummy_cost_generator.x


_dummy_cost_generator.x = -1  # type: ignore


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.log_path = Path.cwd() / "test_logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.clearTestFiles()

        self.entropy = 99217604857427484066604220485342406204
        self.generator = np.random.default_rng(seed=self.entropy)

        _dummy_edge_generator.x = -1
        _dummy_weight_generator.x = -1
        _dummy_cost_generator.x = -1

        weight_generator = GenericWeightGenerator(_dummy_weight_generator, "Test weight generator")
        edge_generator = GenericEdgeGenerator(_dummy_edge_generator, "Test edge generator")
        cost_generator = GenericEdgeCostGenerator(_dummy_cost_generator, "Test cost generator")

        unweighted_vertices = lattice(dimension=2, size=2)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator)
        self.epidemic = SIEpidemic(vertex_set=vertices, edge_cost_generator=cost_generator,
                                   edge_generator=edge_generator, mu=0.5, zeta=1.5)

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
                                  name="test", result_fn=self.result_fn)
        experiment.execute()
        self.assertEqual(experiment.results, [0, 1, 2, 3, 4, 5, 6])

    def testBasicLog(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=False,
                                  name="test", result_fn=self.result_fn)
        experiment.execute()

        self.assertTrue((self.log_path / f"test-results.pickle").exists())
        self.assertTrue((self.log_path / f"test-functions.pickle").exists())
        self.assertTrue((self.log_path / f"test-settings.cfg").exists())

        self.assertEqual(experiment.load_results(), [0, 1, 2, 3, 4, 5, 6])

    def testFullLog(self):
        saved_experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=True,
                                  name="test", result_fn=self.result_fn)
        saved_experiment.save()
        saved_experiment.execute()

        for i in range(7):
            self.assertTrue((self.log_path / f"test-run-{i}.pickle").exists())
            self.assertTrue((self.log_path / f"test-run-{i}.gt").exists())

        self.assertTrue((self.log_path / f"test-results.pickle").exists())
        self.assertTrue((self.log_path / f"test-functions.pickle").exists())
        self.assertTrue((self.log_path / f"test-settings.cfg").exists())

        loaded_experiment = SIExperiment.load_from_file(folder=self.log_path, name="test")

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
        """Currently this is a hack. This thing really should be able to load an SIEpidemic from settings, but it
        can't. The main obstacle to this is the vertex set, which we really don't want to pickle with every single
        run. Probably wants a LoggableFunction."""
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=True,
                                  name="test", result_fn=self.result_fn, seed=self.entropy)
        experiment.save()

        loaded_experiment = SIExperiment.load_from_file(folder=self.log_path, name="test")

        self.assertEqual(loaded_experiment.seed, experiment.seed)
        self.assertEqual(loaded_experiment.run_count, experiment.run_count)
        self.assertEqual(loaded_experiment.resample_edges, experiment.resample_edges)
        self.assertEqual(loaded_experiment.resample_costs, experiment.resample_costs)
        self.assertEqual(loaded_experiment.log_path, experiment.log_path)
        self.assertEqual(loaded_experiment.full_log, experiment.full_log)
        self.assertEqual(loaded_experiment.name, experiment.name)

        loaded_experiment.epidemic = deepcopy(experiment.epidemic)
        # Property maps get broken by a deepcopy operation
        loaded_graph = loaded_experiment.epidemic.graph
        loaded_experiment.epidemic.infection_times = loaded_graph.vertex_properties["infection_times"]
        loaded_experiment.epidemic.infectors = loaded_graph.vertex_properties["infectors"]
        loaded_experiment.epidemic.edge_costs = loaded_graph.edge_properties["edge_costs"]

        experiment.execute()

        # Sample functions don't get called on the first run.
        _dummy_edge_generator.x = 0
        _dummy_weight_generator.x = 0
        _dummy_cost_generator.x = 0
        loaded_experiment.execute()

        self.assertEqual(loaded_experiment.results, experiment.results)

    def testConfigFormat(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=self.initial_vertex_fn, log_path=self.log_path, full_log=False,
                                  name="test", result_fn=self.result_fn, seed=self.entropy)
        experiment.save()
        with open(self.log_path / "test-settings.cfg") as f:
            saved_settings = f.read()
        expected_cfg = textwrap.dedent(f"""\
            name==test
            log_path=={self.log_path}
            full_log==False
            run_count==7
            resample_edges==True
            resample_costs==True
            seed=={self.entropy}
            reset_seed==True
            mu==0.5
            zeta==1.5
            dimension==2
            vertex_description==Integer lattice containing all points in {{0, 1}}^2
            Initial vertex selector of type <class 'SIExperiment.GenericInitialVertexFunction'>:
            \tdescription==Zero vertex
            Test result extractor of type <class 'SIExperiment.GenericResultFunction'>:
            \tdescription==Edge count
            Edge cost generator of type <class 'SIEpidemic.GenericEdgeCostGenerator'>:
            \tdescription==Test cost generator
            Edge set generator of type <class 'SIEpidemic.GenericEdgeGenerator'>:
            \tdescription==Test edge generator
            Vertex weight generator of type <class 'WeightedVertexSet.GenericWeightGenerator'>:
            \tdescription==Test weight generator
            Distance function on vertex set of type <class 'VertexSet.TorusDistance'>:
            \td==2
            \tsize==2
            """)
        self.assertEqual(saved_settings, expected_cfg)

if __name__ == '__main__':
    unittest.main()
