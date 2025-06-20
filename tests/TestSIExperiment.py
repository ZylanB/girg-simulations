from pathlib import Path
import unittest
import textwrap

from SIEpidemic import GenericEdgeCostGen, GenericEdgeGen, FPPCostGen, GirgGen
from SIExperiment import SIExperiment, GenericInitialVertexFunction, GenericResultFunction
from VertexSet import GenericVertexSetGen, FixedVertexSet, EuclideanDistance, PoissonPointProcess
from WeightedVertexSet import GenericWeightGen, PowerLawWeightGen, IdentityWeightScaler

"""These dummy generator functions must all be defined outside the test class or their closure will include the test 
class, which in turn includes an epidemic, which in turn includes a graph-tools graph. This means that when they get 
pickled as part of SIEpidemic.save_config, dill pickles their closure along with them, pickling a graph-tools graph 
and raising an exception. This only matters when writing unit tests."""


TEST_DIR = Path.cwd() / "test_files"


def _test_weight_gen(vertices, _):
    _test_weight_gen.x += 1
    return [_test_weight_gen.x + .25 * i for i in range(vertices.size)]


_test_weight_gen.x = -1  # type: ignore

_graphs = [[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)],
           [],
           [(0, 1)],
           [(0, 1), (0, 2)],
           [(0, 1), (0, 2), (1, 2)],
           [(0, 1), (0, 2), (1, 2), (0, 3)],
           [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]]


def _test_edge_gen(_, __):
    _test_edge_gen.x += 1
    return _graphs[_test_edge_gen.x]


_test_edge_gen.x = -1  # type: ignore


def _test_cost_gen(_):
    _test_cost_gen.x += 1
    return _test_cost_gen.x


_test_cost_gen.x = -1  # type: ignore


def _test_vertex_gen(_):
    _test_vertex_gen.x += 1
    vertices = {i: (float(i),) for i in range(_test_vertex_gen.x)}
    vertex_set = FixedVertexSet(vertices, metric=EuclideanDistance(d=1), description="test")(_)
    return vertex_set


_test_vertex_gen.x = 3  # type: ignore


def _reset_gen_globals() -> None:
    _test_edge_gen.x = -1  # type: ignore
    _test_weight_gen.x = -1  # type: ignore
    _test_cost_gen.x = -1  # type: ignore
    _test_vertex_gen.x = 3  # type: ignore


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.log_path = TEST_DIR
        self.log_path.mkdir(parents=True, exist_ok=True)
        clear_test_files()

        self.entropy = 99217604857427484066604220485342406204

        _reset_gen_globals()

        self.varying_weight_gen = GenericWeightGen(_test_weight_gen, "Varying test weight generator")
        self.varying_edge_gen = GenericEdgeGen(_test_edge_gen, "Varying test edge generator")
        self.varying_cost_gen = GenericEdgeCostGen(_test_cost_gen, "Varying test cost generator")
        self.varying_vertex_gen = GenericVertexSetGen(_test_vertex_gen, "Varying test vertex generator")

        self.initial_vertex_fn = GenericInitialVertexFunction(lambda _, __: 0, "Zero vertex")
        self.edge_count = GenericResultFunction(lambda epi, _: epi.graph.num_edges(), "Edge count")
        self.sum_weights = GenericResultFunction(lambda epi, _: sum(epi.vertex_set.weights), description="Total weight")
        self.vertex_count = GenericResultFunction(lambda epi, _: epi.vertex_set.size, description="Vertex count")
        self.total_cost = GenericResultFunction(lambda epi, _: sum([epi.edge_costs[e] for e in epi.graph.edges()]),
                                                description="Total edge cost")

        self.experiment_args = {
            'vertex_gen': self.varying_vertex_gen,
            'weight_gen': self.varying_weight_gen,
            'edge_gen': self.varying_edge_gen,
            'cost_gen': self.varying_cost_gen,
            'run_count': 7,
            'initial_vertex_fn': self.initial_vertex_fn,
            'log_path': self.log_path,
            'name': 'test',
            'full_log': False,
            'resample_vertices': False,
            'resample_weights': False,
            'resample_edges': False,
            'resample_costs': False,
            'mu': 1.,
            'zeta': 1.,
            'seed': self.entropy
        }

    def tearDown(self):
        clear_test_files()

    def test_execute(self):
        self.experiment_args['run_count'] = 1
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.execute()
        self.assertEqual(experiment.results, [6])

    def test_no_resample_edges(self):
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.execute()
        self.assertEqual(experiment.results, [6, 6, 6, 6, 6, 6, 6])

    def test_resample_edges(self):
        self.experiment_args['resample_edges'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.execute()
        self.assertEqual(experiment.results, [6, 0, 1, 2, 3, 4, 5])

    def test_resampling_vertices_resamples_edges(self):
        self.experiment_args['resample_vertices'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.execute()
        self.assertEqual(experiment.results, [6, 0, 1, 2, 3, 4, 5])

    def test_no_resample_costs(self):
        experiment = SIExperiment(**self.experiment_args, result_fn=self.total_cost)
        experiment.execute()
        self.assertEqual(experiment.results, [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75])

    def test_resample_costs(self):
        self.experiment_args['resample_costs'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.total_cost)
        experiment.execute()
        self.assertEqual(experiment.results, [.125*(6*i+3) + .375*(6*i+4) + .375*(6*i+5) for i in range(7)])

    def test_resampling_edges_resamples_costs(self):
        self.experiment_args['resample_edges'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.total_cost)
        experiment.execute()
        self.assertEqual(experiment.results, [3.75, 0., 0., 0., .125*11, .125*15, .125*19+.375*20])

    def test_resampling_vertices_resamples_costs(self):
        self.experiment_args['resample_vertices'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.total_cost)
        experiment.execute()
        self.assertEqual(experiment.results, [3.75, 0, 27.0, 236.25, 723.375, 2670.625, 6572.375])

    def test_no_resample_weights(self):
        experiment = SIExperiment(**self.experiment_args, result_fn=self.sum_weights)
        experiment.execute()
        self.assertEqual(experiment.results, [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

    def test_resample_weights(self):
        self.experiment_args['resample_weights'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.sum_weights)
        experiment.execute()
        self.assertEqual(experiment.results, [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5])

    def test_resampling_vertices_resamples_weights(self):
        self.experiment_args['resample_vertices'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.sum_weights)
        experiment.execute()
        self.assertEqual(experiment.results, [sum([i + x*.25 for x in range(i+4)]) for i in range(7)])

    def test_no_resample_vertices(self):
        experiment = SIExperiment(**self.experiment_args, result_fn=self.vertex_count)
        experiment.execute()
        self.assertEqual(experiment.results, [4, 4, 4, 4, 4, 4, 4])

    def test_resample_vertices(self):
        self.experiment_args['resample_vertices'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.vertex_count)
        experiment.execute()
        self.assertEqual(experiment.results, [4, 5, 6, 7, 8, 9, 10])

    def test_basic_log(self):
        self.experiment_args['resample_edges'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.execute()

        self.assertTrue((self.log_path / f"test-results.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-functions.pickle").exists())
        self.assertTrue((self.log_path / f"test-experiment-settings.cfg").exists())

        for i in range(7):
            self.assertFalse((self.log_path / f"test-vertices-run-{i}.pickle").exists())
            self.assertFalse((self.log_path / f"test-graph-run-{i}.gt").exists())
        self.assertFalse((self.log_path / f"test-vertices.pickle").exists())
        self.assertFalse((self.log_path / f"test-epidemic-settings.pickle").exists())
        self.assertFalse((self.log_path / f"test-vertex-settings.pickle").exists())

        self.assertEqual(experiment.load_results(), [6, 0, 1, 2, 3, 4, 5])

    def test_full_log(self):
        self.experiment_args['full_log'] = True
        self.experiment_args['resample_vertices'] = True
        self.experiment_args['resample_edges'] = True
        saved_experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
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

        self.assertEqual(loaded_experiment.results, [6, 0, 1, 2, 3, 4, 5])
        self.assertEqual(loaded_experiment.load_results(), [6, 0, 1, 2, 3, 4, 5])

        for i in range(7):
            loaded_run = loaded_experiment.load_run(i)
            self.assertEqual(loaded_run.mu, 1.)
            self.assertEqual(loaded_run.zeta, 1.)

            original_edges = set(_graphs[i])
            loaded_edges = {(int(e.source()), int(e.target())) for e in loaded_run.graph.edges()}
            self.assertEqual(original_edges, loaded_edges)

    def test_config(self):
        self.experiment_args['full_log'] = True
        self.experiment_args['resample_edges'] = True
        self.experiment_args['resample_vertices'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
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
        self.assertEqual(loaded_experiment.resample_vertices, experiment.resample_vertices)
        self.assertEqual(loaded_experiment.resample_weights, experiment.resample_weights)
        self.assertEqual(loaded_experiment.log_path, experiment.log_path)
        self.assertEqual(loaded_experiment.full_log, experiment.full_log)
        self.assertEqual(loaded_experiment.name, experiment.name)
        self.assertEqual(loaded_experiment.results, experiment.results)

    def test_config_format(self):
        self.maxDiff = None
        self.experiment_args['resample_edges'] = True
        experiment = SIExperiment(**self.experiment_args, result_fn=self.edge_count)
        experiment.save_config()
        with open(self.log_path / "test-experiment-settings.cfg") as f:
            saved_settings = f.read()
        expected_cfg = textwrap.dedent(f"""\
            name==test
            log_path=={Path.cwd()}/test_files
            full_log==False
            run_count==7
            resample_edges==True
            resample_costs==False
            resample_weights==False
            resample_vertices==False
            seed==99217604857427484066604220485342406204
            mu==1.0
            zeta==1.0
            Initial vertex selector of type <class 'SIExperiment.GenericInitialVertexFunction'>:
            \tdescription==Zero vertex
            Test result extractor of type <class 'SIExperiment.GenericResultFunction'>:
            \tdescription==Edge count
            Vertex set generator of type <class 'VertexSet.GenericVertexSetGen'>:
            \tdescription==Varying test vertex generator
            Vertex weight generator of type <class 'WeightedVertexSet.GenericWeightGen'>:
            \tdescription==Varying test weight generator
            Edge set generator of type <class 'SIEpidemic.GenericEdgeGen'>:
            \tdescription==Varying test edge generator
            Edge cost generator of type <class 'SIEpidemic.GenericEdgeCostGen'>:
            \tdescription==Varying test cost generator
            """)
        self.assertEqual(saved_settings, expected_cfg)


class SeedSavingTests(unittest.TestCase):
    def setUp(self):
        self.log_path = TEST_DIR
        self.log_path.mkdir(parents=True, exist_ok=True)
        clear_test_files()

        self.entropy = 74062891498417372046333540909508738617

        vertex_gen = PoissonPointProcess(dimension=2, size=10)
        weight_gen = PowerLawWeightGen(tau=2.5, ell=IdentityWeightScaler())
        edge_gen = GirgGen(alpha=1.5, scale_factor=1 / 10)
        cost_gen = FPPCostGen(lambda_=1)
        result_fn = GenericResultFunction(lambda _, __: None, "Throw away results")
        initial_vertex_fn = GenericInitialVertexFunction(lambda _, __: 0, "Zero vertex")

        self.original = SIExperiment(vertex_gen=vertex_gen, weight_gen=weight_gen, edge_gen=edge_gen, cost_gen=cost_gen,
                                     run_count=4, resample_edges=True, resample_costs=True, resample_weights=True,
                                     initial_vertex_fn=initial_vertex_fn, log_path=self.log_path, full_log=True,
                                     name="test", result_fn=result_fn, seed=self.entropy, resample_vertices=True,
                                     mu=1., zeta=1.)
        self.original.execute()

    def tearDown(self):
        clear_test_files()

    def test_rerun_same_seed(self):
        # If we rerun a randomised experiment from the same seed we should get the same data for every run.
        second = SIExperiment.load_from_file(folder=self.log_path, name="test", rerun=True, use_old_seed=True,
                                            new_name="test2")
        second.execute()

        self.assertEqual(self.original.seed, second.seed)

        for i in range(4):
            original_run = self.original.load_run(i)
            second_run = second.load_run(i)

            self.assertEqual(set(original_run.vertex_set.positions), set(second_run.vertex_set.positions))
            self.assertEqual(set(original_run.vertex_set.ids), set(second_run.vertex_set.ids))

            self.assertEqual(original_run.vertex_set.weights, second_run.vertex_set.weights)

            self.assertEqual(set(original_run.graph.edges()), set(second_run.graph.edges()))

            for edge in set(original_run.graph.edges()):
                self.assertEqual(original_run.edge_costs[edge], second_run.edge_costs[edge])

    def test_rerun_new_seed(self):
        # If we rerun an experiment from a new seed we should get different data for every run (with very high
        # probability).
        second = SIExperiment.load_from_file(folder=self.log_path, name="test", rerun=True, use_old_seed=False,
                                            new_name="test2")
        second.execute()

        self.assertNotEqual(self.original.seed, second.seed)

        for i in range(4):
            original_run = self.original.load_run(i)
            second_run = second.load_run(i)

            self.assertNotEqual(list(original_run.vertex_set.positions), list(second_run.vertex_set.positions))

            original_weights = original_run.vertex_set.weights
            rerun_weights = second_run.vertex_set.weights
            for j in range(len(original_weights)):
                if j >= len(rerun_weights):
                    break
                self.assertNotEqual(original_weights[j], rerun_weights[j])

            self.assertNotEqual(set(original_run.graph.edges()), set(second_run.graph.edges()))


def clear_test_files():
    """Clear out existing files from previous tests to make sure new ones are created."""
    if not TEST_DIR.exists():
        return
    for child in TEST_DIR.iterdir():
        if child.is_file() and child.suffix in [".gt", ".cfg", ".pickle"]:
            child.unlink()


if __name__ == '__main__':
    unittest.main()
