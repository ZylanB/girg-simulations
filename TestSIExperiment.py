import unittest
from SIExperiment import *
from SIEpidemic import SIEpidemic
from VertexSet import lattice
from WeightedVertexSet import WeightedVertexSet


class FileIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entropy = 99217604857427484066604220485342406204
        cls.generator = np.random.default_rng(seed=entropy)

        def weight_generator(_):
            weight_generator.x += 1.
            return [weight_generator.x, weight_generator.x + .25, weight_generator.x + .5, weight_generator.x + .75]
        weight_generator.x = 0.

        def cost_generator():
            cost_generator.i += 1
            return cost_generator.i
        cost_generator.i = 0

        cls.graphs = [[],
                     [(0, 1)],
                     [(0, 1), (0, 2)],
                     [(0, 1), (0, 2), (1, 2)],
                     [(0, 1), (0, 2), (1, 2), (0, 3)],
                     [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)],
                     [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]]

        def edge_generator(_):
            edge_generator.i += 1
            return cls.graphs[edge_generator.i]
        edge_generator.i = -1

        unweighted_vertices = lattice(dimension=2, size=2)
        vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator)
        cls.epidemic = SIEpidemic(vertex_set=vertices, edge_cost_generator=cost_generator,
                                  edge_generator=edge_generator, mu=0.5, zeta=1.5)
        cls.log_path = Path.cwd() / "test_log.pickle"

    def testRun(self):
        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=lambda gen: 0, log_path=self.log_path, full_log=False,
                                  result_fn=lambda graph, gen: len(graph.edges))
        self.assertEqual(experiment.results, [0, 1, 2, 3, 4, 5, 6])

    def testFullLog(self):
        self.log_path.unlink(missing_ok=True)
        for i in range(7):
            (Path.cwd() / f"test_log-run-{i}.gt").unlink(missing_ok=True)
            (Path.cwd() / f"test_log-run-{i}.pickle").unlink(missing_ok=True)

        experiment = SIExperiment(epidemic=self.epidemic, run_count=7, resample_edges=True, resample_costs=True,
                                  initial_vertex_fn=lambda gen: 0, log_path=self.log_path, full_log=True,
                                  result_fn=lambda graph, gen: len(graph.edges))
        experiment.execute()
        self.assertEqual(experiment.load_results(), [0, 1, 2, 3, 4, 5, 6])

        for i in range(7):
            experiment.load_run(i)
            self.assertEqual(self.epidemic.mu, 0.5)
            self.assertEqual(self.epidemic.zeta, 1.5)
            self.assertEqual(set(self.graphs[i]), set(experiment.epidemic.graph.edges))


if __name__ == '__main__':
    unittest.main()
