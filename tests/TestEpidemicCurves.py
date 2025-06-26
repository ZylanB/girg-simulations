import unittest

import config
from Region import Region
from figures.EpidemicCurves import (_get_infection_list, _get_run_datum, InfectionTimesAndRegions, InfectionDatum,
                                    log_points, RunDatum)
from SIEpidemic import FixedGraphGen, ConstantCostGen
from SIExperiment import FixedInitialVertex, SIExperiment
from VertexSet import EarthDistance, FixedVertexSet
from WeightedVertexSet import FixedWeightGen


class TestLogPoints(unittest.TestCase):
    def test(self):
        self.assertEqual(log_points(max_datum=16, precision=0), [1, 2, 4, 8, 16])
        self.assertEqual(log_points(max_datum=16, precision=1), [1, 2, 3, 4, 6, 8, 12, 16])
        result = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
        self.assertEqual(log_points(max_datum=64, precision=2), result)
        result.extend([80, 96, 112])
        self.assertEqual(log_points(max_datum=112, precision=2), result)
        result.append(117)
        self.assertEqual(log_points(max_datum=117, precision=2), result)


class TestResultFunction(unittest.TestCase):
    def test(self):
        # This experiment is running on a path graph with unit cost edges, so it should be fully predictable.
        points = {"New York": (40.7128, -74.006), "Bristol": (51.4545, -2.5879), "Eindhoven": (51.4231, 5.4623),
            "Kyoto": (35.0116, 135.7681), "Oxford": (51.752, -1.2577), "Los Angeles": (34.0549, -118.2426)}
        vertex_gen = FixedVertexSet(points, metric=EarthDistance(), description="test")
        weight_gen = FixedWeightGen({k: 1. for k in points.keys()}, description="test")
        edge_gen = FixedGraphGen([("New York", "Eindhoven"), ("Eindhoven", "Bristol"), ("Bristol", "Kyoto"),
                                  ("Kyoto", "Los Angeles"), ("Los Angeles", "Oxford")], description="test")
        cost_gen = ConstantCostGen(c=1.)
        initial_vertex_fn = FixedInitialVertex("New York")
        result_fn = InfectionTimesAndRegions(cutoff=5, precision=0)

        experiment = SIExperiment(vertex_gen=vertex_gen, weight_gen=weight_gen, edge_gen=edge_gen, cost_gen=cost_gen,
                                  initial_vertex_fn=initial_vertex_fn, result_fn=result_fn, run_count=1, mu=0.,
                                  resample_vertices=False, resample_weights=False, resample_edges=False, zeta=0.,
                                  resample_costs=False, full_log=False, name="test", log_path=config.TEST_FOLDER)
        experiment.execute()

        infections = [InfectionDatum(time=0., region=Region.US), InfectionDatum(time=1., region=Region.EU),
                      InfectionDatum(time=2., region=Region.EU), InfectionDatum(time=3., region=Region.OTHER),
                      InfectionDatum(time=4., region=Region.US), InfectionDatum(time=5., region=Region.EU)]
        self.assertEqual(_get_infection_list(experiment.current_run), infections)

        datum = RunDatum(i_points=[1, 2, 4, 5], infection_times=[0., 1., 3., 4.],
                         region_counts={Region.US: [1, 1, 1, 2], Region.EU: [0, 1, 2, 2], Region.OTHER: [0, 0, 1, 1]})
        self.assertEqual(datum, experiment.results[0])
        self.assertEqual(_get_run_datum(infections, i_points=[1, 2, 4, 5]), experiment.results[0])



if __name__ == '__main__':
    unittest.main()
