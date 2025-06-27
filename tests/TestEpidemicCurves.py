import unittest

import config
from Region import Region
from figures.EpidemicCurves import (_get_infection_list, _get_run_datum, InfectionTimesAndRegions, InfectionDatum,
                                    log_points, RunDatum, process_infection_times)
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


class TestDataProcessing(unittest.TestCase):
    def test(self):
        # Fictional vertices: 4 in the US, 4 in the EU, 2 elsewhere.
        # Fictional run histories:
        # run_a: t=0 US, t=1 EU,    t=2 other, t=3 US,  t=4 US,     t=5 EU,  t=6 other, t=7 EU,  t=8 EU,  t=9 US
        # run_b: t=0 EU, t=2 US,    t=4 other, t=10 US, t=12 other, t=14 US, t=16 EU,   t=18 EU, t=20 US, t=22 EU
        # run_c: t=0 US, t=3 other, t=6 other, t=9 EU,  t=11 US,    t=15 EU, t=18 EU,   t=21 EU, t=24 US, t=27 US
        i_points = [1, 2, 4, 6, 8, 10]
        run_a = RunDatum(i_points=i_points, infection_times=[0., 1., 3., 5., 7., 9.],
                         region_counts={Region.US: [1, 1, 2, 3, 3, 4], Region.EU: [0, 1, 1, 2, 3, 4],
                                        Region.OTHER: [0, 0, 1, 1, 2, 2]})
        run_b = RunDatum(i_points=i_points, infection_times=[0., 2., 10., 14., 18., 22.],
                         region_counts={Region.US: [0, 1, 2, 3, 3, 4], Region.EU: [1, 1, 1, 1, 3, 4],
                                        Region.OTHER: [0, 0, 1, 2, 2, 2]})
        run_c = RunDatum(i_points=i_points, infection_times=[0., 3., 9., 15., 21., 27.],
                         region_counts={Region.US: [1, 1, 1, 2, 2, 4], Region.EU: [0, 0, 1, 2, 4, 4],
                                        Region.OTHER: [0, 1, 2, 2, 2, 2]})
        curves = process_infection_times(runs=[run_a, run_b, run_c], top=75, bottom=25)

        self.assertEqual(curves.median_curve, [0., 2., 9., 14., 18., 22.])
        self.assertEqual(curves.bottom_curve, [0., 1., 3., 5., 7., 9.])
        self.assertEqual(curves.top_curve, [0., 3., 10., 15., 21., 27.])

        # The median run is run_a at 1 infection (as the first run in the list with the median value), run_c at 4
        # infections, and run_b at all other i_points. The medoid, which is what we should be getting, is always run_b.
        self.assertEqual(curves.region_medians, run_b.region_counts)


if __name__ == '__main__':
    unittest.main()
