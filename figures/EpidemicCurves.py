from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import functools
from math import ceil, log2, log10
from typing import List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import ConnectionPatch
import numpy as np
from sklearn.linear_model import LinearRegression

import config
from Region import Region, region_from_position
from SIEpidemic import FPPCostGen, SIEpidemic
from SIExperiment import FixedInitialVertex, ResultFunction, SIExperiment
from figures.EpidemicCurvesConfig import (CurveParams, CURVE_PARAMS, PlotParams, PLOT_PARAMS, PLOT_PRECISION, RUN_COUNT,
                                          SEEDS)


@dataclass
class InfectionDatum:
    """An infection time stored alongside its region."""
    time: float
    region: Region


@dataclass
class RunDatum:
    """All the information we'd need to plot an epidemic curve for a given run if we were doing it in isolation."""
    i_points: Sequence[int]
    infection_times: Sequence[float]  # infection_times[x] is the time taken to infect i_points[x] vertices.
    # region_counts[R][x] is the number of infected vertices in region R at time infection_times[x].
    region_counts: Mapping[Region, List[int]]


class InfectionTimesAndRegions(ResultFunction):
    """Returns a list of InfectionDatums sorted by time."""
    def __init__(self, cutoff: int, precision: int = PLOT_PRECISION):
        self._function = functools.partial(self._get_result, cutoff=cutoff, precision=precision)

    @staticmethod
    def _get_result(epidemic: SIEpidemic, _: np.random.Generator, cutoff: int, precision: int) -> RunDatum:
        infection_list = get_infection_list(epidemic)
        i_points = log_points(max_datum=cutoff, precision=precision)
        return get_run_datum(infection_list, i_points)


def get_infection_list(epidemic: SIEpidemic) -> List[InfectionDatum]:
    """Returns a list of all infections in the epidemic, sorted by time. Each infection has its region stored
    alongside it in an InfectionDatum."""
    infections = []
    for v in epidemic.graph.vertices():
        v_position = epidemic.vertex_set.id_to_position(int(v))
        v_region = region_from_position(v_position)
        new_infection = InfectionDatum(time=epidemic.infection_times[v], region=v_region)
        infections.append(new_infection)

    infections.sort(key=lambda x: x.time)
    return infections


def get_run_datum(infection_data: List[InfectionDatum], i_points: List[int]) -> RunDatum:
    """Takes the raw infection data from a run and turns it into a RunDatum, containing i_points, the times at which we
    pass the infection thresholds given by i_points, and regional breakdowns of infections at those times."""
    infection_times = []
    region_counts = {r: [] for r in Region}
    region_running_totals = {r: 0 for r in Region}
    for x, datum in enumerate(infection_data):
        region_running_totals[datum.region] += 1
        if x+1 in i_points:
            infection_times.append(datum.time)
            for region in region_counts.keys():
                region_counts[region].append(region_running_totals[region])

    return RunDatum(i_points=i_points, infection_times=infection_times, region_counts=region_counts)


def generate_data(params: CurveParams, seed: int, name: str) -> List[RunDatum]:
    """Returns sorted lists of RUN_COUNT infection times for independent infections of the given graph."""
    graph = params.graph()
    cost_gen = FPPCostGen(lambda_=1)
    result_fn = InfectionTimesAndRegions(cutoff=params.infection_cutoff)
    initial_fn = FixedInitialVertex(name=params.initial_name)
    experiment = SIExperiment(**graph.experiment_params, resample_costs=True, initial_vertex_fn=initial_fn,
                              result_fn=result_fn, run_count=RUN_COUNT, log_path=config.FIGURE_FOLDER, mu=params.mu,
                              zeta=params.zeta, seed=seed, full_log=False, name=name, cost_gen=cost_gen)

    if experiment.results_exist:
        print("Loading from file...")
        return experiment.load_results()

    experiment.execute()
    return experiment.results


@dataclass
class EpidemicCurveData:
    """Data to pass to matplotlib to plot the actual graphs."""
    i_points: Sequence[int]  # Points to plot on the y axis
    bottom_curve: Sequence[float]  # Points to plot on the x axis for the bottom error curve
    median_curve: Sequence[float]  # Points to plot on the x axis for the middle line
    top_curve: Sequence[float]     # Points to plot on the x axis for the top error curve

    # Points to plot on the y axis in the geographic inset. So region_medians[EU][x] answers: "When i_points[x] total
    # vertices were infected, how many of those were in the EU? All of these are taken from a single medoid run so
    # they add up to the total infection count.
    region_medians: Mapping[Region, Sequence[int]]


def log_points(max_datum: int, precision: int) -> List[int]:
    """Returns a suitable scale to sample from when plotting from 1 to max_datum on a log scale. Higher precision
    samples more points - it counts up using the [precision+1] most significant bits in binary."""
    points = list(range(1, 2**precision))
    max_point = ceil(log2(max_datum))
    for n in range(precision, max_point + 1):
        points.extend(list(range(2**n, min(max_datum, 2**(n+1)), 2**(n-precision))))
    points.append(max_datum)
    return points


def process_infection_times(runs: List[RunDatum], bottom: int, top: int) -> EpidemicCurveData:
    """Takes a sorted list of lists of infection times (one per run) and returns the curve data to plot. 'bottom' and
    'top' are the error bars, so e.g. bottom = 25 and top = 75 returns error curves covering the middle two quartiles.
    'cutoff' is the point to cut off the epidemic, so e.g. saturation = 40,000 will stop plotting a run at the point
    where 40,000 nodes are infected."""
    i_points = runs[0].i_points
    for run in runs:
        if i_points != run.i_points:
            raise RuntimeError("Something has gone very wrong, different runs are sampling at different points!")

    # Goes from [[run 1 time 1, ..., run 1 time n], ..., [run n time 1, ..., run n time n]] to
    # [[run 1 time 1, ..., run n time 1], ..., [run 1 time n, ..., run n time n]]
    zipped_times = list(zip(*[datum.infection_times for datum in runs]))
    bottom_curve = [float(np.percentile(infections, bottom, method="nearest")) for infections in zipped_times]
    median_curve = [float(np.percentile(infections, 50, method="nearest")) for infections in zipped_times]
    top_curve = [float(np.percentile(infections, top, method="nearest")) for infections in zipped_times]

    l2_from_median = lambda r: sum([(r.infection_times[x] - median_curve[x]) ** 2 for x in range(len(i_points))])
    medoid_run = min(runs, key=l2_from_median)

    return EpidemicCurveData(i_points=i_points, bottom_curve=bottom_curve, median_curve=median_curve,
                             top_curve=top_curve, region_medians=medoid_run.region_counts)


def plot_psi_inset(curve_data: EpidemicCurveData, infection_range: Tuple[int, int]):
    """Plots an inset on the current figure showing a linear regression (on a log-log scale) determining the slope
    of the graph in the given infection_range of y coordinates."""
    # Sets start_index to the first plotted infection count greater than infection_range[0]-1, i.e. the first plotted
    # infection count which is at least infection_range[0].
    start_index = bisect_right(curve_data.i_points, infection_range[0] - 1)
    # Sets end_point to the first plotted infection count greater than infection_range[1]. This means the last point
    # of the start_index:end_point slice will be the last plotted infection count which is at most infection_range[1].
    end_index = bisect_left(curve_data.i_points, infection_range[1] + 1)

    # These are the data points we'll use for the regression.
    time_coords = curve_data.median_curve[start_index:end_index]
    infection_coords = curve_data.i_points[start_index:end_index]
    log_times = [log10(t) for t in time_coords]
    log_infections = [log10(x) for x in infection_coords]

    # Run the linear regression.
    reshaped_log_times = [[t] for t in log_times]
    model = LinearRegression()
    model.fit(reshaped_log_times, log_infections)

    # Points to plot in the inset.
    regression_infection_coords = [model.intercept_ + model.coef_[0] * t for t in log_times]

    # Pass to inset.
    main_axes = plt.gca()
    inset_axes = plt.axes((0.15, 0.47, 0.28, 0.43))
    plt.sca(inset_axes)

    plt.grid(visible=True)

    # Bare axes with log/log scale.
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    inset_axes.tick_params(which="both", bottom=False, top=False, left=False, right=False, labelbottom=False,
                           labelleft=False)

    # Zoomed-in plot of linear regression overlaid on the true value.
    plt.xlim(time_coords[0], time_coords[-1])
    plt.ylim(infection_coords[0], infection_coords[-1])
    plt.plot(time_coords, infection_coords, linewidth=6, color="gray", linestyle="dashed")
    plt.plot(time_coords, [10 ** x for x in regression_infection_coords], linewidth=3, color="red")

    # Print the slope of the linear regression.
    plt.title(rf"$2\psi = {round(model.coef_[0], 2)}$")

    # Indicate boundaries of the inset plot with dotted grey lines. Set the coordinates first.
    inset_right_edge_x = inset_axes.get_xlim()[1]
    inset_bottom_edge_y = inset_axes.get_ylim()[0]
    inset_top_edge_y = inset_axes.get_ylim()[1]
    inset_bottom_right = (inset_right_edge_x, inset_bottom_edge_y)
    inset_top_right = (inset_right_edge_x, inset_top_edge_y)
    main_bottom_left = (time_coords[0], infection_coords[0])
    main_top_right = (time_coords[-1], infection_coords[-1])

    # Pass back to main figure before drawing the lines.
    plt.sca(main_axes)
    bottom_patch = ConnectionPatch(xyA=main_bottom_left, coordsA=main_axes.transData, xyB=inset_bottom_right,
                                   coordsB=inset_axes.transData, color="grey", linewidth=2, linestyle="dotted")
    plt.gcf().add_artist(bottom_patch)
    top_patch = ConnectionPatch(xyA=main_top_right, coordsA=main_axes.transData, xyB=inset_top_right,
                                coordsB=inset_axes.transData, color="grey", linewidth=2, linestyle="dotted")
    plt.gcf().add_artist(top_patch)


def percent_formatter(y, _):
    """Formatter for the y axis of the regional inset."""
    return f"{int(round(y,0))}%"


def plot_region_inset(curve_data: EpidemicCurveData, log_t: bool):
    """Plots an inset on the current figure showing cumulative infection totals in the different regions (EU/US/other).
    If log_t is True, then the time/x axis is shown in logarithmic scale."""
    # At each infection count, get the breakdown into proportions in the EU/US/other regions.
    point_count = len(curve_data.i_points)
    region_medians = curve_data.region_medians
    total_infections = curve_data.i_points
    eu_proportions = [100 * region_medians[Region.EU][x] / total_infections[x] for x in range(point_count)]
    us_proportions = [100 * region_medians[Region.US][x] / total_infections[x] for x in range(point_count)]
    other_proportions = [100 * region_medians[Region.OTHER][x] / total_infections[x] for x in range(point_count)]

    # Get the actual boundaries between regions of the plot.
    eu_cumulative = eu_proportions
    us_cumulative = [eu_proportions[x] + us_proportions[x] for x in range(point_count)]
    other_cumulative = [us_cumulative[x] + other_proportions[x] for x in range(point_count)]

    median_curve = curve_data.median_curve

    # Pass to inset.
    main_axes = plt.gca()
    inset_axes = plt.axes((0.625, 0.15, 0.25, 0.25))
    plt.sca(inset_axes)

    plt.grid(visible=True)
    inset_axes.margins(x=0)  # Margins on the x axis look weird.

    # The y axis here is a percentage of infected vertices from a given region.
    inset_axes.set_yticks([0, 25, 50, 75, 100])
    plt.ylim(0, 100)
    inset_axes.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    if log_t:
        plt.xscale("log", base=10)

    # Colours from Okabe-Ito palette, should be safe for colour-blind viewers/monochrome printers
    plt.fill_between(median_curve, [0]*point_count, eu_cumulative, color="#0072B2")
    plt.fill_between(median_curve, eu_cumulative, us_cumulative, color="#D55E00")
    plt.fill_between(median_curve, us_cumulative, other_cumulative, color="#999999")

    # Pass back to main plot before returning.
    plt.sca(main_axes)


def plot_curve(params: PlotParams, curve_data: EpidemicCurveData, graph_no: int):
    """Plots the epidemic curve for the given infection times with the given parameters."""
    plt.figure(graph_no, figsize=(13,13), clear=True)

    plt.grid(visible=True)

    # y axis is always log, x axis may be log depending on params.
    plt.yscale("log", base=10)
    if params.log_t:
        plt.xscale("log", base=10)

    # Median as a bold line, with shaded error region between top and bottom curve.
    plt.plot(curve_data.median_curve, curve_data.i_points, linewidth=2)
    plt.plot(curve_data.top_curve, curve_data.i_points, linestyle="dashed", color="black", linewidth=.5)
    plt.plot(curve_data.bottom_curve, curve_data.i_points, linestyle="dashed", color="black", linewidth=.5)
    plt.fill_betweenx(curve_data.i_points, curve_data.top_curve, curve_data.bottom_curve, color="#B3C7F7")

    # Plot insets if needed.
    if params.psi_inset_infection_range is not None:
        plot_psi_inset(curve_data=curve_data, infection_range=params.psi_inset_infection_range)
    if params.region_inset:
        plot_region_inset(curve_data=curve_data, log_t=params.log_t)

    plt.savefig(config.FIGURE_FOLDER / f"epidemic_curves_{graph_no}.png")


if __name__ == '__main__':
    if len(CURVE_PARAMS) != len(SEEDS) or len(CURVE_PARAMS) != len(PLOT_PARAMS):
        raise RuntimeError("Mismatched plot parameters!")

    for i in range(len(CURVE_PARAMS)):
        print(f"Getting infection data for graph {i}...")
        data = generate_data(CURVE_PARAMS[i], seed=SEEDS[i], name=f"epidemic-curve-{i}")
        print(f"Processing data for graph {i}...")
        epidemic_curves = process_infection_times(runs=data, bottom=25, top=75)
        print(f"Plotting graph {i}...")
        plot_curve(PLOT_PARAMS[i], epidemic_curves, i)
