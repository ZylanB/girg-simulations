from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import functools
from math import ceil, log2, log10
from typing import List, Mapping, Optional, Sequence, Tuple, Type

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import ConnectionPatch
import numpy as np
from sklearn.linear_model import LinearRegression

import config
from PresetGraph import PresetGraph
from Gowalla import SyntheticGowallaGraph, GowallaGiantGraph
from Region import Region, region_from_position
from SIEpidemic import SIEpidemic, FPPCostGen
from SIExperiment import SIExperiment, ResultFunction, InitialVertexFunction, FixedInitialVertex


RUN_COUNT = 55
PLOT_PRECISION = 4  # Increase to plot at a finer scale


@dataclass
class InfectionDatum:
    time: float
    region: Region


class InfectionTimesAndRegions(ResultFunction):
    """Returns a list of InfectionDatums sorted by time."""
    def __init__(self, cutoff: int, precision: int = PLOT_PRECISION):
        self._function = functools.partial(_get_result, cutoff=cutoff, precision=precision)


def _get_infection_list(epidemic: SIEpidemic) -> List[InfectionDatum]:
    infections = []
    for v in epidemic.graph.vertices():
        v_position = epidemic.vertex_set.id_to_position(int(v))
        v_region = region_from_position(v_position)
        new_infection = InfectionDatum(time=epidemic.infection_times[v], region=v_region)
        infections.append(new_infection)

    infections.sort(key=lambda x: x.time)
    return infections


@dataclass
class RunDatum:
    """All the information we'd need to plot epidemic curves for a given run if we were doing it in isolation."""
    i_points: Sequence[int]
    infection_times: Sequence[float]  # infection_times[x] is the time taken to infect i_points[x] vertices.
    # region_counts[R][x] is the number of infected vertices in region R at time infection_times[x].
    region_counts: Mapping[Region, List[int]]


def _get_run_datum(infection_data: List[InfectionDatum], i_points: List[int]) -> RunDatum:
    """Takes the raw infection data from a run and turns it into a RunDatum."""

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


def _get_result(epidemic: SIEpidemic, _: np.random.Generator, cutoff: int, precision: int) -> RunDatum:
        infection_list = _get_infection_list(epidemic)
        i_points = log_points(max_datum=cutoff, precision=precision)
        return _get_run_datum(infection_list, i_points)


@dataclass
class CurveParams:
    graph: Type[PresetGraph]
    initial: InitialVertexFunction
    result_fn: ResultFunction
    mu: float
    zeta: float


@dataclass
class PlotParams:
    log_t: bool
    region_inset: bool
    psi_inset_infection_range: Optional[Tuple[int, int]]


GOWALLA_INITIAL = 164
GOWALLA_CUTOFF = 64635
GOWALLA_XI_RANGE = (150, 5000)
GOWALLA_PARAMS = {"graph": GowallaGiantGraph, "initial": FixedInitialVertex(name=GOWALLA_INITIAL),
                  "result_fn": InfectionTimesAndRegions(cutoff=GOWALLA_CUTOFF)}
SYN_GOWALLA_CUTOFF = 166667
SYN_GOWALLA_INITIAL = 230563
SYN_GOWALLA_XI_RANGE = (150, SYN_GOWALLA_CUTOFF)
SYN_GOWALLA_PARAMS = {"graph": SyntheticGowallaGraph, "initial": FixedInitialVertex(name=SYN_GOWALLA_INITIAL),
                      "result_fn": InfectionTimesAndRegions(cutoff=SYN_GOWALLA_CUTOFF)}
CURVE_PARAMS: List[CurveParams] = [
    CurveParams(**GOWALLA_PARAMS, mu=0.0, zeta=0.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=0.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=2.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=3.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=0.0, zeta=0.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=1.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=2.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=3.0)
]

PLOT_PARAMS: List[PlotParams] = [
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=True),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=True),
    PlotParams(log_t=True, psi_inset_infection_range=GOWALLA_XI_RANGE, region_inset=True),
    PlotParams(log_t=True, psi_inset_infection_range=GOWALLA_XI_RANGE, region_inset=True),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=False),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=False),
    PlotParams(log_t=True, psi_inset_infection_range=SYN_GOWALLA_XI_RANGE, region_inset=False),
    PlotParams(log_t=True, psi_inset_infection_range=SYN_GOWALLA_XI_RANGE, region_inset=False)
]

SEEDS = [
    257135840227949566337238051408055634111,
    303903392100660574452661293122954292079,
    207575285627819734050983420268628288457,
    336827387627079923696399650174942918686,
    175837567687085788645530192316345259609,
    233657882610370377508087340942845579878,
    139102178419114034645107386879574426047,
    90012362361482529980268819803967658146
]


def generate_data(params: CurveParams, seed: int, name: str) -> List[RunDatum]:
    """Returns sorted lists of RUN_COUNT infection times for independent infections of the given graph."""
    graph = params.graph()
    cost_gen = FPPCostGen(lambda_=1)
    experiment = SIExperiment(**graph.experiment_params, resample_costs=True, initial_vertex_fn=params.initial,
                              result_fn=params.result_fn, run_count=RUN_COUNT, log_path=config.FIGURE_FOLDER,
                              mu=params.mu, zeta=params.zeta, seed=seed, full_log=False, name=name, cost_gen=cost_gen)

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
    # vertices were infected, how many of those were in the EU? All of these are taken from a single "median run" so
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

    region_medians = {r: [] for r in Region}
    for x in range(len(i_points)):
        median_run = [run for run in runs if run.infection_times[x] == median_curve[x]][0]
        for r in Region:
            region_medians[r].append(median_run.region_counts[r][x])

    return EpidemicCurveData(i_points=i_points, bottom_curve=bottom_curve, median_curve=median_curve,
                             top_curve=top_curve, region_medians=region_medians)

def exp_formatter(y, _):
    if y <= 0:
        return str(y)
    exponent = int(np.round(log10(y)))
    return rf"$10^{{{exponent}}}$"

def plot_psi_inset(curve_data: EpidemicCurveData, infection_range: Tuple[int, int]):
    # Sets start_index to the first plotted infection count greater than infection_range[0]-1, i.e. the first plotted
    # infection count which is at least infection_range[0].
    start_index = bisect_right(curve_data.i_points, infection_range[0] - 1)
    # Sets end_point to the first plotted infection count greater than infection_range[1]. This means the last point
    # of the start_index:end_point slice will be the last plotted infection count which is at most infection_range[1].
    end_index = bisect_left(curve_data.i_points, infection_range[1] + 1)
    log_times = [log10(t) for t in curve_data.median_curve[start_index:end_index]]
    log_infections = [log10(x) for x in curve_data.i_points[start_index:end_index]]

    # Run the linear regression.
    reshaped_log_times = [[t] for t in log_times]
    model = LinearRegression()
    model.fit(reshaped_log_times, log_infections)

    # Points to plot in the inset.
    time_coords = curve_data.median_curve[start_index:end_index]
    infection_coords = curve_data.i_points[start_index:end_index]
    regression_infection_coords = [model.intercept_ + model.coef_[0] * t for t in log_times]

    # Pass to inset.
    main_axes = plt.gca()
    inset_axes = plt.axes((0.15, 0.47, 0.28, 0.43))
    plt.sca(inset_axes)

    plt.plot(time_coords, infection_coords, linewidth=6, color="gray", linestyle="dashed")
    plt.plot(time_coords, [10**x for x in regression_infection_coords], linewidth=3, color="red")

    # Bare axes with log/log scale.
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.grid(visible=True)

    inset_axes.tick_params(which="both", bottom=False, top=False, left=False, right=False, labelbottom=False,
                           labelleft=False)

    # Zoomed-in plot of linear regression overlaid on the true value.
    plt.xlim(time_coords[0], time_coords[-1])
    plt.ylim(infection_coords[0], infection_coords[-1])

    # Print the slope of the linear regression.
    plt.title(f"$2\\psi = {round(model.coef_[0], 2)}$")

    # Indicate boundaries of inset plot with dotted grey lines. Set the coordinates first.
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


def plot_region_inset(curve_data: EpidemicCurveData):
    pass


def plot_curve(params: PlotParams, curve_data: EpidemicCurveData, graph_no: int):
    """Plots the epidemic curve for the given infection times with the given parameters."""
    plt.figure(graph_no, figsize=(13,13), clear=True)
    axes = plt.gca()

    plt.yscale("log", base=10)
    axes.yaxis.set_major_formatter(FuncFormatter(exp_formatter))
    if params.log_t:
        plt.xscale("log", base=10)
        axes.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

    plt.xlabel("t" if not params.log_t else "log(t)")
    plt.ylabel("log(I(t))")
    plt.plot(curve_data.median_curve, curve_data.i_points, linewidth=2)
    plt.plot(curve_data.top_curve, curve_data.i_points, linestyle="dashed", color="black", linewidth=.5)
    plt.plot(curve_data.bottom_curve, curve_data.i_points, linestyle="dashed", color="black", linewidth=.5)
    plt.fill_betweenx(curve_data.i_points, curve_data.top_curve, curve_data.bottom_curve, color="#B3C7F7")
    plt.grid(visible=True)

    if params.psi_inset_infection_range is not None:
        plot_psi_inset(curve_data=curve_data, infection_range=params.psi_inset_infection_range)

    if params.region_inset:
        plot_region_inset(curve_data)

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
