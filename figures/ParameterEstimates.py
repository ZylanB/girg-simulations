from dataclasses import dataclass
from typing import Sequence, Tuple

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import config
import figures.colours as colours
from figures.tail_estimation import add_uniform_noise, get_ccdf, hill_estimator
from Gowalla import GowallaGiantGraph
from PresetGraph import GraphData

SEED = 7272300  # Obtained from np.random.SeedSequence().entropy & (2**32 - 1).


def get_degree_sequence(graph_data: GraphData) -> np.ndarray:
    """Returns the degree sequence of the Gowalla graph in decreasing order."""
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=graph_data.vertex_set.count)
    graph.add_edge_list(graph_data.edge_list)

    sequence = graph.get_total_degrees(graph.get_vertices())
    sequence[::-1].sort()  # Sort in reverse order.
    return sequence


@dataclass
class HillData:
    kappa: float  # Kappa parameter for Hill's estimator
    xi: float     # Value of Hill's estimator
    tau: float    # Value of tau (our degree distribution tail) corresponding to xi


def get_hill_coefficients(data: np.ndarray) -> HillData:
    """Returns Hill's estimator for the power-law of the given dataset, which should be sorted in decreasing order.
    Also returns the value of tau corresponding to this estimator if the dataset is a graph's degree sequence."""
    np.random.seed(SEED)  # tail_estimates uses RandomState rather than a Generator, so we seed it this way.

    # Adds independent uniform noise on [-0.5, 0.5] to each data point. Necessary according to Voitalov et al.
    # to avoid issues where Hill's is known to be unstable and converge slowly for integer-valued distributions.
    noisy_data = add_uniform_noise(data, p=1.)

    hill_results = hill_estimator(noisy_data)
    kappa, xi = hill_results[2], hill_results[3]
    # C.f. (11) of Voitalov et al., the CCDF tail exponent is 1/\xi, and our tau is one more than this.
    tau = 1 + 1 / xi
    return HillData(xi=xi, kappa=kappa, tau=tau)


@dataclass
class InsetData:
    edge_ccdf_x: np.ndarray  # x coordinates for the CCDF of the edge lengths.
    edge_ccdf_y: np.ndarray  # y coordinates for the CCDF of the edge lengths.
    edge_regression_y: np.ndarray  # y coordinates for the regression on edge_ccdf.
    alpha: float  # Value of alpha corresponding to the regression on edge_ccdf.
    deviation: float  # One standard deviation error on alpha.


def get_low_degree_edges(count: int, edge_list: Sequence[Tuple[int, int]], cutoff: int) -> np.ndarray:
    """Given an edge list and the number of vertices, returns a list of edges between vertices with degree at most
    the specified cutoff."""
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=count)
    graph.add_edge_list(edge_list)
    degree_mask = graph.new_vertex_property("bool")
    degree_mask.a = graph.get_total_degrees(graph.get_vertices()) <= cutoff
    low_degree_graph = gt.GraphView(graph, vfilt=degree_mask)
    return low_degree_graph.get_edges()


def get_edge_ccdf(graph_data: GraphData, edge_cutoffs: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a CCDF for the edge length of the given graph, conditioned on it lying in the closed interval
    given by edge_cutoffs. If the two returned lists are x and y, and we abbreviate edge_cutoffs to ec, then the
    elements of x are the distinct edge lengths, and for all i:

    y[i] = #(edges with length in [x[i], ec[1]]) / #(edges with length in [ec[0], ec[1]])"""

    vertex_set, edge_list = graph_data.vertex_set, graph_data.edge_list

    if edge_cutoffs[0] < 1:
        raise Exception("The model we're fitting to is completely invalid for edges with length less than 1.")

    # Convert the list of edges into a list of edge lengths.
    u_list = [a for [a, _] in edge_list]
    v_list = [b for [_, b] in edge_list]
    length_list = np.fromiter((vertex_set.distance(u, v) for u, v in zip(u_list, v_list)), dtype="float64",
                              count=len(u_list))

    # Strip out the edges outside the cutoff interval.
    length_list.sort()
    low_boundary = length_list.searchsorted(edge_cutoffs[0], side="left")
    high_boundary = length_list.searchsorted(edge_cutoffs[1], side="right")
    length_list = length_list[low_boundary:high_boundary]

    # Get the conditional CCDF of edge length.
    uniques, counts = np.unique(length_list, return_counts=True)
    cumcounts = np.cumsum(counts) - counts #  i'th entry is #(edges with length < i)
    cumprob = cumcounts.astype(np.double) / length_list.size  # i'th entry is Pr(edge length < i)
    edge_ccdf_x, edge_ccdf_y = uniques, (1. - cumprob)
    return edge_ccdf_x, edge_ccdf_y


def alpha_estimate_data(graph_data: GraphData, edge_cutoffs: Tuple[float, float], rng: np.random.Generator) \
        -> InsetData:
    r"""Uses a non-linear regression on edge lengths to estimate an alpha parameter for a synthetic GIRG version of the
    given graph. Relies on approximations that are weak for small edge lengths and edge lengths large enough for
    geometry to become a serious factor, so considers probabilities conditioned on the edge lying in the (closed)
    edge_cutoffs interval.

    In more detail, we use the following assumptions and approximations:
    - The edge lengths we consider are a subset of [1, N/2].
    - Writing C for the lower end of the edge-cutoff, C^{-d*alpha} >> C^{-d*(tau-1)}. (This is a slightly stronger
      condition than alpha > tau-1.)

    Write E_n(x, y) for the number of edges in [-N/2, N/2]^d with length in [x, y]. For all r > 0, over a pair
    e = (x,y) \in [0,1]^d with ||x-y||=r, write Delta_e(r) := E((1 \wedge W_xW_y/r^d)^alpha) for the expected
    connection probability conditioned on x and y being vertices.

    The first assumption allows us to say E(E_n(x, R)) = \int_{r=L}^R r^{d-1} \Delta_e(r) dr.

    The second assumption allows us to approximate Delta_e(r) ~ r^{-d*alpha} * (tau-1)**2 / (tau-1-alpha)**2 when
    r >= L.

    Given these two assumptions, we get that for all x >= L, E(E_n(x, R)) ~ b* (x**(-d*(a-1)) - R**(-d*(a-1))) for
    a=alpha and some unknown b. This is the model function we use for the regression."""

    edge_ccdf_x, edge_ccdf_y = get_edge_ccdf(graph_data, edge_cutoffs)

    # Downsample before running the curve fit so it runs in a reasonable amount of time
    sample_count = min(100000, edge_ccdf_x.size)
    sample_indices = rng.choice(edge_ccdf_x.size, size=sample_count, replace=False)
    sample_edge_ccdf_x, sample_edge_ccdf_y =  edge_ccdf_x[sample_indices], edge_ccdf_y[sample_indices]

    d = graph_data.vertex_set.dimension
    def model_function(x, a, b):
        return b* (x**(-d*(a-1)) - edge_cutoffs[1]**(-d*(a-1)))
    coefficients, covariance = curve_fit(model_function, sample_edge_ccdf_x, sample_edge_ccdf_y,
                                         bounds=([1.0001, 0], [np.inf, np.inf]), maxfev=10000)

    alpha = coefficients[0]
    b = coefficients[1]
    deviation = np.sqrt(np.diag(covariance))[0]
    edge_regression_y = np.asarray([model_function(x, alpha, b) for x in sample_edge_ccdf_x])
    return InsetData(edge_ccdf_x=sample_edge_ccdf_x, edge_ccdf_y=sample_edge_ccdf_y, edge_regression_y=edge_regression_y,
                     alpha=alpha, deviation=deviation)


def plot_inset(graph_data: GraphData, edge_cutoffs: Tuple[float, float]):
    entropy = 87830759957948721258624078455704374391
    rng = np.random.default_rng(entropy)

    # Get data to plot
    inset_data = alpha_estimate_data(graph_data=graph_data, edge_cutoffs=edge_cutoffs, rng=rng)
    edge_ccdf_x, edge_ccdf_y = inset_data.edge_ccdf_x, inset_data.edge_ccdf_y
    edge_regression_x, edge_regression_y = edge_ccdf_x, inset_data.edge_regression_y
    alpha = inset_data.alpha
    print(f"Alpha = {alpha}, standard deviation = {inset_data.deviation})")

    # Pass to new inset axes.
    main_axes = plt.gca()
    inset_axes = plt.axes((0.25, 0.20, 0.35, 0.20))
    plt.sca(inset_axes)

    # Log plot the CCDF of the edge lengths in km alongside its linear regression.
    plt.grid(visible=True)
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.xlim(1, 10**4)
    plt.ylim(.01, 1)
    plt.step(edge_ccdf_x, edge_ccdf_y, linewidth=6, color=colours.BLUE)
    plt.plot(edge_regression_x, edge_regression_y, linewidth=3, color=colours.RED, linestyle="dashed",
             dash_capstyle="round", dashes=(3, 2))

    plt.figtext(0.1875, 0.1375, rf"$d(1-\alpha)={np.round(2*(1-alpha), 3)},\quad\alpha={np.round(alpha, 3)}$")

    # Pass back to main axes before return.
    plt.sca(main_axes)


def generate_plot(graph_data: GraphData, edge_cutoffs: Tuple[float, float]):
    degrees = get_degree_sequence(graph_data)
    hill_coefficients = get_hill_coefficients(degrees)
    xi, kappa, tau = hill_coefficients.xi, hill_coefficients.kappa, hill_coefficients.tau
    print(f"tau = {tau}")

    # C.f. (11) of Voitalov et al., the CCDF tail exponent is 1/\xi.
    ccdf_exp = 1 / xi
    xmin = degrees[kappa]

    # Plot the CCDF of the degree sequence.
    x_ccdf, y_ccdf = get_ccdf(degrees)
    plt.rcParams['font.size'] = 18
    plt.figure("tau-alpha", figsize=(9, 12))
    axes = plt.gca()
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.step(x_ccdf, y_ccdf, linewidth=6, color=colours.BLUE)

    x = x_ccdf[np.where(x_ccdf >= xmin)]
    # This should just be the unique CCDF value at xmin from how get_ccdf is defined.
    ymin = np.mean(y_ccdf[np.where(x == xmin)])

    # The log of estimator_y (what we want to plot) is a line down from (log xmin, log ymin) with slope -xi.
    # So log(estimator_y) = log(y_0) - xi(log(x) - log(x_0)). So estimator_y = y_0(x_0/x)^xi.
    estimator_y = [ymin * (float(xmin) / k) ** ccdf_exp for k in x]
    # wtf_y = [ymin * (float(xmin) / k) ** (SYN_GOWALLA_TAU-1) for k in x]
    axes.plot(x, estimator_y, color=colours.RED, linewidth=3, linestyle="dashed", dash_capstyle="round")
    # axes.plot(x, wtf_y, color=colours.LIGHT_BLUE, linewidth=3, linestyle="dashed", dash_capstyle="round")
    axes.plot((x[-1]), estimator_y[-1], linestyle="none", marker="o", markerfacecolor="none",
              markeredgecolor=colours.RED, markeredgewidth=3, markersize=20)

    plt.figtext(0.6, 0.8, rf"$\xi_{{\kappa, n}} = {round(xi, 3)}$" + '\n' +
                rf"$\tau = 1 + 1/\xi_{{\kappa, n}} = {round(tau, 3)}$", )

    plot_inset(graph_data=graph_data, edge_cutoffs=edge_cutoffs)

    config.FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURE_FOLDER / "gowalla-tail-estimates.png")


def generate_figure():
    generate_plot(graph_data=GowallaGiantGraph().graph_data, edge_cutoffs=(10, 100))