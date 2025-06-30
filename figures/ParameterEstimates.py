from dataclasses import dataclass
from typing import Sequence, Tuple

from PresetGraph import GraphData
from figures.tail_estimation import add_uniform_noise, get_ccdf, hill_estimator
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import figures.colours as colours
import config
from Gowalla import GowallaGraph


SEED = 7272300  # Obtained from np.random.SeedSequence().entropy & (2**32 - 1).
ALPHA_MAX_DEGREE = 148  # Edge lengths will be taken only from the induced subgraph of nodes with this max degree.
ALPHA_MIN_EDGE_LENGTH = 1  # After calculating the CCDF, we discard edge lengths below MIN/above MAX as outliers.
ALPHA_MAX_EDGE_LENGTH = 1000


def get_degree_sequence(graph_data: GraphData) -> np.ndarray:
    """Returns the degree sequence of the Gowalla graph in decreasing order."""
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=graph_data.vertex_set.size)
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
    edge_regression_x: np.ndarray  # x coordinates for the linear regression on log(edge_ccdf) (a subset of edge_ccdf_x).
    edge_regression_y: np.ndarray  # y coordinates for the linear regression on log(edge_ccdf).
    alpha: float  # Value of alpha corresponding to the slope of the linear regression on log(edge_ccdf).


def get_low_degree_edges(size: int, edge_list: Sequence[Tuple[int, int]], cutoff: int) -> np.ndarray:
    """Given an edge list and the number of vertices, returns a list of edges between vertices with degree at most
    the specified cutoff."""
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=size)
    graph.add_edge_list(edge_list)
    degree_mask = graph.new_vertex_property("bool")
    degree_mask.a = graph.get_total_degrees(graph.get_vertices()) <= cutoff
    low_degree_graph = gt.GraphView(graph, vfilt=degree_mask)
    return low_degree_graph.get_edges()


def alpha_estimate_data(graph_data: GraphData) -> InsetData:
    """Uses a linear regression on log(edge lengths) to estimate an alpha parameter for a synthetic GIRG version of
    the given graph."""
    vertex_set, edge_list = graph_data.vertex_set, graph_data.edge_list

    low_degree_edges = get_low_degree_edges(size=vertex_set.size, edge_list=edge_list, cutoff=ALPHA_MAX_DEGREE)

    # Convert the list of edges into a list of edge lengths.
    u_list = low_degree_edges[:, 0].tolist()
    v_list = low_degree_edges[:, 1].tolist()
    length_list = np.fromiter((vertex_set.distance(u, v) for u, v in zip(u_list, v_list)), dtype="float64",
                              count=len(u_list))

    # Get the CCDF in sorted increasing order of length.
    length_list[::-1].sort()
    edge_ccdf_x, edge_ccdf_y = get_ccdf(length_list)
    edge_ccdf_x, edge_ccdf_y = edge_ccdf_x[::-1], edge_ccdf_y[::-1]

    # Strip out edges in the range we're not running regression on and take logs.
    cutoff_low = edge_ccdf_x.searchsorted(ALPHA_MIN_EDGE_LENGTH, side="right")
    cutoff_high = edge_ccdf_x.searchsorted(ALPHA_MAX_EDGE_LENGTH, side="left")
    edge_regression_x = edge_ccdf_x[cutoff_low + 1:cutoff_high]
    log_edge_ccdf_y = np.log10(edge_ccdf_y[cutoff_low + 1:cutoff_high])

    # Run the regression.
    edge_regression_x_input = np.reshape(np.log10(edge_regression_x), (-1, 1))
    model = LinearRegression()
    model.fit(edge_regression_x_input, log_edge_ccdf_y)
    edge_regression_y = np.power(10, model.coef_[0] * edge_regression_x_input + model.intercept_)

    #  Slope of linear regression is d(1-alpha).
    alpha = 1 - model.coef_[0] / vertex_set.dimension

    return InsetData(edge_ccdf_x=edge_ccdf_x, edge_ccdf_y=edge_ccdf_y, edge_regression_x=edge_regression_x,
                     edge_regression_y=edge_regression_y, alpha=alpha)


def plot_inset(graph_data: GraphData):
    # Get data to plot
    inset_data = alpha_estimate_data(graph_data)
    edge_ccdf_x, edge_ccdf_y = inset_data.edge_ccdf_x, inset_data.edge_ccdf_y
    edge_regression_x, edge_regression_y = inset_data.edge_regression_x, inset_data.edge_regression_y
    alpha = inset_data.alpha

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
    plt.step(edge_ccdf_x, edge_ccdf_y, linewidth=3, color=colours.BLUE)
    plt.plot(edge_regression_x, edge_regression_y, linewidth=6, color=colours.RED, linestyle="dashed",
             dash_capstyle="round")

    plt.figtext(0.1875, 0.1375, rf"$d(1-\alpha)={np.round(2*(1-alpha), 3)},\quad\alpha={np.round(alpha, 3)}$")

    # Pass back to main axes before return.
    plt.sca(main_axes)


def generate_figure():
    graph_data = GowallaGraph().graph_data

    degrees = get_degree_sequence(graph_data)
    hill_coefficients = get_hill_coefficients(degrees)
    xi, kappa, tau = hill_coefficients.xi, hill_coefficients.kappa, hill_coefficients.tau

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
    axes.plot(x, estimator_y, color=colours.RED, linewidth=3, linestyle="dashed", dash_capstyle="round")
    axes.plot((x[-1]), estimator_y[-1], linestyle="none", marker="o", markerfacecolor="none",
              markeredgecolor=colours.RED, markeredgewidth=3, markersize=20)

    plt.figtext(0.6, 0.8, rf"$\xi_{{\kappa, n}} = {round(xi, 3)}$" + '\n' +
                rf"$\tau = 1 + 1/\xi_{{\kappa, n}} = {round(tau, 3)}$", )

    plot_inset(graph_data)

    plt.savefig(config.FIGURE_FOLDER / "gowalla-tail-estimates.png")

if __name__ == '__main__':
    generate_figure()
