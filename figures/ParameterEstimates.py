from dataclasses import dataclass

from figures.tail_estimation import add_uniform_noise, get_ccdf, hill_estimator
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np

import config
from Gowalla import GowallaGraph

SEED = 7272300  # Obtained from np.random.SeedSequence().entropy & (2**32 - 1).

def get_degree_sequence() -> np.ndarray:
    """Returns the degree sequence of the Gowalla graph in decreasing order."""
    graph_data = GowallaGraph().graph_data
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=graph_data.vertex_set.size)
    graph.add_edge_list(graph_data.edge_list)

    sequence = graph.get_total_degrees(graph.get_vertices())
    sequence[::-1].sort()  # Reverses the order.
    return sequence


@dataclass
class HillCoefficients:
    xi: float
    kappa: float


def get_hill_coefficients(data: np.ndarray) -> HillCoefficients:
    np.random.seed(SEED)  # tail_estimates uses RandomState rather than a Generator, so we seed it this way.

    # Adds independent uniform noise on [-0.5, 0.5] to each data point. Necessary according to Voitalov et al.
    # to avoid issues where Hill's is known to be unstable and converge slowly for integer-valued distributions.
    noisy_data = add_uniform_noise(data, p=1.)

    hill_results = hill_estimator(noisy_data)
    kappa, xi = hill_results[2], hill_results[3]
    return HillCoefficients(xi=xi, kappa=kappa)

def generate_figure():
    degrees = get_degree_sequence()
    hill_coefficients = get_hill_coefficients(degrees)
    xi, kappa = hill_coefficients.xi, hill_coefficients.kappa

    # C.f. (11) of Voitalov et al., the CCDF tail exponent is 1/\xi.
    ccdf_exp = 1 / xi
    # Our tau is one more than the CCDF exponent, i.e. Pr(x >= w) ~ 1/w^{tau - 1}.
    gowalla_tau = 1 + ccdf_exp
    xmin = degrees[kappa]

    # Plot the CCDF of the degree sequence.
    x_ccdf, y_ccdf = get_ccdf(degrees)
    plt.rcParams['font.size'] = 18
    plt.figure("tau-alpha", figsize=(9, 12))
    axes = plt.gca()
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.step(x_ccdf, y_ccdf, linewidth=6, color="#999999")

    x = x_ccdf[np.where(x_ccdf >= xmin)]
    # This should just be the unique CCDF value at xmin from how get_ccdf is defined.
    ymin = np.mean(y_ccdf[np.where(x == xmin)])

    # The log of estimator_y (what we want to plot) is a line down from (log xmin, log ymin) with slope -xi.
    # So log(estimator_y) = log(y_0) - xi(log(x) - log(x_0)). So estimator_y = y_0(x_0/x)^xi.
    estimator_y = [ymin * (float(xmin) / k) ** ccdf_exp for k in x]
    axes.plot(x, estimator_y, color="#D55E00", linewidth=3, linestyle="dashed")
    axes.plot((x[-1]), estimator_y[-1], color="#D55E00", linestyle="none",
              marker="o", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=3, markersize=20)

    plt.figtext(0.6, 0.8, rf"$\xi_{{\kappa, n}} = {round(xi, 3)}$" + '\n' +
                rf"$\tau = 1 + 1/\xi_{{\kappa, n}} = {round(gowalla_tau, 3)}$", )

    plt.savefig(config.FIGURE_FOLDER / "gowalla-tail-estimates.png")

if __name__ == '__main__':
    generate_figure()
