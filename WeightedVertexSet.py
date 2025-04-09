from VertexSet import VertexSet
from typing import Any, Callable, Dict, List, Optional
import girg_sampling.girgs as gs
import numpy as np


class WeightedVertexSet:
    def __init__(self, vertices: VertexSet, weight_generator: Callable[[VertexSet], List[float]], mu: float,
                 zeta: float):
        """Stores a vertex set for a GIRG with both spatial and weight data. Vertices should be the underlying vertex
        set. Weight_generator should be a function that (probably randomly) resamples weights for the given
        VertexSet, returning a dictionary from vertex IDs to weights. Mu should be the weight penalty,
        and zeta should be the spatial penalty."""
        self.vertices = vertices
        self.weight_generator = weight_generator
        self.weights = None
        self.mu = mu
        self.zeta = zeta
        self.resample_weights()

    def __getattr__(self, item):
        """Delegation, allows use of members and methods from VertexSet without formal inheritance."""
        vertices = object.__getattribute__(self, 'vertices')
        return getattr(vertices, item)

    def weight(self, id_: int) -> float:
        """Returns the weight of the vertex with the given id."""
        return self.weights[id_]

    def resample_weights(self) -> None:
        """Resamples the vertex weights from the given generator function."""
        self.weights = self.weight_generator(self.vertices)

    def penalty(self, x_id: int, y_id: int) -> float:
        """Returns the total penalty for a possible edge (specified by vertex IDs), not including the random cost."""
        spatial_penalty = self.vertices.distance(x_id, y_id) ** self.zeta
        weight_penalty = (self.weight(x_id) * self.weight(y_id)) ** self.mu
        return spatial_penalty * weight_penalty


def power_law_generator(tau: float, ell: Callable[[float], float] = id,
                        generator: Optional[np.random.Generator] = None) -> Callable[[VertexSet], List[float]]:
    """Returns a weight sampling function for WeightedVertexSet which samples weights W i.i.d. from a power law, taking
    Pr(W >= x) = 1 / x^{\tau - 1} and using the specified RNG, then applies the given scaling map to each weight."""
    if tau <= 2:
        raise ValueError("tau must be greater than 2 for the expected degrees to be finite.")
    if generator is None:
        generator = np.random.default_rng()

    return lambda vertices: _power_law_sample(vertices=vertices, tau=tau, scaling=ell, generator=generator)


def _power_law_sample(vertices: VertexSet, tau: float, scaling: Callable[[float], float],
                      generator: np.random.Generator) -> List[float]:
    """Samples weights W for the given VertexSet i.i.d. from a power law, taking Pr(W >= x) = 1 / x^{\tau - 1}
    and using the specified RNG, then applies the given scaling map to each weight."""

    # The upper bound isn't included in the range, so this samples a 31-bit integer to pass to Cython.
    C_seed = generator.integers(low=0, high=2**31)
    weights = gs.generateWeights(n=vertices.size, ple=tau, seed=C_seed)
    if scaling is not id:
        efficient_ell = np.vectorize(scaling)
        weights = efficient_ell(weights)

    # We map an arbitrary weight to each vertex ID since they're all i.i.d. anyway.
    return weights


def fixed_weights_generator(weights: Dict[Any, float]) -> Callable[[VertexSet], List[float]]:
    """Takes a dictionary mapping vertex names to weights, and returns a constant weight 'sampling function' which just
    returns a list mapping each vertex ID to its weight."""
    return lambda vertices: [weights[vertices.id_to_name(i)] for i in range(len(weights))]


def from_degrees_generator(degrees: Dict[Any, int]) -> Callable[[VertexSet], List[float]]:
    """Returns a weight sampling function for WeightedVertexSet which 'samples' weights by estimating them based on
    a supplied dictionary mapping vertex names to vertex degrees in the base graph. Used for the Gowalla dataset.

    IMPORTANT: We really don't understand why this seems to work. Do not submit the paper before we do. A more
    principled approach is likely going to need our estimated value of alpha. John thinks the ideologically correct
    form of this is probably going to map a degree d to max(1, d/C) for some C depending on alpha and the average
    degree, as a result of correcting for three sources of error:

    1. The true degree distribution is going to have a constant scaling factor, i.e. Pr(d(v) >= x) ~ C_1 / x^{\tau-1}
    for some constant C_1. We can estimate C_1 by comparing the average degree to the expected average degree with
    \int_1^\infty 1/x^{\tau-1} = 1 / (\tau - 2).
    2. The expected degrees aren't the same as the weights. This is probably roughly E(d(v)) ~ C_2 * W_v for some
    constant factor C_2 depending on \alpha - need to take a look at Johannes's old calculations. IIRC there's no nice
    closed form for the integral but we can at least do a numerical approximation even if it turns out not to be
    well-approximated by a constant factor.
    3. You can't have a weight smaller than 1, so every vertex of degree less than a certain threshold should be
    considered to have weight 1.

    By themselves, 1. and 2. wouldn't affect the relative penalty values at all - vertices would still be infected in
    the same order - but in conjunction with 3 they become significant."""
    average_degree = sum(degrees.values()) / len(degrees)
    penalty = 2 * average_degree / 3

    def weight_estimate(name: Any) -> float:
        return max(1., degrees[name] - penalty)

    weights = {name: weight_estimate(name) for name in degrees.keys()}
    return fixed_weights_generator(weights)
