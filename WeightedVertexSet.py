from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, List, Sequence

import dill  # type: ignore
import girg_sampling.girgs as gs  # type: ignore
import numpy as np

from LoggableFunction import LoggableFunction
from VertexSet import VertexSet, VertexSetGen


class WeightGen(LoggableFunction[[VertexSet, np.random.Generator], Sequence[float]]):
    """Function to generate a list of weights for the given vertex set."""
    @property
    def function_role(self):
        return "Vertex weight generator"


class GenericWeightGen(WeightGen):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str):
        self.description = description
        self._function = _function


class WeightedVertexSet:
    def __init__(self, vertex_gen: VertexSetGen, weight_gen: WeightGen, rng: np.random.Generator):
        """Stores a vertex set for a GIRG with both spatial and weight data. Vertices should be the underlying vertex
        set. Weight_gen should be a function that (probably randomly) resamples weights for the given
        VertexSet, returning a dictionary from vertex IDs to weights."""
        self.vertex_gen = vertex_gen
        self.vertices = self.vertex_gen(rng)
        self.weight_gen = weight_gen
        self.weights: Sequence[float] = []
        self.resample_weights(rng)

    def __getattr__(self, item):
        """Delegation, allows use of members and methods from VertexSet without formal inheritance."""
        vertices = object.__getattribute__(self, 'vertices')
        return getattr(vertices, item)

    def weight(self, id_: int) -> float:
        """Returns the weight of the vertex with the given id."""
        return self.weights[id_]

    def resample_weights(self, rng: np.random.Generator) -> None:
        """Resamples the vertex weights from the given WeightGen function."""
        self.weights = self.weight_gen(self.vertices, rng)

    def resample_vertices(self, rng: np.random.Generator):
        """Resamples the whole vertex set, including the weights, from the given VertexGen and WeightGen functions."""
        self.vertices = self.vertex_gen(rng)
        self.resample_weights(rng)

    def penalty(self, x_id: int, y_id: int, mu: float, zeta: float) -> float:
        """Returns the total penalty for a possible edge (specified by vertex IDs), not including the random cost."""
        spatial_penalty = self.vertices.distance(x_id, y_id) ** zeta
        weight_penalty = (self.weight(x_id) * self.weight(y_id)) ** mu
        return spatial_penalty * weight_penalty

    def save_configuration(self, path: Path) -> None:
        """Save the generator functions to file; this is often much smaller than the full vertex set."""
        with open(path, "wb") as file:
            dill.dump(self.weight_gen, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.vertex_gen, file, protocol=dill.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_configuration(cls, path: Path, rng: np.random.Generator) -> "WeightedVertexSet":
        try:
            with open(path, "rb") as file:
                weight_gen = dill.load(file)
                vertex_gen = dill.load(file)
        except Exception:
            print("Could not load WeightedVertexSet configuration file.")
            raise
        return WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=rng)


class EdgeWeightScaler(LoggableFunction[[float, np.random.Generator], float]):
    """A (possibly random) function that can play the role of ell in a power law weight distribution."""
    @property
    def function_role(self):
        return "Edge weight scaler in power-law distribution"


class IdentityWeightScaler(EdgeWeightScaler):
    """No weight scaling at all."""
    def __init__(self):
        self._function = lambda x, _: x


@dataclass
class GenericEdgeWeightScaler(EdgeWeightScaler):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self._function = _function
        self.description = description


class PowerLawWeightGen(WeightGen):
    """Returns a weight sampling function for WeightedVertexSet which samples weights W i.i.d. from a power law, taking
    Pr(W >= x) = ell(x) / x^{\tau - 1} and using the specified RNG, then applies the given scaling map to each
    weight."""
    def __init__(self, tau: float, ell: EdgeWeightScaler) -> None:
        self.tau = tau
        self.ell = ell
        if tau <= 2:
            raise ValueError("tau must be greater than 2 for the expected degrees to be finite.")

        self._function = lambda vertices, rng: self._power_law_sample(vertices=vertices, tau=tau, scaling=ell, rng=rng)

    @staticmethod
    def _power_law_sample(vertices: VertexSet, tau: float, scaling: EdgeWeightScaler, rng: np.random.Generator) \
            -> List[float]:
        """Samples weights W for the given VertexSet i.i.d. from a power law, taking Pr(W >= x) = 1 / x^{\tau - 1}
        and using the specified RNG, then applies the given scaling map to each weight."""
        # The upper bound isn't included in the range, so this samples a 31-bit integer to pass to Cython.
        C_seed = rng.integers(low=0, high=2 ** 31)
        weights = gs.generateWeights(n=vertices.size, ple=tau, seed=C_seed)
        curried_scaler = lambda weight: scaling(weight, rng)
        if type(scaling) is not IdentityWeightScaler:
            efficient_ell = np.vectorize(curried_scaler)
            weights = efficient_ell(weights)

        # We map an arbitrary weight to each vertex ID since they're all i.i.d. anyway.
        return weights


class FixedWeightGen(WeightGen):
    def __init__(self, weights: Mapping[Any, float], description: str) -> None:
        """Takes a dictionary mapping vertex names to weights, and returns a constant weight 'sampling function' which
        just returns a list mapping each vertex ID to its weight."""
        self.description = description
        self._function = lambda vertices, _: [weights[vertices.id_to_name(i)] for i in range(len(weights))]


def create_from_degrees_gen(degrees: Mapping[Any, int], description: str) -> FixedWeightGen:
    r"""Returns a weight sampling function for WeightedVertexSet which 'samples' weights by estimating them based on
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
    return FixedWeightGen(weights, description)
