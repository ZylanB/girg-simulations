from __future__ import annotations
import functools
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import dill  # type: ignore # Extension of pickle that supports encoding/decoding functions.
import girg_sampling.girgs as gs  # type: ignore
import graph_tool as gt  # type: ignore
from graph_tool.topology import shortest_distance  # type: ignore
import numpy as np

from LoggableFunction import LoggableFunction
from WeightedVertexSet import WeightedVertexSet, FixedWeightGen
from VertexSet import Lattice


class EdgeCostGen(LoggableFunction[[np.random.Generator], float]):
    """Function to generate the random part of a single edge cost (the "L" per our notation)."""
    @property
    def function_role(self):
        return "Edge cost generator"


class GenericEdgeCostGen(EdgeCostGen):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self.description = description
        self._function = _function


class EdgeGen(LoggableFunction[[WeightedVertexSet, np.random.Generator], Sequence[Tuple[int, int]]]):
    """Function to generate the edges of our graph; these are stored in a tuple since sets aren't Hashable, but are
    nevertheless undirected."""
    @property
    def function_role(self):
        return "Edge set generator"


class GenericEdgeGen(EdgeGen):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self.description = description
        self._function = _function


class SIEpidemic:
    """Stores a spatial graph whose edge weights determine the spread of an SI model epidemic. Edge_cost_generator
    should return a sample of the *random* part of an edge's cost, i.e. not including degree or spatial penalties
    (which are calculated in WeightedVertexSet). Mu should be the weight penalty, and zeta should be the spatial
    penalty. Name determines the filenames used for logging."""
    def __init__(self, vertex_set: WeightedVertexSet, cost_gen: EdgeCostGen, mu: float, zeta: float,
                 edge_gen: EdgeGen, name: str, rng: np.random.Generator) -> None:
        self.vertex_set = vertex_set
        self.cost_gen = cost_gen
        self.edge_gen = edge_gen
        self.mu = mu
        self.zeta = zeta
        self.name = name

        self.graph = gt.Graph(directed=False)
        self.graph.add_vertex(n=self.vertex_set.size)
        self.edge_costs = self.graph.new_edge_property("double")
        self.graph.edge_properties["edge_costs"] = self.edge_costs

        self.sample_edges(rng)
        self.sample_edge_costs(rng)

        # These will be set on calling self.run_infection, which takes the initial vertex as an argument.
        # Uninfected vertices have infinite infection time. Vertices without an infector (including the source vertex)
        # have infector id -1.
        self.infection_times = self.graph.new_vertex_property("double")
        self.graph.vertex_properties["infection_times"] = self.infection_times
        self.infectors = self.graph.new_vertex_property("int64_t")
        self.graph.vertex_properties["infectors"] = self.infectors
        self.initial_vertex = None

    def sample_edges(self, rng: np.random.Generator) -> None:
        """(Re)samples the edge set of the graph from the given vertex set. Vertex IDs in the graph correspond to
        vertex IDs from the WeightedVertexSet."""
        self.graph.clear_edges()
        edges = self.edge_gen(self.vertex_set, rng)
        self.graph.add_edge_list(edges)

    def sample_edge_costs(self, rng: np.random.Generator) -> None:
        """(Re)samples only the edge costs while maintaining the current edge set."""
        # This is fairly well-optimised, and needs to be - it will be called with 10M+ edges.

        print("Sampling edge costs...")

        # Start with the numpy arrays of edge sources/destinations/IDs to avoid costly lookups.
        edge_array = self.graph.get_edges([self.graph.edge_index])
        u_array, v_array, id_array = edge_array.T

        # Pull these into local variables to avoid recomputing them.
        edge_count = self.graph.num_edges()
        cost_gen = functools.partial(self.cost_gen, rng)
        penalty = functools.partial(self.vertex_set.penalty, mu=self.mu, zeta=self.zeta)

        # Actually sample and compute the edge costs, again storing them in an nparray.
        new_costs = np.empty(edge_count, dtype=float)
        for i in range(edge_count):
            new_costs[id_array[i]] = cost_gen() * penalty(u_array[i], v_array[i])

        # Directly reassign the edge property's array to this new array rather than going edge-by-edge.
        self.edge_costs.a[:] = np.asarray(new_costs)

        # The next step in optimisation here would be to vectorise penalty and cost_gen. Vectorising cost_gen would be
        # easy, but only cut running times by 10-15% or so. Vectorising penalty would be more significant but would
        # require encoding the distances as edge properties, which is a much bigger change.

    def run_infection(self, initial_vertex_id: int) -> None:
        """Runs an SI infection on the stored graph, storing the results in self.infection_time and self.infector.
        For each Vertex v in self.graph, self.infection_time[v] will be the infection time of v and self.infector[v]
        will be the node that infected v; these default to infinity (np.inf) and None respectively if v isn't in
        the same component as the initial vertex."""
        print("Running infection...")
        self.initial_vertex = self.graph.vertex(initial_vertex_id)

        # Distances need to be initialised to infinity before running graph-tools' Dijkstra implementation.
        self.infection_times.set_value(np.inf)
        self.infectors.set_value(-1)

        shortest_distance(g=self.graph, source=self.initial_vertex, weights=self.edge_costs,
                          dist_map=self.infection_times, pred_map=self.infectors)

    @property
    def is_infected(self) -> gt.VertexPropertyMap:
        """Returns an vertex property which is true for all vertices which are actually infected."""
        return self.infection_times.transform(lambda t: t != np.inf)

    def infection_path(self, vertex_id: int) -> Optional[List[int]]:
        """Returns the path by which the given vertex was infected in the current run as a list from source to sink.
        Returns None if the vertex was never infected."""
        current_vertex = vertex_id
        if self.infection_times[current_vertex] == np.inf:
            return None

        # Prepending an item to a list takes linear time in Python, so we instead find the path from target to source
        # and then reverse it.
        path = [current_vertex]
        while current_vertex != self.initial_vertex:
            current_vertex = self.infectors[current_vertex]
            path.append(current_vertex)
        path.reverse()

        return path

    def spatial_annulus_property(self, inner_radius: float, outer_radius: float) -> gt.VertexPropertyMap:
        """Returns a vertex property which is true for vertices in the given annulus from the initial vertex."""
        # Normally letting numpy do things its own way is faster, so I'm not pulling things out into a list until I
        # have to.
        initial_vertex_id = self.graph.vertex_index[self.initial_vertex]
        initial_vertex_position = self.vertex_set.id_to_position(initial_vertex_id)
        annulus = self.vertex_set.get_ids_in_annulus(center=initial_vertex_position, inner_radius=inner_radius,
                                                     outer_radius=outer_radius)

        return self.graph.vertex_index.transform(f=lambda id_: id_ in annulus, value_type="bool")

    def median_infection_at_boundary(self, radius: float, error: float) -> float:
        """Returns the median infection time among all vertices whose spatial distance from the initial infection lies
        in [radius, radius + error], including vertices that were never infected."""
        in_annulus_property = self.spatial_annulus_property(radius, radius + error)
        in_annulus_filter = in_annulus_property.get_array().astype(bool)
        annulus_infection_times = self.infection_times.get_array()[in_annulus_filter]
        return np.median(annulus_infection_times)

    def proportional_cost(self, vertex_id: int) -> Optional[float]:
        """Returns the cost of the most expensive edge in the infection path to the given vertex divided by the total
        cost, or None if the vertex was never infected or was the source."""
        path = self.infection_path(vertex_id)
        if path is None or len(path) == 1:
            return None

        path_edge_costs = [self.edge_costs[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        return max(path_edge_costs) / sum(path_edge_costs)

    def proportional_length(self, vertex_id: int) -> Optional[float]:
        """Returns the length of the longest edge (spatially) in the infection path to the given vertex divided by the
        sum of all edge lengths, or None if the vertex was never infected or was the source."""
        path = self.infection_path(vertex_id)
        if path is None or len(path) == 1:
            return None

        path_edge_lengths = [self.vertex_set.distance(path[i], path[i+1]) for i in range(len(path) - 1)]
        return max(path_edge_lengths) / sum(path_edge_lengths)

    def hop_count(self, vertex_id: int) -> Optional[int]:
        """Returns the number of edges in the infection path to the given vertex, or None if the vertex was never
        infected."""
        path = self.infection_path(vertex_id)
        if path is None:
            return None
        return len(path) - 1

    @property
    def config_filename(self) -> str:
        return f"{self.name}-epidemic-settings.pickle"

    @property
    def vertex_config_filename(self) -> str:
        return f"{self.name}-vertex-settings.pickle"

    def vertex_filename(self, run_index: Optional[int]) -> str:
        if run_index is None:
            return f"{self.name}-vertices.pickle"
        return f"{self.name}-vertices-run-{run_index}.pickle"

    def graph_filename(self, run_index: Optional[int]) -> str:
        if run_index is None:
            return f"{self.name}-graph.gt"
        return f"{self.name}-graph-run-{run_index}.gt"

    def save_config(self, folder: Path) -> None:
        """Logs only the data needed to re-run the current epidemic to the given folder, with filenames depending
        on run_index. This is often significantly more space-efficient and faster."""
        with open(folder / self.config_filename, "wb") as file:
            dill.dump(self.edge_gen, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.cost_gen, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.mu, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.zeta, file, protocol=dill.HIGHEST_PROTOCOL)
        self.vertex_set.save_config(folder / self.vertex_config_filename)

    def save_vertices(self, folder: Path, run_index: Optional[int]) -> None:
        """Logs the current vertex set in full to the given folder with filename depending on run_index."""
        with open(folder / self.vertex_filename(run_index), "wb") as file:
            dill.dump(self.vertex_set, file, protocol=dill.HIGHEST_PROTOCOL)

    def save_graph(self, folder: Path, run_index: Optional[int]) -> None:
        """Logs the current epidemic graph (with all infection information) to the given folder with filename
        depending on run_index."""
        self.graph.save(str(folder / self.graph_filename(run_index)))

    @staticmethod
    def _empty_epidemic(name: str) -> SIEpidemic:
        """Creates an empty SIEpidemic object which can be initialised manually. Used when loading from files."""
        vertex_gen = Lattice(size=1, dimension=1)
        weight_gen = FixedWeightGen(weights={0: 1}, description="Empty epidemic")
        weighted_vertices = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen,
                                              rng=np.random.default_rng())
        edge_gen = FixedGraphGen(edges=[], description="Empty graph")
        cost_gen = ConstantCostGen(c=0)
        return SIEpidemic(vertex_set=weighted_vertices, cost_gen=cost_gen, edge_gen=edge_gen, mu=0., zeta=0.,
                          name=name, rng=np.random.default_rng())

    @classmethod
    def load_from_config(cls, folder: Path, name: str, rng: np.random.Generator) -> "SIEpidemic":
        """Loads an SIEpidemic with configuration information only present in the given folder and returns it,
        recreating the vertex set and graph. If information for the given run_index is not present, falls back to
        run_index None (intended for experiments where e.g. the vertex set is kept constant across all runs)."""
        return_value = cls._empty_epidemic(name=name)

        try:
            with open(folder / return_value.config_filename, "rb") as file:
                return_value.edge_gen = dill.load(file)
                return_value.cost_gen = dill.load(file)
                return_value.mu = dill.load(file)
                return_value.zeta = dill.load(file)
        except Exception:
            print("Could not load SIEpidemic configuration file.")
            raise

        vertices = WeightedVertexSet.load_from_config(path=folder / return_value.vertex_config_filename, rng=rng)
        return_value.vertex_set = vertices
        return_value.graph.add_vertex(n=vertices.size)
        return_value.sample_edges(rng)
        return_value.sample_edge_costs(rng)

        return return_value

    @classmethod
    def load_full(cls, folder: Path, name: str, run_index: Optional[int]) -> "SIEpidemic":
        """Loads an SIEpidemic with configuration, vertex and graph information present in the given folder and
        returns it. If information for the given run_index is not present, falls back to run_index None (intended
        for experiments where e.g. the vertex set is kept constant across all runs)."""
        return_value = cls._empty_epidemic(name=name)

        try:
            with open(folder / return_value.config_filename, "rb") as file:
                return_value.edge_gen = dill.load(file)
                return_value.cost_gen = dill.load(file)
                return_value.mu = dill.load(file)
                return_value.zeta = dill.load(file)
        except Exception:
            print("Could not load SIEpidemic configuration file.")
            raise

        if (folder / return_value.vertex_filename(run_index)).exists():
            vertex_path = folder / return_value.vertex_filename(run_index)
        else:
            vertex_path = folder / return_value.vertex_filename(None)
        try:
            with open(vertex_path, "rb") as file:
                return_value.vertex_set = dill.load(file)
        except Exception:
            print("Could not load WeightedVertexSet.")
            raise

        if (folder / return_value.graph_filename(run_index)).exists():
            graph_path = folder / return_value.graph_filename(run_index)
        else:
            graph_path = folder / return_value.graph_filename(None)
        try:
            return_value.graph = gt.load_graph(str(graph_path))
        except Exception:
            print("Could not load graph.")
            raise

        return_value.infection_times = return_value.graph.vertex_properties["infection_times"]
        return_value.infectors = return_value.graph.vertex_properties["infectors"]
        return_value.edge_costs = return_value.graph.edge_properties["edge_costs"]
        return return_value


class GirgGen(EdgeGen):
    """Generates an SIEpidemic whose graph is a GIRG on the given vertex_set with the given long-range parameter
    alpha and the given RNG seed. The connection probability between u and v is max(1, W_uW_v/|u-v|^d)^\alpha. If
    average_degree is set then the GIRG library uses binary search to find a constant c to scale the weights by which
    gives the appropriate average degree.

    The GIRG generation code we're using requires everything to be scaled down to [0,1]^d, using connection probability
    max(1, W_uW_v/n|u-v|^d)^\alpha, and we don't want to change that. As long as we're working in [0, n^{1/d}]^d, this
    is equivalent to scaling everything down by a factor of n^{1/d}. The one catch is that we need to pass this in as
    an argument rather than generating it from the vertex set, since otherwise if we have a PPP with slightly less than
    n points then we might end up with points outside [0, 1]^d and the generation code will crash. So instead we pass
    1/n^{1/d} in as the scale_factor argument."""
    def __init__(self, alpha: float, scale_factor: float, average_degree: Optional[float] = None) -> None:
        self._function = functools.partial(self._sample_edges, alpha=alpha, scale_factor=scale_factor,
                                           average_degree=average_degree)
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.average_degree = average_degree

    @staticmethod
    def _sample_edges(vertex_set: WeightedVertexSet, rng: np.random.Generator, alpha: float, scale_factor: float,
                      average_degree: Optional[float]) -> List[Tuple[Any, Any]]:
        seed = rng.integers(low=0, high=2 ** 31)  # girg-sampling takes 31-bit seeds, "high" is not inclusive.
        weights = [vertex_set.weight(i) for i in range(vertex_set.size)]

        """The GIRG generator creates a GIRG with connection probability between u and v given by max(1, 
        W_uW_v/n|u-v|^d)^alpha. It also requires all points to lie in [0,1]^d. So we need to scale everything down 
        by a factor of n^{1/d}."""
        positions = [vertex_set.id_to_position(i) for i in range(vertex_set.size)]
        scaled_positions = []
        for position in positions:
            scaled_position = [position[i] * scale_factor for i in range(vertex_set.dimension)]
            scaled_positions.append(scaled_position)

        if average_degree is not None:
            c = gs.scaleWeights(weights=weights, desiredAvgDegree=average_degree, dimension=vertex_set.dimension,
                                alpha=alpha)
            weights = [c * weight for weight in weights]

        return gs.generateEdges(weights=weights, positions=scaled_positions, alpha=alpha,
                                scale=scale_factor ** vertex_set.dimension, seed=seed)


class FixedGraphGen(EdgeGen):
    """'Edge generator' for a constant graph such as the Gowalla dataset, to be passed to SIEpidemic. The edges
    should be specified as a list of pairs of vertex IDs."""
    def __init__(self, edges: Sequence[Tuple[int, int]], description: str) -> None:
        self.description = description
        self._function = lambda _, __: edges


class FPPCostGen(EdgeCostGen):
    """Edge cost generator for first passage percolation (i.e. the random part of edge costs are i.i.d. exponentially
    distributed) with the given parameter."""
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_
        self._function = lambda rng: rng.exponential(scale=lambda_)


class ConstantCostGen(EdgeCostGen):
    """Edge cost generator for testing purposes that sets the random part of all edge costs to the given constant."""
    def __init__(self, c: float) -> None:
        self.c = c
        self._function = lambda _: c
