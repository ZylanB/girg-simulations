from __future__ import annotations

import dill  # Extension of pickle that supports encoding/decoding functions.

import numpy as np
import graph_tool as gt
from graph_tool.topology import shortest_distance
import girg_sampling.girgs as gs
from pathlib import Path

from WeightedVertexSet import WeightedVertexSet, fixed_weights_generator
from typing import Any, Callable, List, Optional, Tuple
from VertexSet import lattice


class SIEpidemic:
    """Stores a spatial graph whose edge weights determine the spread of an SI model epidemic. Edge_cost_generator
    should return a sample of the *random* part of an edge's cost, i.e. not including degree or spatial penalties
    (which are calculated in WeightedVertexSet). Mu should be the weight penalty, and zeta should be the spatial
    penalty."""
    def __init__(self, vertex_set: WeightedVertexSet, edge_cost_generator: Callable[[], float],
                 edge_generator: Callable[[WeightedVertexSet], List[Tuple[int, int]]], mu: float, zeta: float):
        self.vertex_set = vertex_set
        self.edge_cost_generator = edge_cost_generator
        self.edge_generator = edge_generator
        self.mu = mu
        self.zeta = zeta

        self.graph = gt.Graph(directed=False)
        self.graph.add_vertex(n=self.vertex_set.size)
        self.edge_costs = self.graph.new_edge_property("double")
        self.graph.edge_properties["edge_costs"] = self.edge_costs

        self.sample_edges()
        self.sample_edge_costs()

        # These will be set on calling self.run_infection, which takes the initial vertex as an argument.
        # Uninfected vertices have infinite infection time. Vertices without an infector (including the source vertex)
        # have infector id -1.
        self.infection_times = self.graph.new_vertex_property("double")
        self.graph.vertex_properties["infection_times"] = self.infection_times
        self.infectors = self.graph.new_vertex_property("int64_t")
        self.graph.vertex_properties["infectors"] = self.infectors
        self.initial_vertex = None

    def sample_edges(self):
        """(Re)samples the edge set of the graph from the given vertex set. Vertex IDs in the graph correspond to
        vertex IDs from the WeightedVertexSet."""
        self.graph.clear_edges()
        edges = self.edge_generator(self.vertex_set)
        self.graph.add_edge_list(edges)

    def sample_edge_costs(self):
        """(Re)samples only the edge costs while maintaining the current edge set."""
        for edge in self.graph.edges():
            u_id = self.graph.vertex_index[edge.source()]
            v_id = self.graph.vertex_index[edge.target()]
            new_cost = self.edge_cost_generator() * self.vertex_set.penalty(u_id, v_id, mu=self.mu, zeta=self.zeta)
            self.edge_costs[edge] = new_cost

    def run_infection(self, initial_vertex_id: int):
        """Runs an SI infection on the stored graph, storing the results in self.infection_time and self.infector.
        For each Vertex v in self.graph, self.infection_time[v] will be the infection time of v and self.infector[v]
        will be the node that infected v; these default to infinity (np.inf) and None respectively if v isn't in
        the same component as the initial vertex."""
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

        return len(self.infection_path(vertex_id)) - 1

    def save_to_file(self, folder: Path, name: str):
        """Logs all data in the current epidemic to the given file in pickle format. The epidemic will be saved in two
        files, one name.gt file and one name.pickle file."""
        with open(folder / f"{name}.pickle", "wb") as file:
            dill.dump(self.vertex_set, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.edge_generator, file, protocol=dill.HIGHEST_PROTOCOL)
            dill.dump(self.edge_cost_generator, file, protocol=dill.HIGHEST_PROTOCOL)

        self.graph.save(str(folder / f"{name}.gt"))

    @staticmethod
    def _empty_epidemic() -> SIEpidemic:
        """Creates an empty SIEpidemic object which can be initialised manually. Used when loading from files."""
        unweighted_vertices = lattice(size=1, dimension=1)
        weight_gen = fixed_weights_generator(weights={0: 1})
        weighted_vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_gen)
        edge_gen = fixed_graph_generator([])
        cost_gen = lambda: 0
        return SIEpidemic(vertex_set=weighted_vertices, edge_cost_generator=cost_gen, edge_generator=edge_gen, mu=0.,
                          zeta=0.)

    @staticmethod
    def load_from_file(folder: Path, name: str) -> SIEpidemic:
        """Loads an SIEpidemic saved via the save_to_file method as name.gt and name.pickle and returns the resulting
        object."""
        with open(folder / f"{name}.pickle", "rb") as file:
            vertices = dill.load(file)
            edge_gen = dill.load(file)
            cost_gen = dill.load(file)
            graph = gt.load_graph(str(folder / f"{name}.gt"))

            return_value = SIEpidemic._empty_epidemic()
            return_value.vertex_set = vertices
            return_value.edge_generator = edge_gen
            return_value.edge_cost_generator = cost_gen
            return_value.graph = graph
            return_value.infection_times = graph.vertex_properties["infection_times"]
            return_value.infectors = graph.vertex_properties["infectors"]
            return_value.edge_costs = graph.edge_properties["edge_costs"]

            return return_value


def girg_generator(alpha: float, scale_factor: float, generator: Optional[np.random.Generator] = None,
                   average_degree: Optional[float] = None) -> Callable[[WeightedVertexSet], List[Tuple[int, int]]]:
    """Generates an SIEpidemic whose graph is a GIRG on the given vertex_set with the given long-range parameter
    alpha and the given RNG seed. The connection probability between u and v is max(1, W_uW_v/|u-v|^d)^\alpha. If
    average_degree is set then the GIRG library uses binary search to find a constant c to scale the weights by which
    gives the appropriate average degree. The generator here should only be used for testing purposes, in which case
    it will be used to generate random seeds - otherwise you should leave this as None and let the sampling library
    generate its own seed.

    The GIRG generation code we're using requires everything to be scaled down to [0,1]^d, using connection probability
    max(1, W_uW_v/n|u-v|^d)^\alpha, and we don't want to change that. As long as we're working in [0, n^{1/d}]^d, this
    is equivalent to scaling everything down by a factor of n^{1/d}. The one catch is that we need to pass this in as
    an argument rather than generating it from the vertex set, since otherwise if we have a PPP with slightly less than
    n points then we might end up with points outside [0, 1]^d and the generation code will crash. So instead we pass
    1/n^{1/d} in as the scale_factor argument."""
    def generator_to_return(vertex_set: WeightedVertexSet) -> List[Tuple[Any, Any]]:
        seed = None
        if generator:
            seed = generator.integers(low=0, high=2**31)  # girg-sampling takes 31-bit seeds, "high" is not inclusive.

        weights = [vertex_set.weight(i) for i in range(vertex_set.size)]

        """The GIRG generator creates a GIRG with connection probability between u and v given by max(1, 
        W_uW_v/n|u-v|^d)^alpha. It also requires all points to lie in [0,1]^d. So we need to scale everything down by 
        a factor of n^{1/d}."""
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

    return generator_to_return


def fixed_graph_generator(edges: List[Tuple[int, int]]) -> Callable[[WeightedVertexSet], List[Tuple[int, int]]]:
    """'Edge generator' for a constant graph such as the Gowalla dataset, to be passed to SIEpidemic. The edges
    should be specified as a list of pairs of vertex IDs."""
    return lambda vertices: edges


def fpp_generator(lambda_: float, generator: Optional[np.random.Generator] = None) -> Callable[[], float]:
    """Edge cost generator for first passage percolation (i.e. the random part of edge costs are i.i.d. exponentially
    distributed) with the given parameter."""
    if generator is None:
        generator = np.random.default_rng()
    return lambda: generator.exponential(scale=lambda_)


def constant_generator(c: float) -> Callable[[], float]:
    """Edge cost generator for testing purposes that sets the random part of all edge costs to the given constant."""
    return lambda: c
