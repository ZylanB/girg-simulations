import numpy as np
import graph_tool as gt
from graph_tool.topology import shortest_distance
import girg_sampling.girgs as gs

from WeightedVertexSet import WeightedVertexSet
from typing import Any, Callable, List, Optional, Tuple


class SIEpidemic:
    """Stores a spatial graph whose edge weights determine the spread of an SI model epidemic. Edge_cost_generator
    should return a sample of the *random* part of an edge's cost, i.e. not including degree or spatial penalties (
    which are calculated in WeightedVertexSet)."""
    def __init__(self, vertex_set: WeightedVertexSet, edge_cost_generator: Callable[[], float],
                 edge_generator: Callable[[WeightedVertexSet], List[Tuple[int, int]]]):
        self.vertex_set = vertex_set
        self.edge_cost_generator = edge_cost_generator
        self.edge_generator = edge_generator

        self.graph = gt.Graph(directed=False)
        self.graph.add_vertex(n=self.vertex_set.size)
        self.edge_costs = self.graph.new_edge_property("double")

        self.sample_edges()
        self.sample_edge_costs()

        # These will be set on calling self.run_infection, which takes the initial vertex as an argument.
        # Uninfected vertices have infinite infection time. Vertices without an infector (including the source vertex)
        # have infector id -1.
        self.infection_times = self.graph.new_vertex_property("double")
        self.infectors = self.graph.new_vertex_property("int64_t")
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
            new_cost = self.edge_cost_generator() * self.vertex_set.penalty(u_id, v_id)
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


def girg_generator(alpha: float, scale_factor: float, average_degree: Optional[float] = None,
                   seed: Optional[int] = None) -> Callable[[WeightedVertexSet], List[Tuple[int, int]]]:
    """Generates an SIEpidemic whose graph is a GIRG on the given vertex_set with the given long-range parameter
    alpha and the given RNG seed. The GIRG sampling library unfortunately works in [0,1]^d, with the connection
    probability from u to v given by min(1, W_uW_v/n||u-v||^d)^alpha - as such, before we pass positions in,
    we need to scale our space down to fit in [0,1]. The correct scale factor is E(#(vertices))^{1/d}. If
    average_degree is set then the GIRG library uses binary search to find a constant c to scale the weights by which
    gives the appropriate average degree. If seed is not set then the sampling library takes care of it."""
    def generator(vertex_set: WeightedVertexSet) -> List[Tuple[Any, Any]]:
        weights = [vertex_set.weight(i) for i in range(vertex_set.size)]

        positions = [vertex_set.id_to_position(i) for i in range(vertex_set.size)]
        scaled_positions = []
        for position in positions:
            scaled_positions.append([position[i] / scale_factor for i in range(vertex_set.dimension)])

        if average_degree is not None:
            c = gs.scaleWeights(weights=weights, desiredAvgDegree=average_degree, dimension=vertex_set.dimension,
                                alpha=alpha)
            weights = [c * weight for weight in weights]

        return gs.generateEdges(weights=weights, positions=scaled_positions, alpha=alpha, seed=seed)

    return generator


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


def unit_generator() -> Callable[[], float]:
    """Edge cost generator for testing purposes that sets the random part of all edge costs to 1."""
    return lambda: 1.0
