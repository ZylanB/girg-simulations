from dataclasses import dataclass
import functools
import itertools
from math import pow
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import haversine  # type: ignore
import numpy as np

from LoggableFunction import LoggableFunction


class Metric(LoggableFunction[[Sequence[float], Sequence[float]], float]):
    """Distance function between two positions in a VertexSet. Should be symmetric."""
    @property
    def function_role(self):
        return "Distance function on vertex set"


class GenericMetric(Metric):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str):
        self.description = description
        self._function = _function


@dataclass(slots=True)
class VertexData:
    """Stores unique identifiers for a single vertex in the network. id_ is the numerical ID as used by graph_tool -
    this must be an integer between 0 and the number of vertices in the network. position is the vertex's coordinates
    in space. name is an arbitrary user-defined name."""
    id_: int
    position: Sequence[float]
    name: Any


class VertexSet:
    def __init__(self, dimension: int, metric: Metric):
        """Stores a vertex set for a GIRG with spatial data and provides some spatial utility functions. Usage:
        Construct with VertexSet(dimension, distance), then set the actual vertices using either setPointsFromIds or
        setPoints."""
        self.dimension = dimension  # Dimension of the space (used for e.g. edge probabilities)
        # Distance function - should take two points in the space and return the distance between them
        self.metric = metric
        # Points in the vertex set. Note we don't store these as NumPy arrays as iterating over these or reading
        # individual values is super-slow. We'll store a NumPy array separately for when it's useful and use the
        # position_to_id map to go from the raw positions stored in the NumPy array to the IDs stored here.
        self._vertex_id_dict: Dict[int, VertexData] = {}
        self._vertex_position_dict: Dict[Sequence[float], VertexData] = {}
        self._vertex_name_dict: Dict[Any, VertexData] = {}

    def name_to_data(self, name: str) -> VertexData:
        return self._vertex_name_dict[name]

    def name_to_position(self, name: Any) -> Sequence[float]:
        return self._vertex_name_dict[name].position

    def name_to_id(self, name: Any) -> int:
        return self._vertex_name_dict[name].id_

    def position_to_data(self, position: Sequence[float]) -> VertexData:
        return self._vertex_position_dict[position]

    def position_to_name(self, position: Sequence[float]) -> Any:
        return self._vertex_position_dict[position].name

    def position_to_id(self, position: Sequence[float]) -> int:
        return self._vertex_position_dict[position].id_

    def id_to_data(self, id_: int) -> VertexData:
        return self._vertex_id_dict[id_]

    def id_to_name(self, id_: int) -> Any:
        return self._vertex_id_dict[id_].name

    def id_to_position(self, id_: int) -> Sequence[float]:
        return self._vertex_id_dict[id_].position

    def distance(self, id_x, id_y):
        """Returns the distance between the two points with the given IDs."""
        return self.metric(self.id_to_position(id_x), self.id_to_position(id_y))

    def set_points_from_names(self, points: Mapping[Any, Sequence[float]]):
        """Initialises points from a pre-existing dictionary mapping names to coordinates"""
        for id_, (name, position) in enumerate(points.items()):
            new_vertex = VertexData(id_=id_, position=position, name=name)
            self._vertex_id_dict[id_] = new_vertex
            self._vertex_position_dict[position] = new_vertex
            self._vertex_name_dict[name] = new_vertex

    def set_points(self, points: Sequence[Sequence[float]]):
        """Initialises points without pre-set names (generating arbitrary names)"""
        self.set_points_from_names({i: points[i] for i in range(len(points))})

    @property
    def vertex_data(self):
        """Returns a list of all VertexData objects currently stored."""
        return self._vertex_id_dict.values()

    @property
    def count(self) -> int:
        """Returns the total number of vertices in the vertex set."""
        return len(self._vertex_id_dict)

    @property
    def positions(self) -> Iterable[Sequence[float]]:
        """Returns an iterator over all positions in the vertex set."""
        return self._vertex_position_dict.keys()

    @property
    def names(self) -> Iterable[Any]:
        """Returns an iterator over all vertex names in the vertex set."""
        return self._vertex_name_dict.keys()

    @property
    def ids(self) -> Iterable[int]:
        """Returns an iterator over all vertex IDs in the vertex set."""
        return self._vertex_id_dict.keys()

    def get_ids_in_annulus(self, center: Sequence[float], inner_radius: float, outer_radius: float) -> List[int]:
        """Returns a list of IDs of points whose distance from center lies in [inner_radius, outer_radius]."""
        id_list = []
        for vertex in self.vertex_data:
            distance = self.metric(center, vertex.position)
            if (distance >= inner_radius) and (distance <= outer_radius):
                id_list.append(vertex.id_)
        return id_list

    def get_ids_in_ball(self, center: Sequence[float], radius: float) -> List[int]:
        """Returns a list of IDs of points which lie in the closed ball of radius r centered at center."""
        return self.get_ids_in_annulus(center=center, inner_radius=0, outer_radius=radius)


class EuclideanDistance(Metric):
    """Returns the distance between x and y in a Euclidean space of dimension d."""
    def __init__(self, d: int):
        self.dimension = d
        self._function = functools.partial(self._euclidean_distance, d=d)

    @staticmethod
    def _euclidean_distance(x: Sequence[float], y: Sequence[float], d: int) -> float:
        return pow(sum(abs(x[i] - y[i]) ** d for i in range(len(x))), 1 / d)


class TorusDistance(Metric):
    def __init__(self, d: int, size: float):
        """Returns the distance between x and y on [0,size]^d considered as a torus."""
        self.dimension = d
        self.size = size
        self._function = functools.partial(self._torus_distance, d=d, size=size)

    @staticmethod
    def _torus_distance(x: Sequence[float], y: Sequence[float], size: float, d: int) -> float:
        l1_distances = [0.] * d
        for i in range(d):
            interval_distance = abs(x[i] - y[i])
            l1_distances[i] = interval_distance if interval_distance <= size / 2 else size - interval_distance

        return pow(sum(x ** d for x in l1_distances), 1 / d)


class EarthDistance(Metric):
    """Returns the distance in kilometres between x and y on the surface of Earth, where x and y are given in
        (latitude, longitude) format. Uses the Haversine formula (so it assumes the earth is a sphere)."""
    def __init__(self):
        self._function = lambda x, y: haversine.haversine(x, y, unit=haversine.Unit.KILOMETERS)


class VertexSetGen(LoggableFunction[[np.random.Generator], VertexSet]):
    @property
    def function_role(self) -> str:
        return "Vertex set generator"


class GenericVertexSetGen(VertexSetGen):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self.description = description
        self._function = _function


class FixedVertexSet(VertexSetGen):
    """Returns a fixed vertex set with the given metric and with dimension inferred from the given map of vertex IDs to
    positions in space."""
    def __init__(self, points: Mapping[Any, Sequence[float]], metric: Metric, description: str):
        dimension = len(next(iter(points.values())))  # Check the dimension of an arbitrary point
        for point in points.values():
            if len(point) != dimension:
                raise ValueError("Points must all have the same dimension!")

        self.metric = metric
        self.dimension = dimension
        self.description = description
        self._function = functools.partial(self._get_vertex_set, points=points)

    def _get_vertex_set(self, _: np.random.Generator, points: Mapping[Any, Sequence[float]]) -> VertexSet:
        return_value = VertexSet(dimension=self.dimension, metric=self.metric)
        return_value.set_points_from_names(points)
        return return_value


class Lattice(VertexSetGen):
    """Returns a VertexSet for the integer lattice spanning [0, size]^dimension under Euclidean distance."""
    def __init__(self, dimension: int, size: int):
        def _function(_):
            # Curry the dimension into the distance
            vertex_set = VertexSet(dimension=dimension, metric=TorusDistance(d=dimension, size=size))

            # Note this generates size^d points, not (size+1)^d points, as the torus wraps at the boundaries.
            one_axis_points = [float(i) for i in range(0, size)]
            points = list(itertools.product(one_axis_points, repeat=dimension))
            vertex_set.set_points(points)

            return vertex_set

        self.size = size
        self.dimension = dimension
        self._function = _function


class PoissonPointProcess(VertexSetGen):
    """Returns a VertexSet for a Poisson point process of density 1 in [0, size]^dimension using the specified RNG,
    with a planted point in the center and using torus distance."""
    def __init__(self, dimension: int, size: float):
        def _function(rng: np.random.Generator) -> VertexSet:
            # We simulate a PPP by choosing Po(size^dimension) points independently and uniformly at random.
            point_count = rng.poisson(size ** dimension)
            points = []
            for i in range(point_count):
                next_point_coordinates = tuple(rng.uniform(low=0., high=size, size=dimension))
                points.append(next_point_coordinates)

            vertex_set = VertexSet(dimension=dimension, metric=TorusDistance(dimension, size))
            vertex_set.set_points(points)
            return vertex_set

        self.size = size
        self.dimension = dimension
        self._function = _function
