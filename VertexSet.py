from math import pow
import haversine
import functools
import itertools
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass(slots=True)
class VertexData:
    """Stores unique identifiers for a single vertex in the network. id_ is the numerical ID as used by graph_tool -
    this must be an integer between 0 and the number of vertices in the network. position is the vertex's coordinates
    in space. name is an arbitrary user-defined name."""
    id_: int
    position: Tuple[float, ...]
    name: Any


class VertexSet:
    def __init__(self, dimension: int, metric: Callable[[Tuple[float, ...], Tuple[float, ...]], float]):
        """Stores a vertex set for a GIRG with spatial data and provides some spatial utility functions. Usage:
        Construct with VertexSet(dimension, distance), then set the actual vertices using either setPointsFromIds or
        setPoints."""
        self.dimension = dimension  # Dimension of the space (used for e.g. edge probabilities)
        # Distance function - should take two points in the space and return the distance between them
        self.metric = metric
        # Points in the vertex set. Note we don't store these as NumPy arrays as iterating over these or reading
        # individual values is super-slow. We'll store a NumPy array separately for when it's useful and use the
        # position_to_id map to go from the raw positions stored in the NumPy array to the IDs stored here.
        self._vertex_id_dict = {}
        self._vertex_position_dict = {}
        self._vertex_name_dict = {}

    def name_to_data(self, name: str) -> VertexData:
        return self._vertex_name_dict[name]

    def name_to_position(self, name: Any) -> Tuple[float, ...]:
        return self._vertex_name_dict[name].position

    def name_to_id(self, name: Any) -> int:
        return self._vertex_name_dict[name].id_

    def position_to_data(self, position: Tuple[float, ...]) -> VertexData:
        return self._vertex_position_dict[position]

    def position_to_name(self, position: Tuple[float, ...]) -> Any:
        return self._vertex_position_dict[position].name

    def position_to_id(self, position: Tuple[float, ...]) -> int:
        return self._vertex_position_dict[position].id_

    def id_to_data(self, id_: int) -> VertexData:
        return self._vertex_id_dict[id_]

    def id_to_name(self, id_: int) -> Any:
        return self._vertex_id_dict[id_].name

    def id_to_position(self, id_: int) -> Tuple[float, ...]:
        return self._vertex_id_dict[id_].position

    def distance(self, id_x, id_y):
        """Returns the distance between the two points with the given IDs."""
        return self.metric(self.name_to_position(id_x), self.name_to_position(id_y))

    def setPointsFromNames(self, points: Dict[Any, Tuple[float, ...]]):
        """Initialises points from a pre-existing dictionary mapping names to coordinates"""
        for id_, (name, position) in enumerate(points.items()):
            new_vertex = VertexData(id_=id_, position=position, name=name)
            self._vertex_id_dict[id_] = new_vertex
            self._vertex_position_dict[position] = new_vertex
            self._vertex_name_dict[name] = new_vertex

    def setPoints(self, points: List[Tuple[float, ...]]):
        """Initialises points without pre-set names (generating arbitrary names)"""
        self.setPointsFromNames({i: points[i] for i in range(len(points))})

    @property
    def vertex_data(self):
        """Returns a list of all VertexData objects currently stored."""
        return self._vertex_id_dict.values()

    @property
    def size(self) -> int:
        """Returns the total number of vertices in the vertex set."""
        return len(self._vertex_id_dict)

    @property
    def positions(self) -> Set[Tuple[float, ...]]:
        """Returns the set of all positions in the vertex set."""
        return set(self._vertex_position_dict.keys())

    @property
    def names(self) -> Set[Any]:
        """Returns the set of all vertex names in the vertex set."""
        return set(self._vertex_name_dict.keys())

    @property
    def ids(self) -> Set[int]:
        return set(self._vertex_id_dict.keys())

    def getIdsInAnnulus(self, center: Tuple[float, ...], inner_radius: float, outer_radius: float) -> List[int]:
        """Returns a list of IDs of points whose distance from center lies in [inner_radius, outer_radius]."""
        id_list = []
        for vertex in self.vertex_data:
            distance = self.metric(center, vertex.position)
            if (distance >= inner_radius) and (distance <= outer_radius):
                id_list.append(vertex.id_)
        return id_list

    def getIdsInBall(self, center: Tuple[float, ...], radius: float) -> List[int]:
        """Returns a list of IDs of points which lie in the closed ball of radius r centered at center."""
        return self.getIdsInAnnulus(center=center, inner_radius=0, outer_radius=radius)


def euclideanDistance(x: Tuple[float, ...], y: Tuple[float, ...], d: int) -> float:
    """Returns the distance between x and y in a Euclidean space of dimension d."""
    return pow(sum((x[i] - y[i])**d for i in range(len(x))), 1/d)


def euclideanDistanceFunction(d: int) -> Callable[[Tuple[float, ...], Tuple[float, ...]], float]:
    """Returns the distance *function* for a Euclidean space of dimension d, i.e. currying d into euclideanDistance."""
    return functools.partial(euclideanDistance, d=d)


def torusDistance(x: Tuple[float], y: Tuple[float], size: float, d: int) -> float:
    """Returns the distance between x and y on [0,size]^d considered as a torus."""
    l1_distances = [0]*d
    for i in range(d):
        euclidean_distance = abs(x[i] - y[i])
        l1_distances[i] = euclidean_distance if euclidean_distance <= size/2 else size - euclidean_distance

    return pow(sum(x**d for x in l1_distances), 1/d)


def torusDistanceFunction(d: int, size: float) -> Callable[[Tuple[float, ...], Tuple[float, ...]], float]:
    """Returns the Torus distance *function* for [0,size]^d, i.e. currying d and size into euclideanDistance."""
    return functools.partial(torusDistance, d=d, size=size)


def earthDistance(x: Tuple[float, ...], y: Tuple[float, ...]) -> float:
    """Returns the distance in kilometres between x and y on the surface of Earth, where x and y are given in
    (latitude, longitude) format. Uses the Haversine formula (so it assumes the earth is a sphere)."""
    return haversine.haversine(x, y)


def lattice(dimension: int, size: int) -> VertexSet:
    """Returns a VertexSet for the integer lattice spanning [0, size]^dimension under Euclidean distance."""
    # Curry the dimension into the distance
    return_value = VertexSet(dimension=dimension, metric=torusDistanceFunction(d=dimension, size=size))

    # Note this generates size^d points, not (size+1)^d points, as the torus wraps at the boundaries.
    one_axis_points = [float(i) for i in range(0, size)]
    points = list(itertools.product(one_axis_points, repeat=dimension))
    return_value.setPoints(points)

    return return_value


def poissonPointProcess(dimension: int, size: float, generator: Optional[np.random.Generator] = None) -> VertexSet:
    """Returns a VertexSet for a Poisson point process of density 1 in [0, size]^dimension using the specified RNG,
    with a planted point in the center and using torus distance."""
    if generator is None:
        generator = np.random.default_rng()

    # We simulate a PPP by choosing Po(size^dimension) points independently and uniformly at random.
    point_count = generator.poisson(size**dimension)
    points = []
    for i in range(point_count):
        next_point_coordinates = tuple(generator.uniform(low=0., high=size, size=dimension))
        points.append(next_point_coordinates)

    return_value = VertexSet(dimension=dimension, metric=torusDistanceFunction(dimension, size))
    return_value.setPoints(points)
    return return_value
