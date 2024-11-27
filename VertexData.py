from math import pow
import haversine
import functools
import itertools
import numpy as np
from typing import Callable, List, Dict, Tuple, Any, Optional


class VertexData:
    def __init__(self, dimension: int, distance_function: Callable[[Tuple[float, ...], Tuple[float, ...]], float]):
        """Stores a vertex set for a GIRG and provides some spatial utility functions. Usage: Construct with
        VertexData(dimension, distance), then set the actual vertices using either setPointsFromIds or
        setPoints."""
        self.dimension = dimension  # Dimension of the space (used for e.g. edge probabilities)
        # Distance function - should take two points in the space and return the distance between them
        self.distance = distance_function
        # Points in the vertex set. Note we don't store these as NumPy arrays as iterating over these or reading
        # individual values is super-slow. We'll store a NumPy array separately for when it's useful and use the
        # position_to_id map to go from the raw positions stored in the NumPy array to the IDs stored here.
        self.id_to_position = None
        self.position_to_id = None

    def setPointsFromIds(self, points: Dict[Any, Tuple[float, ...]]):
        """Initialises points from a pre-existing dictionary mapping IDs to coordinates"""
        self.id_to_position = points
        self.position_to_id = {position: point_id for point_id, position in points.items()}

    def setPoints(self, points: List[Tuple[float, ...]]):
        """Initialises points without pre-set IDs (generating arbitrary IDs)"""
        self.id_to_position = {i: points[i] for i in range(len(points))}
        self.position_to_id = {points[i]: i for i in range(len(points))}

    @property
    def positions(self):
        return self.id_to_position.values()

    @property
    def ids(self):
        return self.position_to_id.values()

    def getIdsInAnnulus(self, center: Tuple[float, ...], inner_radius: float, outer_radius: float) -> List[int]:
        """Returns a list of IDs of points whose distance from center lies in [inner_radius, outer_radius]."""
        id_list = []
        for point_id, point in self.id_to_position.items():
            distance = self.distance(center, point)
            if (distance >= inner_radius) and (distance <= outer_radius):
                id_list.append(point_id)
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


def lattice(dimension: int, size: int) -> VertexData:
    """Returns a VertexData for the integer lattice spanning [0, size]^dimension under Euclidean distance."""
    # Curry the dimension into the distance
    return_value = VertexData(dimension=dimension, distance_function=torusDistanceFunction(d=dimension, size=size))

    # Note this generates size^d points, not (size+1)^d points, as the torus wraps at the boundaries.
    one_axis_points = [float(i) for i in range(0, size)]
    points = list(itertools.product(one_axis_points, repeat=dimension))
    return_value.setPoints(points)

    return return_value


def poissonPointProcess(dimension: int, size: float, generator: Optional[np.random.Generator]) -> VertexData:
    """Returns a VertexData for a Poisson point process of density 1 in [0, size]^dimension using the specified RNG,
    with a planted point in the center and using torus distance."""
    if generator is None:
        generator = np.random.default_rng()

    # We simulate a PPP by choosing Po(size^dimension) points independently and uniformly at random.
    point_count = generator.poisson(size**dimension)
    points = []
    for i in range(point_count):
        next_point_coordinates = tuple(generator.uniform(size=dimension))
        points.append(next_point_coordinates)

    return_value = VertexData(dimension=dimension, distance_function=torusDistanceFunction(dimension, size))
    return_value.setPoints(points)
    return return_value
