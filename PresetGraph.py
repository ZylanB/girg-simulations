from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypedDict

import dill  # type: ignore
import graph_tool.all as gt  # type: ignore
import csv
import numpy as np

import config
from SIEpidemic import EdgeCostGen, EdgeGen, GenericEdgeGen, SIEpidemic
from VertexSet import GenericVertexSetGen, VertexSet, VertexSetGen
from WeightedVertexSet import GenericWeightGen, WeightedVertexSet, WeightGen


class PresetSIEpidemicArgs(TypedDict):
    """This is just for type checking when a PresetGen supplies keyword arguments to the SIEpidemic constructor."""
    vertex_gen: Optional[VertexSetGen]
    weight_gen: Optional[WeightGen]
    edge_gen: Optional[EdgeGen]


class PresetSIExperimentArgs(TypedDict):
    """This is just for type checking when a PresetGen supplies keyword arguments to the SIExperiment constructor."""
    vertex_gen: Optional[VertexSetGen]
    weight_gen: Optional[WeightGen]
    edge_gen: Optional[EdgeGen]
    resample_edges: bool
    resample_weights: bool
    resample_vertices: bool


@dataclass
class GraphData:
    vertex_set: VertexSet
    weights: Sequence[float]  # List of weights indexed by vertex ID
    edge_list: Sequence[Tuple[int, int]]  # List of edges indexed by vertex ID, direction doesn't matter


def extract_giant_data(data: GraphData) -> GraphData:
    """Pulls out vertex, weight and edge data for the largest component of the epidemic's graph. Ties are broken
    deterministically. Pulled out of the SIEpidemic class for efficiency/neatness."""
    print("Loading graph...")
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=data.vertex_set.count)
    graph.add_edge_list(data.edge_list)

    print("Finding giant component...")
    giant = gt.extract_largest_component(graph)

    print("Renumbering vertices of giant component...")
    induced_vertex_ids = [int(v) for v in giant.vertices()]
    name_to_pos_dict = {data.vertex_set.id_to_name(i): data.vertex_set.id_to_position(i) for i in induced_vertex_ids}
    giant_vertex_set = VertexSet(dimension=2, metric=data.vertex_set.metric)
    giant_vertex_set.set_points_from_names(name_to_pos_dict)
    old_to_new_ids = {induced_vertex_ids[i]: i for i in range(len(induced_vertex_ids))}
    giant_weights = [0.] * giant_vertex_set.count
    for i in induced_vertex_ids:
        giant_weights[old_to_new_ids[i]] = data.weights[i]
    giant_edge_list = [(old_to_new_ids[e.source()], old_to_new_ids[e.target()]) for e in giant.edges()]
    return GraphData(vertex_set=giant_vertex_set, weights=giant_weights, edge_list=giant_edge_list)


class PresetGraph:
    """This class is intended for the common use case of generating a single graph that gets re-used for many
    experiments, e.g. from the Gowalla dataset."""
    def __init__(self, seed_override: Optional[int] = None, base_folder: Path = config.DATA_FOLDER,
                 generator_args: Optional[Dict[str, Any]] = None):
        seed = seed_override if seed_override is not None else self.default_seed
        self.rng = np.random.default_rng(seed)
        self.base_folder = base_folder
        self.generator_args = generator_args if generator_args is not None else {}

    @property
    def default_seed(self) -> int:
        """We need to use a different RNG to generate the graph than we use to run the SIExperiment it's used in,
        or the results of the experiment for a given seed will depend on whether the graph's files are present. As
        such, intended use is to generate a random seed (np.random.SeedSequence().entropy) once and then hardcode it
        as the DEFAULT_SEED of the inherited class, with the hard-coding providing a record for reproducibility in the
        potential absence of log files."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Description of the graph for logging purposes."""
        raise NotImplementedError

    @property
    def graph_generator_class(self) -> Type[PresetGen]:
        raise NotImplementedError

    def _vertex_gen_fn(self, _: np.random.Generator):
        # Note we throw away the RNG that comes from the calling SIExperiment, since whether or not it gets called
        # will depend on whether or not the graph data already exists.
        generator = self.graph_generator_class(rng=self.rng, base_folder=self.base_folder, **self.generator_args)
        if not generator.data_exists:
            generator.create_data()
            generator.save_data()
            return generator.vertices
        print("Loading vertices...")
        return generator.vertices_from_file()

    @property
    def graph_data(self) -> GraphData:
        """Returns the vertices/weights/edges of the graph in "raw" form, e.g. for use with another PresetGen."""
        generator = self.graph_generator_class(rng=self.rng, base_folder=self.base_folder, **self.generator_args)
        if not generator.data_exists:
            generator.create_data()
            generator.save_data()
            return GraphData(vertex_set=generator.vertices, weights=generator.weights, edge_list=generator.edge_list)

        print("Loading data...")
        return GraphData(vertex_set=generator.vertices_from_file(), weights=generator.weights_from_file(),
                         edge_list=generator.edge_list_from_file())

    @property
    def vertex_gen(self):
        """Vertex generator function to pass into an SIExperiment. If the graph data exists, loads the vertex set and
        returns it. Otherwise, creates all the graph files and returns the vertex set. This approach means when we log
        the vertex generator function we don't have to store the entire (large) vertex set."""
        return GenericVertexSetGen(_function=self._vertex_gen_fn, description=self.description)

    def _weight_gen_fn(self, _: VertexSet, __: np.random.Generator):
        # Note we throw away the RNG that comes from the calling SIExperiment, since whether or not it gets called
        # will depend on whether or not the graph data already exists.
        generator = self.graph_generator_class(rng=self.rng, base_folder=self.base_folder, **self.generator_args)
        if not generator.data_exists:
            generator.create_data()
            generator.save_data()
            return generator.weights
        print("Loading weights...")
        return generator.weights_from_file()

    @property
    def weight_gen(self):
        """Weight generator function to pass into an SIExperiment. If the graph data exists, loads the vertex weights
        and returns them. Otherwise, creates all the graph files and returns the weights. This approach means when we
        log the weight generator function we don't have to store the entire (large) weight list."""
        return GenericWeightGen(_function=self._weight_gen_fn, description=self.description)

    def _edge_gen_fn(self, _: WeightedVertexSet, __: np.random.Generator):
        # Note we throw away the RNG that comes from the calling SIExperiment, since whether or not it gets called
        # will depend on whether or not the graph data already exists.
        generator = self.graph_generator_class(rng=self.rng, base_folder=self.base_folder, **self.generator_args)
        if not generator.data_exists:
            generator.create_data()
            generator.save_data()
            return generator.edge_list
        print("Loading edges...")
        return generator.edge_list_from_file()

    @property
    def edge_gen(self):
        """Edge generator function to pass into an SIExperiment. If the graph data exists, loads the edge list and
        returns it. Otherwise, creates all the graph files and returns the edges. This approach means when we log the
        edge generator function we don't have to store the entire (large) edge list."""
        return GenericEdgeGen(_function=self._edge_gen_fn, description=self.description)

    def create_epidemic(self, cost_gen: EdgeCostGen, mu: float, zeta: float, name: str,
                        rng: np.random.Generator) -> SIEpidemic:
        vertex_set = WeightedVertexSet(vertex_gen=self.vertex_gen, weight_gen=self.weight_gen, rng=rng)
        return SIEpidemic(vertex_set=vertex_set, edge_gen=self.edge_gen, cost_gen=cost_gen, mu=mu, zeta=zeta,
                          name=name, rng=rng)

    @property
    def experiment_params(self) -> PresetSIExperimentArgs:
        """Pre-filled parameters to build an SIExperiment to be run on this graph, setting the generators for vertices,
        weights and edges and preventing them from being resampled."""
        return {'vertex_gen': self.vertex_gen, 'weight_gen': self.weight_gen, 'edge_gen': self.edge_gen,
                'resample_vertices': False, 'resample_weights': False, 'resample_edges': False}


class PresetGen:
    """This is a helper class for PresetGraph that actually creates the graph. The two need to be kept separate because
    PresetGen stores things like vertices and edges as members, which will be picked up as part of the closures of
    PresetGraph.vertex_gen etc. and then saved in full as part of logging (taking excessive extra space and time)."""
    SAVED_VERTEX_FILENAME = "vertices.pickle"
    SAVED_WEIGHT_FILENAME = "weights.json"
    SAVED_EDGE_FILENAME = "edges.csv"

    def __init__(self, rng: np.random.Generator, base_folder: Path):
        self.rng = rng
        self.base_folder = base_folder
        self.vertices: Optional[VertexSet] = None
        self.weights: Optional[Sequence[float]] = None
        self.edge_list: Optional[Sequence[Tuple[int, int]]] = None

    @property
    def save_folder(self) -> Path:
        """Path in which generated files are stored."""
        return self.base_folder / self.name

    @property
    def vertex_path(self) -> Path:
        """Path in which the graph's vertices are stored."""
        return self.save_folder / self.SAVED_VERTEX_FILENAME

    @property
    def weight_path(self) -> Path:
        """Path in which the graph's weights are stored."""
        return self.save_folder / self.SAVED_WEIGHT_FILENAME

    @property
    def edge_path(self) -> Path:
        """Path in which the graph's edges are stored."""
        return self.save_folder / self.SAVED_EDGE_FILENAME

    @property
    def name(self) -> str:
        """Subfolder name in which to store generated files."""
        raise NotImplementedError

    @property
    def data_exists(self) -> bool:
        """Returns true if the graph has already been created and we don't need to re-create it."""
        return self.vertex_path.exists() and self.weight_path.exists() and self.edge_path.exists()

    def create_data(self) -> None:
        raise NotImplementedError

    def save_data(self):
        self.save_folder.mkdir(parents=True, exist_ok=True)

        print("Saving vertex set...")
        with open(self.vertex_path, "wb") as file:
            dill.dump(self.vertices, file)  # type: ignore

        print("Saving weight list...")
        with open(self.weight_path, "wb") as file:
            dill.dump(self.weights, file)  # type: ignore

        # Using csv rather than dill because dill is slooooowwww for large objects and the edge sets can be 180+MB.
        print("Saving edge list...")
        with open(self.edge_path, "w") as file:
            writer = csv.writer(file, delimiter=',')  # type: ignore
            for entry in self.edge_list:
                writer.writerow(entry)

        print("All saved.")

    def vertices_from_file(self) -> VertexSet:
        with open(self.vertex_path, "rb") as file:
            return dill.load(file)

    def weights_from_file(self) -> Sequence[float]:
        with open(self.weight_path, "rb") as file:
            return dill.load(file)

    def edge_list_from_file(self) -> Sequence[Tuple[int, int]]:
        edge_list = []
        with open(self.edge_path, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                edge_list.append((int(row[0]), int(row[1])))
        return edge_list
