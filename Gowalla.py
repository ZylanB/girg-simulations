from __future__ import annotations
from collections import defaultdict
import datetime
from dataclasses import dataclass
import gzip
from pathlib import Path
import shutil
from typing import List, Optional, Sequence, Tuple, Type

import dill  # type: ignore
import graph_tool.all as gt  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import requests

import config
from PresetGraph import extract_giant_data, GraphData, PresetGraph, PresetGen
from SIEpidemic import GirgGen
from SIExperiment import InitialVertexFunction
from VertexSet import EarthDistance, VertexSet, PoissonPointProcess, TorusDistance
from WeightedVertexSet import PowerLawWeightGen, IdentityWeightScaler, WeightedVertexSet

# Parameter values for synthetic GIRGs to mimic Gowalla.
SYN_GOWALLA_ALPHA = 1.17
SYN_GOWALLA_TAU = 2.8
SYN_GOWALLA_SIZE = 500


def obtain_data_file(url: str, dest_path: Path):
    """Downloads the given .gz (not .tar.gz) file from the given URL, unzips it in the given folder to the given
    name, and deletes the original .gz file."""
    dest_folder = dest_path.parent
    dest_folder.mkdir(parents=True, exist_ok=True)

    local_file = dest_folder / Path(url).name
    print(f"Downloading from {url} to {local_file}...")

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with local_file.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download completed.")

    print("Extracting archive...")
    with gzip.open(local_file, "rb") as gz, open(dest_path, "wb") as out:
        shutil.copyfileobj(gz, out)  # type: ignore
    print("Extraction completed.")

    print(f"Deleting temporary file {local_file}...")
    local_file.unlink()
    print("Temporary file deleted.")


@dataclass(slots=True)
class CheckIn:
    """Holds a single raw user check-in from the corresponding data file."""
    user_id: int
    time: datetime.datetime
    latitude: float
    longitude: float
    location_id: int

    @staticmethod
    def from_line(line: bytes) -> CheckIn:
        """Given a line of the Gowalla dataset, returns a corresponding CheckIn object."""
        sections = line.decode(encoding="utf-8").replace("\n", "").split("\t")
        time_format = "%Y-%m-%dT%H:%M:%SZ"
        return CheckIn(user_id=int(sections[0]), time=datetime.datetime.strptime(sections[1], time_format),
                       latitude=float(sections[2]), longitude=float(sections[3]), location_id=int(sections[4]))


@dataclass
class TieDatum:
    """Contains information about a user whose position is ambiguous from the check-in data file. The (discretised)
    possibilities are contained in modal_positions as latitude/longitude pairs."""
    user_id: int
    modal_positions: List[Tuple[float, float]]

    @property
    def max_distance(self) -> float:
        """Returns the maximum possible difference our choice of position for the user could make, i.e. the maximum
        distance between any pair of coordinates in self.modal_positions."""
        d = EarthDistance()
        return max([d(x, y) for x in self.modal_positions for y in self.modal_positions])


class GowallaGraph(PresetGraph):
    @property
    def description(self) -> str:
        return "Preset graph generated from the Gowalla dataset with the GowallaGen class."

    @property
    def default_seed(self) -> int:
        return 300574007011361565096923362210003471256

    @property
    def graph_generator_class(self) -> Type[PresetGen]:
        return GowallaGen


class GowallaGen(PresetGen):
    """Creates the Gowalla dataset from scratch, downloading the relevant files from _VERTEX_DATA_URL and
    _EDGE_DATA_URL and storing them at _VERTEX_DATA_PATH and _EDGE_DATA_PATH if need be. VERTEX_FILENAME and
    EDGE_FILENAME are the filenames to save the processed vertex and edge data to."""
    _VERTEX_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    _EDGE_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
    _VERTEX_DATA_PATH = config.DATA_FOLDER / "gowalla_data" / "Gowalla_totalCheckins.txt"
    _EDGE_DATA_PATH = config.DATA_FOLDER / "gowalla_data" / "Gowalla_edges.txt"

    def __init__(self, rng: np.random.Generator, base_folder: Path):
        if not self._snap_data_present():
            self._obtain_snap_data()
        self.ties: List[TieDatum] = []
        super().__init__(rng=rng, base_folder=base_folder)

    @property
    def name(self) -> str:
        return "gowalla_data"

    @classmethod
    def _obtain_snap_data(cls):
        """Downloads any of the Gowalla data not already present locally."""
        if not cls._VERTEX_DATA_PATH.exists():
            obtain_data_file(url=cls._VERTEX_DATA_URL, dest_path=cls._VERTEX_DATA_PATH)
        if not cls._EDGE_DATA_PATH.exists():
            obtain_data_file(url=cls._EDGE_DATA_URL, dest_path=cls._EDGE_DATA_PATH)

    @classmethod
    def _snap_data_present(cls):
        """Returns true if all the raw Gowalla data has already been downloaded from the SNAP site."""
        return cls._VERTEX_DATA_PATH.exists() and cls._EDGE_DATA_PATH.exists()

    def create_data(self) -> None:
        self.vertices = self._create_vertices()
        self.weights = self._create_weights(self.vertices)
        self.edge_list = self._create_edges(self.vertices)

    def _create_vertices(self) -> VertexSet:
        """Reads in the raw vertex data for the Gowalla dataset (downloading it first if needed), adds weights based
        on vertex degrees, and returns the resulting vertex set with mu and zeta initialised to 0. Nodes without
        position information from check-ins are omitted from the vertex set."""
        if not self._snap_data_present():
            self._obtain_snap_data()

        print("Reading check-ins...")
        user_id_dict = defaultdict(list)
        with open(self._VERTEX_DATA_PATH, "rb") as file:
            for i, line in enumerate(file):
                if i % 100000 == 0 and i > 0:
                    print(f"Check-in {i / 1000000}M/6.44M complete.")
                new_check_in = CheckIn.from_line(line)
                user_id_dict[new_check_in.user_id].append(new_check_in)

        print("Mapping user IDs to positions...")
        user_id_to_position = {}
        for id_, check_in_list in user_id_dict.items():
            position, tie_datum = self._get_user_position(check_in_list)
            user_id_to_position[id_] = position
            if tie_datum is not None:
                self.ties.append(tie_datum)

        print("Generating vertex set...")
        vertex_set = VertexSet(dimension=2, metric=EarthDistance())
        vertex_set.set_points_from_names(user_id_to_position)

        return vertex_set

    def _create_weights(self, vertices: VertexSet) -> List[float]:
        if vertices is None:
            raise RuntimeError(f"Vertex set has not been created yet.")

        print("Reading vertex degrees for weight initialisation...")
        names = set(vertices.names)
        degree_dict = {x: 0 for x in names}
        with open(self._EDGE_DATA_PATH, "rb") as file:
            # Each edge appears twice, once in each direction, so it's correct to only increment the first endpoint.
            for line in file:
                user_id_0, user_id_1 = self._parse_edge_line(line)
                # Not all nodes have position information encoded - these are omitted from the graph, so have no weight.
                if user_id_0 not in names or user_id_1 not in names:
                    continue
                degree_dict[user_id_0] += 1

        print("Generating weights...")
        weights = [float(degree_dict[vertices.id_to_name(i)]) for i in range(vertices.size)]
        return weights

    def _get_user_position(self, check_in_list: Sequence[CheckIn]) -> Tuple[Tuple[float, float], Optional[TieDatum]]:
        """Given a list of CheckIns for a user, discretise their positions to a (very) roughly 25kmx25km grid,
        choose their most common position on that grid, then return a random CheckIn position restricted to that grid
        square. NB this 25kmx25km grid is the same metric used in the original Gowalla paper, but also NB it's
        implemented as .25x.25 boxes in latitude/longitude, so the actual size of the cells will change substantially
        depending on the longitude."""
        if not check_in_list:
            raise ValueError("Check-in list is empty!")

        # Rounds x to the nearest .25
        def discretise(x):
            return round(4*x, 0)/4

        position_dict = defaultdict(list)
        for check_in in check_in_list:
            disc_positions = (discretise(check_in.latitude), discretise(check_in.longitude))
            position_dict[disc_positions].append((check_in.latitude, check_in.longitude))

        max_position_count = max([len(x) for x in position_dict.values()])
        modal_disc_positions = []
        for disc_position, mapped_positions in position_dict.items():
            if len(mapped_positions) == max_position_count:
                modal_disc_positions.append(disc_position)

        user_id = check_in_list[0].user_id
        if len(modal_disc_positions) == 1:
            tie_datum = None
        else:
            tie_datum = TieDatum(user_id=user_id, modal_positions=modal_disc_positions)
        disc_position_index = self.rng.integers(0, len(modal_disc_positions))
        disc_positions = modal_disc_positions[disc_position_index]

        position_index = self.rng.integers(0, len(position_dict[disc_positions]))
        return position_dict[disc_positions][position_index], tie_datum

    @staticmethod
    def _parse_edge_line(line: bytes) -> Tuple[int, int]:
        """Parses a single line of the Gowalla edge data file into a tuple of two user IDs, returned in the same order
        as in the file."""
        # Each line is in the format "[id_0]\t[id_1]\n", where the IDs are integers.
        id_strings = line.decode(encoding="utf-8").replace("\n", "").split("\t")
        return int(id_strings[0]), int(id_strings[1])

    def _create_edges(self, vertices: VertexSet) -> List[Tuple[int, int]]:
        """Reads the edge set for the Gowalla dataset, downloading it first if needed, and returns the resulting list
        of vertex ID pairs. Takes the vertex set as an argument since vertices without positional data should not be
        included, and calculating this is slow enough that we don't want to do it twice."""
        if vertices is None:
            raise RuntimeError(f"Vertex set has not been created yet.")

        if not self._snap_data_present():
            self._obtain_snap_data()

        print("Reading edge list...")
        edge_list = []
        vertex_names = set(vertices.names)  # Compute the set only once to save time.
        with open(self._EDGE_DATA_PATH, "rb") as file:
            for i, line in enumerate(file):
                if i % 100000 == 0 and 0 < i < 1900000:
                    print(f"Edge {i / 1000000}M/1.9M complete.")
                id_0, id_1 = self._parse_edge_line(line)
                if id_0 not in vertex_names or id_1 not in vertex_names:
                    continue
                # Each edge appears twice, once in each direction.
                if id_0 < id_1:
                    edge_list.append((vertices.name_to_id(id_0), vertices.name_to_id(id_1)))

        return edge_list


class GowallaInitialVertexFn(InitialVertexFunction):
    def __init__(self):
        self._function = lambda _, __: 164
        self.description = "User at (49.50, 11.44) near Nuremburg."


class SyntheticGowallaGraph(PresetGraph):
    @property
    def default_seed(self) -> int:
        return 276482048599935051615627698249100747258

    @property
    def description(self) -> str:
        return f"""Synthetic GIRG with Gowalla parameters: tau = {SYN_GOWALLA_TAU}, alpha = {SYN_GOWALLA_ALPHA}, 
            generated on the torus [0,{SYN_GOWALLA_SIZE})^2. Restricted to the giant component."""

    @property
    def graph_generator_class(self) -> Type[PresetGen]:
        return SyntheticGowallaGen


class SyntheticGowallaGen(PresetGen):
    @property
    def name(self) -> str:
        return "syn-gowalla"

    def create_data(self) -> None:
        print("Generating GIRG...")
        vertex_gen = PoissonPointProcess(dimension=2, size=SYN_GOWALLA_SIZE)
        weight_gen = PowerLawWeightGen(tau=SYN_GOWALLA_TAU, ell=IdentityWeightScaler())
        vertex_set = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=self.rng)
        edge_gen = GirgGen(alpha=SYN_GOWALLA_ALPHA, scale_factor=1 / SYN_GOWALLA_SIZE)
        edges = edge_gen(vertex_set, self.rng)

        original_data = GraphData(vertex_set=vertex_set.vertices, weights=vertex_set.weights, edge_list=edges)
        giant_data = extract_giant_data(original_data)
        self.vertices, self.weights, self.edge_list = giant_data.vertex_set, giant_data.weights, giant_data.edge_list


class GowallaGiantGraph(PresetGraph):
    @property
    def default_seed(self) -> int:
        return 243696392762333123792045834049537897896

    @property
    def description(self) -> str:
        return f"""Preset graph generated from the giant component of the Gowalla dataset with GowallaGiantGen."""

    @property
    def graph_generator_class(self) -> Type[PresetGen]:
        return GowallaGiantGen


class GowallaGiantGen(PresetGen):
    @property
    def name(self):
        return "gowalla-giant"

    def create_data(self) -> None:
        original_gowalla = GowallaGraph()
        giant_data = extract_giant_data(original_gowalla.graph_data)
        self.vertices, self.weights, self.edge_list = giant_data.vertex_set, giant_data.weights, giant_data.edge_list


def plot_tie_data(data: Sequence[TieDatum]):
    """INTERNAL USE: Displays a plot that roughly indicates the quality of our current tie-breaking approach."""
    distance_data = np.asarray([datum.max_distance for datum in data])
    bin_count = 50

    # Space bins evenly on a log scale.
    bins = np.logspace(np.log10(distance_data.min()), np.log10(distance_data.max()), bin_count)

    plt.figure()
    plt.hist(distance_data, bins=bins)  # type: ignore
    plt.xscale('log')
    plt.xlabel('Max change in distance due to tie')
    plt.ylabel('Number of ties')
    plt.title(f"{len(distance_data)} total ties")

    pct_50, pct_75, pct_90 = np.percentile(distance_data, [50, 75, 90])
    for pct, label in zip([pct_50, pct_75, pct_90], ['50th percentile', '75th percentile', '90th percentile']):
        plt.axvline(pct, linestyle='--')
        plt.text(pct, plt.ylim()[1] * 0.9, label, rotation=90, va='top', fontsize='small')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
