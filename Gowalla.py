from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import datetime
import requests
import gzip
from collections import defaultdict
from pathlib import Path
import shutil
import dill  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from SIEpidemic import SIEpidemic, EdgeGenerator, FixedGraphGenerator, EdgeCostGenerator
from WeightedVertexSet import WeightedVertexSet, create_from_degrees_generator
from VertexSet import FixedVertexSet, EarthDistance


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


class GowallaDataReader:
    """Creates the Gowalla dataset from scratch, downloading the relevant files from _VERTEX_DATA_URL and
    _EDGE_DATA_URL and storing them at _VERTEX_DATA_PATH and _EDGE_DATA_PATH if need be. vertex_path is the path to
    save pickled vertex data to, and edge_path is the path to save pickled edge data to."""
    _VERTEX_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    _EDGE_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
    _VERTEX_DATA_PATH = Path.cwd() / "Gowalla_totalCheckins.txt"
    _EDGE_DATA_PATH = Path.cwd() / "Gowalla_edges.txt"

    def __init__(self, vertex_path: Optional[Path] = None, edge_path: Optional[Path] = None,
                 generator: Optional[np.random.Generator] = None):
        self.vertex_path = vertex_path
        self.edge_path = edge_path
        self.generator = np.random.default_rng() if generator is None else generator

        if not self._snap_data_present():
            self._obtain_snap_data()
        self.vertices, self.tie_data = self._create_vertices()
        self.edge_list = self._create_edges(self.vertices)

    def save_files(self):
        """Saves the recreated Gowalla data in pickled form to self.vertex_path and self.edge_path."""
        if not (self.vertex_path and self.edge_path):
            raise RuntimeError("Attempting to save data to an empty path.")

        print("Saving weighted vertex set...")
        with open(self.vertex_path, "wb") as file:
            dill.dump(self.vertices, file)

        print("Saving edge list...")
        with open(self.edge_path, "wb") as file:
            dill.dump(self.edge_list, file)

    @classmethod
    def _obtain_snap_data(cls):
        """Downloads any of the Gowalla data not already present locally."""
        if not cls._VERTEX_DATA_PATH.exists():
            cls._obtain_data_file(url=cls._VERTEX_DATA_URL, dest_path=cls._VERTEX_DATA_PATH)
        if not cls._EDGE_DATA_PATH.exists():
            cls._obtain_data_file(url=cls._EDGE_DATA_URL, dest_path=cls._EDGE_DATA_PATH)

    @staticmethod
    def _obtain_data_file(url: str, dest_path: Path):
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

    @classmethod
    def _snap_data_present(cls):
        """Returns true if all the raw Gowalla data has already been downloaded from the SNAP site."""
        return cls._VERTEX_DATA_PATH.exists() and cls._EDGE_DATA_PATH.exists()

    def _create_vertices(self) -> Tuple[WeightedVertexSet, List[TieDatum]]:
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
        ids_to_positions = {}
        ties = []
        for id_, check_in_list in user_id_dict.items():
            position, tie_datum = self._get_user_position(check_in_list)
            ids_to_positions[id_] = position
            if tie_datum is not None:
                ties.append(tie_datum)

        print("Reading vertex degrees for weight initialisation...")
        degree_dict = {x: 0 for x in ids_to_positions.keys()}
        with open(self._EDGE_DATA_PATH, "rb") as file:
            # Each edge appears twice, once in each direction, so it's correct to only increment the first endpoint.
            # Note that not all nodes have position information encoded.
            for line in file:
                id_0, id_1 = self._parse_edge_line(line)
                if id_0 not in degree_dict or id_1 not in degree_dict:
                    continue
                degree_dict[id_0] += 1

        print("Generating weighted vertex set...")
        vertex_generator = FixedVertexSet(metric=EarthDistance(), points=ids_to_positions,
                                          description="Gowalla dataset")
        weight_generator = create_from_degrees_generator(degree_dict, description="Gowalla dataset")
        weighted_vertices = WeightedVertexSet(vertex_generator=vertex_generator, weight_generator=weight_generator)

        return weighted_vertices, ties

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
        disc_position_index = self.generator.integers(0, len(modal_disc_positions))
        disc_positions = modal_disc_positions[disc_position_index]

        position_index = self.generator.integers(0, len(position_dict[disc_positions]))
        return position_dict[disc_positions][position_index], tie_datum

    @staticmethod
    def _parse_edge_line(line: bytes) -> Tuple[int, int]:
        """Parses a single line of the Gowalla edge data file into a tuple of two user IDs, returned in the same order
        as in the file."""
        # Each line is in the format "[id_0]\t[id_1]\n", where the IDs are integers.
        id_strings = line.decode(encoding="utf-8").replace("\n", "").split("\t")
        return int(id_strings[0]), int(id_strings[1])

    @classmethod
    def _create_edges(cls, vertices: WeightedVertexSet) -> List[List[int]]:
        """Reads the edge set for the Gowalla dataset, downloading it first if needed, and returns the resulting list
        of vertex ID pairs. Takes the vertex set as an argument since vertices without positional data should not be
        included, and calculating this is slow enough that we don't want to do it twice."""
        if not cls._snap_data_present():
            cls._obtain_snap_data()

        print("Reading edge list...")
        edge_list = []
        vertex_names = vertices.names  # Compute the set only once to save time.
        with open(cls._EDGE_DATA_PATH, "rb") as file:
            for i, line in enumerate(file):
                if i % 100000 == 0 and 0 < i < 1900000:
                    print(f"Edge {i / 1000000}M/1.9M complete.")
                id_0, id_1 = cls._parse_edge_line(line)
                if id_0 not in vertex_names or id_1 not in vertex_names:
                    continue
                # Each edge appears twice, once in each direction.
                if id_0 < id_1:
                    edge_list.append([vertices.name_to_id(id_0), vertices.name_to_id(id_1)])

        return edge_list


class GowallaSIEpidemic(SIEpidemic):
    """An SIEpidemic on the Gowalla graph. Creates the Gowalla data from scratch using a GowallaDataReader if needed,
    otherwise just unpickles it."""
    _SAVED_VERTEX_PATH = Path.cwd() / "gowalla_vertices.pickle"
    _SAVED_EDGE_PATH = Path.cwd() / "gowalla_edges.pickle"

    def __init__(self, edge_cost_generator: EdgeCostGenerator, mu: float, zeta: float, name: str,
                 generator: Optional[np.random.Generator] = None):
        if not self._saved_graph_present():
            print("Gowalla data not present. Recreating...")
            data_reader = GowallaDataReader(vertex_path=self._SAVED_VERTEX_PATH, edge_path=self._SAVED_EDGE_PATH)
            data_reader.save_files()

        vertex_set = self._load_vertices(mu, zeta)
        edge_generator = self._load_edges()
        super().__init__(vertex_set=vertex_set, edge_cost_generator=edge_cost_generator, edge_generator=edge_generator,
                         mu=mu, zeta=zeta, generator=generator, name=name)

    @classmethod
    def _saved_graph_present(cls):
        """Returns true if the WeightedVertexSet and edge list of the Gowalla graph have already been saved locally."""
        return cls._SAVED_VERTEX_PATH.exists() and cls._SAVED_EDGE_PATH.exists()

    def _load_vertices(self, mu: float, zeta: float) -> WeightedVertexSet:
        """Reads the vertex set for the Gowalla dataset from VERTEX_DATA_PATH and returns a WeightedVertexSet with
        the appropriate penalties mu and zeta."""
        print("Loading vertex set...")
        with open(self._SAVED_VERTEX_PATH, "rb") as file:
            return_set = dill.load(file)

        return_set.mu = mu
        return_set.zeta = zeta
        return return_set

    def _load_edges(self) -> EdgeGenerator:
        """Reads the edge set for the Gowalla dataset from EDGE_DATA_PATH and returns an edge generator that can be
        passed into an SIEpidemic."""
        print("Loading edge generator...")
        with open(self._SAVED_EDGE_PATH, "rb") as file:
            edges = dill.load(file)

        return FixedGraphGenerator(edges, description="Interactions from the Gowalla dataset")


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
