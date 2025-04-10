from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import datetime
import requests
import tarfile
from collections import defaultdict
from pathlib import Path
import dill

import numpy as np
from SIEpidemic import SIEpidemic, fixed_graph_generator
from WeightedVertexSet import WeightedVertexSet, from_degrees_generator
from VertexSet import VertexSet, earth_distance


@dataclass(slots=True)
class CheckIn:
    user_id: int
    time: datetime.datetime
    latitude: float
    longitude: float
    location_id: int

    @staticmethod
    def from_line(line: str) -> CheckIn:
        """Given a line of the Gowalla dataset, returns a corresponding CheckIn object."""
        sections = line.replace("\n", "").split("\t")
        time_format = "%Y-%m-%dT%H:%M:%SZ"
        return CheckIn(user_id=int(sections[0]), time=datetime.datetime.strptime(sections[1], time_format),
                       latitude=float(sections[2]), longitude=float(sections[3]), location_id=int(sections[4]))


@dataclass(slots=True)
class User:
    user_id: int
    position: Tuple[float, float]


class GowallaSIEpidemic(SIEpidemic):
    _VERTEX_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    _EDGE_DATA_URL = f"https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
    _VERTEX_DATA_PATH = Path.cwd() / "Gowalla_totalCheckins.txt"
    _EDGE_DATA_PATH = Path.cwd() / "Gowalla_edges.txt"
    _SAVED_VERTEX_PATH = Path.cwd() / "gowalla_vertices.pickle"
    _SAVED_EDGE_PATH = Path.cwd() / "gowalla_edges.pickle"

    def __init__(self, edge_cost_generator: Callable[[], float], mu: float, zeta: float, recreate: bool = False,
                 generator: Optional[np.random.Generator] = None):
        if recreate:
            self._obtain_gowalla_data()
            self._create_gowalla_vertices(generator)
            self._create_gowalla_edges()

        vertex_set = self._load_gowalla_vertices(mu, zeta)
        edge_generator = self._load_gowalla_edges()
        super().__init__(vertex_set=vertex_set, edge_cost_generator=edge_cost_generator, edge_generator=edge_generator)

    @staticmethod
    def _obtain_data_file(url: str, dest_path: Path):
        """Downloads the given .gz (not .tar.gz) file from the given URL, untars it in the given folder, and deletes
        the original .gz file."""
        dest_path.mkdir(parents=True, exist_ok=True)

        local_file = dest_path / Path(url).name
        print(f"Downloading from {url} to {local_file}...")

        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with local_file.open(mode="wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Download completed.")

        print("Extracting archive...")
        with tarfile.open(local_file, "r:gz") as tar:
            tar.extractall(path=dest_path)
        print("Extraction completed.")

        print(f"Deleting temporary file {local_file}...")
        local_file.unlink()
        print("Temporary file deleted.")

    @classmethod
    def _obtain_gowalla_data(cls):
        """Downloads any of the Gowalla data not already present locally."""
        if not cls._VERTEX_DATA_PATH.exists():
            cls._obtain_data_file(url=cls._VERTEX_DATA_URL, dest_path=cls._VERTEX_DATA_PATH.parent)
        if not cls._EDGE_DATA_PATH.exists():
            cls._obtain_data_file(url=cls._EDGE_DATA_URL, dest_path=cls._EDGE_DATA_PATH.parent)

    @classmethod
    def _create_gowalla_graph(cls):
        cls._create_gowalla_vertices()
        cls._create_gowalla_edges()

    @classmethod
    def _gowalla_data_present(cls):
        return cls._VERTEX_DATA_PATH.exists() and cls._EDGE_DATA_PATH.exists()

    @classmethod
    def _gowalla_graph_present(cls):
        return cls._SAVED_VERTEX_PATH.exists() and cls._SAVED_EDGE_PATH.exists()

    @staticmethod
    def _get_user_position(check_in_list: List[CheckIn], generator: Optional[np.random.Generator] = None) \
            -> Tuple[float, float]:
        """Given a list of CheckIns for a user, return their modal position after rounding latitude and longitude to
        two decimal places, roughly a half-mile radius. In the event of a tie, choose randomly (using the generator
        provided)."""
        if generator is None:
            generator = np.random.default_rng()

        if not check_in_list:
            raise ValueError("Check-in list is empty!")

        position_counts = defaultdict(int)
        for check_in in check_in_list:
            discretised_position = (round(check_in.latitude, 2), round(check_in.longitude, 2))
            position_counts[discretised_position] += 1

        max_position_count = max(position_counts, key=position_counts.get)
        modal_positions = [pos for pos, count in position_counts.items() if count == max_position_count]
        return modal_positions[generator.integers(0, len(modal_positions))]

    @staticmethod
    def _parse_edge_line(line: bytes) -> Tuple[int, int]:
        """Parses a single line of the Gowalla edge data file into a tuple of two user IDs, returned in the same order
        as in the file."""
        # Each line is in the format "[id_0]\t[id_1]\n", where the IDs are integers.
        id_strings = line.decode(encoding="utf-8").replace("\n", "").split("\t")
        return int(id_strings[0]), int(id_strings[1])

    @classmethod
    def _create_gowalla_vertices(cls, generator: Optional[np.random.Generator] = None):
        """Pulls the vertex set for the Gowalla dataset from loc-gowalla-totalCheckins.txt at the specified path,
        available for download at https://snap.stanford.edu/data/loc-gowalla.html, adds weights, and pickles it to
        GOWALLA_VERTEX_PATH. Mu and zeta are both initialised to 0."""
        if not cls._gowalla_data_present():
            cls._obtain_gowalla_data()

        print("Reading check-ins...")
        user_id_dict = defaultdict(list)
        with open(cls._VERTEX_DATA_PATH, "rb") as file:
            for line in file:
                new_check_in = CheckIn.from_line(str(line))
                user_id_dict[new_check_in.user_id].append(new_check_in)

        print("Mapping user IDs to positions...")
        ids_to_positions = {}
        for id_, check_in_list in user_id_dict.items():
            position = cls._get_user_position(check_in_list, generator)
            ids_to_positions[id_] = position

        print("Reading vertex degrees for weight initialisation...")
        degree_dict = defaultdict(int)
        with open(cls._EDGE_DATA_PATH, "rb") as file:
            # Each edge appears twice, once in each direction, so it's correct to only increment the first endpoint.
            for line in file:
                id_0, id_1 = cls._parse_edge_line(line)
                degree_dict[id_0] += 1

        print("Generating weighted vertex set...")
        unweighted_vertices = VertexSet(dimension=2, metric=earth_distance)
        unweighted_vertices.set_points_from_names(ids_to_positions)
        weight_generator = from_degrees_generator(degree_dict)
        weighted_vertices = WeightedVertexSet(vertices=unweighted_vertices, weight_generator=weight_generator,
                                              mu=0., zeta=0.)

        print("Saving weighted vertex set...")
        with open(cls._SAVED_VERTEX_PATH, "wb") as file:
            dill.dump(weighted_vertices, file)

    @classmethod
    def _load_gowalla_vertices(cls, mu: float, zeta: float) -> WeightedVertexSet:
        """Reads the vertex set for the Gowalla dataset from VERTEX_DATA_PATH, creating it first if it doesn't
        exist, and returns a WeightedVertexSet with the appropriate penalties mu and zeta."""
        if not cls._gowalla_graph_present():
            cls._create_gowalla_graph()

        print("Loading vertex set...")
        with open(cls._VERTEX_DATA_PATH, "rb") as file:
            return_set = dill.load(file)

        return_set.mu = mu
        return_set.zeta = zeta
        return return_set

    @classmethod
    def _create_gowalla_edges(cls):
        """Reads the edge set for the Gowalla dataset from GOWALLA_EDGE_PATH and pickles it to GOWALLA_EDGE_PATH."""
        if not cls._gowalla_data_present():
            cls._obtain_gowalla_data()

        print("Reading edge list...")
        edge_list = []
        with open(cls._EDGE_DATA_PATH, "rb") as file:
            for line in file:
                id_0, id_1 = cls._parse_edge_line(line)
                # Each edge appears twice, once in each direction. TODO add a unit test for this.
                if id_0 < id_1:
                    edge_list.append([id_0, id_1])

        print("Saving edge list...")
        with open(cls._EDGE_DATA_PATH, "rb") as file:
            dill.dump(edge_list, file)

    @classmethod
    def _load_gowalla_edges(cls) -> Callable[[WeightedVertexSet], List[Tuple[int, int]]]:
        """Reads the edge set for the Gowalla dataset from EDGE_DATA_PATH, creating it first if it doesn't exist,
        and returns an edge generator that can be passed into an SIEpidemic."""
        if not cls._gowalla_graph_present():
            cls._create_gowalla_graph()

        print("Loading edge generator...")
        with open(cls._EDGE_DATA_PATH, "rb") as file:
            edges = dill.load(file)

        return fixed_graph_generator(edges)
