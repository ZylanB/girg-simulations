from collections import defaultdict
from enum import Enum
from math import floor
from pathlib import Path
from typing import DefaultDict, Tuple, Callable, Set, List

import cartopy.crs as ccrs
import cartopy.feature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import config
from Gowalla import GowallaGiantGraph, SyntheticGowallaGraph
from Region import Region, region_from_position
from SIEpidemic import SIEpidemic, FPPCostGen
from VertexSet import TorusDistance
from figures.figures_common import collate_curves, GOWALLA_INITIAL, SYN_GOWALLA_INITIAL


def get_map_to_torus(centre_x: float, centre_y: float, side: float) \
        -> Callable[[float, float], Tuple[float, float]]:
    """Returns a projection mapping a point (x,y) into the torus of width and height side centered at
    (centre_x, centre_y)."""
    def projection(x, y):
        new_x = (x - centre_x + side/2) % side
        new_y = (y - centre_y + side/2) % side
        return new_x, new_y
    return projection


def get_map_to_europe(centre_lat: float, centre_long: float) -> Callable[[float, float], Tuple[float, float]]:
    """Returns a projection mapping a point (lat, long)  onto an azimuthal equidistant map with (x, y) coordinates.
    This projection preserves distances and angles from the initial point of infection, allowing for easy observation
    of spread."""
    plate = ccrs.PlateCarree()
    raw_projection = ccrs.AzimuthalEquidistant(central_latitude=centre_lat, central_longitude=centre_long)
    def projection(lat, long):
        """No, this is not a bug. Cartopy orders points by (longitude, latitude), matching coordinate geometry.
        The codebase (and haversine metric code) order points by (latitude, longitude), matching cartography.
        So we do in fact need to reorder the coordinates."""
        return raw_projection.transform_point(x=long, y=lat, src_crs=plate)
    return projection

class HeatMapMode(Enum):
    EUROPE = 1
    TORUS = 2


class EpidemicHeatMap:
    """This class turns data from 2d spatial epidemics into pretty heatmap images showing the spread of the
    infection. Order: initialise, then call load_data, then call generate_heatmap, then call export_to_canvas."""
    def __init__(self, x_pixels: int, y_pixels: int, mode: HeatMapMode, epidemic: SIEpidemic):
        """
        x_pixels and y_pixels control the resolution of the heatmap. NB if x_pixels/y_pixels is not equal to (x_max -
        x_min)/(y_max - y_min) then the pixels won't be square.

        If mode == EUROPE then the heatmap will be rendered on a map of Europe in a LAEA projection, reading the vertex
        co-ordinates as latitude-longitude pairs. Otherwise, mode == TORUS and the heatmap will be rendered on a blank
        square.
        """

        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.mode = mode
        self.epidemic = epidemic

        """self.projection_map will be applied to all positions before processing them. In TORUS mode it does nothing,
        in EUROPE mode it projects into azimuthal equidistant (which represents distances and angles accurately from
        the initial infection). (x_min, y_min) and (x_max, y_max) are the lower-left and upper-right corners of the 
        area to be rendered in this coordinate system."""
        self.projection_map: Callable[[float, float], Tuple[float, float]]

        if mode == HeatMapMode.EUROPE:
            origin_lat, origin_long = epidemic.vertex_set.name_to_position(GOWALLA_INITIAL)
            self.projection_map = get_map_to_europe(centre_lat=origin_lat, centre_long=origin_long)
            # These just need to be a box containing Europe that looks reasonable.
            self.x_min, self.y_min = (-1900000, -1500000)
            self.x_max, self.y_max = (2000000, 2500000)


        elif mode == HeatMapMode.TORUS:
            metric = epidemic.vertex_set.metric
            if type(metric) is not TorusDistance:
                raise Exception("This epidemic isn't on a torus, but torus mode was selected.")
            self.x_min = self.y_min = 0, metric.size
            self.x_max = self.y_max = 0, metric.size
            origin_x, origin_y = epidemic.vertex_set.name_to_position(SYN_GOWALLA_INITIAL)
            self.projection_map = get_map_to_torus(centre_x=origin_x, centre_y=origin_y, side=metric.size)

        else:
            raise Exception(f"Unsupported HeatMapMode {mode}.")

        """ 
        Let P_{i,j} be the i'th pixel from the left and the j'th pixel from the top of the heatmap, counting from 0.
        For a pixel P, let t(P) be the earliest infection time among all vertices in P. heatmap_mesh[j][i] 
        will contain the position of P_{i,j} in the list of all pixels when sorted under the key P |-> t(P). So the 
        first pixel to be infected has pixel order 0, the second has pixel order 1, and so on. Ties will be broken 
        arbitrarily, and pixels containing no vertices are set to -1. 
        """
        self.heatmap_mesh = self._get_heatmap_mesh()

    def _get_heatmap_mesh(self) -> List[List[int]]:
        """Initialises self.heatmap_mesh as specified in __init__."""

        vertices = self.epidemic.vertex_set.vertices

        if vertices.dimension != 2:
            raise Exception("Only two-dimensional graphs are supported.")

        # full_epidemic_data[(i,j)] will contain all infection times for vertices in the (i,j)'th pixel.
        # Pixels containing no vertices are omitted.
        full_epidemic_data: DefaultDict[Tuple[int, int], Set[float]] = defaultdict(set)
        pixel_width = (self.x_max - self.x_min) / self.x_pixels
        pixel_height = (self.y_max - self.y_min) / self.y_pixels

        for id_ in vertices.ids:
            position = vertices.id_to_position(id_)
            europe_valid = self.mode == HeatMapMode.EUROPE and region_from_position(position) == Region.EU
            torus_valid = self.mode == HeatMapMode.TORUS
            if europe_valid or torus_valid:
                x, y = self.projection_map(position[0], position[1])
                i = floor((x - self.x_min) / pixel_width)
                j = floor((y - self.y_min) / pixel_height)
                if i >= self.x_pixels or j >= self.y_pixels:
                    raise Exception("Position out of bounds, chosen projection doesn't display all vertices in Europe")
                infection_time = self.epidemic.infection_times[id_]
                full_epidemic_data[(i, j)].add(infection_time)

        # representative_data is a list of (i,j,time) tuples, where time is the earliest infection time of any vertex
        # in the (i,j)'th pixel, sorted by time.
        representative_data: List[Tuple[int, int, float]] = []
        for pixel, times in full_epidemic_data.items():
            representative_data.append((pixel[0], pixel[1], min(times)))
        representative_data.sort(key=lambda d: d[2])

        # We now read this back into self.heatmap_mesh to set it as specified, first initialising every value to -1.
        heatmap_mesh = [[-1] * self.x_pixels for _ in range(self.y_pixels)]
        for index, (i, j, _) in enumerate(representative_data):
            """Again, not a bug. Matplotlib wants heatmaps to use matrix notation, so the first coordinate is the
            row and the second coordinate is the column. So we have to reverse coordinates *again*."""
            heatmap_mesh[j][i] = index
        return heatmap_mesh

    def export_to_canvas(self, save_path: Path):
        # NB The old figures used the jet colormap, which is deprecated these days because human eyes are weird.
        cmap = mpl.colormaps['plasma_r']  # First infection = 0 = yellow -> purple -> blue = 1 = last infection
        cmap.set_bad(alpha=0)  # Don't draw colours for masked values
        np_mesh = np.asarray(self.heatmap_mesh)
        masked_heatmap = np.ma.masked_array(np_mesh, mask=(np_mesh == -1))  # Mask out pixels with no data

        plt.figure(figsize=(8, 8), dpi=1200)
        x_mesh_points = np.linspace(self.x_min, self.x_max, self.x_pixels + 1)
        y_mesh_points = np.linspace(self.y_min, self.y_max, self.y_pixels + 1)

        if self.mode == HeatMapMode.TORUS:
            ax = plt.axes((.05, .05, .9, .9))
            ax.pcolormesh(x_mesh_points, y_mesh_points, masked_heatmap, cmap=cmap)
            ax.set_axis_off()
        elif self.mode == HeatMapMode.EUROPE:
            # Draw map of Europe in background
            origin_lat, origin_long = self.epidemic.vertex_set.name_to_position(GOWALLA_INITIAL)
            raw_projection = ccrs.AzimuthalEquidistant(central_latitude=origin_lat, central_longitude=origin_long)
            ax = plt.axes((.05, .05, .9, .9), projection=raw_projection)
            ax.set_extent([self.x_min, self.x_max, self.y_min, self.y_max], crs=raw_projection)
            ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.6)
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.4)
            ax.pcolormesh(x_mesh_points, y_mesh_points, masked_heatmap, cmap=cmap)
            centre_x, centre_y = self.projection_map(origin_lat, origin_long)
            ax.plot(centre_x, centre_y, marker='x', markersize=15, color='green')
        else:
            raise Exception(f"Unsupported HeatMapMode {self.mode}.")

        plt.savefig(save_path, dpi=1200)
        plt.savefig(save_path.with_suffix(".eps"), dpi=1200, format="eps")


def generate_real_plot(mu: float, zeta: float, path: Path, rng: np.random.Generator):
    graph = GowallaGiantGraph()
    cost_gen = FPPCostGen(lambda_=1)
    epidemic = graph.create_epidemic(cost_gen=cost_gen, mu=mu, zeta=zeta, name=path.name, rng=rng)
    epidemic.run_infection(initial_vertex_id=epidemic.vertex_set.name_to_id(GOWALLA_INITIAL))

    heatmap = EpidemicHeatMap(x_pixels=460, y_pixels=370, epidemic=epidemic, mode=HeatMapMode.EUROPE)
    heatmap.export_to_canvas(path)


def generate_syn_plot(mu: float, zeta: float, path: Path, rng: np.random.Generator):
    graph = SyntheticGowallaGraph()
    cost_gen = FPPCostGen(lambda_=1)
    epidemic = graph.create_epidemic(cost_gen=cost_gen, mu=mu, zeta=zeta, name=path.name, rng=rng)
    epidemic.run_infection(initial_vertex_id=SYN_GOWALLA_INITIAL)

    heatmap = EpidemicHeatMap(x_pixels=400, y_pixels=400, epidemic=epidemic, mode=HeatMapMode.TORUS)
    heatmap.export_to_canvas(path)


def generate_figures():
    entropy = 281056517691655338767508099688269354802
    rng = np.random.default_rng(entropy)

    parameter_list = [{"mu": 0, "zeta": 0}, {"mu": 1, "zeta": 1}, {"mu": 1, "zeta": 2}, {"mu": 1, "zeta": 3}]

    base_folder = config.FIGURE_FOLDER / "heatmaps"
    if not base_folder.exists():
        base_folder.mkdir(parents=True)

    for i, p in enumerate(parameter_list):
        generate_real_plot(mu=p["mu"], zeta=p["zeta"], path=base_folder / f"real_heatmap_{i}.png", rng=rng)
        generate_syn_plot(mu=p["mu"], zeta=p["zeta"], path=base_folder / f"syn_heatmap_{i}.png", rng=rng)

    real_paths = [base_folder / f"real_heatmap_{i}.png" for i in range(4)]
    syn_paths = [base_folder / f"syn_heatmap_{i}.png" for i in range(4)]
    collate_curves(output_path=base_folder / "combined_heatmaps.png", figure_paths=syn_paths + real_paths)

if __name__ == "__main__":
    generate_figures()
