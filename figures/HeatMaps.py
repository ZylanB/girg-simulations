"""
PLAN:
1. Run an infection on real/synthetic Gowalla with given parameters.
2. For real Gowalla, throw away vertices outside the desired plot area.
3. Iterate over vertices to divide pixels into equal-size buckets by region.
    - Synthetic: Buckets are equal-side squares in the torus.
    - Real: Buckets are equal-area regions in Europe, which can't be squares because projection. (The old code uses
      projection.) Easiest way seems to be using cartopy: transform the latitude/longitude points into Lambert
      azimuthal equal-area projection (which preserves area but not angles) using cartopy.crs.CRS.transform_points
      with a cartopy.crs.PlateCarree as the first argument. This gives you x/y pairs on the area-respecting projection,
      so then we can bin as normal.
   Both synthetic and real ultimately use buckets that are equal-size squares on a 2d plane, so that part of the code
   should be common - the real Gowalla code should just transform the coordinates first.
4. Process the info from #1 and #2 into colours for a heatmap by picking a representative infection time, ordering
   the buckets according to that representative, then going blue -> green -> red. Representative should be configurable,
   probably we want first infection but might as well check median/average too. Old code uses matplotlib.colors for
   this, LinearSegmentedColormap, results look good so let's do the same.
5. Project the heatmap onto a map of Europe for real and a square for synthetic. Old code uses a random map for this,
   cartopy looks easier and less copyright infringement-y (and allows for accurate pixels).

Parameters should be: mu=zeta=0, mu=zeta=1, mu=1 and zeta=2, mu=1 and zeta=3. Arrange plots in a 4x2 rectangle
with synthetic on top, this can be done as in EpidemicCurves.py.
"""
from collections import defaultdict
from enum import Enum
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

# TODO move these into config or Gowalla.py
from EpidemicCurvesConfig import GOWALLA_INITIAL, SYN_GOWALLA_INITIAL


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

        # self.projection_map will be applied to all positions before processing them. In TORUS mode it does nothing,
        # in EUROPE mode it projects into LAEA (which represents areas accurately but not angles). (x_min, y_min)
        # and (x_max, y_max) are the lower-left and upper-right corners of the area to be rendered in this coordinate
        # system.
        self.projection_map: Callable[[Tuple[float, float]], Tuple[float, float]]

        if mode == HeatMapMode.EUROPE:
            # These just need to be a box containing Europe that looks reasonable.
            self.x_min = -12
            self.y_min = 34
            self.x_max = 35
            self.y_max = 72
            self.raw_projection = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)
            plate = ccrs.PlateCarree()
            self.projection_map = lambda x, y: self.raw_projection.transform_point(x=x, y=y, src_crs=plate)

        elif mode == HeatMapMode.TORUS:
            metric = epidemic.vertex_set.metric
            if metric is not TorusDistance:
                raise Exception("This epidemic isn't on a torus, but torus mode was selected.")
            self.x_min = self.y_min = 0
            self.x_max = self.y_max = metric.size
            self.projection_map = lambda x, y: x

        else:
            raise Exception(f"Unsupported HeatMapMode {mode}.")

        """ 
        Let P_{i,j} be the i'th pixel from the left and the j'th pixel from the top of the heatmap, counting from 0.
        For a pixel P, let t(P) be the earliest infection time among all vertices in P. heatmap_mesh[(i,j)] 
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

        for position in vertices.positions:
            europe_valid = self.mode == HeatMapMode.EUROPE and region_from_position(position) == Region.EU
            torus_valid = self.mode == HeatMapMode.TORUS
            if europe_valid or torus_valid:
                x, y = self.projection_map(position[0], position[1])
                i = (x - self.x_min) // pixel_width
                j = (y - self.y_min) // pixel_height
                infection_time = self.epidemic.infection_times[vertices.id_from_position(position)]
                full_epidemic_data[(i, j)].add(infection_time)

        # representative_data is a list of (i,j,time) tuples, where time is the earliest infection time of any vertex
        # in the (i,j)'th pixel, sorted by time.
        representative_data: List[Tuple[int, int, float]] = []
        for pixel, times in full_epidemic_data.items():
            representative_data.append((pixel[0], pixel[1], min(times)))
        representative_data.sort(key=lambda d: d[2])

        # We now read this back into self.heatmap_mesh to set it as specified, first initialising every value to -1.
        heatmap_mesh = [[-1] * self.y_pixels for _ in range(self.x_pixels)]
        for i, j, _, index in enumerate(representative_data):
            heatmap_mesh[i][j] = index
        return heatmap_mesh

    def export_to_canvas(self, save_path: Path):
        # NB The old figures used the jet colormap, which is deprecated these days because human eyes are weird.
        cmap = mpl.colormaps['plasma']  # First infection = 0 = yellow -> purple -> blue = 1 = last infection
        cmap.set_bad(alpha=0)  # Don't draw colours for masked values
        masked_heatmap = np.ma.masked_array(self.heatmap_mesh, mask=-1)  # Mask out pixels with no data

        fig = plt.figure(figsize=(8, 8))
        if self.mode == HeatMapMode.TORUS:
            ax = plt.axes()
        elif self.mode == HeatMapMode.EUROPE:
            # Draw map of Europe in background
            ax = plt.axes(projection=self.raw_projection)
            ax.set_extent([self.x_min, self.x_max, self.y_min, self.y_max], crs=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.8)
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
        else:
            raise Exception(f"Unsupported HeatMapMode {self.mode}.")

        ax.pcolormesh(self.heatmap_mesh, cmap=cmap)
        plt.savefig(save_path)


# TODO Plots need to be square and of equal size
def generate_real_plot(mu: float, zeta: float, path: Path, rng: np.random.Generator):
    graph = GowallaGiantGraph()
    cost_gen = FPPCostGen(lambda_=1)
    epidemic = graph.create_epidemic(cost_gen=cost_gen, mu=mu, zeta=zeta, name=path.name, rng=rng)
    epidemic.run_infection(initial_vertex_id=GOWALLA_INITIAL)

    # TODO Remove non-Europe vertices before getting heatmap.

    heatmap = EpidemicHeatMap(x_pixels=460, y_pixels=370, epidemic=epidemic, mode=HeatMapMode.EUROPE)
    heatmap.export_to_canvas(path)


def generate_syn_plot(mu: float, zeta: float, path: Path, rng: np.random.Generator):
    graph = SyntheticGowallaGraph()
    cost_gen = FPPCostGen(lambda_=1)
    epidemic = graph.create_epidemic(cost_gen=cost_gen, mu=mu, zeta=zeta, name=path.name, rng=rng)
    epidemic.run_infection(initial_vertex_id=SYN_GOWALLA_INITIAL)

    heatmap = EpidemicHeatMap(x_pixels=500, y_pixels=500, epidemic=epidemic, mode=HeatMapMode.TORUS)
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

    # TODO Move collate_images out of EpidemicCurves into a common library then use it to bung these into a 4x2 grid.