from collections import defaultdict
from enum import Enum
from math import floor
from pathlib import Path
from typing import DefaultDict, Tuple, Callable, Set, List
import config
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from HeatMaps import get_map_to_europe
from Region import Region, region_from_position
from Gowalla import GowallaGiantGraph
from SIEpidemic import SIEpidemic, FPPCostGen
from figures_common import GOWALLA_INITIAL

base_folder = config.FIGURE_FOLDER / "heatmaps"
if not base_folder.exists():
    base_folder.mkdir(parents=True)

def select_target_vertex():
    graph = GowallaGiantGraph()
    vertices = graph.graph_data.vertex_set
    vertex_ids = list(vertices.ids)

    target_vertex = np.random.choice(vertex_ids)
    while region_from_position(vertices.id_to_position(target_vertex)) != Region.EU:
        target_vertex = np.random.choice(vertex_ids)
    
    return target_vertex



def generate_infection_path_figure(path: Path):

    parameter_list = [{"mu": 0, "zeta": 0, "regime": "explosive"}, {"mu": 1, "zeta": 1, "regime": "quasi_exp"},
                       {"mu": 1, "zeta": 2, "regime": "polynomial"}, {"mu": 1, "zeta": 3, "regime": "geometric"}]

    target_vertex = select_target_vertex()

    plt.figure(figsize=(8, 8), dpi=1200)

    for i, p in enumerate(parameter_list):
        entropy = 281056517691655338767508099688269354802
        rng = np.random.default_rng(entropy)
        graph = GowallaGiantGraph()
        cost_gen = FPPCostGen(lambda_=1)

        epidemic = graph.create_epidemic(cost_gen=cost_gen, mu=p["mu"], zeta=p["zeta"], name="infectionPath", rng=rng)
        epidemic.run_infection_on_area(initial_vertex_id=epidemic.vertex_set.name_to_id(GOWALLA_INITIAL),region = Region.EU, rng=rng)

        infection_path = epidemic.infection_path(target_vertex)

        origin_lat, origin_long = epidemic.vertex_set.name_to_position(GOWALLA_INITIAL)
        target_lat, target_long = epidemic.vertex_set.id_to_position(target_vertex)

        projection_map = get_map_to_europe(centre_lat=origin_lat, centre_long=origin_long)


        path_coordinates = []
        for u in infection_path:
            lat_u, long_u = epidemic.vertex_set.id_to_position(u)
            path_coordinates.append(projection_map(lat_u,long_u))
        
        x,y = zip(*path_coordinates)

        x_min, y_min = (-1900000, -1500000)
        x_max, y_max = (2000000, 2500000)
        
        raw_projection = ccrs.AzimuthalEquidistant(central_latitude=origin_lat, central_longitude=origin_long)

        ax = plt.axes((.05, .05, .9, .9), projection=raw_projection)
        ax.set_extent([x_min, x_max, y_min, y_max], crs=raw_projection)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.6)
        ax.add_feature(cartopy.feature.BORDERS, linewidth=0.4)
        ax.plot(x, y,linestyle='--',marker='o',markersize=5,linewidth=0.8)
        centre_x, centre_y = projection_map(origin_lat, origin_long)
        target_x, target_y = projection_map(target_lat, target_long)
        ax.plot(centre_x, centre_y, marker='x', markersize=15, color='green')
        ax.plot(target_x, target_y, marker='x', markersize=15, color='orange')
        plt.savefig(path + "/infection_paths_" + p["regime"] + "_" + str(target_vertex) + ".png", dpi=1200)
    #plt.savefig(path.with_suffix(".eps"), dpi=1200, format="eps")



generate_infection_path_figure('/home/zylan/python/girg-simulations-refactor-heatmaps/images')





