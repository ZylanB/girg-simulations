from enum import Enum
from typing import Tuple

import cartopy.io.shapereader as shapereader
from shapely.geometry import Point, shape
from shapely.ops import unary_union


class Region(Enum):
    """Geographic regions for use with real-world datasets like Gowalla."""
    US = 1
    EU = 2
    OTHER = 3

shape_path = shapereader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
reader = shapereader.Reader(shape_path)
europe_geoms = [shape(record.geometry) for record in reader.records() if
                record.attributes["CONTINENT"] == "Europe" and record.attributes["NAME"] != "Russia"]
europe_geom = unary_union(europe_geoms).simplify(tolerance=0.1)
us_record = [record for record in reader.records() if record.attributes["NAME"] == "United States of America"][0]
us_geom = shape(us_record.geometry).simplify(tolerance=0.1)

def region_from_position(position: Tuple[float, float]) -> Region:
    """Returns the region for the given latitude/longitude pair."""
    # Check position is a valid latitude/longitude pair.
    if not (-90 <= position[0] <= 90) or not (-180 <= position[1] <= 180):
        return Region.OTHER

    # Shapely/cartopy stores coordinates the other way round to SNAP, with longitude first.
    point = Point(position[1], position[0])
    if europe_geom.contains(point):
        return Region.EU
    if us_geom.contains(point):
        return Region.US
    return Region.OTHER