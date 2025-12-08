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

import config


class PixelBins:
    # TODO This is the bit that gets unit-tested.
    def __init__(self, x_min, x_max, x_bins, y_min, y_max, y_bins, representative_picker):
        pass

    def add_to_bin(self, position, datum):
        pass

    def load_bins(self, data):
        pass

    def generate_heatmap(self):
        # TODO Replace a list of data in each bin with a single datum using representative_picker (which is e.g.
        #  min or median).
        # TODO Map this onto colour scales using matplotlib.colors.LinearSegmentedColorMap.
        pass

    def export_to_canvas(self):
        # TODO saves heatmap on blank canvas with matplotlib with optional map background
        pass


def generate_real_plot(mu, zeta, pixels_per_side, path):
    # TODO Load Gowalla and run an SIEpidemic with mu and zeta (inline).
    # TODO Remove non-Europe vertices (inline).
    # TODO Instantiate a PixelBins with coordinates matching Europe (inline).
    # TODO Load in data using PixelBins.load_bins, then generate the heatmap using PixelBins.generate_heatmap (inline).
    # TODO Render the heatmap on a LAEA map of Europe using PixelBins.export_to_canvas and carto_py.
    pass


def generate_syn_plot(mu, zeta, pixels, path):
    # TODO Load SynGowalla and run an SIEpidemic with mu and zeta (inline).
    # TODO Instantiate a PixelBins with a simple torus (inline).
    # TODO Load in data using PixelBins.load_bins, then generate the heatmap using PixelBins.generate_heatmap (inline).
    # TODO Render the heatmap on a blank canvas using PixelBins.export_to_canvas.
    pass


# Probably the above two functions only generate the PixelBins to facilitate full unit testing with a toy epidemic.
# Also probably PixelBins gets renamed to HeatMapCreator or similar.


def generate_figures():
    parameter_list = [{"mu": 0, "zeta": 0}, {"mu": 1, "zeta": 1}, {"mu": 1, "zeta": 2}, {"mu": 1, "zeta": 3}]

    base_folder = config.FIGURE_FOLDER / "heatmaps"
    if not base_folder.exists():
        base_folder.mkdir(parents=True)

    for i, p in enumerate(parameter_list):
        generate_real_plot(mu=p["mu"], zeta=p["zeta"], path=base_folder / f"real_heatmap_{i}.png")
        generate_syn_plot(mu=p["mu"], zeta=p["zeta"], path=base_folder / f"syn_heatmap_{i}.png")

    # TODO Move collate_images out of EpidemicCurves into a common library then use it to bung these into a 4x2 grid.