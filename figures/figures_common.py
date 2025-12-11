from pathlib import Path
from math import ceil
from typing import List

from PIL import Image


# Initial vertex names for infections in the real and synthetic Gowalla graphs.
GOWALLA_INITIAL = 164
SYN_GOWALLA_INITIAL = 230563


def collate_curves(output_path: Path, figure_paths: List[Path]):
    """ Combines all figures together into a square grid with four plots per row. Scales gracefully if the number of
    plots changes. If plots are non-uniform in size, keeps the grid squares uniform in size and aligns each plot
    to the left and bottom of its square."""
    plot_count = len(figure_paths)

    image_dimensions = [Image.open(path).size for path in figure_paths]
    max_width = max([dim[0] for dim in image_dimensions])
    max_height = max([dim[1] for dim in image_dimensions])

    canvas = Image.new("RGB", (4 * max_width, ceil(plot_count / 4) * max_height), (255, 255, 255))

    for x, path in enumerate(figure_paths):
        row, column = x // 4, x % 4
        image = Image.open(path)
        _, height = image.size
        canvas.paste(image, (column * max_width, row * max_height + (max_height - height)))

    canvas.save(output_path)