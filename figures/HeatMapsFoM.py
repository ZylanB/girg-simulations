from pathlib import Path

import numpy as np

import config
from HeatMaps import HeatMapMode, EpidemicHeatMap
from SIEpidemic import GirgGen, FPPCostGen, SIEpidemic
from VertexSet import PoissonPointProcess
from WeightedVertexSet import PowerLawWeightGen, IdentityWeightScaler, WeightedVertexSet


def generate_plot(mu: float, path: Path):
    entropy = 105789816241624634282915676311936418383
    rng = np.random.default_rng(seed=entropy)

    print(f"Plotting mu={mu}.")
    print("Generating vertex set...")
    vertex_gen = PoissonPointProcess(dimension=2, size=750)
    weight_gen = PowerLawWeightGen(tau=2.3, ell=IdentityWeightScaler())
    vertex_set = WeightedVertexSet(vertex_gen=vertex_gen, weight_gen=weight_gen, rng=rng)

    print("Generating GIRG...")
    edge_gen = GirgGen(alpha=5, scale_factor=1/750)
    cost_gen = FPPCostGen(lambda_=1)
    epidemic = SIEpidemic(vertex_set=vertex_set, edge_gen=edge_gen, cost_gen=cost_gen, mu=mu, zeta=0,
                          name="FoM-synthetic", rng=rng)

    epidemic.run_infection(initial_vertex_id=0)

    print("Generating heatmap...")
    heatmap = EpidemicHeatMap(x_pixels=300, y_pixels=300, epidemic=epidemic, mode=HeatMapMode.TORUS)
    print("Saving heatmap...")
    heatmap.export_to_canvas(path)
    print("Done!")

def generate_figures():
    base_folder = config.FIGURE_FOLDER / "fom_heatmaps"
    if not base_folder.exists():
        base_folder.mkdir(parents=True)

    for i, mu in enumerate([0, 0.5, 1, 2]):
        generate_plot(mu=mu, path=base_folder / f"heatmap_{i}.png")

if __name__ == "__main__":
    generate_figures()