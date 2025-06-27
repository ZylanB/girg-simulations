from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type

from Gowalla import GowallaGiantGraph, SyntheticGowallaGraph
from PresetGraph import PresetGraph


# Default RNG seeds to use for our plots (generated via np.random.SeedSequence().entropy).
SEEDS = [
    257135840227949566337238051408055634111,
    303903392100660574452661293122954292079,
    207575285627819734050983420268628288457,
    336827387627079923696399650174942918686,
    175837567687085788645530192316345259609,
    233657882610370377508087340942845579878,
    139102178419114034645107386879574426047,
    90012362361482529980268819803967658146
]


@dataclass
class CurveParams:
    """Parameters important to generating and extracting the data needed for a given plot."""
    graph: Type[PresetGraph]  # Which graph to run the infections on.
    initial_name: Any         # The name (from VertexSet) of the fixed vertex to start the infections on.
    infection_cutoff: int     # The saturation point at which we stop considering new infections.
    mu: float                 # Degree penalisation parameter (see WeightedVertexSet.penalty).
    zeta: float               # Spatial penalisation parameter (see WeightedVertexSet.penalty).


@dataclass
class PlotParams:
    """Parameters important to creating a given plot from its data."""
    log_t: bool         # Whether to plot the time axis on a log scale.
    region_inset: bool  # Whether to include an inset displaying regional infections.
    # Range to plot for a linear regression inset determining the slope of the graph. (A value of None means no inset.)
    psi_inset_infection_range: Optional[Tuple[int, int]]


RUN_COUNT = 55      # Number of infections to simulate per graph.
PLOT_PRECISION = 4  # Increase to plot more points per graph.
FILENAME_BASE = "epidemic-curve-"  # Start of plot filenames


GOWALLA_INITIAL = 164
GOWALLA_CUTOFF = 64635
GOWALLA_XI_RANGE = (150, 5000)
GOWALLA_PARAMS = {"graph": GowallaGiantGraph, "initial_name": GOWALLA_INITIAL, "infection_cutoff": GOWALLA_CUTOFF}
SYN_GOWALLA_CUTOFF = 166667
SYN_GOWALLA_INITIAL = 230563
SYN_GOWALLA_XI_RANGE = (150, SYN_GOWALLA_CUTOFF)
SYN_GOWALLA_PARAMS = {"graph": SyntheticGowallaGraph, "initial_name": SYN_GOWALLA_INITIAL,
                      "infection_cutoff": SYN_GOWALLA_CUTOFF}
CURVE_PARAMS = [
    CurveParams(**GOWALLA_PARAMS, mu=0.0, zeta=0.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=0.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=2.0),
    CurveParams(**GOWALLA_PARAMS, mu=1.0, zeta=3.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=0.0, zeta=0.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=1.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=2.0),
    CurveParams(**SYN_GOWALLA_PARAMS, mu=1.0, zeta=3.0)
]


PLOT_PARAMS = [
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=True),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=True),
    PlotParams(log_t=True, psi_inset_infection_range=GOWALLA_XI_RANGE, region_inset=True),
    PlotParams(log_t=True, psi_inset_infection_range=GOWALLA_XI_RANGE, region_inset=True),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=False),
    PlotParams(log_t=False, psi_inset_infection_range=None, region_inset=False),
    PlotParams(log_t=True, psi_inset_infection_range=SYN_GOWALLA_XI_RANGE, region_inset=False),
    PlotParams(log_t=True, psi_inset_infection_range=SYN_GOWALLA_XI_RANGE, region_inset=False)
]
