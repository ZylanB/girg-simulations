# Simulating Infections on GIRGs and Drawing Images

This repository contains the code used to generate experimental results for the paper "Degree-dependent and 
distance-dependent contact rates interpolate between explosive, exponential and polynomial epidemic growth" by Zylan 
Benjert, Júlia Komjáthy, John Lapinskas, Johannes Lengler and Ulysse Schaller.

The code in the main folder focuses on generating spatial graphs, running SI infections on them, and extracting usable 
results:

**VertexSet.py:** Code for generating vertex sets of a spatial graphs with arbitrary metrics as VertexSets. Includes 
support for torus, Euclidean and haversine metrics on lattice, fixed and Poisson point process vertex sets. 

**WeightedVetexSet.py:** Code for adding weights to a VertexSet intended to coincide with average degree up to a 
constant factor, forming a WeightedVertexSet. Includes support for power-law distributions. 

**SIEpidemic.py:** Code for adding edge sets to WeightedVertexSets to form an SIEpidemic. (The graph itself is stored
in graph-tool format.) Includes support for GIRGs via our fork of Weyand's sampling library 
[here](https://github.com/Hamiltonicity/cpp-girgs-fork/) and Gavenčiak's Python bindings for it 
[here](https://github.com/Hamiltonicity/py-girgs-fork/), which modify the edge connection probability. 
Includes code to run SI epidemics on these graphs with customisable cost functions including FPP.

**SIExperiment.py:** Code for running repeated epidemics to extract results from them, with tha ability to resample any 
subset of the vertices, weights, edges, and transmission costs. Includes logging support.

**LoggableFunction.py:** Supporting code for the above files allowing for smoother automatic logging of exactly
what generator functions are being used in a given SIExperiment.

**PresetGraph.py:** Code for quickly generating a fixed graph as an SIEpidemic, e.g. the actual Gowalla network 
or our synthetic Gowalla network (a GIRG with the Gowalla network's estimated parameters). In particular, allows for
automatic caching to save time on graph generation.

**Gowalla.py:** Code to download the Gowalla dataset from SNAP and generate a copy of the Gowalla dataset as set out
in our paper, as well as a synthetic Gowalla network with parameters as estimated in our paper.

**Region.py:** Supporting code for Gowalla.py, providing an easy interface for cartopy for checking whether a given
vertex is in the US, Europe, or neither.

The code in the "figures" folder is focused on generating specific results for our paper:

**EpidemicCurves.py:** Generates the epidemic curves on the real and synthetic Gowalla network as shown in Figure 3.

**HeatMaps.py:** Generates the heatmaps for SI infections on the real and synthetic Gowalla network as shown in Figure 
4.

**HeatMapsFoM.py:** Generates similar heatmaps on lattices for the final version of another paper, "Four universal 
growth regimes in degree-dependent first passage percolation on spatial random graphs" by Júlia Komjáthy, John 
Lapinskas, Johannes Lengler and Ulysse Schaller.

**ParameterEstimates.py:** Estimates tau and alpha parameters for the synthetic Gowalla network from the actual Gowalla
network as discussed in our paper. Also contains code for Figure S8 in the SI.

The code in the "tests" folder is a suite of unit tests to ensure the rest of the code behaves as expected. In 
particular, some of our random generation code uses unit tests based around the DKW inequality to ensure the resulting
graphs follow the correct distribution up to a small possible error in total variation distance. Further explanation
is in **TestDistribution.py**.

**Installation instructions:** In addition to the dependencies in requirements.txt, you will need to install the GIRG
sampling libraries linked above and [graph-tool](https://graph-tool.skewed.de/), which are not available on pip. 
Graph-tool is easiest to install via conda, while the Python bindings for the GIRG sampling library are easiest to 
install via poetry, so this is non-trivial; more detailed instructions are available in its repo, but we are not 
experts on Python build environments and they are provided on an "as-is" basis. Graph-tool is also Linux-based, so if 
you are on a Windows machine you will need to use WSL. 
