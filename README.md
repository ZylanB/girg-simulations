# Simulating Infections on GIRGs and Drawing Images
The two main files are background_functions and drawing_functions, both names are fairly self-explanatory. 
Any functions that have to do with generating the graphs, running the simulations, or really anything that doesnt generate an image is in background_functions.
Code that generates pictures through matplotlib is all in drawing_functions, where there are multiple ways of generating and saving heatmaps, geodesics images, and some other images I've made in the past. 

There is also a gowalla folder with a gowalla.py file, this includes all my code used for the gowalla dataset and also some .pickle files which i created to save time working with the data from the Gowalla dataset. I didn't really clean this up so apologies if it is kind of messy. \

Below ill write some quick explanations about how the functions are usually structured, if needed I can write more extensive documentation. There are also comments throughout the code with notes on how certain functions work. 

### Argument Explanations
Most functions here will use the same arguments so here is a quick overview of what the most common ones mean: \
##### Graph Generation 
`lim` / `n` / `size`: These arguments determine the size of the graph, usually when 'lim' is used it is for a Lattice-type underlying vertex set. Where lim indicates the size of the x/y-axis. For example lim = 100 creates a lattice of 101x101, also `lim` has to be an even number. `n` is usually used when the underlying vertex set is a PPP. `size` is mostly used in drawing_functions when there is ambiguity on when Z^d or a PPP will be used. \
`d`: The dimension on which the vertex set is created. \
`tau`: The power law parameter used for determining the weights. \
`alpha`: The long range penalty parameter used for edge connections. \
`mu`: The penalty on transmission costs on edges. \
`deg`: Usually set to **None**. This influences the average degree of the graph (see the graph generation package at https://github.com/gavento/girg-sampling). 

##### Infection Simulation 
`g`: The saved graph in graph-tool format (https://graph-tool.skewed.de/static/doc/index.html). \
`status`: Binary variable indicating infected or not (1 means infected, 0 means not infected). \
`weights`: Weights according to a power law distribution for each node. \
`L_rv`: Exponential(1) random variables used for the edge transmission costs \
`vertex_set`: One of three options, "Z1","Z2","PPP", this indicates what the underlying vertex set of the graph is. Respectively Z^1, Z^2, or a PPP. Defaults at "Z2". \
`ratio`: Defaults to 1. If set to anything less than 1 it will cut the infection simulation at when the given ratio of nodes is infected. \
`origin_index`: The index of the node at which the infection starts. \
`method`: Equal to either 1 or 2, and determines the edge costs. 1 is standard edge costs using the weights of the nodes, 2 uses the degrees - penalty, penalty is changable within the code itself (not very neat I know Im sorry) 

##### Drawing Pictures
`savefig`: Boolean, if set to True it saves the picture as a .png in the current directory. \
`title`: Changes the title of the matplotlib image, also when used for functions where multiple images are created it adds the title to the filename. \
`marker`: Adds some text to the filename for storing. 





