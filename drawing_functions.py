import background_functions as bf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import *
import random
import numpy as np

# Generates a single heatmap matrix for given parameters, if savefig is set to True it saves the heatmap as a figure
# If set to False (Standard) the function simply returns the heatmap matrix
# (only works for d = 2 thats why it is already entered as such) but works for both PPP and Z^2. Take note that doing this for a PPP is currently very poorly optimized 
# Unless you take a value for n that is smaller than something like 10k it will take a very long time to run.
# 10k takes ~10 minutes, I haven't looked into optimizing this much yet 
def single_heatmap(size,tau,alpha,mu, vertex_set = "Z2", title = "0", savefig = False, ratio = 1, origin_index = None, method = 1, deg = None, seed = None):

    match vertex_set:
        case "Z2":
            g,pos,st,w,L_rv = bf.Lattice(size,2,tau,alpha,deg,seed)
            infs,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu,vertex_set,ratio,origin_index,method)
            heatmap_matrix = bf.heatmapMatrix(infs,size)
            
            if savefig: 
                fig = plt.figure(figsize=(14,14))
                bf.draw(heatmap_matrix,title)
                fig.savefig("Heatmap_" + str(title))

        case "PPP":
            g,pos,st,w,L_rv,distL = bf.PPPGirg(size,2,tau,alpha,deg,seed)
            infs,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu,vertex_set,ratio,origin_index,method)
            heatmap_matrix = bf.setLattice(size,g,pos,infs,noInfecs,mu)

            if savefig:
                fig = plt.figure(figsize=(14,14))
                colors1 = plt.cm.jet(np.linspace(0.,1,255))
                colors2 = plt.cm.Reds(np.linspace(0,1,1))
                colors = np.vstack((colors2,colors1))
                mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)
                bf.draw(heatmap_matrix,title,cmap = mymap)
                fig.savefig("Heatmap_" + str(title))

    if not savefig:
        return heatmap_matrix
    
# Generates 4 heatmaps for given parameters (given in list form with 4 entries) saves them automatically as seperate plots
# as well as all together. add a marker argument if you want the file with all four to have a different name than the standard one
# Title list is not necessary, it will otherwise simply call them 1,2,3,4
def four_heatmaps(lim,tau_list,alpha_list,mu_list, vertex_set = "Z2", title_list = [1,2,3,4], marker = "", ratio = 1, origin_index = None, method = 1, deg = None, seed = None):

    if len(tau_list) != 4 or len(alpha_list) != 4 or len(mu_list) != 4 or len(title_list) != 4:
        raise ValueError("One of your parameter lists is not of length 4.")
    
    tau_1, tau_2, tau_3, tau_4 = tau_list
    alpha_1, alpha_2, alpha_3, alpha_4 = alpha_list
    mu_1, mu_2, mu_3, mu_4 = mu_list
    title_1,title_2,title_3,title_4 = title_list

    heatmap_matrix_1 = single_heatmap(lim,tau_1,alpha_1,mu_1,vertex_set,title_1)
    heatmap_matrix_2 = single_heatmap(lim,tau_2,alpha_2,mu_2,vertex_set,title_2)
    heatmap_matrix_3 = single_heatmap(lim,tau_3,alpha_3,mu_3,vertex_set,title_3)
    heatmap_matrix_4 = single_heatmap(lim,tau_4,alpha_4,mu_4,vertex_set,title_4)

    fig = plt.figure(figsize=(14,14))
    colors1 = plt.cm.jet(np.linspace(0.,1,255))
    colors2 = plt.cm.Reds(np.linspace(0,1,1))
    colors = np.vstack((colors2,colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)

    match vertex_set:
        case "Z2":
            cmap = "jet"
        case "PPP":
            cmap = mymap

    ax1 = fig.add_subplot(2,2,1)
    bf.draw(heatmap_matrix_1,title_1,cmap = cmap)
    extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_" + str(title_1),bbox_inches = extent1.expanded(1.3,1.4))

    ax2 = fig.add_subplot(2,2,2)
    bf.draw(heatmap_matrix_2,title_2,cmap = cmap)
    extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_" + str(title_2),bbox_inches = extent2.expanded(1.3,1.4))

    ax3 = fig.add_subplot(2,2,3)
    bf.draw(heatmap_matrix_3,title_3,cmap = cmap)
    extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_" + str(title_3),bbox_inches = extent3.expanded(1.3,1.27))

    ax4 = fig.add_subplot(2,2,4)
    bf.draw(heatmap_matrix_4,title_4,cmap = cmap)
    extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_" + str(title_4),bbox_inches = extent4.expanded(1.3,1.27))

    fig.savefig('Four_heatmaps_' + str(marker) + '.png')

# Generates four heatmaps on the exact same graph, just mu varying, (exp(1)'s also stay the same!)
# saves each heatmap seperately and as a complete picture with all 4.
def four_heatmaps_same_graph(lim,tau,alpha,mu_list,vertex_set = "Z2", title_list = [1,2,3,4], marker = "", ratio = 1, origin_index = None, deg = None, seed = None):

    if len(mu_list) != 4 or len(title_list) != 4:
        raise ValueError("One of your parameter lists is not of length 4.")
    
    mu_1, mu_2, mu_3, mu_4 = mu_list
    title_1,title_2,title_3,title_4 = title_list

    match vertex_set:
        case "Z2":
            g,pos,st,w,L_rv = bf.Lattice(lim,2,tau,alpha,deg,seed)
            infs_1,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_1,vertex_set,origin_index)
            infs_2,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_2,vertex_set,origin_index)
            infs_3,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_3,vertex_set,origin_index)
            infs_4,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_4,vertex_set,origin_index)
            heatmap_matrix_1 = bf.heatmapMatrix(infs_1,lim)
            heatmap_matrix_2 = bf.heatmapMatrix(infs_2,lim)
            heatmap_matrix_3 = bf.heatmapMatrix(infs_3,lim)
            heatmap_matrix_4 = bf.heatmapMatrix(infs_4,lim)
        case "PPP":
            g,pos,st,w,L_rv,distL = bf.PPPGirg(lim,2,tau,alpha,deg,seed)
            infs_1,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_1,vertex_set,origin_index)
            infs_2,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_2,vertex_set,origin_index)
            infs_3,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_3,vertex_set,origin_index)
            infs_4,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu_4,vertex_set,origin_index)
            heatmap_matrix_1 = bf.setLattice(lim,g,pos,infs_1,noInfecs,mu_1)
            heatmap_matrix_2 = bf.setLattice(lim,g,pos,infs_2,noInfecs,mu_2)
            heatmap_matrix_3 = bf.setLattice(lim,g,pos,infs_3,noInfecs,mu_3)
            heatmap_matrix_4 = bf.setLattice(lim,g,pos,infs_4,noInfecs,mu_4)

    fig = plt.figure(figsize=(14,14))
    colors1 = plt.cm.jet(np.linspace(0.,1,255))
    colors2 = plt.cm.Reds(np.linspace(0,1,1))
    colors = np.vstack((colors2,colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)

    match vertex_set:
        case "Z2":
            cmap = "jet"
        case "PPP":
            cmap = mymap

    ax1 = fig.add_subplot(2,2,1)
    bf.draw(heatmap_matrix_1,title_1,cmap = cmap)
    extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_same_graph_" + str(title_1),bbox_inches = extent1.expanded(1.3,1.4))

    ax2 = fig.add_subplot(2,2,2)
    bf.draw(heatmap_matrix_2,title_2,cmap = cmap)
    extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_same_graph_" + str(title_2),bbox_inches = extent2.expanded(1.3,1.4))

    ax3 = fig.add_subplot(2,2,3)
    bf.draw(heatmap_matrix_3,title_3,cmap = cmap)
    extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_same_graph_" + str(title_3),bbox_inches = extent3.expanded(1.3,1.27))

    ax4 = fig.add_subplot(2,2,4)
    bf.draw(heatmap_matrix_4,title_4,cmap = cmap)
    extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("four_heatmaps_same_graph_" + str(title_4),bbox_inches = extent4.expanded(1.3,1.27))

    fig.savefig('Four_heatmaps_same_graph_' + str(marker) + '.png')


def circleInfectionTime(size,d,tau,alpha,mu,r, eps = 0.5, vertex_set = "Z2", marker = "", savefig = False, deg = None, seed = None, ratio = 1, origin_index = None, method = 1):

    match vertex_set:
        case "Z1":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
        case "Z2":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
        case "PPP":
            g,pos,st,w,L_rv = bf.PPPGirg(size,d,tau,alpha,deg,seed)
    
    infs,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu,vertex_set,method,origin_index)
    inf_time_set = bf.circlePeriod(g,pos,infs,r,eps,vertex_set,origin_index)

    fig = plt.figure(figsize=(14,14))
    plt.plot(inf_time_set)
    plt.xticks([0,len(inf_time_set)],['$-\pi$','$\pi$'])
    plt.xlabel("Angle of circle")
    plt.ylabel("Infection time")
    plt.title("Infection Time at a Circle of Radius r Away From the Origin Node")
    if not savefig:
        plt.show()
    if savefig:
        fig.savefig("circlePeriod_" + str(marker) +".png")

# When using any of the following functions for drawing geodesics pictures, make sure to use the "size" argument in any of the functions correctly with the corresponding vertex_set.
# This means; if using Z1 or Z2, "size" will be interpreted as "lim" in the generating lattice functions from background_functions. Meaning the size of the x/y-axis. 
# When using a PPP as a vertex set, "size" will be interpreted as the total amount of vertices. In my testing, 750 was around the highest i could get 2D lattices to go without it heavily impacting computation time. 
# For a PPP this was around 750k/1 million.

def radiusTime(size,d,tau,alpha,mu,r_num, vertex_set = "Z2", marker = "", savefig = False, deg = None, seed = None, ratio = 1, origin_index = None, method = 1):

    match vertex_set:
        case "Z1":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
        case "Z2":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
        case "PPP":
            g,pos,st,w,L_rv = bf.PPPGirg(size,d,tau,alpha,deg,seed)
    
    infs,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu,vertex_set,method,origin_index)
    r_list, median_times = bf.radiusCoords(g,pos,infs,noInfecs,r_num,vertex_set)

    fig = plt.figure(figsize=(14,14))
    plt.plot(r_list,median_times)
    plt.xlabel("r (distance from origin node)")
    plt.ylabel("median infection time")
    plt.title("Median Infection Time at a Circle of Radius r Away From the Origin Node as a function of r")
    if not savefig:
        plt.show()
    if savefig:
        fig.savefig("radiusTime_" + str(marker) +".png")

# This draws 5 geodesics, proportional cost of the longest edge, proportional cost of the most expensive edge, proportional length of the longest edge, degree of the starting node of the longest edge, and the hopcount.
# These geodesics are averaged over "sample_amount" (type: int, preferably divisible by 6 in the Z2 case for selection purposes but this is not a requirement) nodes on a single graph. 
# For most of my pictures I used a mu ranging from 0 to 2.5 in 201 steps (mu_num gives the amount of steps). 
# Note that this does tend to take a long time to run. as it has to run 201 simulations, on larger graphs this can quickly take up to anywhere between 10-25 hours.
def drawGeodesics(size,d,tau,alpha,mu_min,mu_max,mu_num,sample_amount, vertex_set = "Z2", marker = "", deg = None, seed = None, ratio = 1, origin_index = None, method = 1):

    mu_list = np.linspace(mu_min,mu_max,mu_num)

    expl_cutoff = (3-tau)/2
    polylog_cutoff = 3-tau
    poly_cutoff = (3-tau)/min([1,d*(alpha-2)]) + 1/d

    vline_list = []
    if polylog_cutoff < mu_max and mu_min < polylog_cutoff:
        vline_list.append(polylog_cutoff)
    if poly_cutoff < mu_max and mu_min < poly_cutoff:
        vline_list.append(poly_cutoff)
    if expl_cutoff < mu_max and mu_min < expl_cutoff:
        vline_list.append(expl_cutoff)

    match vertex_set:
        case "Z1":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
            lim = (g.num_vertices()/2)
            sampled_nodes = random.sample(range(int(lim*0.9),lim-1),int(sample_amount/2)) + random.sample(range(0,int(lim*0.1)),int(sample_amount/2))
        case "Z2":
            g,pos,st,w,L_rv = bf.Lattice(size,d,tau,alpha,deg,seed)
            lim = (sqrt(g.num_vertices())-1)
            sampled_nodes = random.sample(range(0,int(lim)*2),int(sample_amount/6)) + random.sample(range((lim+1)**2 - 2*lim,(lim+1)**2 ),int(sample_amount/6)) 
            + random.sample(range(0,(lim+1)**2,lim+1),int(sample_amount/6)) + random.sample(range(1,(lim+1)**2,lim+1),int(sample_amount/6))
            + random.sample(range(lim,(lim+1)**2,lim+1),int(sample_amount/6)) + random.sample(range(lim-1,(lim+1)**2,lim+1),int(sample_amount/6))
        case "PPP":
            g,pos,st,w,L_rv,distL = bf.PPPGirg(size,d,tau,alpha,deg,seed)
            distL_sorted = sorted(distL, reverse=True)
            distL_sorted = [u if u[1] not in noInfecs else [0,0] for u in distL_sorted]
            sampled_nodes = random.sample(distL_sorted[0:int(g.num_vertices()*0.05)],int(sample_amount))

    prop_cost_list = []
    propME_cost_list = []
    prop_length_list = []
    degree_list = []
    hopcount_list = []

    for mu in mu_list:

        infs,tc,noInfecs = bf.infectionSpread(g,st,w,L_rv,mu,vertex_set,method,origin_index)

        prop_cost_averaging_list = []
        propME_cost_averaging_list = []
        prop_length_averaging_list = []
        degree_averaging_list = []
        hopcount_averaging_list = []

        for v in sampled_nodes:
            prop_cost,propME_cost = bf.proportionalCost(g,pos,infs,tc,v,vertex_set,origin_index)
            prop_cost_averaging_list.append(prop_cost)
            propME_cost_averaging_list.append(propME_cost)
            prop_length_averaging_list.append(bf.proportionalLength(g,pos,infs,tc,v,vertex_set,origin_index))
            degree_averaging_list.append(bf.degreeLongestEdge(g,pos,infs,tc,v,vertex_set,origin_index))
            hopcount_averaging_list.append(bf.hopCount(g,pos,infs,tc,v,vertex_set,origin_index))
        
        prop_cost_list.append(np.mean(prop_cost_averaging_list))
        propME_cost_list.append(np.mean(propME_cost_averaging_list))
        prop_length_list.append(np.mean(prop_length_averaging_list))
        degree_list.append(np.mean(degree_averaging_list))
        hopcount_list.append(np.mean(hopcount_averaging_list))

    plt.figure(figsize=(10,10))
    plt.vlines(vline_list,ymin=0,ymax=max(prop_cost_list),colors='r',linestyles='dashed')
    plt.xlabel("mu")
    plt.ylabel("proportional cost ")
    plt.title("Proportional cost of the longest edge")
    plt.plot(mu_list,prop_cost_list)
    plt.savefig("ProportionalCost_" + str(marker) + ".png")

    plt.figure(figsize=(10,10))
    plt.vlines(vline_list,ymin=0,ymax=max(propME_cost_list),colors='r',linestyles='dashed')
    plt.xlabel("mu")
    plt.ylabel("proportional cost")
    plt.title("Proportional cost of the most expensive edge")
    plt.plot(mu_list,propME_cost_list)
    plt.savefig("ProportionalCostME_" + str(marker) + ".png")

    plt.figure(figsize=(10,10))
    plt.vlines(vline_list,ymin=0,ymax=max(prop_length_list),colors='r',linestyles='dashed')
    plt.xlabel("mu")
    plt.ylabel("proportional length")
    plt.title("Proportional length of the longest edge")
    plt.plot(mu_list,prop_length_list)
    plt.savefig("ProportionalLength_" + str(marker) + ".png")

    plt.figure(figsize=(10,10))
    plt.vlines(vline_list,ymin=0,ymax=max(degree_list),colors='r',linestyles='dashed')
    plt.xlabel("mu")
    plt.ylabel("degree")
    plt.title("Degree of starting node of the longest edge")
    plt.plot(mu_list,degree_list)
    plt.savefig("DegreeLongestEdge_" + str(marker) + ".png")

    plt.figure(figsize=(10,10))
    plt.vlines(vline_list,ymin=0,ymax=max(hopcount_list),colors='r',linestyles='dashed')
    plt.xlabel("mu")
    plt.ylabel("hops")
    plt.title("Amount of hops between start node and a far away node")
    plt.plot(mu_list,hopcount_list)
    plt.savefig("HopCount_" + str(marker) + ".png")






