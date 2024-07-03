import girg_sampling.girgs as gs
import graph_tool.all as gt
import numpy as np 
from math import *
import time
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
from line_profiler import LineProfiler
import pickle

# Functions needed for generating a 2D Lattice, only implemented for d = 1,2
def genGrid(lim,d):
    gridpoints = []

    if d == 1: 
        for x in range(lim+1):
            gridpoints.append([x])

    if d == 2:
        for x in range(lim,-1,-1):
            for y in range(0,lim+1,1):
                p = [float(y),float(x)]
                gridpoints.append(p)

    return gridpoints

#lim is the x/y-axis limit of grid points, so lim = 500 means 500 rows and 500 columns starting at 0, meaning 501^2 amount of nodes, (if d = 2)
def genLattice(lim,d,tau,alpha, deg = None, seed = None):
    start_time = time.time()
    nodes = np.asarray(genGrid(lim,d))/(lim + 0.0001)
    weights = np.asarray(gs.generateWeights((lim+1)**2,ple=tau, seed = seed), dtype ='float64')
    weights_scaled = weights
    if deg is not None:
        beta = gs.scaleWeights(weights,deg,d,alpha)
        weights_scaled = weights*beta
    edges = gs.generateEdges(weights_scaled,nodes,alpha, seed = seed)
    nodes = nodes*(lim + 0.0001)

    g = gt.Graph(edges,directed=False)

    pos = g.new_vertex_property("vector<long double>")
    status = g.new_vertex_property("int")

    for u in g.vertices():
        pos[u] = (nodes[int(u)][0] , nodes[int(u)][1])
        status[u] = 0
        if pos[u][0] < lim:
            g.edge(int(u),int(u)+1,add_missing=True)
        if pos[u][1] > 0:
            g.edge(int(u),int(u)+lim+1,add_missing=True)

    print("Graph Generation: --- %s seconds ---" % (time.time() - start_time))
    return g,pos,status,weights_scaled

# Generating a GIRG on a underlying vertex set which is a PPP scaled to a box of dimensions sqrt(n)
def genGirg(n,d,tau,alpha,deg = None, seed = None):
    start_time = time.time()
    box_d = sqrt(n)
    nodes = np.asarray(gs.generatePositions(n,dimension= d, seed = seed))
    nodes = np.insert(nodes,0,[0.50,0.50],axis=0)
    weights = np.asarray(gs.generateWeights(n+1,ple=tau, seed = seed))
    weights_scaled = weights
    if deg is not None:
        beta = gs.scaleWeights(weights,deg,d,alpha)
        weights_scaled = weights*beta
    edges = gs.generateEdges(weights_scaled,nodes,alpha, seed = seed)
    nodes = nodes*box_d

    g = gt.Graph(edges,directed=False)

    pos = g.new_vertex_property("vector<long double>")
    status = g.new_vertex_property("int")

    dist_list = []

    for u in g.vertices():
        pos[u] = ( nodes[int(u)][0] , nodes[int(u)][1])
        dist = max(min(abs( pos[u][0] - pos[0][0]), box_d - abs(pos[u][0] - pos[0][0])), min(abs( pos[u][1] - pos[0][1]), box_d - abs(pos[u][1] - pos[0][1])))
        dist_list.append([dist,u])
        status[u] = 0
    print("Graph Generation: --- %s seconds ---" % (time.time() - start_time))
    return g,pos,status,weights,dist_list



# Generating exp(1)'s for each edge in graph G
def L_exponentials(g):
    L_rv = g.new_edge_property("float")
    for e in g.edges():
        L_rv[e] = np.random.exponential(1)
    return g,L_rv

# The following are simply generating the graphs and exp(1)'s for the edge costs together. Saves a line in future code, unless you want to vary the exponentials then use the previous function seperately
def Lattice(lim,d,tau,alpha,deg = None, seed = None):
    g,pos,st,w = genLattice(lim,d,tau,alpha,deg,seed)
    g, L_rv = L_exponentials(g)
    return g,pos,st,w,L_rv 

def PPPGirg(n,d,tau,alpha,deg = None, seed = None):
    g,pos,st,w,dist_list = genGirg(n,d,tau,alpha,deg, seed)
    g, L_rv = L_exponentials(g)
    return g,pos,st,w,L_rv,dist_list


# Function for simulating the infection, returns a dictionary where they keys are the indexes of the nodes. 
# dict[key] returns a list with 3 values, where the [0] is the infection index, [1] is the time it took to infect
# and [2] is the index of the node which infected [key]. 
# argument "vertex_set" is either "Z1", "Z2", or "PPP", this is for determining the origin_index node (1-d lattice, 2-d lattice, PPP respectively as underlying vertex sets), automatically set to "Z2" as this is what ive used most
def infectionSpread(g,status,weights,L_rv,mu, vertex_set = "Z2", ratio = 1, origin_index = None, method = 1):
    start_time = time.time()
    if vertex_set not in ["Z1","Z2","PPP"]:
        raise TypeError("The vertex_set argument entered is not one of Z1,Z2 or PPP")
    
    num_vertices = g.num_vertices()
    cutoff = ratio*num_vertices
    trans_cost = g.new_edge_property("float")

    for e in g.edges():
        match method:
            case 1:
                trans_cost[e] = L_rv[e]*(weights[int(e.source())]*weights[int(e.target())])**mu
            case 2:
                penalty = 16 # Change this to change the penalty within the edge cost degree calculation
                trans_cost[e] = L_rv[e] * (max(1,e.source().out_degree()-penalty)*(max(1,e.target().out_degree()-penalty)))**mu

    if origin_index is None:   
        match vertex_set:
            case "Z1":
                origin_index = (g.num_vertices()/2)/2
            case "Z2":
                origin_index = ((sqrt(g.num_vertices())-1)+1)*((sqrt(g.num_vertices())-1)/2) + ((sqrt(g.num_vertices())-1)/2)
            case "PPP": 
                origin_index = 0

    inf_nodes = {origin_index: [0,0,-1]} 
    status[origin_index] = 1 # 1 = infected
    time_passed = 0
    steps = 0
    Q = PriorityQueue()

    for e in g.vertex(origin_index).out_edges():
        Q.put((trans_cost[e],int(e.target()),int(g.vertex(origin_index))))

    k = 0
    while not Q.empty():
        if len(inf_nodes) > cutoff:
            break
        next_inf = Q.get()
        steps += 1
        while status[next_inf[1]] == 0:  
            k += 1          
            time_passed = next_inf[0]
            for e in g.vertex(next_inf[1]).out_edges():
                if status[e.target()] == 1:
                    continue
                Q.put((trans_cost[e]+time_passed,int(e.target()),int(g.vertex(next_inf[1]))))
            inf_nodes[next_inf[1]] = [k, time_passed,next_inf[2]]
            status[next_inf[1]] = 1

    not_infected_vertices = []
    for u in g.vertices():
            if status[u] == 0:
                not_infected_vertices.append(u)
            status[u] = 0

    print("Infection Simulation: --- %s seconds ---" % (time.time() - start_time))

    return inf_nodes, trans_cost, not_infected_vertices

# Code for generating a morans index value, this only works on a 2D lattice for k = 1 (rook definition of neighbors), on a 1D lattice it works for any k, does not work on PPP
def moransIndex(g,d,k,infs,lim): 
    start_time = time.time()

    tc_mean = (g.num_vertices() + 1)/2
    big_N = g.num_vertices()
    big_W = 0
    first_sum = 0
    bottom_sum = 0

    weight_dict = {}
    for i in range(0,k):
        weight_dict[i+1] = (1/2)**i

    for u in g.vertices():
        u_neighbors = []
        second_sum = 0
        ind_u = infs[int(u)][0]

        if d == 2:  
            if int(u) % (lim+1) == 0:
                u_neighbors = [int(u)+lim,int(u)+1]
            elif (int(u)+1) % (lim+1) == 0:
                u_neighbors = [int(u)-1,int(u)-lim]
            else:
                u_neighbors = [int(u)-1,int(u)+1]
            if (lim-floor(int(u)/(lim+1))) == lim:
                u_neighbors.append(int(u)+(lim+1))
                u_neighbors.append(int(u) + (lim + lim**2))
            elif (lim - floor(int(u)/(lim+1))) == 0:
                u_neighbors.append(int(u)-(lim+1))
                u_neighbors.append(int(u)-(lim+lim**2))
            else:
                u_neighbors.append(int(u)+(lim+1))
                u_neighbors.append(int(u)-(lim+1))

        if d == 1:
            for i in range(int(u)-k,int(u)):
                if i < 0:
                    u_neighbors.append(g.num_vertices()+i)
                else:
                    u_neighbors.append(i) 
            for i in range(int(u)+1,int(u)+k+1):
                if i > (g.num_vertices()-1):
                    u_neighbors.append(abs(g.num_vertices()-i))
                else:
                    u_neighbors.append(i)

        bottom_sum += (ind_u-tc_mean)**2

        for v in u_neighbors:
            if d == 1:
                dist = round(min(abs(int(u)-v),(g.num_vertices()) - abs(int(u)-v)))
                weight = weight_dict[dist]
            if d == 2:
                weight = 1
            big_W += weight
            second_sum += weight*(ind_u - tc_mean)*(infs[int(v)][0] - tc_mean)
        first_sum += second_sum

    morans_Index = (big_N/big_W)*(first_sum/bottom_sum) 

    print("Calculating Moran's I: --- %s seconds ---" % (time.time() - start_time))

    return morans_Index


#Generates a heatmap matrix for a 2D Lattice
def heatmapMatrix(infs,lim):
    start_time = time.time()
    infMatrix = np.zeros([lim+1,lim+1])
    i = -1
    for row in range(lim+1):
        for col in range(lim+1):
            i += 1 
            infMatrix[row][col] = infs[i][0]
    print("Generating Heatmap: --- %s seconds ---" % (time.time() - start_time))
    return infMatrix

#Draws the heatmap
def draw(m,title,cmap = 'jet'):
    plt.imshow(m,cmap,interpolation ='spline16')
    plt.title(title)

def setLattice(n,g,pos,infs,noinfecs):
    start_time = time.time()
    lim = int(sqrt(n))
    matrix = np.zeros([lim,lim])
    for row in range(lim):
        for col in range(lim):
            infec_order_list = []
            for u in g.vertices():
                if abs(pos[u][0] - row) <= 0.499 and abs(pos[u][1] - col) <= 0.499 and u not in noinfecs:
                    infec_order_list.append(infs[int(u)][0])
            if len(infec_order_list) > 0:
                matrix[row][col] = np.median(infec_order_list)
    print("Generating Heatmap: --- %s seconds ---" % (time.time() - start_time))
    return matrix


# This returns an array of infection times of nodes on a circle at radius r sorted from -pi to pi
# eps is an argument used to determine the error radius for points along the circle
def circlePeriod(g,pos,infs,r,eps = 0.5,vertex_set = "Z2",origin_index = None):
    vertex_set_r = []
    if origin_index is None:   
        match vertex_set:
            case "Z1":
                lim = (g.num_vertices()/2)
                origin_index = lim/2
                og_pos_x = origin_index
            case "Z2":
                lim = (sqrt(g.num_vertices())-1)
                origin_index = (lim+1)*(lim/2) + (lim/2)
                og_pos_x = pos[origin_index][0]
                og_pos_y = pos[origin_index][1]
            case "PPP": 
                lim = sqrt(g.num_vertices())
                origin_index = 0
                og_pos_x = pos[origin_index][0]
                og_pos_y = pos[origin_index][1]
    for u in g.vertices():
        match vertex_set:
            case "Z1":
                dist = min(abs(int(u) - og_pos_x), lim - abs(int(u) - og_pos_x))
                ang = 0
            case "Z2":
                dist = max(min(abs(pos[u][0] - og_pos_x), lim - abs(pos[u][0] - og_pos_x)), min(abs(pos[u][1] - og_pos_y), lim - abs(pos[u][1] - og_pos_y)))
                ang = atan2(pos[u][1]-og_pos_y,pos[u][0]-og_pos_x)    
            case "PPP":
                dist = max(min(abs(pos[u][0] - og_pos_x), lim - abs(pos[u][0] - og_pos_x)), min(abs(pos[u][1] - og_pos_y), lim - abs(pos[u][1] - og_pos_y))) 
                ang = atan2(pos[u][1]-og_pos_y,pos[u][0]-og_pos_x)   
        if dist >= (r-eps) and dist <= (r+eps):
            vertex_set_r.append([u,ang])
    vertex_set_r = sorted(vertex_set_r,key=lambda x: x[1])
    vertex_set_r = np.asarray(vertex_set_r)
    vertex_set_r = vertex_set_r[:,0]
    inf_time_set = []
    for u in vertex_set_r:
        inf_time_set.append(infs[u][1])
    return inf_time_set

# This returns the median infection time of nodes at a distance r of the origin_index
def medianBorder(g,pos,infs,r,noInfecs):
    vertex_set_r = []
    origin_index = 0
    og_pos_x = pos[origin_index][0]
    og_pos_y = pos[origin_index][1]
    eps = 0.05
    for u in g.vertices():
        dist = sqrt((pos[u][0]-og_pos_x)**2 + (pos[u][1] - og_pos_y)**2)
        if dist >= (r-eps) and dist <= (r+eps):
            if u not in noInfecs:
                vertex_set_r.append(u)
    inf_time_set = []
    for u in vertex_set_r:
        inf_time_set.append(infs[int(u)][1])
    return np.median(inf_time_set)

# This returns a list of different distances r, and a list of the median times at these r
def radiusCoords(g,pos,infs,noInfecs,r_num,vertex_set = "Z2"):
    start_time = time.time()

    match vertex_set:
        case "Z1":
            max_dist = g.num_vertices()/2
        case "Z2": 
            max_dist = sqrt(2)*((sqrt(g.num_vertices())-1)/2)
        case "PPP":
            max_dist = sqrt(2*g.num_vertices())

    r_list = np.linspace(1,max_dist,r_num)

    median_times = []

    for r in r_list: 
        med = medianBorder(g,pos,infs,r,noInfecs)
        median_times.append(med)

    print("Calculating Median Infection Time at Radii r: --- %s seconds ---" % (time.time() - start_time))

    return r_list,median_times

# This finds the infection path of any infected node. It returns an edge path which for each edge has a list of [distance of edge,cost of edge]
# And a node path, which simply returns a list of the indices of the nodes which lead to infection of the chosen node. 
# Note the lists are returned in reverse
# For the Z2 case of calculating distances, it uses the way the grid is structured to get the distances, it is tedious to write out but i have it on paper somewhere if you need it
def findInfectionPath(g,pos,infs,tc,v,vertex_set = "Z2",origin_index = None):
    box_d = sqrt(g.num_vertices())
    if origin_index is None:   
        match vertex_set:
            case "Z1":
                lim = (g.num_vertices()/2)
                origin_index = lim/2
            case "Z2":
                lim = (sqrt(g.num_vertices())-1)
                origin_index = (lim+1)*(lim/2) + (lim/2)
            case "PPP": 
                origin_index = 0
    edge_path = []
    node_path = []
    prev = v
    while prev != origin_index:
        match vertex_set:
            case "Z1":
                dist = min(abs(int(infs[prev][2]) - prev), lim - abs(int(infs[prev][2]) - prev))
            case "Z2":
                dist = max(min(abs((int(infs[prev][2]) % (lim+1)) - (prev % (lim+1))), lim - abs((int(infs[prev][2]) % (lim+1)) - (prev % (lim+1)))), min(abs((lim - floor(int(infs[prev][2])/(lim+1))) - (lim - floor(prev/(lim+1)))), 1 - abs((lim - floor(int(infs[prev][2])/(lim+1))) - (lim - floor(prev/(lim+1))))))    
            case "PPP":
                dist = max(min(abs( pos[int(infs[prev][2])][0] - pos[prev][0]), box_d - abs(pos[int(infs[prev][2])][0] - pos[prev][0])), min(abs( pos[int(infs[prev][2])][1] - pos[prev][1]), box_d - abs(pos[int(infs[prev][2])][1] - pos[prev][1])))        
        cost = tc[g.edge(infs[prev][2],prev)]
        edge_path.append([dist,cost])
        node_path.append(prev)
        prev = infs[prev][2]
    node_path.append(origin_index)
    return edge_path,node_path

def proportionalCost(g,pos,infs,tc,v,vertex_set = "Z2",origin_index = None):
    edge_path, node_path = findInfectionPath(g,pos,infs,tc,v,vertex_set,origin_index)
    longest_edge = max(edge_path)
    longest_cost = longest_edge[1]
    edge_path = np.asarray(edge_path)
    proportional_cost_most_expensive_edge = max(edge_path[:,1])/sum(edge_path[:,1])
    proportional_cost = longest_cost/sum(edge_path[:,1])
    return proportional_cost, proportional_cost_most_expensive_edge

def proportionalLength(g,pos,infs,tc,v,vertex_set = "Z2",origin_index = None):
    edge_path, node_path = findInfectionPath(g,pos,infs,tc,v,vertex_set,origin_index)
    longest_edge = max(edge_path)
    edge_path = np.asarray(edge_path)
    return longest_edge[0]/sum(abs(edge_path[:,0]))

def hopCount(g,pos,infs,tc,v,vertex_set = "Z2",origin_index = None):
    edge_path, node_path = findInfectionPath(g,pos,infs,tc,v,vertex_set,origin_index)
    return len(edge_path)

def degreeLongestEdge(g,pos,infs,tc,v,vertex_set = "Z2",origin_index = None):
    edge_path, node_path = findInfectionPath(g,pos,infs,tc,v,vertex_set,origin_index)
    longest_edge = max(edge_path)
    long_ind = np.where(edge_path[:,0]==longest_edge[0])[0][0] + 1
    origin_node_longest_edge = node_path[long_ind]
    return g.vertex(origin_node_longest_edge).out_degree()



