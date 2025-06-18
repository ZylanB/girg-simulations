import graph_tool.all as gt
import numpy as np 
from math import *
import time
import random
from queue import PriorityQueue
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from matplotlib.patches import ConnectionPatch
import pickle
import networkx as nx 
from sklearn.linear_model import LinearRegression
from PIL import Image
import matplotlib.ticker as ticker
import background_functions as bf
import haversine
import drawing_functions as df


path = '/home/zylan/python/gowalla/gow_graph_mode_new.pickle'

gow_areas = {"US": [(65,100),(32,95)],"Europe": [(80,115),(145,200)],"full": [(0,115),(0,315)]} # ,"Japan": [(70,91),(282, 313)],"Australia": [(56,100),(273,313)]}

gow_alpha = 1.15
gow_tau = 2.647

def L_exponentials(g):
    L_rv = g.new_edge_property("float")
    for e in g.edges():
        L_rv[e] = np.random.exponential(1)
    return g,L_rv

#pos is lat/long ([0]-[1])
def globeDistance(lat_u,long_u,lat_v,long_v,shifted = False):
    phi_1 = lat_u*pi/180
    phi_2 = lat_v*pi/180
    dPhi = (lat_v-lat_u)*pi/180
    dLambda = (long_v-long_u)*pi/180
    a = sin(dPhi/2)*sin(dPhi/2) + cos(phi_1)*cos(phi_2)*sin(dLambda/2)*sin(dLambda/2)
    c = 2*atan2(sqrt(a),sqrt(1-a))
    return 673.1*c # 6731 gives in 1's km
    if shifted: 
        lat_u -= 46
        lat_v -= 46
        long_u -= 160
        long_v -= 160
    return haversine.haversine((lat_u,long_u),(lat_v,long_v))/10

def new_infectionSpread(g,pos,id,status,L_rv,mu,zeta = 0,method = 2,ratio = 1,origin_index = None, penalize = False, shifted = False): 
    start_time = time.time()
    num_vertices = g.num_vertices()
    cutoff = ratio*num_vertices
    trans_cost = g.new_edge_property("float")
    penalty = 0 
    if penalize:
        penalty = (2/3)*gt.vertex_average(g,"total")[0]
    for e in g.edges():
        distance = globeDistance(pos[id[e.source()]][0],pos[id[e.source()]][1],pos[id[e.target()]][0],pos[id[e.target()]][1],shifted = shifted)
        match method:
            case 1: #Just Spatial penalization
                trans_cost[e] = L_rv[e]*(max(1,distance))**zeta
            case 2: #Spatial and Degree penalization
                trans_cost[e] = L_rv[e]*(max(1,distance)**zeta)*((max(1,e.source().out_degree()-penalty)*(max(1,e.target().out_degree()-penalty)))**mu)
    if origin_index is None:    
        origin_index = 164
    inf_nodes = {origin_index: [0,0,-1]} #INDEX OF INFECTED, TIME PASSED, NODE WHICH HAS INFECTED CURRENT NODE
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
    #print("Steps = ", steps)
    return inf_nodes, trans_cost, not_infected_vertices

###### The following functions I wrote months ago so I do not know if they still work ######
                   ######## Starts here ##########
def sortDistances(g,pos,id,origin = 316):
    start_time = time.time()
    dist_list = []
    for u in g.vertices():
        dist = sqrt( (pos[id[int(u)]][0]-pos[id[int(origin)]][0])**2  + (pos[id[int(u)]][1]-pos[id[int(origin)]][1])**2 )
        dist_list.append([dist,u])
    print("Dist: --- %s seconds ---" % (time.time() - start_time))
    return dist_list

def findInfectionPath(g,pos,id,infs,tc,v,origin_index = None):
    if origin_index is None:    
        origin_index = 164
    edge_path = []
    node_path = []
    prev = v
    while prev != origin_index:
        next = infs[prev][2]
        lat_u = pos[id[v]][0]
        long_u = pos[id[v]][1]
        lat_v = pos[id[next]][0]
        long_v = pos[id[next]][1]
        dist = globeDistance(lat_u,long_u,lat_v,long_v)                     
        cost = tc[g.edge(next,prev)]
        edge_path.append([dist,cost])
        node_path.append(prev)
        prev = infs[prev][2]
    node_path.append(origin_index)
    return edge_path,node_path

def medianBorder(g,pos,infs,r,id,noInfecs):
    vertex_set_r = []
    origin_index = 6700
    og_pos_x = pos[id[origin_index]][0]
    og_pos_y = pos[id[origin_index]][1]
    eps = 0.2
    for u in g.vertices():
        dist = sqrt((pos[id[u]][0]-og_pos_x)**2 + (pos[id[u]][1] - og_pos_y)**2)
        if dist >= (r-eps) and dist <= (r+eps):
            if u not in noInfecs:
                vertex_set_r.append(u)
    inf_time_set = []
    for u in vertex_set_r:
        inf_time_set.append(infs[int(u)][1])
    return np.median(inf_time_set)

def radiusCoords(g,st,id,pos,Lrv,mu):
    start_time = time.time()
    infs,tc,noinfecs = new_infectionSpread(g,st,Lrv,mu)
    distL = sortDistances(g,pos,id)
    distL_sorted = sorted(distL, reverse=True)
    max_dist = distL_sorted[0][0]
    r_list = np.linspace(1,max_dist,int(max_dist/3))
    median_times = []
    for r in r_list: 
        med = medianBorder(g,pos,infs,r,id,noinfecs)
        median_times.append(med)
    print("--- %s seconds ---" % (time.time() - start_time))
    return r_list,median_times

def runGeodesics(mu_min,mu_max,num,version): 

    with open('gow_graph_mode.pickle', 'rb') as data:
        g,st,id,pos = pickle.load(data)

    start_time = time.time()
    mu_list = np.linspace(mu_min,mu_max,num)
    distL = sortDistances(g,pos,id)
    g, L_rv = L_exponentials(g)
    ignore_infs, ignore_tc, noInfecs = new_infectionSpread(g,st,L_rv,0,id)

    start_time2 = time.time()
    distL_sorted = sorted(distL, reverse=True)
    distL_sorted = [u if u[1] not in noInfecs else [0,0] for u in distL_sorted]
    print("Sorting Time: --- %s seconds ---" % (time.time() - start_time2))

    sampled_nodes = random.sample(distL_sorted[0:int(g.num_vertices()*0.02)],20)
    prop_cost_list = []
    propME_cost_list = []
    prop_length_list = []
    degree_list = []
    hopcount_list = []
    k=0
    for mu in mu_list:
        k += 1
        print(k)
        if k == 1 or k == 50 or k == 100 or k == 150 or k == 200:
            print("quarter, k = " + str(k))
            print("--- %s minutes ---" % ((time.time() - start_time)/60))
        infs, tc, ignore_list = new_infectionSpread(g,st,L_rv,mu,id)
        prop_cost_averaging_list = []
        propME_cost_averaging_list = []
        prop_length_averaging_list = []
        degree_averaging_list = []
        hopcount_averaging_list = []
        for u in sampled_nodes:
            ep, node_p = findInfectionPath(g,pos,id,infs,tc,u[1])
            longest_edge = max(ep)
            longest_cost = longest_edge[1]
            ep = np.asarray(ep)
            proportional_cost_most_expensive_edge = max(ep[:,1])/sum(ep[:,1])
            proportional_cost = longest_cost/sum(ep[:,1])
            proportional_length = longest_edge[0]/sum(abs(ep[:,0]))
            long_ind = np.where(ep[:,0]==longest_edge[0])[0][0] + 1
            origin_node_longest_edge = node_p[long_ind]
            degree = g.vertex(origin_node_longest_edge).out_degree()
            prop_cost_averaging_list.append(proportional_cost)
            propME_cost_averaging_list.append(proportional_cost_most_expensive_edge)
            prop_length_averaging_list.append(proportional_length)
            degree_averaging_list.append(degree)
            hopcount_averaging_list.append(len(ep))
        prop_cost_list.append(np.mean(prop_cost_averaging_list))
        propME_cost_list.append(np.mean(propME_cost_averaging_list))
        prop_length_list.append(np.mean(prop_length_averaging_list))
        degree_list.append(np.mean(degree_averaging_list))
        hopcount_list.append(np.mean(hopcount_averaging_list))
    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("proportional cost ")
    plt.title("Proportional cost of the longest edge")
    plt.plot(mu_list,prop_cost_list)
    plt.savefig("PropCost_avgd_" + str(version) + ".png")

    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("proportional cost")
    plt.title("Proportional cost of the most expensive edge")
    plt.plot(mu_list,propME_cost_list)
    plt.savefig("PropCostME_" + str(version) + ".png")

    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("proportional length")
    plt.title("Proportional length of the longest edge")
    plt.plot(mu_list,prop_length_list)
    plt.savefig("PropLength_avgd_" + str(version) + ".png")

    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("degree")
    plt.title("Degree of starting node of the longest edge")
    plt.plot(mu_list,degree_list)
    plt.savefig("DegreeLongestEdge_" + str(version) + ".png")

    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("hops")
    plt.title("Amount of hops between start node and a far away node")
    plt.plot(mu_list,hopcount_list)
    plt.savefig("HopCount_" + str(version) + ".png")

    plt.figure(figsize=(10,10))
    plt.xlabel("mu")
    plt.ylabel("hops")
    plt.title("Amount of hops between start node and a far away node (Zoomed in)")
    plt.plot(mu_list[0:136],hopcount_list[0:136])
    plt.savefig("HopCountZoomed_" + str(version) + ".png")

###### The following functions I wrote months ago so I do not know if they still work ######
                     ########## Ends here #############

# Min max X-cords = -45.91, 70.08
# Min Max Y-cords = -159.67, 176.92

def rank_normalize(matrix):
    start_time = time.time()
    nonzero_values = np.sort(matrix[matrix != 0])
    value_to_rank = {val: rank+1 for rank, val in enumerate(nonzero_values)}
    ranked_matrix = np.zeros_like(matrix, dtype=int)
    for val, rank in value_to_rank.items():
        ranked_matrix[matrix == val] = rank
    print("Matrix Normalizing Time: --- %s minutes ---" % ((time.time() - start_time)/60))
    return ranked_matrix

def setLattice(g,id,pos,infs_list,noinfecs,area = gow_areas["full"],method = "first",matrix_scaling = 1): #matrix_scaling needs to be an integer otherwise wont work
    start_time = time.time()
    x_size = area[1][1] - area[1][0] + 1
    y_size = area[0][1] - area[0][0] + 1
    filter = g.new_vertex_property("bool")
    for u in g.vertices():
        if pos[id[u]][0] >= area[0][0] and pos[id[u]][0] <= area[0][1] and pos[id[u]][1] >= area[1][0] and pos[id[u]][1] <= area[1][1]:
            filter[u] = True
    g = gt.GraphView(g,filter)
    matrix_1 = np.zeros([y_size*matrix_scaling,x_size*matrix_scaling])
    matrix_2 = np.zeros([y_size*matrix_scaling,x_size*matrix_scaling])
    matrix_3 = np.zeros([y_size*matrix_scaling,x_size*matrix_scaling])
    matrix_4 = np.zeros([y_size*matrix_scaling,x_size*matrix_scaling])
    nodes_in_box = []
    for row in range(y_size*matrix_scaling):
        print(row)
        print("Matrix Setting Time: --- %s minutes ---" % ((time.time() - start_time)/60))
        for col in range(x_size*matrix_scaling):
            y_cord = row/matrix_scaling + area[0][0]
            x_cord = col/matrix_scaling + area[1][0]
            infec_order_list_1 = []
            infec_order_list_2 = []
            infec_order_list_3 = []
            infec_order_list_4 = [] 
            for u in g.iter_vertices():
                if abs(pos[id[u]][0] - y_cord) <= (0.5/matrix_scaling) and abs(pos[id[u]][1] - x_cord) <= (0.5/matrix_scaling) and u not in noinfecs:
                    infec_order_list_1.append(infs_list[0][u][0]) 
                    infec_order_list_2.append(infs_list[1][u][0]) 
                    infec_order_list_3.append(infs_list[2][u][0]) 
                    infec_order_list_4.append(infs_list[3][u][0]) 
            if len(infec_order_list_1) > 0:
                if method == "median":
                    matrix_1[row][col] = np.median(infec_order_list_1)
                    matrix_2[row][col] = np.median(infec_order_list_2)
                    matrix_3[row][col] = np.median(infec_order_list_3)
                    matrix_4[row][col] = np.median(infec_order_list_4)
                if method == "random":
                    matrix_1[row][col] = np.random.choice(infec_order_list_1)
                    matrix_2[row][col] = np.random.choice(infec_order_list_2)
                    matrix_3[row][col] = np.random.choice(infec_order_list_3)
                    matrix_4[row][col] = np.random.choice(infec_order_list_4)
                if method == "first":
                    matrix_1[row][col] = np.min(infec_order_list_1)
                    matrix_2[row][col] = np.min(infec_order_list_2)
                    matrix_3[row][col] = np.min(infec_order_list_3)
                    matrix_4[row][col] = np.min(infec_order_list_4)
            nodes_in_box.append(len(infec_order_list_1))
    matrix_1 = rank_normalize(matrix_1)
    matrix_2 = rank_normalize(matrix_2)
    matrix_3 = rank_normalize(matrix_3)
    matrix_4 = rank_normalize(matrix_4)
    print("Matrix Setting Time: --- %s minutes ---" % ((time.time() - start_time)/60))
    print("Mean nodes in box:", np.mean(nodes_in_box))
    print("Standard Deviation of nodes in box:", np.std(nodes_in_box))
    print("Median nodes in box:", np.median(nodes_in_box))
    return matrix_1,matrix_2,matrix_3,matrix_4

def heatmapsGowalla(g,st,id,pos,origin_index,mList,zList,marker,zoom_areas = None,infection_area=gow_areas["full"],tc_method = 2,method = "first",penalize = False,interpolation = "none",matrix_scaling=1,save_matrices=False):   
    mu_1, mu_2, mu_3, mu_4 = mList[0],mList[1],mList[2],mList[3]
    z_1, z_2, z_3, z_4 = zList[0],zList[1],zList[2],zList[3]
    g, Lrv = L_exponentials(g)
    filter = g.new_vertex_property("bool")
    for u in g.vertices():
        if pos[id[u]][0] >= infection_area[0][0] and pos[id[u]][0] <= infection_area[0][1] and pos[id[u]][1] >= infection_area[1][0] and pos[id[u]][1] <= infection_area[1][1]:
            filter[u] = True
    g = gt.GraphView(g,filter)
    infs1,tc1,noinfecs1 = new_infectionSpread(g,pos,id,st,Lrv,mu_1,zeta = z_1,method = tc_method,origin_index=origin_index,penalize=penalize)
    infs2,tc1,noinfecs2 = new_infectionSpread(g,pos,id,st,Lrv,mu_2,zeta = z_2,method = tc_method,origin_index=origin_index,penalize=penalize)
    infs3,tc1,noinfecs3 = new_infectionSpread(g,pos,id,st,Lrv,mu_3,zeta = z_3,method = tc_method,origin_index=origin_index,penalize=penalize)
    infs4,tc1,noinfecs4 = new_infectionSpread(g,pos,id,st,Lrv,mu_4,zeta = z_4,method = tc_method,origin_index=origin_index,penalize=penalize)
    m_1,m_2,m_3,m_4 = setLattice(g,id,pos,[infs1,infs2,infs3,infs4],noinfecs1,area=infection_area,method=method,matrix_scaling=matrix_scaling)
    if save_matrices:
        k = 0
        for m in [m_1,m_2,m_3,m_4]:
            with open("/home/zylan/python/pickled files/matrix_"+str(marker) + "_" + str(k) + ".pickle",'wb') as handle:
                pickle.dump(m,handle,protocol=pickle.HIGHEST_PROTOCOL)
            k+=1
    colors1 = plt.cm.jet_r(np.linspace(0.,1,255))
    colors2 = plt.cm.Reds(np.linspace(0,1,1))
    colors = np.vstack((colors2,colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)
    regimes = [("explosive",m_1), ("exponential",m_2),("polynomial",m_3),("geometric",m_4)]
    if zoom_areas == None:
        zoom_areas = {"full": [(0,m_1.shape[0]/matrix_scaling),(0,m_1.shape[1]/matrix_scaling)]}
    for area in zoom_areas.keys():
        y_lim_min = zoom_areas[area][0][0]*matrix_scaling
        y_lim_max = zoom_areas[area][0][1]*matrix_scaling
        x_lim_min = zoom_areas[area][1][0]*matrix_scaling
        x_lim_max = zoom_areas[area][1][1]*matrix_scaling
        for reg in regimes: 
            fig = plt.figure(figsize=(11,11))
            plt.axis('off')
            plt.imshow(reg[1],cmap=mymap,interpolation = interpolation)
            plt.xlim(x_lim_min,x_lim_max)
            plt.ylim(y_lim_min,y_lim_max)
            fig.savefig(reg[0]  + str(marker) + "_" + str(area) + '.png')

def selectArea(g,id,pos,x_coords,y_coords):
    nodes = []
    x_min,x_max = x_coords[0] - 46,x_coords[1] - 46
    y_min,y_max = y_coords[0] - 160,y_coords[1] - 160
    for u in g.vertices():
        if pos[id[u]][0] <= x_max and pos[id[u]][0] >= x_min and pos[id[u]][1] <= y_max and pos[id[u]][1] >= y_min:
            nodes.append(int(u))
    print("size of area:", len(nodes))
    return nodes

def findInfectionPathMaxDegree(g,infs,v,origin_index = None):
    if origin_index is None:   
        origin_index = 0
    node_path = []
    prev = v
    while prev != origin_index:       
        node_path.append(prev)
        prev = infs[prev][2]

    node_path.append(origin_index)
    degrees = []

    for u in node_path:
        degrees.append(g.vertex(u).out_degree())
    return node_path

def plotMaxDegrees(mu_min,mu_max,num,sample_amount,marker,area,origin_index):
    start_time = time.time()
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    g,Lrv = L_exponentials(g)
    nodes = selectArea(g,id,pos,area[0],area[1])
    mu_list = np.linspace(mu_min,mu_max,num)
    sampled_nodes = np.random.choice(nodes,sample_amount)
    degrees = []
    for mu in mu_list:
        print(marker + " " + str(mu))
        infs,tc,noninfs = new_infectionSpread(g,pos,id,st,Lrv,mu,origin_index=origin_index)
        degrees_mu = []
        for u in sampled_nodes:
            if g.vertex(u) not in noninfs:
                degrees_mu.append(findInfectionPathMaxDegree(g,infs,u,origin_index=origin_index))
        degrees.append(np.median(degrees_mu))
    plt.figure(figsize=(10,10))
    plt.title("Maximum Degree on " + str(sample_amount) + " Sampled Paths")
    plt.xlabel("mu")
    plt.ylabel("Maximum Degree")
    plt.plot(mu_list,degrees)
    plt.savefig("max_degrees_gowalla" + str(marker) + ".png")
    plt.figure(figsize=(10,10))
    plt.title("Maximum Degree on " + str(sample_amount) + " Sampled Paths")
    plt.xlabel("mu")
    plt.ylabel("log(Maximum Degree)")
    plt.plot(mu_list,np.log(degrees))
    plt.savefig("max_degrees_gowalla_log" + str(marker) + ".png")

def findNodesInArea(x_min,x_max,y_min,y_max,min_degree): 
    with open(path, 'rb') as data:
            g,st,id,pos = pickle.load(data)
    options1 = []
    for key in pos.keys():
        if pos[key][0] > y_min and pos[key][0] < y_max and pos[key][1] > x_min and pos[key][1] < x_max:
            if g.vertex(id[key]).out_degree() > min_degree:
                options1.append(key)
    return options1


def drawNXGraph(area):
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    nodes = selectArea(g,id,pos,area[0],area[1])
    filter = g.new_vertex_property("bool")
    g_nx = nx.Graph()
    nx_pos = {}
    while g_nx.number_of_nodes() < 4000:
        sampled_nodes = np.random.choice(nodes,2000)
        for u in sampled_nodes:
            filter[u] = True
        subgraph = gt.GraphView(g,filter)
        g_nx = nx.Graph()
        for e in subgraph.edges():
            g_nx.add_edge(int(e.source()),int(e.target()))
    for u in subgraph.vertices():
        nx_pos[int(u)] = (pos[id[int(u)]][0],pos[id[int(u)]][1])
    return g_nx,nx_pos

    
def epidemicCurve(mu,zeta,num_runs,infection_area = gow_areas["full"],origin_index = 164,penalize=True,rewire = False):
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    for key in pos.keys():
        pos[key][0] += 46
        pos[key][1] += 160
    filter = g.new_vertex_property("bool")
    for u in g.vertices():
        if pos[id[u]][0] >= infection_area[0][0] and pos[id[u]][0] <= infection_area[0][1] and pos[id[u]][1] >= infection_area[1][0] and pos[id[u]][1] <= infection_area[1][1]:
            filter[u] = True
    g = gt.GraphView(g,filter)
    if rewire:
        gt.random_rewire(g,model='configuration')
    g,L_rv = L_exponentials(g)
    infs,tc,noninfecs = new_infectionSpread(g,pos,id,st,L_rv,mu,zeta,method=2,ratio=1,origin_index=origin_index,penalize=penalize,shifted = True)
    I_max = len(infs)
    I_t = [1,2,3]
    for n in range(2,ceil(np.log(I_max)/np.log(2))):
        I_t.append(2**n)
        if (2**n + 2**(n-1) + 2**(n-2)) <= I_max:
            I_t.append(2**n + 2**(n-2))
            I_t.append(2**n + 2**(n-1))
            I_t.append(2**n + 2**(n-1) + 2**(n-2))
    I_t.append(int((I_t[-1]+I_max)/2))
    I_t.append(I_max)

    all_t = []
    quantile_dict = {"ci": []}

    gow_areas = {"US": [(60,105),(37,90)],"Europe": [(80,115),(145,200)],"full": [(0,115),(0,315)]}
    loc_distribution = {"EU" : [], "US": [], "Other": []}
    for i in range(num_runs):
        print(i)
        t_points = []
        us_amnts = []
        eu_amnts = []
        oth_amnts = []
        g,L_rv = L_exponentials(g)
        infs,tc,noninfecs = new_infectionSpread(g,pos,id,st,L_rv,mu,zeta,method=2,ratio=1,origin_index=origin_index,penalize=penalize,shifted = True)
        infs_list = list(infs)
        for I_amount in I_t:
            vertex = infs_list[I_amount-1]
            t_points.append(infs[vertex][1])
            us_amnt = 0
            eu_amnt = 0
            other_amnt = 0
            for u in infs_list[0:I_amount]:
                if pos[id[u]][0] >= gow_areas["US"][0][0] and pos[id[u]][0] <= gow_areas["US"][0][1] and pos[id[u]][1] >= gow_areas["US"][1][0] and pos[id[u]][1] <= gow_areas["US"][1][1]:
                    us_amnt += 1
                elif pos[id[u]][0] >= gow_areas["Europe"][0][0] and pos[id[u]][0] <= gow_areas["Europe"][0][1] and pos[id[u]][1] >= gow_areas["Europe"][1][0] and pos[id[u]][1] <= gow_areas["Europe"][1][1]:
                    eu_amnt += 1
                else:
                    other_amnt += 1
            eu_amnts.append(eu_amnt/I_amount)
            us_amnts.append(us_amnt/I_amount)
            oth_amnts.append(other_amnt/I_amount)
            


        all_t.append(t_points)
        loc_distribution["EU"].append(eu_amnts)
        loc_distribution["US"].append(us_amnts)
        loc_distribution["Other"].append(oth_amnts)

    median_t = []
    median_eu = []
    median_us = []
    median_other = []
    median_us_abs = []
    median_oth_abs = []

    for k in range(len(I_t)):
        indexed_list = [l[k] for l in all_t]
        indexed_list_eu = [l[k] for l in loc_distribution["EU"]]
        indexed_list_us = [l[k] for l in loc_distribution["US"]]
        indexed_list_oth = [l[k] for l in loc_distribution["Other"]]
        median_t.append(np.median(indexed_list))
        median_eu.append(np.median(indexed_list_eu))
        index = np.where(np.asarray(indexed_list_eu) == np.median(indexed_list_eu))[0][0]
        median_us_abs.append(indexed_list_us[index])
        median_oth_abs.append(indexed_list_oth[index])
        median_us.append(np.median(indexed_list_us))
        median_other.append(np.median(indexed_list_oth))
        quantile_dict["ci"].append((np.percentile(indexed_list,25),np.percentile(indexed_list,75)))


    median_loc_distribution = {"EU" : median_eu, "US": median_us, "Other": median_other, "US_abs": median_us_abs, "Other_abs": median_oth_abs}
    return median_t,I_t,quantile_dict, median_loc_distribution


def degDist(g,marker,log = False): #includes linear regression 
    start_time = time.time()
    degrees = [] 
    degrees_y = []
    degrees_x = []
    degrees_full_x = []
    degrees_full_y = []
    for u in g.vertices(): 
        degrees.append(u.out_degree())
    print(max(degrees))
    for d in range(1,max(degrees)):
        degrees_full_x.append(d)
        degrees_full_y.append(sum(i >= d-1 for i in degrees))
        if d >= floor(e**3) and d <= ceil(e**6):
            degrees_x.append(d)
            degrees_y.append(sum(i >= d-1 for i in degrees))
    degrees_x = np.asarray(np.log(degrees_x))
    degrees_y = np.asarray(np.log(degrees_y))
    degrees_x = degrees_x.reshape((-1,1))
    model = LinearRegression().fit(degrees_x,degrees_y)
    lr_y = []
    for d in degrees_x:
        lr_y.append(model.intercept_ + model.coef_[0]*d)
    if log:
        plt.plot(np.log(degrees_full_x),np.log(degrees_full_y))
        plt.plot(degrees_x,lr_y,color='r')
        plt.xlabel("log(degree)")
        plt.ylabel("log(amount)")
        plt.title("amount of degrees > d, slope of LR = " + str(model.coef_[0]))
        plt.savefig("degreedist_log_LR_" + str(marker) + ".png")
        plt.close()
    if not log:
        plt.plot(degrees_x,degrees_y)
        plt.xlabel("degree")
        plt.ylabel("amount")
        plt.title("amount of degrees > d from degree e^2 to e^7")
        plt.savefig("degreedist" + str(marker) + ".png")
        plt.close()

    print("Time: --- %s seconds ---" % (time.time() - start_time))

def log_ticks(max):
    val = 0
    ticks = [0]
    labels = [0]
    k = 0
    while val < max: 
        val = np.e**(k+1)
        ticks.append(val)
        labels.append(r'$e^{'+str(k+1)+'}$')
        k += 1
    return ticks,labels

def edgeLengthDist(g,id,pos,marker,hist = False):
    deg_filter = g.new_vertex_property("bool")
    for u in g.vertices():
        deg_filter[u] = True
        if g.vertex(u).out_degree() > e**4:
            deg_filter[u] = False
    g = gt.GraphView(g,deg_filter)
    edge_lengths = []
    edges_y = []
    edges_x = []
    edges_full_x = []
    edges_full_y = []
    for edge in g.edges():
        lat_u = pos[id[edge.source()]][0]
        long_u = pos[id[edge.source()]][1]
        lat_v = pos[id[edge.target()]][0]
        long_v = pos[id[edge.target()]][1]
        edge_lengths.append(floor(globeDistance(lat_u,long_u,lat_v,long_v)))
    for l in range(1,ceil(max(edge_lengths))):
        if l <= ceil(e**(5)):
            edges_x.append(l)
            edges_y.append(sum((i == l) for i in edge_lengths))
        edges_full_x.append(l)
        edges_full_y.append(sum((i == l) for i in edge_lengths))
    edges_x = np.asarray(np.log(edges_x))
    edges_y = np.asarray(np.log(edges_y))/np.log(len(edge_lengths))
    edges_x = edges_x.reshape((-1,1))
    model = LinearRegression().fit(edges_x,edges_y)
    lr_y = []
    for d in edges_x:
        lr_y.append(model.intercept_ + model.coef_[0]*d)
    plt.figure(figsize=(12,8))
    edges_full_y = np.asarray(np.log(edges_full_y))/np.log(len(edge_lengths))
    plt.plot(np.log(edges_full_x),edges_full_y)
    plt.plot(edges_x,lr_y,color='r')
    #plt.xscale('log',base=10)
    #plt.xlim((-0.5,5.5))
    plt.ylim((0,1))
    if hist: 
        plt.hist(edge_lengths,bins = 30,log = True, color="forestgreen",alpha=0.7)
    plt.xlabel("length")
    plt.ylabel("Fractional amount")
    plt.title("fraction of edges with length > l (in 1's kms) \n " + r'$d(1-\alpha)$' +  " = " + str(model.coef_[0]*np.log(len(edge_lengths))))
    plt.savefig("edge_length_dist_log_LR_" + str(marker) + ".png")
    plt.close()

def lonlat_to_pixels(lon, lat, map_width, map_height):
    x = (lon + 180) * (map_width / 360)
    lat_rad = np.radians(lat)
    merc_n = np.log(np.tan((np.pi / 4) + (lat_rad / 2)))
    y = (map_height / 2) - (map_width * merc_n / (2 * np.pi))
    return int(x), int(y)

def mapDraw(): #still work in progress because i need to add heatmap to it
    with open(path, 'rb') as data:
            g,st,id,pos = pickle.load(data)
    W = 7000    
    H = 6440
    shifted_x = []
    shifted_y = []
    for key in pos.keys():
        long = pos[key][1]
        lat = pos[key][0]
        x,y = lonlat_to_pixels(long,lat,W,H)
        shifted_x.append(x)
        shifted_y.append(y)
    eu_x1,eu_y1 = lonlat_to_pixels(gow_areas["US"][1][0]-160,gow_areas["US"][0][0]-46,W,H)
    eu_x2,eu_y2 = lonlat_to_pixels(gow_areas["US"][1][1]-160,gow_areas["US"][0][1]-46,W,H)
    map_image_path = '/home/zylan/python/Images Old/map.png'
    image = Image.open(map_image_path)

    dpi = 800
    plt.figure(figsize=(W/dpi, H/dpi),dpi=dpi)
    plt.imshow(image)  
    plt.scatter(shifted_x,shifted_y,s=0.1,c='r')
    plt.xlim((eu_x1,eu_x2))
    plt.ylim((eu_y1,eu_y2))

    plt.axis('off')  
    plt.savefig("map_nodes_US.png")

def moransI(g,pos,id,infs): #is somehow wrong but i dont know why yet
    start_time = time.time()
    tc_mean = (len(infs)+1)/2
    n = len(infs)
    upper_sum = 0
    lower_sum = 0
    w = 0 
    for u in infs.keys():
        diff = infs[u][0]-tc_mean
        lower_sum += diff**2
        lat_u = pos[id[u]][0]
        long_u = pos[id[u]][1]
        phi_1 = lat_u*pi/180
        for e in g.vertex(u).out_edges():
            lat_v = pos[id[e.target()]][0]
            long_v = pos[id[e.target()]][1]
            phi_2 = lat_v*pi/180
            dPhi = (lat_v-lat_u)*pi/180
            dLambda = (long_v-long_u)*pi/180
            a = sin(dPhi/2)*sin(dPhi/2) + cos(phi_1)*cos(phi_2)*sin(dLambda/2)*sin(dLambda/2)
            c = 2*atan2(sqrt(a),sqrt(1-a))
            dist = 6731*c
            if dist <= 0.1: 
                dist = 0.1
        
            weight = 1/dist
            w += weight
            upper_sum += weight*diff*(infs[int(e.target())][0]-tc_mean)
            
    print("Calculating Moran's I: --- %s seconds ---" % (time.time() - start_time))
    return (n/w)*(upper_sum/lower_sum)

def exp_formatter(y, pos):
    if y == 0:
        return "0"
    exponent = int(np.round(np.log10(y)))
    return rf"$10^{{{exponent}}}$"

def epidemic_curves(params,num_runs,file_marker,infection_area=gow_areas["full"],origin_index=164,penalize=True,rewire = False):
    plt.figure(1,figsize=(13,13),clear=True)
    plt.yscale("log",base=10)
    plt.xscale("log",base=10)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
    k = 1
    for mu,zeta,label in params:
        k += 2
        ec_x,y_ec,q_dict,location_dist = epidemicCurve(mu,zeta,num_runs,infection_area=infection_area,origin_index=origin_index,penalize=penalize,rewire=rewire)
        y_ec = np.asarray(y_ec)
        EU_prop_nonlog = np.asarray(location_dist["EU"])
        US_prop_nonlog = np.asarray(location_dist["US"])
        Oth_prop_nonlog = np.asarray(location_dist["Other"])
        US_prop_abs_nonlog = np.asarray(location_dist["US_abs"])
        Oth_prop_abs_nonlog = np.asarray(location_dist["Other_abs"])
        if infection_area == gow_areas["full"]:
            cutoff = e**11
        if infection_area == gow_areas["Europe"]:
            cutoff = e**8
        if infection_area == gow_areas["US"]:
            cutoff = e**(8.5)
        saturation_index = np.where(np.array(y_ec) > cutoff)[0][0]
        y = y_ec[0:saturation_index-1]
        x = ec_x[0:saturation_index-1] 
        EU_prop = EU_prop_nonlog[0:saturation_index-1]
        US_prop = US_prop_abs_nonlog[0:saturation_index-1]
        Oth_prop = Oth_prop_abs_nonlog[0:saturation_index-1]
        plt.figure(k-1,figsize=(15,15),clear=True)
        plt.rc('xtick', labelsize=27)
        plt.rc('ytick', labelsize=27)
        max_q10 = [q[1] for q in q_dict["ci"][0:saturation_index-1]]
        min_q10 = [q[0] for q in q_dict["ci"][0:saturation_index-1]]
        #plt.xlabel("Time",fontsize=18)
        #plt.ylabel("I(t)",fontsize=18)
        plt.yscale("log",base=10)
        ax = plt.gca()
        #ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
        if label == "polynomial" or label == "pure geometric": 
            plt.xscale("log",base=10)
            #ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
        plt.figure(1)
        plt.plot(max_q10,y,linestyle = "dashed", color = "black", alpha = 0.5)
        plt.plot(min_q10,y,linestyle = "dashed", color = "black", alpha = 0.5)
        plt.fill_betweenx(y,max_q10,min_q10,color="#B3C7F7")
        plt.plot(x,y,linewidth=2,label="mu = " + str(mu) + " zeta = " + str(zeta)+ " (" + str(label) +")", color = 'black')
        plt.text(x[-1]-0.5,y[-20],"mu = " + str(mu) + " zeta = " + str(zeta)+ " (" + str(label) +")",rotation=83)
        plt.figure(k-1)
        plt.tick_params(axis='both', which='minor', width=1.2,length=7)   
        plt.rc('xtick', labelsize=27)
        plt.rc('ytick', labelsize=27)
        main_ax = plt.gca()
        plt.plot(x,y,color='black')
        #plt.plot(max_q10,y,linestyle = "dashed", color = "black", alpha = 0.5)
        #plt.plot(min_q10,y,linestyle = "dashed", color = "black", alpha = 0.5 )
        plt.fill_betweenx(y,max_q10,min_q10,color="#B3C7F7",label = "30%")
        plt.grid(visible=True)
        #plt.xlabel("Time",fontsize=18)
        #plt.ylabel("I(t)",fontsize=18)
        inset_ax2 = plt.axes([0.62,0.15,0.25,0.25])
        plt.sca(inset_ax2)
        ax = plt.gca()
        if label == "polynomial" or label == "pure geometric":
            plt.xscale("log",base=10)
            #ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
        plt.grid(visible=True)
        plt.stackplot(x,EU_prop*100,US_prop*100,Oth_prop*100,labels=["EU","US","Other"],colors=["blue","red","grey"])
        if label == "polynomial" or label == "pure geometric":
            inset_ax1 = plt.axes([0.15,0.47,0.28,0.43])
            plt.sca(inset_ax1)
            x_lr = np.array(np.log10(x))
            y_lr = np.array(np.log10(y))
            lr_start = np.where(y_lr>=2.17)[0][0] 
            lr_end = np.where(y_lr>=3.70)[0][0]
            x_lr=x_lr[lr_start:lr_end]
            y_lr=y_lr[lr_start:lr_end]
            x_lr = x_lr.reshape((-1,1))
            model = LinearRegression().fit(x_lr,y_lr)
            lr_y = []
            for d in x_lr:
                lr_y.append(model.intercept_ + model.coef_[0]*d)
            lr_y = np.asarray(lr_y)
            plt.plot(x,y,linewidth=4,label="mu = " + str(mu) + "zeta = " + str(zeta)+ " (" + str(label) +")")
            plt.plot(10**x_lr,10**lr_y,color="red",label="Linear Regression with slope\n" + str(model.coef_[0]),linewidth=3,marker='s',markersize = 8)
            #plt.grid(visible=True)
            #plt.ylabel("log(I(t))")
            #plt.xlabel("log(t)")                         
            plt.title(r"$2\psi = $" + str(round(model.coef_[0],2)), y = 0, fontsize=23)
            #plt.legend()
            ax = plt.gca()
            plt.grid(visible=True)
            plt.xlim((10**x_lr[0])[0] - 10**0.2,(10**x_lr[-1])[0] + 10**0.2)
            plt.ylim(10**2.17,10**3.69)
            plt.xscale("log",base=10)
            plt.yscale("log",base=10)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            #ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
            #ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter)) 
            xlim_inset = inset_ax1.get_xlim()
            ylim_inset = inset_ax1.get_ylim()
            bottom_right_corner = (xlim_inset[1],ylim_inset[0])
            top_right_corner = (xlim_inset[1],ylim_inset[1])
            plt.sca(main_ax)
            con = ConnectionPatch(xyA=((10**x_lr[0])[0], (10**y_lr[0])), coordsA=main_ax.transData,
                xyB=bottom_right_corner, coordsB=inset_ax1.transData,
                color="grey", linewidth=2, linestyle="--") 
            plt.gcf().add_artist(con)
            con2 = ConnectionPatch(xyA=((10**x_lr[-1])[0], (10**y_lr[-1])), coordsA=main_ax.transData,
                xyB=top_right_corner, coordsB=inset_ax1.transData,
                color="grey", linewidth=2, linestyle="--") 
            plt.gcf().add_artist(con2)
        #plt.title("Mu = " + str(mu) + " Zeta = " + str(zeta) + "(" + str(label) +")\n Median taken over 50 runs with 10/30% percentiles\n Cut off when I(t) > e**10")
        #plt.legend()
        plt.sca(main_ax)
        plt.rc('xtick', labelsize=27)
        plt.rc('ytick', labelsize=27)
        plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_"+str(file_marker)+".png")
        # plt.figure(2*k-1,figsize=(12,12),clear=True)
        # ax = plt.gca()
        # x_prop = x
        # if label == "polynomial" or label == "pure geometric": 
        #     x_prop = np.log(x)
        # plt.grid(visible=True)
        # #plt.ylabel("I(t)")
        # y_prop = np.asarray(np.log(y))
        # plt.axis('off')
        # plt.plot(x_prop,y_prop,color='black')
        # plt.fill_between(x_prop,0,EU_prop*y_prop,color='blue',alpha=0.7,label = "Proportion of EU Infected")
        # plt.fill_between(x_prop,EU_prop*y_prop,(US_prop+EU_prop)*y_prop,color='red',alpha = 0.7,label = "Proportion of US Residents Infected")
        # plt.fill_between(x_prop,(US_prop+EU_prop)*y_prop,(US_prop+Oth_prop+EU_prop)*y_prop,color='grey',alpha = 0.7,label = "Proportion of Other Residents Infected")
        # plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_proportions"+str(file_marker)+".png", transparent=True)
        # plt.figure(figsize=(12,12),clear=True)
        # ax = plt.gca()
        # plt.xscale("log",base=10)
        # #ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
        # plt.grid(visible=True)
        # plt.ylabel("Percentage")
        # plt.title("Mu = " + str(mu) + " Zeta = " + str(zeta) + "(" + str(label) +")\n Median taken over 50 runs")
        # plt.stackplot(ec_x,EU_prop_nonlog*100,US_prop_abs_nonlog*100,Oth_prop_abs_nonlog*100,labels=["EU","US","Other"],colors=["blue","red","grey"])
        # plt.legend()
        # plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_evolving"+str(file_marker)+".png")
        # plt.figure(figsize=(12,12),clear=True)
        # ax = plt.gca()
        # if label == "polynomial" or label == "pure geometric": 
        #     plt.xscale("log",base=10)
        #     #ax.xaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
        # plt.grid(visible=True)
        # plt.ylabel("Absolute Number of Infected")
        # plt.plot(ec_x[0:saturation_index-1],(EU_prop_nonlog*y_ec)[0:saturation_index-1],label="Absolute number of infections of EU", color='blue',linewidth=2)
        # plt.plot(ec_x[0:saturation_index-1],(US_prop_nonlog*y_ec)[0:saturation_index-1],label="Absolute number of infections of US", color='red',linewidth=2)
        # plt.plot(ec_x[0:saturation_index-1],(Oth_prop_nonlog*y_ec)[0:saturation_index-1],label="Absolute number of other infections", color='grey',linewidth=2)
        # plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_absolutes"+str(file_marker)+".png")
    plt.figure(1)
    plt.xlabel("log(t)")
    plt.ylabel("log(I(t))")
    plt.grid(visible=True)
    plt.title("Epidemic curves\n Median taken over 51 runs\nGowalla dataset")
    #plt.legend()
    plt.savefig("epidemic_curves_together_" + str(file_marker)+".png")
    plt.close()

def typicalCostDistancesGIRG(n,r_list, amount_runs):
    means = {}
    for r in r_list:
        means[r] = []
    for i in range(amount_runs):
        g,pos,st,w,distL = bf.genGirg(n,2,2.8,1.17)
        g, L_rv = bf.L_exponentials(g)
        infs,tc,noninfecs = bf.infectionSpread(g,st,w,L_rv,1,2,vertex_set="PPP",pos=pos)
        for r in r_list:
            means[r].append(bf.typicalCostDistance(g,pos,infs,noninfecs,origin_index=0,r=r))
    plt.figure(figsize=(12,12))
    plt.xlabel("run #")
    plt.ylabel("Estimated eta")
    plt.hlines(0.34,0,amount_runs,colors='r')
    for r in r_list: 
        mean_points = np.asarray(means[r])
        mean_points = np.log(mean_points)/np.log(r)
        print(r,mean_points)
        plt.plot(mean_points,label="r = " + str(r))
    plt.legend()
    plt.title("Estimated eta from typical cost distance on GIRG with 50k nodes")
    plt.savefig("Costdistances_eta_3.png")


def typicalCostDistances1D(max_n,d,mu,zeta,est_eta,filename):
    etas = []
    n_list = np.linspace(10000,max_n,5)
    n_list = np.insert(n_list,0,1000,axis = 0)
    for n in n_list: 
        g,pos,st,w,distL = bf.genGirg(n,d,2.8,1.17)
        g, L_rv = bf.L_exponentials(g)
        infs,tc,noninfecs = bf.infectionSpread(g,st,w,L_rv,mu=mu,zeta=zeta,d=d,vertex_set="PPP",pos=pos)
        r = 0.35*(g.num_vertices()**(1/d))
        eta = np.log(bf.typicalCostDistance1D(g,pos,infs,noninfecs,origin_index=0,r=r,d=d))/np.log(r)
        print(n,eta)
        etas.append(eta)
    plt.figure(figsize=(12,12))
    plt.xlabel("n")
    plt.ylabel("Estimated eta")
    plt.hlines(est_eta,0,max_n,colors='r')
    plt.plot(n_list,etas)
    print(n_list,etas)
    plt.savefig(str(filename))

def edgeCostsBoxes(max_n,d,mu,zeta,est_eta,gamma,filename):
    etas = []
    n_list = np.linspace(10000,max_n,15)
    n_list = np.insert(n_list,0,1000,axis = 0)
    for n in n_list: 
        g,pos,st,w,distL = bf.genGirg(n,d,2.8,1.17)
        g, L_rv = bf.L_exponentials(g)
        eta = bf.edgeCostBetweenBoxes(g,st,w,pos,L_rv,d,mu,zeta,gamma)
        print(n,eta)
        etas.append(eta)
    plt.figure(figsize=(12,12))
    plt.xlabel("n")
    plt.ylabel("Estimated eta")
    plt.title("gamma =" + str(gamma) + "estimated eta from cheapest edge between boxes")
    plt.hlines(est_eta,0,max_n,colors='r')
    plt.plot(n_list,etas)
    print(n_list,etas)
    plt.savefig(str(filename))

edgeCostsBoxes(500000,1,1,1.17,0.34,0.90,"boxesCost_1d_0.9.png")
edgeCostsBoxes(500000,2,1,2,0.34,0.90,"boxesCost_2d_0.9.png")



typicalCostDistances1D(500000,1,1,0.83,0.1,"1D_eta_estimates_loweta")
typicalCostDistances1D(250000,2,1,1.76,0.1,"2D_eta_estimates_loweta")
typicalCostDistances1D(250000,3,1,2.59,0.1,"3D_eta_estimates_loweta")
# typicalCostDistances1D(500000,1,1,1.63,0.8,"1D_eta_estimates_higheta")
# typicalCostDistances1D(250000,2,1,2.46,0.8,"2D_eta_estimates_higheta")
# typicalCostDistances1D(250000,3,1,3.29,0.8,"3D_eta_estimates_higheta")
# typicalCostDistances1D(500000,1,1,1.33,0.5,"1D_eta_estimates_mideta")
# typicalCostDistances1D(250000,2,1,2.16,0.5,"2D_eta_estimates_mideta")
# typicalCostDistances1D(250000,3,1,2.99,0.5,"3D_eta_estimates_mideta")

#typicalCostDistancesGIRG(1500000,[400,600,800],1)

        

# gt.random_rewire(g,model='configuration')

# edgeLengthDist(g,id,pos,"rewired_equal",hist=False)


# for key in pos.keys():
#     pos[key][0] += 46
#     pos[key][1] += 160

# heatmapsGowalla(g,st,id,pos,origin_index=164,mList=[0,1,1,1],zList=[0,1,2,3],marker="full_scaled",zoom_areas = gow_areas,infection_area=gow_areas["full"],
#               tc_method = 2,method = "first",penalize = False,interpolation = "none",matrix_scaling=3,save_matrices=True)

#df.four_heatmaps_same_graph(500,2.8,1.17,[0,1,1,1],z_list=[0,1,2,3],vertex_set="Z2",marker ="SFP_751",method=4,penalize=False)
# params1 = [(0,0,"explosive"),(1,1,"exponential"),(1,2,"polynomial"),(1,3,"pure geometric")]
#params2 = [(1,2,"polynomial")]
# epidemic_curves(params1,51,file_marker="nuremberg_full",infection_area=gow_areas["full"],origin_index=164,penalize=False,rewire=False)
#bf.epidemic_curves(200000,1.17,2.78,params2,1,"girg")
# g,pos,st,w = bf.genLattice(350,2,2.77,1.15)
# g,Lrv = bf.L_exponentials(g)
# infs1,tc1,noninfs1 = bf.infectionSpread(g,st,w,Lrv,0,0,vertex_set="Z2",method=4,pos=pos)
# infs2,tc2,noninfs2 = bf.infectionSpread(g,st,w,Lrv,1,1,vertex_set="Z2",method=4,pos=pos)
# infs3,tc3,noninfs3 = bf.infectionSpread(g,st,w,Lrv,1,2,vertex_set="Z2",method=4,pos=pos)
# infs4,tc4,noninfs4 = bf.infectionSpread(g,st,w,Lrv,1,3,vertex_set="Z2",method=4,pos=pos)
# m1 = bf.heatmapMatrix(infs1,350)
# m2 = bf.heatmapMatrix(infs2,350)
# m3 = bf.heatmapMatrix(infs3,350)
# m4 = bf.heatmapMatrix(infs4,350)
# m1 = rank_normalize(m1)
# m2 = rank_normalize(m2)
# m3 = rank_normalize(m3)
# m4 = rank_normalize(m4)
# plt.figure(figsize=(12,12))
# plt.axis('off')
# bf.draw(m1,title ="",interpolation="none")
# plt.savefig("explosive_girg_gowalla.png")
# plt.figure(figsize=(12,12))
# plt.axis('off')
# bf.draw(m2,title ="",interpolation="none")
# plt.savefig("exponential_girg_gowalla.png")
# plt.figure(figsize=(12,12))
# plt.axis('off')
# bf.draw(m3,title ="",interpolation="none")
# plt.savefig("polynomial_girg_gowalla.png")
# plt.figure(figsize=(12,12))
# plt.axis('off')
# bf.draw(m4,title ="",interpolation="none")
# plt.savefig("geometric_girg_gowalla.png")

# # node index 316 for a point in the US (sorta east coast
# # gow_areas = {"US": [(60,105),(37,90)],"Europe": [(80,115),(145,200)],"full": [(0,115),(0,315)]}
# # 21 is LA, 15 is NYC, 164 is give or take Nuremburg, 72 is london, 198 is Rotterdam
# # pos[1] is latitude, pos[0] is longitude'
