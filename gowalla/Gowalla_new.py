import girg_sampling.girgs as gs
import graph_tool.all as gt
import numpy as np 
from math import *
import time
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from line_profiler import LineProfiler
import pickle
import networkx as nx 
from sklearn.linear_model import LinearRegression
from PIL import Image

path = '/home/zbenj/.local/python/GIRGs/Datasets/Gowalla/gow_graph_mode.pickle'


'''

gow_checkins_txt = open("/root/python/Gowalla_totalCheckins.txt",'r')
gow_checkins_string = gow_checkins_txt.readlines()
check_in_data = {}
for x in gow_checkins_string:
    x = x.replace("\n","")
    x_split = x.split("\t")
    if int(x_split[0]) not in check_in_data.keys():
        check_in_data[int(x_split[0])] = [[round(float(x_split[2]),2)],[round(float(x_split[3]),2)]]
    if int(x_split[0]) in check_in_data.keys():
        check_in_data[int(x_split[0])][0].append(round(float(x_split[2]),2))
        check_in_data[int(x_split[0])][1].append(round(float(x_split[3]),2))
gow_checkins_txt.close()


pos = {}
for key in check_in_data.keys():
    print(key)
    values_x, counts_x = np.unique(check_in_data[key][0], return_counts = True)
    index = np.where(check_in_data[key][0] == values_x[np.argmax(counts_x)])[0][0]
    pos[key] = [check_in_data[key][0][index],check_in_data[key][1][index]]

with open('gow_pos_mode.pickle','wb') as outfile:
    pickle.dump(pos,outfile,protocol=pickle.HIGHEST_PROTOCOL)
breakpoint()

pos_r = {}
for key in check_in_data.keys():
    index = np.random.randint(0,len(check_in_data[key][0]))
    pos_r[key] = [check_in_data[key][0][index],check_in_data[key][1][index]]

with open('gow_pos_random.pickle','wb') as outfile:
    pickle.dump(pos_r,outfile,protocol=pickle.HIGHEST_PROTOCOL)


with open('gow_pos_random.pickle','rb') as data:
    gow_pos_dict = pickle.load(data)

gow_edges_txt = open("/root/python/Gowalla_edges.txt",'r')
gow_edges_string = gow_edges_txt.readlines()
gow_edge_list= []
for x in gow_edges_string:
    x = x.replace("\n","")
    if int(x.split('\t')[0]) in gow_pos_dict.keys() and int(x.split('\t')[1]) in gow_pos_dict.keys():
        gow_edge_list.append((int(x.split('\t')[0]),int(x.split('\t')[1])))
gow_edges_txt.close()

def setGraph(edges,pos_dict):
    start_time = time.time()
    g = gt.Graph(directed = False)
    g.add_vertex(196590)
    status = g.new_vertex_property("int")
    id = g.new_vertex_property("int")
    del_list = []
    for u in g.vertices():
        id[u] = int(u)
        if id[u] not in pos_dict.keys():
            del_list.append(u)
        status[u] = 0
    g.add_edge_list(gow_edge_list)
    g.remove_vertex(del_list,fast = True)        
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return g,status,id

g,st,id = setGraph(gow_edge_list,gow_pos_dict)
breakpoint()
with open('gow_graph_mode_random.pickle','wb') as outfile:
    pickle.dump([g,st,id,gow_pos_dict],outfile,protocol=pickle.HIGHEST_PROTOCOL)
    
'''

areas = {"US": [(60,105),(37,90)],"Europe": [(80,115),(145,200)]}

def L_exponentials(g):
    L_rv = g.new_edge_property("float")
    for e in g.edges():
        L_rv[e] = np.random.exponential(1)
    return g,L_rv

def new_infectionSpread(g,pos,id,status,L_rv,mu,zeta = 0,method = 1,ratio = 1,origin_index = None, penalize = False): 
    start_time = time.time()
    num_vertices = g.num_vertices()
    cutoff = ratio*num_vertices
    trans_cost = g.new_edge_property("float")
    penalty = 0 
    if penalize:
        penalty = (2/3)*gt.vertex_average(g,"total")[0]
    i = 0
    for e in g.edges():
        i += 1
        if i % 1000000 == 0:
            print(i)
        match method:
            case 1: 
                trans_cost[e] = L_rv[e]*(sqrt((pos[id[e.source()]][0] - pos[id[e.target()]][0])**2 + (pos[id[e.source()]][1] - pos[id[e.target()]][1])**2))**mu
            case 2: 
                trans_cost[e] = L_rv[e]*((sqrt((pos[id[e.source()]][0] - pos[id[e.target()]][0])**2 + (pos[id[e.source()]][1] - pos[id[e.target()]][1])**2))**mu)*((max(1,e.source().out_degree()-penalty)*(max(1,e.target().out_degree()-penalty)))**zeta)
    #print("Edge Cost Simulation: --- %s seconds ---" % (time.time() - start_time))
    if origin_index is None:    
        origin_index = 316
    inf_nodes = {origin_index: [0,0,-1]} #INDEX OF INFECTED, TIME PASSED, NODE WHICH HAS INFECTED CURRENT NODE
    status[origin_index] = 1 # 1 = infected
    time_passed = 0
    steps = 0
    Q = PriorityQueue()
    for e in g.vertex(origin_index).out_edges():
        Q.put((trans_cost[e],int(e.target()),int(g.vertex(origin_index))))
    #print("STARTING QUEUE OF 0", Q.queue)
    k = 0
    while not Q.empty():
        if len(inf_nodes) > cutoff:
            break
        #print("TIME PASSED AT STEP ", steps, " IS ", time_passed)
        next_inf = Q.get()
        steps += 1
        while status[next_inf[1]] == 0:  
            k += 1          
            #print("NEXT INFECTED PAIR IS ", next_inf)
            time_passed = next_inf[0]
            #print("-----LIST OF NEW EDGES------")
            for e in g.vertex(next_inf[1]).out_edges():
                #print(e, trans_cost[e])
                if status[e.target()] == 1:
                    continue
                Q.put((trans_cost[e]+time_passed,int(e.target()),int(g.vertex(next_inf[1]))))
            #print("------END LIST--------")
            #print("NEW QUEUE IS ", Q.queue)
            inf_nodes[next_inf[1]] = [k, time_passed,next_inf[2]]
            status[next_inf[1]] = 1
    #print(len(inf_nodes))
    not_infected_vertices = []
    for u in g.vertices():
            if status[u] == 0:
                not_infected_vertices.append(u)
            status[u] = 0
    #print("Infection Simulation: --- %s seconds ---" % (time.time() - start_time))
    #print("Steps = ", steps)
    return inf_nodes, trans_cost, not_infected_vertices

def sortDistances(g,pos,id,origin = 6700):
    start_time = time.time()
    dist_list = []
    for u in g.vertices():
        dist = sqrt( (pos[id[int(u)]][0]-pos[id[int(origin)]][0])**2  + (pos[id[int(u)]][1]-pos[id[int(origin)]][1])**2 )
        dist_list.append([dist,u])
    print("Dist: --- %s seconds ---" % (time.time() - start_time))
    return dist_list


def findInfectionPath(g,pos,id,infs,tc,v,origin_index = None):
    if origin_index is None:    
        origin_index = 6700
    edge_path = []
    node_path = []
    prev = v
    while prev != origin_index:
        dist = sqrt( (pos[id[prev]][0] - pos[id[infs[prev][2]]][0])**2 + (pos[id[prev]][1] - pos[id[infs[prev][2]]][1]) **2)                        
        cost = tc[g.edge(infs[prev][2],prev)]
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


def checkAlpha(version):
    with open('gow_graph_mode.pickle', 'rb') as data:
        g,st,id,pos = pickle.load(data)
    g,Lrv = L_exponentials(g)
    x1,y1 = radiusCoords(g,st,id,pos,Lrv,0)
    x2,y2 = radiusCoords(g,st,id,pos,Lrv,2)
    fig, axis = plt.subplots(1,2, figsize=(15,11))
    axis[0].plot(x1,y1)
    axis[0].set_title("mu = 0")
    axis[0].set_xlabel("|x|")
    axis[0].set_ylabel("Dc(0,x)")
    axis[1].plot(x2,y2)
    axis[1].set_xlabel("|x|")
    axis[1].set_ylabel("Dc(0,x)")
    axis[1].set_title("mu = 2")
    fig.savefig("alphaCheck_" + str(version) + ".png")
    fig2, axis2 = plt.subplots(1,2, figsize=(15,11))
    axis2[0].plot(x1,y1)
    axis2[0].set_xlabel("|x|")
    axis2[0].set_ylabel("Dc(0,x)")
    axis2[1].plot(x2,np.log10(y2))
    axis2[1].set_xlabel("|x|")
    axis2[1].set_ylabel("log(Dc(0,x))")
    axis2[0].set_title("mu = 0")
    axis2[1].set_title("mu = 2")
    fig2.savefig("alphaCheck_log_" + str(version) + ".png")


def moransIndex(g,infs,lim): #DistanceDecay currently only works for k = 1 (rook definition of neighbors)
    start_time = time.time()
    tc_mean = (g.num_vertices() + 1)/2
    big_N = g.num_vertices()
    big_W = 0
    first_sum = 0
    bottom_sum = 0
    for u in g.vertices():
        u_neighbors = []
        second_sum = 0
        ind_u = infs[int(u)][0]  
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
        bottom_sum += (ind_u-tc_mean)**2
        for v in u_neighbors:
            weight = 1
            big_W += weight
            second_sum += weight*(ind_u - tc_mean)*(infs[int(v)][0] - tc_mean)
        first_sum += second_sum
    morans_Index = (big_N/big_W)*(first_sum/bottom_sum) 
    print("Calculating Moran's I: --- %s seconds ---" % (time.time() - start_time))
    return morans_Index


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


# Min max X-cords = -45.91, 70.08
# Min Max Y-cords = -159.67, 176.92

def setLattice(origin_index,mu,zeta = 0,tc_method = 1,method = "median"):
    start_time = time.time()
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    reverse_ids = {}
    for u in g.vertices():
        reverse_ids[id[u]] = int(u)
    matrix = np.zeros([116,337])
    g, Lrv = L_exponentials(g)
    infs,tc,noinfecs = new_infectionSpread(g,pos,id,st,Lrv,mu,zeta = zeta,method = tc_method,origin_index=origin_index)
    for i in pos.keys():
        pos[i][0] += 46
        pos[i][1] += 160
    for row in range(116):
        for col in range(337):
            infec_order_list = []
            for i in pos.keys():
                if abs(pos[i][0] - row) <= 0.499 and abs(pos[i][1] - col) <= 0.499 and reverse_ids[i] not in noinfecs:
                    infec_order_list.append(infs[reverse_ids[i]][0])
            if len(infec_order_list) > 0:
                if method == "median":
                    matrix[row][col] = np.median(infec_order_list)
                if method == "random":
                    matrix[row][col] = np.random.choice(infec_order_list)
    print("Matrix Setting Time: --- %s minutes ---" % (time.time() - start_time))
    return matrix

def heatmapsGowalla(origin_index,mList,zList,marker,areas = {},tc_method = 1,method = "median"):
    mu_1, mu_2, mu_3, mu_4 = mList[0],mList[1],mList[2],mList[3]
    z_1, z_2, z_3, z_4 = zList[0],zList[1],zList[2],zList[3]
    m_1 = setLattice(origin_index,mu_1,zeta = z_1,tc_method = tc_method, method = method)
    m_2 = setLattice(origin_index,mu_2,zeta = z_2,tc_method = tc_method, method = method)
    m_3 = setLattice(origin_index,mu_3,zeta = z_3,tc_method = tc_method, method = method)
    m_4 = setLattice(origin_index,mu_4,zeta = z_4,tc_method = tc_method, method = method)
    colors1 = plt.cm.jet(np.linspace(0.,1,255))
    colors2 = plt.cm.Reds(np.linspace(0,1,1))
    colors = np.vstack((colors2,colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)
    areas["full"] = [(0,115),(0,315)]
    for area in areas.keys():
        y_lim_min = areas[area][0][0]
        y_lim_max = areas[area][0][1]
        x_lim_min = areas[area][1][0]
        x_lim_max = areas[area][1][1]
        
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(2,2,1)
        plt.imshow(m_1,cmap=mymap,interpolation = 'spline16')
        plt.xlim(x_lim_min,x_lim_max)
        plt.ylim(y_lim_min,y_lim_max)
        plt.title("mu = " + str(mu_1) + " zeta = " + str(z_1))
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('linear'  +str(marker) + '.png',bbox_inches = extent1.expanded(1.3,1.4))

        ax2 = fig.add_subplot(2,2,2)
        plt.imshow(m_2,cmap=mymap,interpolation = 'spline16')
        plt.xlim(x_lim_min,x_lim_max)
        plt.ylim(y_lim_min,y_lim_max)
        plt.title("mu = " + str(mu_2) + " zeta = " + str(z_2))
        extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('polynomial'  +str(marker) + '.png',bbox_inches = extent2.expanded(1.3,1.4))

        ax3 = fig.add_subplot(2,2,3)
        plt.imshow(m_3,cmap=mymap,interpolation = 'spline16')
        plt.xlim(x_lim_min,x_lim_max)
        plt.ylim(y_lim_min,y_lim_max)      
        plt.title("mu = " + str(mu_3) + " zeta = " + str(z_3))
        extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('polylog'  +str(marker) + '.png',bbox_inches = extent3.expanded(1.3,1.27))

        ax4 = fig.add_subplot(2,2,4)
        plt.imshow(m_4,cmap=mymap,interpolation = 'spline16')
        plt.xlim(x_lim_min,x_lim_max)
        plt.ylim(y_lim_min,y_lim_max)       
        plt.title("mu = " + str(mu_4) + " zeta = " + str(z_4))
        extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('explosive'  +str(marker) + '.png',bbox_inches = extent4.expanded(1.3,1.27))
        fig.savefig('allHeatmaps'  + str(marker) + area + '.png')

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

'''
with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
options1 = []
options2 = []
for key in pos.keys():
    if pos[key][0] > (80-46) and pos[key][0] < (84-46) and pos[key][1] > (72-160) and pos[key][1] < (75-160):
        if g.vertex(id[key]).out_degree() > 10:
            options1.append(key)
    if pos[key][0] > (95-46) and pos[key][0] < (100-46) and pos[key][1] > (170-160) and pos[key][1] < (175-160):
        if g.vertex(id[key]).out_degree() > 10:
            options2.append(key)
print(options1[0])
print(options2[0])
'''

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
    
def saveFig(area):
    fig = plt.figure(figsize=(11,11))
    nx_g,nx_pos = drawNXGraph(areas[area])
    nx.draw_networkx_nodes(nx_g,pos=nx_pos,node_size=10,node_color="forestgreen")
    nx.draw_networkx_edges(nx_g, pos=nx_pos, edge_color="black")
    print(nx_g.number_of_nodes(),nx_g.number_of_edges())
    plt.savefig("US_500v.png")

def epidemicCurve(mu,zeta,origin_index):
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    g,Lrv = L_exponentials(g)
    infs,tc,noninfs = new_infectionSpread(g,pos,id,st,Lrv,mu,zeta=zeta,origin_index=origin_index)
    t_max = infs[list(infs)[-1]][1]
    t_points = np.linspace(0,t_max,100)
    I_t = []
    for t in t_points:
        I_t.append(sum(1 for v in infs.values() if v[1] <= t))
    return t_points,I_t

def show(mList,zList,origin_index):
    mu_1, mu_2, mu_3, mu_4 = mList[0],mList[1],mList[2],mList[3]
    z_1, z_2, z_3, z_4 = zList[0],zList[1],zList[2],zList[3]
    x1,y1 = epidemicCurve(mu_1,z_1,origin_index)
    x2,y2 = epidemicCurve(mu_2,z_2,origin_index)
    x3,y3 = epidemicCurve(mu_3,z_3,origin_index)
    x4,y4 = epidemicCurve(mu_4,z_4,origin_index)
    plt.plot(np.log(x1),np.log(y1), label = ("mu = " + str(mu_1) +" zeta = " + str(z_1)))
    plt.show()
    plt.plot(np.log(x2),np.log(y2), label = ("mu = " + str(mu_2) +" zeta = " + str(z_2)))
    plt.show()
    plt.plot(np.log(x3),np.log(y3), label = ("mu = " + str(mu_3) +" zeta = " + str(z_3)))
    plt.plot(np.log(x4),np.log(y4), label = ("mu = " + str(mu_4) +" zeta = " + str(z_4)))
    plt.legend()
    plt.show()
    plt.savefig("waaaaaa.png")


def degDist(marker,log = False): 
    start_time = time.time()
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    degrees = [] 
    degrees_y = []
    degrees_x = []
    for u in g.vertices(): 
        degrees.append(u.out_degree())
    for d in range(1,max(degrees)):
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
        plt.plot(degrees_x,degrees_y)
        plt.plot(degrees_x,lr_y,color='r')
        plt.xlabel("log(degree)")
        plt.ylabel("log(amount)")
        plt.title("amount of degrees > d, slope of LR = " + str(model.coef_[0]))
        plt.savefig("degreedist_log" + str(marker) + ".png")
    if not log:
        plt.plot(degrees_x,degrees_y)
        plt.xlabel("degree")
        plt.ylabel("amount")
        plt.title("amount of degrees > d")
        plt.savefig("degreedist" + str(marker) + ".png")

    print("Time: --- %s minutes ---" % (time.time() - start_time))



def edgeLengthDist():
    with open(path, 'rb') as data:
        g,st,id,pos = pickle.load(data)
    edge_lengths = []
    edges_y = []
    edges_x = []
    for edge in g.edges():
        if edge.source().out_degree() < e**4 and edge.target().out_degree() < e**4:
            edge_lengths.append(sqrt((pos[id[edge.source()]][0] - pos[id[edge.target()]][0])**2 + (pos[id[edge.source()]][1] - pos[id[edge.target()]][1])**2))
    for l in range(0,ceil(max(edge_lengths))):
        edges_x.append(l)
        edges_y.append(sum((i >= l) for i in edge_lengths))
    plt.plot(np.log(edges_x),np.log(edges_y))
    plt.xlabel("log(length)")
    plt.ylabel("log(amount)")
    plt.title("amount of edges with length > l with removal of high weight vertices (> e^4)")
    plt.savefig("edge_length_dist_log_cml_pruned.png")


def mapDraw():
    with open('gow_graph_mode_mode.pickle', 'rb') as data:
            g,st,id,pos = pickle.load(data)

    W = 7000    
    H = 6352
    shifted_pos = {}
    shifted_x = []
    shifted_y = []
    for key in pos.keys():
        long = pos[key][1]
        lat = pos[key][0]
        sec = 1/cos(lat*pi/180)
        y = (1-np.log(tan((pi/4)+((lat*pi)/180)/2))/pi)/2
        y_other = (6352/2) + (lat*sec*6352)/(2*210)
        new_y = y*H
        sign = 1
        if new_y < (6352/2):
            sign = -1
        diff = abs(new_y-(6352/2))
        scaled_y = diff*sign*(1+(sec-1)/8) + (6352/2)
        shifted_pos[key] = [scaled_y,((long+180)/360)*W]
        shifted_x.append(shifted_pos[key][1])
        shifted_y.append(shifted_pos[key][0])

    map_image_path = '/root/python/map.png'
    image = Image.open(map_image_path)

    dpi = 100
    plt.figure(figsize=(W/dpi, H/dpi),dpi=dpi)
    plt.imshow(image)  
    plt.scatter(shifted_x,shifted_y,s=0.3,c='r')


    plt.axis('off')  
    plt.savefig("map_nodes.png")

def moransI(g,pos,id,infs,noinfecs):
    start_time = time.time()
    tc_mean = (g.num_vertices()+1)/2
    n = g.num_vertices()
    upper_sum = 0
    lower_sum = 0
    w = 0 
    for u in g.vertices():
        print(int(u))
        diff = -tc_mean
        if u not in noinfecs:
            diff = infs[u][0]-tc_mean
        lower_sum += diff**2
        for e in g.vertex(u).out_edges():
            dist = sqrt((pos[id[u]][0]-pos[id[e.target()]][0])**2 +(pos[id[u]][1]-pos[id[e.target()]][1])**2)
            if dist == 0: 
                dist = 0.1
            weight = 1/dist
            w += weight
            if e.target() in noinfecs:
                upper_sum += weight*diff*(-tc_mean)
                continue
            upper_sum += weight*diff*(infs[int(e.target())][0]-tc_mean)
            
    print("Calculating Moran's I: --- %s seconds ---" % (time.time() - start_time))
    return (n/w)*(upper_sum/lower_sum)

with open(path, 'rb') as data:
    g,st,id,pos = pickle.load(data)

g, lrv = L_exponentials(g)
for mu in [0,7,1.1,1.7,1.9]:
    infs,tc,noninfecs = new_infectionSpread(g,pos,id,st,lrv,mu)
    print(moransI(g,pos,id,infs,noninfecs))




    








#we find 316 for first big cluster and 164 for second big cluster with this

#plotMaxDegrees(0,3,100,5000,"_europe",areas["Europe"],origin_index=164)
#plotMaxDegrees(0,3,100,5000,"_us",areas["US"],origin_index=316)

#heatmapsGowalla(164,[0.2,0.4,0.7,1],[0.1,0.4,0.7,1],"_mixed_pen_US_random",areas = areas,method = "random")
#heatmapsGowalla(164,[1,0.7,0.5,0.3],[0.2,0.4,0.7,1],"_mixed_pen_Europe_random",areas = areas,method = "random")

#316 for first big cluster, and 164 for second big cluster

