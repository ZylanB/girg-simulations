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


def L_exponentials(g):
    L_rv = g.new_edge_property("float")
    for e in g.edges():
        L_rv[e] = np.random.exponential(1)
    return g,L_rv

def new_infectionSpread(g,status,L_rv,mu,id,ratio = 1,origin_index = None, method = 1): #13104 is at pos 0,0
    num_vertices = g.num_vertices()
    cutoff = ratio*num_vertices
    trans_cost = g.new_edge_property("float")
    for e in g.edges():
        trans_cost[e] = L_rv[e] * ((e.source().out_degree())*(e.target().out_degree()))**mu
    start_time = time.time()
    if origin_index is None:    
        origin_index = 6700
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
    print("Infection Simulation: --- %s seconds ---" % (time.time() - start_time))
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
    with open('gow_graph.pickle', 'rb') as data:
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

    with open('gow_graph.pickle', 'rb') as data:
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

def setLattice(mu):
    start_time = time.time()
    with open('gow_graph.pickle', 'rb') as data:
        g,st,id,pos = pickle.load(data)
    reverse_ids = {}
    for u in g.vertices():
        reverse_ids[id[u]] = int(u)
    matrix = np.zeros([116,337])
    mu = 1
    g, Lrv = L_exponentials(g)
    infs,tc,noinfecs = new_infectionSpread(g,st,Lrv,mu,id)
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
                matrix[row][col] = np.median(infec_order_list)
    print("Matrix Setting Time: --- %s minutes ---" % (time.time() - start_time))
    return matrix

def heatmapsGowalla(mList,marker):
    mu_1, mu_2, mu_3, mu_4 = mList[0],mList[1],mList[2],mList[3]
    m_1 = setLattice(mu_1)
    m_2 = setLattice(mu_2)
    m_3 = setLattice(mu_3)
    m_4 = setLattice(mu_4)
    colors1 = plt.cm.jet(np.linspace(0.,1,255))
    colors2 = plt.cm.Reds(np.linspace(0,1,1))
    colors = np.vstack((colors2,colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_cmap',colors)
    fig = plt.figure(figsize=(16,20))

    ax1 = fig.add_subplot(2,2,1)
    plt.imshow(m_1,cmap=mymap,interpolation = 'spline16')
    plt.title("mu = " + str(mu_1))
    extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('linear'  +str(marker) + '.png',bbox_inches = extent1.expanded(1.3,1.4))

    ax2 = fig.add_subplot(2,2,2)
    plt.imshow(m_2,cmap=mymap,interpolation = 'spline16')
    plt.title("mu = " + str(mu_2))
    extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('polynomial'  +str(marker) + '.png',bbox_inches = extent2.expanded(1.3,1.4))

    ax3 = fig.add_subplot(2,2,3)
    plt.imshow(m_3,cmap=mymap,interpolation = 'spline16')
    plt.title("mu = " + str(mu_3))
    extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('polylog'  +str(marker) + '.png',bbox_inches = extent3.expanded(1.3,1.27))

    ax4 = fig.add_subplot(2,2,4)
    plt.imshow(m_4,cmap=mymap,interpolation = 'spline16')
    plt.title("mu = " + str(mu_4))
    extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('explosive'  +str(marker) + '.png',bbox_inches = extent4.expanded(1.3,1.27))
    fig.savefig('allHeatmaps'  + str(marker) + '.png')


#heatmapsGowalla([2,1,0.5,0],"_gowalla2")

#6700 for first big cluster, and 6000 for second big cluster