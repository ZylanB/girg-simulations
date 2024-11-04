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

path_checkins = ''
path_edges = ''


gow_checkins_txt = open(path_checkins,'r')
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

pos_r = {}
for key in check_in_data.keys():
    index = np.random.randint(0,len(check_in_data[key][0]))
    pos_r[key] = [check_in_data[key][0][index],check_in_data[key][1][index]]

with open('gow_pos_random.pickle','wb') as outfile:
    pickle.dump(pos_r,outfile,protocol=pickle.HIGHEST_PROTOCOL)


with open('gow_pos_random.pickle','rb') as data:
    gow_pos_dict = pickle.load(data)

gow_edges_txt = open(path_edges,'r')
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
    
