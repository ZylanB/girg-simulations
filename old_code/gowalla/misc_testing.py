import girg_sampling.girgs as gs
import graph_tool.all as gt
import numpy as np 
from math import *
import time
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import networkx as nx 
from sklearn.linear_model import LinearRegression
from PIL import Image
import pandas as pd 
import plotly.express as px
from scipy.optimize import minimize

path = '/home/zylan/python/gowalla/gow_graph_mode.pickle'

gow_areas = {"US": [(60,105),(37,90)],"Europe": [(80,115),(145,200)],"full": [(0,115),(0,315)]}

gow_alpha = 1.15
gow_tau = 2.78

with open(path, 'rb') as data:
    g,st,id,pos = pickle.load(data)

for key in pos.keys():
        pos[key][0] += 46
        pos[key][1] += 160
infection_area = gow_areas["US"]
filter = g.new_vertex_property("bool")
for u in g.vertices():
    if pos[id[u]][0] >= infection_area[0][0] and pos[id[u]][0] <= infection_area[0][1] and pos[id[u]][1] >= infection_area[1][0] and pos[id[u]][1] <= infection_area[1][1]:
        filter[u] = True
g = gt.GraphView(g,filter) 

def degCounts(g): #includes linear regression 
    degrees = [] 
    degree_counts = []
    for u in g.vertices(): 
        degrees.append(u.out_degree())
    for d in range(1,max(degrees)+1):
        degree_counts.append(degrees.count(d))
    return degrees, degree_counts

degs, dc = degCounts(g)

with open('gowalla_degrees_US.txt',"w") as f:
    for d in range(max(degs)):
        f.write(str(d+1) + " " + str(dc[d]) + "\n")

        

    