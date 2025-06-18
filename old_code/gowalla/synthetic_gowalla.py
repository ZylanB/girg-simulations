import background_functions as bf
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
import gowalla.Gowalla_new as gow


def findGoodGraph(n,alpha,tau):
    inf_percentage = 0
    while inf_percentage <= 0.70:
        g,pos,st,w,dist_list = bf.genGirg(n,2,tau,alpha)
        g,Lrv = bf.L_exponentials(g)
        infs,tc,noninfecs = bf.infectionSpread(g,st,w,Lrv,mu=0,zeta=0,vertex_set="PPP",ratio = 1,origin_index=None,method=1,penalize=False,pos=pos)
        inf_percentage = len(infs)/g.num_vertices()
    return g,pos,st,w

def epidemicCurve(g,pos,st,w,mu,zeta,num_runs):
    g,L_rv = bf.L_exponentials(g)
    infs,tc,noninfecs = bf.infectionSpread(g,st,w,L_rv,mu,zeta=zeta,vertex_set="PPP",ratio=1,origin_index=None,method=4,penalize=True,pos=pos)
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
    quantile_dict = {"q10" : [], "q30": []}

    for i in range(num_runs):
        t_points = []
        g,L_rv = bf.L_exponentials(g)
        infs,tc,noninfecs = bf.infectionSpread(g,st,w,L_rv,mu,zeta=zeta,vertex_set="PPP",ratio=1,origin_index=None,method=4,penalize=True,pos=pos)
        infs_list = list(infs)
        for I_amount in I_t:
            vertex = infs_list[I_amount-1]
            t_points.append(infs[vertex][1])
        all_t.append(t_points)

    median_t = []

    for k in range(len(I_t)):
        indexed_list = [l[k] for l in all_t]
        median_t.append(np.median(indexed_list))
        quantile_dict["q10"].append((np.percentile(indexed_list,40),np.percentile(indexed_list,60)))
        quantile_dict["q30"].append((np.percentile(indexed_list,20),np.percentile(indexed_list,80)))

    return median_t,I_t,quantile_dict

def epidemic_curves(n,alpha,tau,params,num_runs,marker):
    g,pos,st,w = findGoodGraph(n,alpha,tau)
    plt.figure(1,figsize=(13,13),clear=True)
    k = 1
    for mu,zeta,label in params:
        k += 2
        ec_x,y_ec,q_dict = epidemicCurve(g,pos,st,w,mu,zeta,num_runs)
        saturation_index = np.where(np.array(y_ec) > int(0.95*g.num_vertices()))[0][0]
        y = np.log(y_ec[0:saturation_index-1]) 
        x = ec_x[0:saturation_index-1] 
        plt.figure(k-1,figsize=(13,13),clear=True)
        max_q10 = [q[1] for q in q_dict["q10"][0:saturation_index-1]]
        min_q10 = [q[0] for q in q_dict["q10"][0:saturation_index-1]]
        max_q30 = [q[1] for q in q_dict["q30"][0:saturation_index-1]]
        min_q30 = [q[0] for q in q_dict["q30"][0:saturation_index-1]]
        plt.xlabel("t")
        if label == "polynomial" or label == "pure geometric":
            x = np.log(x)  
            max_q10 = np.log(max_q10)
            min_q10 = np.log(min_q10)
            max_q30 = np.log(max_q30)
            min_q30 = np.log(min_q30)
            plt.xlabel("log(t)")
        plt.figure(1)
        if label == "stretched_exp" or label == "explosive":
            plt.plot(np.log(x),y,linewidth=2,label="mu = " + str(mu) + "zeta = " + str(zeta)+ " (" + str(label) +")")
        if label == "polynomial" or label =="pure geometric":
            plt.plot(x,y,linewidth=2,label="mu = " + str(mu) + "zeta = " + str(zeta)+ " (" + str(label) +")")
        plt.figure(k-1)
        plt.plot(x,y)
        plt.plot(max_q10,y,linestyle = "dashed", color = "black")
        plt.plot(min_q10,y,linestyle = "dashed", color = "black")
        plt.fill_betweenx(y,max_q30,max_q10,color="darkgrey",label = "30%")
        plt.fill_betweenx(y,max_q10,min_q10,color= "lightgrey",label = "10%")
        plt.fill_betweenx(y,min_q10,min_q30,color="darkgrey")
        if label == "polynomial" or label == "pure geometric":
            x = np.array(x)
            y = np.array(y)
            lr_start = np.where(y>=2)[0][0] 
            plt.figure(50+k,figsize=(13,13),clear=True)
            plt.plot(x,y,linewidth=2,label="mu = " + str(mu) + "zeta = " + str(zeta)+ " (" + str(label) +")")
            x=x[lr_start:]
            y=y[lr_start:]
            x = x.reshape((-1,1))
            model = LinearRegression().fit(x,y)
            lr_y = []
            for d in x:
                lr_y.append(model.intercept_ + model.coef_[0]*d)
            plt.plot(x,lr_y,color="red",label="Linear Regression with slope\n" + str(model.coef_[0]))
            plt.grid(visible=True)
            plt.ylabel("log(I(t))")
            plt.xlabel("log(t)")
            plt.title("Mu = " + str(mu) + " Zeta = " + str(zeta) + "(" + str(label) +")\n Median taken over 50 runs with 10/30% percentiles")
            plt.legend()
            plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_"+str(marker)+"_REGRESSION.png")
        plt.figure(k-1)
        plt.grid(visible=True)
        plt.ylabel("log(I(t))")
        plt.title("Mu = " + str(mu) + " Zeta = " + str(zeta) + "(" + str(label) +")\n Median taken over 50 runs with 10/30% percentiles")
        plt.legend()
        plt.savefig("mu_"+str(mu)+"_zeta_"+str(zeta)+"_ec_"+str(marker)+".png")
    plt.figure(1)
    plt.xlabel("log(t)")
    plt.ylabel("log(I(t))")
    plt.grid(visible=True)
    plt.title("Epidemic curves\n Median taken over 50 runs\nalpha = " + str(alpha) + " tau = " + str(tau))
    plt.legend()
    plt.savefig("epidemic_curves_together_" + str(marker)+".png")
    plt.close()

path = '/home/zylan/python/gowalla/gow_graph_mode.pickle'
with open(path, 'rb') as data:
    g2,st2,id2,pos2 = pickle.load(data)

for key in pos2.keys():
    pos2[key][0] += 46
    pos2[key][1] += 160

gow_areas = {"US": [(60,105),(37,90)],"Europe": [(80,115),(145,200)],"full": [(0,115),(0,315)]}

epidemic_curves(10000,2.25,2.2,[(0,0,"explosive"),(0.6,0,"stretched_exp"),(0.9,0,"polynomial"),(1.3,0,"polynomial"),(2.5,0,"pure geometric")],50,"alpha_2.25_tau_2.2")
#epidemic_curves(100000,1.6,2.4,[(0,0,"explosive"),(1,0,"stretched_exp"),(1,1,"polynomial"),(1,2,"pure geometric"),],50,"alpha_1.6_tau_2.4")
#epidemic_curves(100000,1.15,2.65,[(0,0,"explosive"),(1,0,"stretched_exp"),(1,1,"stretched_exp"),(1,2,"polynomial"),(1,3,"pure geometric")],50,"alpha_1.15_tau_2.65")
#gow.epidemic_curves([(0,3,"pure geometric")],50,"gowalla2")
# gow.heatmapsGowalla(g2,st2,id2,pos2,origin_index=164,mList=[0,1,1,1],zList=[0,1,2,3],marker="Europe_only_2",
#                  infection_area = gow_areas["Europe"],tc_method = 2,method = "first",penalize = True,interpolation = "none",draw_epidemic_curves = False,matrix_scaling=1)
#gow.heatmapsGowalla(g2,st2,id2,pos2,origin_index=164,mList=[0,0,0,0],zList=[0,1,2,3],marker="World",zoom_areas=gow_areas,
#                 infection_area = gow_areas["full"],tc_method = 2,method = "first",penalize = True,interpolation = "none",draw_epidemic_curves = False,matrix_scaling=1)
# gow.heatmapsGowalla(g,st,id,pos,origin_index=164,mList=[0,0,0,0],zList=[0,1,2,3],marker="US_only",
#                 infection_area = gow_areas["US"],tc_method = 2,method = "first",penalize = True,interpolation = "none",draw_epidemic_curves = False,matrix_scaling=1)