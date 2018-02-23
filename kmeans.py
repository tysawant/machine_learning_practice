# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:16:14 2017

@author: Tushar
"""
import math
from random import random
import numpy as np
import matplotlib.pyplot as plt
import colorsys
########################################################
def datagen(n):
    points = []
    for i in range(n):
        x = random()
        y = random()
        points.append([x,y])
    return points

def hexCol(N):
    HSV = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    colr = []
    for rgb in HSV:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        colr.append('#%02x%02x%02x' % tuple(rgb))
    return colr

def centroidgen(points,nc,r):
    centroids = []
    [p,q]= [sum(x)/len(x) for x in zip(*points)]
    for i in range(nc):
        theta = 2*math.pi*random()
        s = r*random()
        centroids.append((p+s*math.cos(theta), q+s*math.sin(theta)))
    return centroids
        
def clustergen(pt,c):
    dist = []
    for i in range(len(pt)):
        for j in range(len(c)):
            dist.append(math.hypot(c[j][0] - pt[i][0], c[j][1] - pt[i][1]))
        pt[i].insert(0,np.argmin(dist)+1)
        dist = []
    #print(pt)
    return pt

def kmeans(pt,c,itr):
    for l in range(itr):
        #pt = sorted(pt)
        k = []
        cnew = []
        y = np.unique([x[0] for x in pt])
        #nc = len(y)
        for i in y:
            for j in range(len(pt)):
                if len(pt[j]) == 3:
                    if pt[j][0] == i:
                        del pt[j][0]
                        k.append(pt[j])
            p,q=[sum(x)/len(x) for x in zip(*k)]
            cnew.append([p,q])
            k = []
        pt = clustergen(pt,cnew)
    w = len(np.unique([x[0] for x in pt]))
    print("The number of clusters requested was",nc,
          ", and",w," clusters were generated!")
    return pt
        
n = 100
r = 1
itr = 100
nc = 2
file = open("realdata.txt", "r")
real_data = list(map(str,file))
for i in range(len(real_data)):
    real_data[i] = real_data[i].split()

a = {}
a.setdefault("index", [])
a.setdefault("X", [])
a.setdefault("Y", [])
for i in range(len(real_data)):
    a["index"].append(real_data[i][0])
    a["X"].append(float(real_data[i][1]))
    a["Y"].append(float(real_data[i][2]))

X_data = a["X"]
Y_data = a["Y"]
dat = list(map(list, zip(X_data,Y_data)))
orig = dat
cen = centroidgen(dat,nc,r)
cl = clustergen(dat,cen)
fin_cl = kmeans(cl,cen,itr)
colors = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.', 'k.', 'c.']
clusters = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"
            , "Cluster 6", "Cluster 7", "Cluster 8", "Cluster 9", "Cluster 10"]
hexcolor = hexCol(nc)
plt.figure();
for b in range(len(fin_cl)):
    plt.scatter(fin_cl[b][1],fin_cl[b][2], color = hexcolor[(fin_cl[b][0])-1])

plt.xlabel("Length")
plt.ylabel("Width")
plt.show()

