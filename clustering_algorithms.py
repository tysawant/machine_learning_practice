# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:23:01 2017

@author: Tushar
"""
import numpy as np
import math
from random import random
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import sklearn.cluster as cluster
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from collections import Counter
from scipy import sparse as sp
#################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
# Ported from : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###################################################################################
def check_clusterings(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
# Ported from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html
    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred
###################################################################################
def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")
#Ported from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency
###################################################################################
def fowlkes_mallows_score(labels_true, labels_pred, sparse=False):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples, = labels_true.shape
# Ported from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html
    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0. else 0.
###################################################################################    
def cluster_arrange(labels, Z):
    c_dat = []
    for j in range(len(labels)):
        c_dat.append([labels[j],Z[j]])

    ls = []
    label_n = []
    clust = []
    for i in range(max(Z)+1):
        for j in range(len(c_dat)):
            if c_dat[j][1] == i:
                ls.append(c_dat[j])
        lst = [x[0] for x in ls]
        label_n.append(lst)
        comm = Counter(lst).most_common(1)[0][0]
        for k in range(len(ls)):
            clust.append(comm)
            ls = []

    label_n = [item for sublist in label_n for item in sublist]
    return label_n, clust
####################################################################################
def centroidgen(points,nc,r):
    centroids = []
    [p,q]= [sum(x)/len(x) for x in zip(*points)]
    for i in range(nc):
        theta = 2*math.pi*random()
        s = r*random()
        centroids.append((p+s*math.cos(theta), q+s*math.sin(theta)))
    return centroids
####################################################################################        
def clustergen(pt,c):
    dist = []
    for i in range(len(pt)):
        for j in range(len(c)):
            dist.append(math.hypot(c[j][0] - pt[i][0], c[j][1] - pt[i][1]))
        pt[i].insert(0,np.argmin(dist))
        dist = []
    #print(pt)
    return pt
#####################################################################################
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
    print("The number of clusters requested was",total_digits,
          ", and",w," clusters were generated!")
    return pt
######################################################################################
# LOAD DATA
digits = load_digits()
data = scale(digits.data)

samples, features = data.shape
total_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

reduced_data = PCA(n_components=2).fit_transform(data)
reduced_data = list(map(list, reduced_data))
#####################################################################################
# K-MEANS CLUSTERING
#kmeans = KMeans(init='k-means++', n_clusters=total_digits)
#kmeans.fit(reduced_data)
#Z = kmeans.predict(reduced_data)
cen = centroidgen(reduced_data,total_digits,r=1)
cl = clustergen(reduced_data,cen)
fin_cl = kmeans(cl,cen,itr=100)
Z = [x[0] for x in fin_cl]
print("The Fowlkes Mallow score for K-means is: ", fowlkes_mallows_score(labels, Z))
#####################################################################################
# CONFUSION MATRIX
label_n, clust = cluster_arrange(labels, Z)
cnf_matrix = metrics.confusion_matrix(label_n, clust)
np.set_printoptions(precision=2)
######################################################################################
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target),
                      title='Confusion matrix, without normalization')
######################################################################################
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target), normalize=True,
                      title='Normalized confusion matrix')
plt.show()
#######################################################################################
# AGGLOMERATIVE CLUSTERING
agglo = cluster.AgglomerativeClustering(n_clusters=10,linkage='ward')
Z = agglo.fit_predict(reduced_data)
print("The Fowlkes Mallow score for Agglomerative Clustering is: ", fowlkes_mallows_score(labels, Z))
#####################################################################################
# CONFUSION MATRIX
label_n, clust = cluster_arrange(labels, Z)
cnf_matrix = metrics.confusion_matrix(label_n, clust)
np.set_printoptions(precision=2)
######################################################################################
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target),
                      title='Confusion matrix, without normalization')
######################################################################################
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target), normalize=True,
                      title='Normalized confusion matrix')
plt.show()
#######################################################################################
# AFFINITY PROPOGATION
affinity = cluster.AffinityPropagation()
affinity.fit(reduced_data)
Z = affinity.predict(reduced_data)
print("The Fowlkes Mallow score for Affinity Propogation is: ", fowlkes_mallows_score(labels, Z))
#####################################################################################
# CONFUSION MATRIX
label_n, clust = cluster_arrange(labels, Z)
cnf_matrix = metrics.confusion_matrix(label_n, clust)
np.set_printoptions(precision=2)
######################################################################################
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target),
                      title='Confusion matrix, without normalization')
######################################################################################
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(digits.target), normalize=True,
                      title='Normalized confusion matrix')
plt.show()
#######################################################################################
