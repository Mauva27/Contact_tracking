import numpy as np
import os
import sys
import glob
from scipy.spatial.distance import cdist, pdist,squareform
from read_lif import Reader

def get_features

def removeBigOverlap(centers,threshold):
    ##### find relative big particles, and compute distances with all other particles
    ##### if it's too close to other particles then delete big particles

    index = centers[:,3]>threshold ##### index with boolean
    big_index = np.arange(len(centers))[index] ##### inedex with numbers
    N_big = len(big_index)
    big_centers = centers[index] ##### centers of big particles
    distance = cdist(centers[:,:3],big_centers[:,:3])

    big_overlap=[]
    for i in range(N_big):
        ##### distance with itself is zero
        ##### radius cut off is the raidus of the big particle
        if (distance[:,i]<big_centers[i,3]*1.2).sum()>1:
            big_overlap.append(big_index[i])
    removeBig = np.delete(centers,big_overlap,axis=0)
    return removeBig


def compute_distance(centers):
    distance = pdist(centers[:,:3])
    D = squareform(distance)
    D[D==0]=np.inf

    threshold = centers[:,3].mean() #### radius cutoff is the average radius size
    tooClose = np.where(D<threshold)
    print tooClose[0].shape[0],'pairs are too close'
    return tooClose

def remove_overlap(centers):
    tooClose = compute_distance(centers)
    new = np.vstack((tooClose[0],tooClose[1])).T
    overlap = []
    for i in range(tooClose[0].shape[0]):
        a = new[i][0]
        b = new[i][1]
        r1 = centers[a][3]
        r2 = centers[b][3]
        if r1<r2:
            overlap.append(b) #### compare radii of two close particles and keep the big one
        else:
            overlap.append(a)

    new_centers = np.delete(centers,overlap,axis=0)
    print '##### after removing overlap'
    new_tooClose = compute_distance(new_centers)
    print 'Number of particles left is',new_centers.shape[0]
    return new_centers
