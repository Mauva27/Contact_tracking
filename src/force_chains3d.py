import numpy as np
from matplotlib import pyplot as plt
import os
from copy import deepcopy
from scipy import ndimage

import contactAnalysis as ca
from contactAnalysis import draw_spheres as ds
from contactAnalysis import figures as fig
from contactAnalysis import g_r
from contactAnalysis import analysis

##### f direction is contacts to particles, r direction is opposite
def compute_stress(particles,contact_cen,FonParticle,dists):
    particle_centre = particles[:,:3]
    p_rad = particles[:,3]
    parToCon_list, stress_tensor = [],[]
    for p in range(len(particles)):
        numF = FonParticle[p][-1].shape[0]
        if numF>0:
            sigmas = 0
            for f in range(numF):
                force = FonParticle[p][1][f]
                fx,fy,fz = force[0],force[1],force[2]

                c_idx = (FonParticle[p][2][f]).astype(int)
                contact = contact_cen[c_idx]
                parToCon = particles[p,:3] - contact
                parToCon_list.append((p,c_idx,parToCon))

                dist = dists[p][c_idx]
                rx,ry,rz = parToCon[0],parToCon[1],parToCon[2]

                sigma = np.array((fx*rx,fx*ry,fx*rz,fy*rx,fy*ry,fy*rz,fz*rx,fz*ry,fz*rz)).reshape(3,3)
                sigmas = sigmas + sigma
        else:
            sigmas = np.zeros((3,3))
        stress = sigmas/(4*np.pi*p_rad[p]**3/3) ### 2d is area or volume?
        stress_tensor.append((p,stress))
    return stress_tensor, parToCon_list

def principal_stress(stress_tensor):
    eigvals,major_stress,minor_stress,stress_trace,anisotropy = [],[],[],[],[]
    for i in range(len(stress_tensor)):
        w,v = np.linalg.eig(stress_tensor[i][1])
        'w, not ordered eigenvalues'
        'v, normalized eigenvectors'
        trace = sum(w)
        major = w.max()
        minor = w.min()
        ani = major - minor
        idx = np.where(w == minor)[0][0]
        eigvals.append((w,v[:,idx]))
        'eigvals, all eigenvalues and minor principle eigenvector'
        major_stress.append(major)
        minor_stress.append(minor)
        stress_trace.append(trace)
        anisotropy.append(ani)

    # eigvals = np.array(eigvals)
    # major_stress = np.array(major_stress)
    # minor_stress = np.array(minor_stress)
    return major_stress, minor_stress, eigvals,stress_trace,anisotropy#, major_theta

def find_next_particle(i,chain_color, color, uu,vv,ww,Stress,eigvals, Neigh,XYZ,pairs,threshold):
    minorStress=Stress
    neigh_index = Neigh[i].astype(int)
    p1 = XYZ[i][:3]

    costhreshold = np.cos(threshold*np.pi/180)

    tobecolored = np.zeros(len(neigh_index)).astype(bool)

    for j,idx in enumerate(neigh_index):
        p2 = XYZ[idx][:3]
        l = p2 - p1
        x1 = minorStress[i]*eigvals[i][1][0]
        y1 = minorStress[i]*eigvals[i][1][1]
        z1 = minorStress[i]*eigvals[i][1][2]
        uu[i]=x1
        vv[i]=y1
        ww[i]=z1
        cosa = np.abs(np.dot(l,[x1,y1,z1]))/np.linalg.norm(l)/np.abs(minorStress[i])

        if cosa >= costhreshold:
            ll = p1 - p2
            x2 = minorStress[idx]*eigvals[idx][1][0]
            y2 = minorStress[idx]*eigvals[idx][1][1]
            z2 = minorStress[idx]*eigvals[idx][1][2]
            cosb = np.abs(np.dot(ll,[x2,y2,z2]))/np.linalg.norm(ll)/np.abs(minorStress[idx])

            if cosb >= costhreshold:
                if chain_color[idx] >-1:
                    chain_color[chain_color ==  color]=chain_color[idx]
                else:
                    tobecolored[j]=True
                    uu[idx]=x2
                    vv[idx]=y2
                    ww[idx]=z2
                pairs.append((i,idx))

    if np.any(tobecolored):
        chain_color[neigh_index[tobecolored]]=color
        return int(neigh_index[tobecolored][0])
    else:
        return False
#                     chain_color[idx]=color
#                     return idx

#     return False

def find_chains(pcen,ncontacts,Neighbours,minor_stress,eigvals,threshold=45):
    oldids = np.arange(pcen.shape[0])
    non_alone = np.where(ncontacts>=1)[0]
    nonaloneids = oldids[non_alone]
    newids =  np.arange(len(nonaloneids))
    XYZ = pcen[non_alone]
    Stress = np.array(minor_stress)[non_alone]
    Neigh= []
    pairs = []
    for j,list_of_neighbours in Neighbours:
        # print 'j'
        # print 'nei',list_of_neighbours
        new_list =[]
        for p in list_of_neighbours:
        # if any(non_alone==p):
            new_list.append(newids[non_alone==p][0])
        Neigh.append(np.array(new_list))
    NN = len(Neigh)
    print (NN)
    print ('shape',XYZ.shape[0])
    assert NN==XYZ.shape[0],"Array sizes are not matching..."

    chain_color=-1*np.ones(NN)
    color = 0
    uu,vv, ww = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    for p in range(NN):
        if chain_color[p]==-1:
            color+=1
            chain_color[p] = color
            next_particle = p
            while next_particle is not False:
                next_particle = find_next_particle(next_particle,chain_color,color,uu,vv,ww,
                                Stress,eigvals,Neigh,XYZ,pairs,threshold)

    return chain_color, uu, vv, ww, XYZ,pairs,Neigh

def color_chains(chain_color,particle_center,non_alone):
    chainlength = []
    for i in range(1,int(chain_color.max()+1)):
        idx = np.where(chain_color==i)[0]
        chainlength.append((i,len(idx)))
    chainlength=np.array(chainlength)
    print ('the longest chain is',chainlength[:,1].max(),'particles')

    long_chain_color = chainlength[chainlength[:,1]>=0][:,0]
    real_chain = chainlength[long_chain_color-1]

    coloredP = particle_center[non_alone]
    labelled_chain = np.zeros(len(coloredP))
    length = np.zeros(len(coloredP))
    c = 0
    for i in long_chain_color:
        if i>-1:
            s = np.where(chain_color==i)[0]
            c+=1
            labelled_chain[s] = c
            length[s] = len(s)

    return chainlength,labelled_chain,coloredP,length
