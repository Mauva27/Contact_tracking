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
    particle_centre = particles[:,:2]
    p_rad = particles[:,2]
    parToCon_list, stress_tensor = [],[]
    for p in range(len(particles)):
        numF = FonParticle[p][-1].shape[0]
        if numF>0:
            sigmas = 0
            for f in range(numF):
                force = FonParticle[p][1][f]
                fx,fz = force[0],force[1]

                c_idx = (FonParticle[p][2][f]).astype(int)
                contact = contact_cen[c_idx]
                parToCon = particles[p,:2] - contact
                parToCon_list.append((p,c_idx,parToCon))

                dist = dists[p][c_idx]
                rx,rz = parToCon[0],parToCon[1]

                sigma = np.array((fx*rx,fx*rz,fz*rx,fz*rz)).reshape(2,2)
                sigmas = sigmas + sigma
        else:
            sigmas = np.zeros((2,2))
        stress = sigmas/(4*np.pi*p_rad[p]**3/3) ### 2d is area or volume?
        stress_tensor.append((p,stress))
    return stress_tensor, parToCon_list

def principal_stress(stress_tensor):
    major_stress =[]
    minor_stress = []
    major_theta = []
    for p in range(len(stress_tensor)):
        s_xx = stress_tensor[p][1][0][0]
        s_xz = stress_tensor[p][1][0][1]
        s_zz = stress_tensor[p][1][1][1]
        s_x = (s_xx+s_zz)/2+np.sqrt(((s_xx-s_zz)/2)**2+s_xz**2)
        s_z = (s_xx+s_zz)/2-np.sqrt(((s_xx-s_zz)/2)**2+s_xz**2)
        major_stress.append(s_x)
        minor_stress.append(s_z)
        if s_xx != s_zz:
            tan_2theta = 2*s_xz/(s_xx-s_zz)
#             print (s_xx-s_zz, s_xx,s_zz)
            theta = np.arctan(tan_2theta)/2
        else:
            theta = 0
        major_theta.append(theta)
    return major_stress, minor_stress, major_theta

def find_next_particle(i,chain_color, color, uu,vv,Stress,Angle, Neigh,XY,threshold=90):
    minorStress=Stress
    majorAngle=Angle
    # XY = high_pcen[non_alone]
    # print Neigh
    neigh_index = Neigh[i].astype(int)
    # print neigh_index
    # print 'xy is',XY
    p1 = XY[i][:2]
    # print 'p1 is', p1

    costhreshold = np.cos(threshold*np.pi/180)

    tobecolored = np.zeros(len(neigh_index)).astype(bool)

    for j,idx in enumerate(neigh_index):
        p2 = XY[idx][:2]
        l = p2 - p1
        x1 = minorStress[i]*np.cos(majorAngle[i])
        y1 = minorStress[i]*np.sin(majorAngle[i])
        uu[i]=x1
        vv[i]=y1
        cosa = np.abs(np.dot(l,[x1,y1]))/np.linalg.norm(l)/np.abs(minorStress[i])

        if cosa >= costhreshold:
            ll = p1 - p2
            x2 = minorStress[idx]*np.cos(majorAngle[idx])
            y2 = minorStress[idx]*np.sin(majorAngle[idx])
            cosb = np.abs(np.dot(ll,[x2,y2]))/np.linalg.norm(ll)/np.abs(minorStress[idx])

            if cosb >= costhreshold:
                if chain_color[idx] >-1:
                    chain_color[chain_color ==  color]=chain_color[idx]
                else:
                    tobecolored[j]=True
                    uu[idx]=x2
                    vv[idx]=y2

    if np.any(tobecolored):
        chain_color[neigh_index[tobecolored]]=color
        return int(neigh_index[tobecolored][0])
    else:
        return False
#                     chain_color[idx]=color
#                     return idx

#     return False

def find_chains(high_pcen,n_p_high,highStress_neighbours,high_minor_stress,high_major_angle):
    oldids = np.arange(high_pcen.shape[0])
    non_alone = np.where(n_p_high>=1)[0]
    nonaloneids = oldids[non_alone]
    newids =  np.arange(len(nonaloneids))
    XY = high_pcen[non_alone]
    Stress = np.array(high_minor_stress)[non_alone]
    Angle = np.array(high_major_angle)[non_alone]

    Neigh= []
    for j,list_of_neighbours in highStress_neighbours:
        new_list =[]
        for p in list_of_neighbours:
            print non_alone == p
            new_list.append(newids[non_alone==p][0])
        Neigh.append(np.array(new_list))
    NN = len(Neigh)

    assert NN==XY.shape[0],"Array sizes are not matching..."

    chain_color=-1*np.ones(NN)
    color = 0
    uu,vv = np.zeros(NN), np.zeros(NN)
    for p in range(NN):
        if chain_color[p]==-1:
            color+=1
            chain_color[p] = color
            next_particle = p
            while next_particle is not False:
                next_particle = find_next_particle(next_particle,chain_color,color,uu,vv,
                                Stress,Angle,Neigh,XY)

    return chain_color, uu, vv, XY
