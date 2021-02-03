import numpy as np
from matplotlib import pyplot as plt

def remove_edge_particles(image,particles):
    particle_centre = particles[:,:3]
    particle_size = particles[:,3]

    edge_particle = []
    x_border = image.shape[2]
    y_border = image.shape[1]
    z_border = image.shape[0]
    d_edge = particle_size.mean()
    for p in range(len(particles)):
        edgex = x_border-particle_centre[p,0]
        edgey = y_border-particle_centre[p,1]
        edgez = z_border-particle_centre[p,2]
        if edgex<d_edge or edgex>(x_border-d_edge):
            edge_particle.append(p)
        if edgey<d_edge or edgey>(y_border-d_edge):
            edge_particle.append(p)
        if edgez<d_edge or edgez>(z_border-d_edge):
            edge_particle.append(p)
    edge_particle = (np.unique(edge_particle)).astype(int)

    from copy import deepcopy
    pcen_inbox = deepcopy(particle_centre)
    pcen_inbox[edge_particle]=0
    pcen_inbox_index = np.delete(np.arange(0,len(particle_centre),1),edge_particle)
    # pcen_inbox = np.delete(pcen,edge_particle,0)
    print ('number of particles at the edges',edge_particle.shape[0])
    return edge_particle, pcen_inbox_index

def cell_z_edges(image,z_range=True):
    from scipy.signal import argrelextrema
    lx,ly,lz = image.shape[2],image.shape[1],image.shape[0]
    intensity_avg = image.mean(axis=(1,2))
    local_max = argrelextrema(intensity_avg,np.greater)
#     local_min = argrelextrema(intensity_avg,np.less)
    if z_range == True:
        z_first = local_max[0][0]
        z_last = local_max[0][-1]
        V_box = lx*ly*(z_last-z_first)
    else:
        V_box = lx*ly*lz
        z_first = 0
        z_last = lz
    return V_box, z_first,z_last

def volume_fraction(particles,image,factor=1.,z_range=True):
    particle_center = particles[:,:3]
    p_rad = particles[:,3]*factor
    std_p = np.std(p_rad)
    pd = std_p / p_rad.mean()

    edge_particle, pcen_inbox_index = remove_edge_particles(image,particles)

    N = len(particle_center[pcen_inbox_index])
    print ('Num of inside particles',N,'factor', factor, len(particles))
    V = 0
    for p in pcen_inbox_index:
        v_p = 4/3.*np.pi*p_rad[p]**3
        V +=v_p

    if z_range==True:
        V_box, z_first,z_last = cell_z_edges(image,z_range)
    else:
        lx = image.shape[2] - 2*p_rad.mean()
        ly = image.shape[1] - 2*p_rad.mean()
        lz = image.shape[0] - 2*p_rad.mean()
        V_box = lx * ly * lz

        z_first,z_last = 0, image.shape[0]

    phi = V/V_box
    return phi,pd,z_first,z_last

def phi_of_z(image,folder,bin_z=30,T=1):
    plt.figure(figsize=(6,4.5))
    for t in range(T):
#         plt.figure(figsize=(6,4.5))
        p_cen=np.loadtxt(folder+"t%01d/c1/particle_center.txt"%t)
        g_r = np.loadtxt(folder+'t%01d/c1/particle_center.xyz_CC_r.hist'%t)
        p_rad_gr = g_r[np.where(g_r[:,1]==g_r[:,1][g_r[:,0]<30].max())[0]][0][0]
        V_box, z_first,z_last = cell_z_edges(image,z_range=True)
        #z_range=True, cell height is less than image.shape[0]
        z = z_last-z_first
        hist, bin_edges = np.histogram(p_cen[:,2],bins=bin_z,range=(z_first,z_last))
        bin_mid = (bin_edges[1]-bin_edges[0])/2.+bin_edges[:-1]
        p_radius = p_rad_gr/2.
        v_cell = z/bin_z*image.shape[1]**2
        local_phi = 4./3*np.pi*p_radius**3*hist/v_cell

        plt.plot((bin_mid-z_first)/z,local_phi,'o-',color=plt.cm.magma(t*1./T),label='t=%01d'%t)
        plt.xlabel('$z/H$'),plt.ylabel('$\phi_c$')
        plt.legend(frameon=False,ncol=3,loc=4)
        plt.savefig(folder+"phi_of_z_bin%03d.pdf"%bin_z)
            # plt.savefig(folder+"t%01d/c1/phi_of_z_bin%03d.pdf"%(t,bin_z))
    return hist, bin_edges

def phi_of_z_GLASS(image,folder,bin_z=30,T=1):
    plt.figure(figsize=(6,4.5))
    for t in range(T):
#         plt.figure(figsize=(6,4.5))
        p_cen=np.loadtxt(folder+"t%02d/c1/particle_center.txt"%t)
        g_r = np.loadtxt(folder+'t%02d/c1/particle_center.xyz_CC_r.hist'%t)
        p_rad_gr = g_r[np.where(g_r[:,1]==g_r[:,1][g_r[:,0]<30].max())[0]][0][0]
        V_box, z_first,z_last = cell_z_edges(image,z_range=True)
        #z_range=True, cell height is less than image.shape[0]
        z = z_last-z_first
        hist, bin_edges = np.histogram(p_cen[:,2],bins=bin_z,range=(z_first,z_last))
        bin_mid = (bin_edges[1]-bin_edges[0])/2.+bin_edges[:-1]
        p_radius = p_rad_gr/2.
        v_cell = z/bin_z*image.shape[1]**2
        local_phi = 4./3*np.pi*p_radius**3*hist/v_cell
#         plt.plot((bin_mid-z_first)/z,hist.astype(float)/sum(hist),'o-',color=plt.cm.magma(t*1./T),label='t=%01d'%t)
#         plt.xlabel('$z/H$'),plt.ylabel(r'$\rho_p$')
        plt.plot((bin_mid-z_first)/z,local_phi,'o-',color=plt.cm.magma(t*1./T),label='t=%01d'%t)
        plt.xlabel('$z/H$'),plt.ylabel('$\phi_c$')
        plt.legend(frameon=False,ncol=3,loc=4)
        plt.savefig(folder+"phi_of_z_bin%03d.pdf"%bin_z)
        # plt.savefig(folder+"t%01d/c1/phi_of_z_bin%03d.pdf"%(t,bin_z))
    return hist, bin_edges
