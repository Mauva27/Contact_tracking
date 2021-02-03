from __future__ import print_function
import numpy as np
import glob
from matplotlib import pyplot as plt
import sys, os


import builtins

def myprint(screen,*args, **kwargs):
    """My custom print() function."""
    if screen:
        return __builtins__.print(*args, **kwargs)

def gr(filename,dr,xyres,zres, screen=True):
	#     filename= ''

	#     myprint(screen,"#----------------------------------------------------------------")
	#     myprint(screen,"#-----------Tool for determination of G(r)-----------------------")
	#     myprint(screen,"#----------------------------------------------------------------")

	#     if len(sys.argv)<5:
	#         myprint(screen,"ERROR: missing mandatory argument!")
	#         myprint(screen,"usage: Gr coordfile  binsize[pixel] xres[micron] zres[micron]")
	#         myprint(screen,"example SeriesXX_coords.txt 0.5 0.29 0.25")
	#         exit(11)
	# #     if len(sys.argv) >= 5:
	#       filename= sys.argv[1]
	#       dr = float(sys.argv[2])
	#       xyres=float(sys.argv[3])
	#       zres=float(sys.argv[4])

    import scipy as sp
    from scipy import spatial,stats
    from numpy import random
    import time

    start_time = time.time()
    np.seterr(divide='ignore', invalid='ignore')

    myprint(screen,"# file %s"%filename)
    C= np.loadtxt(filename,skiprows=2,usecols=(1, 2,3))


    myprint(screen,"# size before cutting borders: ")

    myprint(screen,"# x size: %d %d "%(C[:,0].min(),C[:,0].max()))
    myprint(screen,"# y size: %d %d "%(C[:,1].min(),C[:,1].max()))
    myprint(screen,"# z size: %d %d "%(C[:,2].min(),C[:,2].max()))

    border=8

    C = C[np.logical_not(np.logical_or(C[:,0]<C[:,0].min()+border, C[:,0]>C[:,0].max()-border))]
    C = C[np.logical_not(np.logical_or(C[:,1]<C[:,1].min()+border, C[:,1]>C[:,1].max()-border))]
    C = C[np.logical_not(np.logical_or(C[:,2]<C[:,2].min()+border, C[:,2]>C[:,2].max()-border))]


    num_particles=len(C)

    myprint(screen,"# size after cutting borders: ")
    myprint(screen,"# x size: %d %d "%(C[:,0].min(),C[:,0].max()))
    myprint(screen,"# y size: %d %d "%(C[:,1].min(),C[:,1].max()))
    myprint(screen,"# z size: %d %d "%(C[:,2].min(),C[:,2].max()))


    myprint(screen,"# Number of Particles: %d "%num_particles)
    myprint(screen,"# init done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    #create random numbers in the same range as the real particles
    ID0=np.random.randint(C[:,0].min(),C[:,0].max()+1,len(C))
    ID1=np.random.randint(C[:,1].min(),C[:,1].max()+1,len(C))
    ID2=np.random.randint(C[:,2].min(),C[:,2].max()+1,len(C))


    ID=np.vstack((ID0,ID1))
    ID=(np.vstack((ID,ID2))).T

    #take different zres into account
    mul=np.array([1,1,xyres/zres])
    ID=np.multiply(ID, mul)
    #calculate distances
    RID=sp.spatial.distance.pdist(ID, 'euclidean').flatten()
    myprint(screen,"# ideal gas done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    mul=np.array([1,1,xyres/zres])
    C=np.multiply(C, mul)
    RC=sp.spatial.distance.pdist(C, 'euclidean').flatten()

    myprint(screen,"# particles done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    bins=np.arange(1,RC.max(),dr)

    myprint(screen,"# bin size: %f array [%f,%f]"%(dr,1,RC.max()))
    myprint(screen,"# number of bins: %d"%(len(bins)))

    H,bins,binnumbers=sp.stats.binned_statistic(RC, RC, statistic='count', bins=bins)
    HID,binsID,binnumbersID=sp.stats.binned_statistic(RID, RID, statistic='count', bins=bins)

    #calculate bin centers and divide ParticleHist/IdealGasHist
    bincenters = 0.5*(bins[1:]+bins[:-1])
    hist=H/HID
    #take care of 0/0 and x/0, set to 0
    hist[np.isnan(hist)] = 0
    hist[np.isinf(hist)] = 0
    #save result in file
    f=open(filename+"_CC_r.hist",'wb')
    np.savetxt(f, np.column_stack((bincenters,hist,H,HID)), fmt='%f')
    f.close()

    myprint(screen,"# binning done %1.3f s"%(time.time() - start_time))
    return bincenters,hist,H,HID

def first_minimum(GR):
	particle_sigma = GR[:200].argmax()
	first_minimum = GR[particle_sigma:particle_sigma*2].argmin() + particle_sigma
	return particle_sigma, first_minimum
#
# def blockmyprint(screen,):
#     sys.stdout = open(os.devnull, 'w')
#
# def enablemyprint(screen,):
#     sys.stdout = sys.__stdout__

def gr2d(filename,dr,xyres,zres, screen=True):
    import scipy as sp
    from scipy import spatial,stats
    from numpy import random
    import time

    start_time = time.time()
    np.seterr(divide='ignore', invalid='ignore')

    myprint(screen,"# file %s"%filename)
    C= np.loadtxt(filename,skiprows=2,usecols=(1, 2,3))


    myprint(screen,"# size before cutting borders: ")

    myprint(screen,"# x size: %d %d "%(C[:,0].min(),C[:,0].max()))
    myprint(screen,"# y size: %d %d "%(C[:,1].min(),C[:,1].max()))

    border=8

    C = C[np.logical_not(np.logical_or(C[:,0]<C[:,0].min()+border, C[:,0]>C[:,0].max()-border))]
    C = C[np.logical_not(np.logical_or(C[:,1]<C[:,1].min()+border, C[:,1]>C[:,1].max()-border))]

    num_particles=len(C)

    myprint(screen,"# size after cutting borders: ")
    myprint(screen,"# x size: %d %d "%(C[:,0].min(),C[:,0].max()))
    myprint(screen,"# y size: %d %d "%(C[:,1].min(),C[:,1].max()))

    myprint(screen,"# Number of Particles: %d "%num_particles)
    myprint(screen,"# init done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    #create random numbers in the same range as the real particles
    ID0=np.random.randint(C[:,0].min(),C[:,0].max()+1,len(C))
    ID1=np.random.randint(C[:,1].min(),C[:,1].max()+1,len(C))


    ID=np.vstack((ID0,ID1)).T

    #calculate distances
    RID=sp.spatial.distance.pdist(ID, 'euclidean').flatten()

    myprint(screen,"# ideal gas done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    mul=np.array([1,1,xyres/zres])
    C=np.multiply(C, mul)
    RC=sp.spatial.distance.pdist(C, 'euclidean').flatten()

    myprint(screen,"# particles done %1.3f s"%(time.time() - start_time))
    start_time = time.time()

    bins=np.arange(1,RC.max(),dr)

    myprint(screen,"# bin size: %f array [%f,%f]"%(dr,1,RC.max()))
    myprint(screen,"# number of bins: %d"%(len(bins)))

    H,bins,binnumbers=sp.stats.binned_statistic(RC, RC, statistic='count', bins=bins)
    HID,binsID,binnumbersID=sp.stats.binned_statistic(RID, RID, statistic='count', bins=bins)

    #calculate bin centers and divide ParticleHist/IdealGasHist
    bincenters = 0.5*(bins[1:]+bins[:-1])
    hist=H/HID
    #take care of 0/0 and x/0, set to 0
    hist[np.isnan(hist)] = 0
    hist[np.isinf(hist)] = 0
    #save result in file
    f=open(filename+"_CC_r.hist",'wb')
    np.savetxt(f, np.column_stack((bincenters,hist,H,HID)), fmt='%f')
    f.close()

    myprint(screen,"# binning done %1.3f s"%(time.time() - start_time))
    return bincenters,hist,H,HID

def gr_plot(filefolder):
    g_r_list = glob.glob(filefolder+'*.xyz')

    # sort filenames using Cp
    Cp_list = []
    for i in range(len(g_r_list)):
        Cp = g_r_list[i][-5]
        Cp_list.append(Cp)
    sortedCp = np.argsort(Cp_list)
    sorted_gr_list =  np.array(g_r_list)[sortedCp]

    g_rs = []
    # blockmyprint(screen,)
    for (i,filename) in enumerate(sorted_gr_list):
        r,GR,IGR,IG=gr(str(filename),1,1,1,screen=False)
        particle_sigma = GR[r<200].argmax()
        g_rs.append((r,GR,particle_sigma))
    # enablemyprint(screen,)
    # Cp_Cpcrit = np.array((1.6,3.2,3.9,4.1,4.5))
    plt.figure(figsize=(8,6))
    for i in range(len(g_rs)):
        # print g_rs[i][2]*2
        # plt.plot(g_rs[i][0]/g_rs[i][2],(g_rs[i][1]+i*1),'o-', color=plt.cm.viridis(i*1./len(g_rs)),
        #          label='$%s$  $\phi_c=%s$' %(Cp_Cpcrit[i], sorted_gr_list[i][62:66]))
        # plt.plot(g_rs[i][0]/g_rs[i][2],(g_rs[i][1]+i*0),'o-', color=plt.cm.viridis(i*1./len(g_rs)),
        #          label='$\phi_c=%s$  $C_p=%s$' %(sorted_gr_list[i][62:66], sorted_gr_list[i][68:72]))
        plt.plot(g_rs[i][0]/g_rs[i][2],(g_rs[i][1]+i*1),'o-', color=plt.cm.viridis(i*1./len(g_rs)),
                 label='t=$%s$ ' %i)
    plt.xlim(0,6), plt.ylim(0,11)
    plt.xlabel('$r/\sigma$', size=25),plt.ylabel('$g(r)$', size=25)
    plt.xticks(size=20),plt.yticks(size=20)
    plt.legend(ncol=2,frameon=False,prop={'size':15})
    plt.tight_layout()
    plt.savefig(filefolder+'g_rs.pdf')
    # plt.close()
