import numpy as np
from matplotlib import pyplot as plt

def loadtxt_tcc_population(fname, cluster):
    xyz=np.loadtxt(fname, skiprows=2,usecols=[1,2,3])
    _cluster=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_%s"%cluster,skiprows=3,dtype='S1' ) #S1 means: 1 single character string
    return _cluster,cluster

def tcc_gross(folders,folder_idx):
    i = 0
    population=[]
    for f in range(len(folder_idx)):
        folder = folders[folder_idx[f]]
        fname=folder + 't%02d/c1/TCC/particle_center.xyz'%i
        xyz=np.loadtxt(fname, skiprows=2,usecols=[1,2,3])
        N=xyz.shape[0]

        tcc_pool = ['sp3b','5A','6A','7A','8B','9B','10B','11F','12E','13B','HCP','FCC']
        N_species = len(tcc_pool)

        for j in range(N_species):
            _cluster, cluster = loadtxt_tcc_population(fname,tcc_pool[j])
            _cluster_idx = _cluster == 'C'

            N_cluster = np.round(float(sum(_cluster_idx))/len(_cluster),2)

            population.append((tcc_pool[j], N_cluster))
    return population

def tcc_gross_time(folder,time_range):
    population=[]
    for i in range(len(time_range)):
        t = time_range[i]
        fname=folder + 't%02d/c1/TCC/particle_center.xyz'%t
        xyz=np.loadtxt(fname, skiprows=2,usecols=[1,2,3])
        N=xyz.shape[0]

        tcc_pool = ['sp3b','5A','6A','7A','8B','9B','10B']
        N_species = len(tcc_pool)

        for j in range(N_species):
            _cluster, cluster = loadtxt_tcc_population(fname,tcc_pool[j])
            _cluster_idx = _cluster == 'C'

            N_cluster = np.round(float(sum(_cluster_idx))/len(_cluster),2)

            population.append((tcc_pool[j], N_cluster))
    return population

class tcc():
    def __init__(self):
        super().__init__()

    "Find gross TCC"
    def TCC_index(value_name,T):
        for i in range(T):
            fname=folder + 't%02d/c1/TCC/particle_center.xyz'%i
            xyz=np.loadtxt(fname, skiprows=2,usecols=[1,2,3])
            #  read the 10B particles

            _4A=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_sp3b",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _5A=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_5A",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _6A=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_6A",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _7A=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_7A",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _8B=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_8B",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _9B=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_9B",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _10B=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_10B",skiprows=3,dtype='S1' ) #S1 means: 1 single character string
            _BCC=np.loadtxt(fname+".rcAA30.rcAB30.rcBB30.Vor1.fc0.88.PBCs0.raw_BCC_9",skiprows=3,dtype='S1' ) #S1 means: 1 single character string

            N=xyz.shape[0]

            non_alone = np.loadtxt(folder+'t%01d/c2/particle_middle/non_alone.txt'%i)
            non_alone = non_alone.astype(int)

            _4A_ = _4A[non_alone]
            _5A_ = _5A[non_alone]
            _6A_ = _6A[non_alone]
            _7A_ = _7A[non_alone]
            _8B_ = _8B[non_alone]
            _9B_ = _9B[non_alone]
            _10B_ = _10B[non_alone]
            _BCC_ = _BCC[non_alone]

            _4A_idx = _4A_ == 'C'
            _5A_idx = _5A_ == 'C'
            _6A_idx = _6A_ == 'C'
            _7A_idx = _7A_ == 'C'
            _8B_idx = _8B_ == 'C'
            _9B_idx = _9B_ == 'C'
            _10B_idx = _10B_ == 'C'
            _BCC_idx = _BCC_ == 'C'

            N_4A = np.round(float(sum(_4A_idx))/len(non_alone),2)
            N_5A = np.round(float(sum(_5A_idx))/len(non_alone),2)
            N_6A = np.round(float(sum(_6A_idx))/len(non_alone),2)
            N_7A = np.round(float(sum(_7A_idx))/len(non_alone),2)
            N_8B = np.round(float(sum(_8B_idx))/len(non_alone),2)
            N_9B = np.round(float(sum(_9B_idx))/len(non_alone),2)
            N_10B = np.round(float(sum(_10B_idx))/len(non_alone),2)
            N_BCC = np.round(float(sum(_BCC_idx))/len(non_alone),2)
            tcc_idx = np.vstack((_4A_idx,_5A_idx,_6A_idx,_7A_idx,_8B_idx,_9B_idx,_10B_idx,_BCC_idx))
            tcc_N = np.vstack((N_4A,N_5A,N_6A,N_7A,N_8B,N_9B,N_10B,N_BCC))
            values = np.loadtxt(folder + 't%01d/c2/particle_middle/stress_tensor/%s.txt'%(i,value_name))

    #         tccChains, tcc_count, index_cluster = tcc_in_chains(tcc_nonalone,labelled_chain)
    #         plt.figure()
    #         plt.title('Stress Trace t = %01d'%i)
            for j in range(len(tcc_pool)):
                plt.figure()
                plt.title('Stress Trace t = %01d'%i)
                plt.hist(values[non_alone][tcc_idx[j]],bins=10,normed=True,histtype='step',label='%s (%.2f)'%(tcc_pool[j],tcc_N[j]))
    #         plt.hist(values[non_alone][_4A_idx],bins=10,normed=True,histtype='step',label='4A(%.2f)'%N_4A)
    #         plt.hist(values[non_alone][_5A_idx],bins=10,normed=True,histtype='step',label='5A(%.2f)'%N_5A)
    #         plt.hist(values[non_alone][_6A_idx],bins=10,normed=True,histtype='step',label='6A(%.2f)'%N_6A)
    #         plt.hist(values[non_alone][_7A_idx],bins=10,normed=True,histtype='step',label='7A(%.2f)'%N_7A)
    #         plt.hist(values[non_alone][_8B_idx],bins=10,normed=True,histtype='step',label='8B(%.2f)'%N_8B)
    #         plt.hist(values[non_alone][_9B_idx],bins=10,normed=True,histtype='step',label='9B(%.2f)'%N_9B)
    #         plt.hist(values[non_alone][_10B_idx],bins=10,normed=True,histtype='step',label='10B(%.2f)'%N_4A)
    #         plt.hist(values[non_alone][_BCC_idx],bins=10,normed=True,histtype='step',label='BCC(%.2f)'%N_4A)
                plt.yscale('log')
                if value_name == 'stress_trace':
                    plt.xlabel('Stress Trace'),plt.ylabel('pdf')
                if value_name == 'minor_stress':
                    plt.xlabel('Minor Stress'),plt.ylabel('pdf')
                if value_name == 'major_stress':
                    plt.xlabel('Major Stress'),plt.ylabel('pdf')
                plt.legend(frameon=False,loc=2)
                plt.savefig(folder + 't%01d/%s_pdf_cluster_t%01d_gross_%s.pdf'%(i,value_name,i,tcc_pool[j]))
    #         plt.savefig(folder + '%s_pdf_cluster_t%01d_gross.pdf'%(value_name,i))
