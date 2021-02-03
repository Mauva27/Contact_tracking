import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import ndimage

import contactAnalysis as ca
from contactAnalysis import draw_spheres as ds
from contactAnalysis import figures as fig
from contactAnalysis import analysis
from contactAnalysis import properties

def plot_Pn_cp(folders,folder_idx,folders_cp,save_folder):
    N = len(folder_idx)
    i = 0
    plt.figure()
    for f in folder_idx:
        ###### plot P(N)
        nNeigh = np.loadtxt(folders[f]+'t%01d/c2/particle_middle/nNeighbors.txt'%i)
        N_avg = np.round(nNeigh.mean(),1)
        Ncounts, Nedges =np.histogram(nNeigh,bins=12,normed=True)
        Nbin = (Nedges[:-1]+Nedges[1:])/2.
        plt.plot(Nbin,Ncounts,'o-',ms=4,color=plt.cm.viridis(f*1./N),label=r'%s $\bar N$=%.1f'%(folders_cp[f],N_avg))
        plt.legend(ncol=1,frameon=False,fontsize=8,labelspacing=0.2)
        plt.xlabel('$N$'),plt.ylabel('$P(N)$')
        plt.savefig(save_folder+'/p(n).pdf')

def plot_Pz_cp(folders,folder_idx,folders_cp,save_folder):
    N = len(folder_idx)
    i = 0
    plt.figure()
    for f in folder_idx:
        ###### plot P(Z)
        nCon = np.loadtxt(folders[f]+'t%01d/c2/particle_middle/nContacts.txt'%i)
        nCon = nCon[nCon<14]
        Z_avg = np.round(nCon.mean(),1)
        Zcounts, Zedges =np.histogram(nCon,bins=12,normed=True)
        Zbin = (Zedges[:-1]+Zedges[1:])/2.
        plt.plot(Zbin,Zcounts,'o-',ms=4,color=plt.cm.viridis(f*1./N),label=r'%s $\bar Z$=%.1f'%(folders_cp[f],Z_avg))
        plt.legend(ncol=1,frameon=False,fontsize=8,labelspacing=0.2)
        plt.xlabel('$Z$'),plt.ylabel('$P(Z)$')
        plt.savefig(save_folder+'p(Z).pdf')

def force_pdf_cp(folders,folder_idx,thr_range,folders_cp,save_folder,m=1):
    plt.figure()
    N=len(folder_idx)
    i=0
    for f in folder_idx:
        thr = thr_range[f]
        folder=folders[f]
        sizes = np.loadtxt(folder+'t0/c2/particle_middle/volumeResiduals_threshold%02d.txt'%thr)
        H,edges= np.histogram(sizes, bins=30)
        halfbwidth = (edges[1]-edges[0])/2
        x =( edges[:-1]+halfbwidth)
        xx =(x- x[H.argmax()])/(sizes.std()) # scaling with both peak value and std
        xxx =x/sizes.mean()  # scaling with peak value
        xxxx = x/x[5:][H[5:].argmax()]

        if m==0:
            plt.plot(x,H,'o-',label='%s'%folders_cp[f], color=plt.cm.viridis(i*1./N))
            plt.xlabel(r"$f$", size=20)
            plt.legend(frameon=False,fontsize=12)
            plt.savefig(save_folder+'p_f_size.pdf')
        if m==1:
            plt.plot(xx,H,'o-',label='%s'%folders_cp[f], color=plt.cm.viridis(i*1./N))
            plt.xlabel(r"$(f-f_{max})/\sigma_f$", size=20)
            plt.legend(frameon=False,fontsize=12)
            plt.savefig(save_folder+'p_f_scale.pdf')
        if m==2:
            plt.plot(xxx,H,'o-',label='%s'%folders_cp[f], color=plt.cm.viridis(i*1./N))
            plt.xlabel(r"$f/<f>$", size=20)
            plt.legend(frameon=False,fontsize=12)
            plt.savefig(save_folder+'p_f_scale_mean.pdf')
        if m==3:
            plt.plot(xxxx,H,'o-',label='%s'%folders_cp[f], color=plt.cm.viridis(i*1./N))
            plt.xlabel(r"$f/f_{max}$", size=20)
            plt.legend(frameon=False,fontsize=12)
            plt.savefig(save_folder+'p_f_scale_max.pdf')

        plt.yscale('log')
        plt.ylabel('Counts')
        i+=1

def plot_PnPz_cp(folders,folder_idx,folders_cp,save_folder):
#     N = len(folder_idx)
    N = len(folders)
    i = 0
    plt.figure()
    for f in folder_idx:
        con_thr=con_thresholds[f][2]

        nNeigh = np.loadtxt(folders[f]+'t%01d/c2/particle_middle/nNeighbors_conThr%02d.txt'%(i,con_thr))
        N_avg = np.round(nNeigh.mean(),1)
        Ncounts, Nedges =np.histogram(nNeigh,bins=14,normed=True)
        Nbin = (Nedges[:-1]+Nedges[1:])/2.
        plt.plot(Nbin,Ncounts,'--',ms=4,color=plt.cm.terrain(f*1./N),label=r'%s $\bar N$=%.1f'%(folders_cp[f],N_avg))

        nCon = np.loadtxt(folders[f]+'t%01d/c2/particle_middle/nContacts_conThr%02d.txt'%(i,con_thr))
        nCon = nCon[nCon<14]
        Z_avg = np.round(nCon.mean(),1)
        Zcounts, Zedges =np.histogram(nCon,bins=12,normed=True)
        Zbin = (Zedges[:-1]+Zedges[1:])/2.
        plt.plot(Zbin,Zcounts,'-',ms=4,color=plt.cm.terrain(f*1./N),label=r'%s $\bar Z$=%.1f'%(folders_cp[f],Z_avg))
        plt.legend(ncol=1,frameon=False,fontsize=8,labelspacing=0.2)
        plt.xlabel('$Z$'),plt.ylabel('$P(Z)$')
        plt.savefig(save_folder+'p(N)_p(Z)_otsu*1.2.pdf')

def forceChainLength_cp(folders,cp_list,idx):
    N = len(folders)
    i = 0
    plt.figure()
    for f in idx:
        folder = folders[f]
        conthr = con_thresholds[f][2]
        length = np.loadtxt(folders[f]+'TCC/chainlength/chainlength_t00_conThr%02d.txt'%conthr)
        counts,edges = np.histogram(length,bins=10,normed=True)
        bin_edges = (edges[:-1]+edges[1:])/2
        plt.plot(bin_edges,counts,'o-',ms=8,color=plt.cm.viridis(f**(1/2.)*3./N),label='%.1f'%folders_cp[f])
#         plt.hist(length,bins=10,histtype='step',normed=True,color=plt.cm.viridis(f*1./(max(idx)+1)),label='%.2f'%folders_cp[f],alpha=0.6)
    plt.yscale('log')
    plt.xlabel('Chain size [particles]'),plt.ylabel('Probability')
    plt.legend(ncol=1,frameon=False,fontsize=12)
    plt.savefig('/PhD_3rd/Experiments/STED/gel_analysis/polymers/convex_hull/chainlength_hist.pdf')

'correlation between tcc cluster and force chains'
def tcc_in_chains(tcc_xyz,labelled_chain):
    tccChains = []
    for i in range(len(tcc_xyz)):
        tcc = tcc_xyz[i][0]
        color = labelled_chain[i]
        if tcc != 'A' and color!=0:
            tccChains.append((tcc,color,i))
    tccChains = np.array(tccChains)
    clusters = np.unique(tccChains[:,0])

    tcc_count = []
    index_cluster = []
    for c in range(len(clusters)):
        a = 0
        for j in range(len(tccChains)):
            if tccChains[j][0] == clusters[c]:
                a += 1
                index_cluster.append((clusters[c],tccChains[j][-1]))
        tcc_count.append((clusters[c],a/float(tcc_xyz.shape[0])))
    index_cluster = np.array(index_cluster)
    return tccChains,tcc_count,index_cluster
