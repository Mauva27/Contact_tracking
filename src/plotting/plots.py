import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
# import matplotlib.animation as animation
from .__quichi__ import __quichi__
from ..read import load_image, load_movie_frames
from . import sans_serif
import ipdb

def over_plot(media,mode,directory,nframes,prefix,ws,fmt,data,ny,frame_rate):
    '''
    shows all the features found in pretrack
    '''
    pl.close()
    if mode == 'single':
        frame = data.keys()
        if media == 'Images':
            img = load_image(directory, prefix,ws,fmt)
        if media == "Movie":
            img = load_movie_frames(directory, prefix,ws,fmt)
        pl.figure(figsize = (5,5), num='Tracked particles')
        pl.imshow(img, cmap = 'gray')
        pl.plot(data[frame[0]][:,0], -data[frame[0]][:,1]+ny, lw = 0, marker = 'o', ms = 5, mec = 'r', mfc = 'None', zorder = 1)
        pl.xticks([])
        pl.yticks([])
        pl.xlabel('{} particles'.format(data[frame[0]].shape[0]))
    if mode == 'multi':
        fk = sorted(data.keys())

        if (nframes > 1) & (nframes <= 20):
            cols = 5
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(10,3), num='Tracked particles')
            axs = axs.ravel()

            for i, f in enumerate(range):
                i,f = int(i), int(f)
                if media == 'Images':
                    img = load_image(directory,prefix,f,fmt)
                if media == 'Movie':
                    img = load_movie_frames(directory,prefix,f,fmt)
                axs[i].imshow(img, cmap = 'gray')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].plot(data[f][:,0], -data[f][:,1]+ny, lw = 0,ms = 3, marker = 'o', mec = 'r', mfc = 'None', zorder = 1)
                axs[i].set_title('t = {:0.3f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
        elif (nframes > 20):
            cols = 5
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(10,3), num='Tracked particles')
            axs = axs.ravel()
            for i,f in enumerate(range):
                f = int(f)
                if media == 'Images':
                    img = load_image(directory,prefix,f,fmt)
                if media == 'Movie':
                    img = load_movie_frames(directory,prefix,f,fmt)
                axs[i].imshow(img, cmap = 'gray')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].plot(data[f][:,0], -data[f][:,1]+ny, lw = 0,ms = 3, marker = 'o', mec = 'r', mfc = 'None')
                axs[i].set_title('t = {:0.2f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))

def plot_clusters(mode,nframes,data,ids):
    frames = sorted(data.keys())
    pl.close()
    if mode == 'single':
        fig = pl.figure(figsize = (4,4),num = 'Clusters in frame {}'.format(frames[0]))
        ax = fig.gca()
        for ckey in data[frames[0]]:
            ax.scatter(data[frames[0]][ckey]['coords'][:,0], data[frames[0]][ckey]['coords'][:,1], s = 3, alpha = 0.25)
            if ids:
                ax.text(data[frames[0]][ckey]['com'][0],data[frames[0]][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    elif mode == 'multi':
        fk = sorted(data.keys())
        if (nframes > 1) & (nframes <= 20):
            cols = nframes / (float(nframes)*0.25)
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(12,3.5), num='Clusters')
            for i, f in enumerate(range):
                i,f = int(i), int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                for ckey in data[f]:
                    axs[i].scatter(data[f][ckey]['coords'][:,0], data[f][ckey]['coords'][:,1], s = 1, alpha = 0.25)
                    if ids:
                        axs[i].text(data[f][ckey]['com'][0],data[f][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
                    axs[i].set_xlabel('Frame {}'.format(f))
        elif (nframes > 20):
            cols = nframes / (float(nframes)*0.2)
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(13,2.5), num='Tracked particles')
            axs = axs.ravel()
            for i,f in enumerate(range):
                f = int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                for ckey in data[f]:
                    axs[i].scatter(data[f][ckey]['coords'][:,0], data[f][ckey]['coords'][:,1], s = 1, alpha = 0.25)
                    if ids:
                        axs[i].text(data[f][ckey]['com'][0],data[f][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
                    axs[i].set_xlabel('Frame {}'.format(f))


def plot_interface(data):
    fig,axs = pl.subplots(1,5,figsize=(13,3), num='Interface')
    axs = axs.ravel()

    clusters = sorted(data.keys())

    for i in range(5):
        axs[i].scatter(data[clusters[i]]['interior'][:,0], data[clusters[i]]['interior'][:,1], s=5, color = 'b')
        axs[i].scatter(data[clusters[i]]['boundary'][:,0], data[clusters[i]]['boundary'][:,1], s=5, color = 'm')
        axs[i]. set_xticks([])
        axs[i]. set_yticks([])



def plot_boop(media,mode,directory,nframes,prefix,ws,initial_framef,fmt,data, frame_rate):
    '''
    scatters psi6 for every particle
    '''
    pl.close()
    cmap = mpl.colors.ListedColormap(__quichi__, '__quichi__')
    # cmap_r = mpl.colors.ListedColormap(__quichi___[::-1])
    if mode == 'single':
        pl.figure(figsize = (5,5), num='BOOP')
        pl.scatter(data[ws][:,0], data[ws][:,1], c = data[ws][:,-2], cmap = cmap,s = 10)
        pl.xticks([])
        pl.yticks([])
        pl.colorbar(shrink = 0.8)
    elif mode == 'multi':
        fk = sorted(data.keys())
        if (nframes > 1) & (nframes <= 20):
            cols = nframes / (float(nframes)*0.25)
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(12,4.5), num='BOOP in tracked particles')

            for i, f in enumerate(range):
                i,f = int(i), int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                sc = axs[i].scatter(data[f][:,0], data[f][:,1] , c = data[f][:,-2], cmap = cmap)
                axs[i].set_title('t = {:0.3f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
                cbar = pl.colorbar(sc, ax = axs[i], orientation='horizontal', shrink = 0.7, aspect = 10, pad = 0.12, ticks = [0.2,0.4,0.6,0.8])
        elif (nframes > 20):
            cols = nframes / (float(nframes)*0.2)
            which = nframes / cols
            range = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(range), figsize=(13,4.5), num='Tracked particles')
            axs = axs.ravel()
            for i,f in enumerate(range):
                f = int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                sc = axs[i].scatter(data[f][:,0], data[f][:,1], c = data[f][:,-2], cmap = cmap)
                axs[i].set_title('t = {:0.2f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
                cbar = pl.colorbar(sc, ax = axs[i], orientation='horizontal', shrink = 0.7, aspect = 10, pad = 0.12, ticks = [0.2,0.4,0.6,0.8])

def draw_trajs(data,ids):
    '''
    draw in different colors the linked trajectories
    '''
    pl.close()
    fig,ax = pl.subplots(figsize = (4,4), num = 'Trajectories')
    for t in data:
        ax.plot(data[t][:,0], data[t][:,1], alpha = 0.75,label = t)
    ax.set_xticks([])
    ax.set_yticks([])
    pl.axis('off')
    if ids:
        pl.legend()

def quivers(data):
    frames = data.keys()
    cmap = mpl.colors.ListedColormap(__quichi__, '__quichi__')
    norm = mpl.colors.Normalize(vmin=frames[0],vmax=frames[-1])
    cm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cm.set_array([])

    alpha = np.linspace(0.1,1,len(frames))

    pl.close()
    pl.figure(figsize = (5,5))

    for i,f in enumerate(frames):
        pl.quiver(data[f][:,0],data[f][:,1],data[f][:,-2],data[f][:,-1], color = cm.to_rgba(f), alpha = alpha[i])

def plot_density(exp_data,sim_data,error,dimensions,lw,ticks,path,filename,scale = None, labels = [None, None], range = [None, None], shift = 0, save = None):
    markers = ['o', 's', 'p', 'h', 'D', 'v', '^', '*', 'X', 'P', 'H', '>', 'd']
    greens = ['#74C050','#15925e','#94AC3B']
    if kind == 'density':
        cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', greens)
        cnomrexp = mpl.colors.Normalize(vmin=0, vmax=20)
        cexp= mpl.cm.ScalarMappable(norm=cnomrexp, cmap=cmap)
        cexp.set_array([])

    fig, ax = pl.subplots(1,1, figsize=dimensions)
    if not range[0]==None: ax.set_xlim(range[0])
    if not range[1]==None: ax.set_ylim(range[1])

    if not labels[0]==None: ax.set_xlabel(labels[0])
    if not labels[1]==None: ax.set_ylabel(labels[1])

    for i,ephi in enumerate(exp_data):
        for j,epe in enumerate(sorted(exp_data[ephi])):
            ax.plot(epe,exp[ephi][epe][0],lw = lw,marker = markers[i])
            if error:
                ax.plot(epe,exp[ephi][epe][0],yerr=xp[ephi][epe][1],w = lw,marker = markers[i])

def plot_dynamics(exp_data,sim_data,dimensions,lw,ticks,path,filename,scale=None,labels=[None,None],range=[None,None],save=None):
    cmap = mpl.colors.ListedColormap(__quichi__[::-1], '__quichi__')
    cnorm = mpl.colors.Normalize(vmin=0,vmax=len(data[phi].keys()))
    cexp = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cexp.set_array([])

    markers = ['o', 's', 'p', 'h', 'D', 'v', '^', '*', 'X', 'P', 'H', '>', 'd']

    fig = pl.figure(figsize = dimensions)
    ax = fig.gca()

    for i,ephi in enumerate(exp_data):
        for j,epe in enumerate(sorted(exp_data[ephi])):
            ax.plot(exp[ephi][epe][:,0],exp[ephi][epe][:,1],lw = lw,marker = markers[i])
