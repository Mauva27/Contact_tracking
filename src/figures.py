import pylab as plt
import numpy as np
import os
import sys
import glob

def draw_circles(xs, ys, rs,**kwargs):
	for x,y,r in zip(xs,ys,rs):
		circle = plt.Circle((x,y), radius=r, **kwargs)
		plt.gca().add_patch(circle)

def plot_slice(_3dImage,slice):
	plt.imshow(_3dImage[slice,:,:])

def plot_slice_save(_3dImage,savepath):
	slices = _3dImage.shape[0]
	for slice in range(slices):
		plt.imshow(_3dImage[slice,:,:])
		plt.savefig(savepath+'%04d.pdf'%slice)
		plt.close()

def plot_profile_z(_3dImage,style='--',label=None):
	plt.plot(_3dImage.mean(axis=(1,2)),style, label='%s'% (label,))
	plt.xlabel('Z'), plt.ylabel('Intensity')
	# plt.xlabel('z',size=20), plt.ylabel('Intensity',size=20)
	# plt.xticks(size=15), plt.yticks(size=15)
	plt.legend(loc=2)

def multifigures_xy_xz4(Image1,Image2,slice):
	fig=plt.figure()
	num_figure = 4
	fig.add_subplot(1,num_figure,1)
	plt.imshow(Image1[slice,:,:], cmap=plt.cm.afmhot)
	fig.add_subplot(1,num_figure,2)
	plt.imshow(Image2[slice,:,:], cmap=plt.cm.afmhot)
	fig.add_subplot(1,num_figure,3)
	plt.imshow(Image1[:,slice,:], cmap=plt.cm.afmhot)
	fig.add_subplot(1,num_figure,4).axis('off')
	plt.imshow(Image2[:,slice,:], cmap=plt.cm.afmhot)
	fig.set_size_inches(15,15)
	plt.tight_layout()

def multifigures_xy3(Image1,slice1,slice2,slice3):
	fig=plt.figure()
	num_figure = 3
	fig.add_subplot(1,num_figure,1)
	plt.imshow(Image1[slice1,:,:])
	fig.add_subplot(1,num_figure,2)
	plt.imshow(Image1[slice2,:,:])
	fig.add_subplot(1,num_figure,3).axis('off')
	plt.imshow(Image1[slice3,:,:])
	fig.set_size_inches(15,15)
	plt.tight_layout()

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        # if image.ndim == 2:
        plt.imshow(image, cmap=plt.cm.jet)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	# fig.tight_layout()
    # plt.show()

def write_xyz_file(filename, p_cen):
	f = open(filename+'.xyz','w')
	f.write(str(p_cen.shape[0])+'\n'+'Atoms'+'\n')
	for p in range(p_cen.shape[0]):
	    f.write('%s %s %s %s %s' %('A',p_cen[p][0],p_cen[p][1],p_cen[p][2],'\n'))
	f.close()

def write_xyz_file_p(filename, p_cen,Property):
	f = open(filename+'.xyz','w')
	f.write(str(p_cen.shape[0])+'\n'+'Atoms'+'\n')
	for p in range(p_cen.shape[0]):
	    f.write('%s %s %s %s %s %s' %('A',p_cen[p][0],p_cen[p][1],p_cen[p][2],Property[p],'\n'))
	f.close()

def histogram(histograms, style='o-',bins_num=30,xlabel=None,ylabel=None,labels=None,text=None,directory=None):
	plt.figure(figsize=(8,6))
	if len(histograms) > 100:
		counts, bin_edges = np.histogram(histograms,bins=bins_num[i])
		bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
		plt.plot(bin_centres,counts,style,ms=8,lw=3,label= '%s' %labels,color=plt.cm.viridis(1*1./len(histograms)))
	else:
		for i in range(len(histograms)):
			counts, bin_edges = np.histogram(histograms[i],bins=bins_num[i])
			bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
			plt.plot(bin_centres,counts,style,ms=8,lw=3,label='%s'%labels[i],color=plt.cm.viridis(i*1./len(histograms)))
			if directory == None:
				Pass
			else:
				# np.savetxt(directory+'%s_histogram.txt'%labels[i],bin_centres)
				np.savetxt(directory+'%s_histogram.txt'%labels[i],np.vstack((bin_centres,counts)))
	axes = plt.gca()
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.legend(prop={'size': 25})
	plt.xlabel('%s' % xlabel,size=22)
	plt.ylabel('%s' % ylabel,size=22)
	# plt.legend(bbox_to_anchor=(0,0,1,1),ncol=3)
	plt.legend()
	plt.tight_layout()
	xmin, xmax = axes.get_xlim()
	ymin, ymax = axes.get_ylim()
	if text!=None:
		for j in range(len(text)):
			plt.text(xmax*0.75,ymax*(0.7-j*0.1), '$ %s = %s $'%(text[j][0],np.around(text[j][1],decimals=2)),size=18)

def plot_scatter(x,y,marker='o',color='b',xlabel=None, ylabel=None,yscale='linear'):
	plt.figure(figsize=(8,6))
	plt.scatter(x,y,marker=marker,s=50)
	plt.xlabel('%s' % xlabel,size=22)
	plt.ylabel('%s' % ylabel,size=22)
	plt.yscale('%s' % yscale)
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.tight_layout()

def coordination_plot(filefolder):
	coordination_list = glob.glob(filefolder+'Coordination*.txt')
	coordinations = []
	N_ave=[]
	for filename in np.sort(coordination_list):
		print (filename)
		coor = np.loadtxt(filename)
		N_average = sum(x * y for x, y in zip(coor[0],coor[1]))/sum(coor[1])
		N_ave.append(N_average)
		# for i in range(coor[0].shape):
		# 	coor[0][i] * coor[1][i]
		# for i in range()
		coordinations.append(coor)

    # ncontacts_list = glob.glob(filefolder+'Contact*.txt')
    # ncontacts = []
    # for filename in np.sort(ncontacts_list):
	# 	con = np.loadtxt(filename)
	# 	ncontacts.append(con)
    # print len(coordination_list)
    # print len(ncontacts_list)
    # Cp_Cpcrit=np.array((1.6,3.2,3.9,4.1,4.5))
    # plot all sample force histograms
	plt.figure(figsize=(8,6))
	for i in range(len(coordinations)):
		# plt.plot(coordinations[i][0],coordinations[i][1]/coordinations[i][1].sum(),'o-', lw=2,color=plt.cm.viridis(i*1./len(coordinations)),
		# 	label='$C_p=%s$ $\phi_c=%s$ $\\bar{N}=%s$' %(np.sort(coordination_list)[i][67:71], np.sort(coordination_list)[i][-8:-4],np.around(N_ave[i],1)))
		# plt.plot(coordinations[i][0],coordinations[i][1]/coordinations[i][1].sum(),'o-', lw=2,color=plt.cm.viridis(i*1./len(coordinations)),
		# 	label='$%s$ $\phi_c=%s$ $\\bar{N}=%s$' %(Cp_Cpcrit[i], np.sort(coordination_list)[i][-8:-4],np.around(N_ave[i],1)))
		# plt.plot(ncontacts[i][0],ncontacts[i][1],'o--',lw=2, color=plt.cm.viridis(i*1./len(ncontacts)))
			# label='$C_p=%s$ $\phi_c=%s$' %(np.sort(coordination_list)[i][67:71], np.sort(coordination_list)[i][-8:-4]))
		plt.plot(coordinations[i][0],coordinations[i][1]/coordinations[i][1].sum(),'o-', lw=2,color=plt.cm.viridis(i*1./len(coordinations)),
			label='t=$%s$ $\\bar{N}=%s$' %(i,np.around(N_ave[i],1)))
		plt.xticks(size=20)
		plt.yticks(size=20)
		plt.legend(ncol=1,frameon=False,prop={'size':12})
		plt.xlabel("Number of neighbours, "r"$N$", size=25), plt.ylabel('P(N)', size=25)
	plt.tight_layout()
	plt.savefig(filefolder+'coordination_plot_pdf_Nave.pdf')
    # plt.close()

def ncontacts_plot(filefolder):
	ncontacts_list = glob.glob(filefolder+'*.txt')
	ncontacts = []
	Z_ave = []
	for filename in np.sort(ncontacts_list):
		print (filename)
		con = np.loadtxt(filename)
		Z_average = sum(x * y for x, y in zip(con[0],con[1]))/sum(con[1])
		Z_ave.append(Z_average)
		ncontacts.append(con)

    # plot all sample force histograms
    # Cp_Cpcrit=np.array((1.6,3.2,3.9,4.1,4.5))
    # print Cp_Cpcrit
	plt.figure(figsize=(8,6))
	for i in range(len(ncontacts)):
		# plt.plot(ncontacts[i][0],ncontacts[i][1]/ncontacts[i][1].sum(),'o-',lw=2, color=plt.cm.viridis(i*1./len(ncontacts)),
		# 	label='$C_p=%s$ $\phi_c=%s$ $\\bar{Z}=%s$' %(np.sort(ncontacts_list)[i][58:62], np.sort(ncontacts_list)[i][-8:-4],np.around(Z_ave[i],1)))
		# plt.plot(ncontacts[i][0],ncontacts[i][1]/ncontacts[i][1].sum(),'o-',lw=2, color=plt.cm.viridis(i*1./len(ncontacts)),
		# 	label='$%s$ $\phi_c=%s$ $\\bar{Z}=%s$' %(Cp_Cpcrit[i], np.sort(ncontacts_list)[i][-8:-4],np.around(Z_ave[i],1)))
		plt.plot(ncontacts[i][0],ncontacts[i][1]/ncontacts[i][1].sum(),'o-',lw=2, color=plt.cm.viridis(i*1./len(ncontacts)),
			label='t=$%s$ $\\bar{Z}=%s$' %(i,np.around(Z_ave[i],1)))
		plt.xticks(size=20)
		plt.yticks(size=20)
		plt.legend(ncol=1,frameon=False,prop={'size':12})
		plt.xlabel("Number of contacts, "r"$Z$", size=25), plt.ylabel('P(Z)', size=25)
	plt.tight_layout()
	plt.savefig(filefolder+'ncontacts_plot_pdf_Zave.pdf')
    # plt.close()

def plot_quiver(particle_center,local_f_sum,local_sum,imageC2,filename=None,theta=None):
	x = particle_center[:,0]
	y = particle_center[:,1]
	if theta == None:
		uf = np.array(local_f_sum)[:,0]
		vf = np.array(local_f_sum)[:,1]
	else:
		uf = np.array(local_sum) * np.cos(theta)
		vf = np.array(local_sum) * np.sin(theta)
	uf_norm = uf/np.sqrt(uf**2+vf**2)
	vf_norm = vf/np.sqrt(uf**2+vf**2)
	R = local_sum

	plt.figure(figsize=(9,9))
	plt.quiver(x,y,uf_norm,vf_norm,R,units='inches',scale=4)
	L = imageC2.shape[0]
	plt.xlim(0,L),plt.ylim(0,L)
	plt.xticks(size=20),plt.yticks(size=20)
	plt.imshow(imageC2,alpha=0.6, cmap=plt.cm.bone)
	if filename != None:
		plt.savefig(filename)

def draw_circles_index(xs, ys, rs, indexs,**kwargs):
	for x,y,r,index in zip(xs,ys,rs,indexs):
		circle = plt.Circle((x,y), radius=r, **kwargs)
		plt.gca().add_patch(circle)
		plt.gca().annotate('%s'%index,xy=(x,y),fontsize=8.,color='#FFFFFF',ha='center')



def pd_hist(p_rad,filefolder,bin_number=10):
    std_p = np.std(p_rad)
    pd = std_p / p_rad.mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(p_rad, bins=bin_number)
    ax.set_xlabel('diameter(pixels)'),ax.set_ylabel('counts')
    ax.text(0.1,0.8,r'$\bar{\sigma} = %s$'%np.round(p_rad.mean(),1),fontsize=15,transform=ax.transAxes)
    plt.text(0.1,0.7,'$PD = %s$'%np.round(p_rad.std()/p_rad.mean(),2),fontsize=15, transform=ax.transAxes)
    plt.savefig(filefolder+'p_rad_distribution.pdf')
    plt.close()

def draw_tracked_p_xy(particle_center,Image,filefolder,factor=1.1,z_space=1.,test_frame=False):
	hdiam= np.average(particle_center[:,3]*factor)
	print ('factor is ',factor)
	print ('mean radius is',hdiam)

	print (particle_center.shape[1])

	if particle_center.shape[1]<5:
		particle_center = np.insert(particle_center,4,0,axis=1)
	else:
		particle_center =particle_center
	# my_dpi=96.
	# Npixels = 256
	if test_frame == False:
		lz = Image.shape[0]
	else:
		lz = test_frame

	for frame in range(lz):
		fig, ax = plt.subplots(figsize=(9,9))
		# fig, ax = plt.subplots(figsize=(Npixels/my_dpi,Npixels/my_dpi),dpi=my_dpi)
		ax.imshow(Image[frame,:,:],vmax=100., cmap=plt.cm.gray)
	    # ax.axis('off')
	    # fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
		select=(particle_center[:,2]>frame-hdiam*z_space)*(particle_center[:,2]<frame+hdiam*z_space)
		index = (np.arange(len(particle_center)))[select]
		c=particle_center[select,:]
		x,y,z, r, intensity=c[:,0],c[:,1],c[:,2],c[:,3]*factor,c[:,4]
		h=np.abs(frame-z)
		a=np.sqrt(r**2-h**2)
		#
		# small = r <3.5 #3.5
		# big = r>= 3.5
		# draw_circles(x[big],y[big],a[big],facecolor='none', edgecolor='r')
		# draw_circles(x[small],y[small],a[small],facecolor='none', edgecolor='b')
		draw_circles(x,y,a,facecolor='none', edgecolor='r',lw=3)
		##### To write index of big droplets
		# for j in range(big.sum()):
		# 	plt.text(x[big][j],y[big][j],np.arange(len(particle_center))[select][big][j],fontsize=5,color='white')
		##### To write colloids tracked intensity on droplets
		# for j in range(len(x)):
		# 	var_idx = (np.arange(len(particle_center)))[select][j]
		# # 	print var_idx, variance[var_idx]
		# 	plt.text(x[j],y[j],var_idx),fontsize=5,color='white')
		# 	if intensity[j]/r[j]<=-0.25:
		# 		plt.text(x[j],y[j],np.round(intensity[j]/r[j],2),fontsize=5,color='yellow')
		# 	if intensity[j]/r[j]>-0.25:
		# 		plt.text(x[j],y[j],np.round(intensity[j]/r[j],2),fontsize=5,color='white')
		fig.savefig(filefolder+"/xy%04d.png"%frame)#,dpi=my_dpi)
		plt.close()

def draw_tracked_p_xy_label(particle_center,Image,mask,predicted,filefolder,
factor=1.1,z_space=1.,test_frame=False,pltImage2=False,alpha=0.5):
	hdiam= np.average(particle_center[:,3]*factor)
	print ('factor is ',factor)
	print ('mean radius is',hdiam)

	print (particle_center.shape[1])

	if particle_center.shape[1]<5:
		particle_center = np.insert(particle_center,4,0,axis=1)
	else:
		particle_center =particle_center
	if test_frame == False:
		lz = Image.shape[0]
	else:
		lz = test_frame
	clusters = np.unique(predicted)
	N_cluster = len(clusters)
	for frame in range(lz):

		fig, ax = plt.subplots()
		ax.imshow(Image[frame,:,:], cmap=plt.cm.gray)
		if pltImage2==True:
			ax.imshow(mask[frame,:,:],alpha=alpha)

		select=(particle_center[:,2]>frame-hdiam*z_space)*(particle_center[:,2]<frame+hdiam*z_space)
		index = (np.arange(len(particle_center)))[select]
		c=particle_center[select,:]
		x,y,z, r, intensity=c[:,0],c[:,1],c[:,2],c[:,3]*factor,c[:,4]
		h=np.abs(frame-z)
		a=np.sqrt(r**2-h**2)

		colors=iter(plt.cm.rainbow(np.linspace(0,1,N_cluster)))
		for j in range(N_cluster):
			cluster = clusters[j]
			cluster_idx = predicted[index]==cluster
			co=next(colors)
			draw_circles(x[cluster_idx],y[cluster_idx],a[cluster_idx],facecolor='none', edgecolor=co)

		fig.savefig(filefolder+"/xy%04d.pdf"%frame)#,dpi=my_dpi)
		plt.close()

def draw_tracked_p_xz(particle_center,Image,filefolder,factor=1.1,z_space=1.,test_frame=False):
	hdiam= np.average(particle_center[:,3]*factor)
	if test_frame == False:
		ly = Image.shape[1]
	else:
		ly = test_frame
	for frame in range(ly):
	    fig, ax = plt.subplots()
	    ax.imshow(Image[:,frame,:], cmap=plt.cm.gray)
	    select=(particle_center[:,1]>frame-hdiam*z_space)*(particle_center[:,1]<frame+hdiam*z_space)
	    c=particle_center[select,:]
	    x,y,z, r, intensity=c[:,0],c[:,1],c[:,2],c[:,3],c[:,4]
	    h=np.abs(frame-y)
	    a=np.sqrt(r**2-h**2)
	    draw_circles(x,z,a,facecolor='none', edgecolor='r')
	    fig.savefig(filefolder+"/xz%04d.pdf"%frame)#,dpi=my_dpi)
	    plt.close()

def draw_tracked_p_yz(particle_center,Image,filefolder,factor=1.1,z_space=1.,test_frame=False):
	print (particle_center.shape[1])
	if particle_center.shape[1]<5:
		particle_center = np.insert(particle_center,4,0,axis=1)
	else:
		particle_center =particle_center

	hdiam= np.average(particle_center[:,3]*factor)
	if test_frame == False:
		lx = Image.shape[2]
	else:
		lx = test_frame
	for frame in range(lx):
	    fig, ax = plt.subplots()
	    ax.imshow(Image[:,:,frame], cmap=plt.cm.gray)
	    select=(particle_center[:,2]>frame-hdiam*z_space)*(particle_center[:,2]<frame+hdiam*z_space)
	    c=particle_center[select,:]
	    x,y,z, r, intensity=c[:,0],c[:,1],c[:,2],c[:,3],c[:,4]
	    h=np.abs(frame-x)
	    a=np.sqrt(r**2-h**2)
	    draw_circles(y,z,a,facecolor='none', edgecolor='r')
	    fig.savefig(filefolder+"/yz%04d.pdf"%frame)#,dpi=my_dpi)
	    plt.close()
