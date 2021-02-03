import numpy as np
import pylab as plt
from scipy import ndimage
from copy import deepcopy
import glob
from tqdm import tqdm

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts

    #     histograms
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def hist_equal_z(_image,slices,target=200):
    from copy import deepcopy
    M=deepcopy(_image)
    for z in range(slices[0],slices[1]):
        M[z,:,:]=hist_match(_image[z, :,:], _image[target,:,:])
    return M

def hist_equal_z_phiNormalise(_image,slices,densityProfile,target=200):
    # densityProfile[0] - histogram values of phi_z
    # densityProfile[1] - bins of phi_z
    from copy import deepcopy
    M=deepcopy(_image)

    delta_rho = densityProfile[1][1]-densityProfile[1][0]
    target_idx = np.where(((densityProfile[1]>(target-delta_rho)) & (densityProfile[1]<(target+delta_rho))))
    # where target image is allocated in the bins

    for z in range(slices[0],slices[1]):
        M[z,:,:]=hist_match(_image[z, :,:], _image[target,:,:])
    return M

def image_substract(image):
    if len(image.shape) == 3:
        sub = []
        for i in range(image.shape[0]):
            sub_i = image[i,:,:].mean(axis=0)
            sub.append(sub_i)
        sub = np.array(sub)

        lz,ly,lx = image.shape[0],image.shape[1],image.shape[2]
        image_sub = np.zeros((lz,ly,lx))
        for i in range(lz):
            image_sub[i,:,:] = image[i,:,:]/sub[i]*sub[i].mean()

    if len(image.shape) == 2:
        sub = image.mean(axis=0)
        image_sub = image/sub*sub.mean()

    return image_sub

def threshold_top_bottom(_3dImage,top=180,bottom=0):
	Copy_I2=deepcopy(_3dImage)
	Copy_I2[_3dImage>=top]=0
	Copy_I2[_3dImage<bottom]=0

	return Copy_I2

def coordination(particle_center, maxdistance):
    from scipy.spatial.distance import pdist,squareform, cdist
    dr=pdist(particle_center)
    D=squareform(dr)
    D[D==0]=np.inf
    coordinations=(D<maxdistance).sum(axis=0)

    # neighbours = np.array(np.where(D<maxdistance))
    # neighbours = neighbours.T
    neighbours=[]
    neigh_pairs = []
    for i in range(D.shape[0]):
        neighs = np.where(D[i]<maxdistance)[0]
        neighbours.append((i,neighs))
        for j in neighs:
            neigh_pairs.append((i,j))
            # for m in range(len(neigh_pairs)):
            #     print m
            #     if set([i,j]) == set(neigh_pairs[m]):
            #         print set([i,j]) == set(neigh_pairs[m])
            #         pass
            #     else:
            #         neigh_pairs.append((i,j))
    # repeat = []
    # for i in range(len(neigh_pairs)):
    #     for j in range(i+1,len(neigh_pairs)):
    #         if set(neigh_pairs[i]) == set(neigh_pairs[j]):
    #             repeat.append((i,j))

    dist_nei = []
    for par in range(len(neighbours)):
        for neigh in neighbours[par][1]:
            dist_nei.append((par,neigh,D[par][neigh]))
    # print '# dist_nei is a list of (particle, its neighbour, distances between these two particles)'
    return (dist_nei, coordinations,neigh_pairs)

def middle_point(particle_center, nei_index, _3dImage):
    particle_middle = []
    particle_pairs = []
    for i in range(len(nei_index)):
        p1,p2 = nei_index[i][0],nei_index[i][1]
        x1,x2 = particle_center[p1][0],particle_center[p2][0]
        y1,y2 = particle_center[p1][1],particle_center[p2][1]
        z1,z2 = particle_center[p1][2],particle_center[p2][2]
        r1,r2 = particle_center[p1][3],particle_center[p2][3]

        d12 = nei_index[i][2]
        delta = (d12 - (r1+r2))/2
        x = (x2-x1) * (1-(r1+delta)/d12) + x1
        y = (y2-y1) * (1-(r1+delta)/d12) + y1
        z = (z2-z1) * (1-(r1+delta)/d12) + z1
        particle_middle.append((x,y,z))
        particle_pairs.append((p1,p2))

    "assign the middle point center to value 1, the rest of the image is 0"
    n = int(np.rint(particle_middle).max())

    mask = np.zeros((_3dImage.shape[0],_3dImage.shape[1],_3dImage.shape[2]))
    for i in range(len(particle_middle)):
        x = int(np.rint(particle_middle[i][0])) # tilt the image
        y = int(np.rint(particle_middle[i][1]))
        z = int(np.rint(particle_middle[i][2]))
        # if x > _3dImage.shape[0]:
 #            x = _3dImage.shape[0]
        mask[z][y][x]=1. #only integer index?
    return mask, particle_middle,particle_pairs

def middle_point_pt(particle_center, nei_index):
    from scipy.spatial.distance import euclidean
    particle_middle = []
    for i in range(len(nei_index)):
        p1,p2 = nei_index[i][0],nei_index[i][1]
        x1,x2 = particle_center[p1][0],particle_center[p2][0]
        y1,y2 = particle_center[p1][1],particle_center[p2][1]
        z1,z2 = particle_center[p1][2],particle_center[p2][2]
        r1,r2 = particle_center[p1][3],particle_center[p2][3]

        d12 = euclidean((x1,y1,z1),(x2,y2,z2))
        delta = (d12 - (r1+r2))/2
        x = (x2-x1) * (1-(r1+delta)/d12) + x1
        y = (y2-y1) * (1-(r1+delta)/d12) + y1
        z = (z2-z1) * (1-(r1+delta)/d12) + z1
        particle_middle.append((x,y,z))
    return particle_middle

def draw_spheres(middle,original_image, threshold,s_iter=3,dil_iter=1):
	# generate a structure close to a sphere
	mask = middle*(original_image>threshold)
	struct = ndimage.generate_binary_structure(3, 2)
	struct_iter = ndimage.iterate_structure(struct,s_iter).astype(int)
	spheres = ndimage.binary_dilation(mask, structure=struct_iter, iterations=dil_iter).astype(mask[:100,:,:].dtype)
	# convolution
	# sph = ndimage.filters.convolve(mask, struct_iter)

	print ('sphere size is', np.sum(struct_iter))
	return spheres, mask, np.sum(struct_iter)

def draw_shperes_fromCoordinates(particle_center, _3dImage, iteration):
    # assign the middle point to value 1, the rest of the image is 0
    n = int(np.rint(particle_center).max())
    mask = np.zeros((_3dImage.shape[0],_3dImage.shape[1],_3dImage.shape[2]))
    for i in range(len(particle_center)):
        x = int(np.rint(particle_center[i][0])) # tilt the image
        y = int(np.rint(particle_center[i][1]))
        z = int(np.rint(particle_center[i][2]))
        mask[x][y][z]=1. #only integer index?
    struct = ndimage.generate_binary_structure(3,2)
    struct_iter = ndimage.iterate_structure(struct,iterations=iteration).astype(int)
    spheres = ndimage.binary_dilation(mask, structure=struct_iter, iterations=1).astype(mask[:100,:,:].dtype)
    return spheres

def large_small_mask(spheres,labelled_spheres, sphere_size,threshold):
	mask_large = deepcopy(spheres)
	mask_small = deepcopy(spheres)
	print (threshold)
	small_area = sphere_size<threshold
	large_area = sphere_size>=threshold
	to_be_zeroed_small = small_area[labelled_spheres]
	to_be_zeroed_large = large_area[labelled_spheres]
	mask_large[to_be_zeroed_small] = 0
	mask_small[to_be_zeroed_large] = 0
	return mask_large, mask_small

def label_mask(mask):
	labelled_mask, num_lab_mask = ndimage.label(mask)
	mask_size = ndimage.sum(mask,labelled_mask, range(0,num_lab_mask+1))
	return labelled_mask, num_lab_mask, mask_size

def binary_image(image,threshold,binary=True):
    #### binary=True, return [0,1] images
    #### binary=False, return [0,values] images
    from copy import deepcopy
    bin_image = deepcopy(image)
    bin_image[bin_image<threshold]=0
    if binary==True:
        bin_image[bin_image>=threshold]=1
    return bin_image

def find_smallPx_inSphere(labelled_spheres,labelled_raw):
    #### labelled_sphere is the mask with sphere_size
    #### labelled_raw is the mask with Ostu thresholding of imageC2
    # con_in_sphere =[]
    small_px = []
    for i in tqdm(range(1,labelled_spheres.max())):  #### 0 is the background
        num = np.unique(labelled_raw[labelled_spheres == i])

        nonzero_num = num[1:]
        if len(nonzero_num) > 1:
            small_px.append((i,nonzero_num))
    return small_px#,con_in_sphere

def remove_smallPx(small_px,volumes_raw,labelled_raw,sphere_size):
    contacts_mask = deepcopy(labelled_raw)
    small_remove=[]
    for i in tqdm(range(len(small_px))):
        sphere_idx = small_px[i][0]
        idx = small_px[i][1]
        vol = volumes_raw[idx]
        sorted_px =  np.sort(vol)[::-1] #### from big to small pixels
        sort_idx = np.argsort(vol)[::-1]
        _max_px, _2max_px = sorted_px[0],sorted_px[1]
        if sphere_size[sphere_idx]>263:
            if _max_px/_2max_px < 2:
                small_remove.append(idx[sort_idx][2:])
            else:
                small_remove.append(idx[sort_idx][1:])
        else:
            small_remove.append(idx[sort_idx][1:])

    for i in tqdm(range(len(small_remove))):
        tobe_zero = small_remove[i]
        for j in range(len(tobe_zero)):
            contacts_mask[contacts_mask==tobe_zero[j]]=0
    return contacts_mask

def force_pdf(image_contacts, mask, minimum_intensity):
    ROI = (image_contacts>minimum_intensity).astype(int)
    residualContacts =( ROI)*mask
    residualLabels=np.unique(residualContacts[residualContacts>0])
    volumeResiduals = ndimage.sum(residualContacts>0,residualContacts, residualLabels)
    volumeResiduals = volumeResiduals[volumeResiduals<800] #?????
    print (minimum_intensity,len(volumeResiduals))
#     print np.bincount(maxima)
    H,edges= np.histogram(volumeResiduals, bins=30)
    halfbwidth = (edges[1]-edges[0])/2
    x =( edges[:-1]+halfbwidth)
    xx =(x- x[H.argmax()])/(volumeResiduals.std()) # scaling with both peak value and std
    xxx =x/ volumeResiduals.mean()  # scaling with peak value
    xxxx = x/x[5:][H[5:].argmax()] #[1:] in case the max p(f) is at x=0
    print (H[5:].argmax(),x[5:][H[5:].argmax()])
    return x, xx,xxx,xxxx, H, volumeResiduals,minimum_intensity


def gauss(x,A,mu, sigma):
    from scipy.optimize import curve_fit
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
# print "Optimised parameters",popt
# print "Covariance matrix",pcov

def force_plot(histograms, otsu, threshold_range,otsu_threshold, scaling=0,fitting=True):
    fig = plt.figure(figsize=(8,6))
    m = int(scaling)
    # h[0] - raw distribution      h[2] - bin value
    # h[1] - scaled by peak value    h[4] - contact sizes
    for i,h in enumerate(histograms):
        plt.plot(h[m], h[4],'o-',label='threshold=%d'%threshold_range[i], color=plt.cm.viridis(i*1./len(histograms)))

    plt.plot(otsu[m],otsu[4],'-k',lw=8, label="otsu threshold = %d"%otsu_threshold)

    # popt,pcov = curve_fit(gauss, otsu[m],otsu[2])
    # popt2,pcov2 = curve_fit(gauss, histograms[3][m],histograms[3][2])
    if fitting == True:
        plt.plot(otsu[m],gauss(otsu[m],*popt), 'r--', lw=2,label='Gaussian fit')
        plt.plot(histograms[3][m],gauss(histograms[3][m],*popt2), 'r--', lw=2,label='Gaussian fit%d'%threshold_range[3])

    if m == 0:
        plt.xlabel(r"$f$", size=20)
        # plt.xlim(0,600)
    if m == 1:
        plt.xlabel(r"$(f-f_{max})/\sigma_f$", size=20)
    if m == 2:
        plt.xlabel(r"$f/<f>$", size=20)
    if m == 3:
        plt.xlabel(r"$f/f_{max}$", size=20)

    plt.yscale('log')
    plt.ylim(1,30000)
    plt.ylabel('counts', size=20)
    plt.xticks(size=15),plt.yticks(size=15)
    plt.legend(loc=9,ncol=3,frameon=False)
    plt.tight_layout()
    return scaling#, popt, popt2
    # m=0 no scaling, m=1 scaled by sd, m=2 scaled by the average value

def forcePDF_plot(filefolder, scaling=False):
    m = int(scaling)
    # reading all forcepdf filenames
    force_list = glob.glob(filefolder+'*.txt')
    # sort filenames using Cp
    Cp_list = []
    for i in range(len(force_list)):
        Cp = force_list[i][-5]
        print (Cp)
        # Cp = force_list[i][66:70]
        Cp_list.append(Cp)
    sortedCp = np.argsort(Cp_list)
    sortedForce_list = np.array(force_list)[sortedCp]

    forcePDFs = []
    # for filename in force_list:
    for filename in sortedForce_list:
        print (filename)
        forcepdf = np.loadtxt(filename)
        forcePDFs.append(forcepdf)
    forcePDFsarr = np.array(forcePDFs)

    # plot all sample force histograms
    # Cp_Cpcrit=np.array((0,1,2,3,4,5))
    # Cp_Cpcrit=np.array((1.6,3.2,3.9,4.1,4.5,6))
    plt.figure(figsize=(8,6))
    for i in range(len(forcePDFs)):
        # plt.plot(forcePDFs[i][:,m],forcePDFs[i][:,2]/forcePDFs[i][:,2].sum(),'o-', color=plt.cm.viridis(i*1./len(forcePDFs)),
        #          label='$\phi_c=%s$  $C_p=%s$' %(sortedForce_list[i][60:64], sortedForce_list[i][66:70]))
        plt.plot(forcePDFs[i][:,m],forcePDFs[i][:,2]/forcePDFs[i][:,2].sum(),'o-', color=plt.cm.viridis(i*1./len(forcePDFs)),
                 label='t=$%s$' % sortedForce_list[i][-5])
        # popt,pcov = curve_fit(gauss, forcePDFs[i][:,m],forcePDFs[i][:,2]/forcePDFs[i][:,2].sum())
        # plt.plot(forcePDFs[i][:,m],gauss(forcePDFs[i][:,m],*popt), '--', lw=2,
        #         color=plt.cm.viridis(i*1./len(forcePDFs)),label='fit=%s'%round(popt[-1],2))
        # plt.plot(forcePDFs[i][:,m],forcePDFs[i][:,2]/forcePDFs[i][:,2].sum(),'o-', color=plt.cm.viridis(i*1./len(forcePDFs)),
        #          label='$%s$ $\phi_c=%s$' %(Cp_Cpcrit[i], sortedForce_list[i][60:64]))
    plt.yscale('log')
#         plt.ylim(0,50000)
    # plt.ylabel('counts', size=25)
    plt.xticks(size=20),plt.yticks(size=20)
    plt.legend(ncol=2,frameon=False,prop={'size':15},loc='best')
    if m == 0:
        plt.xlabel(r"$f$", size=25)
        plt.ylabel('P($f$)', size=25)
        plt.tight_layout()
        plt.savefig(filefolder+'forcePDFs_noscaling.pdf')
    if m == 1:
        plt.xlabel(r"$(f-f_{peak})/f_{std}$", size=25)
        plt.ylabel('P', size=25)
        # lt.xlabel(r"$f/f_{peak}$", size=25)
        # plt.ylabel('P($f/f_{peak}$)', size=25)
        plt.tight_layout()
        plt.savefig(filefolder+'forcePDFs_scaling_%s_new.pdf'%scaling)
    # plt.close()

'save histograms of force distributions'
def write_histogram_file(filename, histogram):
    f = open(filename+'.txt','w')
    f.write('%s'%histogram[-1]+'\n')
    for i in range(len(histogram[0])):
        f.write('%s %s %s %s %s %s'%(histogram[0][i],histogram[1][i],histogram[2][i],histogram[3][i],histogram[4][i],'\n'))
    f.close()

    ff = open(filename+'_residualSize.txt','w')
    ff.write('%s'%histogram[-1]+'\n')
    for j in range(len(histogram[-2])):
        ff.write('%s %s'%(histogram[-2][j],'\n'))
    ff.close()

def find_nan(tracked_centers):
    nan_li = []
    for i in range(len(tracked_centers)):
        if np.isnan(tracked_centers[i]).any():
            nan_li.append(i)
    nan_li = np.array(nan_li)
    new_centers = np.delete(np.array(tracked_centers),nan_li,0)
    return new_centers,nan_li

def label_contactCentre(image_contacts, mask, minimum_intensity):
    ROI = (image_contacts>minimum_intensity).astype(int)
    residualContacts =(ROI)*mask
    residualLabels=np.unique(residualContacts[residualContacts>0])

    coms=ndimage.measurements.center_of_mass(residualContacts>0,residualContacts,index=range(1,residualLabels.max()+1))
    maxs=ndimage.measurements.maximum_position(residualContacts>0,residualContacts,index=range(1,residualLabels.max()+1))
    new_coms,nan_li = find_nan(coms)
    new_maxs = np.delete(maxs,nan_li,0)
    shifted_maxs=(np.array(new_coms)+np.array(new_maxs))/2.

    not_nan = np.setdiff1d(residualLabels,(nan_li+1))
    print (len(nan_li),len(not_nan),len(residualLabels))
    contact_size = ndimage.sum(residualContacts>0,residualContacts, not_nan)

    return new_maxs,new_coms,shifted_maxs,contact_size,residualContacts,residualLabels,nan_li

def contact_centre_save(centres,contactSize,directory):
    contacts=np.vstack({tuple(row) for row in zip(centres[:,2],centres[:,1],centres[:,0],
            contactSize)})
    np.savetxt(directory,contacts)
