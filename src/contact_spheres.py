import numpy as np
import pylab as pl
from skimage import filters
from scipy import ndimage
from copy import deepcopy
import glob
from tqdm import tqdm
from scipy.spatial.distance import pdist,squareform, cdist
from copy import deepcopy
import utils


class Force:
    '''
    Main algorithm for the detection of contacts by means of pixel volumes.
    Params:
        - cimg      :   array. Original contacts image array
        - thres     :   int. Min intensity value for pixels, e.g. from Otsu filter
        - s_iter        :   int. Number of iterations for the structure
        - d_iter        :   int. Number of titerations for the image dilation
        - dist_thres    :   Max distance for identifyig neighs. Usually first minimum of g(r)

    '''
    def __init__(self,params,centres,contact_imgs,max_dist):
        self.params = params
        self.centres = centres
        self.contact_imgs = contact_imgs
        self.max_dist = max_dist
        self.d_iter = params['d_iter']
        self.s_iter = params['s_iter']

    def get_neighs(self,centres):
        dr = pdist(centres)
        D = squareform(dr)
        D[D==0] = np.inf
        coordination = (D < self.max_dist).sum(axis=0)
        neighbours = {}
        for i in range(D.shape[0]):
            this_neigh = np.where(D[i] < self.max_dist)[0]
            if len(this_neigh) != 0:
                neigh_pairs = []
                for n in this_neigh:
                    neigh_pairs.append([n, D[i][n]])
                neighbours[i] = np.vstack(neigh_pairs)
        return neighbours

    def get_middle_points(self,neighs,centres):
        imgsize = self.img.shape
        assert len(imgsize) == 3, ('A three-dimensional array is needed')
        mask = np.zeros((imgsize[0], imgsize[1], imgsize[2]))
        particle_middle = []
        middle = {}
        for i, n  in enumerate(neighs):
            if len(neighs[n]) != 0:
                p1,p2 = n,neighs[n][:,0].astype('int')
                x1,x2 = centres[p1][0],centres[p2][:,0]
                y1,y2 = centres[p1][1],centres[p2][:,1]
                z1,z2 = centres[p1][2],centres[p2][:,2]
                r1,r2 = centres[p1][3],centres[p2][:,3]

                d12 = neighs[n][:,1]
                delta = (d12 - (r1+r2)) / 2
                x = (x2-x1) * (1-(r1+delta)/d12) + x1
                y = (y2-y1) * (1-(r1+delta)/d12) + y1
                z = (z2-z1) * (1-(r1+delta)/d12) + z1
                particle_middle.append(np.vstack((x,y,z)).T)
                middle[n] = np.vstack((p2,x,y,z)).T

        particle_middle = np.vstack(particle_middle)
        for j in range(particle_middle.shape[0]):
            x_ = np.rint(particle_middle[j][0]).astype('int')
            y_ = np.rint(particle_middle[j][1]).astype('int')
            z_ = np.rint(particle_middle[j][2]).astype('int')
            mask[x_][y_][z_] = 1.
        return mask, middle

    def draw_spheres(self,mid_mask,thres):
        all_mask = mid_mask * (self.img > thres)
        struct = ndimage.generate_binary_structure(3, 2)
        struct_iter = ndimage.iterate_structure(struct,self.s_iter).astype(int)
        spheres = ndimage.binary_dilation(all_mask, structure = struct_iter, iterations=self.d_iter).astype(all_mask[:,:,:].dtype)
        struct_size = np.sum(struct_iter)
        return spheres, all_mask, struct_size

    def label_mask(self,mask):
    	labelled_mask, num_lab_mask = ndimage.label(mask)
    	mask_size = ndimage.sum(mask,labelled_mask, range(0,num_lab_mask+1))
    	return labelled_mask, num_lab_mask, mask_size

    def label_contacts(image_contacts, mask, thres):
        ROI = (image_contacts > thres).astype(int)
        residualContacts = (ROI) * mask
        residualLabels = np.unique(residualContacts[residualContacts > 0])

        coms=ndimage.measurements.center_of_mass(residualContacts > 0,residualContacts,index = range(1,residualLabels.max()+1))

        maxs = ndimage.measurements.maximum_position(residualContacts > 0,residualContacts,index = range(1,residualLabels.max()+1))

        new_coms,nan_li = find_nan(coms)
        new_maxs = np.delete(maxs,nan_li,0)
        shifted_maxs=(np.array(new_coms) + np.array(new_maxs)) / 2.

        not_nan = np.setdiff1d(residualLabels,(nan_li + 1))
        print (len(nan_li),len(not_nan),len(residualLabels))
        contact_size = ndimage.sum(residualContacts > 0,residualContacts, not_nan)

        return new_maxs,new_coms,shifted_maxs,contact_size,residualContacts,residualLabels,nan_li

    def get_contacts(self,neighs,mask):
        ROI = (self.img > self.otsu_thres).astype(int)
        residualContacts = ROI * mask
        residualLabels = np.unique(residualContacts[residualContacts > 0])

        coms = ndimage.measurements.center_of_mass(residualContacts > 0,residualContacts,index = range(1,residualLabels.max()+1))
        maxs = ndimage.measurements.maximum_position(residualContacts > 0,residualContacts,index = range(1,residualLabels.max()+1))

        new_coms,nan_li = utils.find_nan(coms)
        new_maxs = np.delete(maxs,nan_li,0)
        shifted_coms = (np.array(new_coms) + np.array(new_maxs)) / 2.

        not_nan = np.setdiff1d(residualLabels,(nan_li + 1))
        contact_sizes = ndimage.sum(residualContacts > 0,residualContacts, not_nan)
        contact_sizes = contact_sizes[contact_sizes < 800] #Some arbitrary limit
        contact_sizes = contact_sizes.reshape(-1,1)
        new_coms_ = np.hstack((new_coms,contact_sizes))
        new_maxs_ = np.hstack((new_maxs,contact_sizes))
        shifted_coms_ = np.hstack((shifted_coms,contact_sizes))

        # contacts['coms'] = new_coms
        # contacts['maxs'] = new_maxs
        # contacts['shifted_coms'] = shifted_coms
        # contacts['contact_sizes'] = contact_sizes
        return new_coms_, new_maxs_, shifted_coms_, contact_sizes

    def force_pdf(self,contact_sizes):#,mask, thres):
        # ROI = (self.img > thres).astype(int)
        # residualContacts = ROI * mask
        # residualLabels = np.unique(residualContacts[residualContacts > 0])
        # volumeResiduals = ndimage.sum(residualContacts > 0,residualContacts, residualLabels)
        # volumeResiduals = volumeResiduals[volumeResiduals < 800] #Some arbitrary limit
        hist,bedges = np.histogram(contact_sizes, bins=30,normed = True)
        halfbwidth = (bedges[1] - bedges[0]) / 2
        bcentres = (bedges[:-1] + halfbwidth)
        bcentres_scaled = bcentres / np.mean(contact_sizes)  # scaling with peak value
        bcentres_scaled_ = (bcentres - bcentres[hist.argmax()]) / (np.std(contact_sizes)) # scaling with both peak value and std
        bcentres_adj = bcentres / bcentres[5:][hist[5:].argmax()] #[1:] in case the max p(f) is at x=0
        return bcentres, bcentres_scaled_, bcentres_scaled ,bcentres_adj, hist#, volumeResiduals,minimum_intensity

    def get_forces(self,neighs,mask,contact_sizes,mode,range=10,step=4):
        thr_low = self.otsu_thres - range
        thr_high = self.otsu_thres + range
        delta_thr = step
        histograms = {}
        if mode == 'scan':
            for intensity in range(thr_low,thr_high,delta_thr):
                _,_,_,contact_sizes_ = self.get_contacts(neighs,mask)
                histograms['otsu_{}'.format(intensity)] = self.force_pdf(contact_sizes_)
        elif mode == 'normal':
            histograms['otsu_{}'.format(self.otsu_thres)] = self.force_pdf(contact_sizes)
        return histograms

    def get_todo(self,mode = 'normal'):
        contacts = {}
        for t in self.contact_imgs:
            contacts.setdefault(t,{})
            self.img = deepcopy(self.contact_imgs[t])
            edges = deepcopy(self.img)
            bgd =  filters.threshold_otsu(edges.ravel())
            self.otsu_thres = filters.threshold_otsu(edges[edges>bgd].ravel())
            centres = self.centres[t]['centres']
            neighs = self.get_neighs(centres)
            middle_mask, particle_middle = self.get_middle_points(neighs,centres)
            spheres, all_mask, struct_size  = self.draw_spheres(middle_mask,thres=0)
            labelled_mask, num_lab_mask, mask_size = self.label_mask(spheres)
            new_coms, new_max, shifted_coms, contact_sizes = self.get_contacts(neighs,labelled_mask)
            if mode == 'scan':
                contact_sizes_ == None
                force_hists = self.get_forces(neighs,labelled_mask,contact_sizes_,mode,range=10,step=4)
            else:
                force_hists = self.get_forces(neighs,labelled_mask,contact_sizes,mode)
            contacts[t]['coms'] = new_coms
            contacts[t]['max_com'] = new_max
            contacts[t]['shifted_coms'] = shifted_coms
            contacts[t]['contact_sizes'] = contact_sizes
            contacts[t]['force_hist'] = force_hists
            contacts[t]['mask'] = labelled_mask

        return contacts
