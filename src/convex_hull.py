import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import ndimage
from scipy.optimize import linprog
from skimage.morphology import disk,ball
from tqdm import tqdm
from contactAnalysis import draw_spheres as ds

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

#### Convex hull of tracked residual contacts
# def convex_hull_contacts(labelled_contacts):
#     convex_hull = deepcopy(labelled_contacts)
#     NonZeroIndex = np.array(np.where(convex_hull>0)).T
#     newImage = convex_hull[convex_hull>0]
#
#     for i in tqdm(range(1,labelled_contacts.max()+1)):
#         points = NonZeroIndex[np.where(newImage==i)[0]]
#         if len(points)>0:
#             z_min,z_max = points[:,0].min(),points[:,0].max()
#             y_min,y_max = points[:,1].min(),points[:,1].max()
#             x_min,x_max = points[:,2].min(),points[:,2].max()
#
#             test_cube = [[z,y,x] for z in range(z_min,z_max+1)
#                      for y in range(y_min,y_max+1)
#                      for x in range(x_min,x_max+1)]
#
#
#             for c in points:
#                 test_cube.remove(np.ndarray.tolist(c))
#
#             for m in range(len(test_cube)):
#                 p = test_cube[m]
#                 if in_hull(points,p):
#                     convex_hull[p[0],p[1],p[2]] = i
#
#     return convex_hull

def convex_hull_contacts(labelled_contacts):
    convex_hull = deepcopy(labelled_contacts)
    findobj = ndimage.find_objects(labelled_contacts.astype('int'))
    num_zero = 0
    for i in tqdm(range(labelled_contacts.max().astype('int'))):
        indices =findobj[i]
        if indices!=None:
            test_cube = labelled_contacts[indices]
            points = np.array(np.nonzero(test_cube)).T
            zero_idx = np.array(np.nonzero(test_cube==0))  #### create a binary mask first which is opposite value to the image
            for m in range(zero_idx.shape[1]):
                p = list([zero_idx[0][m],zero_idx[1][m],zero_idx[2][m]])
                if in_hull(points,p):       #### if pixel==0 is inside the convex hull made of (points)
                    convex_hull[indices][p[0],p[1],p[2]] = i+1  ### find_obj starts from 1 not 0
        else:
            num_zero += 1
    print ('volume of zero is', num_zero)
    return convex_hull

#### Using sobel filter to find particle edges from the sphere mask
def sobel_on_particleImage(_Image):
    sob_Image=[]
    for i in range(len(_Image)):
        im1=_Image[i,:,:].astype('int32')
        dx1=ndimage.sobel(im1,0)#,mode='constant',cval=1)
        dy1=ndimage.sobel(im1,1)#,mode='constant',cval=1)
        mag1=np.hypot(dx1, dy1)
        mag1 *=255.0/np.max(mag1)
        sob_Image.append(mag1)
    sob_Image = np.array(sob_Image)

    return sob_Image

def binSpheres_from_pcen(_ImageC1, pcen, radius_factor=0.9 ):
    lz,ly,lx = _ImageC1.shape
    sphere_mask = np.zeros((lz,ly,lx)) ### z,y,x
    sphere_sob = np.zeros((lz,ly,lx))

    edges_sphere = []

    # for j in np.linspace(0.9,1.,11):
    for j in [radius_factor]:
        middle_image = np.zeros((lz,ly,lx))
        n=0
#         for i in tqdm(range(1)):
        for i in tqdm(range(len(pcen))):
            px = int(np.rint(pcen[i][0]))
            py = int(np.rint(pcen[i][1]))
            pz = int(np.rint(pcen[i][2]))
            rad = int(np.rint(pcen[i][3]*j))

            xmin,ymin,zmin = max(0,px-rad),max(0,py-rad),max(0,pz-rad)
            xmax,ymax,zmax = min(px+rad+1,lx),min(py+rad+1,ly),min(pz+rad+1,lz)

            _x = slice(xmin,xmax,1)  ##### slice index of the cube in x
            _y = slice(ymin,ymax,1)
            _z = slice(zmin,zmax,1)
            cube = sphere_mask[_z,_y,_x]

            if cube.shape==ball(rad).shape:
                middle_image[_z,_y,_x]=ball(rad)
                  #### sobel on one sphere

            if (np.array(cube.shape)==np.array(ball(rad).shape)).sum()>1:
                cubeLz,cubeLy,cubeLx = cube.shape
                if cubeLz < ball(rad).shape[0]:
                    if zmax == lz:
                        middle_image[_z,_y,_x]=ball(rad)[:cubeLz,:,:]
                    else:
                        middle_image[_z,_y,_x]=ball(rad)[-cubeLz:,:,:]
                elif cubeLy < ball(rad).shape[0]:
                    if ymax == ly:
                        middle_image[_z,_y,_x]=ball(rad)[:,:cubeLy,:]
                    else:
                        middle_image[_z,_y,_x]=ball(rad)[:,-cubeLy:,:]
                elif cubeLx < ball(rad).shape[0]:
                    if xmax == lx:
                        middle_image[_z,_y,_x]=ball(rad)[:,:,:cubeLx]
                    else:
                        middle_image[_z,_y,_x]=ball(rad)[:,:,-cubeLx:]

                edges_sphere.append((_z,_y,_x))
                n+=1

            sob_middle = sobel_on_particleImage(middle_image[_z,_y,_x])
            nonzero_idx = np.array(np.where(sob_middle>0))

            nonzero_idx_all = deepcopy(nonzero_idx)
            nonzero_idx_all[0] = nonzero_idx[0]+zmin
            nonzero_idx_all[1] = nonzero_idx[1]+ymin
            nonzero_idx_all[2] = nonzero_idx[2]+xmin

            nonzero_idx = np.ndarray.tolist(nonzero_idx)
            nonzero_idx_all = np.ndarray.tolist(nonzero_idx_all)

            sphere_sob[nonzero_idx_all] = 1
#             sphere_sob[nonzero_idx_all] = sob_middle[nonzero_idx]


        lab_sphere_mask, num_sphere_mask, sphere_mask_size = ds.label_mask(sphere_sob)
        print ('sphere at edges',n,',' ,'factor',j,sphere_mask_size)
        print ('sphere drawn',num_sphere_mask, ',','tracked centers',len(pcen))

        return sphere_sob, num_sphere_mask
