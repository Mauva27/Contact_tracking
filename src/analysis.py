import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist

def find_neighbours(particle_radius, distance, cutoff, contact_centre):
	# Criteria, each contact MUST have TWO adjacent particle neighbours
	delta = 0.06* particle_radius
	d_tolerance =  particle_radius + delta
	incontact = distance < (d_tolerance)
	print (d_tolerance)
	print ('total distances', distance.shape[0]*distance.shape[1])
	print ('incontact distances',incontact[incontact==True].shape[0])

	Nc = len(contact_centre)
	neighs_particles = [] # search for the particle neighbours for each contact
	particle_idx=np.arange(distance.shape[0])
	con_multiNei_idx = []
	for con in range(Nc):
	    neighs_particles.append(particle_idx[incontact[:,con]])
	    con_multiNei_idx.append(con)
	print ('contacts have two neighbours', len(con_multiNei_idx))
	new_con = contact_centre[con_multiNei_idx]
	return neighs_particles

def link_particles(distance, contact_centre,contact_size,particles):
	# contact_centre_reverse = contacts[:,:3]
	# contact_centre = contact_centre_reverse[:,::-1]
    # contact coordinates are stored as z,y,x
	# contact_centre = contacts[:,:3]
	particle_centre = particles[:,:3]
	# contact_size = contacts[:,3]
	particle_size = particles[:,3]

	which_particles=[]
	for c in range(len(contact_centre)):
	    closest=np.argsort(distance[:,c])[:2]
	    which_particles.append((closest, distance[closest,c]) )

	forces=[]
	p_tb_connected=[]
	connections=[]
	p1_cen,p2_cen=[],[]

	for i in range(len(contact_centre)):   #     looping over the contacts
		linked_particles=which_particles[i][0]
		p1,p2=linked_particles[0],linked_particles[1]
		r1,r2=particle_size[p1],particle_size[p2]
		rtilde=r1*r2/(r1+r2)
		force=contact_size[i]/rtilde
		forces.append(force)
		p1_cen.append((particle_centre[p1,0],particle_centre[p1,1],particle_centre[p1,2]))
		p2_cen.append((particle_centre[p2,0],particle_centre[p2,1],particle_centre[p2,2]))
		p_tb_connected.append((i,p1,p2))
		connections.append((particle_centre[linked_particles,0],
							particle_centre[linked_particles,1],
							particle_centre[linked_particles,2]))

	# for i in range(len(contact_centre)):   #     looping over the contacts
	#     linked_particles=which_particles[i][0]
	#     if np.max(which_particles[i][1])<1000000000:
    #     # accept all close distance as contacts
	# 		p1,p2=linked_particles[0],linked_particles[1]
	# 		r1,r2=particle_size[p1],particle_size[p2]
	# 		rtilde=r1*r2/(r1+r2)
	# 		force=contact_size[i]/rtilde
	# 		forces.append(force)
	# 		p1_cen.append((particle_centre[p1,0],particle_centre[p1,1],particle_centre[p1,2]))
	# 		p2_cen.append((particle_centre[p2,0],particle_centre[p2,1],particle_centre[p2,2]))
	# 		p_tb_connected.append((i,p1,p2))
	# 		connections.append((particle_centre[linked_particles,0],
	# 							particle_centre[linked_particles,1],
	# 							particle_centre[linked_particles,2]))


	forces=np.array(forces)
	p_tb_connected=np.array(p_tb_connected)
	connections=np.array(connections)
	print (len(forces))
	return np.array(p1_cen), np.array(p2_cen), p_tb_connected, connections, forces
	#for each contact, choose the cloeset 2 particles,
	#record their index and distance to the contact

def find_force_vector(contact_centre, p1_cen, p2_cen,forces):
	from numpy import hypot
	# contact_centre_reverse = contacts[:,:3]
	# contact_centre = contact_centre_reverse[:,::-1]
	# contact_centre = contacts[:,:3]
	ac = np.zeros((contact_centre.shape[0],contact_centre.shape[1]))
	bc = np.zeros((contact_centre.shape[0],contact_centre.shape[1]))
	ab = p2_cen - p1_cen

	force_aligned = []
	ab_plot =[]
	for con in range(len(contact_centre)):
		##### force direction is contacts to particles(inwards)
		ac[con] = contact_centre[con] - p1_cen[con]
		bc[con] = contact_centre[con] - p2_cen[con]
		dotProduct = np.dot(ac[con],bc[con])

		ac_mag = hypot(hypot(ac[con][0],ac[con][1]),ac[con][2])
		bc_mag = hypot(hypot(bc[con][0],bc[con][1]),bc[con][2])
		ab_unit = ab[con]/np.linalg.norm(ab,axis=1)[con]


            # force_aligned.append((contact_centre[con],ab_unit*forces[con])) # vector ca, ab direction for particle A
            # force_aligned.append((contact_centre[con],(-ab_unit)*forces[con])) # vector cb same direction of ab
		ac_unit = ac[con]/np.linalg.norm(ac,axis=1)[con]
		bc_unit = bc[con]/np.linalg.norm(bc,axis=1)[con]
		force_aligned.append((contact_centre[con],ac_unit*forces[con]))
		force_aligned.append((contact_centre[con],bc_unit*forces[con]))

		ab_plot.append((p1_cen[con][0],p1_cen[con][1],p1_cen[con][2],
				ab[con][0],ab[con][1],ab[con][2],forces[con]))

	force_aligned = np.array(force_aligned) # filtered contact to particle vector with magnitude of contact sizes
	ab_plot = np.array(ab_plot)
	# np.savetxt(folder+'particle_middle/new/c2/ab_plot.txt',ab_plot)

	# force_vector: contacts centre(x,y,z), force (x,y,z)
	force_vector = np.reshape(force_aligned,(force_aligned.shape[0],6))
	print ('sum of all forces',np.sum(force_vector[:,-3:], axis=0)) # sum all the force vectors
	print (force_vector.shape)

	return force_vector

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
    edge_particle = np.unique(edge_particle)

    from copy import deepcopy
    pcen_inbox = deepcopy(particle_centre)
    pcen_inbox[edge_particle]=0
    pcen_inbox_index = np.delete(np.arange(0,len(particle_centre),1),edge_particle)
    # pcen_inbox = np.delete(pcen,edge_particle,0)
    print ('number of particles at the edges',edge_particle.shape[0])
    return edge_particle, pcen_inbox_index

def n_contacts_pparticle(p_tb_connected,force_vector,particles):
    # Calculate number of contacts per particle
    # p_idx = paritcle_index, contacts_index, forces vector(x,y,z)
    particle_centre = particles[:,:3]
    p_rad_mean = particles[:,3].mean()

    p_idx = []
    for i in range(len(p_tb_connected)):
        p_idx.append((p_tb_connected[i][1],p_tb_connected[i][0]))
        p_idx.append((p_tb_connected[i][2],p_tb_connected[i][0]))
    p_idx = np.concatenate((np.array(p_idx),force_vector[:,-3:]), axis=1)

    local_sum,local_sum_moduli,ncontacts,force_per_p=[],[],[],[]
    local_f_list = []
    local_f_sum = []
    # loop over all the particles
    for p in range(len(particles)):
    #     for every particle, find all the contacts (and forces) on it
        if abs(particle_centre[int(p_idx[p,0])]-256).max()< 100 * p_rad_mean:
            local_f=p_idx[p_idx[:,0]==p,-3:]
            c_idx = p_idx[p_idx[:,0]==p,1]
            local_f_list.append((p,local_f,c_idx))
            ncontacts.append(local_f.shape[0])
    #     this is a matrix nx3
    # First: compute the sum along the x,y,z directions (=net force), and its norm
            local_sum.append(np.linalg.norm(local_f.sum(axis=0)))
            local_f_sum.append(local_f.sum(axis=0))
    # Second: compute the norm of every force, and then sum
            local_sum_moduli.append(np.linalg.norm(local_f.T,axis=0).sum())
            force_per_p.append(local_f.sum(axis=0))
    # find the order of particles that sorts the sum of moduli
    order=np.argsort(local_sum_moduli)
    local_sum=np.array(local_sum)
    local_sum_moduli=np.array(local_sum_moduli)
    ncontacts=np.array(ncontacts)
    force_per_p = np.array(force_per_p)
    print ('sum of force',force_per_p.sum(axis=0))
    return ncontacts,local_sum,local_sum_moduli, order, local_f_list,local_f_sum

def coordination(particle_center, maxdistance):
	from scipy.spatial.distance import pdist,squareform, cdist
	dr=pdist(particle_center)
	D=squareform(dr)
	D[D==0]=np.inf
	coordinations=(D<maxdistance).sum(axis=0)
	return coordinations

def compression_tension(coordinations,ncontacts):
    non_contact = coordinations - ncontacts
    comp_tension_df = ncontacts - non_contact
    return comp_tension_df

def force_vector_unit(force_vector):
    force_vector_unit =[]
    for i in range(len(force_vector)/2):
        absolute_value = np.linalg.norm(force_vector[i*2][-3:])
        new_vector = force_vector[i*2][-3:]/absolute_value
        force_vector_unit.append(new_vector)
    force_vector_unit = np.array(force_vector_unit)

def percolation(Cluster,image,particles):
	###### check if the gel is percolating
    particle_centre = particles[:,:3]
    particle_rad = particles[:,3]
    biggest=np.bincount(Cluster).argmax()
    largest=particle_centre[Cluster==biggest]
    box_len = float(image.shape[1])
    print ('box length is',box_len)
    score= largest.ptp(axis=0)/box_len
    perc_criterion = (box_len- 2*particle_rad.mean())/box_len  # L - 2*sigma
    print ('criteria',perc_criterion)
    if np.any(score >perc_criterion ):
	    print ("It percolates",score)
    else:
	    print ("It does NOT percolate",score)

def fractal_dimension():
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    image_tri = rgb2gray(plt.imread('/Users/jundong/Downloads/Sierpinski.png'))

    # finding all the non-zero pixels
    pixels =[]
    for i in range(image_tri.shape[0]):
        for j in range(image_tri.shape[1]):
            if image_tri[i,j] > 0:
                pixels.append((i,j))

    Lx = image_tri.shape[1]
    Ly = image_tri.shape[0]
    print (Lx, Ly)
    pixels = np.array(pixels)
    plt.plot(pixels[:,1], pixels[:,0], '.', ms=0.01)
    plt.show()
    print (pixels.shape)

    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales = np.logspace(1,8, num=20, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    for scale in scales:
    #     print 'Scale ', scale
        H, edges = np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns),1)
    plt.plot(np.log(scales),np.log(Ns), 'o', mfc='none')
    plt.plot(np.log(scales),np.polyval(coeffs, np.log(scales)))
    plt.xlabel('log $\epsilon$')
    plt.ylabel('log N')
    # plt.imshow(image_tri).

    print ('the Hausdorff dimension is', -coeffs[0])

def particle_boxcentre(pcen,p_radius,image,centre_cutoff):
	# particles in the centre of box
	p_boxcen = []
	lx,ly,lz = image.shape[2],image.shape[1],image.shape[0]
	x_cen, y_cen, z_cen = lx/2,ly/2,lz/2
	box_cen_dist = centre_cutoff * p_radius
	for p in range(len(pcen)):
	    cendx = abs(x_cen-pcen[p,0])
	    cendy = abs(y_cen-pcen[p,1])
	    cendz = abs(z_cen-pcen[p,2])
	    if all((cendx,cendy,cendz) < box_cen_dist):
	        p_boxcen.append(p)
	p_boxcen = np.array(p_boxcen)
	print (pcen[p_boxcen].max(),pcen[p_boxcen].min())
	print ('number of particle in the centre of the box',p_boxcen.shape[0])

	return p_boxcen

def particle_smallbox():
	# Check sum of forces in each small region
	x_box ,y_box ,z_box = [],[],[]
	delta =110
	R_x=np.arange(0,551, delta)
	numbox = len(R_x)
	for x in pcen[:,0]:
	    for n in range(0,(numbox-1)):
	        if R_x[n]<= x < R_x[n+1]:
	            x_box.append((x,n))
	for y in pcen[:,1]:
	    for n in range(0,(numbox-1)):
	        if R_x[n]<= y < R_x[n+1]:
	            y_box.append((y,n))
	for z in pcen[:,2]:
	    for n in range(0,(numbox-1)):
	        if R_x[n]<= z < R_x[n+1]:
	            z_box.append((z,n))
	# print np.insert(np.array(x_box),1,np.array(y_box)[:,0], axis=1)
	box = np.concatenate((np.array(x_box),np.array(y_box),np.array(z_box)),axis=1)
	# print len(box[:,1:6:2])
