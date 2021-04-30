import sys
import os
import numpy as np
import pickle
from src.trackingParticles import track
from src.read_lif import Images
from src import utils

params = {}
params['tz']        = 744
params['nz']        = 25
params['nt']        = 2
params['blur']      = 1
params['k']         = 1.6
params['initial']   = 0
params['edge_cut']  = 0
params['fmt']       = 'tif'

directory = '/Volumes/Mac/'
filename  = 'EG_Water_60_40_centres.tif'
filename_ = filename[:-4]

data = directory + filename

imgs = Images(data, params)
stack = imgs.read_images()

tr =  track(stack,params,thres=-2)
features = tr.get_features()
pickle.dump(features, open(directory + 'features.p','wb'))
type_ = 'xyz'
which = 'centres'
utils.saving_xyz(directory,filename_,features,type_,which)
