from IPython.display import display,clear_output
from ipywidgets import widgets, interact, Output
from read_lif import Reader, Images
import pandas as pd
import numpy as np

class Select:
    def __init__(self,data, params):
        self.data = data
        self.which = params['which']
        self.initial = params['initial']
        self.nz = params['nz']
        self.edge_cut = params['edge_cut']

        if self.which == 'Lif':
            assert '.lif' in self.data, ('Input format does not correspond to a lif file')
            reader = Reader(self.data)
            self.files = reader.getSerieInfo()
            assert len(self.files) != 0, 'Lif file is empty'
            self.output_widget = widgets.Output()
            self.widget = widgets.VBox()
            self.selection = None
            self.main()
            self.display_lif(self.selection)
        elif self.which == 'Images':
            self.fmt = params['fmt']
            assert self.fmt in self.data, ('Input format does not correspond to a tiff file')
            imgs = Images(self.data, self.initial,self.nz,self.edge_cut)
            self.xyz = imgs.read_images()
            self.display_tiff()

    def main(self):
        self.options = []
        for i in self.files:
            dimensions = self.files[i]['Dimensions']
            if len(dimensions) == 2:
                self.options.append('{}, {}D, N Channels:{}, X:{}, Y:{}'.format(self.files[i]['Filename'],
                                                                    self.files[i]['N_dimensions'],
                                                                    self.files[i]['N_channels'],
                                                                    dimensions[0][1],dimensions[1][1]))
            else:
                self.options.append('{}, {}D, N Channels:{}, X:{}, Y:{}, Z:{}'.format(self.files[i]['Filename'],
                                                                    self.files[i]['N_dimensions'],
                                                                    self.files[i]['N_channels'],
                                                                    dimensions[0][1],dimensions[1][1],dimensions[2][1]))

        self.series = widgets.Dropdown(layout={'width': 'max-content'},description = 'Select Serie',options=self.options,value=None)
        self.widget.children = [self.series,self.output_widget]
        self.series.observe(self.update, ['value'])

    def update(self,_):
        self.selection = self.options.index(self.series.value)
        self.display(self.selection)

    def display_lif(self,selection):
        info = pd.DataFrame(np.nan,index=[],columns=['Filename','Dimensions','X','Y','Z','Channels'])
        if selection != None:
            clear_output()
            display(self.widget)
            info.at[0,'Filename'] = self.files[selection]['Filename']
            info.at[0,'Dimensions'] = self.files[selection]['N_dimensions']
            info.at[0,'X'] = self.files[selection]['Dimensions'][0][1]
            info.at[0,'Y'] = self.files[selection]['Dimensions'][1][1]
            info.at[0,'Z'] = self.files[selection]['Dimensions'][2][1]
            chs = ''
            for ch in self.files[selection]['Channels']:
                chs += '{}'.format(ch)
            info['Channels'] = chs
            display(info.head())

    def display_tiff(self):
        info = pd.DataFrame(np.nan,index=[],columns=['X','Y','Z'])
        info.at[0,'X'] = self.xyz.shape[0]
        info.at[0,'Y'] = self.xyz.shape[1]
        info.at[0,'Z'] = self.xyz.shape[2]
        display(info.head())

    def return_tiff(self):
        return self.xyz


class MainMenu:
    '''
    Generates interactive menu of the main tracking options
    '''
    def __init__(self,media = 'Tiff'):
        self.media = media
        self.output_widget = widgets.Output()
        self.options = widgets.VBox()
        self.main()
        self.update_options(self.media)

    def main(self):
        self.media_wid = widgets.ToggleButtons(options=['Lif', 'Images'], description='Media Input', value = 'Lif')
        self.media_wid.observe(self.media_update, ['value'])

        # self.mode_wid = widgets.ToggleButtons(options=['multi', 'single'], description='Tracking mode',tooltips=['Multiple frame tracking', 'Single frame tracking. Which singles must be specified'])
        # self.mode_wid.observe(self.mode_update, ['value'])

        self.fmt = widgets.Dropdown(options = ['.tif','.png','.jpg'], description  = 'Frame format')

        self.initial_frame = widgets.IntText(description='Initial frame')

        self.nframes = widgets.IntText(description='Z Frames',value = 5)

        self.k = widgets.FloatSlider(min = 1,max = 10,step = 0.2,description = 'k (Blur)',value = 1.6)

        self.edge_cutoff = widgets.FloatText(description = 'Edge cut')

        self.plot = widgets.Checkbox(description='Show plots',value = True)

        self.options.children = [self.media_wid]

    def media_update(self,_):
        self.media= self.media_wid.value
        self.update_options(x = self.media_wid.value)

    # def mode_update(self,_):
    #     self.mode_option = self.mode_wid.value
    #     self.update_options(x = self.media_wid.value,y = self.mode_wid.value)


    def update_options(self,x):
        if self.media == 'Images':
            self.options.children = [
                self.media_wid,
                self.fmt,
                self.nframes,
                self.initial_frame,
                self.k,
                self.edge_cutoff,
                self.plot,
                self.output_widget
            ]
        if self.media == 'Lif':
            self.options.children = [
                self.media_wid,
                self.nframes,
                self.initial_frame,
                self.k,
                self.edge_cutoff,
                self.plot,
                self.output_widget
                ]


def return_params(options):
    '''
    reads the selected options from interactive menu and return these parameters in a dictionary
    '''
    o = [(options[i].value) for i in range(len(options)-1)]
    p = {}
    if o[0] == 'Lif':
        p['which']      = o[0]
        p['nz']         = o[1]
        p['initial']    = o[2]
        p['blur']       = o[3]
        p['edge_cut']   = o[4]
        p['plot']       = o[5]
        return p
    if o[0] == 'Images':
        p['which']      = o[0]
        p['fmt']        = o[1]
        p['nz']         = o[2]
        p['initial']    = o[3]
        p['blur']       = o[4]
        p['edge_cut']   = o[5]
        p['plot']       = o[6]
        return p

def interacetive_menu():
    mode = widgets.ToggleButtons(options = ['Normal','Interactive'],description='Display mode', value = 'Interactive')
    return mode


def cl_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    dc= widgets.FloatSlider(min=0.5,max=2.0,description = 'Dist cut',value = 1.25,step=0.25)
    mn= widgets.IntSlider(min=3,max=7,description = 'Num neigh',value = 5)
    mcs= widgets.FloatText(description = 'Min cluster size',value = 10)
    ids = widgets.Checkbox(description = 'Cluster Id', value = False)
    iso = widgets.Checkbox(description = 'Isolated', value = False)
    save = widgets.Checkbox(description = 'Save', value = False)
    return widgets.VBox([input,dc,mn,mcs,ids,iso])

def save():
    '''
    generates a menu for saving ditionaries as .p .txt .xyz or both files
    '''
    print 'Saving options'
    save = widgets.Dropdown(options = ['all','pickle/xyz','pickle/txt','txt/xyz','pickle','txt','xyz'],description = 'Format', value = 'txt')
    filename = widgets.Text(description = 'File name', placeholder='eg. raw_coords')
    return widgets.VBox([save,filename])

def boop_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    wd = widgets.Dropdown(description = 'What input data?', options = ['Features','Trajectories'])
    plot = widgets.Checkbox(description = 'Plot', value = True)
    return widgets.VBox([input,wd,plot])

def af_menu():
    a = widgets.Dropdown(options = ['S','OD','O','R'],value = 'S',description = 'Arena')
    save = widgets.Checkbox(value = False, description = 'Save')
    return widgets.VBox([a,save])

def af_c_menu():
    a = widgets.Dropdown(options = ['30','60','120'],value = '30',description = 'Well size')
    save = widgets.Checkbox(value = False, description = 'Save')
    return widgets.VBox([a,save])

def po_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    plot = widgets.Checkbox(description = 'Plot', value = True)
    save  = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,plot,save])

def track_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    max_disp = widgets.FloatSlider(min=0.1,max=11.0,description = 'Max disp',value = 0.5)
    memory = widgets.IntSlider(description = 'Memory',min=0,max = 5,value = 2)
    predict = widgets.Checkbox(description = 'Trajectory predictor', value  = False)
    rn1 = widgets.IntSlider(description = 'Range 1', min = 0,max=11,value = 3)
    rn2 = widgets.IntSlider(description = 'Range 2', min = 0,max=11,value = 3)
    return widgets.VBox([input,predict,max_disp,memory,rn1,rn2])

def filter_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    pinfilter = widgets.FloatSlider(min=0.1,max=11,description = 'Pinned cutoff',value = 0.5)
    ids = widgets.Checkbox(description = 'Show IDs', value = False)
    return widgets.VBox([input,pinfilter,ids])

def input():
    '''
    geneates a menu for reading data of specific format, either from .txt files or directly from dictionaries
    '''
    return widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')

def msd_menu():
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    plot = widgets.Checkbox(description = 'Plot', value = True)
    fit = widgets.Checkbox(description = 'Fit MSD', value = True)
    points = widgets.IntText(description = 'Points',value = 100)
    vel = widgets.FloatText(description = '$ \\upsilon$ (px $s^{-1}$)',value = 100)
    tau = widgets.FloatText(description = '$ \\tau $',value = 0.5)
    refs = widgets.Checkbox(description = 'Slope reference', value = True)
    kind = widgets.Dropdown(options = ['ABP', 'Quincke', 'Passive'], description = 'MSD fit type', value = 'ABP')
    save  = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,plot,fit,kind,points,tau,vel,refs,save])

def ov_menu():
    '''
    generates a menu for specific parameters to compute and fit the overlap Q(t) functions. See src.characterisation.overlap.py for details
    '''
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    plot = widgets.Checkbox(description = 'Plot', value = True)
    fit =widgets.Checkbox(description = 'Fit', value = True)
    points = widgets.IntText(description = 'No. of points',value = 100)
    tau = widgets.FloatText(description = '$ \\tau $',value = 1.)
    save  = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,plot,fit,points,tau,save])

def return_ov_params(options):
    '''
    returns parameters selected in ov_menu()
    '''
    o = [(options[i].value) for i in range(len(options))]
    p = {}
    p['input'] = o[0]
    p['plot'] = o[1]
    p['fit'] = o[2]
    p['points'] = o[3]
    p['tau'] = o[4]
    p['save'] = o[5]

    return p

def g6r_menu():
    '''
    generates a menu for specific parameters to compute and fit the time correlation of the hexagonal order. See src.characterisation.g6.py for details
    '''
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    nbins = widgets.IntText(description = 'Nbins', value = 100)
    bs = widgets.FloatText(description = 'Binwidth', value = 0.25)
    plot = widgets.Checkbox(description = 'Plot', value = True)
    save = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,nbins,bs,plot,save])

def return_g6r_params(options):
    '''
    returns parameters selected in g6r_menu()
    '''
    o = [(options[i].value) for i in range(len(options))]
    p = {}
    p['input'] = o[0]
    p['nbins'] = o[1]
    p['bs'] = o[2]
    p['plot'] = o[3]
    p['save'] = o[4]

    return p

def dim_menu():
    '''
    generates menu for the molecules notebook in order to display and save data from indtified dimers
    '''
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    plot = widgets.Checkbox(description = 'Plot', value = True)
    save = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,plot,save])

def return_dim_params(options):
    '''
    return parameters drom dim_menu()
    '''
    o = [(options[i].value) for i in range(len(options))]
    p = {}
    p['input'] = o[0]
    p['plot'] = o[1]
    p['save'] = o[2]
    return p

def wdim_menu():
    '''
    generates menu for the molecules notebook in order to display and save data of specific dimers
    '''
    input = widgets.Dropdown(description = 'Input data', options = ['Dictionary','Pickle','Text file'], value = 'Dictionary')
    which = widgets.IntText(description = 'Which dimmer')
    plot = widgets.Checkbox(description = 'Plot', value = True)
    save = widgets.Checkbox(description = 'Save', value = True)
    return widgets.VBox([input,which,plot,save])

def return_wdim_params(options):
    '''
    return parameters drom wdim_menu()
    '''
    o = [(options[i].value) for i in range(len(options))]
    p = {}
    p['input'] = o[0]
    p['which'] = o[1]
    p['plot'] = o[2]
    p['save'] = o[3]
    return p
