# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


#Name model
modelname = 'heterog_1000'


import os
from pathlib import Path
import sys
import numpy as np
import flopy
import SGD
import config
import datetime

print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))
# Name model

# repo = Path('/scratch/users/ianpg/henry')
repo = Path('/Users/ianpg/Documents/ProjectsLocal/SWIsmall')
workdir = repo.joinpath('work')
figdir = workdir.joinpath('figs')
datadir = repo.joinpath('data')
objdir = repo.joinpath('data', 'objs')
model_ws = repo.joinpath('work', modelname)    
    
for p in (workdir,figdir,datadir,objdir,model_ws):
    if not p.exists():
        p.mkdir()

# sys.path.append(repo.joinpath('notebook').as_posix())
sw_exe = config.swexe  # set the exe path for seawat
print('Model workspace:', model_ws)

def load_obj(dirname,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)

def save_obj(dirname,obj,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def make_timestamp(YMD=True,HM=True):
    import datetime
    if YMD:
        ymd = '%Y%m%d'
    else:
        ymd = ''
    if HM:
        hm = '%H%M'
    else:
        hm = ''
    if YMD and HM:
        sep = '_'
    else:
        sep = ''
    return datetime.datetime.now().strftime('{}{}{}'.format(ymd,sep,hm))

#Create new MC_file
def create_MC_file():
    ts = make_timestamp()
    MC_dir = Path(os.path.join(m.model_ws, 'MC_expt_' + ts))
    if not MC_dir.exists():
        MC_dir.mkdir()
    m.MC_file = MC_dir.joinpath('expt.txt')
    with m.MC_file.open('w') as wf:
        wf.close
    print(m.MC_file)
    return

#nearest value in array
def find_nearest(array,value):
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    idx.astype('int')
    return array[idx]

#take distance in meters, return column in model
def loc_to_col(locs):
    cols = [int(find_nearest((np.arange(ncol)*delc),loc)) for loc in locs]
    return cols

#make a line across the grid
def get_line(start, end,allrows=1,nrow=None):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        if allrows==1:
            if nrow is None:
                nrow = m.nrow
            for row in range(nrow):
                coord = (y, row, x) if is_steep else (x, row, y)
                points.append(coord)
        else:
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


#make all cells=0 above the line from get_line()
#Calculate freshwater head based on column of saltwater above each node (rho*g*z)
def shade_above(nlay,nrow,ncol,point_list,third_dim=1):
    import numpy as np
    grd = np.ones((nlay,nrow,ncol),dtype='int')
    ocean_hf = []
    if len(point_list)==0:
        return grd,ocean_hf
    for (lay,row,col) in point_list:
        grd[lay,:,col] = -1 #assign ocean ibound to -1
        grd[:lay,:,col] = 0 #assign cells above ocean to 0
        hf = densefresh/densesalt*ocean_elev - (densesalt - densefresh)/densefresh*(henry_botm[lay] +.5*delv)
        for irow in range(nrow):
            ocean_hf.append((int(lay),int(irow),int(col),hf))
    ocean_hf = tuple(np.array(ocean_hf).T)
    ocean_hf = (ocean_hf[0].astype('int'),
                ocean_hf[1].astype('int'),
                ocean_hf[2].astype('int'),
                ocean_hf[3])
    return grd,ocean_hf

def get_ocean_right_edge(m,ocean_line_tuple,startlay=None,col=None):
    import numpy as np
    point_list = []
    if col is None:
        col = m.ncol-1
    #If there is no vertical side boundary, return bottom-right corner node
    if len(ocean_line_tuple)==0:
        if startlay is None:
            startlay = 0
    elif max(ocean_line_tuple[0])==m.nlay:
        startlay = m.nlay
    elif max(ocean_line_tuple[0])<m.nlay:
        startlay = max(ocean_line_tuple[0])
    for lay in range(startlay,m.nlay):
        for row in range(m.nrow):
            point_list.append((lay,row,col))
    point_list = tuple(np.array(point_list).T)
    return point_list

def add_pumping_wells(wel_data,ssm_data,n_wells,flx,rowcol,kper):
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    new_weldata = wel_data
    new_ssmdata = ssm_data
    wel_cells = []
    for k in range(n_wells):
        row,col = rowcol[k]
        for i in range(nper):
            if i in kper:
                for j in range(nlay):
                    #WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append([j,row,col,-flx[k]*delv_weight[j]])
                    wel_cells.append((j,row,col))
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    new_ssmdata[i].append([j,row,col,Cfresh,itype['WEL']]) #since it's a sink, conc. doesn't matter
            else:
                for j in range(nlay):
                    #WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append([j,row,col,0])
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    new_ssmdata[i].append([j,row,col,Cfresh,itype['WEL']]) #since it's a sink, conc. doesn't matter
                    wel_cells.append((j,row,col))
                continue
    wel_cells = tuple(np.array(list(set(wel_cells))).T)
    return new_weldata, new_ssmdata,wel_cells

#Add recharge if desired
def make_rech_array(low=1e-2,high=1e0):
    import scipy.stats as sts
    llow,lhigh = np.log10((low,high))
    rech = np.exp(sts.uniform.rvs(size=1,loc=llow,scale=lhigh-llow)[0])
    return rech/(nrow*ncol)

def add_recharge_cells(recharge_generator,const=1,*args):
    if const==1:
        rech_data = recharge_generator(*args)
    else:
        rech_data = {}
        for i in range(nper):
            rech_array = recharge_generator(*args)
        rech_data[i] = rech_array
    return rech_data

def sample_dist(distclass,size,*args):
    smp = distclass.rvs(*args,size=size)
    if size==1:
        smp=smp[-1]
    return smp

def write_sample(fname,varname,distclass,sample):
    fout= open(fname,"a")
    fout.write(varname + ',' + str(type(distclass)) + ',' + str(sample) + '\n')
    fout.close()
    return

def pec_num(delv,delc,delr,al):
    delL = (delv,delc,delr) #length in the lay,row,col directions
    pec_num = [round(d/al,2) for d in delL]
    for num,point  in zip(pec_num,('lay','row','col')):
        print('Pe = {} in the {} direction'.format(num,point))
    return pec_num

#%%
#Name model



sw_exe = config.swexe #set the exe path for seawat
print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))

#Model discretization
#Lx = 2000.
Lx = 3000.
Ly = 600.
Lz = 80.

nlay = int(Lz/3)
nrow = int(Ly/30)
ncol = int(Lx/30)

dim = tuple([int(x) for x in (nlay,nrow,ncol)])

henry_top = 3.3
ocean_elev = 0

#delv_first = Lz/nlay
delv_first = 4
botm_first = henry_top-delv_first

delv = (Lz-delv_first) / (nlay-1)
delr = Lx / ncol
delc = Ly / nrow

henry_botm = np.hstack(([botm_first],np.linspace(botm_first-delv,henry_top-Lz,nlay-1)))
delv_vec = np.hstack((delv_first,np.repeat(delv,nlay-1)))
delv_weight = [x/np.sum(delv_vec) for x in delv_vec]

beachslope = .05
ocean_col = [np.floor(ncol-1).astype('int'),ncol-1] #Manually done to make sure it's in the right place rn
inland_elev = beachslope*ocean_col[0]*delr
offshore_elev = -beachslope*(ocean_col[1]-ocean_col[0])*delr


#Period data
nyrs= 1
Lt = 360*nyrs #Length of time in days
perlen = list(np.repeat(180,int(Lt/180)))
nstp = list(np.ones(np.shape(perlen),dtype=int))

nper = len(perlen)
steady = [False for x in range(len(perlen))] #Never steady
itmuni = 4 #time unit 4= days
lenuni = 2 #length unit 2 = meter
tsmult = 1.8
ssm_data = None
verbose = True

print('Model setup: \n'
      'nlay: {}\n'
      'nrow: {}\n'
      'ncol: {}\n'
      'Total cells: {}\n'
      'Total time: {} days\n'
      'nper: {}\n'.format(nlay,nrow,ncol,nlay*nrow*ncol,Lt,nper))
# In[4]:

#Create basic model instance and dis pacakge
m = flopy.seawat.Seawat(modelname, exe_name=sw_exe, model_ws=model_ws,verbose=verbose)
SGD.ModelSGD.Seawat2SGD(m)  #convert to subclass ModelSGD
print(m.namefile)

# Add DIS package to the MODFLOW model
dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=nper, delr=delr,
                               delc=delc,
                               laycbd=0, top=henry_top,
                               botm=henry_botm, perlen=perlen, nstp=nstp,
                               steady=steady,itmuni=itmuni,lenuni=lenuni,
                               tsmult=tsmult)
create_MC_file()
# In[5]:

#Hydraulic conductivity field
hkSand = 1e2  #horizontal hydraulic conductivity m/day
hkClay = hkSand*.01
hk = hkSand*np.ones((nlay,nrow,ncol), dtype=np.int32)

#plt.figure(),plt.imshow((hk[:,0,:])),plt.colorbar(),plt.title('Sill:{}'.format(sill)),plt.show()

#Set Hydraulic properties
sy = 0.24
ss = 1e-5
por = 0.3
vka = 1 # = vk/hk
al = 1 #longitudinal dispersivity (m) from Walther et al. 2017
dmcoef = 2e-9 #m2/day

#Variable density parameters
Csalt = 35.0001
Cfresh = 0.
densesalt = 1025.
densefresh = 1000.
denseslp = (densesalt - densefresh) / (Csalt - Cfresh)
#denseslp = 0 #trick for testing constant density



# In[8]:
#Winter is even stress periods, summer is odd SP.
#Winter= wells OFF, natural precip (rech) ON, irrigation rech OFF,
#Summer = wells ON, irrigation rech (farm_rech) ON,  precip (rech) OFF
kper_odd = list(np.arange(1,nper,2))
kper_even = list(np.arange(0,nper,2))

#BCs
bc_ocean = 'GHB'
bc_right_edge = 'GHB'
bc_inland = 'GHB'
add_wells = 0
n_wells = 0
rech_on = 0

#Inland
calc_inland_head = 0 #calculate from hgrad
manual_inland_head = 0.3184
start_fresh_yn = 1
ocean_shead = [ocean_elev for x in range(len(perlen))]
ocean_ehead = ocean_shead

# save cell fluxes to unit 53
ipakcb = 53

#Create ocean boundary at top of model
ocean_col_vec = (0,0,np.arange(ocean_col[0],ocean_col[1]+1))
ocean_coords = (0,slice(0,nrow),slice(ocean_col[0],ocean_col[1]+1)) #top of the model
ocean_bool = np.zeros((nlay,nrow,ncol), dtype=bool)
ocean_bool[0,:,np.arange(ocean_col[0],ocean_col[1]+1)] = 1
m.ocean_arr = ocean_bool


if calc_inland_head == 1:
    head_inland = ocean_col[0]*delc*hgrad + ocean_elev
else:
    head_inland = manual_inland_head

####IN TESTING#####
#Create a line of where the ocean is, and any nodes on right edge below ocean
offshore_lay = (np.abs(henry_botm-offshore_elev)).argmin().astype('int')
if ocean_col[0] == ncol-1:
    ocean_line = []
    bc_ocean = 'XXX'
else:
    ocean_line = get_line((0,ocean_col[0]),(offshore_lay,ocean_col[1]),allrows=1,nrow=1)

ocean_line_tuple = tuple(np.array(ocean_line).T) #use this for indexing numpy arrays
right_edge = get_ocean_right_edge(m,ocean_line_tuple,
                                  int(np.where(henry_botm==find_nearest(henry_botm,ocean_elev))[0]))
left_edge = get_ocean_right_edge(m,ocean_line_tuple,
                                  int(np.where(henry_botm==find_nearest(henry_botm,head_inland))[0]),
                                col=0)

#Create ibound
ibound,ocean_hf = shade_above(nlay,nrow,ncol,ocean_line) #don't set ibound of ocean
#ibound[:right_edge[0][0],right_edge[1][0],right_edge[2][0]] = 0
#ibound[:right_edge[0][0],right_edge[1][0],0] = 0

if bc_ocean == 'GHB':
    ibound[ocean_line_tuple]=1


#Set starting heads
strt = np.ones((nlay,nrow,ncol),dtype=np.int)+3


#Transport BCs
if start_fresh_yn == 1:
    sconc = Cfresh*np.ones((nlay, nrow, ncol), dtype=np.float32) #Begin fresh
elif start_fresh_yn == 0:
    sconc = Csalt*np.ones((nlay, nrow, ncol), dtype=np.float32) #Begin SW-saturated
else:
    sconc = Cfresh*np.ones((nlay, nrow, ncol), dtype=np.float32) #Begin SW-saturated
    sconc[:,:,int(np.floor(ncol/2)):-1] = Csalt

if ocean_hf:
    sconc[ocean_hf[0:3]] = Csalt
sconc[right_edge] = Csalt
sconc[:,:,0] = Cfresh

icbund = np.ones((nlay, nrow, ncol), dtype=np.int)
icbund[np.where(ibound==-1)] = -1

head_inland_sum_wint = (3.0,3.0) #m in summer, m in winter

def make_bc_dicts(head_inland_sum_wint=head_inland_sum_wint):
    #Ocean and inland boundary types
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    chd_data = {}
    ssm_data = {}
    ghb_data = {}
    wel_data = {}
    for i in range(nper):
        dat_chd = []
        dat_ssm = []
        dat_ghb = []
        dat_wel = []
        #Ocean boundary
        if ocean_hf:
            for j in range(np.size(ocean_hf[0])):
                if bc_ocean=='CHD':
                    #CHD: {stress_period: [lay,row,col,starthead,endhead]}
                    dat_chd.append([ocean_line_tuple[0][j],
                                ocean_line_tuple[1][j],
                                ocean_line_tuple[2][j],
                                ocean_shead[i],
                                ocean_ehead[i]])
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([ocean_line_tuple[0][j],
                                ocean_line_tuple[1][j],
                                ocean_line_tuple[2][j],
                                Csalt,
                                itype['CHD']])
                elif bc_ocean=='GHB':
                    #GHB: {stress period: [lay,row,col,head level,conductance]}
                    #conductance c = K*A/dL; assume horizontal flow at outlet,
                    #and calculate length to be at edge of ocean cell, as opposed to mipoint
                    # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
                    dat_ghb.append([ocean_hf[0][j],
                                   ocean_hf[1][j],
                                   ocean_hf[2][j],
                                   #ocean_hf[3][j],
                                    ocean_elev,
                                   2*hkSand*delc*delv_vec[ocean_hf[0][j]]/delr])
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([ocean_hf[0][j],
                                   ocean_hf[1][j],
                                   ocean_hf[2][j],
                                   Csalt,
                                   itype['GHB']])
        else:
            pass
        #Right edge boundary
        if bc_right_edge=='GHB':
            for j in range(np.size(right_edge[0])):
                #GHB: {stress period: [lay,row,col,head level,conductance]}
                #conductance c = K*A/dL; assume horizontal flow at outlet,
                #and calculate length to be at edge of ocean cell, as opposed to mipoint
                # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
                dat_ghb.append([right_edge[0][j],
                               right_edge[1][j],
                               right_edge[2][j],
                               #ocean_hf[3][j],
                                ocean_elev,
                               2*hkSand*delc*delv_vec[right_edge[0][j]]/delr])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                dat_ssm.append([right_edge[0][j],
                               right_edge[1][j],
                               right_edge[2][j],
                               Csalt,
                               itype['GHB']])
        else:
            pass
        #Inland boundary
        if bc_inland=='GHB':
            if i in kper_odd:
                head_inland = head_inland_sum_wint[0]
            elif i in kper_even:
                head_inland = head_inland_sum_wint[1]
            left_edge = get_ocean_right_edge(m,ocean_line_tuple,
                  int(np.where(henry_botm==find_nearest(henry_botm,head_inland))[0]),
                col=0)
            for j in range(np.size(left_edge[0])):
                dat_ghb.append([left_edge[0][j],
                               left_edge[1][j],
                               left_edge[2][j],
                                head_inland,
                               2*hkSand*delc*delv_vec[left_edge[0][j]]/delr])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                dat_ssm.append([left_edge[0][j],
                               left_edge[1][j],
                               left_edge[2][j],
                               Cfresh,
                               itype['GHB']])
        elif bc_inland=='WEL':
            for j in range(nlay):
                for k in range(nrow):
                    #WEL: {stress_period: [lay,row,col,flux]}
                    dat_wel.append([j,k,0,influx*delv_weight[j]/nrow])
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([j,k,0,Cfresh,itype['WEL']])
        chd_data[i] = dat_chd
        ssm_data[i] = dat_ssm
        ghb_data[i] = dat_ghb
        wel_data[i] = dat_wel

    #saving concentrations at specified times
    #timprs = [k for k in range(1,np.sum(perlen),50)]
    return chd_data, ssm_data, ghb_data, wel_data

chd_data, ssm_data, ghb_data, wel_data = make_bc_dicts(head_inland_sum_wint)
wel_data_base,ssm_data_base = wel_data,ssm_data
timprs = np.round(np.linspace(1,np.sum(perlen),20),decimals=0)

save_obj(m.MC_file.parent,wel_data_base,'wel_data_base')
save_obj(m.MC_file.parent,ssm_data_base,'ssm_data_base')

# In[9]:

#### ADD WELL AND RECHARRGE DATA####
#Winter is even stress periods, summer is odd SP.
#Winter= wells OFF, natural precip (rech) ON, irrigation rech OFF,
#Summer = wells ON, irrigation rech (farm_rech) ON,  precip (rech) OFF

##Add recharge data
rech = 1e-6

#Assign the location of the farms
farm_leftmargin = 10
farm_leftmargin = int(ncol/2)
farm_uppermargin = 1
nfarms = 4
farm_size = (200,200) #size in meters (row,col)
farm_size_rowcol = (int(farm_size[0]/delc),int(farm_size[1]/delr)) #size of farm in number of row,col
farm_loc_list = []
farm_orig = []
for x in range(int(nfarms/2)):
    for y in range(2):
        farm_loc_r = []
        farm_loc_c = []
        for z1 in range(farm_size_rowcol[0]):
            for z2 in range(farm_size_rowcol[1]):
                farm_loc_r.append(farm_uppermargin + y*(farm_size_rowcol[0]+2) + z1)
                farm_loc_c.append(farm_leftmargin + x*(farm_size_rowcol[1]+2) + z2)
                if (z1==0) and (z2==0):
                    farm_orig.append((farm_loc_r[-1],farm_loc_c[-1])) #upper left of ea. farm=loc of well
        farm_loc_list.append((np.array(farm_loc_r),np.array(farm_loc_c)))
farm_loc = (np.array(farm_loc_r),np.array(farm_loc_c))

## Add well data
n_wells = nfarms
wel_flux = list(np.ones(n_wells)*0)
top_lay = int(np.where(henry_botm==find_nearest(henry_botm,ocean_elev))[0])+1
wel_data,ssm_data,wel_cells = add_pumping_wells(wel_data_base.copy(),ssm_data_base.copy(),n_wells,wel_flux,farm_orig,kper_odd)


hk[wel_cells] = hkSand


## Add farm recharge data
farm_rech_flux = [flx*0.2 for flx in wel_flux]
farm_rech = np.zeros((nrow,ncol),dtype=np.float)

for i in range(nfarms):
    farm_rech[farm_loc_list[i]] = farm_rech_flux[i]/np.prod(farm_size)
#Set rech_data for winter and summer
rech_data = {}
for i in range(len(perlen)):
    if i in kper_even:
        rech_data[i] = rech
    elif i in kper_odd:
        rech_data[i] = farm_rech

# In[10]:

riv_loc = get_line((0,0),(0,ncol-1),allrows=1,nrow=nrow)
riv_loc = [x for x in riv_loc if x[1]==int(nrow/2)]
riv_loc = tuple(np.array(riv_loc).T)

riv_grad = .001
rbot_vec = np.linspace(riv_grad*Lx,ocean_elev,ncol)

#Stage and conductance:
stage = 1
cond = 10
riv_grad = .001

def write_river_data(riv_loc,stage,cond,riv_grad,kper,ssm_data):

    ####ADD RIVER DATA####
    rbot_vec = np.linspace(riv_grad*Lx,ocean_elev,ncol)

    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    riv_data = {}
    new_ssm_data = ssm_data
    for i in range(nper):
        dat_riv = []
        if i in kper:
            for j in range(np.size(riv_loc[0])):
                #RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                    riv_loc[1][j],
                                    riv_loc[2][j],
                                    stage+rbot_vec[riv_loc[2][j]],
                                    cond,
                                    rbot_vec[riv_loc[2][j]]])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        else:
            for j in range(np.size(riv_loc[0])):
                #RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                    riv_loc[1][j],
                                    riv_loc[2][j],
                                    rbot_vec[riv_loc[2][j]], #set stage as bottom of river
                                    cond,
                                    rbot_vec[riv_loc[2][j]]])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        riv_data[i] = dat_riv
    return riv_data,new_ssm_data

riv_data,ssm_data = write_river_data(riv_loc,stage,cond,riv_grad,kper_even,ssm_data)

# In[9]:

#Output control
oc_data = {}
for kper in range(nper):
    oc_data[(kper,0)] = ['save head','save budget']


# In[10]:

#Create instances in flopy
bas = flopy.modflow.ModflowBas(m, ibound, strt=strt)
if bc_ocean=='CHD' or bc_inland=='CHD' :
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chd_data)
if bc_ocean=='GHB' or bc_inland=='GHB'or bc_right_edge=='GHB':
    ghb = flopy.modflow.ModflowGhb(m, stress_period_data=ghb_data)

rch = flopy.modflow.ModflowRch(m, rech=rech_data)
wel = flopy.modflow.ModflowWel(m, stress_period_data=wel_data, ipakcb=ipakcb)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=riv_data)
# Add LPF package to the MODFLOW model
lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=vka, ipakcb=ipakcb,laytyp=1,laywet=1,
                              ss=ss,sy=sy)

# Add PCG Package to the MODFLOW model
pcg = flopy.modflow.ModflowPcg(m, hclose=1.e-8)

# Add OC package to the MODFLOW model
oc = flopy.modflow.ModflowOc(m,
                             stress_period_data=oc_data,
                             compact=True)

#Create the basic MT3DMS model structure
btn = flopy.mt3d.Mt3dBtn(m,
                         laycon=lpf.laytyp, htop=henry_top,
                         dz=dis.thickness.get_value(), prsity=por, icbund=icbund,
                         sconc=sconc, nprs=1,timprs=timprs)
adv = flopy.mt3d.Mt3dAdv(m, mixelm=-1)
dsp = flopy.mt3d.Mt3dDsp(m, al=al, dmcoef=dmcoef)
gcg = flopy.mt3d.Mt3dGcg(m, iter1=50, mxiter=1, isolve=1, cclose=1e-5)
ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

#vdf = flopy.seawat.SeawatVdf(m, iwtable=0, densemin=0, densemax=0,denseref=1000., denseslp=0.7143, firstdt=1e-3)
vdf = flopy.seawat.SeawatVdf(m, mtdnconc=1, mfnadvfd=1, nswtcpl=0, iwtable=1,
                             densemin=0., densemax=0., denseslp=denseslp, denseref=densefresh)

# In[11]:

#printyn = 0
#gridon=0
#rowslice=0
#rowslice=farm_orig[0][0]
#m.plot_hk_ibound(rowslice=rowslice,printyn=printyn,gridon=gridon);


# In[12]:

#Write input
m.write_input()

# Try to delete the output files, to prevent accidental use of older files
try:
    os.remove(os.path.join(model_ws,'MT3D.CNF'))
    os.remove(os.path.join(model_ws,'MT3D001.MAS'))
    os.remove(os.path.join(model_ws, 'MT3D001.UCN'))
    os.remove(os.path.join(model_ws, modelname + '.hds'))
    os.remove(os.path.join(model_ws, modelname + '.cbc'))
except:
    pass

#%%
runyn = False

if runyn:
    #Run model
    ts = make_timestamp()
    v = m.run_model(silent=False, report=True)
    for idx in range(-3, 0):
        print(v[1][idx])
else:
    print('Not running model!')
# In[14]:

#Post-processing functions
def extract_hds_conc(m,per):
    fname = Path(m.model_ws).joinpath(Path(m.name).parts[-1] + '.hds').as_posix()

    hdobj = flopy.utils.binaryfile.HeadFile(fname)
    times = hdobj.get_times()
    hds = hdobj.get_data(totim=times[per])
    hds[np.where(ibound != 1)] = np.nan
    hds[np.where((hds>1e10) | (hds<-1e10))] = np.nan

    # Extract final timestep salinity
    fname = Path(m.model_ws).joinpath('MT3D001.UCN').as_posix()
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    times = ucnobj.get_times()
    kstpkper = ucnobj.get_kstpkper()
    conc = ucnobj.get_data(totim=times[per])
    conc[np.where(ibound != 1)] = np.nan
    conc[np.where((conc>1e10) | (conc<-10))] = np.nan
    return conc,hds

# Make head and quiver plot
import utils
def basic_plot(m,per,backgroundplot,rowslice=0,printyn=0,contoursyn=1,**kwargs):
    printyn = 1

    f, axs = plt.subplots(1, figsize=(6, 2))

    plt.tight_layout()

    #Plot discharge and ibound
    mm = flopy.plot.ModelCrossSection(ax=axs, model=m, line={'row':rowslice})

    #Plot background
    backgroundpatch,lbl = cpatchcollection,label = plot_background(mm,backgroundplot,'conc(g/L)')
    lvls = Cfresh + (Csalt-Cfresh)*np.array([.05,.5,.95])
    if contoursyn==1:
        CS = mm.contour_array(backgroundplot,levels=lvls,colors='white')
        plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    #mm.contour_array(hds,head=hds)
    mm.plot_ibound()
    mm.plot_bc(ftype='GHB',color='blue')
    if m.Wel:
        mm.plot_bc(ftype='WEL',color='black')
    #Plot discharge
    utils.plotdischarge(m,color='white',per=per,rowslice=rowslice,**kwargs);
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Flow during period {} of {}'.format(np.arange(nper)[per],nper-1))
    plt.subplots_adjust(bottom=.1)

    #align plots and set colorbar
    f.subplots_adjust(left=.1,right=0.88)
    cbar_ax = f.add_axes([0.90, 0.1, 0.02, 0.7])
    cb = f.colorbar(cpatchcollection,cax=cbar_ax)
    cb.set_label(label)
    if printyn == 1:
        plt.savefig(m.MC_file.parent.joinpath(ts + 'flowvec_row' + str(rowslice) +
                                 '_per' + str(per) + '_' + lbl[:3] + '.png').as_posix(),dpi=150)
    plt.show()
    return CS


def plotdischarge(modelname,model_ws,color='w',per=-1,scale=50,rowslice=0):
    fname = os.path.join(model_ws, '' + modelname + '.cbc')
    budobj = flopy.utils.CellBudgetFile(fname)
    qx = budobj.get_data(text='FLOW RIGHT FACE')[per]
    qz = budobj.get_data(text='FLOW LOWER FACE')[per]

    # Average flows to cell centers
    qx_avg = np.empty(qx.shape, dtype=qx.dtype)
    qx_avg[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol-1] + qx[:, :, 1:ncol])
    qx_avg[:, :, 0] = 0.5 * qx[:, :, 0]
    qz_avg = np.empty(qz.shape, dtype=qz.dtype)
    qz_avg[1:, :, :] = 0.5 * (qz[0:nlay-1, :, :] + qz[1:nlay, :, :])
    qz_avg[0, :, :] = 0.5 * qz[0, :, :]

    y, x, z = dis.get_node_coordinates()
    X, Z = np.meshgrid(x, z[:, 0, 0])
    iskip = 1 #how many cells to skip, 1 means plot every cell

    ax = plt.gca()
    cpatchcollection = ax.quiver(X[::iskip, ::iskip], Z[::iskip, ::iskip],
              qx_avg[::iskip, rowslice, ::iskip], -qz_avg[::iskip, rowslice, ::iskip],
              color=color, scale=scale, headwidth=4, headlength=2,
              headaxislength=1, width=0.0025)
    return cpatchcollection

def permute_kstpkper(ucnobj):
    kstpkper = ucnobj.get_kstpkper()
    kstpkper_unique = []
    index_unique = []
    niter = 0
    for entry in kstpkper:
        if not entry in kstpkper_unique:
            kstpkper_unique.append(entry)
            index_unique.append(niter)
        niter += 1
    return kstpkper_unique, index_unique

def kstpkper_from_time(ucnobj,tottim):
    kstpkpers = ucnobj.get_kstpkper()
    times = ucnobj.get_times()
    timeind = times.index(tottim)
    kstpkper = kstpkpers[timeind]
    return kstpkper

def kstpkper_ind_from_kstpkper(ucnobj,kstpkper=(0,0)):
    kstpkpers = ucnobj.get_kstpkper()
    kstpkper_unique = permute_kstpkper(ucnobj)[0]
    kstpkper_ind = kstpkper_unique.index(kstpkper)
    return kstpkper_ind

def get_salt_outflow(m,kstpkper=None,totim=None):
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    totim = ucnobj.get_times()[-1]
    if kstpkper==None:
        kstpkper = ucnobj.get_kstpkper()[-1]
    ocean_conc = ucnobj.get_data(kstpkper=kstpkper)
    return ocean_conc

def plot_background(mm,array,label=None):
    if label==None:
        label = [k for k,v in globals().items() if v is array][-1]
    if label=='hk':
        norm=matplotlib.colors.LogNorm()
        vmin=hkClay
        vmax=hkSand
        cmap='jet'
    else:
        norm = None
        vmin=None
        vmax=None
        cmap='jet'
    cpatchcollection = mm.plot_array(array,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    cpatchcollection.set_label(label)
    return cpatchcollection,label

def plot_mas(m):
    # Load the mas file and make a plot of total mass in aquifer versus time
    fname = os.path.join(m.model_ws, 'MT3D001.MAS')
    mas = flopy.mt3d.Mt3dms.load_mas(fname)
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    plt.xlabel('Time (d)')
    plt.ylabel('Mass (kg)')
    plt.title('Mass of salt within model through time')
    lines = ax.plot(mas.time, mas.total_mass)
    plt.show()
    return mas

# Make head and quiver plot
def basic_plot(m,per,backgroundplot,rowslice=0,printyn=0,contoursyn=1,**kwargs):
    printyn = 1

    f, axs = plt.subplots(1, figsize=(6, 2))

    plt.tight_layout()

    #Plot discharge and ibound
    mm = flopy.plot.ModelCrossSection(ax=axs, model=m, line={'row':rowslice})

    #Plot background
    backgroundpatch,lbl = cpatchcollection,label = plot_background(mm,backgroundplot,'conc(g/L)')
    lvls = Cfresh + (Csalt-Cfresh)*np.array([.05,.5,.95])
    if contoursyn==1:
        CS = mm.contour_array(backgroundplot,levels=lvls,colors='white')
        plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    #mm.contour_array(hds,head=hds)
    mm.plot_ibound()
    mm.plot_bc(ftype='GHB',color='blue')
    if m.Wel:
        mm.plot_bc(ftype='WEL',color='black')
    #Plot discharge
    utils.plotdischarge(m,color='white',per=per,rowslice=rowslice,**kwargs);
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Flow during period {} of {}'.format(np.arange(nper)[per],nper-1))
    plt.subplots_adjust(bottom=.1)

    #align plots and set colorbar
    f.subplots_adjust(left=.1,right=0.88)
    cbar_ax = f.add_axes([0.90, 0.1, 0.02, 0.7])
    cb = f.colorbar(cpatchcollection,cax=cbar_ax)
    cb.set_label(label)
    if printyn == 1:
        plt.savefig(m.MC_file.parent.joinpath(ts + 'flowvec_row' + str(rowslice) +
                                 '_per' + str(per) + '_' + lbl[:3] + '.png').as_posix(),dpi=150)
    plt.show()
    return CS


# In[19]:
per = [-1,-2]
rowslice = 1

try:
    mas = plot_mas(m)
    ts=''
    for p in per:
        conc,hds = extract_hds_conc(m,p)
        basic_plot(m,p,conc,rowslice=rowslice,scale=70,iskip=3,printyn=1,contoursyn=1)
    m.plot_hk_ibound(rowslice=rowslice,gridon=0)
except:
    print('Cant find concentration file, not making plot...')
    pass

#%%

def add_to_paramdict(paramdict,paramname,val):
    if paramdict is None:
        paramdict = {}
    if  paramname in list(paramdict.keys()):
        paramdict[paramname].append(val)
    else:
        #paramdict.update(paramname=[val])
        paramdict[paramname] = [val]
    return


def record_salinity(m,totim=None,writeyn=True,fname_write=None,ts_hms=None):
    from pathlib import Path
    import datetime
    if ts_hms is None:
        ts_hms = datetime.datetime.now().strftime('%H-%M-%S')
    # Extract final timestep salinity
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    if totim is None:
        totim = ucnobj.get_times()[-1]
    conc = ucnobj.get_data(totim=totim)
    if writeyn:
        if fname_write is None:
            fname_write = m.MC_file.parent.joinpath('conc_' + str(int(totim)) + '_' + ts_hms + '.npy')
        print(fname_write)
        np.save(fname_write,conc)
    return conc

def copy_rename(src_file, dst_file):
    import shutil
    from pathlib import Path
    shutil.copy(str(Path(src_file)),str(Path(dst_file)))
    return

def get_hds(m,kstpkper=None):
    f_hds = Path(m.name + '.hds')
    hdsobj = flopy.utils.binaryfile.HeadFile(f_hds.as_posix())
    if kstpkper is None:
        kstpkper = hdsobj.get_kstpkper()[-1]
    return hdsobj.get_data(kstpkper=kstpkper)

def get_base_hds_conc(f_basecase=None):
    if f_basecase is None:
        f_basecase = Path(model_ws).parent.joinpath('base_case_3m','base_case_3m.nam')
    m_temp =flopy.seawat.Seawat.load(f_basecase.as_posix(), exe_name = config.swexe,model_ws = f_basecase.parent.as_posix())
    conc_final = record_salinity(m_temp,writeyn=False)
    hds_final = get_hds(m_temp)
    return hds_final,conc_final

# In[22]:

def get_yn_response(prompt):
    while True:
        try:
            resp = str(input(prompt))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        if resp[0] is 'y':
            value = True
            break
        elif resp[0] is 'n':
            value = False
            break
        else:
            print('This didnt work right. Try again')
            continue
    return value

def get_value(prompt):
    while True:
        try:
            resp = str(input(prompt))
            break
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
    return resp

def check_MC_inputParams():
    if m.MC_file is not None:
        use_existing_MCfile = get_yn_response("m.MC_file already exists, continue using this experiment?")
    else:
        use_existing_MCfile = False
    if use_existing_MCfile:
        if m.inputParams is not None:
            if len(m.inputParams)>0:
                add_to_inputParams = get_yn_response("m.inputParams already has entries, do you want to add to it?")
            else:
                add_to_inputParams =False
            if add_to_inputParams:
                pass
            else:
                m.inputParams = {}
        else:
            m.inputParams = {}
    else:
        load_existing_MCfile = get_yn_response("load MC file?")
        if load_existing_MCfile:
            f = get_value("path to MC_file (path/to/test.expt): ")
            m.inputParams = load_obj(Path(f),'inputParams')
            print('loaded .pkl file!')
        else:
            create_MC_file()
            m.inputParams = {}
    return

def idx2centroid(node_coord_tuple,idx_tuple):
    z_pt = node_coord_tuple[2][idx_tuple]
    x_pt = node_coord_tuple[1][idx_tuple[2]]
    y_pt = node_coord_tuple[0][idx_tuple[1]]
    return (z_pt,y_pt,x_pt)

def rem_last_ind_from_dict(dict):
    dict_filt = {}
    for k,v in dict.items():
        vnew = v[:-1]
        dict_filt[k] = vnew
    return dict_filt


def filt_inds_from_dict(dict,inds):
    i=0
    dict_false = {}
    dict_true = {}
    for k,v in dict.items():
        vtrue = [x for i, x in enumerate(v) if i in inds]
        vfalse = [x for i, x in enumerate(v) if i not in inds]
        dict_true[k] = vtrue
        dict_false[k] = vfalse
        i+=1
    return dict_true,dict_false

def sample_uniform(low, high, shape, logyn):
    '''
    #Samples a uniform distribution the nummber of times shown in 
    low: low value in dist
    high: high value in dist
    shape: shape of samples
    logyn: if True, samples as a log-normal distribution. 
        If False, samples as a uniform distribution. Returned values are *not* in logspace  
    '''

    if logyn:
        log_param_list = np.random.uniform(np.log(low), np.log(high), shape)
        param_list = np.exp(log_param_list)
    else:
        param_list = np.random.uniform(low, high, shape)
    return param_list

def normal_transform(data1,mu1,mu2,sig1,sig2):
    a = sig2/sig1
    b = mu2 - mu1 * a
    return a*data1 + b




######################################################################
########################### NEW FUNCTION #############################
######################################################################
def create_varlist(tot_it,heterogenous=1,saveyn=True):
    varlist = {}
    #head_inland_sum
    logyn = False
    low = -1.1
    high = 0
    varlist['head_inland_sum'] = sample_uniform(low, high, tot_it, logyn)
    
    #head_inland_wint
    logyn = False
    low = 0
    high = 3
    varlist['head_inland_wint'] = sample_uniform(low, high, tot_it, logyn)
    
    if heterogenous==0:
    #HOMOGENOUS ONLY
        # log_hk
        logyn = True
        low = 80
        high = 80
        varlist['hk'] = sample_uniform(low, high, tot_it, logyn)
    
        ##por: porosity
        logyn = False
        low = .2
        high = .5
        varlist['por'] = sample_uniform(low, high, tot_it, logyn)
    
    elif heterogenous in [1,2]:
        
        #########HETEROGENOUS ONLY ##############
        
        #CF_glob: global clay-fraction (mu in the random gaussian simulation)
        logyn = False
        low = .1
        high = .9
        varlist['CF_glob'] = sample_uniform(low, high, tot_it, logyn)
        
        #CF_var: variance in clay-fraction (sill in the random gaussian simulation)
        logyn = False
        low = .001
        high = .05
        varlist['CF_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #hk_var: variance in hk (sill in the random gaussisan simulation)
        logyn = True
        low = .005
        high = .175
        varlist['hk_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #seed for random gaussian simulation
        varlist['seed'] = np.arange(1,tot_it+1)
    
        #hk_mean: mean in hk (set constant)
        varlist['hk_mean'] = np.ones(tot_it)*np.log10(50)
        
        #por_mean: global porosity (mu in the random gaussian simulation)
        logyn = False
        low = 0.3
        high = 0.4
        varlist['por_mean'] = sample_uniform(low, high, tot_it, logyn)
        
        #por_var: variance in porosity (sill in the random gaussian simulation)
        logyn = True
        low = .00001
        high = .005
        varlist['por_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #vario_type: model for random gaussian simulation
        varlist['vario_type'] = ['Gaussian' if v==1 else 'Exponential' for v in np.random.randint(0,2,tot_it)]
    
        #corr_len
        logyn = False
        low = 250
        high = 1000
        varlist['corr_len'] = sample_uniform(low, high, tot_it, logyn)
    
        #corr_len_zx
        # equal to lz/lx
        low= .01
        high = .1
        logyn = False
        varlist['corr_len_zx'] = sample_uniform(low, high, tot_it, logyn)
    
    
        #corr_len_yx
        # equal to ly/lx
        low= 0.1
        high = 1
        varlist['corr_len_yx'] = sample_uniform(low, high, tot_it, logyn)
    
        #clay_lyr_yn
        varlist['clay_lyr_yn'] = np.random.randint(0,2,tot_it,dtype=bool)
               
    
    #### END INSERTED BLOCK ########
    
    
    # vka: ratio of vk/hk
    logyn = False
    low = 1 / 20
    high = 1
    varlist['vka'] = sample_uniform(low, high, tot_it, logyn)
    
    # al: #longitudinal dispersivity (m)
    logyn = False
    low = 0.1
    high = 20
    varlist['al'] = sample_uniform(low, high, tot_it, logyn)
    
    # dmcoef: #dispersion coefficient (m2/day)
    #      log-uniform [1e-10,1e-5] #2e-9 from Walther et al
    logyn = True
    low = 1e-10
    high = 1e-5
    varlist['dmcoef'] = sample_uniform(low, high, tot_it, logyn)
    
    # sy: specific yield
    logyn = False
    low = 0.1
    high = 0.4
    varlist['sy'] = sample_uniform(low, high, tot_it, logyn)

    # ss: specific storage
    logyn = False
    low = 5.0e-5
    high = 5.0e-3
    varlist['ss'] = sample_uniform(low, high, tot_it, logyn)
    
    # Wel
    logyn = True
    # low = 1e2
    # high = 1e3
    low = 10
    high = 5e2
    varlist['wel'] = sample_uniform(low, high, (4, tot_it), logyn)
    
    # rech
    logyn = True
    low = 3.5e-4 
    high = 1.5e-3
    varlist['rech'] = sample_uniform(low, high, tot_it, logyn)
    
    # farm rech as a fraction of well extraction
    logyn = False
    low = .05
    high = .20
    varlist['rech_farm'] = sample_uniform(low, high, tot_it, logyn)
    
    # riv_stg
    logyn = False
    low = 0.5
    high = 1.5
    varlist['riv_stg'] = sample_uniform(low, high, tot_it, logyn)
    
    # riv_cond
    logyn = True
    low = 0.1
    high = 100
    varlist['riv_cond'] = sample_uniform(low, high, tot_it, logyn)
    
    #Success log
    varlist['success'] = np.ones(tot_it,dtype=np.int)*-1
    
    varlist['it'] = np.arange(tot_it)
    
    # Save
    save_obj(m.MC_file.parent, varlist, 'varlist')
    print('Saved file', m.MC_file.parent.joinpath('varlist.pkl'))
    return varlist
    

#%%
def update_run_model(varlist, it, m=m, homogenous=2, runyn=True,
                     plotyn=False,silent=True,start_basecase=True,
                     f_basecase=None,pooling=True,output=None,results=[]):
    import simulationFFT
    # Make timestamp
    ts = make_timestamp()
    print('Running it {} at time {}'.format(it,ts))
    
    if start_basecase:
        strt,sconc = get_base_hds_conc(f_basecase)
    
    if pooling:
        model_ws_orig = Path(m.model_ws).as_posix() + ''
        tmp = Path(model_ws).joinpath('tmp{}'.format(it))
        if not tmp.exists():
            tmp.mkdir()
        m.model_ws = tmp.as_posix()
        print('temp ws', m.model_ws)

    # unpack values from varlist
    vka = varlist['vka'][it]
    al = varlist['al'][it]
    dmcoef = varlist['dmcoef'][it]
    riv_stg = varlist['riv_stg'][it]
    riv_cond = varlist['riv_cond'][it]
    head_inland_sum = varlist['head_inland_sum'][it]
    head_inland_wint = varlist['head_inland_wint'][it]
    wel = varlist['wel'][:,it]
    rech_farm_pct = varlist['rech_farm'][0]
    rech_farm = [rech_farm_pct*flx/np.prod(farm_size) for flx in wel]
    rech_precip = varlist['rech'][it]
    
    CF_glob = varlist['CF_glob'][it]
    CF_var = varlist['CF_var'][it]
    seed = varlist['seed'][it]
    hk_mean = varlist['hk_mean'][it]
    hk_var = varlist['hk_var'][it]
    por_mean = varlist['por_mean'][it]
    por_var = varlist['por_var'][it]
    corr_len = varlist['corr_len'][it]
    corr_len_yx = varlist['corr_len_yx'][it]
    corr_len_zx = varlist['corr_len_zx'][it]
    clay_lyr_yn = varlist['clay_lyr_yn'][it]
    vario_type = varlist['vario_type'][it]
    
    #set ghb data and create dicts
    chd_data, ssm_data_base, ghb_data, wel_data_base = make_bc_dicts((head_inland_sum,head_inland_wint))
    save_obj(m.MC_file.parent,wel_data_base,'wel_data_base')
    save_obj(m.MC_file.parent,ssm_data_base,'ssm_data_base')

    ssm_data = {}
    # write recharge data
    
    rech_farm_mat = np.zeros((nrow,ncol),dtype=np.float32)
    for i in range(len(rech_farm)):
        rech_farm_mat[farm_loc_list[i]] = rech_farm[i]
    
    rech_data = {}
    for i in range(len(perlen)):
        if i in kper_even:
            rech_data[i] = rech_precip
        elif i in kper_odd:
            rech_data[i] = rech_farm_mat


    # write wel data
    # ssm_data_base = load_obj(m.MC_file.parent, 'ssm_data_base')
    # wel_data_base = load_obj(m.MC_file.parent, 'wel_data_base')
    wel_data, ssm_data, wel_cells = add_pumping_wells(wel_data_base,
                                                      ssm_data_base,
                                                      n_wells,flx=wel,
                                                      rowcol=farm_orig,
                                                      kper=kper_odd)

    # Write river data--take SSM data from WEL!!
    riv_grad = .0005
    riv_data, ssm_data = write_river_data(
        riv_loc, riv_stg, riv_cond, riv_grad, kper_even, ssm_data)

    
    if homogenous==1:
        CF_grid = 1
        hk_grid = 10**hk_mean
        por_grid = .4
    elif homogenous==2:
        #Create Gaussian Simulation
        lcol = int(corr_len/delr)
        llay = int(corr_len*corr_len_zx/np.mean(delv))
        lrow = int(corr_len*corr_len_yx/delc)
    #     fft_grid = np.exp(simulationFFT.simulFFT(nrow, nlay, ncol, mu, sill, vario_type, lrow , llay, lcol))   
        CF_grid = simulationFFT.simulFFT(nrow,nlay, ncol,CF_glob,CF_var,vario_type, lrow , llay, lcol,seed=seed)
        hk_grid = 10**normal_transform(CF_grid,CF_glob,hk_mean,np.sqrt(CF_var),np.sqrt(hk_var))
        por_grid = normal_transform(CF_grid,CF_glob,por_mean,np.sqrt(CF_var),np.sqrt(por_var))
        CF_grid[CF_grid > 1.] = 1.
        CF_grid[CF_grid < 0.] = 0.
        por_grid[por_grid > 1.] = .99
        por_grid[por_grid < 0.] = 0.01    

        hk_grid[wel_cells] = np.max((hk_grid.max(),200))
        np.save(m.MC_file.parent.joinpath('{}_hk.npy'.format(ts)),hk_grid)
        
    
    
    ###### Reassign, run record ######
    # Reassign to model object
    # assign_m()
    bas = flopy.modflow.ModflowBas(m, ibound, strt=strt)
    if bc_ocean == 'CHD' or bc_inland == 'CHD':
        chd = flopy.modflow.ModflowChd(m, stress_period_data=chd_data)
    if bc_ocean == 'GHB' or bc_inland == 'GHB' or bc_right_edge == 'GHB':
        ghb = flopy.modflow.ModflowGhb(m, stress_period_data=ghb_data)
    wel = flopy.modflow.ModflowWel(
        m, stress_period_data=wel_data, ipakcb=ipakcb)
    rch = flopy.modflow.ModflowRch(m, rech=rech_data)
    riv = flopy.modflow.ModflowRiv(m, stress_period_data=riv_data)

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(m, hk=hk_grid, vka=vka, ipakcb=ipakcb,
                                   laytyp=1,laywet=1,ss=ss,sy=sy)
    # Add PCG Package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(m, hclose=1.e-8)

    # Add OC package to the MODFLOW model
    oc = flopy.modflow.ModflowOc(m,
                                 stress_period_data=oc_data,
                                 compact=True)

    # Create the basic MT3DMS model structure
    btn = flopy.mt3d.Mt3dBtn(m,
                             laycon=lpf.laytyp, htop=henry_top,
                             dz=dis.thickness.get_value(), prsity=por_grid, icbund=icbund,
                             sconc=sconc, nprs=1, timprs=timprs)
    adv = flopy.mt3d.Mt3dAdv(m, mixelm=-1)
    dsp = flopy.mt3d.Mt3dDsp(m, al=al, dmcoef=dmcoef)
    gcg = flopy.mt3d.Mt3dGcg(m, iter1=50, mxiter=1, isolve=1, cclose=1e-5)
    ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

    #vdf = flopy.seawat.SeawatVdf(m, iwtable=0, densemin=0, densemax=0,denseref=1000., denseslp=0.7143, firstdt=1e-3)
    vdf = flopy.seawat.SeawatVdf(m, mtdnconc=1, mfnadvfd=1, nswtcpl=0, iwtable=1,
                                 densemin=0., densemax=0., denseslp=denseslp, denseref=densefresh)

    # Write input
    m.write_input()

    # Try to delete the output files, to prevent accidental use of older files
    flist = [os.path.join(model_ws, 'MT3D.CNF'),
             os.path.join(model_ws, 'MT3D001.MAS'),
             os.path.join(model_ws, modelname + '.hds'),
             os.path.join(model_ws, 'MT3D001.UCN'),
             os.path.join(model_ws, 'MT3D001.UCN'),
             os.path.join(model_ws, modelname + '.cbc')]
    for f in flist:
        try:
            os.remove(f)
        except:
            pass

    # Plot model? 
    if plotyn:
        m.plot_hk_ibound(rowslice=farm_orig[0][0],gridon=True)
        
    # Run model
    if runyn:
        v = m.run_model(silent=silent, report=True)
        for idx in range(-3, 0):  # Report
            print(v[1][idx])

        # Record success/failure and store data
        varlist['success'][it] = v[0]

        if v[0] is False:
            pass
        else:
            # Record final salinity as .npy, also move full CBC and UCN files
            # to expt folder
            fname = os.path.join(m.model_ws, 'MT3D001.UCN')
            totim = flopy.utils.binaryfile.UcnFile(fname).get_times()[-1]
            conc_fname = 'conc{}_{}_totim{}.UCN'.format(
                it, ts, str(int(totim)))
            _ = record_salinity(
                m, ts_hms=ts, fname_write=m.MC_file.parent.joinpath(conc_fname))
            copy_rename(os.path.join(m.model_ws, 'MT3D001.UCN'),
                        m.MC_file.parent.joinpath(conc_fname).as_posix())
    if pooling:
        import shutil
        try:
            # [print(p) for p in tmp.iterdir() if (p.suffix is not '.UCN')]
            [p.unlink() for p in tmp.iterdir() if (p.suffix not in ('.UCN','.list'))]
            # shutil.rmtree(tmp.as_posix())
            # tmp.rmdir()
        except:
            print('didnt work!')
            pass
        m.model_ws = model_ws_orig
        print('resetting ws:',m.model_ws)

        if output is None:
            return (it,varlist['success'][it])
        else:
            output.put((it,varlist['success'][it]))
            # results.append((it,varlist['success'][it]))
            return
    else:
        return m, varlist


def make_varlist_array(varlist,nwel=4):
    flag=0
    i=0
    for k,v in varlist.items():
        if flag==0:
            varlist_arr= np.zeros((len(varlist)+nwel,len(v)),dtype=np.float)
            flag=1
        if k is 'wel':
            for j in range(nwel):
                varlist_arr[i,:] = v[j,:]
                i+=1
        elif k is 'vario_type':
            varlist_arr[i,:] = [0 if model is 'Gaussian' else 1 for model in v]
            i+=1
        else:
            varlist_arr[i,:] = np.asarray(v,dtype=np.float)
            i+=1
    return varlist_arr
#%%



# if __name__ == '__main__':
#     update_run_model(varlist, it)

tot_it = 2
seed = 1
np.random.seed(seed)
varlist = create_varlist(tot_it)


it_select = np.arange(tot_it)
# it_select = (1,)
runyn=True
pooling=False
# output = mp.Queue()

for it in it_select:
   m, varlist = update_run_model(varlist,it,runyn=runyn,
                                 start_basecase=True,silent=False,
                                 pooling=pooling)

varlist_arr = make_varlist_array(varlist)
savemat(m.MC_file.parent.joinpath('inputParams.mat').as_posix(),varlist)
np.save(m.MC_file.parent.joinpath('inputParams_all.npy'),varlist_arr)
save_obj(m.MC_file.parent,varlist,'varlist_final')
save_obj(m.MC_file.parent,m.dis.get_node_coordinates(),'yxz')

#%%
'''
TEST MP
'''   


# import os
# import concurrent.futures
# runyn=True
# pooling=True
# start_basecase=True
# silent=True
# kwargs ={'pooling':pooling,
#         'runyn':runyn,
#         'start_basecase':start_basecase,
#         'silent':silent
#         }


# # start threads
# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     future_to_file = [executor.submit(update_run_model, varlist, it,**kwargs) for it in range(tot_it)]

#     for future in concurrent.futures.as_completed(future_to_file):
#         # print('future',future)
#         # f = future_to_file[future]
#         if future.exception() is not None:
#            print(' exception: %s' % (future.exception()))
#         # run() doesn't return anything so `future.result()` is always `None`


# , args=(varlist,it),) for it in it_select]


# import multiprocessing as mp
# from multiprocessing import Queue, Semaphore
    
# if __name__ == '__main__':
#     print('Running MP:...')
#     it_select = np.arange(tot_it)
#     results = []
#     # it_select = (1,)
#     runyn=True
#     pooling=True
#     start_basecase=True
#     silent=True
#     output = mp.Queue()
#     concurrency = 7
#     sema = Semaphore(concurrency)

#     processes = [mp.Process(target=update_run_model, args=(varlist,it),kwargs={'pooling':pooling,
#                                                                             'runyn':runyn,
#                                                                             'start_basecase':start_basecase,
#                                                                             'silent':silent,
#                                                                             'output':output,
#                                                                             'sema':sema}) for it in it_select]
#     # Run processes
#     for p in processes:
#        p.start()
#     print('All processes started')
#     # Exit the completed processes
#     import time
#     time.sleep(10)

#     results = []
#     while 1:
#         running = any(p.is_alive() for p in processes)
#         while not output.empty():
#            results.append(output.get(False))
#         if not running:
#             break


#         # for p in processes:
#     #    p.terminate()
#     print('All data gathered')

#     for p in processes:
#        p.join()
#     print('All processes joined')
#     # Get process results from the output queue
#     # results = [output.get() for p in processes]
#     print('All done \nresults:',results)
#     save_obj(m.MC_file.parent,results,'results')





#%%
'''
RUN THE MC!
'''
it_select = np.arange(tot_it)
# it_select = (1,)
runyn=True
pooling=False
# output = mp.Queue()

for it in it_select:
   m, varlist = update_run_model(varlist,it,runyn=runyn,
                                 start_basecase=True,silent=False,
                                 pooling=pooling)

varlist_arr = make_varlist_array(varlist)
savemat(m.MC_file.parent.joinpath('inputParams.mat').as_posix(),varlist)
np.save(m.MC_file.parent.joinpath('inputParams_all.npy'),varlist_arr)
save_obj(m.MC_file.parent,varlist,'varlist_final')
save_obj(m.MC_file.parent,m.dis.get_node_coordinates(),'yxz')


#%%

#%%
#import multiprocessing as mp
#it_select = (1,)
#runyn=True
#pooling=True
#start_basecase=True
#silent=False
#output = mp.Queue()
#
#
#processes = [mp.Process(target=update_run_model, args=(m, varlist,it)) for it in range(tot_it)]
##
##kwargs={'output':output,
##                                'pooling':pooling,
##                                'runyn':runyn,
##                                'start_basecase':start_basecase,
##                                'silent':silent}
## Run processes
#for p in processes:
#    p.start()
#
## Exit the completed processes
#for p in processes:
#    p.join()
#
## Get process results from the output queue
#results = [output.get() for p in processes]


#%%
#import multiprocessing as mp
#
#pool = mp.Pool(processes=4)
#results = [pool.apply(update_run_model, args=(m,varlist,it)) for it in range(2)]
#




# # In[19]:
# plotyn = False
# if plotyn:
#     per = np.arange(nper)
#     mas = plot_mas(m)
#     rowslice = 1
#     ts=''
#     for p in per:
#         conc,hds = extract_hds_conc(m,p)
#         basic_plot(m,p,conc,rowslice=rowslice,scale=70,iskip=3,printyn=1,contoursyn=1)
#     m.plot_hk_ibound(rowslice=rowslice,gridon=0,printyn=1)

#%%
#Plot success and failed runs to see if there are any parameters causing the trouble
#printyn=1
#filt = np.arange(len(runlog))[runlog]
#i=0
#input_fail = {}
#input_success = {}
#for k,v in m.inputParams.items():
#    vsucc = [x for i, x in enumerate(v) if i in filt]
#    vfail = [x for i, x in enumerate(v) if i not in filt]
#    input_fail[k] = vfail
#    input_success[k] = vsucc
#    if i==0:
#        Nsucc = len(vsucc)
#        Nfail = len(vfail)
#    i+=1
#
#
#succMat = np.zeros((Nsucc,len(input_success)))
#for i,key in enumerate(input_success):
#    succMat[:,i] = np.asarray(input_success[key])
#
#failMat = np.zeros((Nfail,len(input_fail)))
#for i,key in enumerate(input_fail):
#    failMat[:,i] = np.asarray(input_fail[key])
#
#f, axs = plt.subplots(nrows=5,ncols=4, figsize=(6, 8))
#it=0
#for i in range(5):
#    for j in range(4):
#        ttl = str(it) + str(list(input_success.keys())[it])
#        plt.sca(axs[i,j])
#        plt.title(ttl)
#        plt.hist(succMat[:,it],label='succes')
#        plt.hist(failMat[:,it],label='fail')
#        plt.tick_params(
#            axis='y',          # changes apply to the x-axis
#            which='both',      # both major and minor ticks are affected
#            bottom=False,      # ticks along the bottom edge are off
#            top=False,         # ticks along the top edge are off
#            left=False,
#            right=False,
#            labelleft=False)
#        it+=1
#        if it==1:
#            plt.legend()
#if printyn == 1:
#    plt.savefig(str(m.MC_file.parent.joinpath('failplot.png')),dpi=200)
#



#%% Calculate hausdorff matrix and export
#import hausdorff_from_dir
##importlib.reload(hausdorff_from_dir)
#pers = np.cumsum(perlen)
#totims =  np.r_[pers[::5],pers[-1]]
##totims = (2340.0,4860.0,7200.0)
#fnames = hausdorff_from_dir.create_concmat_from_ucndir(m.MC_file.parent,totims=totims,modsize=hk.shape)
#yxz = load_obj(m.MC_file.parent,'yxz')
#
#for i in range(len(totims)):
#    hausdorff_from_dir.compute_export_hausdorff(m.MC_file.parent,
#                                                conc_mat=np.load(m.MC_file.parent.joinpath(fnames[i])),
#                                                yxz=yxz,suffix=str(totims[i]))
#
#
#
#
#
#

