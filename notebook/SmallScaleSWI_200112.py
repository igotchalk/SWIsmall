# coding: utf-8
'''
Small-scale SWI
Saltwater intrusion model paired with AEM data acquisition
for use with sensitivity analysis
'''


# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %aimport SGD,utils
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as sts
import flopy
import SGD
import config
import hausdorff_from_dir

print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))
# Name model
modelname = 'homogenous'
tot_it = 10

if sys.platform == "darwin":
    repo = Path(__file__).parent.parent  # repo-->notebook-->__file__
    model_ws = repo.joinpath('work', modelname)
elif sys.platform == "win32":
    repo = Path('E:\Projects\DelawareSGD')
    model_ws = os.path.join('E:\Projects\DelawareSGD', 'work', modelname)

# if sys.platform == "darwin":
#     model_ws = Path('/Users/ianpg/Documents/ProjectsLocal/DelawareSGD/work').joinpath(modelname)
# elif sys.platform == "win32":
#     model_ws = Path(r'E:\Projects\DelawareSGD\work').joinpath(modelname)
# #     model_ws = os.path.join('E:\Projects\DelawareSGD','work',modelname)

if not model_ws.exists():
    model_ws.mkdir()

# sys.path.append(repo.joinpath('notebook').as_posix())
sw_exe = config.swexe  # set the exe path for seawat
print('Model workspace:', os.path.abspath(model_ws))

workdir = Path('../')
figdir = workdir.joinpath('work', 'figs')
datadir = workdir.joinpath('data')
objdir = workdir.joinpath('data', 'objs')


# utility functions
def load_obj(dirname, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)


def save_obj(dirname, obj, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Create new MC_file
def create_MC_file():
    import datetime
    ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    MC_dir = Path(os.path.join(m.model_ws, 'MC_expt_' + ts))
    if not MC_dir.exists():
        MC_dir.mkdir()
    m.MC_file = MC_dir.joinpath('expt.txt')
    with m.MC_file.open('w') as wf:
        wf.close
    print(m.MC_file)
    return


# nearest value in array
def find_nearest(array, value):
    import numpy as np
    idx = (np.abs(array - value)).argmin()
    idx.astype('int')
    return array[idx]


# take distance in meters, return column in model
def loc_to_col(locs):
    cols = [int(find_nearest((np.arange(ncol) * delc), loc)) for loc in locs]
    return cols


# make a line across the grid
def get_line(start, end, allrows=1, nrow=None):
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
        if allrows == 1:
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


# make all cells=0 above the line from get_line()
# Calculate freshwater head based on column of saltwater above each node
# (rho*g*z)
def shade_above(nlay, nrow, ncol, point_list, third_dim=1):
    import numpy as np
    grd = np.ones((nlay, nrow, ncol), dtype='int')
    ocean_hf = []
    if len(point_list) == 0:
        return grd, ocean_hf
    for (lay, row, col) in point_list:
        grd[lay, :, col] = -1  # assign ocean ibound to -1
        grd[:lay, :, col] = 0  # assign cells above ocean to 0
        hf = densefresh / densesalt * ocean_elev - \
            (densesalt - densefresh) / densefresh * \
            (henry_botm[lay] + .5 * delv)
        for irow in range(nrow):
            ocean_hf.append((int(lay), int(irow), int(col), hf))
    ocean_hf = tuple(np.array(ocean_hf).T)
    ocean_hf = (ocean_hf[0].astype('int'),
                ocean_hf[1].astype('int'),
                ocean_hf[2].astype('int'),
                ocean_hf[3])
    return grd, ocean_hf


def get_ocean_right_edge(m, ocean_line_tuple, startlay=None, col=None):
    import numpy as np
    point_list = []

    if col is None:
        col = m.ncol - 1
    # If there is no vertical side boundary, return bottom-right corner node
    if len(ocean_line_tuple) == 0:
        if startlay is None:
            startlay = 0
    elif max(ocean_line_tuple[0]) == m.nlay:
        startlay = m.nlay
    elif max(ocean_line_tuple[0]) < m.nlay:
        startlay = max(ocean_line_tuple[0])
    for lay in range(startlay, m.nlay):
        for row in range(m.nrow):
            point_list.append((lay, row, col))
    point_list = tuple(np.array(point_list).T)
    return point_list

# create random clay blocks throughout model (quick dirty way to do
# heterogeneity)


def rand_clay_blocks(lithmat, hkClay, numblocks, sizeblocks):
    nlay, nrow, ncol = lithmat.shape
    lay_block = np.random.randint(1, nlay - sizeblocks[0], numblocks)
    row_block = np.random.randint(0, nrow - sizeblocks[1] + 1, numblocks)
    col_block = np.random.randint(1, ncol - sizeblocks[2], numblocks)
    lithmat_blocks = lithmat.copy()
    for i in range(numblocks):
        block_coords = [slice(lay_block[i], lay_block[i] + sizeblocks[0]),
                        slice(row_block[i], row_block[i] + sizeblocks[1]),
                        slice(col_block[i], col_block[i] + sizeblocks[2])]
        lithmat_blocks[block_coords] = hkClay
    return lithmat_blocks


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
        use_existing_MCfile = get_yn_response(
            "m.MC_file already exists, continue using this experiment?")
    else:
        use_existing_MCfile = False
    if use_existing_MCfile:
        if m.inputParams is not None:
            if len(m.inputParams) > 0:
                add_to_inputParams = get_yn_response(
                    "m.inputParams already has entries, do you want to add to it?")
            else:
                add_to_inputParams = False
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
            m.inputParams = load_obj(Path(f), 'inputParams')
            print('loaded .pkl file!')
        else:
            create_MC_file()
            m.inputParams = {}
    return


def add_pumping_wells(wel_data, ssm_data, n_wells, flx, rowcol, kper):
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    new_weldata = wel_data
    new_ssmdata = ssm_data
    wel_cells = []
    for k in range(n_wells):
        row, col = rowcol[k]
        for i in range(nper):
            if i in kper:
                for j in range(nlay):
                    # WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append(
                        [j, row, col, -flx[k] * delv_weight[j]])
                    wel_cells.append((j, row, col))
                    # SSM: {stress_period: [lay,row,col,concentration,itype]}
                    # since it's a sink, conc. doesn't matter
                    new_ssmdata[i].append([j, row, col, Cfresh, itype['WEL']])
            else:
                for j in range(nlay):
                    # WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append([j, row, col, 0])
                    # SSM: {stress_period: [lay,row,col,concentration,itype]}
                    # since it's a sink, conc. doesn't matter
                    new_ssmdata[i].append([j, row, col, Cfresh, itype['WEL']])
                    wel_cells.append((j, row, col))
                continue
    wel_cells = tuple(np.array(list(set(wel_cells))).T)
    return new_weldata, new_ssmdata, wel_cells


# Add recharge if desired
def make_rech_array(low=1e-2, high=1e0):
    llow, lhigh = np.log10((low, high))
    rech = np.exp(sts.uniform.rvs(size=1, loc=llow, scale=lhigh - llow)[0])
    return rech / (nrow * ncol)


def add_recharge_cells(recharge_generator, const=1, *args):
    if const == 1:
        rech_data = recharge_generator(*args)
    else:
        rech_data = {}
        for i in range(nper):
            rech_array = recharge_generator(*args)
        rech_data[i] = rech_array
    return rech_data


def sample_dist(distclass, size, writeyn=0, model=None, varname=None, log_backtransf=0, *args):
    smp = distclass.rvs(*args, size=size)
    if log_backtransf == 1:
        smp = tuple([10**x for x in smp])
    if size == 1:
        smp = smp[-1]
    if writeyn == 1:
        write_sample(model.MC_file, varname, distclass, smp)
    return smp


def write_sample(fname, varname, distclass, sample):
    fout = open(fname, "a")
    fout.write(varname + ',' + str(type(distclass)) + ',' + str(sample) + '\n')
    fout.close()
    return


def truncate_grf(grid, lith_props, hk_vals, log10trans=True, plotyn=False, saveyn=False):
    grid_cutoffs = []
    for q in np.cumsum(lith_props):
        grid_cutoffs.append(np.quantile(grid, q))

    if plotyn:
        h = plt.hist(grid.flatten())
        for cutoff in grid_cutoffs:
            plt.vlines(cutoff, 0, 14000)
        plt.show()

    outgrid = np.ones(grid.shape, dtype=np.float32)
    for i, cutoff in reversed(list(enumerate(grid_cutoffs))):
        outgrid[np.where(grid < cutoff)] = hk_vals[i]

    if plotyn:
        f, axs = plt.subplots(2, 1, sharex=True)
        axs[0].imshow(grid[:, 0, :])
        axs[1].imshow(outgrid[:, 0, :])
        if saveyn:
            plt.savefig(m.MC_file.parent.joinpath(
                'Truncated_GRF.png').as_posix(), resolution=300)
    if log10trans:
        return np.power(10, outgrid)
    else:
        return outgrid

#%%
# Name model


def pec_num(delv, delc, delr, al):
    delL = (delv, delc, delr)  # length in the lay,row,col directions
    pec_num = [round(d / al, 2) for d in delL]
    for num, point in zip(pec_num, ('lay', 'row', 'col')):
        print('Pe = {} in the {} direction'.format(num, point))
    return pec_num
#%%

sw_exe = config.swexe  # set the exe path for seawat
print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))
print('Model workspace:', os.path.abspath(model_ws))
# Model grid
# Grid: 0.5 *1*1m • Size:70m*20m

Lx = 3000.
Ly = 600.
Lz = 80.

xul = 488930.5 - 3  # minimum x value - 3
yul = 4271464.0 - 32  # minimum y value -32 to place in front of breakwater
rotation = 0

henry_top = 3
ocean_elev = 0
delv_first = 5


botm_first = henry_top - delv_first

nlay = int(Lz * 1 / 3)
nrow = int(Ly * (1 / 30))
ncol = int(Lx * (1 / 30))

delv = (Lz - delv_first) / (nlay - 1)
delr = Lx / ncol
delc = Ly / nrow

henry_botm = np.hstack(([botm_first], np.linspace(
    botm_first - delv, henry_top - Lz, nlay - 1)))
delv_vec = np.hstack((delv_first, np.repeat(delv, nlay - 1)))
delv_weight = [x / np.sum(delv_vec) for x in delv_vec]

beachslope = .05

# dev
# Create ocean boundary at top of model
# Manually done to make sure it's in the right place rn
ocean_col = [np.floor(ncol - 1).astype('int'), ncol - 1]

inland_elev = beachslope * ocean_col[0] * delr
offshore_elev = -beachslope * (ocean_col[1] - ocean_col[0]) * delr

#beach_arr = matlib.repmat(np.linspace(inland_elev,offshore_elev,num=ncol),nrow,1)
#henry_top = ocean_elev*np.ones((nrow,ncol), dtype=np.float) + beach_arr
#henry_botm = matlib.repmat(np.linspace(ocean_elev - delv, ocean_elev - Lz, nlay),ncol,1).transpose() + beach_arr[1]
# henry_botm = np.transpose(np.tile(henry_botm,(nrow,1,1)),
#                          (1,0,2))
# dev


# Period data
Lt = 360 * 20  # Length of time in days
perlen = list(np.repeat(180, int(Lt / 180)))
nstp = list(np.ones(np.shape(perlen), dtype=int))

nper = len(perlen)
steady = [False for x in range(len(perlen))]  # Never steady
itmuni = 4  # time unit 4= days
lenuni = 2  # length unit 2 = meter
tsmult = 1.8
ssm_data = None
verbose = True


# Create basic model instance and dis pacakge
m = flopy.seawat.Seawat(modelname, exe_name=sw_exe,
                        model_ws=model_ws, verbose=verbose)
SGD.ModelSGD.Seawat2SGD(m)  # convert to subclass ModelSGD
print(m.namefile)

# Add DIS package to the MODFLOW model
dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=nper, delr=delr,
                               delc=delc, xul=xul, yul=yul, rotation=rotation,
                               laycbd=0, top=henry_top,
                               botm=henry_botm, perlen=perlen, nstp=nstp,
                               steady=steady, itmuni=itmuni, lenuni=lenuni,
                               tsmult=tsmult)

# Hydraulic conductivity field
hkSand = 80.  # horizontal hydraulic conductivity m/day
hkClay = 1.
lithmat = hkSand * np.ones((nlay, nrow, ncol),
                           dtype=np.int32)  # sandy background
addclay_yn = 0


######## ADDED BLOCK #########

# Hydraulic conductivity field
hkSand = 1e2  # horizontal hydraulic conductivity m/day
hkClay = hkSand * .01

heterogenous = 1  # 0:homogenous,1:variogram,2:MPS
clay_lyr = True


if heterogenous == 1:
    import simulationFFT
    mu = np.log(hkSand)
    sill = .1
    modeltype = 'Exponential'
    llay = int(20 / np.mean(delv))
    lrow = int(2000 / delc)
    lcol = int(2000 / delr)

    fft_grid = np.exp(simulationFFT.simulFFT(
        nrow, nlay, ncol, mu, sill, modeltype, lrow, llay, lcol))
    grid = np.log10(fft_grid)
    #lith_props = [0.2,0.5,0.3]
    #hk_vals = [-1,0,2]
    lith_props = [0.2, 0.8]
    hk_vals = [0, 2]

    log10trans = True
    plotyn = True
    hk = truncate_grf(grid, lith_props, hk_vals,
                      log10trans=True, plotyn=False, saveyn=True)

    if clay_lyr:
        lyr_ind = np.where((henry_botm < -2) & (henry_botm > -15))
        hk[lyr_ind, :, :int(ncol * 3 / 4)] = 10**hk_vals[0]
        if plotyn:
            plt.imshow(hk[:, 0, :])

elif heterogenous == 2:
    import sgs_mod
    nodes = 20
    marg = .5
    search_ellipse = (2000, 500, 20, 0, 0, 0)  # (max,med,min,az,dip,rake)
    grid_size = (Lz, Ly, Lx)
    grid_cells = dim
    rotind = [1, 2, 0]  # simulation in a Y,X,Z grid
    constrain = 1
    expgridfile, outgrid, rotind, grid_cells_sgems = sgs_mod.snesim_grid(m.name, Path(model_ws),
                                                                         grid_size, grid_cells, search_ellipse=search_ellipse,
                                                                         TIfile=None, TIname=None, marg=marg, seed=1, nodes=nodes,
                                                                         nreals=1, output=False, rmfiles=False, rotind=rotind, constrain=constrain)
    outgrid = sgs_mod.read_sgems_grid(
        expgridfile, grid_cells, grid_cells_sgems)
    outgrid = outgrid.squeeze()
    hk = np.zeros(dim, dtype=np.float)
    hk[np.where(outgrid == 0)] = hkClay
    hk[np.where(outgrid == 1)] = hkSand
    f, axs = plt.subplots(2, 1, sharex=True)
    plt.sca(axs[0])
    plt.imshow(hk[-3, :, :])
    axs[0].set_title('top view')
    plt.sca(axs[1])
    plt.imshow(hk[:, 0, :])
    axs[1].set_title('side view')
else:
    hk = hkSand * np.ones((nlay, nrow, ncol), dtype=np.int32)

# plt.figure(),plt.imshow((hk[:,0,:])),plt.colorbar(),plt.title('Sill:{}'.format(sill)),plt.show()

# Set Hydraulic properties
sy = 0.24
ss = 1e-5
por = 0.3
vka = 1  # = vk/hk
al = 1  # longitudinal dispersivity (m) from Walther et al. 2017
dmcoef = 2e-9  # m2/day

# Variable density parameters
Csalt = 35.0001
Cfresh = 0.
densesalt = 1025.
densefresh = 1000.
denseslp = (densesalt - densefresh) / (Csalt - Cfresh)
# denseslp = 0 #trick for testing constant density
######## END ADDED BLOCK ####


# add random low conductivity regions
if addclay_yn == 1:
    lithmat = rand_clay_blocks(lithmat, hkClay, 100, (2, 1, 5))

# Set Hydraulic properties

hk = lithmat
sy = 0.15
ss = 0.00005
por = 0.2
vka = 1 / 10  # = vk/hk
al = 10  # longitudinal dispersivity (m) from Walther et al. 2017
dmcoef = 2e-9  # m2/day  from Walther et al. 2017

# Variable density parameters
Csalt = 35.0001
Cfresh = 0.
densesalt = 1025.
densefresh = 1000.
denseslp = (densesalt - densefresh) / (Csalt - Cfresh)
# denseslp = 0 #trick for testing constant density


# Model discretization
Lx = 3000.
Ly = 600.
Lz = 80.

nlay = int(Lz / 3)
nrow = int(Ly / 30)
ncol = int(Lx / 30)
#Lx = 300.
#Ly = 100.
#Lz = 20.
#nlay = int(Lz/3)
#nrow = int(Ly/3)
#ncol = int(Lx/3)

dim = tuple([int(x) for x in (nlay, nrow, ncol)])

henry_top = 5
ocean_elev = 0

delv_first = Lz / nlay
#delv_first = 5
botm_first = henry_top - delv_first

delv = (Lz - delv_first) / (nlay - 1)
delr = Lx / ncol
delc = Ly / nrow

henry_botm = np.hstack(([botm_first], np.linspace(
    botm_first - delv, henry_top - Lz, nlay - 1)))
delv_vec = np.hstack((delv_first, np.repeat(delv, nlay - 1)))
delv_weight = [x / np.sum(delv_vec) for x in delv_vec]

beachslope = .05
# Manually done to make sure it's in the right place rn
ocean_col = [np.floor(ncol - 1).astype('int'), ncol - 1]
inland_elev = beachslope * ocean_col[0] * delr
offshore_elev = -beachslope * (ocean_col[1] - ocean_col[0]) * delr


# Period data
nyrs = 20
Lt = 360 * nyrs  # Length of time in days
perlen = list(np.repeat(180, int(Lt / 180)))
nstp = list(np.ones(np.shape(perlen), dtype=int))

nper = len(perlen)
steady = [False for x in range(len(perlen))]  # Never steady
itmuni = 4  # time unit 4= days
lenuni = 2  # length unit 2 = meter
tsmult = 1.8
ssm_data = None
verbose = True

print('Model setup: \n'
      'nlay: {}\n'
      'nrow: {}\n'
      'ncol: {}\n'
      'Total cells: {}\n'
      'Total time: {} days\n'
      'nper: {}\n'.format(nlay, nrow, ncol, nlay * nrow * ncol, Lt, nper))
# In[4]:

# Create basic model instance and dis pacakge
m = flopy.seawat.Seawat(modelname, exe_name=sw_exe,
                        model_ws=model_ws, verbose=verbose)
SGD.ModelSGD.Seawat2SGD(m)  # convert to subclass ModelSGD
print(m.namefile)

# Add DIS package to the MODFLOW model
dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=nper, delr=delr,
                               delc=delc,
                               laycbd=0, top=henry_top,
                               botm=henry_botm, perlen=perlen, nstp=nstp,
                               steady=steady, itmuni=itmuni, lenuni=lenuni,
                               tsmult=tsmult)
create_MC_file()
# In[5]:

# Hydraulic conductivity field
hkSand = 1e2  # horizontal hydraulic conductivity m/day
hkClay = hkSand * .01

heterogenous = 1  # 0:homogenous,1:variogram,2:MPS
clay_lyr = True


if heterogenous == 1:
    import simulationFFT
    mu = np.log(hkSand)
    sill = .1
    modeltype = 'Exponential'
    llay = int(20 / np.mean(delv))
    lrow = int(2000 / delc)
    lcol = int(2000 / delr)

    fft_grid = np.exp(simulationFFT.simulFFT(
        nrow, nlay, ncol, mu, sill, modeltype, lrow, llay, lcol))
    grid = np.log10(fft_grid)
    #lith_props = [0.2,0.5,0.3]
    #hk_vals = [-1,0,2]
    lith_props = [0.2, 0.8]
    hk_vals = [0, 2]

    log10trans = True
    plotyn = True
    hk = truncate_grf(grid, lith_props, hk_vals,
                      log10trans=True, plotyn=False, saveyn=True)

    if clay_lyr:
        lyr_ind = np.where((henry_botm < -2) & (henry_botm > -15))
        hk[lyr_ind, :, :int(ncol * 3 / 4)] = 10**hk_vals[0]
        if plotyn:
            plt.imshow(hk[:, 0, :])

elif heterogenous == 2:
    import sgs_mod
    nodes = 20
    marg = .5
    search_ellipse = (2000, 500, 20, 0, 0, 0)  # (max,med,min,az,dip,rake)
    grid_size = (Lz, Ly, Lx)
    grid_cells = dim
    rotind = [1, 2, 0]  # simulation in a Y,X,Z grid
    constrain = 1
    expgridfile, outgrid, rotind, grid_cells_sgems = sgs_mod.snesim_grid(m.name, Path(model_ws),
                                                                         grid_size, grid_cells, search_ellipse=search_ellipse,
                                                                         TIfile=None, TIname=None, marg=marg, seed=1, nodes=nodes,
                                                                         nreals=1, output=False, rmfiles=False, rotind=rotind, constrain=constrain)
    outgrid = sgs_mod.read_sgems_grid(
        expgridfile, grid_cells, grid_cells_sgems)
    outgrid = outgrid.squeeze()
    hk = np.zeros(dim, dtype=np.float)
    hk[np.where(outgrid == 0)] = hkClay
    hk[np.where(outgrid == 1)] = hkSand
    f, axs = plt.subplots(2, 1, sharex=True)
    plt.sca(axs[0])
    plt.imshow(hk[-3, :, :])
    axs[0].set_title('top view')
    plt.sca(axs[1])
    plt.imshow(hk[:, 0, :])
    axs[1].set_title('side view')
else:
    hk = hkSand * np.ones((nlay, nrow, ncol), dtype=np.int32)

# plt.figure(),plt.imshow((hk[:,0,:])),plt.colorbar(),plt.title('Sill:{}'.format(sill)),plt.show()

# Set Hydraulic properties
sy = 0.24
ss = 1e-5
por = 0.3
vka = 1  # = vk/hk
al = 1  # longitudinal dispersivity (m) from Walther et al. 2017
dmcoef = 2e-9  # m2/day

# Variable density parameters
Csalt = 35.0001
Cfresh = 0.
densesalt = 1025.
densefresh = 1000.
denseslp = (densesalt - densefresh) / (Csalt - Cfresh)
# denseslp = 0 #trick for testing constant density


# In[8]:
# Winter is even stress periods, summer is odd SP.
# Winter= wells OFF, natural precip (rech) ON, irrigation rech OFF,
# Summer = wells ON, irrigation rech (farm_rech) ON,  precip (rech) OFF
kper_odd = list(np.arange(1, nper, 2))
kper_even = list(np.arange(0, nper, 2))

# BCs
bc_ocean = 'GHB'
bc_right_edge = 'GHB'
bc_inland = 'GHB'
add_wells = 0
n_wells = 0
rech_on = 0


# Inland
calc_inland_head = 0  # calculate from hgrad

manual_inland_head = 0.3184


start_fresh_yn = 1

ocean_shead = [ocean_elev for x in range(len(perlen))]
ocean_ehead = ocean_shead

# save cell fluxes to unit 53
ipakcb = 53


# Create ocean boundary at top of model
ocean_col_vec = (0, 0, np.arange(ocean_col[0], ocean_col[1] + 1))
ocean_coords = (0, slice(0, nrow), slice(
    ocean_col[0], ocean_col[1] + 1))  # top of the model
ocean_bool = np.zeros((nlay, nrow, ncol), dtype=bool)
ocean_bool[0, :, np.arange(ocean_col[0], ocean_col[1] + 1)] = 1
m.ocean_arr = ocean_bool


if calc_inland_head == 1:
    head_inland = ocean_col[0] * delc * hgrad + ocean_elev
else:
    head_inland = manual_inland_head

####IN TESTING#####
# Create a line of where the ocean is, and any nodes on right edge below ocean
offshore_lay = (np.abs(henry_botm - offshore_elev)).argmin().astype('int')
if ocean_col[0] == ncol - 1:
    ocean_line = []
    bc_ocean = 'XXX'
else:
    ocean_line = get_line(
        (0, ocean_col[0]), (offshore_lay, ocean_col[1]), allrows=1, nrow=1)

# use this for indexing numpy arrays
ocean_line_tuple = tuple(np.array(ocean_line).T)
right_edge = get_ocean_right_edge(m, ocean_line_tuple,
                                  int(np.where(henry_botm == find_nearest(henry_botm, ocean_elev))[0]))
left_edge = get_ocean_right_edge(m, ocean_line_tuple,
                                 int(np.where(henry_botm == find_nearest(
                                     henry_botm, head_inland))[0]),
                                 col=0)

# Create ibound
ibound, ocean_hf = shade_above(
    nlay, nrow, ncol, ocean_line)  # don't set ibound of ocean

#ibound[:right_edge[0][0],right_edge[1][0],right_edge[2][0]] = 0
#ibound[:right_edge[0][0],right_edge[1][0],0] = 0

if bc_ocean == 'GHB':
    ibound[ocean_line_tuple] = 1


# Set starting heads


strt = np.zeros((nlay, nrow, ncol), dtype=np.int)


# Transport BCs
if start_fresh_yn == 1:
    sconc = Cfresh * np.ones((nlay, nrow, ncol),
                             dtype=np.float32)  # Begin fresh
elif start_fresh_yn == 0:
    sconc = Csalt * np.ones((nlay, nrow, ncol),
                            dtype=np.float32)  # Begin SW-saturated
else:
    sconc = Cfresh * np.ones((nlay, nrow, ncol),
                             dtype=np.float32)  # Begin SW-saturated
    sconc[:, :, int(np.floor(ncol / 2)):-1] = Csalt

if ocean_hf:
    sconc[ocean_hf[0:3]] = Csalt
sconc[right_edge] = Csalt

sconc[:, :, 0] = Cfresh

icbund = np.ones((nlay, nrow, ncol), dtype=np.int)


icbund[np.where(ibound == -1)] = -1


head_inland_sum_wint = (0, 1)  # m in summer, m in winter


def make_bc_dicts(head_inland_sum_wint=(0, 0.3)):
    # Ocean and inland boundary types
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
        # Ocean boundary
        if ocean_hf:
            for j in range(np.size(ocean_hf[0])):
                if bc_ocean == 'CHD':
                    # CHD: {stress_period: [lay,row,col,starthead,endhead]}
                    dat_chd.append([ocean_line_tuple[0][j],
                                    ocean_line_tuple[1][j],
                                    ocean_line_tuple[2][j],
                                    ocean_shead[i],
                                    ocean_ehead[i]])
                    # SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([ocean_line_tuple[0][j],
                                    ocean_line_tuple[1][j],
                                    ocean_line_tuple[2][j],
                                    Csalt,
                                    itype['CHD']])
                elif bc_ocean == 'GHB':
                    # GHB: {stress period: [lay,row,col,head level,conductance]}
                    # conductance c = K*A/dL; assume horizontal flow at outlet,
                    # and calculate length to be at edge of ocean cell, as opposed to mipoint
                    # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
                    dat_ghb.append([ocean_hf[0][j],
                                    ocean_hf[1][j],
                                    ocean_hf[2][j],
                                    # ocean_hf[3][j],
                                    ocean_elev,
                                    2 * hkSand * delc * delv_vec[ocean_hf[0][j]] / delr])
                    # SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([ocean_hf[0][j],
                                    ocean_hf[1][j],
                                    ocean_hf[2][j],
                                    Csalt,
                                    itype['GHB']])
        else:
            pass
        # Right edge boundary
        if bc_right_edge == 'GHB':
            for j in range(np.size(right_edge[0])):
                # GHB: {stress period: [lay,row,col,head level,conductance]}
                # conductance c = K*A/dL; assume horizontal flow at outlet,
                # and calculate length to be at edge of ocean cell, as opposed to mipoint
                # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
                dat_ghb.append([right_edge[0][j],
                                right_edge[1][j],
                                right_edge[2][j],
                                # ocean_hf[3][j],
                                ocean_elev,
                                2 * hkSand * delc * delv_vec[right_edge[0][j]] / delr])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}
                dat_ssm.append([right_edge[0][j],
                                right_edge[1][j],
                                right_edge[2][j],
                                Csalt,
                                itype['GHB']])
        else:
            pass
        # Inland boundary
        if bc_inland == 'GHB':

            if i in kper_odd:
                head_inland = head_inland_sum_wint[0]
            elif i in kper_even:
                head_inland = head_inland_sum_wint[1]
            left_edge = get_ocean_right_edge(m, ocean_line_tuple,
                                             int(np.where(henry_botm == find_nearest(
                                                 henry_botm, head_inland))[0]),
                                             col=0)
            for j in range(np.size(left_edge[0])):
                dat_ghb.append([left_edge[0][j],
                                left_edge[1][j],
                                left_edge[2][j],
                                head_inland,
                                2 * hkSand * delc * delv_vec[left_edge[0][j]] / delr])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}

                dat_ssm.append([left_edge[0][j],
                                left_edge[1][j],
                                left_edge[2][j],
                                Cfresh,
                                itype['GHB']])
        elif bc_inland == 'WEL':
            for j in range(nlay):
                for k in range(nrow):
                    # WEL: {stress_period: [lay,row,col,flux]}
                    dat_wel.append([j, k, 0, influx * delv_weight[j] / nrow])
                    # SSM: {stress_period: [lay,row,col,concentration,itype]}
                    dat_ssm.append([j, k, 0, Cfresh, itype['WEL']])
        chd_data[i] = dat_chd
        ssm_data[i] = dat_ssm
        ghb_data[i] = dat_ghb
        wel_data[i] = dat_wel

    # saving concentrations at specified times
    #timprs = [k for k in range(1,np.sum(perlen),50)]
    return chd_data, ssm_data, ghb_data, wel_data

chd_data, ssm_data, ghb_data, wel_data = make_bc_dicts(head_inland_sum_wint)
wel_data_base, ssm_data_base = wel_data, ssm_data
timprs = np.round(np.linspace(1, np.sum(perlen), 20), decimals=0)


save_obj(m.MC_file.parent, wel_data_base, 'wel_data_base')
save_obj(m.MC_file.parent, ssm_data_base, 'ssm_data_base')

# In[9]:

#### ADD WELL AND RECHARRGE DATA####
# Winter is even stress periods, summer is odd SP.
# Winter= wells OFF, natural precip (rech) ON, irrigation rech OFF,
# Summer = wells ON, irrigation rech (farm_rech) ON,  precip (rech) OFF

# Add recharge data
rech = 1e-6

# Assign the location of the farms
farm_leftmargin = 10
farm_leftmargin = int(ncol / 2)
farm_uppermargin = 1
nfarms = 4
farm_size = (200, 200)  # size in meters (row,col)

# size of farm in number of row,col
farm_size_rowcol = (int(farm_size[0] / delc), int(farm_size[1] / delr))

farm_loc_list = []

farm_orig = []
for x in range(int(nfarms / 2)):
    for y in range(2):
        farm_loc_r = []
        farm_loc_c = []
        for z1 in range(farm_size_rowcol[0]):
            for z2 in range(farm_size_rowcol[1]):
                farm_loc_r.append(farm_uppermargin + y *
                                  (farm_size_rowcol[0] + 2) + z1)
                farm_loc_c.append(farm_leftmargin + x *
                                  (farm_size_rowcol[1] + 2) + z2)
                if (z1 == 0) and (z2 == 0):
                    # upper left of ea. farm=loc of well
                    farm_orig.append((farm_loc_r[-1], farm_loc_c[-1]))
        farm_loc_list.append((np.array(farm_loc_r), np.array(farm_loc_c)))
farm_loc = (np.array(farm_loc_r), np.array(farm_loc_c))


# Add well data
n_wells = nfarms
wel_flux = list(np.ones(n_wells) * 10)
top_lay = int(np.where(henry_botm == find_nearest(
    henry_botm, ocean_elev))[0]) + 1


wel_data, ssm_data, wel_cells = add_pumping_wells(wel_data_base.copy(
), ssm_data_base.copy(), n_wells, wel_flux, farm_orig, kper_odd)


hk[wel_cells] = hkSand


# Add farm recharge data
farm_rech_flux = [flx * 0.2 for flx in wel_flux]


farm_rech = np.zeros((nrow, ncol), dtype=np.float)

for i in range(nfarms):
    farm_rech[farm_loc_list[i]] = farm_rech_flux[i] / np.prod(farm_size)
# Set rech_data for winter and summer
rech_data = {}
for i in range(len(perlen)):
    if i in kper_even:

        rech_data[i] = rech
    elif i in kper_odd:
        rech_data[i] = farm_rech

# In[10]:

riv_loc = get_line((0, 0), (0, ncol - 1), allrows=1, nrow=nrow)
riv_loc = [x for x in riv_loc if x[1] == int(nrow / 2)]
riv_loc = tuple(np.array(riv_loc).T)

riv_grad = .001
rbot_vec = np.linspace(riv_grad * Lx, ocean_elev, ncol)

# Stage and conductance:
stage = 1
cond = 10
riv_grad = .001


def write_river_data(riv_loc, stage, cond, riv_grad, kper, ssm_data):

    ####ADD RIVER DATA####
    rbot_vec = np.linspace(riv_grad * Lx, ocean_elev, ncol)

    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    riv_data = {}
    new_ssm_data = ssm_data
    for i in range(nper):
        dat_riv = []
        if i in kper:
            for j in range(np.size(riv_loc[0])):
                # RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                riv_loc[1][j],
                                riv_loc[2][j],
                                stage + rbot_vec[riv_loc[2][j]],
                                cond,
                                rbot_vec[riv_loc[2][j]]])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        else:
            for j in range(np.size(riv_loc[0])):
                # RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                riv_loc[1][j],
                                riv_loc[2][j],
                                # set stage as bottom of river
                                rbot_vec[riv_loc[2][j]],
                                cond,
                                rbot_vec[riv_loc[2][j]]])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        riv_data[i] = dat_riv
    return riv_data, new_ssm_data

riv_data, ssm_data = write_river_data(
    riv_loc, stage, cond, riv_grad, kper_even, ssm_data)

# In[9]:


# Output control
oc_data = {}
for kper in range(nper):
    oc_data[(kper, 0)] = ['save head', 'save budget']


# In[10]:

# Create instances in flopy
bas = flopy.modflow.ModflowBas(m, ibound, strt=strt)
if bc_ocean == 'CHD' or bc_inland == 'CHD':
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chd_data)
if bc_ocean == 'GHB' or bc_inland == 'GHB'or bc_right_edge == 'GHB':
    ghb = flopy.modflow.ModflowGhb(m, stress_period_data=ghb_data)


rch = flopy.modflow.ModflowRch(m, rech=rech_data)
wel = flopy.modflow.ModflowWel(m, stress_period_data=wel_data, ipakcb=ipakcb)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=riv_data)

# Add LPF package to the MODFLOW model
lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=vka, ipakcb=ipakcb, laytyp=1, laywet=1,
                               ss=ss, sy=sy)

# Add PCG Package to the MODFLOW model
pcg = flopy.modflow.ModflowPcg(m, hclose=1.e-8)

# Add OC package to the MODFLOW model
oc = flopy.modflow.ModflowOc(m,
                             stress_period_data=oc_data,
                             compact=True)

# Create the basic MT3DMS model structure
btn = flopy.mt3d.Mt3dBtn(m,
                         laycon=lpf.laytyp, htop=henry_top,
                         dz=dis.thickness.get_value(), prsity=por, icbund=icbund,
                         sconc=sconc, nprs=1, timprs=timprs)
adv = flopy.mt3d.Mt3dAdv(m, mixelm=-1)
dsp = flopy.mt3d.Mt3dDsp(m, al=al, dmcoef=dmcoef)
gcg = flopy.mt3d.Mt3dGcg(m, iter1=50, mxiter=1, isolve=1, cclose=1e-5)
ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

#vdf = flopy.seawat.SeawatVdf(m, iwtable=0, densemin=0, densemax=0,denseref=1000., denseslp=0.7143, firstdt=1e-3)
vdf = flopy.seawat.SeawatVdf(m, mtdnconc=1, mfnadvfd=1, nswtcpl=0, iwtable=1,
                             densemin=0., densemax=0., denseslp=denseslp, denseref=densefresh)

# # In[11]:

# printyn = 0
# gridon=0
# rowslice=0

# rowslice=farm_orig[0][0]
# m.plot_hk_ibound(rowslice=rowslice,printyn=printyn,gridon=gridon);


# # In[12]:

# Write input
m.write_input()

# Try to delete the output files, to prevent accidental use of older files
try:
    os.remove(os.path.join(model_ws, 'MT3D.CNF'))
    os.remove(os.path.join(model_ws, 'MT3D001.MAS'))
    os.remove(os.path.join(model_ws, 'MT3D001.UCN'))
    os.remove(os.path.join(model_ws, modelname + '.hds'))
    os.remove(os.path.join(model_ws, modelname + '.cbc'))
except:
    pass


# In[13]:

# Run model
import datetime
ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

# run_basemodel = get_yn_response("Run the base case model?")
run_basemodel = False
if run_basemodel:
    v = m.run_model(silent=False, report=True)
    for idx in range(-3, 0):
        print(v[1][idx])

print('MC file in \t', m.MC_file.parent)


# In[134]:


# ## Post-processing results

# In[17]:

# Post-processing functions
def plotdischarge(modelname, model_ws, color='w', per=-1, scale=50, rowslice=0):
    fname = os.path.join(model_ws, '' + modelname + '.cbc')
    budobj = flopy.utils.CellBudgetFile(fname)
    qx = budobj.get_data(text='FLOW RIGHT FACE')[per]
    qz = budobj.get_data(text='FLOW LOWER FACE')[per]

    # Average flows to cell centers
    qx_avg = np.empty(qx.shape, dtype=qx.dtype)
    qx_avg[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol - 1] + qx[:, :, 1:ncol])
    qx_avg[:, :, 0] = 0.5 * qx[:, :, 0]
    qz_avg = np.empty(qz.shape, dtype=qz.dtype)
    qz_avg[1:, :, :] = 0.5 * (qz[0:nlay - 1, :, :] + qz[1:nlay, :, :])
    qz_avg[0, :, :] = 0.5 * qz[0, :, :]

    y, x, z = dis.get_node_coordinates()
    X, Z = np.meshgrid(x, z[:, 0, 0])
    iskip = 1  # how many cells to skip, 1 means plot every cell

    ax = plt.gca()
    cpatchcollection = ax.quiver(X[::iskip, ::iskip], Z[::iskip, ::iskip],
                                 qx_avg[::iskip, rowslice, ::iskip], -
                                 qz_avg[::iskip, rowslice, ::iskip],
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


def kstpkper_from_time(ucnobj, tottim):
    kstpkpers = ucnobj.get_kstpkper()
    times = ucnobj.get_times()
    timeind = times.index(tottim)
    kstpkper = kstpkpers[timeind]
    return kstpkper


def kstpkper_ind_from_kstpkper(ucnobj, kstpkper=(0, 0)):
    kstpkpers = ucnobj.get_kstpkper()
    kstpkper_unique = permute_kstpkper(ucnobj)[0]
    kstpkper_ind = kstpkper_unique.index(kstpkper)
    return kstpkper_ind


def get_salt_outflow(m, kstpkper=None, totim=None):
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    totim = ucnobj.get_times()[-1]
    if kstpkper == None:
        kstpkper = ucnobj.get_kstpkper()[-1]
    ocean_conc = ucnobj.get_data(kstpkper=kstpkper)
    return ocean_conc


def plot_background(mm, array, label=None):
    if label == None:
        label = [k for k, v in globals().items() if v is array][-1]
    if label == 'hk':
        norm = matplotlib.colors.LogNorm()
        vmin = hkClay
        vmax = hkSand
        cmap = 'jet'
    else:
        norm = None
        vmin = None
        vmax = None
        cmap = 'jet'
    cpatchcollection = mm.plot_array(
        array, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    cpatchcollection.set_label(label)
    return cpatchcollection, label


def plot_mas(m):
    # Load the mas file and make a plot of total mass in aquifer versus time
    fname = os.path.join(m.model_ws, 'MT3D001.MAS')
    mas = flopy.mt3d.Mt3dms.load_mas(fname)
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    plt.xlabel('Time (d)')
    plt.ylabel('Mass (kg)')
    plt.title('Mass of salt within model through time')
    ax.plot(mas.time, mas.total_mass)
    plt.show()
    return mas


def add_to_paramdict(paramdict, paramname, val):
    if paramdict is None:
        paramdict = {}
    if paramname in list(paramdict.keys()):
        paramdict[paramname].append(val)
    else:
        # paramdict.update(paramname=[val])
        paramdict[paramname] = [val]
    return


def record_salinity(m, totim=None, fname_write=None, ts_hms=None):
    if ts_hms is None:
        ts_hms = datetime.datetime.now().strftime('%H-%M-%S')
    # Extract final timestep salinity
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    if totim is None:
        totim = ucnobj.get_times()[-1]
    conc = ucnobj.get_data(totim=totim)
    if fname_write is None:
        fname_write = m.MC_file.parent.joinpath(
            'conc_' + str(int(totim)) + '_' + ts_hms + '.npy')
        print(fname_write)
        np.save(fname_write, conc)
    return conc


def copy_rename(src_file, dst_file):
    import shutil
    from pathlib import Path
    shutil.copy(str(Path(src_file)), str(Path(dst_file)))
    return


def idx2centroid(node_coord_tuple, idx_tuple):
    z_pt = node_coord_tuple[2][idx_tuple]
    x_pt = node_coord_tuple[1][idx_tuple[2]]
    y_pt = node_coord_tuple[0][idx_tuple[1]]
    return (z_pt, y_pt, x_pt)


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


tot_it = 2
seed = 1
np.random.seed(seed)
varlist = {}

from scipy.io import savemat

# log_hk
logyn = True
low = 10
high = 100
varlist['hk'] = sample_uniform(low, high, tot_it, logyn)

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

# rech
logyn = True
low = 1e-6 / (nrow * ncol)
high = 1e-1 / (nrow * ncol)
varlist['rech'] = sample_uniform(low, high, tot_it, logyn)

# farm rech
logyn = True
low = 1e-6 / (nrow * ncol)
high = 1e-1 / (nrow * ncol)
varlist['rech_farm'] = sample_uniform(low, high, tot_it, logyn)

# Wel
logyn = True
low = 1e1
high = 1e3
varlist['wel'] = sample_uniform(low, high, (4, tot_it), logyn)

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

# Save
save_obj(m.MC_file.parent, varlist, 'varlist')
print('Saved file', m.MC_file.parent.joinpath('varlist.pkl'))


# Heterogenous simulations hk_all =
# np.zeros((tot_it,nlay,nrow,ncol),dtype=np.float)


def update_run_model(varlist, it, runyn=True):

    #     for k,v in varlist.items():
    #         print('Assigning {} = {} \n'.format(k,v[it]))
    # insert new values
    hk = varlist['hk'][it]
    vka = varlist['vka'][it]
    al = varlist['al'][it]
    dmcoef = varlist['dmcoef'][it]
    riv_stg = varlist['riv_stg'][it]
    riv_cond = varlist['riv_cond'][it]

    ssm_data = {}
    # write recharge data
    farm_rech[farm_loc] = varlist['rech_farm'][it]
    rech_data = {}
    for i in range(len(perlen)):
        if i % 2 == 0:
            rech_data[i] = np.ones(
                (nrow, ncol), dtype=np.float) * varlist['rech'][it]
        else:
            rech_data[i] = farm_rech

    # write wel data
    wel_data, ssm_data, wel_cells = add_pumping_wells(wel_data_base,
                                                      ssm_data_base,
                                                      n_wells, flx=varlist[
                                                          'wel'][:, it],
                                                      rowcol=farm_orig,
                                                      kper=kper_odd)

    # Write river data--take SSM data from WEL!!
    riv_grad = .0005
    riv_data, ssm_data = write_river_data(
        riv_loc, riv_stg, riv_cond, riv_grad, kper_even, ssm_data)

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
    lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=vka, ipakcb=ipakcb, laytyp=1)

    # Add PCG Package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(m, hclose=1.e-8)

    # Add OC package to the MODFLOW model
    oc = flopy.modflow.ModflowOc(m,
                                 stress_period_data=oc_data,
                                 compact=True)

    # Create the basic MT3DMS model structure
    btn = flopy.mt3d.Mt3dBtn(m,
                             laycon=lpf.laytyp, htop=henry_top,
                             dz=dis.thickness.get_value(), prsity=0.2, icbund=icbund,
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

    # Make timestamp
    import datetime
    sep = '-'
    ts = datetime.datetime.now().strftime(
        '%m' + sep + '%d' + sep + '%H' + sep + '%M' + sep + '%S')
    ts_hms = ts.split(sep)[1:]
    ts_hms = sep.join(ts_hms)

    # Run model
    if runyn:
        v = m.run_model(silent=True, report=True)
        for idx in range(-3, 0):  # Report
            print(v[1][idx])

        # Record success/failure and store data
        if it == 0:
            varlist['success'] = v[0]
        else:
            varlist['success'] = np.r_[varlist['success'], v[0]]

        if v[0] is False:
            pass
        else:
            # Record final salinity as .npy, also move full CBC and UCN files
            # to expt folder
            fname = os.path.join(m.model_ws, 'MT3D001.UCN')
            totim = flopy.utils.binaryfile.UcnFile(fname).get_times()[-1]
            conc_fname = 'conc{}_totim{}_{}.UCN'.format(
                it, str(int(totim)), ts_hms)
            _ = record_salinity(
                m, ts_hms=ts_hms, fname_write=m.MC_file.parent.joinpath(conc_fname))
            copy_rename(os.path.join(m.model_ws, 'MT3D001.UCN'),
                        m.MC_file.parent.joinpath(conc_fname).as_posix())
    return m


for it in range(tot_it):
    update_run_model(varlist, it)
    print('Finished iteration ', it + 1, 'out of ', tot_it)


save_obj(m.MC_file.parent, varlist, 'varlist_success')
save_obj(m.MC_file.parent, m.dis.get_node_coordinates(), 'yxz')
print('Saved file ', m.MC_file.parent.joinpath('varlist_success.pkl'))
print('Saved file ', m.MC_file.parent.joinpath('yxz.pkl'))


# ##
# #%% Calculate hausdorff matrix and export
# import hausdorff_from_dir
# #importlib.reload(hausdorff_from_dir)
# pers = np.cumsum(perlen)
# totims = pers[0::10]
# #totims = (2340.0,4860.0,7200.0)
# fnames = hausdorff_from_dir.create_concmat_from_ucndir(m.MC_file.parent,totims=totims,modsize=hk.shape)
# yxz = load_obj(m.MC_file.parent,'yxz')

# for i in range(len(totims)):
#     hausdorff_from_dir.compute_export_hausdorff(m.MC_file.parent,
#                                                 conc_mat=np.load(m.MC_file.parent.joinpath(fnames[i])),
#                                                 yxz=yxz,suffix=str(totims[i]))
