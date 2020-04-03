# utility functions
import pickle
from pathlib import Path
import glob
import os


def load_obj(dirname, name):
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)


def save_obj(dirname, obj, name):
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def unpack_success_files(success_dir):
    from numpy import asarray
    nams = [p.name for p in Path(success_dir).iterdir()]
    success_list= []
    for n in nams:
        success_list.append(load_obj(success_dir,n[:-4]))
    return asarray(success_list),nams


def create_ucn_dict(ucndir,str_start='conc',str_end='_20',saveyn=True):
    fnames = sorted(glob.glob(Path(ucndir).joinpath('*.UCN').as_posix()))
    fnames_part = [Path(f).name for f in fnames]
    inds ={}
    for f in fnames_part:
        start_ind, stop_ind  = f.find(str_start),f.find(str_end)
        inds[int(f[start_ind+len(str_start):stop_ind])] = f
    if saveyn:
        save_obj(ucndir,inds,'ucn_inds')
    return inds


def create_concmat_from_ucndir(dirname, pattern='*.UCN', totims=(2340.0,4860.0,7200.0), modsize=(26,20,100), saveyn=1, Lt=7200):
    import flopy
    import numpy as np
    dirname=Path(dirname)
    ucn_fnames = sorted(glob.glob(dirname.joinpath(pattern).as_posix()),
                         key=os.path.getctime)

    conc_mat = np.zeros((len(totims),len(ucn_fnames),modsize[0],modsize[1],modsize[2]),dtype=np.float)
    filt = []
    for i,fname in enumerate(ucn_fnames):
        print('file {} of {}'.format(i,len(ucn_fnames)))
        ucnobj = flopy.utils.binaryfile.UcnFile(fname)

        for j,tim in enumerate(totims):
            try:
                conc_mat[j,i,:,:,:] = ucnobj.get_data(totim=tim)
            except:
                filt.append(i)
                print('requested time {} not found in file:\n{}'.format(tim, Path(fname).parts[-1]))
    fnames = []
    for i,tim in enumerate(totims):
        fnames.append('conc_mat_totim' + str(int(tim)) + '.npy')
        np.save(dirname.joinpath(fnames[i]),conc_mat[i,:,:,:,:])

    return conc_mat,fnames
