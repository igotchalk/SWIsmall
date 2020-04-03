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


def normal_transform(data1,mu1,mu2,sig1,sig2):
    a = sig2/sig1
    b = mu2 - mu1 * a
    return a*data1 + b

   
def unpack_success_files(success_dir):
    from numpy import asarray
    nams = [p.name for p in Path(success_dir).iterdir() if 
            (p.name.startswith('success') and p.name.endswith('.pkl'))]
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

def create_ucn_from_dir(dirname,ucn_dict=None,
    totims = (1800, 3600, 5400, 7200),
                        modsize = (26,20,100),
                        rmFailed=True,
                        varlist=None,
                        ftype='mat'
                       ):
    from scipy.io import savemat
    import flopy
    import numpy as np
    if ucn_dict is None:
        try:
            load_obj(dirname, 'ucn_dict')
        except:
            ucn_dict = create_ucn_dict(dirname)

    if rmFailed:
        rmlist = []
        if varlist is None:
            try:
                varlist = load_obj(dirname, 'varlist_final')
            except:
                raise(Exception('Cannot find "varlist"'))
        for ind in ucn_dict.keys():
            if varlist['success'][ind]!=1:
                rmlist.append(ind)
        if len(rmlist)>0:
            rmdir = dirname.joinpath('removed')
            if not rmdir.exists():
                rmdir.mkdir()
            for rm_ind in rmlist:
                if not rmdir.joinpath(ucn_dict[rm_ind]).exists():
                    dirname.joinpath(ucn_dict[rm_ind]).replace(rmdir.joinpath(ucn_dict[rm_ind]))
                del ucn_dict[rm_ind]
                print('removed failed iteration from concentration dict: {}\n'.format(rm_ind))
        else:
            print('No failed iterations found')

    conc_mat = {}
    for k,v in ucn_dict.items():
        conc_tmp = np.zeros((len(totims),modsize[0],modsize[1],modsize[2]),dtype=np.float)
        for j,tim in enumerate(totims):
            conc_tmp[j,:,:,:] = flopy.utils.binaryfile.UcnFile(dirname.joinpath(v).as_posix()).get_data(totim=tim)
        conc_mat['conc_mat{}'.format(k)] = conc_tmp
    
    if ftype=='mat':
        conc_mat['times']=totims
        savemat(dirname.joinpath('conc_mat.mat'),conc_mat,  do_compression=True)
        print('saved {}'.format(dirname.joinpath('conc_mat.mat')))
    elif ftype=='npy':
        for k,v in conc_mat.items():
            np.save(dirname.joinpath('conc_mat{}.npy'.format(int(k[8:]))),v)
            print('saved {}'.format(dirname.joinpath('conc_mat{}.npy'.format(int(k[8:])))))
        np.save(dirname.joinpath('conc_mat_totims.npy'),np.asarray(totims))

    else:
        raise(Exception('allowed ftypes: "mat", "npy"'))
    return ucn_dict

def create_CF_from_dir(dirname,ucn_dict=None,
                   modsize = (26,20,100),
                   varlist=None,
                   ftype='mat',
                   saveyn=True):
    
    import simulationFFT

    nlay,nrow,ncol = modsize
    delv,delc,delr = (3,30,30)


    if varlist is None:
        varlist = load_obj(dirname,'varlist_final')
    if ucn_dict is None:
        try:
            ucn_dict = load_obj(dirname,'ucn_dict')
        except:
            ucn_dict = create_ucn_from_dir(dirname,totims = (1,180,360), saveyn=False)

    CF_mat = {}
    for it in ucn_dict.keys():
        CF_glob = varlist['CF_glob'][it]
        CF_var = varlist['CF_var'][it]
        seed = varlist['seed'][it]
        corr_len = varlist['corr_len'][it]
        corr_len_yx = varlist['corr_len_yx'][it]
        corr_len_zx = varlist['corr_len_zx'][it]
        vario_type = varlist['vario_type'][it]

        #Create Gaussian Simulation
        lcol = int(corr_len/delr)
        llay = int(corr_len*corr_len_zx/delv)
        lrow = int(corr_len*corr_len_yx/delc)
        CF_mat['CF{}'.format(it)] = simulationFFT.simulFFT(nrow,nlay, ncol,CF_glob,
                                            CF_var,vario_type, lrow ,
                                            llay, lcol,seed=seed)
    if saveyn:
        if ftype=='mat':
            from scipy.io import savemat
            savemat(dirname.joinpath('CF_mat.mat'),CF_mat, do_compression=True)
            print('saved {}'.format(dirname.joinpath('CF_mat.mat')))
        elif ftype=='npy':
            from numpy import save
            for k,v in CF_mat.items():
                save(dirname.joinpath('CF_mat{}.npy'.format(k)),v)
                print('saved {}'.format(dirname.joinpath('conc_mat{}.npy'.format(k))))
        else:
            raise(Exception('allowed ftypes: "mat", "npy"'))
    return CF_mat



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
