import os
import sys
import numpy as np
import flopy
import sys
job_id=sys.argv[1]

def run(job_id):


    hk_mat = np.logspace(1,3,10)
    hk = hk_mat[job_id]

    workspace = os.path.join('/scratch/users/ianpg/henry')
    # make sure workspace directory exists
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    fext = 'png'
    narg = len(sys.argv)
    iarg = 0
    if narg > 1:
        while iarg < narg - 1:
            iarg += 1
            basearg = sys.argv[iarg].lower()
            if basearg == '--pdf':
                fext = 'pdf'

    # Input variables for the Henry Problem
    Lx = 2.
    Lz = 1.
    nlay = 50
    nrow = 1
    ncol = 100
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    henry_top = 1.
    henry_botm = np.linspace(henry_top - delv, 0., nlay)
    qinflow = 5.702  # m3/day
    dmcoef = 0.57024  # m2/day  Could also try 1.62925 as another case of the Henry problem

    # Create the basic MODFLOW model data
    modelname = 'henry'
    m = flopy.seawat.Seawat(modelname, exe_name="swt_v4", model_ws=workspace)

    # Add DIS package to the MODFLOW model
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=1, delr=delr,
                                   delc=delc, laycbd=0, top=henry_top,
                                   botm=henry_botm, perlen=1.5, nstp=15)

    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, -1] = -1
    bas = flopy.modflow.ModflowBas(m, ibound, 0)

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=hk, ipakcb=53)

    # Add PCG Package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(m, hclose=1.e-8)

    # Add OC package to the MODFLOW model
    oc = flopy.modflow.ModflowOc(m,
                                 stress_period_data={
                                     (0, 0): ['save head', 'save budget']},
                                 compact=True)

    # Create WEL and SSM data
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    wel_data = {}
    ssm_data = {}
    wel_sp1 = []
    ssm_sp1 = []
    for k in range(nlay):
        wel_sp1.append([k, 0, 0, qinflow / nlay])
        ssm_sp1.append([k, 0, 0, 0., itype['WEL']])
        ssm_sp1.append([k, 0, ncol - 1, 35., itype['BAS6']])
    wel_data[0] = wel_sp1
    ssm_data[0] = ssm_sp1
    wel = flopy.modflow.ModflowWel(m, stress_period_data=wel_data)

    # Create the basic MT3DMS model data
    btn = flopy.mt3d.Mt3dBtn(m, nprs=-5, prsity=0.35, sconc=35., ifmtcn=0,
                             chkmas=False, nprobs=10, nprmas=10, dt0=0.001)
    adv = flopy.mt3d.Mt3dAdv(m, mixelm=0)
    dsp = flopy.mt3d.Mt3dDsp(m, al=0., trpt=1., trpv=1., dmcoef=dmcoef)
    gcg = flopy.mt3d.Mt3dGcg(m, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

    # Create the SEAWAT model data
    vdf = flopy.seawat.SeawatVdf(m, iwtable=0, densemin=0, densemax=0,
                                 denseref=1000., denseslp=0.7143, firstdt=1e-3)

    # Write the input files
    m.write_input()

    # Try to delete the output files, to prevent accidental use of older files
    try:
        os.remove(os.path.join(workspace, 'MT3D001.UCN'))
        os.remove(os.path.join(workspace, modelname + '.hds'))
        os.remove(os.path.join(workspace, modelname + '.cbc'))
    except:
        pass

    # run the model
    m.run_model()
    
    np.savetxt(os.path.join(workspace,'output' + str(job_id) + '.txt'),np.r_[job_id,hk])    



    return

if __name__ == '__main__':
    success = run(job_id)
