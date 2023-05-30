#!/usr/bin/env python
"""
# Chi-folding
Folding polarizability matrix from unit-cell representation to supercell representation

Generate Chi_sc(qsc,gsc,gscp) from Chi_uc(quc,guc,gucp)

*Comments on notation & naming convention*
    1. For real/reciprocal space, we use Bohr/Bohr^{-1} unit.
    2. Be carefull about distinguishing crystal and Bohr coordinate.
       Unlike q_bohr, physical meaning of q_crys depends on the cell size
    3. We use 'quc' for representing q-vectors in unit-cell BZ.
       We use 'qsc' for representing q-vectors in supercell BZ.
       We use 'gsc' and/or 'hklsc' for representing G-vectors in supercell representation.
       We use 'guc' and/or 'hkluc' for representing G-vectors in unit-cell representation.

*Comments on workflow.*
    Currently we assume that calculations of chi(eps)mat.h5 for supercell are done seperately for each q-points. It might be the case since size of chi_sc for each q-points is huge.


*TODO*
    3. construct Chi_sc seperately


Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
os.sys.path.append("/pscratch/sd/w/wkim94/Chi-folding/src/")
import numpy as np
from mpi4py import MPI
import h5py
from scipy.spatial import KDTree
from typedef import Polarizability

comm = MPI.COMM_WORLD
def main():
    """
    Required file
    chi0mat.h5
    chimat.h5
    qpts_sc w/o q0
    kuc_map_crys.npy
    """
    ########################
    ###### SET PARAMS ######
    ########################
    sc_eps_path = '../'
    uc_eps_path = './epsilon_data_bot/'
    #sc_eps_path = '../sc/3.1-chi-separate/'
    #uc_eps_path = '../uc/3.1-chi/'
    change_nmtx = True
    # kuc_map_crys : real(nqsc, size_sc, 3)
    quc_map_crys = np.load('/pscratch/sd/w/wkim94/Moire/TBG/1.08/CC.KC-full.CH.rebo/Unitcell_3x3x1/Bot/bfold/kuc_map_crys.npy')
    #quc_map_crys = np.load('../sc/2.1-wfn/kuc_map_crys.npy')
    # contains mapped k_uc in crystal_uc

    qpts_crys = get_qpts_crys(sc_eps_path+'qpts', containq0=False)

    iqsc = 5

    # check the number of mpi process
    #if len(quc_map_crys[iqsc,:])%comm.size != 0:
    #    print(f'len(quc_map_crys[iqsc,:])={len(quc_map_crys[iqsc,:])} is not divisible by mpi process')
    #    Error

    ########################
    ###### LOAD DATA ######
    ########################

    #### Read the header of chimat file ###
    chi0_uc = Polarizability.from_hdf5(fn_chimat=uc_eps_path+'./chi0mat.h5')
    chi_uc  = Polarizability.from_hdf5(fn_chimat=uc_eps_path+'./chimat.h5')
    chi0_sc = Polarizability.from_hdf5(fn_chimat=sc_eps_path+'Eps_0/chi0mat.h5')
    chi_sc  = Polarizability.from_split(dirname=sc_eps_path,
                                       qpts_crys_inp=qpts_crys)

    # Change nmtx
    if change_nmtx:
        chi_uc_nmtx = np.loadtxt('/pscratch/sd/w/wkim94/Moire/TBG/1.08/CC.KC-full.CH.rebo/Unitcell_3x3x1/Bot/epsilon_4ry/chimat.nmtx.txt', dtype=np.int32)
        chi0_uc_nmtx = np.loadtxt('/pscratch/sd/w/wkim94/Moire/TBG/1.08/CC.KC-full.CH.rebo/Unitcell_3x3x1/Bot/epsilon_4ry/chi0mat.nmtx.txt',dtype=np.int32)
        if comm.rank == 0:
            print("Change nmtx!")
            os.sys.stdout.flush()
        chi_uc.nmtx = chi_uc_nmtx
        chi0_uc.nmtx = np.array([chi0_uc_nmtx])
    # Change nmtx

    comm.Barrier()

    if comm.rank == 0:
        print('########################')
        print('###### START MAIN ######')
        print('########################')
        os.sys.stdout.flush()

    # For unfolding purpose we assume q0 = 0 exactly.
    chi0_uc.qpts_crys = np.array([[0.0,0.0,0.0]])
    chi0_uc.qpts_bohr = np.array([[0.0,0.0,0.0]])
    chi0_sc.qpts_crys = np.array([[0.0,0.0,0.0]])
    chi0_sc.qpts_bohr = np.array([[0.0,0.0,0.0]])

    tree  = KDTree(chi_uc.qpts_crys)
    #quc_map_iq   = np.zeros((nqsc, sc_size),dtype=np.int32)
    _, quc_map_iq = tree.query(quc_map_crys,k=1,distance_upper_bound=1e-9)
    #print(quc_map_iq)

    # Because the tree doesn't contain the quc0 so quc_map_iq[0,0] is not mapped now.
    # so we assume quc_map_iq[0,0] should corresponding to q0uc and put '0'
    quc_map_iq[0,0] = 0
    #simple_hard_coding(chi0_uc, chi0_sc,chi_uc, chi_sc, quc_map_crys)

    if comm.rank == 0:
        qucG2qscg = get_qucG2qscg(chi0_uc, chi_uc, chi0_sc, chi_sc, quc_map_iq)
    else:
        qucG2qscg = None

    qucG2qscg = comm.bcast(qucG2qscg, 0)

    #if comm.rank == 0:
    #    np.save('qucG2qscg', qucG2qscg)

    comm.Barrier()

    if comm.rank == 0:
        print('\n')
        print('########################')
        print('####### ADD Chi ########')
        print('########################')
        print('\n')
        os.sys.stdout.flush()

    #_zero_mat(f'../sc/3.1-chi-separate/Eps_{iqsc}/chimat.h5')

    #if comm.rank == 0:
    #    _zero_mat(iqsc)

    comm.Barrier()

    if iqsc == 0:
        make_sc0_split(sc_eps_path, chi0_uc, chi_uc, qucG2qscg, quc_map_iq)
    else:
        make_sc_split(iqsc, sc_eps_path, chi0_uc, chi_uc, qucG2qscg, quc_map_iq)

    comm.Barrier()

    #if comm.rank == 0:
    #    _compare2ref(iqsc, sc_eps_path)
    #test_mat   = make_sc(iqsc, chi_sc.get_mat_iq(iqsc-1), chi0_uc, chi_uc, qucG2qscg, quc_map_iq)
    #test_mat  = make_sc0(0, chi0_sc.get_mat_iq(0), chi0_uc, chi_uc, qucG2qscg, quc_map_iq)
    return None

def _zero_mat(iqsc):
    """
    Zero out the matrix element of a hdf5 file
    This function is for debugging.
    """
    #chimat_sc_iq = np.zeros_like(target_mat)
    #dummy_h5 = h5py.File('../sc/3.1.1-chi_lowepscut/chimat_dummy.h5','r+')
    if iqsc == 0:
        fn = f'../sc/3.1-chi-separate/Eps_0/chi0mat.h5'
    else:
        fn = f'../sc/3.1-chi-separate/Eps_{iqsc}/chimat.h5'

    print('\nZero out', fn)
    dummy_h5 = h5py.File(fn, 'r+')
    chimat_sc_iq = dummy_h5['mats/matrix']
    chimat_sc_iq[:, :, :, :, :, :] = np.zeros_like(chimat_sc_iq)
    dummy_h5.close()
    return None


def _compare2ref(iqsc, sc_eps_path):
    print('\nSanity check')
    if iqsc == 0:
        fn_target = f'../sc/3.1-chi-separate_backup/Eps_0/chi0mat.h5'
        fn = sc_eps_path + f'./Eps_0/chi0mat.h5'
    else:
        fn_target = f'../sc/3.1-chi-separate_backup/Eps_{iqsc}/chimat.h5'
        fn = sc_eps_path + f'./Eps_{iqsc}/chimat.h5'

    print('target: ', fn_target)
    target = h5py.File(fn_target,'r')
    target_mat = (target['/mats/matrix'][0,0,0,:,:,0] + \
               1j*target['/mats/matrix'][0,0,0,:,:,1]).T
    target.close()

    chi_sc_at_iqsc = h5py.File(fn,'r')
    chimat_sc_iqsc = chi_sc_at_iqsc['mats/matrix']
    chimat_sc_iqsc = (chi_sc_at_iqsc['/mats/matrix'][0,0,0,:,:,0] + \
                   1j*chi_sc_at_iqsc['/mats/matrix'][0,0,0,:,:,1]).T

    chi_sc_at_iqsc.close()

    subtract_mat = target_mat - chimat_sc_iqsc
    print('\nconstructed')
    print(chimat_sc_iqsc[0,0:20])
    print('\ntarget')
    print(target_mat[0,:20])
    print('\n|subtract|')
    print(np.abs(subtract_mat[0,:20]))
    print('\nnp.max(subtract)')
    print(np.max(np.abs(subtract_mat)))

    return None

def make_sc0_split(sc_eps_path, chi0_uc, chi_uc, qucG2qscg, quc_map_iq):
    """
    Add Chi_uc(quc, guc, qpuc) to Chi_sc(gsc,gscp) at iqsc

    --INPUT--
    sc_eps_path : string
        path for splited chi_sc calculation

    chi0_uc : Polarizability

    chi_uc : Polarizability

    qucG2qscg : int(nqsc,sc_size,nguc)

    quc_map_iq : real(nqsc, size_sc)

    """
    ### LOAD DATA ###
    iqsc = 0
    fn = sc_eps_path + f'./Eps_{iqsc}/chi0mat.h5'
    chi_sc_at_iqsc = h5py.File(fn,'r+',driver='mpio',comm=comm)
    chimat_sc_iqsc = chi_sc_at_iqsc['mats/matrix']
    chi0mat_uc_h5  = h5py.File(chi0_uc.fn_chimat, 'r')
    chimat_uc_h5   = h5py.File(chi_uc.fn_chimat, 'r')

    # Needed qucG2qsc & quc_map_iq
    if comm.rank == 0:
        print('\nStart adding matrix elements\n')
        print('\nWorking on ', fn)
        print('\nNote, this is a special point q0sc')
        os.sys.stdout.flush()
    qucG2qscg_tmp  = qucG2qscg[iqsc,:,:]
    quc_map_iq_tmp = quc_map_iq[iqsc,:]
    nquc_l         = len(quc_map_iq_tmp)

    # Parallel setup
    quc_map_iq_splited = np.array_split(quc_map_iq_tmp, comm.size)
    iquc_tmp2iquc_l    = np.array_split(np.array(range(nquc_l)), comm.size)
    if comm.rank == 0:
        print(f'\nParallel over quc')
        print(f'\nNumber of quc in the master processor: {len(quc_map_iq_splited[0])}')
        os.sys.stdout.flush()
    # Start mapping iteration
    #for iquc_l, iquc_g in enumerate(quc_map_iq_tmp):
    comm.Barrier()
    for iquc_tmp, iquc_g in enumerate(quc_map_iq_splited[comm.rank]):
        iquc_l = iquc_tmp2iquc_l[comm.rank][iquc_tmp]
        if comm.rank == 0:
            print(f'iquc_l: {iquc_l} out of {nquc_l}')
            os.sys.stdout.flush()
        if (iquc_l == 0) and (iquc_g == 0):
            if comm.rank == 0:
                print("We are constructing q0sc from q0uc!")
                os.sys.stdout.flush()
            #chimat_uc_iq = chi0_uc.get_mat_iq(iquc_g)
            chimat_uc_iq = (chi0mat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,0] + \
             1j*chi0mat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,1]).T
            for iguc in range(chi0_uc.nmtx[iquc_g]):
                for jguc in range(chi0_uc.nmtx[iquc_g]):
                    igsc = qucG2qscg_tmp[iquc_l, iguc]
                    jgsc = qucG2qscg_tmp[iquc_l, jguc]
                    chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 0] += np.real(chimat_uc_iq[iguc,jguc])
                    chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 1] += np.imag(chimat_uc_iq[iguc,jguc])
        else:
            #chimat_uc_iq = chi_uc.get_mat_iq(iquc_g)
            chimat_uc_iq = (chimat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,0] + \
             1j*chimat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,1]).T
            for iguc in range(chi_uc.nmtx[iquc_g]):
                for jguc in range(chi_uc.nmtx[iquc_g]):
                    igsc = qucG2qscg_tmp[iquc_l, iguc]
                    jgsc = qucG2qscg_tmp[iquc_l, jguc]
                    chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 0] += np.real(chimat_uc_iq[iguc,jguc])
                    chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 1] += np.imag(chimat_uc_iq[iguc,jguc])
    # End of mapping iteration
    comm.Barrier()
    chi_sc_at_iqsc.close()
    chi0mat_uc_h5.close()
    chimat_uc_h5.close()
    if comm.rank == 0:
        print('\nEnd of adding matrix elements\n')
        os.sys.stdout.flush()

    return None

def make_sc_split(iqsc, sc_eps_path, chi0_uc, chi_uc, qucG2qscg, quc_map_iq):
    """
    Add Chi_uc(quc, guc, qpuc) to Chi_sc(gsc,gscp) at iqsc

    --INPUT--
    iqsc : int
        q-point index for sc. qsc0 correponding to 0

    sc_eps_path : string
        path for splited chi_sc calculation

    chi0_uc : Polarizability

    chi_uc : Polarizability

    qucG2qscg : int(nqsc,sc_size,nguc)

    quc_map_iq : real(nqsc, size_sc)

    """
    ### LOAD DATA ###
    fn = sc_eps_path + f'./Eps_{iqsc}/chimat.h5'
    chi_sc_at_iqsc = h5py.File(fn,'r+',driver='mpio',comm=comm)
    chimat_sc_iqsc = chi_sc_at_iqsc['mats/matrix']
    chi0mat_uc_h5  = h5py.File(chi0_uc.fn_chimat, 'r')
    chimat_uc_h5   = h5py.File(chi_uc.fn_chimat, 'r')

    # Needed qucG2qsc & quc_map_iq
    if comm.rank == 0:
        print('\nStart adding matrix elements\n')
        print('Working on ', fn)
        os.sys.stdout.flush()

    qucG2qscg_tmp  = qucG2qscg[iqsc,:,:]
    quc_map_iq_tmp = quc_map_iq[iqsc,:]
    nquc_l         = len(quc_map_iq_tmp)

    # Parallel setup
    quc_map_iq_splited = np.array_split(quc_map_iq_tmp, comm.size)
    iquc_tmp2iquc_l    = np.array_split(np.array(range(nquc_l)), comm.size)
    if comm.rank == 0:
        print(f'\nParallel over quc')
        print(f'\nNumber of quc in the master processor: {len(quc_map_iq_splited[0])}')
        os.sys.stdout.flush()

    # Start mapping iteration
    comm.Barrier()
    #for iquc_l, iquc_g in enumerate(quc_map_iq_tmp):
    for iquc_tmp, iquc_g in enumerate(quc_map_iq_splited[comm.rank]):
        iquc_l = iquc_tmp2iquc_l[comm.rank][iquc_tmp]
        if comm.rank == 0:
            print(f'iquc_l: {iquc_l} out of {nquc_l}')
            os.sys.stdout.flush()
        #chimat_uc_iq = chi_uc.get_mat_iq(iquc_g)
        chimat_uc_iq = (chimat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,0] + \
             1j*chimat_uc_h5['/mats/matrix'][iquc_g,0,0,:,:,1]).T
        for iguc in range(chi_uc.nmtx[iquc_g]):
            for jguc in range(chi_uc.nmtx[iquc_g]):
                igsc = qucG2qscg_tmp[iquc_l, iguc]
                jgsc = qucG2qscg_tmp[iquc_l, jguc]
                chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 0] += np.real(chimat_uc_iq[iguc,jguc])
                chimat_sc_iqsc[0, 0, 0, jgsc, igsc, 1] += np.imag(chimat_uc_iq[iguc,jguc])
    # End of mapping iteration
    comm.Barrier()
    chi_sc_at_iqsc.close()
    chi0mat_uc_h5.close()
    chimat_uc_h5.close()
    if comm.rank == 0:
        print('\nEnd of adding matrix elements\n')
        os.sys.stdout.flush()
    return None

def get_qpts_crys(fn_qpts_crys, containq0):
    """
    --INPUT--
    fn_qpts_crys : string

    containq0 : bool
        exclude first element (=q0) if True

    --OUTPUT--
    qpts_crys : float(nqsc-1, 3)
    """
    qpts_crys = np.loadtxt(fn_qpts_crys, usecols=(0,1,2),dtype=np.float64)
    if containq0:
        return qpts_crys[1:,:]
    else:
        return qpts_crys


def make_sc0(iqsc, target_mat, chi0_uc, chi_uc, qucG2qscg, quc_map_iq):
    """
    Test Code
    From Chi_uc(quc, guc, gpuc), make a Chi_sc(gsc, gscp) at iqsc
    Currently we read prexisting Chi_sc(gsc,gscp) and copy the shape of it.

    --INPUT--
    iqsc : int
        q-point index for sc. qsc0 correponding to 0

    target_mat : complex128(:,:)

    chi0_uc : Polarizability

    chi_uc  : Polarizability

    qucG2qscg : int(nqsc,sc_size,nguc)

    quc_map_iq : real(nqsc, size_sc)


    --OUTPUT--
    chimat_sc_iq : complex128(:,:)
        constructed chi_sc at iqsc
    """

    #chimat_sc_iq_ = np.zeros_like(target_mat)
    dummy_h5 = h5py.File('../sc/3.1.1-chi_lowepscut/chi0mat_dummy.h5','r+')
    chimat_sc_iq = dummy_h5['mats/matrix']
    chimat_sc_iq[0, 0, 0, :, :, 1] = np.zeros_like(chimat_sc_iq[0, 0, 0, :, :, 1])
    chimat_sc_iq[0, 0, 0, :, :, 0] = np.zeros_like(chimat_sc_iq[0, 0, 0, :, :, 0])
    # Needed qucG2qsc & quc_map_iq
    print('start mapping')
    qucG2qscg_tmp  = qucG2qscg[iqsc,:,:]
    quc_map_iq_tmp =  quc_map_iq[iqsc,:]

    # Start mapping iteration
    for iquc_l, iquc_g in enumerate(quc_map_iq_tmp):
        if (iquc_l == 0) and (iqsc == 0):
            print("We are constructing q0sc from q0uc")
            chimat_uc_iq = chi0_uc.get_mat_iq(iquc_g)
            for iguc in range(chi0_uc.nmtx[iquc_g]):
                for jguc in range(chi0_uc.nmtx[iquc_g]):
                    igsc = qucG2qscg_tmp[iquc_l, iguc]
                    jgsc = qucG2qscg_tmp[iquc_l, jguc]
                    #chimat_sc_iq[igsc,jgsc] = chimat_uc_iq[iguc,jguc]
                    chimat_sc_iq[0, 0, 0, jgsc, igsc, 0] = np.real(chimat_uc_iq[iguc,jguc])
                    chimat_sc_iq[0, 0, 0, jgsc, igsc, 1] = np.imag(chimat_uc_iq[iguc,jguc])
        else:
            chimat_uc_iq = chi_uc.get_mat_iq(iquc_g)
            for iguc in range(chi_uc.nmtx[iquc_g]):
                for jguc in range(chi_uc.nmtx[iquc_g]):
                    igsc = qucG2qscg_tmp[iquc_l, iguc]
                    jgsc = qucG2qscg_tmp[iquc_l, jguc]
                    #chimat_sc_iq[igsc,jgsc] = chimat_uc_iq[iguc,jguc]
                    chimat_sc_iq[0, 0, 0, jgsc, igsc, 0] = np.real(chimat_uc_iq[iguc,jguc])
                    chimat_sc_iq[0, 0, 0, jgsc, igsc, 1] = np.imag(chimat_uc_iq[iguc,jguc])

    subtract_mat = target_mat - (chimat_sc_iq[0,0,0,:,:,0]+1j*chimat_sc_iq[0,0,0,:,:,1]).T
    print('constructed')
    print(chimat_sc_iq[0,0,0,:10,0,0].T +1j*chimat_sc_iq[0,0,0,:10,0,1].T)
    print('target')
    print(target_mat[0,:10])
    print('|subtract|')
    print(np.abs(subtract_mat[0,:50]))
    print('np.max(subtract)')
    print(np.max(np.abs(subtract_mat)))
    dummy_h5.close()

    return None


def make_sc(iqsc, target_mat, chi0_uc, chi_uc, qucG2qscg, quc_map_iq):
    """
    Test Code
    From Chi_uc(quc, guc, gpuc), make a Chi_sc(gsc,gscp) at iqsc
    Currently we read prexisting Chi_sc(gsc,gscp) and copy the shape of it.

    --INPUT--
    iqsc : int
        q-point index for sc. qsc0 correponding to 0

    target_mat : complex128(:,:)

    chi0_uc : Polarizability

    chi_uc : Polarizability

    qucG2qscg : int(nqsc,sc_size,nguc)

    quc_map_iq : real(nqsc, size_sc)


    --OUTPUT--
    chimat_sc_iq : complex128(:,:)
        constructed chi_sc at iqsc
    """
    #chimat_sc_iq = np.zeros_like(target_mat)
    dummy_h5 = h5py.File('../sc/3.1.1-chi_lowepscut/chimat_dummy.h5','r+')
    chimat_sc_iq = dummy_h5['mats/matrix']
    chimat_sc_iq[iqsc-1, 0, 0, :, :, 1] = np.zeros_like(chimat_sc_iq[iqsc-1, 0, 0, :, :, 1])
    chimat_sc_iq[iqsc-1, 0, 0, :, :, 0] = np.zeros_like(chimat_sc_iq[iqsc-1, 0, 0, :, :, 0])

    # Needed qucG2qsc & quc_map_iq
    print('start mapping')
    qucG2qscg_tmp  = qucG2qscg[iqsc,:,:]
    quc_map_iq_tmp =  quc_map_iq[iqsc,:]

    # Start mapping iteration
    for iquc_l, iquc_g in enumerate(quc_map_iq_tmp):
        print(f'iquc_l: {iquc_l}')
        chimat_uc_iq = chi_uc.get_mat_iq(iquc_g)
        for iguc in range(chi_uc.nmtx[iquc_g]):
            for jguc in range(chi_uc.nmtx[iquc_g]):
                igsc = qucG2qscg_tmp[iquc_l, iguc]
                jgsc = qucG2qscg_tmp[iquc_l, jguc]
                #chimat_sc_iq[igsc,jgsc] = chimat_uc_iq[iguc,jguc]
                chimat_sc_iq[iqsc-1, 0, 0, jgsc, igsc, 0] = np.real(chimat_uc_iq[iguc,jguc])
                chimat_sc_iq[iqsc-1, 0, 0, jgsc, igsc, 1] = np.imag(chimat_uc_iq[iguc,jguc])
    # End of mapping iteration

    subtract_mat = target_mat - (chimat_sc_iq[iqsc-1,0,0,:,:,0]+1j*chimat_sc_iq[iqsc-1,0,0,:,:,1]).T
    print('constructed')
    print(chimat_sc_iq[iqsc-1,0,0,0:10,0,0].T +1j*chimat_sc_iq[iqsc-1,0,0,:10,0,1].T)
    print('target')
    print(target_mat[0,:10])
    print('|subtract|')
    print(np.abs(subtract_mat[0,:50]))
    print('np.max(subtract)')
    print(np.max(np.abs(subtract_mat)))
    dummy_h5.close()


    return None




def simple_hard_coding_for_qsc0(chi0_uc, chi0_sc,chi_uc,chi_sc,quc_map_crys):
    """
    From Chi(iquc_global=7), find equivalent elements in Chi(iqsc_global=1)

    """
    # UC
    iquc_l = 1
    uc_gind_eps2rho = chi_uc.gind_eps2rho
    print(chi_uc.components[chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]-1])
    chimat_uc = chi_uc.get_mat_iq(iq=iquc_l)

    # SC
    iqsc_l = 0
    tree = KDTree(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1])
    #tree = KDTree(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1]@chi_sc.bvec_bohr)
    #print('blat', hi_sc.blat)
    print('chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1]')
    print(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1])
    chimat_sc = chi_sc.get_mat_iq(iq=iqsc_l)
    print(len(chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]))
    qucG2qscg = np.zeros((len(chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]])),dtype=np.int32)
    print('init',qucG2qscg)
    for iguc, hkl_uc in enumerate(chi_uc.components[chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]-1]):
        guc_bohr = hkl_uc@chi_uc.bvec_bohr
        gsc_bohr = guc_bohr+chi_uc.qpts_bohr[iquc_l]-chi_sc.qpts_bohr[iqsc_l]
        #gsc_bohr = guc_bohr#+chi_uc.qpts_bohr[iquc_l]
        gsc_crys = gsc_bohr@np.linalg.inv(chi_sc.bvec_bohr)
        dist, igsc = tree.query([gsc_crys],k=1)
        #print(dist)
        if dist[0] < 1e-8:
            print('dist[0]',dist[0])
            print('igsc[0]',igsc[0])
            igsc = igsc[0]
            #print('chimat_sc[igsc,igsc]')
            #print(chimat_sc[igsc,igsc])
            qucG2qscg[iguc] = igsc
        else:
            print(f'Cannot find matched igsc for iguc: {iguc}')
            Error

    print(qucG2qscg)
    for i in range(8):
        for j in range(8):
            print('\n')
            print(f'chimat_sc[{i},{j}]')
            print(chimat_uc[i,i])
            print(f'chimat_sc[qucG2qscg[{i}],qucG2qscg[{j}]]')
            print(chimat_sc[qucG2qscg[i],qucG2qscg[i]])

    return None

def simple_hard_coding(chi0_uc, chi0_sc,chi_uc,chi_sc,quc_map_iq):
    """
    From Chi(iquc_global=7), find equivalent elements in Chi(iqsc_global=1)
    for iqsc_l = 0,
        iquc_l = {7, 8, 9, 10, 11, 12, 13} - 1
    for iqsc_l = 1,
        iquc_l = {14, 15, 16, 17, 18, 19, 20}
    for iqsc_l = 2,
        iquc_l = {21, 22, 23, 24, 25, 26, 27}

    """
    # UC
    iquc_l = 27
    uc_gind_eps2rho = chi_uc.gind_eps2rho
    print(chi_uc.components[chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]-1])
    chimat_uc = chi_uc.get_mat_iq(iq=iquc_l)

    # SC
    iqsc_l = 3
    tree = KDTree(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1])
    #tree = KDTree(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1]@chi_sc.bvec_bohr)
    #print('blat', hi_sc.blat)
    print('chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1]')
    print(chi_sc.components[chi_sc.gind_eps2rho[iqsc_l,0:chi_sc.nmtx[iqsc_l]]-1])
    chimat_sc = chi_sc.get_mat_iq(iq=iqsc_l)
    qucG2qscg = np.empty((len(chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]])),dtype=np.int32)
    for iguc, hkl_uc in enumerate(chi_uc.components[chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]-1]):
        guc_bohr = hkl_uc@chi_uc.bvec_bohr
        gsc_bohr = guc_bohr+chi_uc.qpts_bohr[iquc_l]-chi_sc.qpts_bohr[iqsc_l]
        gsc_crys = gsc_bohr@np.linalg.inv(chi_sc.bvec_bohr)
        dist, igsc = tree.query([gsc_crys],k=1)
        #print(dist)
        if dist[0] < 1e-9:
            #print('dist[0]',dist[0])
            #print('igsc[0]',igsc[0])
            igsc = igsc[0]
            #print('chimat_sc[igsc,igsc]')
            #print(chimat_sc[igsc,igsc])
            qucG2qscg[iguc] = igsc
    print(qucG2qscg)
    for i in range(15):
        for j in range(6):
            print('\n')
            print(f'chimat_uc[{i},{j}]')
            print(chimat_uc[i,i])
            print(f'chimat_sc[qucG2qscg[{i}],qucG2qscg[{j}]]')
            print(chimat_sc[qucG2qscg[i],qucG2qscg[i]])
            print('|diff| = ',np.abs(chimat_uc[i,i]-chimat_sc[qucG2qscg[i],qucG2qscg[i]]))

    return



def get_qucG2qscg(chi0_uc, chi_uc, chi0_sc, chi_sc, quc_map_iq, TOL_SMALL=1e-5):
    """
    Get mapping indices from quc+G to qsc+g
    dat
    real(nG_atquc,ng_atqsc)

    --INPUT--
    chi_sc : Polarizability

    chi_uc : Polarizability

    chi0_sc : Polarizability

    chi0_uc : Polarizability

    quc_map_iq : real(nqsc, size_sc)

    TOL_SMALL  : float64
        Tolerance for distacne in 'sc_crys' unit
        Bigger supercell requires higher TOL_SMALL


    --OUTPUT--
    qucG2qscg : int(nq_sc,sc_size,max(nmatx_uc))
        For a given quc+guc index in Chi_quc, return qsc+gsc index in Chi_qsc.
        Note, for a given qsc, we only consider 'relavant' quc in {quc: quc = qsc+gsc}.
    """
    if comm.rank == 0:
        print('\nStart mapping quc+guc <-> qsc+gsc')
        os.sys.stdout.flush()
    bvec_bohr_uc = chi_uc.bvec_bohr
    bvec_bohr_sc = chi_sc.bvec_bohr
    #print(chi_uc.nmtx)
    # 1. assign the shape of qucG2qscg
    max_rank_uc  = np.max(chi_uc.nmtx)      # maximum rank of chi_uc
    max_rank_uc  = max(max_rank_uc,np.max(chi0_uc.nmtx))
    #chi_uc.nmtx[:] = max_rank_uc
    #chi0_uc.nmtx[:] = np.max(chi0_uc.nmtx)
    nquc         = len(chi0_uc.qpts_bohr) + len(chi_uc.qpts_bohr)
    nqsc         = len(chi0_sc.qpts_bohr) + len(chi_sc.qpts_bohr)        # Assume we only have 'one' q0sc
    sc_size      = np.int32(nquc/nqsc)      # Size of supercell

    # qucG2scg is a zeros. It causes potential problem because even before mapping
    # it is already mapped to '0'
    qucG2qscg    = np.zeros((nqsc,sc_size,max_rank_uc),dtype=np.int32)
    qucG2qscg[:] = -1      # So we put -1 for unmapped points

    # 2. We devide mapping procedure into 3-parts
    # (a). q0sc <-> q0uc
    # (b). q0sc <-> quc
    # (c). qsc  <-> quc

    if comm.rank == 0:
        print("\n2-(a). q0sc <-> q0uc")
        os.sys.stdout.flush()
    # Here, we find mapping between q0sc+gsc and q0uc+guc
    # Note, we q0sc/q0uc are exactly zero vector for mapping purpose!
    q0uc_bohr = chi0_uc.qpts_bohr[0] # We don't use ...
    q0sc_bohr = chi0_sc.qpts_bohr[0] # We don't use ...

    # set of miller id (hkl)_uc of guc at q0uc
    set_hkluc_q0 = chi0_uc.components[chi0_uc.gind_eps2rho[0,:chi0_uc.nmtx[0]]-1]
    # set of miller id (hkl)_sc of gsc at q0sc
    set_hklsc_q0 = chi0_sc.components[chi0_sc.gind_eps2rho[0,:chi0_sc.nmtx[0]]-1]
    q0sc_tree = KDTree(set_hklsc_q0)
    for iguc, hkluc in enumerate(set_hkluc_q0):
        guc_bohr = hkluc@bvec_bohr_uc
        gsc_bohr = guc_bohr # + q0uc_bohr - q0sc_bohr
        gsc_crys = gsc_bohr@np.linalg.inv(bvec_bohr_sc) #should be int
        dist, igsc_lst = q0sc_tree.query([gsc_crys],k=1)
        if dist[0] < TOL_SMALL:
            #print('dist[0]',dist[0])
            #print('igsc[0]',igsc[0])
            igsc = igsc_lst[0]
            qucG2qscg[0,0,iguc] = igsc #qucG2qscg[iqsc,iquc_l,iguc]
        else:
            print('\n2-(a). q0sc <-> q0uc')
            print(f'Cannot find matched igsc for iguc: {iguc}')
            Error

    if comm.rank == 0:
        print("2-(a). q0sc <-> q0uc Done")
        os.sys.stdout.flush()


    if comm.rank == 0:
        print("2-(b). q0sc <-> quc")
        os.sys.stdout.flush()
    # Here, we need to deal with multiple quc=\=0
    # iquc_l represent a local quc index within given qsc
    # iquc_g represent a global quc index in 'chi_uc.qpts'
    # Note! 'chi_uc.qpts' does not include q0 vector
    quc_map_q0sc = quc_map_iq[0,:]
    for iquc_l, iquc_g in enumerate(quc_map_q0sc):
        if iquc_l == 0:
            # we already have done q0sc <-> q0uc
            continue
        else:
            quc_bohr = chi_uc.qpts_bohr[iquc_g]
            set_hkluc_q = chi_uc.components[chi_uc.gind_eps2rho[iquc_g,:chi_uc.nmtx[iquc_g]]-1]
            for iguc, hkluc in enumerate(set_hkluc_q):
                guc_bohr = hkluc@bvec_bohr_uc
                gsc_bohr = guc_bohr + quc_bohr # - q0sc_bohr
                gsc_crys = gsc_bohr@np.linalg.inv(bvec_bohr_sc) #should be int
                dist, igsc_lst = q0sc_tree.query([gsc_crys],k=1)
                if dist[0] < TOL_SMALL:
                    #print('dist[0]',dist[0])
                    #print('igsc[0]',igsc[0])
                    igsc = igsc_lst[0]
                    qucG2qscg[0,iquc_l,iguc] = igsc
                else:
                    print('\n2-(b). q0sc <-> quc')
                    print(f'Cannot find matched igsc for iguc: {iguc}')
                    Error

    if comm.rank == 0:
        print("2-(b). q0sc <-> quc Done")
        os.sys.stdout.flush()
    # 2-(b). q0sc <-> quc Done


    if comm.rank == 0:
        print("2-(c). qsc <-> quc")
        os.sys.stdout.flush()
    # Here, we need to deal with multiple qsc=\=0, quc=\=0
    # iquc_l represent a local quc index within given qsca
    # iquc_g represent a global quc index in 'chi_uc.qpts'
    # Note! 'chi_uc.qpts' does not include qsc0
    for iqsc, qsc_bohr in enumerate(chi_sc.qpts_bohr):
        if comm.rank == 0:
            print(f"      *iqsc = {iqsc} out of {nqsc-1}")
            os.sys.stdout.flush()
        set_hklsc_q = chi_sc.components[chi_sc.gind_eps2rho[iqsc,:chi_sc.nmtx[iqsc]]-1]
        qsc_tree = KDTree(set_hklsc_q)
        # iqsc -> iqsc+1 in chi_sc bcz we don't have qsc0
        quc_map_qsc = quc_map_iq[iqsc+1,:]
        for iquc_l, iquc_g in enumerate(quc_map_qsc):
            quc_bohr = chi_uc.qpts_bohr[iquc_g]
            set_hkluc_q = chi_uc.components[chi_uc.gind_eps2rho[iquc_g,:chi_uc.nmtx[iquc_g]]-1]
            for iguc, hkluc in enumerate(set_hkluc_q):
                guc_bohr = hkluc@bvec_bohr_uc
                gsc_bohr = guc_bohr + quc_bohr - qsc_bohr
                gsc_crys = gsc_bohr@np.linalg.inv(bvec_bohr_sc) #should be int
                dist, igsc_lst = qsc_tree.query([gsc_crys],k=1)
                if dist[0] < TOL_SMALL:
                    #print('dist[0]',dist[0])
                    #print('igsc[0]',igsc[0])
                    igsc = igsc_lst[0]
                    qucG2qscg[iqsc+1,iquc_l,iguc] = igsc
                else:
                    print('\n2-(c). qsc <-> quc')
                    print(f'Cannot find matched igsc for iguc: {iguc}')
                    Error

    if comm.rank == 0:
        print("2-(c). qsc <-> quc Done")
        print('\nEnd of mapping quc+guc <-> qsc+gsc')
        os.sys.stdout.flush()


    return qucG2qscg



def fold_qsc(iq, chi_sc, chi_uc, quc_map_crys):
    """
    make folded Chi^sc(qsc,g,g) for qsc in sc BZ

    """
    quc_lst_local = quc_map_crys[iq,:,:]
    quc_map_iq = [ 6,  7,  8,  9, 10, 11, 12]
    #chi_sc = chi_sc['/mats/matrix'][:].view(np.complex128)
    chi_uc = chi_uc['/mats/matrix'][:].view(np.complex128)
    for iquc in quc_map_iq:
        chi_uc_q = chi_cplx_uc[iq,:,:]
        chi_uc_q_diag = np.diagonal(chi_q)




if __name__ == '__main__':
    main()
