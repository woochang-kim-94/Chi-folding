#!/usr/bin/env python
"""
####################################################
Generate pristine supercell wavefunctions for a given fft grid

Note! Still under development!
Note! In this code, for k-points representation, we use the 2*pi/alat_uc unit
, same as the bfold and map_unk codes
Note! Mapping procedure need to be optimized!
Note! Need to support noncolinear basis
####################################################

Written by W. Kim   Aug. 02. 2022
"""
import h5py
from numpy import *
import os
from os.path import exists
from scipy.spatial import KDTree
bohr = 0.529177211 #Angstrom



def main():
    for ik_sc in range(130, 145):
        for ibnd_sc in range(1, 11):
            find_wfn_v(ik_sc=ik_sc, iv_sc=ibnd_sc)
            find_wfn_c(ik_sc=ik_sc, ic_sc=ibnd_sc)
    return None

def find_wfn_c(ik_sc, ic_sc, savedir='./gen_wfc.save/'):
    alat, A_uc, sc, nv_sc, nc_sc, nF_sc = read_inp('./map.inp')
    B_uc = linalg.inv(A_uc)
    B_uc = transpose(B_uc)
    A_sc = zeros((3,3))
    A_sc[0] = A_uc[0]*sc[0,0] + A_uc[1]*sc[0,1]
    A_sc[1] = A_uc[0]*sc[1,0] + A_uc[1]*sc[1,1]
    A_sc[2] = A_uc[2]
    a_scba_uc = linalg.norm(A_sc[0,:]) # a_sc/a_uc
    B_sc = linalg.inv(A_sc)
    B_sc = transpose(B_sc) #alat_uc!
    print('alat_sc/alat_uc:', a_scba_uc)

    print('\nB_uc in 2pi/alat_uc')
    print(B_uc)
    print('\nB_sc in 2pi/alat_uc')
    print(B_sc)


    print('\nWe are looking for a unitcell wavefunction correspoding to the supercell wavefunction of')
    print('ik_sc: ', ik_sc)
    print('ic: ', ic_sc)


    path_scsets = '../../../Bilayer_SC/qe-nscf/WSe2.save/'
    wfc_sc_fn = path_scsets+'wfc'+str(ik_sc)+'.hdf5'
    prefix = 'WSe2'
    path_UCsets = '../bfold/'
    collect_ibv = load('./collect_ibv.npy')
    collect_ibc = load('./collect_ibc.npy')
    collect_kc = load('./collect_kc.npy')
    kuc_map = load('./kuc_map.npy')
    E_map = load('./E_map.npy')
    #k_sc_tpiba_sc = loadtxt('./kpts_sc.txt')
    k_sc_tpiba_uc = read_kpt_sc('../bfold/kpts_sc', B_sc) # already in 2pi/alat_uc!
    #k_sc_tpiba_uc = k_sc_tpiba_sc*a_scba_uc
    k_sc = k_sc_tpiba_uc[ik_sc-1]
    #print('k_sc in 2pi/alat_sc is:', k_sc_tpiba_sc[ik_sc-1])
    print('k_sc in 2pi/alat_uc is:', k_sc)

    ic_uc = int(collect_ibc[ik_sc-1,ic_sc-1]) #ic_uc is from bottom!
    k_uc = collect_kc[ik_sc-1,ic_sc-1,:]
    tree = KDTree(kuc_map[ik_sc-1,:,:])
    _, ik_uc = tree.query([k_uc],k=1)
    ik_uc = ik_uc[0] + 1

    if not allclose(k_uc, kuc_map[ik_sc-1,ik_uc-1,:]):
        print('Error in mapping procedure!')
        exit()

    print('\nWe found k_uc of index:',ik_uc)
    print('k_uc in 2pi/alat_uc:', k_uc)

    fn = 'UCset'+str(ik_sc)
   # print(E_map[ik_sc-1, ik_uc-1,iv_uc])

    # Now we've found the ik_uc and iv_uc corresponding to the ik_sc and iv_sc

    wfc_fn = path_UCsets+fn+'/'+prefix+'.save/'+'wfc'+str(ik_uc)+'.hdf5'
    unkg_set, miller_uc, npol_uc = read_wfc(wfc_fn)
    phi_uc = unkg_set[ic_uc,:]
    print('Read ', wfc_sc_fn, ' to get the FFT grid of sc at ik_sc')
    miller_sc, rec_latt_sc, xk_sc = get_miller_and_rec_latt_sc_and_xk(wfc_sc_fn)

    if not allclose(xk_sc*(1/bohr)*(alat/(2*pi)), k_sc, rtol=1e-05, atol=1e-05): # bohr^-1 -> tpi/alat_uc
        print('Error!, k_sc is different from the one in', wfc_sc_fn)
        #print('k_sc from file:', xk_sc/(2*pi*bohr/(alat*a_scba_uc))) # 2pi/alat_sc
        print('k_sc from file:', xk_sc*(1/bohr)*(alat/(2*pi)))  # 2pi/alat_uc
        print('k_sc from bfold:', k_sc) # 2pi/alat_uc
        exit()
    #print(dot(k_uc-k_sc, linalg.inv(B_sc)))
    phi_sc = gen_phi_sc(phi_uc, miller_uc, miller_sc, k_uc, k_sc, B_uc, B_sc, npol_uc)
    fn_phi_sc = savedir+'phi_sc._ik.{0:03d}._ic_sc.{1:02d}'.format(ik_sc, ic_sc)
    save(fn_phi_sc, phi_sc)
    #######TEST##############################
    #phi_sc_test, _ = read_wfc(wfc_sc_fn)
    #phi_sc_test = phi_sc_test[nF_sc-iv_sc,:]
    #print(abs(vdot(phi_sc, phi_sc_test))**2)
    #######TEST##############################

    return None



def find_wfn_v(ik_sc, iv_sc, savedir='./gen_wfc.save/'):
    alat, A_uc, sc, nv_sc, nc_sc, nF_sc = read_inp('./map.inp')
    B_uc = linalg.inv(A_uc)
    B_uc = transpose(B_uc)
    A_sc = zeros((3,3))
    A_sc[0] = A_uc[0]*sc[0,0] + A_uc[1]*sc[0,1]
    A_sc[1] = A_uc[0]*sc[1,0] + A_uc[1]*sc[1,1]
    A_sc[2] = A_uc[2]
    a_scba_uc = linalg.norm(A_sc[0,:]) # a_sc/a_uc
    B_sc = linalg.inv(A_sc)
    B_sc = transpose(B_sc) #alat_uc!
    print('alat_sc/alat_uc:', a_scba_uc)

    print('\nB_uc in 2pi/alat_uc')
    print(B_uc)
    print('\nB_sc in 2pi/alat_uc')
    print(B_sc)

    print('\nWe are looking for a unitcell wavefunction correspoding to the supercell wavefunction of')
    print('ik_sc: ', ik_sc)
    print('iv: ', iv_sc)

    path_scsets = '../../../Bilayer_SC/qe-nscf/WSe2.save/'
    wfc_sc_fn = path_scsets+'wfc'+str(ik_sc)+'.hdf5'
    prefix = 'WSe2'
    path_UCsets = '../bfold/'
    collect_ibv = load('./collect_ibv.npy')
    collect_kv = load('./collect_kv.npy')
    kuc_map = load('./kuc_map.npy')
    E_map = load('./E_map.npy')
    #k_sc_tpiba_sc = loadtxt('./kpts_sc.txt')
    k_sc_tpiba_uc = read_kpt_sc('../bfold/kpts_sc', B_sc) # already in 2pi/alat_uc!
    #k_sc_tpiba_uc = k_sc_tpiba_sc*a_scba_uc
    k_sc = k_sc_tpiba_uc[ik_sc-1]
    #print('k_sc in 2pi/alat_sc is:', k_sc_tpiba_sc[ik_sc-1])
    print('k_sc in 2pi/alat_uc is:', k_sc)

    iv_uc = int(collect_ibv[ik_sc-1,iv_sc-1]) #iv_uc is from bottom!
    k_uc = collect_kv[ik_sc-1,iv_sc-1,:]
    tree = KDTree(kuc_map[ik_sc-1,:,:])
    _, ik_uc = tree.query([k_uc],k=1)
    ik_uc = ik_uc[0] + 1

    if not allclose(k_uc, kuc_map[ik_sc-1,ik_uc-1,:]):
        print('Error in mapping procedure!')
        exit()

    print('\nWe found k_uc of index:',ik_uc)
    print('k_uc in 2pi/alat_uc:', k_uc)

    fn = 'UCset'+str(ik_sc)
   # print(E_map[ik_sc-1, ik_uc-1,iv_uc])

    # Now we've found the ik_uc and iv_uc corresponding to the ik_sc and iv_sc

    wfc_fn = path_UCsets+fn+'/'+prefix+'.save/'+'wfc'+str(ik_uc)+'.hdf5'
    unkg_set, miller_uc, npol_uc = read_wfc(wfc_fn)
    phi_uc = unkg_set[iv_uc,:]
    print('Read ', wfc_sc_fn, ' to get the FFT grid of sc at ik_sc')
    miller_sc, rec_latt_sc, xk_sc = get_miller_and_rec_latt_sc_and_xk(wfc_sc_fn)

    if not allclose(xk_sc*(1/bohr)*(alat/(2*pi)), k_sc, rtol=1e-05, atol=1e-05): # bohr^-1 -> tpi/alat_uc
        print('Error!, k_sc is different from the one in', wfc_sc_fn)
        #print('k_sc from file:', xk_sc/(2*pi*bohr/(alat*a_scba_uc))) # 2pi/alat_sc
        print('k_sc from file:', xk_sc*(1/bohr)*(alat/(2*pi)))  # 2pi/alat_uc
        print('k_sc from bfold:', k_sc) # 2pi/alat_uc
        exit()
    #print(dot(k_uc-k_sc, linalg.inv(B_sc)))
    phi_sc = gen_phi_sc(phi_uc, miller_uc, miller_sc, k_uc, k_sc, B_uc, B_sc, npol_uc)
    fn_phi_sc = savedir+'phi_sc._ik.{0:03d}._iv_sc.{1:02d}'.format(ik_sc, iv_sc)
    save(fn_phi_sc, phi_sc)
    #######TEST##############################
    #phi_sc_test, _ = read_wfc(wfc_sc_fn)
    #phi_sc_test = phi_sc_test[nF_sc-iv_sc,:]
    #print(abs(vdot(phi_sc, phi_sc_test))**2)
    #######TEST##############################

    return None


def gen_phi_sc(phi_uc, miller_uc, miller_sc, k_uc, k_sc, B_uc, B_sc, npol):
    ngwx_uc = len(miller_uc)
    ngwx_sc = len(miller_sc)
    tree = KDTree(miller_sc)
    phi_sc = zeros(len(miller_sc)*npol, dtype=complex64)
    if npol == 1:
        for igwx_uc, hkl_uc in enumerate(miller_uc):
            g_uc = dot(hkl_uc, B_uc)
            gauge_dif = g_uc + k_uc - k_sc
            gauge_dif_hkl = dot(gauge_dif, linalg.inv(B_sc))
            dist, igwx_sc = tree.query([gauge_dif_hkl],k=1)
            if dist[0] < 1e-5:
                phi_sc[igwx_sc[0]]=phi_uc[igwx_uc]
                #print(miller_sc[igwx_sc[0]])
                #print(gauge_dif_hkl)
    else:
        for igwx_uc, hkl_uc in enumerate(miller_uc):
            g_uc = dot(hkl_uc, B_uc)
            gauge_dif = g_uc + k_uc - k_sc
            gauge_dif_hkl = dot(gauge_dif, linalg.inv(B_sc))
            dist, igwx_sc = tree.query([gauge_dif_hkl],k=1)
            if dist[0] < 1e-5:
                phi_sc[igwx_sc[0]]=phi_uc[igwx_uc] #spin-up part
                phi_sc[igwx_sc[0]+ngwx_sc]=phi_uc[igwx_uc+ngwx_uc] #spin-down part
                #print(miller_sc[igwx_sc[0]])
                #print(gauge_dif_hkl)
    print(phi_sc)
    print(phi_sc.shape)
    print(linalg.norm(phi_sc))

    return phi_sc



def get_miller_and_rec_latt_sc_and_xk(fn):

    #Read wfc file
    print("Read ",fn)
    f = h5py.File(fn)

    gamma_only = f.attrs['gamma_only'] # bool
    igwx = f.attrs['igwx'] # int
    ik = f.attrs['ik'] # int k-point index
    ispin = f.attrs['ispin'] # int
    nbnd = f.attrs['nbnd'] # int
    ngw = f.attrs['ngw'] # int
    npol = f.attrs['npol'] # int
    scale_factor = f.attrs['scale_factor'] #float
    xk = f.attrs['xk'] # 3 components array in Bohr^-1?

    miller_ids = array(f['MillerIndices']) # (3 x 3)
    savetxt('miller_ids.{0:03d}.txt'.format(ik),miller_ids)
    print("\nMiller indices are saved in a text file")
    rec_latt = zeros((3,3))
    rec_latt[0,:] = array(f['MillerIndices'].attrs['bg1'])
    rec_latt[1,:] = array(f['MillerIndices'].attrs['bg2'])
    rec_latt[2,:] = array(f['MillerIndices'].attrs['bg3'])

    return miller_ids, rec_latt, xk



def read_wfc(fn):

    #Read wfc file
    print("Read ",fn)
    f = h5py.File(fn)

    gamma_only = f.attrs['gamma_only'] # bool
    igwx = f.attrs['igwx'] # int
    ik = f.attrs['ik'] # int k-point index
    ispin = f.attrs['ispin'] # int
    nbnd = f.attrs['nbnd'] # int
    ngw = f.attrs['ngw'] # int
    npol = f.attrs['npol'] # int
    scale_factor = f.attrs['scale_factor'] #float
    xk = f.attrs['xk'] # 3 components array in Bohr^-1?

    miller_ids = array(f['MillerIndices']) # (3 x 3)
    savetxt('miller_ids.{0:03d}.txt'.format(ik),miller_ids)
    print("\nMiller indices are saved in a text file")
    rec_latt = zeros((3,3))
    rec_latt[0,:] = array(f['MillerIndices'].attrs['bg1'])
    rec_latt[1,:] = array(f['MillerIndices'].attrs['bg2'])
    rec_latt[2,:] = array(f['MillerIndices'].attrs['bg3'])

    #rec_latt = (2*pi/alat)*rec_latt


    evc = array(f['evc'])
    if npol == 2:
        print('\nReading non-colinear spinor wavefunction')
        unkg_set = read_evc_non_colinear(evc, igwx, nbnd)
    else:
        unkg_set = read_evc(evc, igwx, nbnd)
#    savetxt('unkg.{0:03d}.txt'.format(ik),unkg_set)
#    print("\nWavefunctions are saved in a text file")
    return unkg_set, miller_ids, npol



def read_kpt_sc(f_name,B_sc):
    # Read f_name file and return in tpba format
    fp = open(f_name)
    fmt = fp.readline().split()
    nkpts = eval(fp.readline().split()[0])
    kpts = loadtxt(f_name, skiprows = 2)
    if fmt[0] == 'tpba':
        return kpts
    if fmt[0] == 'crystal':
        kpts_tpba = zeros((nkpts, 3))
        for ik in range(nkpts):
            if nkpts == 1:
                kpts_tpba[ik] = kpts[0]*B_sc[0] + kpts[1]*B_sc[1]
            else:
                kpts_tpba[ik] = kpts[ik, 0]*B_sc[0] + kpts[ik, 1]*B_sc[1]
        return kpts_tpba


def read_evc(evc, igwx, nbnd):

    print('converting the wavefunction coefficents to numpy format')
    psi_k_set = zeros((nbnd,igwx), dtype=complex64)
    for n, row in enumerate(evc):
        psi = add(row[::2], 1j*row[1::2])
        psi_k_set[n,:] = psi

    print('converting a wavefunction file is done')

    return psi_k_set


def read_evc_non_colinear(evc, igwx, nbnd):
    #if npol == 2 the len of array will be doulbed

    print('converting the wavefunction coefficents to numpy format')
    psi_k_set = zeros((nbnd,igwx*2), dtype=complex64)
    for n, row in enumerate(evc):
        psi = add(row[::2], 1j*row[1::2])
        psi_k_set[n,:] = psi

    print('converting a wavefunction file is done')

    return psi_k_set


def read_inp(f_name):
    fp = open(f_name)
    lines = fp.readlines()
    A_uc = zeros((3,3))
    sc = zeros((2,2), dtype = int32)
    for i in range(len(lines)):
        if "nv_sc" in lines[i]:
            w = lines[i+1].split()
            nv_sc, nc_sc, nF_sc = eval(w[0]), eval(w[1]), eval(w[2])
        if "alat" in lines[i]:
            w = lines[i+1].split()
            alat = eval(w[0])
        if "Unit-cell vectors" in lines[i]:
            for j in range(3):
                w = lines[i+j + 1].split()
                A_uc[j] = array([eval(w[0]), eval(w[1]), eval(w[2])])
        if "Super-cell vectors" in lines[i]:
            for j in range(2):
                w = lines[i+j + 1].split()
                sc[j] = array([eval(w[0]), eval(w[1])])
    return alat, A_uc, sc, nv_sc, nc_sc, nF_sc


main()
