#!/usr/bin/env python
"""
Generate Chi_sc(qsc,g,gp) from Chi_uc(quc,G,Gp)

Comments on notation & naming convention
    1. For real/reciprocal space, we use Bohr/Bohr^{-1} unit.
    2. Be carefull about distinguishing crystal and Bohr coordinate.
       Note, unlike q_bohr, physical meaning of q_crys depends on the cell size
    3. We use 'quc' for representing q-vectors in unit-cell BZ.
       We use 'qsc' for representing q-vectors in supercell BZ.
       We use 'g' or 'gsc' for representing G-vectors in supercell representation.
       We use 'G' or 'Guc' for representing G-vectors in unit-cell representation.

Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
os.sys.path.append("../../../src/")
import numpy as np
import h5py
from scipy.spatial import KDTree
from typedef import Polarizability

def main():
    sc_eps_path = '../sc/3.1-chi/'
    uc_eps_path = '../uc/3.1-chi/'
    chi0_sc = Polarizability.from_hdf5(fn_chimat=sc_eps_path+'./chi0mat.h5')
    chi0_uc = Polarizability.from_hdf5(fn_chimat=uc_eps_path+'./chi0mat.h5')
    chi_sc = Polarizability.from_hdf5(fn_chimat=sc_eps_path+'./chimat.h5')
    chi_uc = Polarizability.from_hdf5(fn_chimat=uc_eps_path+'./chimat.h5')
    #kuc_map_crys : real(nqsc, size_sc, 3)
    #have mapped k_uc in crystal_uc
    quc_map_crys = np.load('../sc/2.1-wfn/kuc_map_crys.npy')
    #nqsc, sc_size, _ = quc_map_crys.shape
    #tree  = KDTree(qpts_uc)
    #quc_map_iq   = np.zeros((nqsc, sc_size),dtype=np.int32)
    #_, iquc = tree.query(quc_map_crys[1,:,:],k=1,distance_upper_bound=1e-9)
    simple_hard_coding(chi0_uc, chi0_sc,chi_uc, chi_sc, quc_map_crys)
    #print(_)
    #print(iquc)
    #qucG2qscg = get_qucG2qscg(chi_sc, chi_uc)
    return None



def simple_hard_coding(chi0_uc, chi0_sc,chi_uc,chi_sc,quc_map_crys):
    """
    From Chi(iquc_global=7), find equivalent elements in Chi(iqsc_global=1)

    """
    # UC
    iquc_l = 6
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
    qucG2qscg = np.empty((len(chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]])),dtype=np.int32)
    for iGuc, hkl_uc in enumerate(chi_uc.components[chi_uc.gind_eps2rho[iquc_l,0:chi_uc.nmtx[iquc_l]]-1]):
        #print(iGuc,hkl_uc)
        Guc_bohr = hkl_uc@chi_uc.bvec_bohr
        print('\n')
        print(hkl_uc)
        print('chimat_uc[iGuc,iGuc]')
        print(chimat_uc[iGuc,iGuc])
        #print(chi_uc.qpts_bohr[iquc_l])
        #print(chi_sc.qpts_bohr[iqsc_l])
        #print(chi_uc.qpts_bohr[iquc_l]-chi_sc.qpts_bohr[iqsc_l])
        gsc_bohr = Guc_bohr+chi_uc.qpts_bohr[iquc_l]-chi_sc.qpts_bohr[iqsc_l]
        gsc_crys = gsc_bohr@np.linalg.inv(chi_sc.bvec_bohr)
        print('gsc_crys:', gsc_crys)
        dist, igsc = tree.query([gsc_crys],k=1)
        #print(dist)
        if dist[0] < 1e-9:
            print('dist[0]',dist[0])
            print('igsc[0]',igsc[0])
            igsc = igsc[0]
            print('chimat_sc[igsc,igsc]')
            print(chimat_sc[igsc,igsc])
            qucG2qscg[iGuc] = igsc
    print(qucG2qscg)
    print('chimat_sc[1,2]')
    print(chimat_uc[1,3])
    print('chimat_sc[qucG2qscg[1],qucG2qscg[2]]')
    print(chimat_sc[qucG2qscg[1],qucG2qscg[3]])

    return



def get_qucG2qscg(chi_sc, chi_uc, quc_map_crys):
    """
    Get mapping indices from quc+G to qsc+g
    dat
    real(nG_atquc,ng_atqsc)

    --INPUT--
    chi_sc : Polarizability

    chi_uc : Polarizability

    quc_map_crys : real(nqsc, size_sc, 3)


    --OUTPUT--
    qucG2qscg : int(nq_sc,sc_size,nG)
        For a given quc+G index in Chi_quc, return qsc+g index in Chi_qsc
        Note, for a given qsc, we only consider relavant 'sc_size' quc.
    """
    nqsc, sc_size, _ = quc_map_crys.shape
    tree = KDTree(chi_uc.qpts_crys)
    quc_map_iq = np.zeros((nqsc, sc_size),dtype=np.int32)
    _, iquc = tree.query(quc_map_crys[1,:,:],k=1,distance_upper_bound=1e-9)
    print(chi_uc.nmtx)
    print(chi_uc.gind_eps2rho.shape)
    print(chi_uc.components.shape)
    for iqsc in range(nqsc):
        qsc_bohr = chi_sc.qpts_bohr[iqsc]
        for iquc_local in range(sc_size):
            #Here iquc_local means index of quc in {qsc + g} in uc BZ
            #for a given qsc.
            quc_crys = quc_map_crys[iqsc,iquc_local,:]
            quc_bohr = quc_crys@chi_uc.bvec_bohr




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


    return phi_sc

if __name__ == '__main__':
    main()

