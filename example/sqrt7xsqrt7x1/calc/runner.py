#!/usr/bin/env python
"""
Generate Chi_sc(qsc,gsc,gscp) from Chi_uc(quc,guc,gucp)

Comments on notation & naming convention
    1. For real/reciprocal space, we use Bohr/Bohr^{-1} unit.
    2. Be carefull about distinguishing crystal and Bohr coordinate.
       Unlike q_bohr, physical meaning of q_crys depends on the cell size
    3. We use 'quc' for representing q-vectors in unit-cell BZ.
       We use 'qsc' for representing q-vectors in supercell BZ.
       We use 'gsc' and/or 'hklsc' for representing G-vectors in supercell representation.
       We use 'guc' and/or 'hkluc' for representing G-vectors in unit-cell representation.

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
    chi0_sc = Polarizability.header_from_hdf5(fn_chimat=sc_eps_path+'./chi0mat.h5')
    chi0_uc = Polarizability.header_from_hdf5(fn_chimat=uc_eps_path+'./chi0mat.h5')
    # For unfolding purpose we assume q0 = 0 exactly.
    chi0_uc.qpts_crys = np.array([[0.0,0.0,0.0]])
    chi0_uc.qpts_bohr = np.array([[0.0,0.0,0.0]])
    chi0_sc.qpts_crys = np.array([[0.0,0.0,0.0]])
    chi0_sc.qpts_bohr = np.array([[0.0,0.0,0.0]])
    chi_sc = Polarizability.header_from_hdf5(fn_chimat=sc_eps_path+'./chimat.h5')
    chi_uc = Polarizability.header_from_hdf5(fn_chimat=uc_eps_path+'./chimat.h5')
    #kuc_map_crys : real(nqsc, size_sc, 3)
    #have mapped k_uc in crystal_uc
    quc_map_crys = np.load('../sc/2.1-wfn/kuc_map_crys.npy')
    #nqsc, sc_size, _ = quc_map_crys.shape
    tree  = KDTree(chi_uc.qpts_crys)
    #quc_map_iq   = np.zeros((nqsc, sc_size),dtype=np.int32)
    _, quc_map_iq = tree.query(quc_map_crys,k=1,distance_upper_bound=1e-9)
    print(quc_map_iq)
    #simple_hard_coding_for_qsc0(chi0_uc, chi0_sc,chi_uc, chi_sc, quc_map_crys)
    get_qucG2qscg(chi0_uc, chi_uc, chi0_sc, chi_sc, quc_map_iq)
    #simple_hard_coding(chi0_uc, chi0_sc,chi_uc, chi_sc, quc_map_crys)
    #print(_)
    #print(iquc)
    #qucG2qscg = get_qucG2qscg(chi_sc, chi_uc)
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

    return

def simple_hard_coding(chi0_uc, chi0_sc,chi_uc,chi_sc,quc_map_iq):
    """
    From Chi(iquc_global=7), find equivalent elements in Chi(iqsc_global=1)

    """
    # UC
    iquc_l = 12
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
    for i in range(8):
        for j in range(8):
            print('\n')
            print(f'chimat_sc[{i},{j}]')
            print(chimat_uc[i,i])
            print(f'chimat_sc[qucG2qscg[{i}],qucG2qscg[{j}]]')
            print(chimat_sc[qucG2qscg[i],qucG2qscg[i]])

    return



def get_qucG2qscg(chi0_uc, chi_uc, chi0_sc, chi_sc, quc_map_iq):
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


    --OUTPUT--
    qucG2qscg : int(nq_sc,sc_size,max(nmatx_uc))
        For a given quc+guc index in Chi_quc, return qsc+gsc index in Chi_qsc.
        Note, for a given qsc, we only consider 'relavant' quc in {quc: quc = qsc+gsc}.
    """
    bvec_bohr_uc = chi_uc.bvec_bohr
    bvec_bohr_sc = chi_sc.bvec_bohr

    # 1. assign the shape of qucG2qscg
    max_rank_uc  = np.max(chi_uc.nmtx)      # maximum rank of chi_uc
    max_rank_uc  = max(max_rank_uc,np.max(chi0_uc.nmtx))
    quc_bohr_lst = np.concatenate((chi0_uc.qpts_bohr,chi_uc.qpts_bohr), axis=0)
    qsc_bohr_lst = np.concatenate((chi0_sc.qpts_bohr,chi_sc.qpts_bohr), axis=0)
    nquc         = len(quc_bohr_lst)
    nqsc         = len(qsc_bohr_lst)        # Assume we only have 'one' q0sc
    sc_size      = np.int32(nquc/nqsc)      # Size of supercell

    # qucG2scg is a zeros. It causes potential problem because even before mapping
    # it is already mapped to '0'
    qucG2qscg    = np.zeros((nqsc,sc_size,max_rank_uc),dtype=np.int32)

    # 2. We devide mapping procedure into 3-parts
    # (a). q0sc <-> q0uc
    # (b). q0sc <-> quc
    # (c). qsc  <-> quc


    print("\n2-(a). q0sc <-> q0uc")
    # Here, we find mapping between q0sc+gsc and q0uc+guc
    # Note, we q0sc/q0uc are exactly zero vector for mapping purpose!
    q0uc_bohr = quc_bohr_lst[0] # We don't use ...
    q0sc_bohr = qsc_bohr_lst[0]

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
        if dist[0] < 1e-8:
            #print('dist[0]',dist[0])
            #print('igsc[0]',igsc[0])
            igsc = igsc_lst[0]
            qucG2qscg[0,0,iguc] = igsc #qucG2qscg[iqsc,iquc_l,iguc]
        else:
            print('\n2-(a). q0sc <-> q0uc')
            print(f'Cannot find matched igsc for iguc: {iguc}')
            Error
    print("2-(a). q0sc <-> q0uc Done")


    print("2- (b). q0sc <-> quc")
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
                if dist[0] < 1e-8:
                    #print('dist[0]',dist[0])
                    #print('igsc[0]',igsc[0])
                    igsc = igsc_lst[0]
                    qucG2qscg[0,iquc_l,iguc] = igsc
                else:
                    print('\n2-(b). q0sc <-> quc')
                    print(f'Cannot find matched igsc for iguc: {iguc}')
                    Error
    print("2- (b). q0sc <-> quc Done")
    print(qucG2qscg[0])
    # 2-(b). q0sc <-> quc Done


    print("2- (c). qsc <-> quc")
    # Here, we need to deal with multiple qsc=\=0, quc=\=0
    # iquc_l represent a local quc index within given qsca
    # iquc_g represent a global quc index in 'chi_uc.qpts'
    # Note! 'chi_uc.qpts' does not include qsc0
    for iqsc, qsc_bohr in enumerate(qsc_bohr_lst):
        if iqsc==0:
            # we already have done q0sc <->
            continue
        else:
            set_hklsc_q = chi_sc.components[chi_sc.gind_eps2rho[iqsc-1,:chi_sc.nmtx[iqsc-1]]-1]
            qsc_tree = KDTree(set_hklsc_q)
            # iqsc -> iqsc-1 in chi_sc bcz we don't have qsc0
            quc_map_qsc = quc_map_iq[iqsc,:]
            for iquc_l, iquc_g in enumerate(quc_map_qsc):
                quc_bohr = chi_uc.qpts_bohr[iquc_g]
                set_hkluc_q = chi_uc.components[chi_uc.gind_eps2rho[iquc_g,:chi_uc.nmtx[iquc_g]]-1]
                for iguc, hkluc in enumerate(set_hkluc_q):
                    guc_bohr = hkluc@bvec_bohr_uc
                    gsc_bohr = guc_bohr + quc_bohr - qsc_bohr
                    gsc_crys = gsc_bohr@np.linalg.inv(bvec_bohr_sc) #should be int
                    dist, igsc_lst = qsc_tree.query([gsc_crys],k=1)
                    if dist[0] < 1e-8:
                        #print('dist[0]',dist[0])
                        #print('igsc[0]',igsc[0])
                        igsc = igsc_lst[0]
                        qucG2qscg[iqsc,iquc_l,iguc] = igsc
                    else:
                        print('\n2-(c). q0sc <-> quc')
                        print(f'Cannot find matched igsc for iguc: {iguc}')
                        Error
    print("2-(c). q0sc <-> quc Done")
    print(qucG2qscg)


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

