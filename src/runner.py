#!/usr/bin/env python
"""
Generate Chi_sc(qsc,g,gp) from Chi_uc(quc,G,Gp)

Note! Still under development!
Note! For q-points representation, we use Bohr unit.

Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
os.sys.path.append("/scratch1/08702/wkim94/Chi-folding/src")
import numpy as np
import h5py
from scipy.spatial import KDTree
from typedef import Polarizability

def main():

    sc_eps_path = '../sc/3.1-chi/'
    uc_eps_path = '../uc/3.1-chi/'
    chimat_sc   = Polarizability(sc_eps_path+'./chimat.h5')
    chimat_uc   = Polarizability(uc_eps_path+'./chimat.h5')
    #kuc_map_crys : real(nqsc, size_sc, 3)
    #have mapped k_uc in crystal_uc
    quc_map_crys = np.load('../sc/2.1-wfn/kuc_map_crys.npy')
    #nqsc, sc_size, _ = quc_map_crys.shape
    #tree  = KDTree(qpts_uc)
    #quc_map_iq   = np.zeros((nqsc, sc_size),dtype=np.int32)
    #_, iquc = tree.query(quc_map_crys[1,:,:],k=1,distance_upper_bound=1e-9)
    #print(_)
    #print(iquc)
    get_qucG2qscg(chimat_sc, chimat_uc)
    exit()

    return None



def get_qucG2qscg(chimat_sc, chimat_uc, quc_map_crys):
    """
    Get mapping between quc+G and qsc+g
    dat
    real(nG_atquc,ng_atqsc)

    --INPUT--
    chimat_sc : Polarizability

    chimat_uc : Polarizability


    --OUTPUT--
    qucG2qscg : int(nq_sc,sc_size,nG)
    For a given quc+G index in Chi_quc, return qsc+g index in Chi_qsc
    Note, for a given qsc, we only consider relavant 'sc_size' quc.
    """
    nqsc, sc_size, _ = quc_map_crys.shape
    print(chimat_uc.nmtx)
    print(chimat_uc.gind_eps2rho.shape)
    print(chimat_uc.components.shape)
    for iqsc in range(nqsc):
        qsc_crys = chimat_sc.qpts[iqsc]
        qsc_bohr = qsc@chimat_sc.bvec_bohr
        for iquc_local in range(sc_size)






    return qucG2qscg

def fold_qsc(iq, chimat_sc, chimat_uc, quc_map_crys):
    """
    make folded Chi^sc(qsc,g,g) for qsc in sc BZ

    """
    quc_lst_local = quc_map_crys[iq,:,:]
    quc_map_iq = [ 6,  7,  8,  9, 10, 11, 12]
    #chi_sc = chimat_sc['/mats/matrix'][:].view(np.complex128)
    chi_uc = chimat_uc['/mats/matrix'][:].view(np.complex128)
    for iquc in quc_map_iq:
        chi_uc_q = chi_cplx_uc[iq,:,:]
        chi_uc_q_diag = np.diagonal(chimat_q)





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

